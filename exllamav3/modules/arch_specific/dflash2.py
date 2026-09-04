"""
DFlash2-specific modules: grouped dynamic convolutions, the conv-wrapped
transformer block, and the top-k candidate selector.

Reference: the ``DFlash2DraftModel`` implementation in the ``dflash`` package.

Dtype layout (measured on real fixtures): this checkpoint's residual stream
reaches |x| ~ 1.5e5 mid-network (bf16-trained by design), far beyond fp16
range, while everything downstream of an RMSNorm stays <= ~50. Hence:
  - residual stream, conv ``finish`` outputs, final state: bf16
  - projections/attention/MLP on normed inputs: stock fp16 exl3 modules
  - cache K/V: bounded (k is post-k_norm, v is post-hidden_norm) -> fp16
    cache layers as usual
"""

from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

from ...model.config import Config
from .. import Module, Linear, RMSNorm, Attention, GatedMLP
from ...util.tensor import to2


def _grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Transcribed from dflash.model._grouped_dynamic_convolve.

    hidden  [b, l, H]; dynamic [b, l, taps, H//group_size]; base [taps, H].
    output[t] = sum_taps  base[tap] * x[t - tap] + dyn[tap][t] * x[t - tap]
    (causal; tap 0 = current position). Caller controls dtype.
    """
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else \
            F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset].to(hidden.dtype), values)
    return output.view(batch, length, hidden_size)


class _DFlash2Norm(Module):
    """Torch-only RMSNorm with bf16 support (ext.rms_norm emits half only).
    Loads the raw bf16 weight; computes in fp32, emits bf16 (or given out_dtype)."""

    def __init__(self, config: Config, key: str, eps: float):
        super().__init__(config, key, None)
        self.module_name = "DFlash2Norm"
        self.eps = eps
        self.weight = None
        self._numel = config.hidden_size
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        return []

    @override
    def weights_numel(self):
        return self._numel

    def forward(self, x, params, out_dtype = None, **kwargs):
        w = self.weight.float()
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim = True)
        y = xf * torch.rsqrt(var + self.eps) * w
        return y.to(out_dtype or torch.bfloat16)

    @override
    def load(self, device: torch.device, **kwargs):
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError("DFlash2 requires a GPU with native bfloat16 support")
        super().load(device, **kwargs)
        self.weight = self.config.stc.get_tensor(
            self.key + ".weight", self.device, optional = False, allow_bf16 = True)
        if self.weight.shape != (self.config.hidden_size,):
            raise ValueError(
                f"Expected {self.key}.weight shape {(self.config.hidden_size,)}, "
                f"got {tuple(self.weight.shape)}"
            )
        self._numel = self.weight.numel()

    @override
    def unload(self):
        self.weight = None
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.weight is not None:
            t[self.key + ".weight"] = self.weight.contiguous()
        return t


class DFlash2DynConv(Module):
    """Two-tap grouped dynamic conv (dflash ``GroupedDynamicCausalConv``).

    Checkpoint tensors (raw, unquantized, bf16):
      {key}.base_kernel        [2, kernel_size, hidden]   (prepare base, finish base)
      {key}.kernel_projection  Linear(hidden -> 2 * kernel_size * groups)

    prepare(): fp16 math (input is post-norm, bounded ~50).
    finish(): bf16 math (input can reach the residual-stream magnitude ~1.5e5).
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        kernel_size: int,
        group_size: int,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "DFlash2DynConv"
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.groups = hidden_size // group_size

        self.proj = Linear(
            config = config,
            key = f"{key}.kernel_projection",
            in_features = hidden_size,
            out_features = 2 * kernel_size * self.groups,
            qmap = qmap,
            trim_padded_out = True,
        )
        self.register_submodule(self.proj)

        self.base_kernel = None
        self.key_base_kernel = f"{key}.base_kernel"
        self.base_kernel_numel = 2 * kernel_size * hidden_size
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        return []

    @override
    def weights_numel(self):
        return self.base_kernel_numel + super().weights_numel()

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.base_kernel = self.config.stc.get_tensor(
            self.key_base_kernel, self.device, optional = False, allow_bf16 = True
        )
        expected_shape = (2, self.kernel_size, self.hidden_size)
        if self.base_kernel.shape != expected_shape:
            raise ValueError(
                f"Expected {self.key_base_kernel} shape {expected_shape}, "
                f"got {tuple(self.base_kernel.shape)}"
            )

    @override
    def unload(self):
        self.base_kernel = None
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.base_kernel is not None:
            t[self.key_base_kernel] = self.base_kernel.contiguous()
        return t

    def prepare(self, x: torch.Tensor, params: dict):
        """x [b, l, H] (post-norm) -> (convolved half, finish-time dynamic half)"""
        x = x.half()
        dyn = self.proj.forward(x, params)
        dyn = dyn.view(*x.shape[:-1], 2, self.kernel_size, self.groups)
        y = _grouped_dynamic_convolve(
            x, dyn[..., 0, :, :], self.base_kernel[0], self.group_size)
        return y, dyn[..., 1, :, :]

    def finish(self, x: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.bfloat16)
        return _grouped_dynamic_convolve(
            x, dynamic, self.base_kernel[1], self.group_size)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        y, dyn = self.prepare(x, params)
        return to2(self.finish(y, dyn), out_dtype, torch.bfloat16)


class DFlash2Block(Module):
    """Reference DFlash2 decoder layer (Qwen3DFlashDecoderLayer):

        r = x; x = attn_norm(x); x, k = attn_conv.prepare(x); x = attn(x);
        x = attn_conv.finish(x, k); x = r + x
        r = x; x = mlp_norm(x);  x, k = mlp_conv.prepare(x);  x = mlp(x);
        x = mlp_conv.finish(x, k); x = r + x

    Residual stream bf16 (reaches ~1.5e5); normed sub-ops fp16 (bounded).
    """

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        attn: Attention,
        mlp: GatedMLP,
        attn_norm: RMSNorm,
        mlp_norm: RMSNorm,
        attn_conv: DFlash2DynConv,
        mlp_conv: DFlash2DynConv,
    ):
        super().__init__(config, key, None)
        self.module_name = "DFlash2Block"
        self.layer_idx = layer_idx
        self.attn = attn
        self.mlp = mlp
        self.attn_norm = attn_norm
        self.mlp_norm = mlp_norm
        self.attn_conv = attn_conv
        self.mlp_conv = mlp_conv
        for m in (attn, mlp, attn_norm, mlp_norm, attn_conv, mlp_conv):
            self.register_submodule(m)

    def optimizer_targets(self):
        return [self.attn.optimizer_targets(), self.mlp.optimizer_targets()]

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        x = x.to(torch.bfloat16)

        y = self.attn_norm.forward(x, params)
        y, kernel = self.attn_conv.prepare(y, params)
        y = self.attn.forward(y, params)
        y = self.attn_conv.finish(y, kernel)
        x = x + y

        y = self.mlp_norm.forward(x, params)
        y, kernel = self.mlp_conv.prepare(y, params)
        y = self.mlp.forward(y, params)
        y = self.mlp_conv.finish(y, kernel)
        x = x + y

        return to2(x, out_dtype, torch.bfloat16)


class DFlash2Selector(Module):
    """Top-k candidate selector (dflash ``CandidateSelector``).

    Checkpoint tensors (raw, unquantized, BARE keys — no .weight suffix):
      candidate_selector.predecessor_codebook  [vocab, rank]
      candidate_selector.successor_codebook    [vocab, rank]
      candidate_selector.hidden_projection     Linear(hidden -> rank, no bias)

    walk(): top-k(16) per row from draft logits, then greedy chained walk
      S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>,  a = previous path token
    """

    def __init__(
        self,
        config: Config,
        key: str,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
    ):
        super().__init__(config, key, None)
        self.module_name = "DFlash2Selector"
        self.vocab_size = vocab_size
        self.rank = rank
        self.top_k = top_k

        self.hidden_proj = Linear(
            config = config,
            key = f"{key}.hidden_projection",
            in_features = hidden_size,
            out_features = rank,
            trim_padded_out = True,
        )
        self.register_submodule(self.hidden_proj)

        self.key_pred = f"{key}.predecessor_codebook"
        self.key_succ = f"{key}.successor_codebook"
        self.pred_codebook = None
        self.succ_codebook = None
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        return []

    @override
    def weights_numel(self):
        return 2 * self.vocab_size * self.rank + super().weights_numel()

    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        # The selector is part of the module list so loading, autosplit and compilation account
        # for its tensors. Proposal generation invokes walk() after the shared target LM head.
        return to2(x, out_dtype, None)

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.pred_codebook = self.config.stc.get_tensor(
            self.key_pred, self.device, optional = False, allow_bf16 = True)
        self.succ_codebook = self.config.stc.get_tensor(
            self.key_succ, self.device, optional = False, allow_bf16 = True)
        expected_shape = (self.vocab_size, self.rank)
        if self.pred_codebook.shape != expected_shape:
            raise ValueError(
                f"Expected {self.key_pred} shape {expected_shape}, "
                f"got {tuple(self.pred_codebook.shape)}"
            )
        if self.succ_codebook.shape != expected_shape:
            raise ValueError(
                f"Expected {self.key_succ} shape {expected_shape}, "
                f"got {tuple(self.succ_codebook.shape)}"
            )

    @override
    def unload(self):
        self.pred_codebook = None
        self.succ_codebook = None
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.pred_codebook is not None:
            t[self.key_pred] = self.pred_codebook.contiguous()
            t[self.key_succ] = self.succ_codebook.contiguous()
        return t

    def walk(
        self,
        hidden: torch.Tensor,        # [b, rows, H] post-norm draft state
        logits: torch.Tensor,        # [b, rows, V] float draft logits
        anchor_ids: torch.Tensor,    # [b]
    ) -> torch.Tensor:
        """Greedily rerank each row's top-k tokens, conditioned on the preceding token."""
        unary, cands = torch.topk(logits, self.top_k, dim = -1, sorted = False)
        unary = unary.float()
        cands = cands.long()
        gate = self.hidden_proj.forward(hidden.half(), params = {}).float()

        pred = anchor_ids.long()
        path = []
        for i in range(logits.shape[1]):
            a_emb = F.embedding(pred, self.pred_codebook).float()
            b_emb = F.embedding(cands[:, i], self.succ_codebook).float()
            scores = unary[:, i] + torch.einsum("br,bkr->bk", a_emb * gate[:, i], b_emb)
            idx = torch.argmax(scores, dim = -1)
            pred = cands[:, i].gather(-1, idx[:, None])[:, 0]
            path.append(pred)
        return torch.stack(path, dim = 1)
