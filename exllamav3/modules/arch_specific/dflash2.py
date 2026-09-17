"""
DFlash2-specific modules: grouped dynamic convolutions, the conv-wrapped
transformer block, and the top-k candidate selector.

Reference: the ``DFlash2DraftModel`` implementation in the ``dflash`` package.

The residual stream stays in fp32, matching the standard transformer path.
RMSNorm and convolution prepare outputs are fp16; convolution finish returns
fp32 before the residual add.
"""

from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

from ...model.config import Config
from .. import Module, Linear, RMSNorm, Attention, GatedMLP
from ...util.tensor import to2

from ...ext import exllamav3_ext as ext


def _grouped_dynamic_convolve_torch(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.float().view(batch, length, groups, group_size)
    dynamic = dynamic.float().view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(min(base.shape[0], length)):
        values = blocks[:, : length - offset]
        kernel = base[offset].float().view(1, 1, groups, group_size)
        weights = kernel + dynamic[:, offset:, offset]
        output[:, offset:] += weights * values
    output = output.view(batch, length, hidden_size)
    if residual is not None:
        residual += output
        return residual
    return output.to(hidden.dtype)


def _grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transcribed from dflash.model._grouped_dynamic_convolve.

    hidden  [b, l, H]; dynamic [b, l, taps, H//group_size] (any strides); base [taps, H].
    output[t] = sum_taps  base[tap] * x[t - tap] + dyn[tap][t] * x[t - tap]
    (causal; tap 0 = current position). Result has hidden's dtype, or, with residual (fp32
    [b, l, H]), is added into residual in place and residual is returned (the finish()
    variant's residual add, fused into the kernel).
    """
    if not hidden.is_cuda:
        return _grouped_dynamic_convolve_torch(hidden, dynamic, base, group_size, residual)
    hidden = hidden.contiguous()
    if residual is not None:
        ext.dflash2_dynconv(hidden, dynamic, base, residual, group_size, True)
        return residual
    output = torch.empty_like(hidden)
    ext.dflash2_dynconv(hidden, dynamic, base, output, group_size, False)
    return output


class DFlash2DynConv(Module):
    """Two-tap grouped dynamic conv (dflash ``GroupedDynamicCausalConv``).

    Checkpoint tensors (raw, unquantized, bf16):
      {key}.base_kernel        [2, kernel_size, hidden]   (prepare base, finish base)
      {key}.kernel_projection  Linear(hidden -> 2 * kernel_size * groups)

    prepare() uses fp16 input/output. finish() adds the fp32 result into the block's
    fp32 residual stream in place (one kernel, no separate add).
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

    def finish(self, x: torch.Tensor, dynamic: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """Sublayer output x [b, l, H] (fp16 or fp32) -> conv result in fp32, or, with residual
        (fp32 [b, l, H]), residual += conv result in place (returned)"""
        if residual is not None:
            return _grouped_dynamic_convolve(
                x, dynamic, self.base_kernel[1], self.group_size, residual = residual)
        return _grouped_dynamic_convolve(
            x.float(), dynamic, self.base_kernel[1], self.group_size)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        y, dyn = self.prepare(x, params)
        return to2(self.finish(y, dyn), out_dtype, torch.float)


class DFlash2Block(Module):
    """Reference DFlash2 decoder layer (Qwen3DFlashDecoderLayer):

        r = x; x = attn_norm(x); x, k = attn_conv.prepare(x); x = attn(x);
        x = attn_conv.finish(x, k); x = r + x
        r = x; x = mlp_norm(x);  x, k = mlp_conv.prepare(x);  x = mlp(x);
        x = mlp_conv.finish(x, k); x = r + x

    Residual stream fp32; normed sub-ops fp16. The residual adds are fused into the
    finish() convolutions.
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
        x = x.float() if x.dtype != torch.float else x
        y = self.attn_norm.forward(x, params, out_dtype = torch.half)
        y, kernel = self.attn_conv.prepare(y, params)
        y = self.attn.forward(y, params)
        x = self.attn_conv.finish(y, kernel, residual = x)

        y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
        y, kernel = self.mlp_conv.prepare(y, params)
        y = self.mlp.forward(y, params)
        x = self.mlp_conv.finish(y, kernel, residual = x)

        return to2(x, out_dtype, torch.float)


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
        return_confidence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Greedily rerank each row's top-k tokens, conditioned on the preceding token.
        Returns the path [b, rows] (and per-row winning scores [b, rows])."""
        out, conf = self.walk_block(hidden, logits, anchor_ids, return_confidence)
        if return_confidence:
            return out[:, 1:], conf[:, 1:]
        return out[:, 1:]


    def walk_block(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        return_confidence: bool = False,
        vocab_size: int | None = None,
        scale: float = 1.0,
        softcap: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """walk() in the generator's block layout: ids [b, rows + 1] = [anchor, path...] and,
        when requested, confidence [b, rows + 1] = [0, winning score...]. logits may be wider
        than vocab_size (padded head) and are scaled / softcapped on the fly. On CUDA the top-k
        and the whole chain run as two kernels (no per-row host round trip, no torch
        intermediates)."""
        vocab_size = vocab_size or logits.shape[-1]
        gate = self.hidden_proj.forward(hidden.half(), params = {})
        anchor_ids = anchor_ids.long()
        bsz, rows = logits.shape[:2]
        cuda = hidden.is_cuda and self.pred_codebook.dtype in (torch.half, torch.bfloat16)
        if cuda and self.top_k in (8, 16, 32) and logits.stride(-1) == 1:
            unary = torch.empty((bsz, rows, self.top_k), dtype = torch.float, device = hidden.device)
            cands = torch.empty((bsz, rows, self.top_k), dtype = torch.long, device = hidden.device)
            ext.dflash2_topk(logits, vocab_size, scale, softcap, unary, cands)
        else:
            logits = logits[..., :vocab_size].float() * scale
            if softcap > 0.0:
                logits = torch.tanh(logits / softcap) * softcap
            unary, cands = torch.topk(logits, self.top_k, dim = -1, sorted = False)
        if cuda:
            out = torch.empty((bsz, rows + 1), dtype = torch.long, device = hidden.device)
            conf = torch.empty((bsz, rows + 1), dtype = torch.float, device = hidden.device) if return_confidence else None
            ext.dflash2_selector_walk(
                unary.float().contiguous(), cands.long().contiguous(), gate.contiguous(),
                self.pred_codebook, self.succ_codebook, anchor_ids.to(hidden.device, non_blocking = anchor_ids.is_pinned()).contiguous(), out, conf,
            )
            return out, conf
        return self._walk_torch(unary.float(), cands.long(), gate.float(), anchor_ids, return_confidence)


    def _walk_torch(self, unary, cands, gate, anchor_ids, return_confidence):
        pred = anchor_ids
        path = [pred]
        confidence = [torch.zeros_like(pred, dtype = torch.float)]
        for i in range(unary.shape[1]):
            a_emb = F.embedding(pred, self.pred_codebook).float()
            b_emb = F.embedding(cands[:, i], self.succ_codebook).float()
            scores = unary[:, i] + torch.einsum("br,bkr->bk", a_emb * gate[:, i], b_emb)
            score, idx = torch.max(scores, dim = -1)
            pred = cands[:, i].gather(-1, idx[:, None])[:, 0]
            path.append(pred)
            confidence.append(score)
        out = torch.stack(path, dim = 1)
        return out, (torch.stack(confidence, dim = 1) if return_confidence else None)
