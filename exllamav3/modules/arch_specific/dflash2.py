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

try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False

    class _DummyTritonLanguage:
        constexpr = object()

    class _DummyTriton:
        @staticmethod
        def jit(fn):
            return fn

    triton = _DummyTriton()
    tl = _DummyTritonLanguage()


@triton.jit
def _grouped_dynamic_convolve_kernel(
    hidden,
    dynamic,
    base,
    output,
    dynamic_stride_b: tl.constexpr,
    dynamic_stride_l: tl.constexpr,
    dynamic_stride_k: tl.constexpr,
    dynamic_stride_g: tl.constexpr,
    length: tl.constexpr,
    hidden_size: tl.constexpr,
    groups: tl.constexpr,
    kernel_size: tl.constexpr,
    group_size: tl.constexpr,
    channel_tiles: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    group_tile = tl.program_id(1)
    position_tile = tl.program_id(2)
    group_idx = group_tile // channel_tiles
    channel_tile = group_tile - group_idx * channel_tiles

    positions = position_tile * BLOCK_L + tl.arange(0, BLOCK_L)
    local_channels = channel_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    channels = group_idx * group_size + local_channels
    position_mask = positions < length
    channel_mask = local_channels < group_size
    acc = tl.zeros((BLOCK_L, BLOCK_C), dtype = tl.float32)

    for offset in range(kernel_size):
        source_positions = positions - offset
        source_mask = position_mask & (source_positions >= 0)
        hidden_offsets = (
            (batch_idx * length + source_positions[:, None]) * hidden_size +
            channels[None, :]
        )
        values = tl.load(
            hidden + hidden_offsets,
            mask = source_mask[:, None] & channel_mask[None, :],
            other = 0.0,
        ).to(tl.float32)
        base_values = tl.load(
            base + offset * hidden_size + channels,
            mask = channel_mask,
            other = 0.0,
        ).to(tl.float32)
        dynamic_offsets = (
            batch_idx * dynamic_stride_b +
            positions * dynamic_stride_l +
            offset * dynamic_stride_k +
            group_idx * dynamic_stride_g
        )
        dynamic_values = tl.load(
            dynamic + dynamic_offsets,
            mask = position_mask,
            other = 0.0,
        ).to(tl.float32)
        acc += values * (base_values[None, :] + dynamic_values[:, None])

    output_offsets = (
        (batch_idx * length + positions[:, None]) * hidden_size +
        channels[None, :]
    )
    tl.store(
        output + output_offsets,
        acc,
        mask = position_mask[:, None] & channel_mask[None, :],
    )


def _grouped_dynamic_convolve_torch(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
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
    return output.view(batch, length, hidden_size).to(hidden.dtype)


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
    if not hidden.is_cuda or not has_triton:
        return _grouped_dynamic_convolve_torch(hidden, dynamic, base, group_size)

    hidden = hidden.contiguous()
    base = base.contiguous()
    batch, length, hidden_size = hidden.shape
    kernel_size = base.shape[0]
    groups = hidden_size // group_size
    output = torch.empty_like(hidden)
    block_l = triton.next_power_of_2(min(length, 16))
    block_c = triton.next_power_of_2(min(group_size, 64))
    channel_tiles = triton.cdiv(group_size, block_c)
    num_warps = 2 if block_l * block_c < 256 else 4
    grid = batch, groups * channel_tiles, triton.cdiv(length, block_l)
    with torch.cuda.device(hidden.device):
        _grouped_dynamic_convolve_kernel[grid](
            hidden,
            dynamic,
            base,
            output,
            dynamic_stride_b = dynamic.stride(0),
            dynamic_stride_l = dynamic.stride(1),
            dynamic_stride_k = dynamic.stride(2),
            dynamic_stride_g = dynamic.stride(3),
            length = length,
            hidden_size = hidden_size,
            groups = groups,
            kernel_size = kernel_size,
            group_size = group_size,
            channel_tiles = channel_tiles,
            BLOCK_L = block_l,
            BLOCK_C = block_c,
            num_warps = num_warps,
            num_stages = 1,
        )
    return output


class DFlash2DynConv(Module):
    """Two-tap grouped dynamic conv (dflash ``GroupedDynamicCausalConv``).

    Checkpoint tensors (raw, unquantized, bf16):
      {key}.base_kernel        [2, kernel_size, hidden]   (prepare base, finish base)
      {key}.kernel_projection  Linear(hidden -> 2 * kernel_size * groups)

    prepare() uses fp16 input/output. finish() uses fp32 input/output so the
    surrounding block keeps its residual stream in fp32.
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

    Residual stream fp32; normed sub-ops fp16.
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
        y = self.attn_norm.forward(x, params, out_dtype = torch.half)
        y, kernel = self.attn_conv.prepare(y, params)
        y = self.attn.forward(y, params)
        y = self.attn_conv.finish(y, kernel)
        x += y

        y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
        y, kernel = self.mlp_conv.prepare(y, params)
        y = self.mlp.forward(y, params)
        y = self.mlp_conv.finish(y, kernel)
        x += y

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
        """Greedily rerank each row's top-k tokens, conditioned on the preceding token."""
        unary, cands = torch.topk(logits, self.top_k, dim = -1, sorted = False)
        unary = unary.float()
        cands = cands.long()
        gate = self.hidden_proj.forward(hidden.half(), params = {}).float()

        pred = anchor_ids.long()
        path = []
        confidence = []
        for i in range(logits.shape[1]):
            a_emb = F.embedding(pred, self.pred_codebook).float()
            b_emb = F.embedding(cands[:, i], self.succ_codebook).float()
            scores = unary[:, i] + torch.einsum("br,bkr->bk", a_emb * gate[:, i], b_emb)
            score, idx = torch.max(scores, dim = -1)
            pred = cands[:, i].gather(-1, idx[:, None])[:, 0]
            path.append(pred)
            if return_confidence:
                confidence.append(score)
        path = torch.stack(path, dim = 1)
        if return_confidence:
            return path, torch.stack(confidence, dim = 1)
        return path
