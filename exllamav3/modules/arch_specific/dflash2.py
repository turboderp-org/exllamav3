from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .. import Module, Linear
from ...model.config import Config

# ---------------------------------------------------------------------------
# Pure math, ported from the reference implementation (z-lab/dflash,
# dflash/model.py, MIT license). The walk/conv functions take plain tensors so
# they can be unit-tested against the reference without loading a model.
# ---------------------------------------------------------------------------

def grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    out[i,c] = sum_t (base[t,c] + delta[i,t,g(c)]) * x[i-t,c]

    taps are zero across the block boundary (the shift is within the provided
    block tensor). hidden: (batch, length, hidden); dynamic: (batch, length, K,
    groups); base: (K, hidden).
    """
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


# ---------------------------------------------------------------------------
# Triton serve path for the grouped dynamic conv (kernel copied from
# turboderp-org/exllamav3#334, RodriMora). The eager grouped_dynamic_convolve
# above stays as the CPU fallback and the test oracle.
# ---------------------------------------------------------------------------

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


def _grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Serve-path grouped dynamic conv: the Triton kernel on CUDA, the eager
    grouped_dynamic_convolve elsewhere. hidden: (batch, length, hidden);
    dynamic: (batch, length, K, groups); base: (K, hidden). Output dtype
    follows hidden (the kernel accumulates in fp32).
    """
    if not hidden.is_cuda:
        return grouped_dynamic_convolve(hidden, dynamic, base, group_size)

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


def conv_prepare(
    hidden: torch.Tensor,
    base_kernel_0: torch.Tensor,
    kernel_projection_weight: torch.Tensor,
    kernel_size: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = hidden.shape[-1] // group_size
    dynamic = F.linear(hidden, kernel_projection_weight).view(
        *hidden.shape[:-1], 2, kernel_size, groups
    )
    return (
        grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], base_kernel_0, group_size),
        dynamic[..., 1, :, :],
    )


def conv_finish(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base_kernel_1: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return grouped_dynamic_convolve(hidden, dynamic, base_kernel_1, group_size)


def selector_select(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    anchor_ids: torch.Tensor,
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    hidden_projection_weight: torch.Tensor | None,
    temperature: float,
    top_k: int,
    return_confidence: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Candidate path selector. Keeps the top-k candidates per block position, scores
    adjacent transitions edge(p->c) = <A[p] * project(h), B[c]> + unary[c], and walks
    one path from the verified anchor. Greedy: argmax walk (q is None). Sampled:
    inverse-CDF walk returning q over the k candidates per position, for the
    lossless rejection-sampling verify.

    hidden: (batch, T, hidden) draft states for the mask positions; logits: (batch,
    T, vocab) draft logits (via the target head); anchor_ids: (batch,) last verified
    token ids; hidden_projection_weight: (rank, hidden) or None when hidden is
    already projected to rank. Returns (path tokens (batch, T), candidates
    (batch, T, k), q (batch, T, k) or None, confidence (batch, T) or None).
    """
    unary, candidates = torch.topk(logits, top_k, dim = -1, sorted = False)
    if hidden_projection_weight is not None:
        hidden = F.linear(hidden, hidden_projection_weight)
    # the codebooks load as fp16; keep the bilinear term in one dtype (no-op for the
    # fp32 parity tests, cast for the loaded model)
    hidden = hidden.to(predecessor_codebook.dtype)
    # anchor ids arrive from the generator's job bookkeeping (CPU); the codebooks live on
    # the model device, so the index tensor has to follow
    predecessor = anchor_ids.to(predecessor_codebook.device)
    # Position-independent gathers, hoisted out of the walk: candidates come
    # from the single topk above, so their codebook rows and the
    # hidden-product need one launch each, not one per position. Only the
    # predecessor embedding stays serial. (~57KB at T=7/k=16/rank=256.)
    m = F.embedding(candidates, successor_codebook) * hidden[:, :, None, :]
    bsz, T, _, _ = m.shape
    # One coalescing copy: a leading-dim slice per step is a view, while
    # m[:, position] is strided and forces bmm to copy it back (7 copies).
    m = m.permute(1, 0, 2, 3).reshape(T * bsz, -1, m.shape[-1])
    path, q_rows, conf_rows = [], [], []
    for position in range(T):
        a = F.embedding(predecessor, predecessor_codebook)
        scores = unary[:, position] + torch.bmm(
            m[position * bsz : (position + 1) * bsz], a[..., None])[..., 0]
        if temperature > 0:
            q = torch.softmax(scores.float() / temperature, dim = -1)
            # NOTE: multinomial, not inverse-CDF: the reference implementation
            # draws this stream, and exact stream parity (same seed -> same
            # path, since the draw feeds the next predecessor) is worth more
            # than the ~0.1ms a searchsorted swap would save. The hoist above
            # carries the launch savings and is bit-exact.
            index = torch.multinomial(q, 1)[:, 0]
            q_rows.append(q)
        else:
            index = torch.argmax(scores, dim = -1)
        if return_confidence:
            conf_rows.append(scores.gather(-1, index[:, None])[:, 0])
        predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
        path.append(predecessor)
    return (
        torch.stack(path, dim = 1),
        candidates,
        torch.stack(q_rows, dim = 1) if q_rows else None,
        torch.stack(conf_rows, dim = 1) if conf_rows else None,
    )


# ---------------------------------------------------------------------------
# ExLlama3 module wrappers
# ---------------------------------------------------------------------------

class GroupedDynamicCausalConv(Module):
    """
    DFlash 2 grouped dynamic depthwise convolution, wrapped around a sublayer:
    prepare() runs before the sublayer (input side), finish() after (output
    side). base_kernel stays a raw fp16 tensor (uncalibrated); the
    kernel_projection Linear carries no qmap and stays fp16 with it.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        kernel_size: int,
        group_size: int,
    ):
        super().__init__(config, key, None)
        self.module_name = "GroupedDynamicCausalConv"
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.groups = hidden_size // group_size

        self.kernel_projection = Linear(
            config = config,
            key = f"{key}.kernel_projection",
            in_features = hidden_size,
            out_features = 2 * kernel_size * self.groups,
            trim_padded_out = True,
        )
        self.register_submodule(self.kernel_projection)
        self.base_kernel = None


    @override
    def optimizer_targets(self):
        raise NotImplementedError()


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.base_kernel = self.config.stc.get_tensor(
            f"{self.key}.base_kernel",
            device,
            float2half = True,
            no_defer = True,
        )


    @override
    def unload(self):
        super().unload()
        self.base_kernel = None


    @override
    def weights_numel(self):
        return super().weights_numel() + (self.base_kernel.numel() if self.base_kernel is not None else 0)


    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.base_kernel is not None:
            t[f"{self.key}.base_kernel"] = self.base_kernel.data.contiguous()
        return t


    def prepare(self, hidden: torch.Tensor, params: dict) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.base_kernel is not None, "GroupedDynamicCausalConv not loaded"
        proj = self.kernel_projection.forward(hidden, params, out_dtype = torch.float)
        groups = hidden.shape[-1] // self.group_size
        dynamic = proj.view(*hidden.shape[:-1], 2, self.kernel_size, groups)
        # The projection runs in fp32 (dynamic deltas are small); the sublayer consuming
        # this output is fp16 (EXL3 GEMMs require kHalf), so cast the conv result back
        # on the way out.
        return (
            _grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size).to(hidden.dtype),
            dynamic[..., 1, :, :],
        )


    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor, params: dict) -> torch.Tensor:
        assert self.base_kernel is not None, "GroupedDynamicCausalConv not loaded"
        return _grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size).to(hidden.dtype)


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype = torch.half) -> torch.Tensor:
        raise NotImplementedError(
            "GroupedDynamicCausalConv wraps a sublayer via prepare()/finish(), not forward()"
        )


class CandidateSelector(Module):
    """
    DFlash 2 candidate path selector. Codebooks and hidden_projection stay raw
    fp16 tensors (uncalibrated): the projection Linear carries no qmap, so it
    can never enter the EXL3 path. Convert still copies the tensors through
    via the module's retain_raw_fp16 caps flag.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
    ):
        super().__init__(config, key, None)
        self.module_name = "CandidateSelector"
        self.vocab_size = vocab_size
        self.rank = rank
        self.top_k = top_k

        self.hidden_projection = Linear(
            config = config,
            key = f"{key}.hidden_projection",
            in_features = hidden_size,
            out_features = rank,
            trim_padded_out = True,
        )
        self.register_submodule(self.hidden_projection)
        self.predecessor_codebook = None
        self.successor_codebook = None
        self.caps.update({"retain_raw_fp16": True})


    @override
    def optimizer_targets(self):
        raise NotImplementedError()


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        # The DFlash2 checkpoints store the codebooks under the bare embedding name
        # (candidate_selector.predecessor_codebook); some trainers append .weight like a
        # standard nn.Embedding. Accept both.
        self.predecessor_codebook = self._load_codebook("predecessor_codebook", device)
        self.successor_codebook = self._load_codebook("successor_codebook", device)


    def _load_codebook(self, name: str, device: torch.device) -> torch.Tensor:
        t = self.config.stc.get_tensor(
            f"{self.key}.{name}",
            device,
            float2half = True,
            optional = True,
            no_defer = True,
        )
        if t is None:
            t = self.config.stc.get_tensor(
                f"{self.key}.{name}.weight",
                device,
                float2half = True,
                no_defer = True,
            )
        return t


    @override
    def unload(self):
        super().unload()
        self.predecessor_codebook = None
        self.successor_codebook = None


    @override
    def weights_numel(self):
        n = super().weights_numel()
        if self.predecessor_codebook is not None:
            n += self.predecessor_codebook.numel() + self.successor_codebook.numel()
        return n


    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.predecessor_codebook is not None:
            t[f"{self.key}.predecessor_codebook.weight"] = self.predecessor_codebook.data.contiguous()
            t[f"{self.key}.successor_codebook.weight"] = self.successor_codebook.data.contiguous()
        return t


    def select(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        temperature: float,
        return_confidence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        proj = self.hidden_projection.forward(hidden, {}, out_dtype = torch.half)
        assert self.predecessor_codebook is not None and self.successor_codebook is not None, \
            "CandidateSelector not loaded"
        return selector_select(
            proj,
            logits,
            anchor_ids,
            self.predecessor_codebook,
            self.successor_codebook,
            None,
            temperature,
            self.top_k,
            return_confidence,
        )


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype = torch.half) -> torch.Tensor:
        # The selector sits in the model's module list so the loader reaches its weights, and
        # the module walk visits every entry in order — here it must leave the state untouched.
        # Selection happens via select(), called from propose() with the draft block states.
        return x
