from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
    (batch, T, k), q (batch, T, k) or None).
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
    path, q_rows = [], []
    for position in range(hidden.shape[1]):
        scores = unary[:, position] + torch.einsum(
            "br,bkr->bk",
            F.embedding(predecessor, predecessor_codebook) * hidden[:, position],
            F.embedding(candidates[:, position], successor_codebook),
        )
        if temperature > 0:
            q = torch.softmax(scores.float() / temperature, dim = -1)
            index = torch.multinomial(q, 1)[:, 0]
            q_rows.append(q)
        else:
            index = torch.argmax(scores, dim = -1)
        predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
        path.append(predecessor)
    return (
        torch.stack(path, dim = 1),
        candidates,
        torch.stack(q_rows, dim = 1) if q_rows else None,
    )


# ---------------------------------------------------------------------------
# ExLlama3 module wrappers
# ---------------------------------------------------------------------------

class GroupedDynamicCausalConv(Module):
    """
    DFlash 2 grouped dynamic depthwise convolution, wrapped around a sublayer:
    prepare() runs before the sublayer (input side), finish() after (output
    side). base_kernel stays a raw fp16 tensor (uncalibrated); the
    kernel_projection Linear quantizes normally.
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
            grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size).to(hidden.dtype),
            dynamic[..., 1, :, :],
        )


    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor, params: dict) -> torch.Tensor:
        assert self.base_kernel is not None, "GroupedDynamicCausalConv not loaded"
        return grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size).to(hidden.dtype)


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype = torch.half) -> torch.Tensor:
        raise NotImplementedError(
            "GroupedDynamicCausalConv wraps a sublayer via prepare()/finish(), not forward()"
        )


class CandidateSelector(Module):
    """
    DFlash 2 candidate path selector. Codebooks stay raw fp16 tensors
    (uncalibrated); hidden_projection quantizes normally.
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
        )
        self.register_submodule(self.hidden_projection)
        self.predecessor_codebook = None
        self.successor_codebook = None


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        )


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype = torch.half) -> torch.Tensor:
        # The selector sits in the model's module list so the loader reaches its weights, and
        # the module walk visits every entry in order — here it must leave the state untouched.
        # Selection happens via select(), called from propose() with the draft block states.
        return x
