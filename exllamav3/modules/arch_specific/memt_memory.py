from __future__ import annotations
from typing_extensions import override
import torch
from ...model.config import Config
from .. import Module
from ...model.model_tp_alloc import TPAllocation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...training.memt import MemTSpec


def _math():
    # The math lives with the training code (plain torch, CPU-tested standalone). Imported lazily:
    # the training package is heavy and this module is imported while the exllamav3 package is
    # still loading (through the architecture registry)
    from ...training import memt
    return memt


class MemTMemory(Module):
    """
    MemT-1.0 native memory block of one full-attention layer (MemT-1.0-Nano on
    Qwen3.6-35B-A3B). Experimental; see doc/memt_plan.md.

    Holds the six released tensors (bf16, unquantized, ~90 MB per layer for
    Nano) and fuses the memory readout into the attention context right before
    ``o_proj`` -- after the backbone's own sigmoid attention gate, exactly
    where the released SGLang integration injects it (``o_proj`` pre-hook).

    Session state is NOT held here. The caller passes it through ``params``:

      params["memt_read"]   dict {layer_idx: state}  -- the snapshot every
                            token of this forward reads. ``state`` is fp32
                            ``[H, D, D]`` (one trajectory, broadcast over the
                            batch) or ``[bsz, H, D, D]``.
      params["memt_commit"] dict {layer_idx: state}  -- when present, this
                            forward also runs the stable GDN-2 scan over its
                            tokens from the given write state and stores the
                            result back into the dict (same key, same rank).
                            Absent during ordinary generation.

    Without either key the module is a no-op (zero readout). With a zero
    state it is an exact no-op as well: the forward is then the plain
    backbone, which is how the memory-free parity gate runs.

    Limitations of this first cut: python path only (the graph-captured
    decode block is bypassed while a state is attached), no tensor
    parallelism, the commit scan is the sequential fp32 reference (fine for
    turn-sized commits; the chunked fla gdn2 kernels are the follow-up), and
    batched requests must all share the state (or pass ``[bsz, H, D, D]``).
    """

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        spec: "MemTSpec",
    ):
        super().__init__(config, key, None)
        self.module_name = "MemTMemory"
        self.layer_idx = layer_idx
        self.spec = spec
        self.weights = None
        self._numel = _math().memt_param_count(spec)

    def tensor_keys(self) -> dict[str, str]:
        return {name: f"{self.key}.{name}" for name in _math().memt_param_shapes(self.spec)}

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        t = {}
        for name, key in self.tensor_keys().items():
            t[name] = self.config.stc.get_tensor(key, device, allow_bf16 = True, no_defer = True)
        w = _math().MemTWeights.from_dict(t)
        w.check(self.spec)
        self.weights = w

    @override
    def unload(self):
        self.device = None
        self.weights = None

    @override
    def get_tensors(self):
        if self.weights is None:
            return {}
        return {f"{self.key}.{n}": v for n, v in self.weights.to_dict().items()}

    @override
    def weights_numel(self):
        return self._numel

    # --- forward -----------------------------------------------------------

    @staticmethod
    def active(params: dict) -> bool:
        return params.get("memt_read") is not None or params.get("memt_commit") is not None

    def fuse(self, x: torch.Tensor, o: torch.Tensor, params: dict) -> torch.Tensor:
        """
        ``x``: the attention block's input after its pre-norm, ``[bsz, seqlen,
        hidden]``; ``o``: the gated attention context, ``[bsz, seqlen, H*D]``.
        Returns ``o`` plus the memory readout (same dtype), and performs the
        commit scan when requested.
        """
        read = params.get("memt_read")
        commit = params.get("memt_commit")
        if read is None and commit is None:
            return o
        assert self.weights is not None, f"{self.key}: memory block not loaded"
        rs = read.get(self.layer_idx) if read is not None else None
        ws = commit.get(self.layer_idx) if commit is not None else None
        if rs is None and ws is None:
            return o
        dev = x.device
        rs_d = rs.to(dev, torch.float32) if rs is not None else None
        ws_d = ws.to(dev, torch.float32) if ws is not None else None
        # Projections in bf16 like the release (its block runs bf16 GEMMs and
        # keeps the norm/gate/decay/state math in fp32).
        xb = x.to(torch.bfloat16)
        out, nxt = _math().memt_block_forward(
            xb, self.weights, self.spec, rs_d,
            commit = ws is not None,
            write_state = ws_d,
        )
        if nxt is not None:
            commit[self.layer_idx] = nxt[0] if ws.dim() == 3 else nxt
        return o + out.to(o.dtype)

    # --- TP (unsupported) ---------------------------------------------------

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError("MemT memory layers do not support tensor parallelism yet")

    def tp_export(self, plan, producer):
        raise NotImplementedError("MemT memory layers do not support tensor parallelism yet")
