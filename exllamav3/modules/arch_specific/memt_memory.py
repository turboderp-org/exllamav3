from __future__ import annotations
from typing_extensions import override
from collections import OrderedDict
import dataclasses
import torch
from ...model.config import Config
from .. import Module
from ...model.model_tp_alloc import TPAllocation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...training.memt import MemTSpec


import os
_graph_enable = os.environ.get("EXL3_MEMT_GRAPH", "1") != "0"   # CUDA-graph the decode readout
_GRAPH_MAX_BSZ = 8
_GRAPH_MAX_QLEN = 4


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
    Reading is additive before a linear map, so decode keeps the graph-captured
    attention block and adds ``o_proj(readout)`` to its output; only commits
    (prefill-sized anyway) take the python path.

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

    Limitations of this first cut: the commit scan is the sequential fp32 reference (fine for
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
        self._graphs = {}   # (bsz, seqlen, dtype) -> (CUDAGraph, static x, static state, static out)
        # Tensor parallelism: this shard holds memory heads [tp_head_first, tp_head_first + spec.num_heads) of
        # the full block, and reads/writes states through the rank's registry (key -> {layer_idx: [1, Hl, D, D]})
        # instead of tensors in params (see tp_import, model_tp_fn.mp_memt_put_state)
        self.tp_head_first = 0
        self.tp_registry: OrderedDict | None = None

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
        self._graphs = {}

    @override
    def get_tensors(self):
        if self.weights is None:
            return {}
        return {f"{self.key}.{n}": v for n, v in self.weights.to_dict().items()}

    @override
    def weights_numel(self):
        return self._numel

    # --- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype = torch.half) -> torch.Tensor:
        raise NotImplementedError("MemTMemory is driven by its Attention block through fuse()")

    @staticmethod
    def active(params: dict) -> bool:
        return params.get("memt_read") is not None or params.get("memt_commit") is not None

    # --- TP state registry ---------------------------------------------------

    def _tp_state(self, key: bytes) -> torch.Tensor:
        entry = self.tp_registry.get(key) if self.tp_registry is not None else None
        if entry is None or self.layer_idx not in entry:
            raise KeyError(f"{self.key}: MemT state {key.hex() if key else key} is not registered on this rank")
        return entry[self.layer_idx]

    def _tp_rows(self, keys: list) -> torch.Tensor | None:
        """[bsz, Hl, D, D] for a batch of registry keys (None rows read the zero state); None if no row reads"""
        if all(k is None for k in keys):
            return None
        if len(keys) == 1:
            return self._tp_state(keys[0])
        ref = next(self._tp_state(k) for k in keys if k is not None)
        rows = [self._tp_state(k)[0] if k is not None else torch.zeros_like(ref[0]) for k in keys]
        return torch.stack(rows, dim = 0)

    def readout(self, x: torch.Tensor, params: dict) -> torch.Tensor | None:
        """
        The memory readout for ``x`` (the attention block's input after its pre-norm, ``[bsz, seqlen,
        hidden]``): ``[bsz, seqlen, H*D]`` in bf16, or None when this forward neither reads nor commits
        on this layer. Performs the commit scan when requested. The readout is added to the gated
        attention context before ``o_proj``, so it is also what the graph-captured decode path adds
        through ``o_proj`` after the fact.
        """
        read = params.get("memt_read")
        commit = params.get("memt_commit")
        if read is None and commit is None:
            return None
        assert self.weights is not None, f"{self.key}: memory block not loaded"
        tp_read = read is not None and "tp_keys" in read
        tp_commit = commit is not None and "tp_in" in commit
        rs = (self._tp_rows(read["tp_keys"]) if tp_read else read.get(self.layer_idx)) if read is not None else None
        ws = (self._tp_state(commit["tp_in"]) if tp_commit else commit.get(self.layer_idx)) if commit is not None else None
        if rs is None and ws is None:
            return None
        dev = x.device
        rs_d = rs.to(dev, torch.float32) if rs is not None else None
        ws_d = ws.to(dev, torch.float32) if ws is not None else None
        if ws is None and _graph_enable and x.shape[0] <= _GRAPH_MAX_BSZ and x.shape[1] <= _GRAPH_MAX_QLEN:
            return self._graph_readout(x, rs_d)
        # Projections in bf16 like the release (its block runs bf16 GEMMs and
        # keeps the norm/gate/decay/state math in fp32).
        xb = x.to(torch.bfloat16)
        out, nxt = _math().memt_block_forward(
            xb, self.weights, self.spec, rs_d,
            commit = ws is not None,
            write_state = ws_d,
        )
        if nxt is not None:
            if tp_commit:
                self.tp_registry.setdefault(commit["tp_out"], {})[self.layer_idx] = nxt.contiguous()
            else:
                commit[self.layer_idx] = nxt[0] if ws.dim() == 3 else nxt
        return out

    def _graph_readout(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Read-only readout replayed as a CUDA graph, one per (bsz, seqlen, dtype): the eager readout is a
        few dozen small kernels per layer (~0.45 ms each on a 3090, ~4.5 ms per token over the ten memory
        layers), the replay ~0.18 ms. Inputs are copied into the graph's static buffers, so any state works.
        """
        bsz, seqlen, _ = x.shape
        if state.dim() == 3:
            state = state.unsqueeze(0)
        state = state.expand(bsz, *state.shape[1:])
        key = (bsz, seqlen, x.dtype)
        g = self._graphs.get(key)
        if g is None:
            # Capture on this layer's device (the model is split over several); the default capture
            # stream would sit on the current device and silently record nothing for the others
            with torch.cuda.device(x.device):
                xs = x.clone()
                ss = state.contiguous().clone()
                fwd = lambda: _math().memt_block_forward(xs.to(torch.bfloat16), self.weights, self.spec, ss)[0]
                side = torch.cuda.Stream(device = x.device)
                side.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(side):
                    for _ in range(3):
                        fwd()
                torch.cuda.current_stream(x.device).wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream = side):
                    out = fwd()
                torch.cuda.current_stream(x.device).wait_stream(side)
            g = self._graphs[key] = (graph, xs, ss, out)
        graph, xs, ss, out = g
        with torch.cuda.device(x.device):
            xs.copy_(x)
            ss.copy_(state)
            graph.replay()
        return out

    def fuse(self, x: torch.Tensor, o: torch.Tensor, params: dict) -> torch.Tensor:
        """``o`` (the gated attention context, ``[bsz, seqlen, H*D]``) plus the memory readout, same dtype"""
        out = self.readout(x, params)
        return o if out is None else o + out.to(o.dtype)

    # --- Tensor parallelism -------------------------------------------------
    # The block is per head: head h owns rows [h*D, (h+1)*D) of the q/k/v/erase/write sections of in_proj,
    # of f_down / g_down / decay_bias, entry h of decay_scale_log, and state slice [h]. The two latent
    # sections of in_proj (decay, gate) and o_norm are shared and replicated. Memory heads line up with the
    # attention's query heads, so a rank keeps the memory heads of the query heads it owns and its readout
    # joins the local attention context before the split o_proj; the layer's existing all-reduce sums it.

    def tp_export(self, plan, producer):
        assert self.weights is not None, f"{self.key}: cannot export an unloaded memory block"
        return {
            "cls": MemTMemory,
            "kwargs": {"key": self.key, "layer_idx": self.layer_idx},
            "spec": dataclasses.asdict(self.spec),
            "weights": {n: producer.send(t) for n, t in self.weights.to_dict().items()},
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan, head_first: int, head_last: int):
        from ...training.memt import MemTSpec, MemTWeights, memt_in_proj_slices
        consumer = local_context["consumer"]
        device = local_context["device"]
        full = MemTSpec(**exported["spec"])
        D = full.head_dim
        spec = dataclasses.replace(full, num_heads = head_last - head_first)
        w = exported["weights"]
        recv = lambda desc, a, b: consumer.recv(desc, cuda = True, slice_dim = 0, first = a, last = b)
        sl = memt_in_proj_slices(full)
        parts = [recv(w["in_proj.weight"], sl[n].start + head_first * D, sl[n].start + head_last * D)
                 for n in ("q", "k", "v", "erase", "write")]
        parts += [recv(w["in_proj.weight"], sl[n].start, sl[n].stop) for n in ("decay_latent", "gate_latent")]
        weights = MemTWeights(
            in_proj = torch.cat(parts, dim = 0).contiguous(),
            f_down = recv(w["f_down.weight"], head_first * D, head_last * D).contiguous(),
            g_down = recv(w["g_down.weight"], head_first * D, head_last * D).contiguous(),
            decay_scale_log = recv(w["decay_scale_log"], head_first, head_last).contiguous(),
            decay_bias = recv(w["decay_bias"], head_first * D, head_last * D).contiguous(),
            o_norm = consumer.recv(w["o_norm.weight"], cuda = True),
        )
        weights.check(spec)
        module = MemTMemory(None, exported["kwargs"]["key"], exported["kwargs"]["layer_idx"], spec)
        module.device = device
        module.weights = weights
        module.tp_head_first = head_first
        module.tp_registry = local_context.setdefault("memt_states", OrderedDict())
        local_context.setdefault("memt_modules", []).append(module)
        torch.cuda.synchronize()
        return module

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError("MemTMemory is split with its Attention block (see Attention.make_tp_allocation)")
