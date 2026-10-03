"""
Park serving-only component models out of VRAM while the shared GPU trains.

A multimodal / MTP-capable checkpoint loads as several component ``Model``
instances sharing one ``Config``: the text trunk plus, when the host serves
them, a vision tower (``component="vision"``) and/or an MTP speculative-
decoding head (``component="mtp"`` / a draft model). Training only ever
touches the text trunk -- the LoRA targets live there and the vision/MTP
components are never targeted -- so during a training burst those components
are dead weight in VRAM (a vision tower + MTP head can be well over a GB on
the 16 GB cards this matters for).

:class:`ModelParker` evicts one such component for the duration of a burst
and brings it back afterwards, using the two module lifecycle operations that
every exllamav3 module already implements and that the autosplit loader's
OOM-rollback path already exercises as a cycle:

  * ``park()`` records each top-level module's device, calls its ``unload()``
    (which frees weights, RoPE tables, scratch buffers AND any KV cache
    layers allocated on the module -- an MTP draft cache comes along for
    free), and flushes the CUDA caching allocator.
  * ``unpark()`` re-runs ``module.load(device)`` against the model's
    safetensors collection, restoring every module to the exact device it
    was parked from, then closes the collection's file handles again.

Restore streams the weights back through the safetensors files rather than a
private host-RAM copy: the OS page cache holds the recently-read tensor data
in system RAM, so in practice unparking is a RAM -> VRAM copy at first-load
speed, with zero per-module-type code and no risk of missing a tensor a
handwritten swap would have to know about. Weights are value-exact by
construction (same loader, same source bytes).

Not supported: tensor-parallel-loaded models (``park()`` asserts) and
CPU-offloaded-MoE components (their expert weights already live in system
RAM through a worker process; nothing to park). Modules already on CPU
(e.g. ``prefer_cpu`` embeddings) are left where they are.

Contents of a parked component's KV cache layers are lost across the cycle
(freed with the module). For an MTP draft cache this is benign: drafted
tokens are always verified by the main model, and the default realtime flow
resets the generator's page table after every ingest anyway.
"""

from __future__ import annotations

import gc

import torch


def _free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ModelParker:
    """
    Park/unpark one loaded component ``Model`` (vision tower, MTP/draft
    head). Both operations are idempotent; construction does not park.
    """

    def __init__(self, model):
        self.model = model
        self._homes = None      # [(module, device)] while parked, else None

    @property
    def parked(self) -> bool:
        return self._homes is not None

    def park(self) -> None:
        """Unload every device-resident top-level module of the model and
        flush the CUDA caching allocator so the freed VRAM is returned."""
        if self._homes is not None:
            return
        assert not getattr(self.model, "loaded_tp", False), \
            "cannot park a tensor-parallel-loaded model"
        homes = []
        for module in self.model.modules:
            device = module.device
            if device is None or torch.device(device).type == "cpu":
                continue
            homes.append((module, device))
            module.unload()
        self._homes = homes
        if homes:
            _free_mem()

    def unpark(self) -> None:
        """Reload every parked module to the device it was parked from
        (weights come back through the safetensors collection -- the OS page
        cache makes this a system-RAM read in practice)."""
        if self._homes is None:
            return
        homes, self._homes = self._homes, None
        if not homes:
            return
        config = self.model.config
        stc = config.stc
        # Mirror the loader (model_ls.py): a vision tower loaded with
        # InferParams.vision_pinned keeps its linear weights in pinned host
        # memory (zero-copy device aliases); module.load() puts them back in
        # VRAM, so re-pin after every restore or the tower silently regrows.
        pin = (
            getattr(getattr(config, "infer_params", None), "vision_pinned", False) and
            getattr(self.model, "component", "text") == "vision"
        )
        for module, device in homes:
            defer = module.can_defer_load()
            if defer:
                stc.begin_deferred_load()
            try:
                module.load(device)
            except Exception:
                if defer:
                    stc.abort_deferred_load()
                raise
            if defer:
                stc.end_deferred_load()
            if pin:
                module.pin_linears()
        stc.close()
        # Mirror Model.load_gen(): release global shared scratch tensors that
        # module loading may have (re)created; modules keep their own refs.
        try:
            from ..util.tensor import g_tensor_cache
            g_tensor_cache.drop_all()
        except ImportError:
            # Standalone import (CPU tests load this file outside the
            # package); the shared-scratch release is an optimization only.
            pass
        _free_mem()
