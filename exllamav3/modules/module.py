from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from ..util.device_copy import to_device
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..model.config import Config
from ..model.model_tp_alloc import TPAllocation
from functools import cached_property

# Use host bounce when moving state from device to device in layer split

class Module(ABC):

    def __init__(
        self,
        config: Config | None,
        key: str,
        qmap: str | None,
    ):
        """
        :param config:
            Model config

        :param key:
            Tensor key, reflects name in .safetensors collection

        :param qmap:
            Label for the hidden state upon entry into the forward function. Used to collect states/Hessian data
            in linear layers during quantization, e.g. to allow sharing between Q/K/V projections that have the same
            input state.
        """
        # Imported here rather than at the top of the file: importing ..model.config at module scope would pull in
        # the model package while the modules package may still be mid-import
        if config is None:
            from ..model.config import NullConfig
            config = NullConfig()
        self.config = config
        self.key = key
        self.alt_key = None
        self.used_alt_key = False
        self.device = None
        self.modules = []
        self.caps = {}
        self.qmap = qmap
        self.num_slices = 1
        self.select_hq_bits = 0
        self.q_priority = 0
        self.q_half_bits = True     # may be quantized at a half-integer bitrate (all decode kernels have instances)
        self.layer_idx = None

    def __iter__(self):
        yield self
        for module in self.modules:
            yield from module

    def find_module(self, key: str):
        for module in self:
            if module.key == key:
                return module

    def can_defer_load(self):
        if len(self.modules) == 0: return True
        return all(module.can_defer_load() for module in self.modules)

    def load(self, device: torch.Device, **kwargs):
        self.device = device
        for module in self.modules:
            module.load(device, **kwargs)

    def pin_linears(self):
        """Move eligible linear-layer weights to pinned host memory, replacing them with
        zero-copy device aliases (see InferParams.vision_pinned). Called by the loader after
        a top-level module's (possibly deferred) load has fully materialized its tensors;
        recurses to Linear, which does the work. Everything else keeps its VRAM tensors."""
        for module in self.modules:
            module.pin_linears()

    def unload(self):
        self.device = None
        for module in self.modules:
            module.unload()

    # Tensor-parallel collectives. tp_reduce (set per module by tp_import) marks a module whose
    # forward ends with a collective over its per-rank partial outputs. tp_owner is the one rank
    # that holds the WHOLE module when the allocator placed it that way (max_devices = 1, or a
    # plan that happened to give every channel to one device): the other ranks then hold stubs
    # contributing zeros, and the collective is a byte-exact broadcast from the owner instead of
    # a sum through the reduce wire (which rounds fp32 outputs to bf16 on the native backend)
    tp_owner: int | None = None

    def tp_collect(self, backend, x: torch.Tensor, contribution: bool = True):
        if self.tp_owner is not None:
            backend.broadcast(x, self.tp_owner)
        else:
            backend.all_reduce(x, contribution)

    @staticmethod
    def tp_single_owner(local_context: dict, *keys: str) -> int | None:
        """
        The single rank whose plan slice is non-empty for every one of keys, or None when the
        module is split across ranks (or its parts are owned by different ranks)
        """
        plan = local_context.get("plan")
        devices = local_context.get("active_devices")
        if plan is None or devices is None:
            return None
        owner = None
        for key in keys:
            owners = [d for d in devices if key in plan[d] and plan[d][key][1] > plan[d][key][0]]
            if len(owners) != 1 or (owner is not None and owners[0] != owner):
                return None
            owner = owners[0]
        return owner

    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        if x.device != self.device:
            # Pinned CPU sources (e.g. the generator's staged input IDs) upload without
            # blocking the host; the copy is stream-ordered ahead of the consuming kernels.
            # Device-to-device moves go through to_device, which bounces them via host memory
            # on platforms where peer copies corrupt data (probed, or EXLLAMA_NO_P2P_COPY)
            nb = x.device.type == "cpu" and x.is_pinned()
            x = to_device(x, self.device, non_blocking = nb)
        return x

    def get_qmaps(self):
        sq = set()
        if self.qmap:
            sq.add(self.qmap)
        for m in self.modules:
            sq.update(m.get_qmaps())
        return sq

    def get_tensors(self):
        return {}

    def weights_numel(self):
        return sum(m.weights_numel() for m in self.modules)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype = torch.half
    ) -> torch.Tensor:
        pass

    def register_submodule(self, module: Module | None):
        if module is not None:
            self.modules.append(module)

    def quant_format_id(self):
        return None

    def can_fuse_residual(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        # Overridden by norm modules that support forward(residual_in = ...)
        return False

    def get_name(self):
        return self.__class__.__name__

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        tpa_list = []
        for m in self.modules:
            tpa_list += m.make_tp_allocation(options)
        return tpa_list

    def tp_export(self, plan, producer):
        """
        Create serializable (dict) collection of module parameters and shared weights to pass to child process.
        """
        raise NotImplementedError()

    @staticmethod
    def tp_import(local_context, plan, loaded):
        """
        Reconstruct module in child process from exported parameters and shared weights, sliced as necessary for
        TP according to plan.
        """
        raise NotImplementedError()

    @cached_property
    def _all_cache_modules(self) -> list[Module]:
        return [m for m in self if m.caps.get("kv_cache")]
    def all_cache_modules(self):
        return self._all_cache_modules

    @cached_property
    def _all_recurrent_modules(self) -> list[Module]:
        return [m for m in self if m.caps.get("recurrent_cache")]
    def all_recurrent_modules(self):
        return self._all_recurrent_modules

    @abstractmethod
    def optimizer_targets(self):
        pass

    def get_compile_sizes(self, stc):
        return stc.get_tensor_sizes(self.key)

    def get_compile_tensors(self, stc):
        return stc.get_tensors(self.key, allow_bf16 = True)

    def autosplit_extra_measure(self, params):
        """
        Extra measuring forwards for the autosplit loader.
        """
        pass