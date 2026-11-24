import math

import torch
from typing_extensions import override

from ..cache.recurrent import host_copy
from ..util.tensor import get_for_device
from .gated_delta_net import GDNState
from .tensor_bundle import TensorBundle


class CommittedRecurrentLayerState:
    """One committed set of FP32 tensors per sequence slot."""

    def __init__(self, module, max_batch_size, max_history, cache_id):
        self.cache_id = cache_id
        self.shapes = module.state_shapes
        self.max_batch_size = max_batch_size
        self.device = None
        self.tensors = ()

    def get_checkpoint_size(self):
        return sum(math.prod(shape) for shape in self.shapes) * torch.float32.itemsize

    def storage_size(self):
        return self.max_batch_size * self.get_checkpoint_size()

    def alloc(self, device):
        self.device = device
        self.tensors = tuple(
            torch.zeros((self.max_batch_size, *shape), device = device, dtype = torch.float32)
            for shape in self.shapes
        )

    def free(self):
        self.device = None
        self.tensors = ()

    def get_state_tensors(self):
        return self.tensors

    @torch.inference_mode()
    def clear(self, slot):
        for tensor in self.tensors:
            tensor[slot].zero_()

    def stash(self, slot, position = 0):
        return tuple(host_copy(tensor[slot]) for tensor in self.tensors)

    @torch.inference_mode()
    def unstash(self, slot, stashed, position = 0):
        for tensor, saved in zip(self.tensors, stashed, strict = True):
            tensor[slot].copy_(saved)


class CommittedRecurrentState(GDNState):
    """Recurrent checkpoints for layers that update the committed slot in place."""

    @override
    def reset(self):
        super().reset()
        for layer in self.cache.get_all_recurrent_layers().values():
            layer.clear(self.slot)

    def can_stash_at(self, position):
        return position == self.position and self.last_history == 0


class RecurrentMixer(TensorBundle):
    """A mixer with committed FP32 state. Use CommittedRecurrentState for its cache."""

    def __init__(self, config, key, layer_idx, state_shapes, tensor_shapes, tensor_keys = None):
        super().__init__(config, key, tensor_shapes, tensor_keys)
        self.state_shapes = state_shapes
        self.layer_idx = layer_idx
        self.layer_state_cls = CommittedRecurrentLayerState
        self.recurrent_layers = []
        self.caps["recurrent_cache"] = True

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        for layer in self.recurrent_layers:
            layer.alloc(device)

    @override
    def unload(self):
        for layer in self.recurrent_layers:
            layer.free()
        super().unload()

    def state_slots(self, x, params):
        if params.get("recurrent_slots") is None:
            return None, None
        cache = params["cache"]
        layer = cache.get_recurrent_layer((self.layer_idx, params.get("layer_instance") or 0))
        return layer, get_for_device(params, "recurrent_slots", x.device)

    def load_state(self, x, params):
        layer, slots = self.state_slots(x, params)
        if layer is None:
            return None, None, tuple(
                x.new_zeros((x.shape[0], *shape), dtype = torch.float32) for shape in self.state_shapes
            )
        slots = slots.long()
        return layer, slots, tuple(tensor.index_select(0, slots) for tensor in layer.tensors)

    @staticmethod
    def save_state(layer, slots, state):
        if layer is not None:
            for target, source in zip(layer.tensors, state, strict = True):
                target.index_copy_(0, slots, source)
