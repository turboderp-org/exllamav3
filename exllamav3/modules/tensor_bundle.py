import math

from typing_extensions import override

from .module import Module


class TensorBundle(Module):
    """Named checkpoint tensors carried through conversion with their original dtype."""

    def __init__(self, config, key, tensor_shapes, tensor_keys = None):
        super().__init__(config, key, None)
        self.tensor_shapes = tensor_shapes
        self.tensor_keys = tensor_keys or {}
        self.tensors = {}

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        for name, shape in self.tensor_shapes.items():
            key = f"{self.key}.{self.tensor_keys.get(name, name)}"
            tensor = self.config.stc.get_tensor(key, device, no_defer = True, allow_bf16 = True)
            if tensor.shape != shape:
                raise ValueError(f"{key}: expected shape {shape}, got {tuple(tensor.shape)}")
            self.tensors[name] = tensor

    @override
    def unload(self):
        self.tensors.clear()
        super().unload()

    @override
    def weights_numel(self):
        return super().weights_numel() + sum(math.prod(shape) for shape in self.tensor_shapes.values())

    @override
    def get_tensors(self):
        return {
            f"{self.key}.{self.tensor_keys.get(name, name)}": tensor
            for name, tensor in self.tensors.items()
        }

    @override
    def optimizer_targets(self):
        return []
