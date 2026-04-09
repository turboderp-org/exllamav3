from __future__ import annotations
from typing_extensions import override
import torch
from .. import Module
from ...model import Config
import torch.nn.functional as F

class Gemma3MMPool(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        patches_per_image: int,
        tokens_per_side: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Gemma3MMPool"
        self.qmap = qmap

        self.patches_per_image = patches_per_image
        self.tokens_per_side = tokens_per_side

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device: torch.device, **kwargs):
        pass

    @override
    def unload(self):
        pass

    @override
    def weights_numel(self):
        return 0

    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        bsz, _, seq_length = x.shape
        x = (
            x.transpose(1, 2)
            .view(bsz, seq_length, self.patches_per_image, self.patches_per_image)
        )
        kernel_size = self.patches_per_image // self.tokens_per_side
        x = (
            F.avg_pool2d(x, kernel_size = kernel_size, stride = kernel_size)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )
        return x
