from __future__ import annotations
import torch.nn.functional as F
from ...util.tensor import to2
from .. import Module, Linear
from ...model import Config
import torch
from typing_extensions import override

class Mistral3PatchMerger(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        merge: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Mistral3PatchMerger"
        self.qmap = qmap

        self.merge = merge
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype

        self.merging_layer = Linear(
            config = config,
            key = f"{key}.merging_layer",
            in_features = hidden_size * merge ** 2,
            out_features = hidden_size
        )

        self.register_submodule(self.merging_layer)

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        bsz, seq_len, dim = x.shape
        h, w = params["features_size"]
        assert bsz == 1

        x = x.view(h, w, dim).permute(2, 0, 1).unsqueeze(0)
        x = F.unfold(x, kernel_size = self.merge, stride = self.merge)
        x = x.view(bsz, dim * self.merge ** 2, -1).transpose(1, 2).contiguous()
        x = self.merging_layer.forward(x, params)

        return to2(x, out_dtype, self.out_dtype)
