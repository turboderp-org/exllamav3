from __future__ import annotations
from typing_extensions import override
import torch.nn.functional as F
from .. import Module, RMSNorm, Linear
from ...model import Config
import torch

class Qwen2_5VLVisionSpatialMerger(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        spatial_merge_unit: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.spatial_merge_unit = spatial_merge_unit
        self.out_dtype = out_dtype

    def optimizer_targets(self):
        raise NotImplementedError()

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

        window_index = params.get("window_index")
        if window_index is None:
            return x

        bsz, seq_len, dim = x.shape
        x = x.reshape(bsz * seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(bsz, seq_len, -1)

        inv_freq = params.get("inv_freq")
        if inv_freq is not None:
            inv_freq = inv_freq.reshape(bsz * seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            inv_freq = inv_freq[window_index, :, :]
            inv_freq = inv_freq.reshape(bsz, seq_len, -1)
            params["inv_freq"] = inv_freq

        return x


class Qwen2_5VLVisionPatchMerger(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        key_up: str,
        key_down: str,
        key_norm: str,
        hidden_size: int,
        merge_size: int,
        out_hidden_size: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.in_size = hidden_size * merge_size
        self.interm_size = hidden_size * merge_size
        self.out_size = out_hidden_size
        self.out_dtype = out_dtype

        self.up = Linear(
            config = config,
            key = f"{key}.{key_up}",
            in_features = self.in_size,
            out_features = self.interm_size,
            qmap = qmap + ".input",
            out_dtype = torch.half,
            pad_to = 1
        )
        self.down = Linear(
            config = config,
            key = f"{key}.{key_down}",
            in_features = self.interm_size,
            out_features = self.out_size,
            qmap = qmap + ".down",
            out_dtype = self.out_dtype,
            allow_input_padding = True,
            pad_to = 1
        )

        self.register_submodule(self.up)
        self.register_submodule(self.down)

        if key_norm:
            self.norm = RMSNorm(
                config = config,
                key = f"{key}.{key_norm}",
                rms_norm_eps = 1e-6,
                out_dtype = torch.half,
            )
            self.register_submodule(self.norm)

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def weights_numel(self):
        numel = self.up.weights_numel() + self.down.weights_numel()
        if self.norm: numel += self.norm.weights_numel()
        return numel

    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        bsz, seqlen, dim = x.shape
        y = self.norm.forward(x, params).to(torch.half)
        y = y.view(-1, self.in_size)

        y = self.up.forward(y, params)
        y = F.gelu(y, approximate = "tanh")
        y = self.down.forward(y, params)
        y = y.view(bsz, -1, self.out_size)

        return y
