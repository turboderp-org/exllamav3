from __future__ import annotations
from typing_extensions import override
import torch.nn.functional as F
from .. import Module, Linear
from ...model import Config
import torch
from ...util.tensor import get_for_device, to2


class Gemma4VisionPatchEmbedder(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        patch_dim: int,
        position_embedding_size: int,
        out_dtype: torch.dtype = torch.float
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.position_embedding_size = position_embedding_size
        self.position_embedding_key = f"{key}.position_embedding_table"
        self.position_embedding_table = None
        self.position_embedding_numel = 0
        self.out_dtype = out_dtype

        self.input_proj = Linear(
            config = config,
            key = f"{key}.input_proj",
            in_features = patch_dim,
            out_features = hidden_size,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.input_proj)


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.position_embedding_table = self.config.stc.get_tensor(
            self.position_embedding_key,
            device,
            float2half = True,
            allow_bf16 = True,
        )
        self.position_embedding_numel = self.position_embedding_table.numel()


    @override
    def unload(self):
        super().unload()
        self.position_embedding_table = None
        self.position_embedding_numel = 0


    @override
    def weights_numel(self):
        return super().weights_numel() + self.position_embedding_numel


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        # Pixel values in range -1..1
        x = 2.0 * (x - 0.5)
        y = self.input_proj.forward(x.half(), params, out_dtype = torch.half)

        # Position IDs
        position_ids = get_for_device(params, "position_ids", self.device)
        pos_x = position_ids[..., 0].reshape(-1)
        pos_y = position_ids[..., 1].reshape(-1)

        # Table is 2x 1D learned embeddings (for x and y tile index, respectively)
        table = self.position_embedding_table
        pos_emb = table[0].index_select(0, pos_x) + table[1].index_select(0, pos_y)
        pos_emb = pos_emb.view(position_ids.shape[0], position_ids.shape[1], self.hidden_size)

        y = to2(y, out_dtype, self.out_dtype)
        y += pos_emb
        return y


class Gemma4VisionPooler(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        key_std_bias: str | None = None,
        key_std_scale: str | None = None,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.std_bias_key = f"{key}.{key_std_bias}" if key_std_bias else None
        self.std_scale_key = f"{key}.{key_std_scale}" if key_std_scale else None
        self.std_bias = None
        self.std_scale = None
        self.numel = 0
        assert bool(self.std_bias_key) == bool(self.std_scale_key), \
            "Must have both std_bias and std_scale or neither"
        self.has_bias_scale = bool(self.std_bias_key)


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.has_bias_scale:
            self.std_bias = self.config.stc.get_tensor(self.std_bias_key, device, allow_bf16 = True)
            self.std_scale = self.config.stc.get_tensor(self.std_scale_key, device, allow_bf16 = True)


    @override
    def weights_numel(self):
        return 2 * self.hidden_size if self.has_bias_scale else 0


    @override
    def unload(self):
        super().unload()
        self.std_bias = None
        self.std_scale = None


    @override
    def optimizer_targets(self):
        return []


    @override
    def get_tensors(self):
        if self.has_bias_scale:
            return {
                self.std_bias_key: self.std_bias.contiguous(),
                self.std_scale_key: self.std_scale.contiguous(),
            }
        else:
            return {}


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        position_ids = get_for_device(params, "position_ids", self.device)
        output_length = int(params["image_output_length"])
        if output_length > x.shape[1]:
            raise ValueError(f"Cannot pool {x.shape[1]} patches to {output_length} soft tokens.")

        if x.shape[1] != output_length:
            input_seq_len = x.shape[1]
            k = int((input_seq_len // output_length) ** 0.5)
            k_squared = k ** 2
            if k_squared * output_length != input_seq_len:
                raise ValueError(f"Cannot pool {x.shape} to {output_length}: {k=}^2 mismatch")
            max_x = position_ids[..., 0].max(dim = -1, keepdim = True)[0] + 1
            kernel_idxs = torch.div(position_ids, k, rounding_mode = "floor")
            kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
            weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared
            x = weights.transpose(1, 2) @ x

        x = x * (self.hidden_size ** 0.5)

        if self.has_bias_scale:
            x -= self.std_bias
            x *= self.std_scale

        return to2(x, out_dtype, torch.float)