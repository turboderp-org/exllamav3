from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from .. import Module, Linear
from ...model import Config
from ...util.tensor import get_for_device, to2


class MuseGlimmerVisionPatchEmbedder(Module):
    """
    Linear patch embedding + bilinearly interpolated learned position embedding, followed by the
    window-attention reorder (spatial merge unit 1). The bilinear gather indices/weights and the
    window index are computed host-side per image and passed in params.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        patch_dim: int,
        out_dtype: torch.dtype = torch.float,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self.position_embedding_key = f"{key}.position_embedding_table.weight"
        self.position_embedding_table = None
        self.position_embedding_numel = 0

        self.patch_embedding = Linear(
            config = config,
            key = f"{key}.patch_embedding",
            in_features = patch_dim,
            out_features = hidden_size,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.patch_embedding)


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
    def get_tensors(self):
        # The Linear submodule exports itself; the embedding table is this module's own tensor
        if self.position_embedding_table is None:
            return {}
        return {self.position_embedding_key: self.position_embedding_table.contiguous()}


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        y = self.patch_embedding.forward(x.half(), params, out_dtype = torch.half).float()

        # Interpolated position embedding, computed in fp32 like the reference
        bilinear_indices = get_for_device(params, "bilinear_indices", self.device)
        bilinear_weights = get_for_device(params, "bilinear_weights", self.device)
        table = self.position_embedding_table
        pos_emb = (table[bilinear_indices].float() * bilinear_weights[:, :, None]).sum(0)
        y += pos_emb.unsqueeze(0)

        # Reorder sequence for window attention, and the rotary grid along with it
        window_index = get_for_device(params, "window_index", self.device, None)
        if window_index is not None:
            y = y[:, window_index]
            inv_freq = params.get("inv_freq")
            if inv_freq is not None:
                params["inv_freq"] = inv_freq[window_index.to(inv_freq.device)]

        return to2(y, out_dtype, self.out_dtype)


class MuseGlimmerVisionPixelShuffle(Module):
    """
    Restores the pre-window ordering (inverse of the patch embedder's reorder), then merges each
    (merge_size x merge_size) block of patches into one token by channel-major concatenation.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        merge_size: int,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)
        self.merge_size = merge_size
        self.out_dtype = out_dtype


    @override
    def optimizer_targets(self):
        return []


    @override
    def weights_numel(self):
        return 0


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        bsz, seq_len, dim = x.shape
        f = self.merge_size

        window_index = get_for_device(params, "window_index", self.device, None)
        if window_index is not None:
            reverse_indices = torch.argsort(window_index)
            x = x[:, reverse_indices]

        grids = params["grid_thw"]
        output = []
        offset = 0
        for t, h, w in grids.tolist():
            t, h, w = int(t), int(h), int(w)
            n_tokens = t * h * w
            chunk = x[0, offset : offset + n_tokens]
            n_out_per_frame = (h // f) * (w // f)
            ds_perm = torch.arange(h * w, device = x.device)
            ds_perm = ds_perm.view(h // f, f, w // f, f).permute(0, 2, 1, 3).reshape(-1)
            if t > 1:
                frame_offsets = (torch.arange(t, device = x.device) * h * w).view(t, 1)
                ds_perm = (ds_perm.unsqueeze(0) + frame_offsets).reshape(-1)
            chunk = chunk[ds_perm]
            chunk = chunk.view(t * n_out_per_frame, f * f, dim)
            chunk = chunk.permute(0, 2, 1).reshape(t * n_out_per_frame, dim * f * f)
            output.append(chunk)
            offset += n_tokens

        y = torch.cat(output, dim = 0).unsqueeze(0)
        return to2(y, out_dtype, self.out_dtype)


class MuseGlimmerVisionAdapter(Module):
    """
    fc1 -> gelu -> fc2 -> gelu, the first stage of the multimodal projection. The projection into
    the text embedding space and the trailing scaleless RMSNorm live in separate modules so every
    tensor stays under its own module's key prefix (checkpoint layout puts vision_projection
    outside model.vision_adapter, and the compiler collects tensors by module-key prefix).
    """

    def __init__(
        self,
        config: Config,
        key: str,
        in_size: int,
        interm_size: int,
        out_dtype: torch.dtype = torch.half,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.out_dtype = out_dtype

        self.fc1 = Linear(
            config = config,
            key = f"{key}.fc1",
            in_features = in_size,
            out_features = interm_size,
            qmap = qmap + ".fc1" if qmap else None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.fc2 = Linear(
            config = config,
            key = f"{key}.fc2",
            in_features = interm_size,
            out_features = interm_size,
            qmap = qmap + ".fc2" if qmap else None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.fc1)
        self.register_submodule(self.fc2)


    @override
    def optimizer_targets(self):
        return []


    @override
    def weights_numel(self):
        return self.fc1.weights_numel() + self.fc2.weights_numel()


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        y = self.fc1.forward(x.half(), params)
        y = F.gelu(y)
        y = self.fc2.forward(y, params)
        y = F.gelu(y)
        return to2(y, out_dtype, self.out_dtype)
