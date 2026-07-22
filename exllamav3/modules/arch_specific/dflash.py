from __future__ import annotations
from typing_extensions import override
import torch
from .. import LayerNorm
from ...model.config import Config
from ...model.model_tp_fn import mp_model_forward_embedding
from ...modules import Module, Linear, RMSNorm
from ...util.rope import RopeSettings, RoPE
from ...util.tensor import get_for_device, to2

class DFlashInputLayer(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        key_norm: str,
        hidden_size: int,
        target_state_size: int,
        mask_token_id: int,
        rms_norm_eps: float,
        native_draft_len: int,
        out_dtype: torch.dtype | None = torch.float,
        qmap: str | None = None,
        key_aux_norms: str | None = None,
        num_aux_norms: int = 0,
    ):
        super().__init__(config, key, None)
        self.module_name = "DFlashInputLayer"
        self.qmap = qmap
        self.key = key
        self.hidden_size = hidden_size
        self.target_state_size = target_state_size
        self.out_dtype = out_dtype
        self.native_draft_len = native_draft_len

        self.proj = Linear(
            config = config,
            key = f"{key}",
            in_features = self.target_state_size,
            out_features = self.hidden_size,
            qmap = (qmap + ".input") if qmap else None,
            out_dtype = out_dtype,
            pad_to = 1
        )

        self.norm = RMSNorm(
            config = config,
            key = f"{key_norm}",
            rms_norm_eps = rms_norm_eps,
            out_dtype = out_dtype,
        )

        self.register_submodule(self.proj)
        self.register_submodule(self.norm)

        # Per-tap norms (Laguna DFlash): each captured target hidden state is RMS-normed
        # individually before the taps are concatenated and projected by fc
        self.aux_norms = []
        if key_aux_norms is not None:
            for i in range(num_aux_norms):
                aux_norm = RMSNorm(
                    config = config,
                    key = f"{key_aux_norms}.{i}",
                    rms_norm_eps = rms_norm_eps,
                    out_dtype = torch.half,
                )
                self.aux_norms.append(aux_norm)
                self.register_submodule(aux_norm)

        self.mask_token_id = mask_token_id

        # Populated by attach_to()
        self.attached_model = None

        self.caps.update({"x_cpu": True})


    def optimizer_targets(self):
        raise NotImplementedError()


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)


    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ):
        bsz, seqlen = x.shape
        noise_mask = torch.full((bsz, self.native_draft_len - 1), self.mask_token_id, dtype = torch.long)
        x = torch.cat((x, noise_mask), dim = -1)
        if not self.attached_model().loaded_tp:
            x = self.attached_model().modules[0].forward(x, params)
        else:
            x = self.attached_model().tp_producer.send(x)
            x = self.attached_model().tp_dispatch_master(mp_model_forward_embedding, (x, params))
        return x
