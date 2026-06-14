from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache
from .. import LayerNorm
from ..module import no_p2p_copy
from ...model.config import Config
from ...model.model_tp_fn import mp_model_forward_embedding
from ...modules import Module, Linear, RMSNorm
from ...util.rope import RopeSettings, RoPE
from ...util.tensor import get_for_device, to2

class Qwen3_5MTPInputLayer(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        key_pre_fc_norm_hidden: str,
        key_pre_fc_norm_embedding: str,
        key_fc: str,
        hidden_size: int,
        rms_norm_eps: float,
        native_draft_len: int,
        out_dtype: torch.dtype | None = torch.float,
        qmap: str | None = None,
        qbits_key = "mtp_bits",
    ):
        super().__init__(config, key, None)
        self.module_name = "Qwen3_5MTPInputLayer"
        self.qmap = qmap
        self.key = key
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self.native_draft_len = native_draft_len
        self.key_pre_fc_norm_hidden = key_pre_fc_norm_hidden
        self.key_pre_fc_norm_embedding = key_pre_fc_norm_embedding
        self.key_fc = key_fc

        # Pre-fc norms applied to incoming hidden state and embeddings before they get concatenated
        self.pre_fc_norm_hidden = RMSNorm(
            config = config,
            key = f"{key_pre_fc_norm_hidden}",
            rms_norm_eps = rms_norm_eps,
            constant_bias = 1.0,
            out_dtype = torch.half,
        )
        self.pre_fc_norm_embedding = RMSNorm(
            config = config,
            key = f"{key_pre_fc_norm_embedding}",
            rms_norm_eps = rms_norm_eps,
            constant_bias = 1.0,
            out_dtype = torch.half,
        )

        # 2H -> H projection
        self.fc = Linear(
            config = config,
            key = f"{key_fc}",
            in_features = hidden_size * 2,
            out_features = hidden_size,
            qmap = "block.mtp.fc",
            out_dtype = out_dtype,
            pad_to = 1,
            qbits_key = qbits_key,
        )

        self.register_submodule(self.pre_fc_norm_hidden)
        self.register_submodule(self.pre_fc_norm_embedding)
        self.register_submodule(self.fc)

        # Populated by attach_to()
        self.attached_model = None

        self.caps.update({"x_cpu": True})


    def optimizer_targets(self):
        raise NotImplementedError()


    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ):
        # Norm for input state
        target_hidden = params.get("target_hidden")
        assert target_hidden is not None, "Qwen3.5 MTP requires target_hidden"
        assert target_hidden.shape[:-1] == x.shape, \
            f"Qwen3.5 MTP token/state shape mismatch: {tuple(x.shape)} vs {tuple(target_hidden.shape)}"
        y = get_for_device(params, "target_hidden", self.device)
        y = self.pre_fc_norm_hidden.forward(y, params)

        # Token embedding
        if not self.attached_model().loaded_tp:
            x = self.attached_model().modules[0].forward(x, params, out_dtype = torch.half)
        else:
            x = self.attached_model().tp_producer.send(x)
            x = self.attached_model().tp_dispatch_master(mp_model_forward_embedding, (x, params))
            x = x.half()
        x = self.pre_fc_norm_embedding.forward(x.to(self.device), params)

        # Project
        x = torch.cat((x.to(y.device), y), dim = -1)
        x = self.fc.forward(x, params)
        return to2(x, out_dtype, self.out_dtype)


    def get_compile_sizes(self, stc):
        return (
            stc.get_tensor_sizes(self.key_pre_fc_norm_hidden) +
            stc.get_tensor_sizes(self.key_pre_fc_norm_embedding) +
            stc.get_tensor_sizes(self.key_fc)
        )

    def get_compile_tensors(self, stc):
        r = {}
        r.update(stc.get_tensors(self.key_pre_fc_norm_hidden, allow_bf16 = True))
        r.update(stc.get_tensors(self.key_pre_fc_norm_embedding, allow_bf16 = True))
        r.update(stc.get_tensors(self.key_fc, allow_bf16 = True))
        return r
