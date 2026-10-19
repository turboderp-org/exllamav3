from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..model.model import Model
from ..modules import RMSNorm, Embedding, TransformerBlock, Linear
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bailing_hybrid import BailingHybridConfig


class BailingHybridMTPModel(Model):
    """
    Ling 3.x MTP head (num_nextn_predict_layers = 1), DeepSeek-V3 shape stored as the layer past
    the trunk: enorm(embedding) and hnorm(target state) concatenated embedding-first into
    eh_proj (2H -> H), one MLA + MoE block like the trunk's full-attention layers, then
    final_layernorm before the shared lm_head. Per the bundled modeling, the target state it
    consumes is the trunk's POST-final-norm state, and the clamp lists do not reach it.
    """

    def __init__(
        self,
        config: BailingHybridConfig,
        key_prefix: str = "model",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        from .bailing_hybrid import bailing_mla, bailing_moe
        assert config.num_mtp_layers == 1, "Ling MTP: one prediction layer"
        idx = config.num_hidden_layers
        key = f"{key_prefix}.layers.{idx}"

        self.input_layer = Qwen3_5MTPInputLayer(
            config = config,
            key = f"{key}.input",
            key_pre_fc_norm_hidden = f"{key}.hnorm",
            key_pre_fc_norm_embedding = f"{key}.enorm",
            key_fc = f"{key}.eh_proj",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
            constant_bias = 0.0,
        )
        self.modules = [self.input_layer]
        self.first_block_idx = len(self.modules)
        self.modules.append(
            TransformerBlock(
                config = config,
                key = key,
                layer_idx = 0,
                attn_norm = RMSNorm(
                    config = config,
                    key = f"{key}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                attn = bailing_mla(config, f"{key}.attention", 0, qbits_key = "mtp_bits"),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{key}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                mlp = bailing_moe(config, f"{key}.mlp", 0.0, 0.0, qbits_key = "mtp_bits"),
            )
        )
        self.last_kv_module_idx = len(self.modules) - 1

        self.final_norm = RMSNorm(
            config = config,
            key = f"{key}.final_layernorm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 1,
            "autosplit_load_fwd": False,
        })
        self.calibration_all_experts = True
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # The input layer embeds; prepare_for_attn wires the attention params
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """Bind to the target: borrow its embedding and lm_head, and have it export the
        post-final-norm state hnorm consumes"""
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)
        target_embed = next((m for m in target.modules if isinstance(m, Embedding)), None)
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)
        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])
        target_norm = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_norm, RMSNorm), "Expected target final RMSNorm immediately before lm_head"
        self.draft_verifier_params = {
            "export_state_norm_keys": {target_norm.key},
        }


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        # The target's head, TP-aware; exports draft confidence when the generator asks
        return self.attached_model().lm_head_argmax(state, params)
