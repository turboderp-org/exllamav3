from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..cache import Cache
from ..ext import exllamav3_ext as ext
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn
from ..modules.module import no_p2p_copy
from ..util.tensor import get_for_device

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig


class Qwen3_5MTPModel(Model):

    def __init__(
        self,
        config: Qwen3_5Config | Qwen3_5MoeConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        # Module list: optional embed, then pre_fc norms + fc, then num_mtp_layers * TransformerBlock, then norm
        self.input_layer = Qwen3_5MTPInputLayer(
            config = config,
            key = "mtp",
            key_pre_fc_norm_hidden = "mtp.pre_fc_norm_hidden",
            key_pre_fc_norm_embedding = "mtp.pre_fc_norm_embedding",
            key_fc = "mtp.fc",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        )

        self.modules = [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.mtp_num_hidden_layers):
            attn = Attention(
                config = config,
                key = f"mtp.layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_kv_heads,
                rope_settings = config.rope_settings,
                sm_scale = None,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                qmap = "block.attn",
                q_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                out_dtype = torch.float,
                interleaved_gate = True,
                qbits_key = "mtp_bits",
            )
            self.attn_modules.append(attn)

            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = f"mtp.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"mtp.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        qbits_key = "mtp_bits",
                    ),
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # Final norm
        self.final_norm = RMSNorm(
            config = config,
            key = "mtp.norm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
            constant_bias = 1.0,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 2,  # MTP-1 with 2-step recurrence
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # MTP doesn't take input_ids through Embedding here — embedding is handled in step()
        # But prepare_for_attn still wires up flash-attn params
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and tell target to export hidden state.

        Qwen3.5/3.6 MTP consumes the trunk's post-final-norm hidden state. This differs from
        DeepSeek/GLM-style MTP heads, which consume a pre-norm residual stream.
        """
        self.input_layer.attached_model = weakref.ref(target)

        # Find the target's embedding (first module of class Embedding)
        target_embed = None
        for m in target.modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        # lm_head is the last module
        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        target_norm = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_norm, RMSNorm), "Expected target final RMSNorm immediately before lm_head"
        self.draft_verifier_params.update({
            "export_state_norm_keys": {target_norm.key},
        })


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}
