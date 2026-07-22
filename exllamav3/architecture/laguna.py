from __future__ import annotations
from typing_extensions import override
import torch

from ..cache.recurrent_util import prepare_for_recurrence
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    BlockSparseMLP,
    Linear,
    GatedMLP,
    SlidingAttention,
    SWAState
)
from ..modules.attn import prepare_for_attn


class LagunaConfig(Config):
    arch_string = "LagunaForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": LagunaModel},
            **kwargs
        )

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # Attention params. Full-attention and SWA layers have different q-head counts (48 vs 64
        # in Laguna-XS), given per layer
        self.head_dim = self.read_cfg(int, "head_dim", no_default)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", no_default)
        self.num_q_heads_list = self.read_cfg(list, "num_attention_heads_per_layer", None)
        if self.num_q_heads_list is None:
            self.num_q_heads_list = [self.num_q_heads] * self.num_hidden_layers

        # Attention output gate: o *= softplus(g_proj(x)), one gate value per head. The config
        # class also defines "per-element" (gate per channel) which no released checkpoint uses
        self.assert_cfg(str, "gating", "per-head")

        # Layer types / sliding window
        self.layer_types = self.read_cfg(list, "layer_types", no_default)
        self.sliding_window = self.read_cfg(int, "sliding_window", -1)
        self.assert_cfg(bool, "swa_attention_sink_enabled", False, True)

        # RoPE is given per layer type, nested in rope_parameters (transformers v5 style)
        rope_parameters = self.read_cfg(dict, "rope_parameters", no_default)

        def rope_settings_for(layer_type):
            synth = dict(self.config_dict)
            synth["rope_parameters"] = rope_parameters[layer_type]
            return self.read_rope_settings_default(
                RopeStyle.NEOX,
                config_dict = synth,
                override_type = "default" if layer_type == "sliding_attention" else None,
            )

        rope_settings_by_type = {lt: rope_settings_for(lt) for lt in set(self.layer_types)}
        self.rope_settings_list = [rope_settings_by_type[lt] for lt in self.layer_types]

        # MLP params. mlp_only_layers are dense (layer 0); every other layer is MoE
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.shared_expert_intermediate_size = self.read_cfg(int, "shared_expert_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.mlp_only_layers = set(self.read_cfg(list, "mlp_only_layers", [0]))
        self.routed_scaling_factor = self.read_cfg(float, "moe_routed_scaling_factor", 1.0)
        self.assert_cfg(int, "decoder_sparse_step", 1, True)
        self.assert_cfg(bool, "norm_topk_prob", True)
        self.assert_cfg(bool, "moe_apply_router_weight_on_input", False, True)
        self.assert_cfg(float, "moe_router_logit_softcapping", 0.0, True)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", 1e-6)


class LagunaModel(Model):
    config_class = LagunaConfig

    def __init__(
        self,
        config: LagunaConfig,
        key_prefix: str = "model",
        swa_full: bool = False,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.swa_full = swa_full

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            is_moe = idx not in config.mlp_only_layers
            is_swa = config.layer_types[idx] == "sliding_attention"

            attn_args = dict(
                config = config,
                key = f"{key_prefix}.layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads_list[idx],
                num_kv_heads = config.num_kv_heads,
                rope_settings = config.rope_settings_list[idx],
                sm_scale = None,
                sliding_window = config.sliding_window if is_swa else -1,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                key_g = "g_proj",
                gate_softplus = True,
                qmap = "block.attn",
                q_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                out_dtype = torch.float,
            )

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = (
                        Attention(**attn_args)
                        if swa_full or not is_swa else
                        SlidingAttention(**attn_args)
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = (
                        BlockSparseMLP(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.moe_intermediate_size,
                            num_experts = config.num_experts,
                            num_experts_per_tok = config.num_experts_per_tok,
                            router_type = "dots",
                            routed_scaling_factor = config.routed_scaling_factor,
                            key_up = "experts.{expert_idx}.up_proj",
                            key_gate = "experts.{expert_idx}.gate_proj",
                            key_down = "experts.{expert_idx}.down_proj",
                            key_routing_gate = "gate",
                            key_e_score_bias = "experts.e_score_correction_bias",
                            qmap = "block.mlp",
                            interm_div = 128.0,  # Routed-expert activations overflow fp16
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            shared_experts = GatedMLP(
                                config = config,
                                key = f"{key_prefix}.layers.{idx}.mlp.shared_expert",
                                hidden_size = config.hidden_size,
                                intermediate_size = config.shared_expert_intermediate_size,
                                key_up = "up_proj",
                                key_gate = "gate_proj",
                                key_down = "down_proj",
                                qmap = "block.mlp",
                                interm_dtype = torch.half,
                                out_dtype = torch.float,
                                select_hq_bits = 2,
                            ),
                        )
                        if is_moe else
                        GatedMLP(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.intermediate_size,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            select_hq_bits = 1,
                        )
                    ),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embed_tokens"

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{key_prefix}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        # SWA layers are recurrent, optionally
        self.recurrent_state_cls = None
        if not self.swa_full:
            self.caps.update({
                "recurrent_states": True,
                "default_recurrent_checkpoint_interval": 6144,
            })
            self.recurrent_state_cls = SWAState


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if not self.swa_full:
            prepare_for_recurrence(input_ids, params, self)
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "〈|EOS|〉"
        if system_prompt:
            p += f"<system>{system_prompt}</system>\n"
        p += f"<user>{prompt}</user>\n"
        p += f"<assistant>"
        return p
