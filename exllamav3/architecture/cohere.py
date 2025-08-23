from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import LayerNorm, Embedding, ParallelDecoderBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn

class CohereConfig(Config):
    arch_string = "CohereForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": CohereModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        self.use_qk_norm = self.read_cfg(int, "use_qk_norm", False)

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.layernorm_eps = self.read_cfg(float, "layer_norm_eps", 1e-05)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", True)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.GPTJ)

        # Logit scale
        self.logit_scale = self.read_cfg(float, "logit_scale", 0.0625)


class CohereModel(Model):
    config_class = CohereConfig

    def __init__(
        self,
        config: CohereConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = "model.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        self.modules += [
            ParallelDecoderBlock(
                config = config,
                key = f"model.layers.{idx}",
                out_dtype = torch.float,
                input_norm = LayerNorm(
                    config = config,
                    key = f"model.layers.{idx}.input_layernorm",
                    layernorm_eps = config.layernorm_eps,
                ),
                attn = Attention(
                    config = config,
                    key = f"model.layers.{idx}.self_attn",
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
                    qmap = "block.parallel",
                    q_norm = LayerNorm(
                        config = config,
                        key = f"model.layers.{idx}.self_attn.q_norm",
                        layernorm_eps = config.layernorm_eps,
                    ) if config.use_qk_norm else None,
                    k_norm = LayerNorm(
                        config = config,
                        key = f"model.layers.{idx}.self_attn.k_norm",
                        layernorm_eps = config.layernorm_eps,
                    ) if config.use_qk_norm else None,
                    out_dtype = torch.float,
                ),
                mlp = GatedMLP(
                    config = config,
                    key = f"model.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.parallel",
                    interm_dtype = torch.float,
                    out_dtype = torch.float,
                ),
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = "model.embed_tokens"

        self.modules += [
            LayerNorm(
                config = config,
                key = "model.norm",
                layernorm_eps = config.layernorm_eps,
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
                caps = {"logits_output": True},
                post_scale = config.logit_scale
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<BOS_TOKEN>"
        if system_prompt:
            p += f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|>"
        p += f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|>"
        p += f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        return p
