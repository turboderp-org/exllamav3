from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, RopeSettings
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn

class Spark2_5Config(Config):
    arch_string = "Spark2_5ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Spark2_5Model},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Per-layer attention pattern (sliding vs full)
        layer_types = self.read_cfg(list, "layer_types", None)
        if layer_types is not None:
            assert len(layer_types) == self.read_cfg(int, "num_hidden_layers", no_default), \
                "Length of layer_types must match num_hidden_layers"
            for t in layer_types:
                if t not in ("sliding_attention", "full_attention"):
                    raise ValueError(f"Unknown layer type in layer_types: {t}")
        self.layer_types = layer_types

        # Sliding window size
        self.sliding_window = self.read_cfg(int, "sliding_window", -1)

        # MLP params
        self.assert_cfg(str, "hidden_act", "gelu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # Headwise output gate (one scalar per query head, applied as a gate on the attention output)
        self.headwise_attn_output_gate = self.read_cfg(bool, "headwise_attn_output_gate", False)
        self.gate_attn_act_mode = self.read_cfg(str, "gate_attn_act_mode", "sigmoid")

        # Per-layer RoPE: full_attention and sliding_attention each carry their own theta and
        # partial_rotary_factor under rope_parameters. The Spark2.5 stack keeps both families
        # on the NEOX variant; the difference is the rotary band width and the base.
        rope_params = self.read_cfg(dict, "rope_parameters", {})
        full_params = rope_params.get("full_attention", {}) if isinstance(rope_params, dict) else {}
        sliding_params = rope_params.get("sliding_attention", {}) if isinstance(rope_params, dict) else {}

        self.rope_settings_full = RopeSettings(
            head_dim = self.head_dim,
            rope_theta = float(full_params.get("rope_theta", 10000.0)),
            partial_rotary_factor = float(full_params.get("partial_rotary_factor", 1.0)),
            max_position_embeddings = self.max_position_embeddings,
            rope_style = RopeStyle.NEOX,
        )
        self.rope_settings_sliding = RopeSettings(
            head_dim = self.head_dim,
            rope_theta = float(sliding_params.get("rope_theta", 10000.0)),
            partial_rotary_factor = float(sliding_params.get("partial_rotary_factor", 1.0)),
            max_position_embeddings = self.max_position_embeddings,
            rope_style = RopeStyle.NEOX,
        )

        # Only sigmoid is wired through the existing headwise-gate kernel path (mul_sigmoid_broadcast_).
        # The reference modeling code also accepts "silu", but that requires x *= silu(y), which is
        # not exposed by Attention. Refuse loudly if a checkpoint asks for it so we don't silently
        # produce wrong outputs.
        if self.headwise_attn_output_gate and self.gate_attn_act_mode not in ("sigmoid",):
            raise ValueError(
                f"Spark2.5: only gate_attn_act_mode='sigmoid' is currently supported, got "
                f"'{self.gate_attn_act_mode}'."
            )

    def get_tensor_name_fixes(self):
        # HF Spark stores the embedding under model.embedding.weight, while the rest of the
        # architecture uses model.embed_tokens. Remap once at load time so the standard
        # Embedding key works.
        return {
            ".embedding.weight": ".embed_tokens.weight"
        }

    def layer_rope_settings(self, layer_idx: int) -> RopeSettings:
        if self.layer_types is None or self.layer_types[layer_idx] == "full_attention":
            return self.rope_settings_full
        return self.rope_settings_sliding

    def layer_sliding_window(self, layer_idx: int) -> int:
        if self.layer_types is None:
            return -1
        if self.layer_types[layer_idx] == "sliding_attention":
            return self.sliding_window
        return -1


class Spark2_5Model(Model):
    config_class = Spark2_5Config

    def __init__(
        self,
        config: Spark2_5Config,
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

        for idx in range(config.num_hidden_layers):
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"model.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"model.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = Attention(
                        config = config,
                        key = f"model.layers.{idx}.self_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        head_dim = config.head_dim,
                        num_q_heads = config.num_q_heads,
                        num_kv_heads = config.num_kv_heads,
                        rope_settings = config.layer_rope_settings(idx),
                        sm_scale = None,
                        sliding_window = config.layer_sliding_window(idx),
                        key_fused_qkv = "q_k_v_proj",
                        key_g = "g_proj" if config.headwise_attn_output_gate else None,
                        key_o = "out_proj",
                        qmap = "block.attn",
                        out_dtype = torch.float,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"model.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"model.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        activation_fn = "gelu",
                        out_dtype = torch.float,
                    ),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = "model.embed_tokens"

        self.modules += [
            RMSNorm(
                config = config,
                key = "model.norm",
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


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # HF chat_template.jinja wraps turns in <|System|>/<|User|>/<|Assistant|> blocks separated
        # by <|end▁of▁sentence|>. The default raw-text path uses a single round, no tools.
        p = "<｜start▁of▁sentence｜>"
        if system_prompt:
            p += f"<|System|>\n{system_prompt}<｜end▁of▁sentence｜>"
        p += f"<|User|>\n{prompt}<｜end▁of▁sentence｜><|Assistant|>\n"
        return p
