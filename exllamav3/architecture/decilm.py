from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn

class DeciLMConfig(Config):
    arch_string = "DeciLMForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": DeciLMModel},
            **kwargs
        )

        # Global attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)

        # Block configs
        self.block_configs = self.read_cfg(list, "block_configs", no_default)
        assert len(self.block_configs) == self.num_hidden_layers, \
            "Number of hidden layers does not match length of block_configs list"


class DeciLMModel(Model):
    config_class = DeciLMConfig

    def __init__(
        self,
        config: DeciLMConfig,
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
        self.last_kv_module_idx = 0
        cache_layer_idx = 0

        for idx, cfg in enumerate(config.block_configs):
            cfg_attn = cfg["attention"]
            cfg_ffn = cfg["ffn"]

            if cfg_attn.get("no_op"):
                attn_norm = None
                attn = None
            else:
                assert not cfg_attn.get("num_sink_tokens"), "DeciLM: num_sink_tokens not supported"
                assert not cfg_attn.get("replace_with_linear"), "DeciLM: replace_with_linear not supported"
                assert not cfg_attn.get("sparsify"), "DeciLM: sparsify not supported"
                assert not cfg_attn.get("unshifted_sink"), "DeciLM: unshifted_sink not supported"
                assert not cfg_attn.get("use_prefill_window_in_sink_attention"), \
                    "DeciLM: use_prefill_window_in_sink_attention not supported"
                attn_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                )
                attn = Attention(
                    config = config,
                    key = f"model.layers.{idx}.self_attn",
                    layer_idx = cache_layer_idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_q_heads // cfg_attn["n_heads_in_group"],
                    rope_settings = config.rope_settings,
                    sm_scale = None,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                )
                cache_layer_idx += 1
                self.last_kv_module_idx = len(self.modules)

            if cfg_ffn.get("no_op"):
                mlp_norm = None
                mlp = None
            else:
                assert not cfg_ffn.get("replace_with_linear"), "DeciLM: replace_with_linear not supported"
                assert not cfg_ffn.get("sparsify"), "DeciLM: sparsify not supported"
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                )
                interm_size = int(2 * cfg_ffn["ffn_mult"] * config.hidden_size / 3)
                interm_size = ((interm_size + 255) // 256) * 256
                mlp = GatedMLP(
                    config = config,
                    key = f"model.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = interm_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.mlp",
                )

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"model.layers.{idx}",
                    attn_norm = attn_norm,
                    attn = attn,
                    mlp_norm = mlp_norm,
                    mlp = mlp,
                )
            ]

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
        p = "<s>"
        if system_prompt:
            p += f"### System:\n"
            p += f"{system_prompt}\n"
        p += f"### User:\n"
        p += f"{prompt}\n"
        p += f"### Assistant:\n"
        return p