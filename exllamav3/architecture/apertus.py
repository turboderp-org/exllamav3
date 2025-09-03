from typing_extensions import override
from ..model.config import Config, no_default
from ..model.model import Model
import torch
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, MLP, Linear
from ..modules.attn import prepare_for_attn

class ApertusConfig(Config):
    arch_string = "ApertusForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": ApertusModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "hidden_act", "xielu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)



class ApertusModel(Model):
    config_class = ApertusConfig

    def __init__(
        self,
        config: ApertusConfig,
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
            TransformerBlock(
                config = config,
                key = f"model.layers.{idx}",
                attn_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.attention_layernorm",
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
                    rope_settings = config.rope_settings,
                    sm_scale = None,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    q_norm = RMSNorm(
                        config = config,
                        key = f"model.layers.{idx}.self_attn.q_norm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    k_norm = RMSNorm(
                        config = config,
                        key = f"model.layers.{idx}.self_attn.k_norm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                mlp = MLP(
                    config = config,
                    key = f"model.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_down = "down_proj",
                    key_alpha_p = f"model.layers.{idx}.mlp.act_fn.alpha_p",
                    key_alpha_n = f"model.layers.{idx}.mlp.act_fn.alpha_n",
                    qmap = "block.mlp",
                    out_dtype = torch.float,
                    interm_dtype = torch.float,
                    activation_fn = "xielu",
                ),
            )
            for idx in range(config.num_hidden_layers)
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
        p = "<s>"
        if system_prompt:
            p += f"<|system_start|>{system_prompt}<|system_end|>"
        p += f"<|user_start|>{prompt}<|user_end|>"
        p += f"<|assistant_start|>"
        return p