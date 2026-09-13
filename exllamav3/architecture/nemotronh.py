from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    MLP,
    BlockSparseMLP,
    Linear,
    Mamba2,
    GDNState,
)
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from .nemotronh_mtp import NemotronHMTPModel

class NemotronHConfig(Config):
    arch_string = "NemotronHForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": NemotronHModel, "mtp": NemotronHMTPModel},
            **kwargs
        )

        # Layer pattern: M = Mamba2, * = attention, - = MLP, E = MoE
        self.hybrid_override_pattern = self.read_cfg(str, "hybrid_override_pattern", no_default)

        # Attention params (NoPE: the reference model applies no positional embeddings)
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Mamba2 params
        self.mamba_num_heads = self.read_cfg(int, "mamba_num_heads", no_default)
        self.mamba_head_dim = self.read_cfg(int, "mamba_head_dim", no_default)
        self.ssm_state_size = self.read_cfg(int, "ssm_state_size", no_default)
        self.n_groups = self.read_cfg(int, "n_groups", no_default)
        self.conv_kernel = self.read_cfg(int, "conv_kernel", 4)
        self.time_step_limit = self.read_cfg(list, "time_step_limit", [0.0, float("inf")])
        self.assert_cfg(str, "mamba_hidden_act", "silu", True)

        # MLP params
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.assert_cfg(str, "mlp_hidden_act", "relu2", True)

        # MoE params (30B-A3B, 120B-A12B): DeepSeek-style sigmoid router with correction bias,
        # non-gated relu2 experts plus one always-on shared expert
        self.num_experts = self.read_cfg(int, "n_routed_experts", 0)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", 0)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", 0)
        self.shared_expert_intermediate_size = self.read_cfg(int, "moe_shared_expert_intermediate_size", 0)
        # Latent MoE (Nemotron-3 Super): routed experts run at moe_latent_size behind a pair of
        # projections (fc1_latent_proj / fc2_latent_proj); the shared expert stays full width
        self.moe_latent_size = self.read_cfg(int, "moe_latent_size", None)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 2.5)
        if self.num_experts:
            self.assert_cfg(int, "n_group", 1, True)
            self.assert_cfg(int, "topk_group", 1, True)
            self.assert_cfg(bool, "norm_topk_prob", True, True)

        # Norms
        self.layer_norm_epsilon = self.read_cfg(float, "layer_norm_epsilon", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        assert len(self.hybrid_override_pattern) == self.num_hidden_layers, \
            "hybrid_override_pattern length does not match num_hidden_layers"
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # MTP head (Nemotron-3 Super): one DeepSeek-style step (enorm/hnorm/eh_proj) followed by the
        # blocks of mtp_hybrid_override_pattern ("*E": attention + latent MoE) and a final norm, under
        # the mtp.* namespace. The component only exists when the tensors are present (a quant made
        # before MTP support gets them from util/convert_mtp.py)
        self.num_mtp_layers = self.read_cfg(int, "num_nextn_predict_layers", 0)
        self.mtp_hybrid_override_pattern = self.read_cfg(str, "mtp_hybrid_override_pattern", "")
        if self.num_mtp_layers == 0 or not any(
            self.stc.has_tensor(f"mtp.layers.0.eh_proj.{t}") for t in ("weight", "trellis")
        ):
            del self.model_classes["mtp"]



def nemotronh_block(
    config: NemotronHConfig,
    key: str,
    layer_idx: int,
    block_type: str,
    qbits_key: str = "bits",
) -> TransformerBlock:
    """One block of the hybrid pattern: M = Mamba2, * = attention, - = MLP, E = MoE. Shared by the
    trunk (backbone.layers.N) and the MTP head (mtp.layers.N, mtp_bits)"""
    norm = RMSNorm(
        config = config,
        key = f"{key}.norm",
        rms_norm_eps = config.layer_norm_epsilon,
    )
    match block_type:
        case "M":
            return TransformerBlock(
                config = config,
                key = key,
                layer_idx = layer_idx,
                attn_norm = norm,
                attn = Mamba2(
                    config = config,
                    key = f"{key}.mixer",
                    layer_idx = layer_idx,
                    hidden_size = config.hidden_size,
                    num_heads = config.mamba_num_heads,
                    head_dim = config.mamba_head_dim,
                    num_groups = config.n_groups,
                    state_size = config.ssm_state_size,
                    rms_norm_eps = config.layer_norm_epsilon,
                    conv_kernel_size = config.conv_kernel,
                    dt_limit = tuple(config.time_step_limit),
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                ),
            )
        case "*":
            return TransformerBlock(
                config = config,
                key = key,
                layer_idx = layer_idx,
                attn_norm = norm,
                attn = Attention(
                    config = config,
                    key = f"{key}.mixer",
                    layer_idx = layer_idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_kv_heads,
                    rope_settings = None,
                    sm_scale = None,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                    qbits_key = qbits_key,
                ),
            )
        case "-":
            return TransformerBlock(
                config = config,
                key = key,
                layer_idx = layer_idx,
                mlp_norm = norm,
                mlp = MLP(
                    config = config,
                    key = f"{key}.mixer",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_down = "down_proj",
                    activation_fn = "relu2",
                    qmap = "block.mlp",
                    out_dtype = torch.float,
                    select_hq_bits = 1,
                    qbits_key = qbits_key,
                ),
            )
        case "E":
            return TransformerBlock(
                config = config,
                key = key,
                layer_idx = layer_idx,
                mlp_norm = norm,
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key}.mixer",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    latent_size = config.moe_latent_size,
                    key_latent_in = "fc1_latent_proj" if config.moe_latent_size else None,
                    key_latent_out = "fc2_latent_proj" if config.moe_latent_size else None,
                    latent_hq_bits = 2,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_routing_gate = "gate",
                    activation_fn = "relu2",
                    router_type = "dots",
                    routed_scaling_factor = config.routed_scaling_factor,
                    n_group = 1,
                    topk_group = 1,
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    qbits_key = qbits_key,
                    shared_experts = MLP(
                        config = config,
                        key = f"{key}.mixer.shared_experts",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.shared_expert_intermediate_size,
                        key_up = "up_proj",
                        key_down = "down_proj",
                        activation_fn = "relu2",
                        qmap = "block.mlp",
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                        qbits_key = qbits_key,
                    ),
                ),
            )
        case _:
            raise ValueError(f"Unknown layer type {block_type!r} in hybrid_override_pattern")


class NemotronHModel(Model):
    config_class = NemotronHConfig

    def __init__(
        self,
        config: NemotronHConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = "backbone.embeddings",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            self.modules.append(nemotronh_block(
                config, f"backbone.layers.{idx}", idx, config.hybrid_override_pattern[idx]
            ))

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = "backbone.embeddings"

        self.modules += [
            RMSNorm(
                config = config,
                key = "backbone.norm_f",
                rms_norm_eps = config.layer_norm_epsilon,
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

        # Mark that we need recurrent cache for generation
        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True,
        })
        self.recurrent_state_cls = GDNState

        self.caps.update({"supports_tp": True})


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n"
            p += f"{system_prompt}<|im_end|>\n"
        p += f"<|im_start|>user\n"
        p += f"{prompt}<|im_end|>\n"
        p += f"<|im_start|>assistant\n"
        return p
