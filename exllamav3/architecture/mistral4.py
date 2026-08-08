from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear, BlockSparseMLP
from ..modules.attn import prepare_for_attn

"""
Text stack of Mistral-Small-4 (Mistral3ForConditionalGeneration with text_config.model_type
"mistral4"): DeepSeek-style MLA attention (q_lora 1024, kv_lora 256, nope 64 + rope 64,
v_head 128) combined with the Ministral-3 Llama-4 position scale on the queries (applied to
the full query by MLAttention, see mla_attn.py), and a 128-expert top-4 MoE with softmax
routing (norm_topk_prob) plus one shared expert. Routed experts are stored as stacked 3D fp8
tensors in nn.Linear (out, in) orientation, gate rows first (HF DeepseekV3Experts layout).
The vision tower and projector are unchanged Mistral3/Pixtral; dispatch happens in
Mistral3Config, which calls mistral4_init_config in place of its own text-field reads.
"""


def mistral4_init_config(config, directory: str, **kwargs):
    from .mistral3 import Mistral3VisionModel
    Config.__init__(
        config,
        directory,
        {"text": Mistral4TextModel, "vision": Mistral3VisionModel},
        **kwargs
    )

    # Latent attention params
    config.hidden_size = config.read_cfg(int, "text_config->hidden_size", no_default)
    config.num_q_heads = config.read_cfg(int, "text_config->num_attention_heads", no_default)
    config.q_lora_rank = config.read_cfg(int, "text_config->q_lora_rank", None)
    config.kv_lora_rank = config.read_cfg(int, "text_config->kv_lora_rank", no_default)
    config.qk_nope_head_dim = config.read_cfg(int, "text_config->qk_nope_head_dim", no_default)
    config.qk_rope_head_dim = config.read_cfg(int, "text_config->qk_rope_head_dim", no_default)
    config.v_head_dim = config.read_cfg(int, "text_config->v_head_dim", no_default)
    config.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    # Reported head dim, for allocation and logging; the cache stores the latent instead
    config.head_dim = config.qk_head_dim

    # MLP params
    config.assert_cfg(str, "text_config->hidden_act", "silu", True)
    config.intermediate_size = config.read_cfg(int, "text_config->intermediate_size", no_default)
    config.moe_intermediate_size = config.read_cfg(int, "text_config->moe_intermediate_size", no_default)
    config.num_shared_experts = config.read_cfg(int, "text_config->n_shared_experts", 1)
    config.num_experts = config.read_cfg(int, "text_config->n_routed_experts", no_default)
    config.num_experts_per_tok = config.read_cfg(int, "text_config->num_experts_per_tok", no_default)
    config.first_k_dense_replace = config.read_cfg(int, "text_config->first_k_dense_replace", 0)
    # The std routing path implements softmax over the selected logits, which matches the
    # reference (full softmax -> top-k -> renormalize) only with norm_topk_prob and no group
    # masking or scaling
    config.assert_cfg(bool, "text_config->norm_topk_prob", True, True)
    n_group = config.read_cfg(int, "text_config->n_group", 1)
    topk_group = config.read_cfg(int, "text_config->topk_group", 1)
    assert n_group in (None, 1) and topk_group in (None, 1), \
        f"Group-limited expert routing (n_group = {n_group}, topk_group = {topk_group}) is not supported"
    routed_scaling_factor = config.read_cfg(float, "text_config->routed_scaling_factor", 1.0)
    assert routed_scaling_factor == 1.0, \
        f"routed_scaling_factor {routed_scaling_factor} != 1.0 is not supported with softmax routing"

    # Norms
    config.rms_norm_eps = config.read_cfg(float, "text_config->rms_norm_eps", no_default)

    # Layers
    config.num_hidden_layers = config.read_cfg(int, "text_config->num_hidden_layers", no_default)
    config.tie_word_embeddings = config.read_cfg(bool, "text_config->tie_word_embeddings", False)

    # RoPE applies to the rope half of the query and the single shared rope key only. The yarn
    # config carries inert mscale/mscale_all_dim 1.0 and llama_4_scaling_beta supersedes the
    # static attention factor (rope.py rule 3), so sm_scale stays plain
    config.assert_cfg(bool, "text_config->rope_interleave", True, True)
    text_cfg = config.read_cfg(dict, "text_config", no_default)
    config.rope_settings = config.read_rope_settings_default(
        RopeStyle.GPTJ,
        override_head_dim = config.qk_rope_head_dim,
        config_dict = text_cfg,
    )
    config.sm_scale = config.qk_head_dim ** -0.5


class Mistral4TextModel(Model):

    def __init__(
        self,
        config,
        key_prefix = "language_model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        # Auto-detect key naming convention (shared with Mistral3Model)
        if config.new_key_style:
            lm = f"model.{key_prefix}"
            head = "lm_head"
        else:
            lm = f"{key_prefix}.model" if key_prefix else "model"
            head = f"{key_prefix}.lm_head" if key_prefix else "lm_head"

        self.modules += [
            Embedding(
                config = config,
                key = f"{lm}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)
        self.modules += [
            TransformerBlock(
                config = config,
                key = f"{lm}.layers.{idx}",
                layer_idx = idx,
                attn_norm = RMSNorm(
                    config = config,
                    key = f"{lm}.layers.{idx}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                attn = MLAttention(
                    config = config,
                    key = f"{lm}.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    num_q_heads = config.num_q_heads,
                    kv_lora_rank = config.kv_lora_rank,
                    qk_nope_head_dim = config.qk_nope_head_dim,
                    qk_rope_head_dim = config.qk_rope_head_dim,
                    v_head_dim = config.v_head_dim,
                    rope_settings = config.rope_settings,
                    q_lora_rank = config.q_lora_rank,
                    sm_scale = config.sm_scale,
                    rms_norm_eps = config.rms_norm_eps,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{lm}.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                mlp = (
                    GatedMLP(
                        config = config,
                        key = f"{lm}.layers.{idx}.mlp",
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
                    if idx < config.first_k_dense_replace else
                    BlockSparseMLP(
                        config = config,
                        key = f"{lm}.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size,
                        num_experts = config.num_experts,
                        num_experts_per_tok = config.num_experts_per_tok,
                        key_up = "experts.{expert_idx}.up_proj",
                        key_gate = "experts.{expert_idx}.gate_proj",
                        key_down = "experts.{expert_idx}.down_proj",
                        # Stacked 3D expert tensors, nn.Linear (out, in) slices, gate rows first
                        key_gate_up_split = "experts.gate_up_proj",
                        key_down_split = "experts.down_proj",
                        transpose_fused_weights = False,
                        ftranspose_after_load = True,
                        frange_dim = 0,
                        gate_up_interleaved = False,
                        key_routing_gate = "gate",
                        key_e_score_bias = None,
                        router_type = "std",
                        interm_div = 128.0,  # Outlier routed experts overflow fp16 (layer 32 expert 67: 5x over)
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        shared_experts = GatedMLP(
                            config = config,
                            key = f"{lm}.layers.{idx}.mlp.shared_experts",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            select_hq_bits = 2,
                        ) if config.num_shared_experts else None,
                    )
                )
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor(head):
            head_alt_key = f"{lm}.embed_tokens"

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{lm}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = head,
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

        # MLA layers currently do not support TP because the latent cache cannot be split by head
        self.caps.update({"supports_tp": False})


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<s>"
        if system_prompt:
            p += f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT]"
        p += f"[INST]{prompt}[/INST]"
        return p
