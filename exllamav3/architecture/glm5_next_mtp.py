from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..model.config import Config
from ..model.model import Model
from ..modules import RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear, BlockSparseMLP
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .glm5_next import Glm5NextConfig


class Glm5NextMTPModel(Model):
    """
    GLM-5.3-Flash MTP head: DeepSeek-V3 shape (enorm/hnorm/eh_proj concat into one trunk-style
    decoder layer, shared_head.norm before the shared lm_head), stored as last indexed layer
    Unlike the trunk layers it carries NO mHC tensors -- it is a plain residual block operating
    on the collapsed (post-mean, post-norm) trunk state.

    The config's index_share_for_mtp_iteration flag would let draft iterations reuse the
    trunk's top-k selection; as with GLM-5.2, this implementation lets the layer's own
    indexer score instead (self-consistent, and identical below index_topk context).
    """

    def __init__(
        self,
        config: Glm5NextConfig,
        key_prefix: str = "model.language_model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        first_mtp_layer = config.num_hidden_layers

        # Input layer: normed token embedding concatenated with normed target hidden state,
        # projected 2H -> H. Same mechanism as GLM-5.2/Qwen3.5/HyV3 MTP with DeepSeek-V3
        # tensor names and plain RMSNorms
        self.input_layer = Qwen3_5MTPInputLayer(
            config = config,
            key = f"{key_prefix}.layers.{first_mtp_layer}.input",
            key_pre_fc_norm_hidden = f"{key_prefix}.layers.{first_mtp_layer}.hnorm",
            key_pre_fc_norm_embedding = f"{key_prefix}.layers.{first_mtp_layer}.enorm",
            key_fc = f"{key_prefix}.layers.{first_mtp_layer}.eh_proj",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
            constant_bias = 0.0,
        )

        self.modules = [self.input_layer]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_mtp_layers):
            key = f"{key_prefix}.layers.{first_mtp_layer + idx}"
            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = MLAttention(
                        config = config,
                        key = f"{key}.self_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        num_q_heads = config.num_q_heads,
                        kv_lora_rank = config.kv_lora_rank,
                        qk_nope_head_dim = config.qk_nope_head_dim,
                        qk_rope_head_dim = config.qk_rope_head_dim,
                        v_head_dim = config.v_head_dim,
                        rope_settings = None,
                        q_lora_rank = config.q_lora_rank,
                        sm_scale = config.sm_scale,
                        rms_norm_eps = config.rms_norm_eps,
                        qmap = "block.attn",
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                        qbits_key = "mtp_bits",
                        indexer_mode = "full",
                        index_n_heads = config.index_n_heads,
                        index_head_dim = config.index_head_dim,
                        index_topk = config.index_topk,
                        index_kpool = config.index_kpool,
                        index_kpool_tail = config.index_kpool_tail,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = BlockSparseMLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size,
                        num_experts = config.num_experts,
                        num_experts_per_tok = config.num_experts_per_tok,
                        key_up = "experts.{expert_idx}.up_proj",
                        key_gate = "experts.{expert_idx}.gate_proj",
                        key_down = "experts.{expert_idx}.down_proj",
                        key_routing_gate = "gate",
                        key_e_score_bias = "gate.e_score_correction_bias",
                        activation_fn = "silu",
                        act_limit = config.swiglu_limit,
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        router_type = "dots",
                        routed_scaling_factor = config.routed_scaling_factor,
                        n_group = config.n_group,
                        topk_group = config.topk_group,
                        qbits_key = "mtp_bits",
                        shared_experts = GatedMLP(
                            config = config,
                            key = f"{key}.mlp.shared_experts",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            activation_fn = "silu",
                            act_limit = config.swiglu_limit,
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            qbits_key = "mtp_bits",
                            select_hq_bits = 2,
                        ) if config.num_shared_experts else None,
                    ),
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # Final norm before the (shared) lm_head
        self.final_norm = RMSNorm(
            config = config,
            key = f"{key_prefix}.layers.{first_mtp_layer + config.num_mtp_layers - 1}.shared_head.norm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 3,
            "autosplit_load_fwd": False,
        })

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # MTP doesn't take input_ids through Embedding here — embedding is handled by the
        # input layer. prepare_for_attn still wires up flash-attn params
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and tell the target to export its
        hidden state. hnorm consumes the trunk's POST-final-norm state (mean-collapsed hc
        streams through model.norm)
        """
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)

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
        assert isinstance(target_norm, RMSNorm), \
            "Expected target final RMSNorm immediately before lm_head"
        self.draft_verifier_params = {
            "export_state_norm_keys": {target_norm.key},
        }


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        ll = self.attached_model().logit_layer_idx
        lm = self.attached_model().modules[ll]
        logits = lm.prepare_for_device(state, params)
        logits = lm.forward(logits, params)
        if params.get("export_draft_conf"):
            # Per-position confidence for the generator's draft truncation: the argmax logit
            # value, over the unpadded vocabulary
            logits = logits[..., :self.attached_model().config.vocab_size]
            conf, ids = torch.max(logits, dim = -1)
            params["draft_conf"] = conf
            return ids
        return torch.argmax(logits, dim = -1)
