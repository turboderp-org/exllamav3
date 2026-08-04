from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Embedding, RMSNorm, Linear, GatedMLP, BlockSparseMLP, TransformerBlock, \
    HyperConnection, ExpandStreams, HyperHead
from ..modules.dsv4 import DSV4Attention
from ..modules.attn import prepare_for_attn
from .deepseek_v4_mtp import DeepseekV4MTPModel

# DeepSeek-V4: hybrid sparse attention (sliding / CSA / HCA per compress_ratios), mHC
# hyper-connection residual streams, hash-MoE bootstrap layers, sqrt-softplus routing.
# Checkpoint uses DeepSeek's native tensor namespace (layers.N.attn.wq_a, hc_attn_fn,
# embed / head / norm, ...). Reference implementation: transformers models/deepseek_v4.

_RATIO_TO_TYPE = {0: "sliding", 4: "csa", 128: "hca"}


class DeepseekV4Config(Config):
    arch_string = "DeepseekV4ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": DeepseekV4Model, "mtp": DeepseekV4MTPModel},
            **kwargs
        )

        # Attention
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", 1)
        assert self.num_kv_heads == 1, "DeepseekV4: expected shared-KV MQA (num_key_value_heads == 1)"
        self.head_dim = self.read_cfg(int, "head_dim", 512)
        self.qk_rope_head_dim = self.read_cfg(int, "qk_rope_head_dim", 64)
        self.q_lora_rank = self.read_cfg(int, "q_lora_rank", no_default)
        self.o_groups = self.read_cfg(int, "o_groups", 8)
        self.o_lora_rank = self.read_cfg(int, "o_lora_rank", 1024)
        self.sliding_window = self.read_cfg(int, "sliding_window", 128)
        self.index_n_heads = self.read_cfg(int, "index_n_heads", 64)
        self.index_head_dim = self.read_cfg(int, "index_head_dim", 128)
        self.index_topk = self.read_cfg(int, "index_topk", 512)

        # Layer schedule: compress_ratios per layer (0 = sliding, 4 = CSA, 128 = HCA); the
        # list may run past num_hidden_layers to describe MTP blocks (ignored here)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        ratios = self.read_cfg(list, "compress_ratios", None)
        if ratios is not None:
            self.layer_types = [_RATIO_TO_TYPE[r] for r in ratios[:self.num_hidden_layers]]
        else:
            inter = ["csa" if i % 2 else "hca" for i in range(max(self.num_hidden_layers - 2, 0))]
            self.layer_types = ["hca"] * min(self.num_hidden_layers, 2) + inter
        self.compress_rate_csa = self.read_cfg(int, "compress_rate_csa", 4)
        self.compress_rate_hca = self.read_cfg(int, "compress_rate_hca", 128)

        # mHC
        self.hc_mult = self.read_cfg(int, "hc_mult", 4)
        self.hc_sinkhorn_iters = self.read_cfg(int, "hc_sinkhorn_iters", 20)
        self.hc_eps = self.read_cfg(float, "hc_eps", 1e-6)

        # MoE
        self.assert_cfg(str, "scoring_func", "sqrtsoftplus", optional = True)
        self.assert_cfg(str, "topk_method", "noaux_tc", optional = True)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.num_shared_experts = self.read_cfg(int, "n_shared_experts", 1)
        self.num_hash_layers = self.read_cfg(int, "num_hash_layers", 3)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)
        self.swiglu_limit = self.read_cfg(float, "swiglu_limit", 10.0)

        # Norms / rope
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", 1e-6)
        self.rope_theta = self.read_cfg(float, "rope_theta", 10000.0)
        self.compress_rope_theta = self.read_cfg(float, "compress_rope_theta", 160000.0)
        self.rope_scaling = self.read_cfg(dict, "rope_scaling", None)

        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # DSpark drafter (mtp.* namespace). num_nextn_predict_layers is unreliable
        # (V4-Flash ships 1 alongside three mtp blocks); the compress_ratios tail past the
        # trunk layers describes the MTP blocks. The component only exists when the
        # checkpoint actually carries the tensors
        self.dspark_block_size = self.read_cfg(int, "dspark_block_size", 0)
        self.dspark_noise_token_id = self.read_cfg(int, "dspark_noise_token_id", 0)
        self.dspark_markov_rank = self.read_cfg(int, "dspark_markov_rank", 256)
        self.dspark_target_layer_ids = self.read_cfg(list, "dspark_target_layer_ids", [])
        # Generator drafter convention (dflash flow): block_size counts the seed position
        self.block_size = self.dspark_block_size + 1
        if ratios is not None and len(ratios) > self.num_hidden_layers:
            self.num_mtp_layers = len(ratios) - self.num_hidden_layers
            self.mtp_layer_types = [_RATIO_TO_TYPE[r] for r in ratios[self.num_hidden_layers:]]
        else:
            self.num_mtp_layers = 0
            self.mtp_layer_types = []
        if self.num_mtp_layers == 0 or not any(
            self.stc.has_tensor(f"mtp.0.attn.wkv.{t}") for t in ("weight", "trellis")):
            del self.model_classes["mtp"]


class DeepseekV4Model(Model):
    config_class = DeepseekV4Config

    def __init__(
        self,
        config: DeepseekV4Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = "embed",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            ),
            ExpandStreams(
                config = config,
                key = "hc_expand",
                hc_mult = config.hc_mult,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            layer_type = config.layer_types[idx]
            key = f"layers.{idx}"
            attn = DSV4Attention(
                config = config,
                key = f"{key}.attn",
                layer_idx = idx,
                layer_type = layer_type,
                hidden_size = config.hidden_size,
                num_q_heads = config.num_q_heads,
                head_dim = config.head_dim,
                rope_head_dim = config.qk_rope_head_dim,
                q_lora_rank = config.q_lora_rank,
                o_groups = config.o_groups,
                o_lora_rank = config.o_lora_rank,
                sliding_window = config.sliding_window,
                compress_rate = {
                    "sliding": None,
                    "csa": config.compress_rate_csa,
                    "hca": config.compress_rate_hca,
                }[layer_type],
                index_n_heads = config.index_n_heads,
                index_head_dim = config.index_head_dim,
                index_topk = config.index_topk,
                rope_theta = config.rope_theta,
                compress_rope_theta = config.compress_rope_theta,
                rope_scaling = config.rope_scaling,
                rms_norm_eps = config.rms_norm_eps,
                qmap = "block.attn",
                out_dtype = torch.float,
                select_hq_bits = 2,
            )
            is_hash = idx < config.num_hash_layers
            mlp = BlockSparseMLP(
                config = config,
                key = f"{key}.ffn",
                hidden_size = config.hidden_size,
                intermediate_size = config.moe_intermediate_size,
                num_experts = config.num_experts,
                num_experts_per_tok = config.num_experts_per_tok,
                key_up = "experts.{expert_idx}.w3",
                key_gate = "experts.{expert_idx}.w1",
                key_down = "experts.{expert_idx}.w2",
                key_routing_gate = "gate",
                key_e_score_bias = "gate.bias",
                key_tid2eid = "gate.tid2eid" if is_hash else None,
                qmap = "block.mlp",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                activation_fn = "silu",
                act_limit = config.swiglu_limit,
                router_type = "sqrtsp_hash" if is_hash else "sqrtsp",
                routed_scaling_factor = config.routed_scaling_factor,
                shared_experts = GatedMLP(
                    config = config,
                    key = f"{key}.ffn.shared_experts",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                    key_up = "w3",
                    key_gate = "w1",
                    key_down = "w2",
                    qmap = "block.mlp",
                    out_dtype = torch.float,
                    activation_fn = "silu",
                    act_limit = config.swiglu_limit,
                    select_hq_bits = 2,
                ),
            )
            def _hc(tag: str):
                return HyperConnection(
                    config = config,
                    key = f"{key}.hc_{tag}",
                    hc_mult = config.hc_mult,
                    hidden_size = config.hidden_size,
                    sinkhorn_iters = config.hc_sinkhorn_iters,
                    hc_eps = config.hc_eps,
                    rms_norm_eps = config.rms_norm_eps,
                )
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(config, f"{key}.attn_norm", config.rms_norm_eps),
                    attn = attn,
                    attn_hc = _hc("attn"),
                    mlp_norm = RMSNorm(config, f"{key}.ffn_norm", config.rms_norm_eps),
                    mlp = mlp,
                    mlp_hc = _hc("ffn"),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("head"):
            head_alt_key = "embed"

        self.modules += [
            HyperHead(
                config = config,
                key = "hc_head",
                hc_mult = config.hc_mult,
                rms_norm_eps = config.rms_norm_eps,
                hc_eps = config.hc_eps,
            ),
            RMSNorm(
                config = config,
                key = "norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True},
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True

        # All attention state is recurrent-style (rings + pools), no paged KV modules
        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
        })
        from ..cache.dsa import DSV4State
        self.recurrent_state_cls = DSV4State

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Hash-MoE layers route by token id (get_for_device copies once per device)
        params["input_ids"] = input_ids
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"{system_prompt}\n\n"
        p += f"<|User|>{prompt}<|Assistant|>"
        return p
