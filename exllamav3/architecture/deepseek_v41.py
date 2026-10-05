from __future__ import annotations
from typing_extensions import override
import numpy as np
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Embedding, RMSNorm, Linear, GatedMLP, BlockSparseMLP, TransformerBlock, \
    HyperConnection, ExpandStreams, HyperHead
from ..modules.dsv4 import DSV4Attention
from ..modules.ngram_embedding import _find_nth_prime_after
from ..modules.arch_specific.deepseek_v41 import EngramLayer
from ..modules.attn import prepare_for_attn

# DeepSeek-V4.1: DeepSeek-V4 blocks where runs of layers read one compressed KV pool and one
# index selection (kv_source_layer_ids / index_source_layer_ids), every mHC site collapses with
# the previous site's weights, and engram layers add hashed n-gram rows to the streams.


class DeepseekV41Config(Config):
    arch_string = "DeepseekV41ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": DeepseekV41Model},
            **kwargs
        )

        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", 1)
        assert self.num_kv_heads == 1, "DeepseekV41: expected shared-KV MQA (num_key_value_heads == 1)"
        self.head_dim = self.read_cfg(int, "text_config->head_dim", 512)
        self.qk_rope_head_dim = self.read_cfg(int, "text_config->qk_rope_head_dim", 64)
        self.q_lora_rank = self.read_cfg(int, "text_config->q_lora_rank", no_default)
        self.o_groups = self.read_cfg(int, "text_config->o_groups", 8)
        self.o_lora_rank = self.read_cfg(int, "text_config->o_lora_rank", 1024)
        self.sliding_window = self.read_cfg(int, "text_config->sliding_window", 128)
        self.index_n_heads = self.read_cfg(int, "text_config->index_n_heads", no_default)
        self.index_head_dim = self.read_cfg(int, "text_config->index_head_dim", 128)
        self.index_topk = self.read_cfg(int, "text_config->index_topk", 512)

        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.compress_ratios = self.read_cfg(list, "text_config->compress_ratios", no_default)[:self.num_hidden_layers]
        self.kv_source_layer_ids = self.read_cfg(list, "text_config->kv_source_layer_ids", no_default)
        self.index_source_layer_ids = self.read_cfg(list, "text_config->index_source_layer_ids", no_default)
        self.candidate_source_layer_id = self.read_cfg(int, "text_config->candidate_source_layer_id", -1)
        self.candidate_topk_blocks = self.read_cfg(int, "text_config->candidate_topk_blocks", 0)
        self.candidate_block_size = self.read_cfg(int, "text_config->candidate_block_size", 1)

        self.hc_mult = self.read_cfg(int, "text_config->hc_mult", 4)
        self.hc_sinkhorn_iters = self.read_cfg(int, "text_config->hc_sinkhorn_iters", 20)
        self.hc_eps = self.read_cfg(float, "text_config->hc_eps", 1e-6)

        self.assert_cfg(str, "text_config->scoring_func", "sqrtsoftplus", optional = True)
        self.assert_cfg(str, "text_config->topk_method", "noaux_tc", optional = True)
        self.moe_intermediate_size = self.read_cfg(int, "text_config->moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "text_config->n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "text_config->num_experts_per_tok", no_default)
        self.num_shared_experts = self.read_cfg(int, "text_config->n_shared_experts", 1)
        self.routed_scaling_factor = self.read_cfg(float, "text_config->routed_scaling_factor", 1.0)
        self.swiglu_limit = self.read_cfg(float, "text_config->swiglu_limit", 10.0)

        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", 1e-6)
        self.rope_theta = self.read_cfg(float, "text_config->rope_theta", 10000.0)
        self.compress_rope_theta = self.read_cfg(float, "text_config->compress_rope_theta", 160000.0)
        self.rope_scaling = self.read_cfg(dict, "text_config->rope_scaling", None)

        self.engram_layer_ids = self.read_cfg(list, "text_config->engram_layer_ids", [])
        self.engram_ngram_size = self.read_cfg(int, "text_config->engram_max_ngram_size", 4)
        self.engram_n_heads = self.read_cfg(int, "text_config->engram_n_heads", 8)
        self.engram_head_dim = self.read_cfg(int, "text_config->engram_head_dim", 256)
        self.engram_pad_token_id = self.read_cfg(int, "text_config->engram_pad_token_id", 2)
        self.engram_compressed_vocab_size = self.read_cfg(int, "text_config->engram_compressed_vocab_size", 0)
        p = self.read_cfg(int, "text_config->engram_vocab_size", 0) - 1
        rows = self.read_cfg(list, "text_config->engram_num_embeddings", [])
        self.engram_head_vocab_sizes, self.engram_multipliers = {}, {}
        for idx, n in zip(self.engram_layer_ids, rows):
            sizes = []
            for _ in range((self.engram_ngram_size - 1) * self.engram_n_heads):
                p = _find_nth_prime_after(p, 1)
                sizes.append(p)
            assert sum(sizes) == n, f"DeepseekV41: unexpected engram table size for layer {idx}"
            self.engram_head_vocab_sizes[idx] = sizes
            bound = (2 ** 63 - 1) // self.engram_compressed_vocab_size // 2
            rng = np.random.default_rng(10007 * idx)
            mult = rng.integers(0, bound, size = self.engram_ngram_size, dtype = np.int64) * 2 + 1
            self.engram_multipliers[idx] = mult.tolist()


class DeepseekV41Model(Model):
    config_class = DeepseekV41Config

    def __init__(
        self,
        config: DeepseekV41Config,
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
            key = f"layers.{idx}"
            ratio = config.compress_ratios[idx]
            kv_source = max((s for s in config.kv_source_layer_ids if s <= idx), default = idx) if ratio else None
            assert not ratio or config.compress_ratios[kv_source] == ratio, \
                f"DeepseekV41: layer {idx} cannot read the pool of layer {kv_source}"
            full = idx in config.index_source_layer_ids
            cand = config.candidate_source_layer_id
            if idx in config.engram_layer_ids:
                self.modules += [
                    EngramLayer(
                        config = config,
                        key = f"{key}.engram",
                        layer_idx = -(idx + 1),
                        hidden_size = config.hidden_size,
                        hc_mult = config.hc_mult,
                        ngram_size = config.engram_ngram_size,
                        heads_per_ngram = config.engram_n_heads,
                        head_dim = config.engram_head_dim,
                        head_vocab_sizes = config.engram_head_vocab_sizes[idx],
                        layer_multipliers = config.engram_multipliers[idx],
                        compressed_vocab_size = config.engram_compressed_vocab_size,
                        pad_token_id = config.engram_pad_token_id,
                        sliding_window = config.sliding_window,
                        rms_norm_eps = config.rms_norm_eps,
                    )
                ]
            attn = DSV4Attention(
                config = config,
                key = f"{key}.attn",
                layer_idx = idx,
                layer_type = "hca" if ratio else "sliding",
                hidden_size = config.hidden_size,
                num_q_heads = config.num_q_heads,
                head_dim = config.head_dim,
                rope_head_dim = config.qk_rope_head_dim,
                q_lora_rank = config.q_lora_rank,
                o_groups = config.o_groups,
                o_lora_rank = config.o_lora_rank,
                sliding_window = config.sliding_window,
                compress_rate = ratio or None,
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
                kv_source = kv_source,
                indexer_mode = ("full" if full else "shared") if ratio else None,
                q_head_norm = False,
                candidate_mode = "source" if idx == cand else "use" if full and 0 <= cand < idx else None,
                candidate_topk_blocks = config.candidate_topk_blocks,
                candidate_block_size = config.candidate_block_size,
            )
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
                qmap = "block.mlp",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                activation_fn = "silu",
                act_limit = config.swiglu_limit,
                router_type = "sqrtsp",
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
                    carry_pre = True,
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

        self.modules += [
            HyperHead(
                config = config,
                key = "hc_head",
                hc_mult = config.hc_mult,
                rms_norm_eps = config.rms_norm_eps,
                hc_eps = config.hc_eps,
                carry_pre = True,
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
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True},
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "supports_tp": False,
        })
        from ..cache.dsa import DSV4State
        self.recurrent_state_cls = DSV4State

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        params["hc_pre"] = None
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"{system_prompt}\n\n"
        p += f"<｜User｜>{prompt}<｜Assistant｜>"
        return p
