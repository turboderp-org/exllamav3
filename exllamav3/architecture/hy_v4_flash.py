from __future__ import annotations
from typing_extensions import override
import numpy as np
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeSettings, RopeStyle, RoPE
from ..modules import (
    RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear, BlockSparseMLP, PLELayer,
)
from ..modules.hyperconnections import ExpandStreams, HyperConnection, HyperHead
from ..modules.attn import prepare_for_attn
from .hy_v4_flash_dfly import HyV4FlashDFlyModel
from .hy_v4_flash_vision import HyV4FlashVisionModel, read_hy_v4_vision_config

# Hy4 flash (HYV4FlashForConditionalGeneration, text component): gated MLA with a learned sink on
# every layer, read through a 512-token window on three layers out of four and through a compressed
# indexer's selection on the rest (pools of four tokens, the best 511 pools plus the query's own);
# identity hyper-connections over four residual streams with a norm after every sublayer; one dense
# MLP followed by latent MoE layers; hashed n-gram memories (Engram) ahead of four layers.
# Reference: transformers models/hy_v4_flash.

# The Engram norms are torch RMSNorms built without an eps, which resolves to the fp32 machine epsilon
ENGRAM_NORM_EPS = 1.1920929e-07


class HyV4FlashConfig(Config):
    arch_string = "HYV4FlashForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": HyV4FlashModel, "mtp": HyV4FlashDFlyModel, "vision": HyV4FlashVisionModel},
            **kwargs
        )
        p = "text_config->"

        # Attention (gated MLA); sliding layers carry their own head count
        self.hidden_size = self.read_cfg(int, p + "hidden_size", no_default)
        self.q_lora_rank = self.read_cfg(int, p + "q_lora_rank", no_default)
        self.kv_lora_rank = self.read_cfg(int, p + "kv_lora_rank", no_default)
        self.num_q_heads = self.read_cfg(int, p + "num_attention_heads", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, p + "qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, p + "qk_rope_head_dim", no_default)
        self.v_head_dim = self.read_cfg(int, p + "v_head_dim", no_default)
        self.swa_num_q_heads = self.read_cfg(int, p + "swa_num_attention_heads", self.num_q_heads)
        self.swa_qk_nope_head_dim = self.read_cfg(int, p + "swa_qk_nope_head_dim", self.qk_nope_head_dim)
        self.swa_qk_rope_head_dim = self.read_cfg(int, p + "swa_qk_rope_head_dim", self.qk_rope_head_dim)
        self.swa_v_head_dim = self.read_cfg(int, p + "swa_v_head_dim", self.v_head_dim)
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.assert_cfg(bool, p + "attention_bias", False, True)

        # Layer schedule. The window counts the query's own position
        self.num_hidden_layers = self.read_cfg(int, p + "num_hidden_layers", no_default)
        self.layer_types = self.read_cfg(list, p + "layer_types", no_default)
        self.layer_types = ["indexed" if t != "sliding_attention" else "sliding" for t in self.layer_types]
        assert len(self.layer_types) >= self.num_hidden_layers
        self.sliding_window = self.read_cfg(int, p + "sliding_window", 512)

        # Compressed indexer
        self.index_n_heads = self.read_cfg(int, p + "index_n_heads", no_default)
        self.index_head_dim = self.read_cfg(int, p + "index_head_dim", no_default)
        self.index_topk = self.read_cfg(int, p + "index_topk", no_default)
        self.index_kpool = self.read_cfg(int, [p + "index_kpool", p + "compressed_indexer_compress_ratio"], 4)

        # RoPE: interleaved pairs over the rope slice, one base per layer type. Text positions are the same
        # on all three M-RoPE planes, which leaves the plain one-dimensional rotation; the rows of an image
        # take a position per plane (frame, row, column), the planes interleaved over the frequencies
        rp = self.read_cfg(dict, p + "rope_parameters", {})
        def rope(names: tuple, default_theta: float, dim: int):
            params = next((rp[n] for n in names if n in rp), {})
            return RopeSettings(
                head_dim = dim,
                rope_theta = float(params.get("rope_theta", default_theta)),
                rope_style = RopeStyle.GPTJ,
                rope_scaling = {
                    "mrope_section": params.get("mrope_section", [12, 10, 10]),
                    "mrope_interleaved": params.get("mrope_interleaved", True),
                },
            )
        self.rope_settings_indexed = rope(("indexed_attention", "deepseek_sparse_attention"), 1e7, self.qk_rope_head_dim)
        self.rope_settings_sliding = rope(("sliding_attention",), 1e4, self.swa_qk_rope_head_dim)

        # Identity hyper-connections
        self.hc_mult = self.read_cfg(int, p + "hc_mult", 4)
        self.hc_eps = self.read_cfg(float, p + "hc_eps", 1e-6)
        self.hc_magnitude = self.read_cfg(float, p + "hc_magnitude", 2.0)

        # MLP params
        self.assert_cfg(str, p + "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, p + "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, p + "moe_intermediate_size", no_default)
        self.moe_latent_size = self.read_cfg(int, p + "routed_expert_hidden_size", no_default)
        self.num_shared_experts = self.read_cfg(int, p + "n_shared_experts", 1)
        self.num_experts = self.read_cfg(int, p + "n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, p + "num_experts_per_tok", no_default)
        self.routed_scaling_factor = self.read_cfg(float, p + "routed_scaling_factor", 1.0)
        self.swiglu_limit = self.read_cfg(float, p + "swiglu_limit", 0.0)
        first_k_dense = self.read_cfg(int, p + "first_k_dense_replace", 1)
        self.mlp_layer_types = self.read_cfg(
            list, p + "mlp_layer_types",
            ["dense" if idx < first_k_dense else "sparse" for idx in range(self.num_hidden_layers)]
        )
        self.assert_cfg(int, p + "n_group", 1, True)
        self.assert_cfg(int, p + "topk_group", 1, True)
        self.assert_cfg(bool, p + "norm_topk_prob", True, True)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, p + "rms_norm_eps", no_default)

        # Engram. Layer ids count from one; the head sizes (hash moduli) come per layer, bigram heads first
        self.engram_layer_ids = [i - 1 for i in self.read_cfg(list, p + "engram_layer_ids", [])]
        self.engram_ngram_size = self.read_cfg(int, p + "engram_max_ngram_size", 3)
        self.engram_n_heads = self.read_cfg(int, p + "engram_n_head_per_ngram", 8)
        self.engram_embed_dim = self.read_cfg(int, p + "engram_n_embed_per_ngram", 768)
        self.engram_pad_id = self.read_cfg(int, p + "engram_pad_id", no_default if self.engram_layer_ids else 0)
        self.engram_seed = self.read_cfg(int, p + "engram_seed", 0)
        self.engram_fp8 = self.read_cfg(bool, p + "engram_fp8_embedding", True)
        sizes = self.read_cfg(dict, p + "engram_layer_head_vocab_sizes", {})
        self.engram_head_sizes = {
            idx: [s for order in sizes[str(idx + 1)] for s in order] for idx in self.engram_layer_ids
        }
        for idx, s in self.engram_head_sizes.items():
            assert len(s) == (self.engram_ngram_size - 1) * self.engram_n_heads, \
                f"HyV4Flash: unexpected Engram head layout for layer {idx}"

        self.image_token_id = self.read_cfg(int, "image_token_id", None)
        self.image_start_token_id = self.read_cfg(int, "image_start_token_id", None)
        self.image_end_token_id = self.read_cfg(int, "image_end_token_id", None)

        # Vision tower, present when the directory carries its tensors
        self.vision = read_hy_v4_vision_config(self)
        if self.vision is None:
            del self.model_classes["vision"]
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # DFly2 drafter (MTP component), present when the directory carries its tensors
        dk = "model.dfly_layers"
        self.dfly_num_layers = self.read_cfg(int, "dfly_num_hidden_layers", 0)
        self.dfly_kv_layers = self.read_cfg(list, "dfly_target_kv_layers", [])
        index_layers = self.read_cfg([list, int], "dfly_target_index_layer", [])
        self.dfly_index_layers = index_layers if isinstance(index_layers, list) else [index_layers] * self.dfly_num_layers
        if (
            not self.dfly_num_layers or not self.stc.has_tensor(f"{dk}.final_norm.weight") or
            # (a model cut short of the layers the drafter shares has no use for it)
            max(self.dfly_kv_layers + self.dfly_index_layers, default = 0) >= self.num_hidden_layers
        ):
            del self.model_classes["mtp"]
            return
        self.dfly_mask_token_id = self.read_cfg(int, "dfly_mask_token_id", no_default)
        self.dfly_local_window = self.read_cfg(int, "dfly_context_local_window", None)
        self.dfly_block_size = self.read_cfg(int, "speculative_config->num_speculative_tokens", 5)
        # Acceptance predictor for the dynamic draft length, when the directory carries it
        self.dfly_confidence_head = self.stc.has_tensor(f"{dk}.confidence_head.proj.weight")
        self.assert_cfg(str, "dfly_mtp_fusion", "slot0", True)
        self.assert_cfg(bool, "dfly_enable_hidden_correction", True, True)
        self.assert_cfg(bool, "dfly_enable_ihc", False, True)
        self.assert_cfg(str, "dfly_mlp_type", "dense", True)
        # The MLP widths are not in the config; a projection's input side gives them, as the weight's
        # second dimension or the length of a quantized tensor's input scales
        def in_features(key: str, default: int) -> int:
            for suffix, dim in ((".weight", 1), (".suh", 0), (".su", 0)):
                meta = self.stc.get_tensor_meta(key + suffix)
                if meta is not None:
                    return meta[key + suffix]["shape"][dim]
            return default
        self.dfly_intermediate_size = self.read_cfg(int, "dfly_intermediate_size", None) or \
            in_features(f"{dk}.0.mlp.down_proj", self.intermediate_size)
        self.dfly_correction_size = self.read_cfg(int, "dfly_hidden_correction_intermediate_size", None) or \
            in_features(f"{dk}.hidden_correction.down_proj", self.hidden_size)

    def engram_multipliers(self, idx: int, compressed_vocab_size: int) -> list:
        """The odd multipliers a layer's n-grams are mixed with: numpy's default generator seeded per layer
        (by its one-based id), bounded so a product with any compressed token id stays inside int64"""
        bound = max(1, (2 ** 63 - 1) // compressed_vocab_size // 2)
        rng = np.random.default_rng(self.engram_seed + 10007 * (idx + 1))
        return (rng.integers(0, bound, size = self.engram_ngram_size, dtype = np.int64) * 2 + 1).tolist()


class HyV4FlashModel(Model):
    config_class = HyV4FlashConfig

    def __init__(
        self,
        config: HyV4FlashConfig,
        key_prefix: str = "model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
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
            key = f"{key_prefix}.layers.{idx}"
            self.modules += self.make_engram(idx, key)
            dense = config.mlp_layer_types[idx] == "dense"

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(config, f"{key}.input_layernorm", config.rms_norm_eps),
                    attn = self.make_attn(idx, key),
                    attn_post_norm = RMSNorm(config, f"{key}.self_attn.post_proj_norm", config.rms_norm_eps),
                    attn_hc = self.make_hc(f"{key}.hc_attn_layer"),
                    mlp_norm = RMSNorm(config, f"{key}.post_attention_layernorm", config.rms_norm_eps),
                    mlp = self.make_mlp(idx, key),
                    mlp_post_norm = RMSNorm(
                        config,
                        f"{key}.mlp.post_fc2_norm" if dense else f"{key}.mlp.combine_output_norm",
                        config.rms_norm_eps,
                    ),
                    mlp_hc = self.make_hc(f"{key}.hc_mlp_layer"),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embed_tokens"

        self.modules += [
            HyperHead(
                config = config,
                key = f"{key_prefix}.hc_head.hc_head",
                hc_mult = config.hc_mult,
                rms_norm_eps = config.rms_norm_eps,
                hc_eps = config.hc_eps,
            ),
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
        self.calibration_all_experts = True
        if config.engram_layer_ids:
            # The Engram layers hash each token with the ones before it; the caller supplies those
            self.caps.update({"prev_ids": config.engram_ngram_size - 1})

        # Image rows take three-axis positions: the caller builds a per-token frequency table for them
        # with the indexed layers' base, and the sliding layers rescale it to their own
        assert config.rope_settings_sliding.rope_scaling == config.rope_settings_indexed.rope_scaling and \
            config.rope_settings_sliding.head_dim == config.rope_settings_indexed.head_dim, \
            "HyV4Flash: the layer types must share their M-RoPE layout"
        self.caps.update({"mrope": True})
        self.g_rope = RoPE("cpu", config.rope_settings_indexed)

    def make_attn(self, idx: int, key: str) -> MLAttention:
        config = self.config
        sliding = config.layer_types[idx] == "sliding"
        return MLAttention(
            config = config,
            key = f"{key}.self_attn",
            layer_idx = idx,
            hidden_size = config.hidden_size,
            num_q_heads = config.swa_num_q_heads if sliding else config.num_q_heads,
            kv_lora_rank = config.kv_lora_rank,
            qk_nope_head_dim = config.swa_qk_nope_head_dim if sliding else config.qk_nope_head_dim,
            qk_rope_head_dim = config.swa_qk_rope_head_dim if sliding else config.qk_rope_head_dim,
            v_head_dim = config.swa_v_head_dim if sliding else config.v_head_dim,
            rope_settings = config.rope_settings_sliding if sliding else config.rope_settings_indexed,
            table_rope_settings = config.rope_settings_indexed if sliding else None,
            q_lora_rank = config.q_lora_rank,
            rms_norm_eps = config.rms_norm_eps,
            qmap = "block.attn",
            out_dtype = torch.float,
            select_hq_bits = 2,
            key_out_gate = "linear_gate",
            key_sinks = "learnable_sink_param",
            window = config.sliding_window if sliding else 0,
            indexer_mode = None if sliding else "full",
            index_n_heads = config.index_n_heads,
            index_head_dim = config.index_head_dim,
            index_topk = config.index_topk,
            index_norm_eps = config.rms_norm_eps,
            index_kpool = 0 if sliding else config.index_kpool,
            index_kpool_mode = "proj",
            index_keys = {
                "wq_b": "query_projection",
                "wk": "token_key_projection",
                "k_norm": "token_key_norm",
                "weights": "head_weight_projection",
                "ape": "intra_group_ape",
            },
            key_indexer = "compressed_indexer",
        )

    def make_mlp(self, idx: int, key: str):
        config = self.config
        if config.mlp_layer_types[idx] == "dense":
            return GatedMLP(
                config = config,
                key = f"{key}.mlp",
                hidden_size = config.hidden_size,
                intermediate_size = config.intermediate_size,
                key_up = "up_proj",
                key_gate = "gate_proj",
                key_down = "down_proj",
                activation_fn = "silu",
                qmap = "block.mlp",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                select_hq_bits = 1,
            )
        # Only the routed experts clamp their activations; they run at the latent width between a pair of
        # projections, while the router and the shared expert read the block input
        return BlockSparseMLP(
            config = config,
            key = f"{key}.mlp",
            hidden_size = config.hidden_size,
            intermediate_size = config.moe_intermediate_size,
            num_experts = config.num_experts,
            num_experts_per_tok = config.num_experts_per_tok,
            latent_size = config.moe_latent_size,
            key_latent_in = "routed_expert_down_proj",
            key_latent_out = "routed_expert_up_proj",
            latent_hq_bits = 2,
            key_up = "experts.{expert_idx}.up_proj",
            key_gate = "experts.{expert_idx}.gate_proj",
            key_down = "experts.{expert_idx}.down_proj",
            key_gate_up_split = "experts.gate_up_proj",
            key_down_split = "experts.down_proj",
            transpose_fused_weights = False,
            key_routing_gate = "gate",
            key_e_score_bias = "gate.e_score_correction_bias",
            activation_fn = "silu",
            act_limit = config.swiglu_limit,
            qmap = "block.mlp",
            interm_dtype = torch.half,
            out_dtype = torch.float,
            router_type = "dots",
            routed_scaling_factor = config.routed_scaling_factor,
            n_group = 1,
            topk_group = 1,
            shared_experts = GatedMLP(
                config = config,
                key = f"{key}.mlp.shared_experts",
                hidden_size = config.hidden_size,
                intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                key_up = "up_proj",
                key_gate = "gate_proj",
                key_down = "down_proj",
                activation_fn = "silu",
                qmap = "block.mlp",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                select_hq_bits = 2,
            ) if config.num_shared_experts else None,
        )

    def make_hc(self, key: str) -> HyperConnection:
        config = self.config
        return HyperConnection(
            config = config,
            key = f"{key}.hc_pre.hc",
            hc_mult = config.hc_mult,
            hidden_size = config.hidden_size,
            sinkhorn_iters = 0,
            hc_eps = config.hc_eps,
            rms_norm_eps = config.rms_norm_eps,
            comb = False,
            post_scale = config.hc_magnitude,
            post_eps = config.hc_eps,
        )

    def make_engram(self, idx: int, key: str) -> list:
        config = self.config
        if idx not in config.engram_layer_ids:
            return []
        ek = f"{key}.engram_layer"
        # The checkpoint carries the token map (tokens that normalize alike share an id); the hash
        # multipliers are drawn against its range
        token_map = config.stc.get_tensor(f"{ek}.lookup_table", "cpu", no_defer = True).long()
        pad = int(token_map[config.engram_pad_id])
        if token_map.shape[0] < config.vocab_size:
            # Ids past the map (the embedding table pads beyond the tokenizer) read as padding
            token_map = torch.cat((token_map, token_map.new_full((config.vocab_size - token_map.shape[0],), pad)))
        H = config.hc_mult
        heads = (config.engram_ngram_size - 1) * config.engram_n_heads
        return [
            PLELayer(
                config = config,
                key = ek,
                layer_idx = -(idx + 1),
                hidden_size = config.hidden_size,
                hc_mult = H,
                ple_embed_dim = (config.engram_ngram_size - 1) * config.engram_embed_dim,
                ngram_size = config.engram_ngram_size,
                heads_per_ngram = config.engram_n_heads,
                eos_token_id = pad,
                conv_kernel_size = 0,
                rms_norm_eps = ENGRAM_NORM_EPS,
                # Image rows hash as the placeholder token and take no update, like padding
                mm_token_id = config.image_token_id,
                mm_gate_off = True,
                off_token_id = config.engram_pad_id,
                ngram_key = f"{ek}.multi_head_embedding.embedding",
                ngram_segment = False,
                ngram_fp8_rows = config.engram_fp8,
                stateless = True,
                norm_constant_bias = 0.0,
                token_map = token_map,
                token_map_key = "lookup_table",
                hash_params = (
                    config.engram_head_sizes[idx],
                    config.engram_multipliers(idx, int(token_map.max()) + 1),
                ),
                key_proj_alt = [f"{ek}.key_projs.{i}" for i in range(H)],
                norm_alt_keys = (
                    [f"{ek}.norm_key.{i}.weight" for i in range(H)],
                    [f"{ek}.norm_query.{i}.weight" for i in range(H)],
                ),
                head_norm_keys = [f"{ek}.lookup_per_head_norms.{h}.weight" for h in range(heads)],
                value_norm_key = "unified_pre_gamma",
                branch_scale_keys = [f"{ek}.unified_per_hc_scales.{i}" for i in range(H)],
            )
        ]

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # The checkpoint's template without reasoning: the system turn always carries the reasoning mode,
        # and the assistant turn opens with a closed think block
        s, m, e = "<｜hy_start:opensource｜>", "<｜hy_middle:opensource｜>", "<｜hy_end:opensource｜>"
        p = f"{s}system{m}{system_prompt or ''}<｜reasoning_mode:opensource｜>reasoning_effort:no_think{e}"
        p += f"{s}user{m}{prompt}{e}"
        p += f"{s}assistant{m}<think:opensource></think:opensource>"
        return p
