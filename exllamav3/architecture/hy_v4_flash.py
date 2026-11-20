from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import RMSNorm, Embedding, GatedMLP, Linear, BlockSparseMLP, ExpandStreams
from ..modules.attn import prepare_for_attn
from ..modules.arch_specific.hy_v4_flash import (
    HyV4Attention, HyV4Block, HyV4Engram, HyV4HyperConnection, HyV4HyperHead, HyV4SideLayer, RawTensors,
)

# Hy4-flash (hy_v4_flash): 42 gated-MLA layers (12 read through the compressed indexer, 30
# through a 512-token window), identity hyper-connections over 4 streams, a dense first MLP and
# 41 latent MoE layers (512 experts of 1536 x 1536 between two shared projections, sigmoid
# router with a score bias, 16 a token), 4 Engram n-gram tables, the DFly2 draft layers and a
# SigLIP2-style vision tower.
#
# Converter-side architecture: the trunk runs the reference arithmetic row by row without a
# cache (calibration and check forwards); the DFly2 and vision components are tensor layouts
# for the uncalibrated side quantization and have no forward here.

SPARSE_TYPES = ("deepseek_sparse_attention", "indexed_attention")


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
            config_key = "text_config",
            **kwargs
        )
        r = self.read_cfg

        # Attention
        self.hidden_size = r(int, "hidden_size", no_default)
        self.num_hidden_layers = r(int, "num_hidden_layers", no_default)
        self.layer_types = r(list, "layer_types", no_default)
        assert len(self.layer_types) == self.num_hidden_layers
        assert all(t in SPARSE_TYPES + ("sliding_attention",) for t in self.layer_types)
        self.q_lora_rank = r(int, "q_lora_rank", no_default)
        self.kv_lora_rank = r(int, "kv_lora_rank", no_default)
        self.attn_dims = {
            "sparse": dict(
                num_heads = r(int, "num_attention_heads", no_default),
                qk_nope_head_dim = r(int, "qk_nope_head_dim", no_default),
                qk_rope_head_dim = r(int, "qk_rope_head_dim", no_default),
                v_head_dim = r(int, "v_head_dim", no_default),
            ),
        }
        s = self.attn_dims["sparse"]
        self.attn_dims["sliding"] = dict(
            num_heads = r(int, "swa_num_attention_heads", 64),
            qk_nope_head_dim = r(int, "swa_qk_nope_head_dim", s["qk_nope_head_dim"]),
            qk_rope_head_dim = r(int, "swa_qk_rope_head_dim", s["qk_rope_head_dim"]),
            v_head_dim = r(int, "swa_v_head_dim", s["v_head_dim"]),
        )
        self.assert_cfg(bool, "gated_mla", True, True)
        self.assert_cfg(bool, "learnable_sink", True, True)
        self.assert_cfg(bool, "sandwich_norm", True, True)
        self.assert_cfg(bool, "attention_bias", False, True)
        self.sliding_window = r(int, "sliding_window", 512)
        rope = r(dict, "rope_parameters", {})
        sparse_rope = next((rope[k] for k in SPARSE_TYPES if k in rope), {})
        self.rope_theta = {
            "sparse": float(sparse_rope.get("rope_theta", 10000000.0)),
            "sliding": float(rope.get("sliding_attention", {}).get("rope_theta", 10000.0)),
        }

        # Compressed indexer
        self.index_topk = r(int, "index_topk", 2048)
        self.index_n_heads = r(int, ["index_n_heads", "compressed_indexer_num_attention_heads"], 16)
        self.index_head_dim = r(int, ["index_head_dim", "compressed_indexer_head_dim"], 128)
        self.index_kpool = r(int, ["compressed_indexer_compress_ratio", "index_kpool"], 4)
        self.assert_cfg(str, "compressed_indexer_key_pooling", "gated", True)
        assert self.index_topk % self.index_kpool == 0

        # Hyper-connections
        self.hc_mult = r(int, "hc_mult", 4)
        self.hc_eps = r(float, "hc_eps", 1e-6)
        self.hc_magnitude = r(float, "hc_magnitude", 2.0)

        # MLP
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.assert_cfg(bool, "moe_router_use_sigmoid", True, True)
        self.assert_cfg(bool, "moe_router_enable_expert_bias", True, True)
        self.assert_cfg(bool, "norm_topk_prob", True, True)
        self.assert_cfg(int, "n_group", 1, True)
        self.assert_cfg(int, "topk_group", 1, True)
        self.intermediate_size = r(int, "intermediate_size", no_default)
        self.moe_intermediate_size = r(int, "moe_intermediate_size", no_default)
        self.routed_expert_hidden_size = r(int, "routed_expert_hidden_size", no_default)
        self.num_experts = r(int, "n_routed_experts", no_default)
        self.num_experts_per_tok = r(int, "num_experts_per_tok", no_default)
        self.num_shared_experts = r(int, "n_shared_experts", 1)
        self.routed_scaling_factor = r(float, "routed_scaling_factor", no_default)
        self.swiglu_limit = r(float, "swiglu_limit", 10.0)
        self.mlp_layer_types = r(list, "mlp_layer_types", None) or \
            (["dense"] * min(1, self.num_hidden_layers) + ["sparse"] * max(self.num_hidden_layers - 1, 0))

        # Engram
        self.engram_layer_ids = r(list, "engram_layer_ids", []) or []          # 1-based: layer idx + 1
        self.engram_max_ngram_size = r(int, "engram_max_ngram_size", 3)
        self.engram_heads_per_ngram = r(int, "engram_n_head_per_ngram", 8)
        self.engram_embed_per_ngram = r(int, "engram_n_embed_per_ngram", 768)
        self.engram_row_dim = self.engram_embed_per_ngram // self.engram_heads_per_ngram
        self.engram_pad_id = r(int, "engram_pad_id", 120002)
        self.engram_seed = r(int, "engram_seed", 0)
        self.engram_fp8 = r(bool, "engram_fp8_embedding", True)
        sizes = r(dict, "engram_layer_head_vocab_sizes", {}) or {}
        self.engram_head_vocab_sizes = {
            int(k): [n for order in v for n in order] for k, v in sizes.items()
        }
        # The reference builds this layer's norms without an eps: torch computes them in fp32 and takes the
        # fp32 machine epsilon, for bf16 inputs too. See HyV4Engram
        self.engram_norm_eps = 1.1920928955078125e-07
        self.video_token_id = r(int, "video_token_id", None)

        # Norms, head
        self.rms_norm_eps = r(float, "rms_norm_eps", no_default)
        self.tie_word_embeddings = r(bool, "tie_word_embeddings", False)

        # DFly2 draft layers (side component, present when the checkpoint carries them)
        self.dfly_num_layers = r(int, "dfly_num_hidden_layers", 0)
        def has_linear(key):   # as stored in the checkpoint, or as an EXL3 group in a pack
            return self.stc.has_tensor(key + ".weight") or self.stc.has_tensor(key + ".trellis")
        self.has_linear = has_linear
        if self.dfly_num_layers == 0 or not has_linear("model.dfly_layers.0.self_attn.q_a_proj"):
            self.dfly_num_layers = 0
            del self.model_classes["mtp"]

        # Vision tower (side component)
        self.vision_num_layers = r(int, "vision_config->num_hidden_layers", 0)
        self.vision_hidden_size = r(int, "vision_config->hidden_size", 0)
        self.vision_intermediate_size = r(int, "vision_config->intermediate_size", 0)
        if self.vision_num_layers == 0 or not has_linear("vit.layers.0.self_attn.q_proj"):
            self.vision_num_layers = 0
            del self.model_classes["vision"]

    def default_max_position_embeddings(self):
        return 1048576

    def attn_type(self, idx: int) -> str:
        return "sparse" if self.layer_types[idx] in SPARSE_TYPES else "sliding"


def make_attn(config: HyV4FlashConfig, key: str, idx: int, layer_type: str, qbits_key: str = "bits",
              qmap: str | None = "block.attn") -> HyV4Attention:
    sparse = layer_type == "sparse"
    return HyV4Attention(
        config = config,
        key = key,
        layer_idx = idx,
        layer_type = layer_type,
        hidden_size = config.hidden_size,
        q_lora_rank = config.q_lora_rank,
        kv_lora_rank = config.kv_lora_rank,
        rope_theta = config.rope_theta[layer_type],
        rms_norm_eps = config.rms_norm_eps,
        sliding_window = 0 if sparse else config.sliding_window,
        index_n_heads = config.index_n_heads if sparse else 0,
        index_head_dim = config.index_head_dim if sparse else 0,
        index_topk = config.index_topk if sparse else 0,
        index_kpool = config.index_kpool if sparse else 0,
        qmap = qmap,
        qbits_key = qbits_key,
        select_hq_bits = 2,
        **config.attn_dims[layer_type],
    )


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
            ),
        ]

        self.first_block_idx = len(self.modules)

        def hc(key):
            return HyV4HyperConnection(config, key, config.hc_mult, config.hc_eps, config.rms_norm_eps,
                                       config.hc_magnitude)

        def norm(key):
            return RMSNorm(config = config, key = key, rms_norm_eps = config.rms_norm_eps)

        for idx in range(config.num_hidden_layers):
            key = f"{key_prefix}.layers.{idx}"
            if (idx + 1) in config.engram_layer_ids:
                self.modules.append(HyV4Engram(
                    config = config,
                    key = f"{key}.engram_layer",
                    layer_idx = -(idx + 1),
                    layer_id = idx + 1,
                    hidden_size = config.hidden_size,
                    hc_mult = config.hc_mult,
                    max_ngram_size = config.engram_max_ngram_size,
                    heads_per_ngram = config.engram_heads_per_ngram,
                    row_dim = config.engram_row_dim,
                    head_vocab_sizes = config.engram_head_vocab_sizes[idx + 1],
                    pad_id = config.engram_pad_id,
                    seed = config.engram_seed,
                    fp8_rows = config.engram_fp8,
                    norm_eps = config.engram_norm_eps,
                    no_ngram_ids = (config.image_token_id, config.video_token_id, config.engram_pad_id),
                ))

            if config.mlp_layer_types[idx] == "dense":
                # every token passes this MLP: promoted with --hq like the shared experts
                mlp = GatedMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                )
                post = norm(f"{key}.mlp.post_fc2_norm")
            else:
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    latent_size = config.routed_expert_hidden_size,
                    key_latent_in = "routed_expert_down_proj",
                    key_latent_out = "routed_expert_up_proj",
                    latent_hq_bits = 2,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_gate = "experts.{expert_idx}.gate_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_gate_up_split = "experts.gate_up_proj",
                    key_down_split = "experts.down_proj",
                    key_routing_gate = "gate",
                    key_e_score_bias = "gate.e_score_correction_bias",
                    transpose_fused_weights = False,
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    activation_fn = "silu",
                    act_limit = config.swiglu_limit,
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
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                    ),
                )
                post = norm(f"{key}.mlp.combine_output_norm")

            self.modules.append(HyV4Block(
                config = config,
                key = key,
                layer_idx = idx,
                attn_hc = hc(f"{key}.hc_attn_layer.hc_pre"),
                attn_norm = norm(f"{key}.input_layernorm"),
                attn = make_attn(config, f"{key}.self_attn", idx, config.attn_type(idx)),
                mlp_hc = hc(f"{key}.hc_mlp_layer.hc_pre"),
                mlp_norm = norm(f"{key}.post_attention_layernorm"),
                mlp = mlp,
                mlp_post_norm = post,
            ))

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            HyV4HyperHead(config, f"{key_prefix}.hc_head", config.hc_mult, config.hc_eps, config.rms_norm_eps),
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
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        self.caps.update({
            "supports_tp": False,
            # the trunk here is the cache-less reference forward: no KV cache, no decode kernels
            "cacheless_only": True,
        })

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # As chat_template.jinja with reasoning_effort no_think: the system turn is always present
        p = "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>"
        if system_prompt:
            p += system_prompt
        p += "<｜reasoning_mode:opensource｜>reasoning_effort:no_think<｜hy_end:opensource｜>"
        p += f"<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>{prompt}<｜hy_end:opensource｜>"
        p += "<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource></think:opensource>"
        return p


def _side_linear(config, key, i, o, bits_key, hq = 0):
    return Linear(config, key, i, o, qmap = "side", qbits_key = bits_key, select_hq_bits = hq)


class HyV4FlashDFlyModel(Model):
    """DFly2 draft layers: 5 gated-MLA layers with dense MLPs that read the target's KV rows and
    indexer picks, plus the fusion, hidden-correction and confidence pieces. Tensor layout only."""
    config_class = HyV4FlashConfig

    def __init__(self, config: HyV4FlashConfig, key_prefix: str = "model.dfly_layers", **kwargs):
        super().__init__(config, **kwargs)
        D = config.hidden_size
        a = config.attn_dims["sparse"]
        H, qk, v = a["num_heads"], a["qk_nope_head_dim"] + a["qk_rope_head_dim"], a["v_head_dim"]
        stc = config.stc
        B = "mtp_bits"

        def out_of(key):
            """(out, in) of a linear from its stored weight, or from its EXL3 tiles in a pack (16 x 16 a tile)"""
            if stc.has_tensor(key):
                return stc.get_tensor_meta(key)[key]["shape"]
            t = key[:-len(".weight")] + ".trellis"
            shape = stc.get_tensor_meta(t)[t]["shape"]
            return [shape[1] * 16, shape[0] * 16]

        for idx in range(config.dfly_num_layers):
            key = f"{key_prefix}.{idx}"
            inter = out_of(f"{key}.mlp.gate_proj.weight")[0]
            lin = [
                _side_linear(config, f"{key}.self_attn.q_a_proj", D, config.q_lora_rank, B, 2),
                _side_linear(config, f"{key}.self_attn.q_b_proj", config.q_lora_rank, H * qk, B, 2),
                _side_linear(config, f"{key}.self_attn.kv_a_proj_with_mqa", D,
                             config.kv_lora_rank + a["qk_rope_head_dim"], B, 2),
                _side_linear(config, f"{key}.self_attn.kv_b_proj", config.kv_lora_rank,
                             H * (a["qk_nope_head_dim"] + v), B, 2),
                _side_linear(config, f"{key}.self_attn.linear_gate", D, H * v, B, 2),
                _side_linear(config, f"{key}.self_attn.o_proj", H * v, D, B, 2),
                _side_linear(config, f"{key}.mlp.gate_proj", D, inter, B),
                _side_linear(config, f"{key}.mlp.up_proj", D, inter, B),
                _side_linear(config, f"{key}.mlp.down_proj", inter, D, B),
            ]
            raw = RawTensors(config, key, [
                "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_a_layernorm.weight",
                "self_attn.kv_a_layernorm.weight", "self_attn.post_proj_norm.weight",
                "self_attn.learnable_sink_param",
            ])
            self.modules.append(HyV4SideLayer(config, key, lin, raw))

        key = f"{key_prefix}.hidden_correction"
        if config.has_linear(f"{key}.gate_proj"):
            hin = out_of(f"{key}.gate_proj.weight")
            self.modules.append(HyV4SideLayer(config, key, [
                _side_linear(config, f"{key}.gate_proj", hin[1], hin[0], B),
                _side_linear(config, f"{key}.up_proj", hin[1], hin[0], B),
                _side_linear(config, f"{key}.down_proj", *reversed(out_of(f"{key}.down_proj.weight")), B),
            ], RawTensors(config, key, ["embed_norm.weight", "hidden_norm.weight"])))
        key = f"{key_prefix}.mtp_fusion"
        if config.has_linear(f"{key}.eh_proj"):
            self.modules.append(HyV4SideLayer(config, key, [
                _side_linear(config, f"{key}.eh_proj", *reversed(out_of(f"{key}.eh_proj.weight")), B),
            ], RawTensors(config, key, ["enorm.weight", "hnorm.weight", "gate"])))
        key = f"{key_prefix}.confidence_head"
        if stc.has_tensor(f"{key}.proj.weight"):
            self.modules.append(HyV4SideLayer(config, key, [], RawTensors(config, key, ["proj.weight", "proj.bias"])))
        key = f"{key_prefix}.final_norm"
        if stc.has_tensor(f"{key}.weight"):
            self.modules.append(HyV4SideLayer(config, key, [], RawTensors(config, key, ["weight"])))

        self.caps.update({"supports_tp": False, "can_quantize": True})


class HyV4FlashVisionModel(Model):
    """Vision tower (27 pre-norm ViT layers) and projector. Tensor layout only: the q/k/v/o and
    MLP projections take --vision_bits, the patch embedding, positions and projector convolutions
    stay as stored."""
    config_class = HyV4FlashConfig

    def __init__(self, config: HyV4FlashConfig, key_prefix: str = "vit", **kwargs):
        super().__init__(config, **kwargs)
        D, I = config.vision_hidden_size, config.vision_intermediate_size
        B = "bits"
        for idx in range(config.vision_num_layers):
            key = f"{key_prefix}.layers.{idx}"
            lin = [
                _side_linear(config, f"{key}.self_attn.q_proj", D, D, B),
                _side_linear(config, f"{key}.self_attn.k_proj", D, D, B),
                _side_linear(config, f"{key}.self_attn.v_proj", D, D, B),
                _side_linear(config, f"{key}.self_attn.o_proj", D, D, B),
                _side_linear(config, f"{key}.mlp.dense_h_to_4h", D, I, B),
                _side_linear(config, f"{key}.mlp.dense_4h_to_h", I, D, B),
            ]
            raw = RawTensors(config, key, [
                "input_layernorm.weight", "input_layernorm.bias",
                "post_attention_layernorm.weight", "post_attention_layernorm.bias",
            ])
            self.modules.append(HyV4SideLayer(config, key, lin, raw))
        key = f"{key_prefix}.embeddings"
        self.modules.append(HyV4SideLayer(config, key, [], RawTensors(config, key, [
            "patch_embedding.weight", "patch_embedding.bias", "position_embedding.weight",
        ])))
        key = f"{key_prefix}.perceive"
        names = ["image_newline", "norm.weight", "proj.0.weight", "proj.0.bias", "proj.2.weight", "proj.2.bias"]
        self.modules.append(HyV4SideLayer(config, key, [], RawTensors(config, key, names, optional = names)))

        self.caps.update({"supports_tp": False, "default_vision_bits": 16})
