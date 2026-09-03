from __future__ import annotations
from typing_extensions import override
import math
import os, json
import numpy as np
import torch
from PIL import Image

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear, BlockSparseMLP,
    GatedDeltaNet,
)
from ..modules.hyperconnections import ExpandStreams, HyperConnection, HyperHead
from ..modules.gated_delta_net import GDNState
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from .glm4v import Glm4VVisionModel, read_glm4v_vision_config, read_glm4v_pp_config
from .mm_processing.glm5_next import glm5_vision_canvas

# GLM-5.3-Flash (Glm5NextForConditionalGeneration, text component): hybrid of KDA linear
# attention (Kimi-style per-channel-decay delta rule, 3:1) and NoPE MLA with a k-pool-
# compressed DSA indexer, DeepSeek-V4-style mHC hyper-connections on every layer (final
# collapse is an unweighted mean), sigmoid noaux_tc MoE with clamped-silu MLPs
# (swiglu_limit), and a DS3-style MTP head. Reference: transformers models/glm5_next.


class Glm5NextConfig(Config):
    arch_string = "Glm5NextForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        from .glm5_next_mtp import Glm5NextMTPModel
        super().__init__(
            directory,
            {"text": Glm5NextModel, "mtp": Glm5NextMTPModel, "vision": Glm5NextVisionModel},
            **kwargs
        )

        # Attention (NoPE MLA) params
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.q_lora_rank = self.read_cfg(int, "text_config->q_lora_rank", no_default)
        self.kv_lora_rank = self.read_cfg(int, "text_config->kv_lora_rank", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, "text_config->qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, "text_config->qk_rope_head_dim", 0)
        assert self.qk_rope_head_dim == 0, \
            "Glm5Next expects NoPE attention (qk_rope_head_dim == 0)"
        self.v_head_dim = self.read_cfg(int, "text_config->v_head_dim", no_default)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.sm_scale = self.qk_head_dim ** -0.5
        self.rope_settings = None

        # Layer schedule
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.layer_types = self.read_cfg(list, "text_config->layer_types", no_default)
        assert len(self.layer_types) >= self.num_hidden_layers and all(
            t in ("linear_attention", "deepseek_sparse_attention") for t in self.layer_types)

        # KDA linear attention
        la = self.read_cfg(dict, "text_config->linear_attn_config", no_default)
        self.linear_num_heads = la["num_heads"]
        self.linear_head_dim = la["head_dim"]
        self.linear_conv_kernel_size = la["short_conv_kernel_size"]
        self.linear_lower_bound = la.get("gate_lower_bound")

        # DSA indexer with k-pool compression
        self.index_n_heads = self.read_cfg(int, "text_config->index_n_heads", no_default)
        self.index_head_dim = self.read_cfg(int, "text_config->index_head_dim", no_default)
        self.index_topk = self.read_cfg(int, "text_config->index_topk", no_default)
        self.index_kpool = self.read_cfg(int, "text_config->index_kpool", no_default)
        self.index_kpool_tail = self.read_cfg(bool, "text_config->index_kpool_always_select_tail", True)
        self.assert_cfg(bool, "text_config->index_kpool_compress", True, True)
        self.indexer_types = self.read_cfg(list, "text_config->indexer_types", no_default)
        assert all(t in ("full", "shared") for t in self.indexer_types)

        # mHC hyper-connections
        self.hc_mult = self.read_cfg(int, "text_config->hc_mult", 4)
        self.hc_sinkhorn_iters = self.read_cfg(int, "text_config->hc_sinkhorn_iters", 20)
        self.hc_eps = self.read_cfg(float, "text_config->hc_eps", 1e-6)
        self.assert_cfg(bool, "text_config->mhc", True, True)

        # MLP params
        self.assert_cfg(str, "text_config->hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "text_config->moe_intermediate_size", no_default)
        self.num_shared_experts = self.read_cfg(int, "text_config->n_shared_experts", 1)
        self.num_experts = self.read_cfg(int, "text_config->n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "text_config->num_experts_per_tok", 8)
        self.routed_scaling_factor = self.read_cfg(float, "text_config->routed_scaling_factor", 1.0)
        self.swiglu_limit = self.read_cfg(float, "text_config->swiglu_limit", 0.0)
        first_k_dense = self.read_cfg(int, "text_config->first_k_dense_replace", 3)
        self.mlp_layer_types = self.read_cfg(
            list, "text_config->mlp_layer_types",
            ["dense" if idx < first_k_dense else "sparse" for idx in range(self.num_hidden_layers)]
        )
        assert all(t in ("dense", "sparse") for t in self.mlp_layer_types)
        self.n_group = self.read_cfg(int, "text_config->n_group", 1)
        self.topk_group = self.read_cfg(int, "text_config->topk_group", 1)
        assert self.n_group in (None, 1) and self.topk_group in (None, 1), \
            "Group-limited expert routing is not supported"
        self.assert_cfg(str, "text_config->scoring_func", "sigmoid", True)
        self.assert_cfg(str, "text_config->topk_method", "noaux_tc", True)
        self.assert_cfg(bool, "text_config->norm_topk_prob", True, True)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)

        # Layers
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # MTP head (model.language_model.layers.{num_hidden_layers}). Quantized models
        # converted without the head lack its tensors; a convert_mtp.py output file placed
        # alongside the model's .safetensors files augments them
        self.num_mtp_layers = self.read_cfg(int, "text_config->num_nextn_predict_layers", 0)

        # Vision tower: GLM4V structure with GLM5.3 deltas
        vc = self.read_cfg(dict, "vision_config", no_default)
        self.vision = read_glm4v_vision_config(vc)

        prep_path = os.path.join(self.directory, "processor_config.json")
        with open(prep_path, encoding = "utf8") as f:
            prep = json.load(f)
        ip = prep["image_processor"]
        vp = prep.get("video_processor", ip)
        self.vision_pp = read_glm4v_pp_config(ip, glm5_next = True)
        self.vision_pp.max_video_tokens = vp.get("max_image_tokens", ip.get("max_image_tokens"))

        self.vision_start_token_id = self.read_cfg(int, "image_start_token_id", no_default)
        self.vision_end_token_id = self.read_cfg(int, "image_end_token_id", no_default)


class Glm5NextModel(Model):
    config_class = Glm5NextConfig

    def __init__(
        self,
        config: Glm5NextConfig,
        key_prefix: str = "model.language_model",
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

            if config.layer_types[idx] == "linear_attention":
                attn = GatedDeltaNet(
                    config = config,
                    key = f"{key}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    k_head_dim = config.linear_head_dim,
                    v_head_dim = config.linear_head_dim,
                    num_k_heads = config.linear_num_heads,
                    num_v_heads = config.linear_num_heads,
                    rms_norm_eps = config.rms_norm_eps,
                    conv_kernel_size = config.linear_conv_kernel_size,
                    key_qkv = "qkv_proj",
                    key_qkv_alt = ["q_proj", "k_proj", "v_proj"],
                    key_conv1d = "conv1d",
                    key_conv1d_q = "q_conv1d",
                    key_conv1d_k = "k_conv1d",
                    key_conv1d_v = "v_conv1d",
                    key_b = "b_proj",
                    key_f_a = "f_a_proj",
                    key_f_b = "f_b_proj",
                    key_g_a = "g_a_proj",
                    key_g_b = "g_b_proj",
                    gate_lower_bound = config.linear_lower_bound,
                    key_a_log = "A_log",
                    key_dt_bias = "dt_bias",
                    key_norm = "o_norm",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                )
            else:
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
                    indexer_mode = config.indexer_types[idx],
                    index_n_heads = config.index_n_heads,
                    index_head_dim = config.index_head_dim,
                    index_topk = config.index_topk,
                    index_kpool = config.index_kpool,
                    index_kpool_tail = config.index_kpool_tail,
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
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    attn_hc = _hc("attn"),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = (
                        GatedMLP(
                            config = config,
                            key = f"{key}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.intermediate_size,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            activation_fn = "silu",
                            act_limit = config.swiglu_limit,
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            select_hq_bits = 1,
                        )
                        if config.mlp_layer_types[idx] == "dense" else
                        BlockSparseMLP(
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
                                select_hq_bits = 2,
                            ) if config.num_shared_experts else None,
                        )
                    ),
                    mlp_hc = _hc("ffn"),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embed_tokens"

        self.modules += [
            HyperHead(
                config = config,
                key = "hc_head",
                hc_mult = config.hc_mult,
                rms_norm_eps = config.rms_norm_eps,
                hc_eps = config.hc_eps,
                mean = True,
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
        self.caps.update({
            "supports_tp": False,
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True,
        })
        self.recurrent_state_cls = GDNState


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = f"[gMASK]<sop>"
        if system_prompt:
            p += f"<|system|>\n{system_prompt}"
        p += f"<|user|>\n{prompt}"
        p += f"<|assistant|>\n"
        return p




class Glm5NextVisionModel(Glm4VVisionModel):
    """
    GLM5.3-Flash vision tower. Derived from Glm4VVisionModel. Deltas carried on config.vision
    and only preprocessing differs in code: a token-budget canvas resize that scales the content
    preserving aspect (never upscaling above the minimum budget) and zero-pads to the aligned
    canvas, instead of the qwen2 aspect warp.
    """

    def preprocess(
        self,
        images: Image | list[Image]
    ) -> (torch.Tensor, tuple):
        from .mm_processing.common import convert_to_rgb, normalize_image
        v = self.config.vision
        pp = self.config.vision_pp
        resample = Image.Resampling(pp.resample)
        image_mean, image_std = tuple(pp.image_mean), tuple(pp.image_std)
        tps = pp.temporal_patch_size
        factor = pp.patch_size * pp.merge_size

        if not isinstance(images, list):
            mode = "image"
            images = [images]
            max_tokens = pp.max_image_tokens
        else:
            mode = "video"
            frames = len(images)
            if frames > 1:
                frames = frames // tps * tps
                images = images[:frames]
            max_tokens = pp.max_video_tokens

        images = [convert_to_rgb(image) for image in images]
        old_size = images[0].size
        assert all(old_size == frame.size for frame in images), \
            "All frames in video must have same dimensions"
        w, h = old_size

        # HF resizes images with num_frames = temporal_patch_size (one duplicated frame)
        num_frames = tps if mode == "image" else max(len(images), tps)
        target_h, target_w = glm5_vision_canvas(num_frames, h, w, tps, factor, pp.min_image_tokens, max_tokens)

        # Content scale preserves aspect and never upscales once the raw pixels meet the
        # minimum budget; the remainder of the canvas is zero-padded
        scale = min(target_h / h, target_w / w)
        if tps * h * w >= tps * factor ** 2 * pp.min_image_tokens:
            scale = min(1.0, scale)
        content_h = max(1, min(target_h, math.floor(h * scale)))
        content_w = max(1, min(target_w, math.floor(w * scale)))

        def to_canvas(image):
            if (content_w, content_h) != image.size:
                image = image.resize((content_w, content_h), resample = resample)
            if (content_w, content_h) != (target_w, target_h):
                canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                canvas.paste(image, (0, 0))
                image = canvas
            return image
        images = [to_canvas(image) for image in images]
        new_size = (target_w, target_h)

        images = [np.array(image).astype(np.float32) for image in images]
        images = [image * pp.rescale_factor for image in images]
        images = [normalize_image(image, image_mean, image_std) for image in images]

        # Identical patch layout to glm4v/qwen2: single frames duplicate across the
        # temporal patch, blocks are merge-major
        patches = np.array(images)
        patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (tps, 1, 1, 1))
        channels = patches.shape[1]
        grid_t = patches.shape[0] // tps
        grid_h = target_h // pp.patch_size
        grid_w = target_w // pp.patch_size
        patches = patches.reshape(
            grid_t,
            tps,
            channels,
            grid_h // v.spatial_merge_size,
            v.spatial_merge_size,
            pp.patch_size,
            grid_w // v.spatial_merge_size,
            v.spatial_merge_size,
            pp.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channels * tps * pp.patch_size ** 2
        )

        return torch.from_numpy(flatten_patches).half(), new_size, (grid_t, grid_h, grid_w)
