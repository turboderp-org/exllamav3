from __future__ import annotations
import json
import os
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from typing_extensions import override
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    Attention,
    Embedding,
    GatedMLP,
    LayerNorm,
    Linear,
    MLP,
    RMSNorm,
    SlidingAttention,
    SWAState,
    TransformerBlock,
)
from ..modules.arch_specific.muse_glimmer import (
    MuseGlimmerVisionAdapter,
    MuseGlimmerVisionPatchEmbedder,
    MuseGlimmerVisionPixelShuffle,
)
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from ..tokenizer import MMEmbedding, Tokenizer
from ..util.file import read_dict
from ..util.rope import RoPE, RopeSettings, RopeStyle
from .mm_processing.common import convert_to_rgb, normalize_image
from .mm_processing.qwen2 import get_qwen2_window_index
from .mm_processing.muse_glimmer import (
    muse_bilinear_pos_emb,
    muse_patchify,
    muse_position_embedding_grid_2d,
    muse_smart_resize,
)


class MuseGlimmerConfig(Config):
    arch_string = "MuseGlimmerForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": MuseGlimmerTextModel, "vision": MuseGlimmerVisionModel},
            **kwargs
        )

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", False)

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", None)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", self.num_q_heads)
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        self.layer_types = self.read_cfg(list, "text_config->layer_types", no_default)
        assert len(self.layer_types) == self.num_hidden_layers, \
            "Length of text_config->layer_types doesn't match number of hidden layers"
        # The reference mask attends to sliding_window tokens including self; the internal (FA)
        # convention counts past tokens only, hence the -1
        self.sliding_window = self.read_cfg(int, "text_config->sliding_window", 0) - 1
        assert self.sliding_window > 0, "Expected text_config->sliding_window > 1"

        # Scaleless QK norm, with Q additionally scaled by qk_scale_factor on top of the standard
        # 1/sqrt(head_dim); scale commutes with rotation so it folds into sm_scale
        self.qk_scale_factor = self.read_cfg(float, "text_config->qk_scale_factor", 1.0)

        # MLP params
        self.assert_cfg(str, "text_config->hidden_activation", "silu", True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)

        # Norms: pre norms use rms_norm_eps, the post (sandwich) norms use post_norm_eps
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)
        self.post_norm_eps = self.read_cfg(float, "text_config->post_norm_eps", self.rms_norm_eps)

        # Output softcap with pre-scale: logits = T * tanh(logits * output_multiplier / T)
        self.final_logit_softcapping = self.read_cfg(float, "text_config->final_logit_softcapping", 0.0)
        self.output_multiplier = self.read_cfg(float, "text_config->output_multiplier", 1.0)

        # RoPE: per-layer theta, 0 = NoPE (the full-attention layers in the released config)
        text_config_dict = self.read_cfg(dict, "text_config", no_default)
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 500000.0,
            config_dict = text_config_dict,
        )
        self.layer_rope_theta = self.read_cfg(
            list,
            "text_config->layer_rope_theta",
            [self.rope_settings.rope_theta] * self.num_hidden_layers,
        )
        assert len(self.layer_rope_theta) == self.num_hidden_layers, \
            "Length of text_config->layer_rope_theta doesn't match number of hidden layers"
        self.layer_rope_settings = []
        for theta in self.layer_rope_theta:
            if not theta:
                self.layer_rope_settings.append(None)
            elif theta == self.rope_settings.rope_theta:
                self.layer_rope_settings.append(self.rope_settings)
            else:
                self.layer_rope_settings.append(replace(self.rope_settings, rope_theta = float(theta)))

        # Vision model settings
        read_vision_config = self.read_cfg(dict, "vision_config", no_default)
        self.vision = read_muse_glimmer_vision_config(read_vision_config)
        self.image_token_id = self.read_cfg(int, "image_token_id", None)

        prep_path = os.path.join(self.directory, "processor_config.json")
        with open(prep_path, encoding = "utf8") as f:
            read_prep_config = json.load(f)
        self.vision_pp = read_muse_glimmer_pp_config(read_prep_config)


def read_muse_glimmer_vision_config(config_dict: dict):
    v = SimpleNamespace(**{
        k: read_dict(config_dict, t, k, no_default)
        for k, t in [
            ("hidden_size", int),
            ("num_hidden_layers", int),
            ("intermediate_size", int),
            ("num_attention_heads", int),
            ("patch_size", int),
            ("patch_temporal", int),
            ("merge_size", int),
            ("pos_emb_height", int),
            ("pos_emb_width", int),
            ("layer_norm_eps", float),
            ("layer_types", list),
        ]
    })
    assert read_dict(config_dict, str, "hidden_act", "gelu") == "gelu", \
        "Expected vision_config->hidden_act to be 'gelu'"
    assert v.pos_emb_height == v.pos_emb_width, \
        "Expected square position embedding table in vision_config"
    v.num_channels = 3
    v.head_dim = v.hidden_size // v.num_attention_heads
    v.rope_theta = read_dict(config_dict, float, "rope_parameters->rope_theta", 10000.0)
    v.patch_dim = v.patch_temporal * v.num_channels * v.patch_size ** 2
    v.window_size = v.pos_emb_height * v.patch_size
    assert len(v.layer_types) == v.num_hidden_layers, \
        "Length of vision_config->layer_types doesn't match number of hidden layers"
    return v


def read_muse_glimmer_pp_config(config_dict: dict):
    ip = read_dict(config_dict, dict, "image_processor", no_default)
    pp = SimpleNamespace(**{
        k: read_dict(ip, t, k, no_default)
        for k, t in [
            ("patch_size", int),
            ("temporal_patch_size", int),
            ("merge_size", int),
            ("max_image_tokens", int),
            ("image_mean", list),
            ("image_std", list),
            ("resample", int),
            ("rescale_factor", float),
        ]
    })
    return pp


class MuseGlimmerTextModel(Model):
    config_class = MuseGlimmerConfig

    def __init__(
        self,
        config: MuseGlimmerConfig,
        key_prefix: str = "model.language_model",
        swa_full: bool = False,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.swa_full = swa_full

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            ),
            # Scaleless RMSNorm on top of the embeddings (reference embed_norm). Also normalizes
            # inserted MM embeddings, but those arrive pre-normalized so the second pass is a no-op
            RMSNorm(
                config = config,
                key = f"{key_prefix}.embed_tokens.embed_norm",
                rms_norm_eps = config.rms_norm_eps,
                unweighted = True,
                out_dtype = torch.float,
            ),
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            is_swa = config.layer_types[idx] == "sliding_attention"
            attn_cls = Attention if swa_full or not is_swa else SlidingAttention

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    attn = attn_cls(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.self_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        head_dim = config.head_dim,
                        num_q_heads = config.num_q_heads,
                        num_kv_heads = config.num_kv_heads,
                        rope_settings = config.layer_rope_settings[idx],
                        sm_scale = config.qk_scale_factor * config.head_dim ** (-0.5),
                        sliding_window = config.sliding_window if is_swa else -1,
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "o_proj",
                        key_g = "gate_proj",
                        full_gate = True,
                        qmap = "block.attn",
                        q_norm = RMSNorm(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.self_attn.qk_norm",
                            rms_norm_eps = config.rms_norm_eps,
                            unweighted = True,
                        ),
                        k_norm = RMSNorm(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.self_attn.qk_norm",
                            rms_norm_eps = config.rms_norm_eps,
                            unweighted = True,
                        ),
                        out_dtype = torch.float,
                    ),
                    attn_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.post_norm_eps,
                        constant_bias = 1.0,
                        out_dtype = torch.float,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.pre_feedforward_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                    mlp_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_feedforward_layernorm",
                        rms_norm_eps = config.post_norm_eps,
                        constant_bias = 1.0,
                        out_dtype = torch.float,
                    ),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embed_tokens"

        self.modules += [
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
                pre_scale = config.output_multiplier,
                softcap = config.final_logit_softcapping,
                caps = {"logits_output": True},
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1
        self.g_rope = RoPE("cpu", config.rope_settings)

        # SWA layers are recurrent, optionally
        self.recurrent_state_cls = None
        if not self.swa_full:
            self.caps.update({
                "supports_tp": False,
                "recurrent_states": True,
                "default_recurrent_checkpoint_interval": 6144,
            })
            self.recurrent_state_cls = SWAState


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if not self.swa_full:
            prepare_for_recurrence(input_ids, params, self)
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<|begin_of_text|>"
        if system_prompt:
            p += f"<|start|>system<|message|>{system_prompt}<|eot|>"
        p += f"<|start|>user<|message|>{prompt}<|eot|>"
        p += "<|start|>assistant"
        return p


class MuseGlimmerVisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: MuseGlimmerConfig) -> dict:
        return (
            config.stc.list_tensors(prefix = "model.vision_tower") |
            config.stc.list_tensors(prefix = "model.vision_adapter") |
            config.stc.list_tensors(prefix = "model.vision_projection")
        )

    def __init__(
        self,
        config: MuseGlimmerConfig,
        key_prefix: str = "model.vision_tower",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "default_vision_bits": 6,
            "supports_tp": False,
        })
        v = self.config.vision

        self.modules += [
            MuseGlimmerVisionPatchEmbedder(
                config = config,
                key = f"{key_prefix}.patch_embedder",
                hidden_size = v.hidden_size,
                patch_dim = v.patch_dim,
            ),
            LayerNorm(
                config = config,
                key = f"{key_prefix}.ln_pre",
                layernorm_eps = v.layer_norm_eps,
            ),
        ]

        for idx in range(v.num_hidden_layers):
            key = f"{key_prefix}.layers.{idx}"
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = LayerNorm(
                        config = config,
                        key = f"{key}.norm1",
                        layernorm_eps = v.layer_norm_eps,
                    ),
                    attn = Attention(
                        config = config,
                        key = f"{key}.attn",
                        layer_idx = idx,
                        hidden_size = v.hidden_size,
                        head_dim = v.head_dim,
                        num_q_heads = v.num_attention_heads,
                        num_kv_heads = v.num_attention_heads,
                        rope_settings = RopeSettings(
                            head_dim = v.head_dim,
                            rope_style = RopeStyle.NEOX,
                        ),
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "proj",
                        qmap = "block.attn",
                        use_cu_seqlens = v.layer_types[idx] == "window_attention",
                    ),
                    mlp_norm = LayerNorm(
                        config = config,
                        key = f"{key}.norm2",
                        layernorm_eps = v.layer_norm_eps,
                    ),
                    mlp = MLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_up = "fc1",
                        key_down = "fc2",
                        activation_fn = "gelu_exact",
                        qmap = "block.mlp",
                        pad_to = 1,
                    ),
                )
            ]

        self.modules += [
            LayerNorm(
                config = config,
                key = f"{key_prefix}.ln_post",
                layernorm_eps = v.layer_norm_eps,
            ),
            MuseGlimmerVisionPixelShuffle(
                config = config,
                key = f"{key_prefix}.pixel_shuffle",
                merge_size = v.merge_size,
            ),
            MuseGlimmerVisionAdapter(
                config = config,
                key = "model.vision_adapter",
                in_size = v.hidden_size * v.merge_size ** 2,
                interm_size = self.config.read_cfg(int, "projector_hidden_size", no_default),
                qmap = "block",
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "model.vision_projection",
                in_features = self.config.read_cfg(int, "projector_hidden_size", no_default),
                out_features = config.hidden_size,
                qmap = "block",
                out_dtype = torch.half,
                pad_to = 1,
            ),
            # Scaleless RMSNorm (reference perception_emb_norm); no tensors to load
            # TODO: Determine if this can be omitted since text model input embeddings are already normalized
            RMSNorm(
                config = config,
                key = "model.vision_projection.perception_emb_norm",
                rms_norm_eps = config.rms_norm_eps,
                unweighted = True,
                out_dtype = torch.half,
            ),
        ]


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids


    def default_load_shape_dtype(self, chunk_size):
        v = self.config.vision
        pp = self.config.vision_pp
        max_patches = pp.max_image_tokens * pp.merge_size ** 2
        return (1, max_patches, v.patch_dim), torch.half


    def default_load_params(self, max_chunk_size):
        pp = self.config.vision_pp
        side = int((pp.max_image_tokens * pp.merge_size ** 2) ** 0.5)
        return self.make_vision_params((1, side, side))


    def make_vision_params(self, grid_thw: tuple) -> dict:
        v = self.config.vision
        t, grid_h, grid_w = grid_thw

        inv_freq = muse_position_embedding_grid_2d(grid_thw, v.head_dim, v.rope_theta)
        window_index, window_cu_seqlens = get_qwen2_window_index(
            [grid_thw],
            v.window_size,
            1,
            v.patch_size,
        )
        window_cu_seqlens = torch.unique_consecutive(torch.tensor(window_cu_seqlens, dtype = torch.int))
        max_seqlen = (window_cu_seqlens[1:] - window_cu_seqlens[:-1]).max().item()
        bilinear_indices, bilinear_weights = muse_bilinear_pos_emb(grid_h, grid_w, v.pos_emb_height)
        if t > 1:
            bilinear_indices = bilinear_indices.repeat(1, t)
            bilinear_weights = bilinear_weights.repeat(1, t)

        return {
            "causal": False,
            "grid_thw": torch.tensor([grid_thw], dtype = torch.int),
            "inv_freq": inv_freq,
            "window_index": window_index,
            "cu_seqlens": window_cu_seqlens,
            "max_seqlen": max_seqlen,
            "bilinear_indices": bilinear_indices,
            "bilinear_weights": bilinear_weights,
        }


    def preprocess(
        self,
        image: Image.Image,
    ) -> (torch.Tensor, tuple, tuple):
        pp = self.config.vision_pp
        resample = Image.Resampling(pp.resample)

        image = convert_to_rgb(image)
        old_size = image.size
        new_size = muse_smart_resize(
            old_size,
            pp.patch_size * pp.merge_size,
            pp.max_image_tokens,
        )
        if old_size != new_size:
            image = image.resize(new_size, resample = resample)

        np_image = np.array(image).astype(np.float32)
        np_image = np_image * pp.rescale_factor
        np_image = normalize_image(np_image, tuple(pp.image_mean), tuple(pp.image_std))
        np_image = np_image.transpose(2, 0, 1)

        patches, grid_h, grid_w = muse_patchify(np_image, pp.patch_size, pp.temporal_patch_size)
        return torch.from_numpy(patches).half(), new_size, (1, grid_h, grid_w)


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image.Image | list[Image.Image],
        text_alias: str | None = None,
    ):
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply a single alias to a list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        pp = self.config.vision_pp
        image_tensor, prep_image_size, grid_thw = self.preprocess(image)
        params = self.make_vision_params(grid_thw)

        embedding_tensor = self.forward(
            image_tensor.unsqueeze(0),
            params = params,
        ).cpu()

        num_emb_tokens = embedding_tensor.shape[1]
        id_start = tokenizer.single_id("<|image_start|>")
        id_end = tokenizer.single_id("<|image_end|>")
        token_string = torch.tensor([[id_start] + [-1] * num_emb_tokens + [id_end]], dtype = torch.long)

        mme = MMEmbedding(
            embeddings = embedding_tensor[0],
            text_alias = text_alias,
            token_string = token_string,
        )

        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_image_size,
            "model_architecture": self.config.architecture,
        })
        return mme
