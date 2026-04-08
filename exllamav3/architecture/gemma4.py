from __future__ import annotations
import json
import os
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from typing_extensions import override
from ..model.config import Config
from ..model.model import Model
from ..modules import (
    Embedding,
    GatedMLP,
    Linear,
    RMSNorm,
    TransformerBlock,
    Attention,
    BlockSparseMLP,
)
from ..modules.arch_specific.gemma4 import (
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
)
from ..modules.attn import prepare_for_attn
from ..tokenizer import MMEmbedding, Tokenizer
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from ..util.file import no_default
from ..util.rope import RopeStyle, RopeSettings
from .mm_processing.common import convert_to_rgb
from .mm_processing.gemma4 import (
    get_aspect_ratio_preserving_size,
    convert_image_to_patches,
)


class Gemma4Config(Config):
    arch_string = "Gemma4ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Gemma4TextModel, "vision": Gemma4VisionModel},
            **kwargs
        )

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", False)

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", no_default)
        self.global_head_dim = self.read_cfg(int, "text_config->global_head_dim", self.head_dim)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", self.num_q_heads)

        # "Don't be evil" ...
        self.num_global_kv_heads = self.read_cfg(int, "text_config->num_global_key_value_heads", self.num_kv_heads)
        self.attention_k_eq_v = self.read_cfg(bool, "text_config->attention_k_eq_v", False)
        self.use_bidirectional_attention = self.read_cfg(str, "text_config->use_bidirectional_attention", None)

        # Layers
        self.layer_types = self.read_cfg(list, "text_config->layer_types", no_default)
        assert len(self.layer_types) == self.num_hidden_layers, \
            "Length of text_config->layer_types key doesn't match number of hidden layers"

        self.sliding_window = self.read_cfg(int, "text_config->sliding_window", -1)
        self.swa_pattern = []
        for layer_type in self.layer_types:
            match layer_type:
                case "sliding_attention":
                    self.swa_pattern.append(self.sliding_window)
                case "full_attention":
                    self.swa_pattern.append(-1)
                case _:
                    raise ValueError(f"Unknown layer type in layer_types: {layer_type}")

        self.assert_cfg(str, "text_config->hidden_activation", "gelu_pytorch_tanh", True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)

        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)
        self.attn_logit_softcapping = self.read_cfg(float, "text_config->attn_logit_softcapping", 0.0)
        self.final_logit_softcapping = self.read_cfg(float, "text_config->final_logit_softcapping", 0.0)

        self.hidden_size_per_layer_input = self.read_cfg(int, "text_config->hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input:
            raise NotImplementedError("Gemma4 per-layer inputs are not implemented yet")

        self.enable_moe_block = self.read_cfg(bool, "text_config->enable_moe_block", False)
        self.num_experts = self.read_cfg(int, "text_config->num_experts", 0)
        self.num_experts_per_tok = self.read_cfg(int, "text_config->top_k_experts", 0)
        self.moe_intermediate_size = self.read_cfg(int, "text_config->moe_intermediate_size", 0)
        if self.enable_moe_block:
            assert self.num_experts > 0, "Gemma4 MoE requires text_config->num_experts"
            assert self.num_experts_per_tok > 0, "Gemma4 MoE requires text_config->top_k_experts"
            assert self.moe_intermediate_size > 0, "Gemma4 MoE requires text_config->moe_intermediate_size"

        self.rope_settings_local = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000.0,
            config_dict = self.read_cfg(dict, "text_config->rope_parameters->sliding_attention", {}),
            override_type = self.read_cfg(
                str,
                "text_config->rope_parameters->sliding_attention->rope_type",
                None,
            )
        )
        self.rope_settings_global = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1000000.0,
            config_dict = self.read_cfg(dict, "text_config->rope_parameters->full_attention", {}),
            override_type = self.read_cfg(
                str,
                "text_config->rope_parameters->full_attention->rope_type",
                None,
            ),
            override_head_dim = self.global_head_dim
        )

        # Vision model settings
        self.image_token_id = self.read_cfg(int, "image_token_id", None)
        self.boi_token_id = self.read_cfg(int, "boi_token_id", None)
        self.eoi_token_id = self.read_cfg(int, "eoi_token_id", None)
        self.vision_soft_tokens_per_image = self.read_cfg(int, "vision_soft_tokens_per_image", no_default)

        self.vision = SimpleNamespace(
            hidden_size = self.read_cfg(int, "vision_config->hidden_size", no_default),
            intermediate_size = self.read_cfg(int, "vision_config->intermediate_size", no_default),
            num_hidden_layers = self.read_cfg(int, "vision_config->num_hidden_layers", no_default),
            num_q_heads = self.read_cfg(int, "vision_config->num_attention_heads", no_default),
            head_dim = self.read_cfg(int, "vision_config->head_dim", no_default),
            patch_size = self.read_cfg(int, "vision_config->patch_size", no_default),
            pooling_kernel_size = self.read_cfg(int, "vision_config->pooling_kernel_size", no_default),
            position_embedding_size = self.read_cfg(int, "vision_config->position_embedding_size", no_default),
            rms_norm_eps = self.read_cfg(float, "vision_config->rms_norm_eps", no_default),
            standardize = self.read_cfg(bool, "vision_config->standardize", False),
            num_channels = 3,
            rope_theta = self.read_cfg(float, "vision_config->rope_theta", 100.0),
        )

        processor_path = os.path.join(self.directory, "processor_config.json")
        with open(processor_path, encoding = "utf8") as f:
            processor_config = json.load(f)
        image_processor = processor_config["image_processor"]

        self.vision_pp = SimpleNamespace(
            do_convert_rgb = image_processor["do_convert_rgb"],
            do_rescale = image_processor["do_rescale"],
            do_normalize = image_processor["do_normalize"],
            image_mean = image_processor["image_mean"],
            image_std = image_processor["image_std"],
            resample = image_processor["resample"],
            rescale_factor = image_processor["rescale_factor"],
            max_soft_tokens = image_processor["max_soft_tokens"],
            patch_size = image_processor["patch_size"],
            pooling_kernel_size = image_processor["pooling_kernel_size"],
        )

        self.vision.num_kv_heads = self.read_cfg(int, "vision_config->num_key_value_heads", self.vision.num_q_heads)
        self.vision.patch_dim = self.vision.num_channels * self.vision.patch_size ** 2
        self.vision.max_patches = self.vision_pp.max_soft_tokens * (self.vision_pp.pooling_kernel_size ** 2)


class Gemma4TextModel(Model):
    config_class = Gemma4Config

    def __init__(
        self,
        config: Gemma4Config,
        key_prefix: str = "model.language_model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        use_moe = config.enable_moe_block

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
                multiplier = torch.tensor(config.hidden_size ** 0.5, dtype=torch.bfloat16).float().item()
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            layer_is_full = config.layer_types[idx] == "full_attention"

            attn = Attention(
                config = config,
                key = f"{key_prefix}.layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.global_head_dim if layer_is_full else config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_global_kv_heads if layer_is_full else config.num_kv_heads,
                rope_settings = config.rope_settings_global if layer_is_full else config.rope_settings_local,
                logit_softcapping = config.attn_logit_softcapping,
                sliding_window = config.swa_pattern[idx],
                use_k_as_v = layer_is_full and config.attention_k_eq_v,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                qmap = "block.attn",
                sm_scale = 1.0,
                q_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                v_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn.v_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    unweighted = True,
                ),
                select_hq_bits = 2 if use_moe else 0,
            )

            mlp = GatedMLP(
                config = config,
                key = f"{key_prefix}.layers.{idx}.mlp",
                hidden_size = config.hidden_size,
                intermediate_size = config.intermediate_size,
                key_up = "up_proj",
                key_gate = "gate_proj",
                key_down = "down_proj",
                qmap = "block.mlp",
                activation_fn = "gelu",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                select_hq_bits = 1 if use_moe else 0,
            )

            if use_moe:
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_gate = "experts.{expert_idx}.gate_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_gate_up_split = "experts.gate_up_proj",
                    key_down_split = "experts.down_proj",
                    key_routing_gate = "router.proj",
                    key_per_expert_scale = "router.per_expert_scale",
                    ftranspose_after_load = False,
                    frange_dim = 1,
                    shared_experts = mlp,
                    alt_residual_channel = True,
                    activation_fn = "gelu",
                    shared_experts_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_feedforward_layernorm_1",
                        rms_norm_eps = config.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                    router_pre_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.router.scale",
                        tensor_weight_suffix = False,
                        rms_norm_eps = config.rms_norm_eps,
                        out_dtype = torch.float,
                        # unweighted = True,
                        constant_scale = config.hidden_size ** -0.5,
                    ),
                    routed_pre_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.pre_feedforward_layernorm_2",
                        rms_norm_eps = config.rms_norm_eps,
                        out_dtype = torch.half,
                    ),
                    routed_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_feedforward_layernorm_2",
                        rms_norm_eps = config.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                    out_dtype = torch.float,
                    qmap = "block.moe",  # Not same H as mlp, due to routed_pre_norm
                )

            block = TransformerBlock(
                config = config,
                key = f"{key_prefix}.layers.{idx}",
                layer_idx = idx,
                key_layer_scalar = "layer_scalar",
                attn_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                attn = attn,
                attn_post_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.pre_feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                mlp = mlp,
                mlp_post_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.post_feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    out_dtype = torch.float,
                ),
            )

            self.modules.append(block)

        self.last_kv_module_idx = len(self.modules) - 1

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
                alt_key = f"{key_prefix}.embed_tokens" if config.tie_word_embeddings else None,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                softcap = config.final_logit_softcapping,
                caps = {"logits_output": True},
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        if config.enable_moe_block:
            self.calibration_all_experts = True

        self.caps.update({
            "supports_tp": False,
            "atomic_mm_prefill": config.use_bidirectional_attention == "vision",
        })


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        _prepare_noncausal_mm_spans(input_ids, params)
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<bos>"
        if system_prompt:
            p += f"<|turn>system\n{system_prompt}<turn|>\n"
        p += f"<|turn>user\n{prompt}<turn|>\n"
        p += "<|turn>model\n"
        return p


def _prepare_noncausal_mm_spans(input_ids, params):
    l = len(input_ids[0])
    if l == 1:
        return

    ids = input_ids[0]
    mask = ids >= FIRST_MM_EMBEDDING_INDEX
    change_points = torch.nonzero(mask[1:] != mask[:-1], as_tuple=True)[0] + 1
    boundaries = torch.cat([torch.tensor([0]), change_points, torch.tensor([l])])
    values = mask[boundaries[:-1]]
    spans = [
        (int(start), int(end), bool(val))
        for start, end, val in zip(boundaries[:-1], boundaries[1:], values)
    ]
    if spans and spans[0][2]:
        assert input_ids.shape[0] == 1, "Gemma4 does not support batched multimodal prefill"
        params["non_causal_spans"] = spans


class Gemma4VisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Gemma4Config) -> dict:
        return (
            config.stc.list_tensors(prefix = "model.vision_tower") |
            config.stc.list_tensors(prefix = "model.embed_vision")
        )

    def __init__(
        self,
        config: Gemma4Config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "supports_tp": False,
        })
        v = self.config.vision

        self.modules += [
            Gemma4VisionPatchEmbedder(
                config = config,
                key = "model.vision_tower.patch_embedder",
                hidden_size = v.hidden_size,
                patch_dim = v.patch_dim,
                position_embedding_size = v.position_embedding_size,
            )
        ]

        for idx in range(v.num_hidden_layers):
            key = f"model.vision_tower.encoder.layers.{idx}"
            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    attn = Attention(
                        config = config,
                        key = f"{key}.self_attn",
                        layer_idx = idx,
                        hidden_size = v.hidden_size,
                        head_dim = v.head_dim,
                        num_q_heads = v.num_q_heads,
                        num_kv_heads = v.num_kv_heads,
                        key_q = "q_proj.linear",
                        key_k = "k_proj.linear",
                        key_v = "v_proj.linear",
                        key_o = "o_proj.linear",
                        sm_scale = 1.0,
                        q_norm = RMSNorm(
                            config = config,
                            key = f"{key}.self_attn.q_norm",
                            rms_norm_eps = v.rms_norm_eps,
                        ),
                        k_norm = RMSNorm(
                            config = config,
                            key = f"{key}.self_attn.q_norm",
                            rms_norm_eps = v.rms_norm_eps,
                        ),
                        v_norm = RMSNorm(
                            config = config,
                            key = f"{key}.self_attn.v_norm",
                            rms_norm_eps = v.rms_norm_eps,
                            unweighted = True,
                        ),
                        rope_settings = RopeSettings(
                            head_dim = v.head_dim // 2,
                            rope_theta = v.rope_theta,
                            rotate_dims = 2,
                        ),
                    ),
                    attn_post_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.pre_feedforward_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_up = "up_proj.linear",
                        key_gate = "gate_proj.linear",
                        key_down = "down_proj.linear",
                        activation_fn = "gelu",
                    ),
                    mlp_post_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_feedforward_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                )
            )

        self.modules += [
            Gemma4VisionPooler(
                config = config,
                key = "model.vision_tower",
                hidden_size = v.hidden_size,
                key_std_bias = f"std_bias" if v.standardize else None,
                key_std_scale = f"std_scale" if v.standardize else None,
            ),

            RMSNorm(
                config = config,
                key = "model.embed_vision.embedding_norm",
                rms_norm_eps = config.rms_norm_eps,
                constant_bias = 1.0,
                out_dtype = torch.half,
                unweighted = True,
            ),
            Linear(
                config = config,
                key = "model.embed_vision.embedding_projection",
                in_features = config.vision.hidden_size,
                out_features = config.hidden_size,
            )
        ]


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids


    def default_load_shape_dtype(self, chunk_size):
        return (1, self.config.vision.max_patches, self.config.vision.patch_dim,), torch.half


    def default_load_params(self, chunk_size):
        h_patches = 45
        w_patches = self.config.vision.max_patches // h_patches
        grid_x, grid_y = np.meshgrid(np.arange(w_patches), np.arange(h_patches), indexing = "xy")
        position_ids = np.stack([grid_x, grid_y], axis = -1).reshape(self.config.vision.max_patches, 2)
        return {
            "input_ids": torch.zeros((1, self.config.vision.max_patches, self.config.vision.patch_dim), dtype = torch.half),
            "position_ids": torch.from_numpy(position_ids).int().unsqueeze(0),
            "image_output_length": self.config.vision_pp.max_soft_tokens,
            "causal": False,
        }


    def preprocess(
        self,
        image: Image.Image,
    ):
        vpp = self.config.vision_pp

        image = convert_to_rgb(image)
        old_size = image.size
        new_size = get_aspect_ratio_preserving_size(
            size = old_size,
            patch_size = vpp.patch_size,
            max_patches = self.config.vision.max_patches,
            pooling_kernel_size = vpp.pooling_kernel_size,
        )

        if new_size != old_size:
            image = image.resize(new_size, resample = Image.Resampling(vpp.resample))

        image_np = np.array(image).astype(np.float32)
        image_np = image_np.transpose(2, 0, 1)

        if vpp.do_rescale:
            image_np *= vpp.rescale_factor

        if vpp.do_normalize:
            image_mean = np.asarray(vpp.image_mean, dtype = np.float32).reshape(-1, 1, 1)
            image_std = np.asarray(vpp.image_std, dtype = np.float32).reshape(-1, 1, 1)
            image_np = (image_np - image_mean) / image_std

        patches = convert_image_to_patches(image_np, vpp.patch_size)
        num_soft_tokens = patches.shape[0] // (vpp.pooling_kernel_size ** 2)

        patch_width = image_np.shape[-1] // vpp.patch_size
        patch_height = image_np.shape[-2] // vpp.patch_size
        grid_x, grid_y = np.meshgrid(np.arange(patch_width), np.arange(patch_height), indexing = "xy")
        positions = np.stack([grid_x, grid_y], axis = -1).reshape(patches.shape[0], 2)

        pixel_values = torch.from_numpy(patches).half().unsqueeze(0)
        image_position_ids = torch.from_numpy(positions).int().unsqueeze(0)
        return pixel_values, image_position_ids, num_soft_tokens, new_size


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image.Image | list[Image.Image],
        text_alias: str | None = None,
    ):
        cfg = self.config

        if isinstance(image, list):
            assert text_alias is None, "Cannot apply a single alias to a list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        pixel_values, image_position_ids, num_soft_tokens, prep_size = self.preprocess(image)

        params = {
            "causal": False,
            "position_ids": image_position_ids,
            "image_output_length": num_soft_tokens,
        }

        embedding_tensor = self.forward(
            pixel_values,
            params = params,
        ).cpu()

        token_string = torch.tensor(
            [[cfg.boi_token_id] + [-1] * num_soft_tokens + [cfg.eoi_token_id]],
            dtype = torch.long,
        )

        mme = MMEmbedding(
            embeddings = embedding_tensor[0],
            text_alias = text_alias,
            token_string = token_string,
        )

        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_size,
            "model_architecture": cfg.architecture,
        })
        return mme
