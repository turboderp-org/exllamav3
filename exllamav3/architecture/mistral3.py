from __future__ import annotations
from typing_extensions import override
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..model.model import Model
from ..util.rope import RopeSettings, RopeStyle
from ..util.file import read_dict, no_value, no_default
from ..util.vision import size_to_longest_edge_and_patch_size, convert_to_rgb, normalize_image
from ..util.tensor import to2
from ..modules import (
    Module,
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    GatedMLP,
    Linear,
    Conv,
    MLP,
    LayerNorm,
)
from ..modules.attn import prepare_for_attn
from ..tokenizer import Tokenizer, MMEmbedding
from types import SimpleNamespace
from PIL import Image

class Mistral3Config(Config):
    arch_string = "Mistral3ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Mistral3Model, "vision": Mistral3VisionModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", None)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "text_config->hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", False)

        # RoPE
        text_cfg = self.read_cfg(dict, "text_config", no_default)
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX, config_dict = text_cfg)

        # Vision model settings
        def unpack_patch_size(patch_temp: dict | int):
            if isinstance(patch_temp, dict):
                h, w = (patch_temp.get(x) for x in ["height", "width"])
                assert w == h, f"Pixtral image preprocessor requires square patches, not {h} x {w}"
                patch_temp = h
            assert isinstance(patch_temp, int), "Unexpected type for patch_size"
            return patch_temp

        self.vision = SimpleNamespace()
        self.vision.head_dim = self.read_cfg(int, ["vision_config->head_dim"], no_default)
        self.vision.num_q_heads = self.read_cfg(int, ["vision_config->num_attention_heads"], no_default)
        self.vision.num_kv_heads = self.read_cfg(int, ["vision_config->num_key_value_heads"], self.vision.num_q_heads)
        self.vision.multimodal_projector_bias = self.read_cfg(bool, ["multimodal_projector_bias"], False)
        self.vision.hidden_size = self.read_cfg(int, ["vision_config->hidden_size"], no_default)
        self.vision.patch_size = unpack_patch_size(self.read_cfg(object, ["vision_config->patch_size"], int(14)))
        self.vision.num_hidden_layers = self.read_cfg(int, ["vision_config->num_hidden_layers"], 24)
        self.vision.intermediate_size = self.read_cfg(int, ["vision_config->intermediate_size"], no_default)
        self.vision.merger_intermediate_size = self.vision.intermediate_size
        self.vision.rms_norm_eps = self.rms_norm_eps
        self.vision.image_size = self.read_cfg(int, ["vision_config->image_size"], 1540)
        self.vision.spatial_merge_size = self.read_cfg(int, ["spatial_merge_size"], 1)
        self.vision.rope_theta = self.read_cfg(int, ["vision_config->rope_theta"], 10000.0)

        vision_cfg = self.read_cfg(dict, "vision_config", no_default)
        self.vision.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX, config_dict = vision_cfg)

        self.vision.num_channels = 3
        self.vision.feature_layer = -1
        self.assert_cfg(int, "vision_config->num_channels", self.vision.num_channels, True)
        self.assert_cfg(int, "vision_feature_layer", self.vision.feature_layer, True)

        # Vision preprocessor
        prep_path = os.path.join(self.directory, "preprocessor_config.json")
        with open(prep_path, encoding = "utf8") as f:
                read_prep_config = json.load(f)
        image_processor_type = read_dict(read_prep_config, str, ["image_processor_type"], no_default)
        assert image_processor_type in ["PixtralImageProcessor", "PixtralImageProcessorFast"], \
            f"Wrong image processor type: {image_processor_type}"
        self.vision_pp = SimpleNamespace()
        self.vision_pp.image_mean = read_dict(read_prep_config, list, ["image_mean"], no_default)
        self.vision_pp.image_std = read_dict(read_prep_config, list, ["image_std"], no_default)
        self.vision_pp.resample = read_dict(read_prep_config, int, ["resample"], no_default)
        self.vision_pp.rescale_factor = read_dict(read_prep_config, float, ["rescale_factor"], no_default)
        self.vision_pp.size = read_dict(read_prep_config, dict, ["size"], no_default)
        self.vision_pp.patch_size = unpack_patch_size(read_dict(read_prep_config, object, ["patch_size"], no_default))

        assert self.vision.patch_size == self.vision_pp.patch_size, \
            "Vision model and vision preprocessor patch sizes do not match"


class Mistral3Model(Model):
    config_class = Mistral3Config

    def __init__(
        self,
        config: Mistral3Config,
        key_prefix = "language_model.",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = key_prefix + "model.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        self.modules += [
            TransformerBlock(
                config = config,
                key = key_prefix + f"model.layers.{idx}",
                attn_norm = RMSNorm(
                    config = config,
                    key = key_prefix + f"model.layers.{idx}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                attn = Attention(
                    config = config,
                    key = key_prefix + f"model.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_kv_heads,
                    rope_settings = config.rope_settings,
                    sm_scale = None,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = key_prefix + f"model.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                mlp = GatedMLP(
                    config = config,
                    key = key_prefix + f"model.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.mlp",
                    # interm_dtype = torch.float,
                    out_dtype = torch.float,
                ),
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor(key_prefix + "lm_head"):
            head_alt_key = key_prefix + "model.embed_tokens"

        self.modules += [
            RMSNorm(
                config = config,
                key = key_prefix + "model.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = key_prefix + "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1


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


class Mistral3PatchMerger(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        merge: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Mistral3PatchMerger"
        self.qmap = qmap

        self.merge = merge
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype

        self.merging_layer = Linear(
            config = config,
            key = f"{key}.merging_layer",
            in_features = hidden_size * merge ** 2,
            out_features = hidden_size
        )

        self.register_submodule(self.merging_layer)

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        bsz, seq_len, dim = x.shape
        h, w = params["features_size"]
        assert bsz == 1

        x = x.view(h, w, dim).permute(2, 0, 1).unsqueeze(0)
        x = F.unfold(x, kernel_size = self.merge, stride = self.merge)
        x = x.view(bsz, dim * self.merge ** 2, -1).transpose(1, 2).contiguous()
        x = self.merging_layer.forward(x, params)

        return to2(x, out_dtype, self.out_dtype)


class Mistral3VisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Mistral3Config) -> dict:
        vlm_tensors = config.stc.list_tensors(prefix = "vision_tower")
        mmp_tensors = config.stc.list_tensors(prefix = "multi_modal_projector")
        return vlm_tensors | mmp_tensors

    def __init__(
        self,
        config: Mistral3Config,
        key_prefix = "vision_tower.",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({"image_input": True})

        self.modules += [
            Conv(
                config = config,
                key = key_prefix + "patch_conv",
                in_channels = config.vision.num_channels,
                out_channels = config.vision.hidden_size,
                kernel_size = (config.vision.patch_size, config.vision.patch_size),
            ),
             RMSNorm(
                config = config,
                key = key_prefix + f"ln_pre",
                rms_norm_eps = config.rms_norm_eps,
            )
        ]

        self.modules += [
            TransformerBlock(
                config = config,
                key = key_prefix + f"transformer.layers.{idx}",
                attn_norm = RMSNorm(
                    config = config,
                    key = key_prefix + f"transformer.layers.{idx}.attention_norm",
                    rms_norm_eps = config.vision.rms_norm_eps
                ),
                attn = Attention(
                    config = config,
                    key = key_prefix + f"transformer.layers.{idx}.attention",
                    layer_idx = idx,
                    hidden_size = config.vision.hidden_size,
                    head_dim = config.vision.head_dim,
                    num_q_heads = config.vision.num_q_heads,
                    num_kv_heads = config.vision.num_kv_heads,
                    rope_settings = RopeSettings(
                        head_dim = config.vision.head_dim,
                        rope_theta = config.vision.rope_theta,
                    ),
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn"
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = key_prefix + f"transformer.layers.{idx}.ffn_norm",
                    rms_norm_eps = config.vision.rms_norm_eps
                ),
                mlp = GatedMLP(
                    config = config,
                    key = key_prefix + f"transformer.layers.{idx}.feed_forward",
                    hidden_size = config.vision.hidden_size,
                    intermediate_size = config.vision.intermediate_size,
                    key_gate = "gate_proj",
                    key_up = "up_proj",
                    key_down = "down_proj",
                    activation_fn = "silu",
                    qmap = "block.mlp",
                    pad_to = 1,
                ),
            )
            for idx in range(config.vision.num_hidden_layers)
        ]

        self.modules += [
            RMSNorm(
                config = config,
                key = f"multi_modal_projector.norm",
                rms_norm_eps = config.vision.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Mistral3PatchMerger(
                config = config,
                key = "multi_modal_projector.patch_merger",
                hidden_size = config.vision.hidden_size,
                merge = config.vision.spatial_merge_size,
                out_dtype = torch.half,
            ),
            MLP(
                config = config,
                key = "multi_modal_projector",
                key_up = "linear_1",
                key_down = "linear_2",
                hidden_size = config.vision.hidden_size,
                intermediate_size = 5120,  # This seems to be hard coded?
                out_size = config.hidden_size,
                activation_fn = "gelu",
                qmap = "block",
            )
        ]

        # Precomputed RoPE table following Transformers implementation
        self.max_edge_features = config.vision_pp.size["longest_edge"] // config.vision_pp.patch_size
        freqs = 1.0 / (
            config.vision.rope_theta ** (
                torch.arange(0, config.vision.head_dim, 2).float()
                / config.vision.head_dim
            )
        )
        h = torch.arange(self.max_edge_features).float()
        w = torch.arange(self.max_edge_features).float()
        freqs_h = torch.outer(h, freqs[::2])
        freqs_w = torch.outer(w, freqs[1::2])
        self.inv_freq = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, self.max_edge_features, 1),
                freqs_w[None, :, :].repeat(self.max_edge_features, 1, 1),
            ],
            dim = -1,
        ).reshape(-1, config.vision.head_dim // 2)


    def preprocess(
        self,
        image: Image
    ) -> (torch.Tensor, tuple):
        """
        Convert input image to the size and format expected by the vision tower. Image is scaled proportionally to
        fit a bounding box of longest_edge x longest_edge pixels as defined by the preprocessor config, while still
        being divisible into tiles of spatial_merge_size x spatial_merge_size input patches. Each such tile will be
        merged into one multimodal feature token by the vision tower.
        """

        patch_2d = (
            self.config.vision_pp.patch_size * self.config.vision.spatial_merge_size,
            self.config.vision_pp.patch_size * self.config.vision.spatial_merge_size,
        )
        longest_edge = self.config.vision_pp.size["longest_edge"]
        resample = Image.Resampling(self.config.vision_pp.resample)
        image_mean = tuple(self.config.vision_pp.image_mean)
        image_std = tuple(self.config.vision_pp.image_std)
        rescale_factor = self.config.vision_pp.rescale_factor

        # Convert to RGB and resize as necessary
        image = convert_to_rgb(image)
        old_size = image.size
        new_size = size_to_longest_edge_and_patch_size(image.size, (longest_edge, longest_edge), patch_2d)
        if old_size != new_size:
            image = image.resize(new_size, resample = resample)

        # Convert to numpy array and normalize
        image = np.array(image).astype(np.float32)
        image = image * rescale_factor
        image = normalize_image(image, image_mean, image_std)

        # Convert to tensor, shape (1, 3, resized_height, resized_width)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).half().unsqueeze(0)
        return image, new_size


    def default_load_shape_dtype(self, chunk_size):
        return (
            (
                1,
                self.config.vision.num_channels,
                self.config.vision_pp.size["longest_edge"],
                self.config.vision_pp.size["longest_edge"]
            ),
            torch.half
        )


    def default_load_params(self):
        return {
            "features_size": (
                self.config.vision_pp.size["longest_edge"] // self.config.vision_pp.patch_size,
                self.config.vision_pp.size["longest_edge"] // self.config.vision_pp.patch_size,
            )
        }


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image | list[Image],
        text_alias: str | None = None,
    ):
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"

            # Images in Mistral3 have uneven numbers of MM tokens so each image is actually processed at bsz 1
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        image_tensor, prep_image_size = self.preprocess(image)
        features_size = (
            prep_image_size[1] // self.config.vision_pp.patch_size,
            prep_image_size[0] // self.config.vision_pp.patch_size,
        )

        # Flattened position ID grid matching inv_freq table
        h, w = features_size
        row_indices = torch.arange(h, dtype = torch.int).unsqueeze(1) * self.max_edge_features
        col_indices = torch.arange(w, dtype = torch.int).unsqueeze(0)
        position_ids_grid = (row_indices + col_indices).flatten().unsqueeze(0)

        embedding_tensor = self.forward(
            image_tensor,
            params = {
                "causal": False,
                "features_size": features_size,
                "inv_freq": self.inv_freq,
                "position_ids": position_ids_grid,
            }
        ).cpu().squeeze(0)

        w //= self.config.vision.spatial_merge_size
        h //= self.config.vision.spatial_merge_size
        id_break = tokenizer.single_id("[IMG_BREAK]")
        id_end = tokenizer.single_id("[IMG_END]")
        token_string = torch.tensor([([-1] * w + [id_break]) * h + [id_end]] , dtype = torch.long)

        mme = MMEmbedding(
            embeddings = embedding_tensor,
            text_alias = text_alias,
            token_string = token_string
        )

        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_image_size,
            "model_architecture": self.config.architecture,
        })

        return mme


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids
