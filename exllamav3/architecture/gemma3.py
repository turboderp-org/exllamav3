from __future__ import annotations
from typing_extensions import override
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..model.model import Model
from ..util.rope import RopeStyle
from ..util.file import read_dict, no_value, no_default
from ..util.vision import convert_to_rgb, normalize_image
from ..modules import (
    Module,
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    GatedMLP,
    Linear,
    Conv,
    PosEmbedding,
    MLP,
    LayerNorm
)
from ..modules.attn import prepare_for_attn
from ..tokenizer import Tokenizer, MMEmbedding
from types import SimpleNamespace
from PIL import Image

class Gemma3Config(Config):
    arch_string = "Gemma3ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Gemma3Model, "vision": Gemma3VisionModel},
            **kwargs
        )

        # Gemma3 quirk, vocab size is implicit on HF versions
        if self.vocab_size is None:
            self.vocab_size = 262208

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", True)

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", 256)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", 2304)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", 8)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", 4)

        self.query_pre_attn_scalar = self.read_cfg(float, "text_config->query_pre_attn_scalar", 256)
        self.attn_logit_softcapping = self.read_cfg(float, "text_config->attn_logit_softcapping", 0.0)

        self.sliding_window = self.read_cfg(int, "text_config->sliding_window", 4096)
        layer_types = self.read_cfg(list, "layer_types", None)
        sliding_window_pattern = self.read_cfg(int, "sliding_window_pattern", 6)

        if layer_types:
            assert len(layer_types) == self.num_hidden_layers, \
                "Length of layer_types key doesn't match number of hidden layers"
            self.swa_pattern = []
            for t in layer_types:
                if t == "sliding_attention":
                    self.swa_pattern.append(self.sliding_window)
                elif t == "full_attention":
                    self.swa_pattern.append(-1)
                else:
                    raise ValueError("Unknown layer type in layer_types")

        elif sliding_window_pattern:
            self.swa_pattern = [
                self.sliding_window if (idx + 1) % sliding_window_pattern != 0 else -1
                for idx in range(self.num_hidden_layers)
            ]

        else:
            self.swa_pattern = [-1 for _ in range(self.num_hidden_layers)]

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", 9216)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", 1e-6)

        # RoPE
        self.rope_settings_global = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1e6,
            config_dict = self.read_cfg(dict, "text_config", no_default)
        )
        self.rope_settings_local = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1e4,
            config_dict = {}
        )

        # Output softcap
        self.final_logit_softcapping = self.read_cfg(float, "text_config->final_logit_softcapping", 0.0)

        # Vision model settings
        self.vision = SimpleNamespace()
        self.vision.num_q_heads = self.read_cfg(int, ["vision_config->num_attention_heads"], no_default)
        self.vision.num_kv_heads = self.read_cfg(int, ["vision_config->num_key_value_heads"], self.vision.num_q_heads)
        self.vision.hidden_size = self.read_cfg(int, ["vision_config->hidden_size"], no_default)
        def_head_dim = self.vision.hidden_size // self.vision.num_q_heads
        self.vision.head_dim = self.read_cfg(int, ["vision_config->head_dim"], def_head_dim)
        self.vision.multimodal_projector_bias = self.read_cfg(bool, ["multimodal_projector_bias"], False)
        self.vision.patch_size = self.read_cfg(int, ["vision_config->patch_size"], no_default)
        self.vision.num_hidden_layers = self.read_cfg(int, ["vision_config->num_hidden_layers"], 24)
        self.vision.intermediate_size = self.read_cfg(int, ["vision_config->intermediate_size"], self.vision.hidden_size)
        self.vision.mm_tokens_per_image = self.read_cfg(int, "mm_tokens_per_image", no_default)
        self.vision.num_channels = 3
        self.vision.layernorm_eps = 1e-6  # Siglip default
        self.vision.image_size = self.read_cfg(int, ["vision_config->image_size"], 896)

        # Vision preprocessor
        prep_path = os.path.join(self.directory, "preprocessor_config.json")
        with open(prep_path, encoding = "utf8") as f:
                read_prep_config = json.load(f)
        image_processor_type = read_dict(read_prep_config, str, ["image_processor_type"], no_default)
        assert image_processor_type == "Gemma3ImageProcessor", \
            f"Wrong image processor type: {image_processor_type}"
        self.vision_pp = SimpleNamespace()
        self.vision_pp.image_mean = read_dict(read_prep_config, list, ["image_mean"], no_default)
        self.vision_pp.image_std = read_dict(read_prep_config, list, ["image_std"], no_default)
        self.vision_pp.resample = read_dict(read_prep_config, int, ["resample"], no_default)
        self.vision_pp.rescale_factor = read_dict(read_prep_config, float, ["rescale_factor"], no_default)
        self.vision_pp.size = read_dict(read_prep_config, dict, ["size"], no_default)


    def default_max_position_embeddings(self):
        # Fixed for Gemma3, usually not present in config.json
        return 131072


class Gemma3TextConfig(Config):
    arch_string = "Gemma3ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Gemma3TextModel},
            **kwargs
        )

        # Gemma3 quirk, vocab size is implicit on HF versions
        if self.vocab_size is None:
            self.vocab_size = 262208

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", True)

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", 256)
        self.hidden_size = self.read_cfg(int, "hidden_size", 2304)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", 8)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", 4)

        self.query_pre_attn_scalar = self.read_cfg(float, "query_pre_attn_scalar", 256)
        self.attn_logit_softcapping = self.read_cfg(float, "attn_logit_softcapping", 0.0)

        self.sliding_window = self.read_cfg(int, "sliding_window", 4096)
        layer_types = self.read_cfg(list, "layer_types", None)
        sliding_window_pattern = self.read_cfg(int, "sliding_window_pattern", 6)

        if layer_types:
            assert len(layer_types) == self.num_hidden_layers, \
                "Length of layer_types key doesn't match number of hidden layers"
            self.swa_pattern = []
            for t in layer_types:
                if t == "sliding_attention":
                    self.swa_pattern.append(self.sliding_window)
                elif t == "full_attention":
                    self.swa_pattern.append(-1)
                else:
                    raise ValueError("Unknown layer type in layer_types")

        elif sliding_window_pattern:
            self.swa_pattern = [
                self.sliding_window if (idx + 1) % sliding_window_pattern != 0 else -1
                for idx in range(self.num_hidden_layers)
            ]

        else:
            self.swa_pattern = [-1 for _ in range(self.num_hidden_layers)]

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.intermediate_size = self.read_cfg(int, "intermediate_size", 9216)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", 1e-6)

        # RoPE
        self.rope_settings_global = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1e6,
        )
        self.rope_settings_local = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1e4,
            config_dict = {}
        )

        # Output softcap
        self.final_logit_softcapping = self.read_cfg(float, "final_logit_softcapping", 0.0)


    def default_max_position_embeddings(self):
        # Fixed for Gemma2, usually not present in config.json
        return 8192


class Gemma3Model(Model):
    config_class = Gemma3Config

    def __init__(
        self,
        config: Gemma3Config | Gemma3TextConfig,
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
                normalize = True,
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
                    constant_bias = 1.0,
                ),
                attn = Attention(
                    config = config,
                    key = key_prefix + f"model.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_kv_heads,
                    rope_settings = config.rope_settings_global if config.swa_pattern[idx] == -1 else config.rope_settings_local,
                    sm_scale = config.query_pre_attn_scalar ** (-0.5),
                    logit_softcapping = config.attn_logit_softcapping,
                    sliding_window = config.swa_pattern[idx],
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    q_norm = RMSNorm(
                        config = config,
                        key = key_prefix + f"model.layers.{idx}.self_attn.q_norm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    k_norm = RMSNorm(
                        config = config,
                        key = key_prefix + f"model.layers.{idx}.self_attn.k_norm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                ),
                attn_post_norm = RMSNorm(
                    config = config,
                    key = key_prefix + f"model.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = key_prefix + f"model.layers.{idx}.pre_feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
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
                    activation_fn = "gelu"
                ),
                mlp_post_norm = RMSNorm(
                    config = config,
                    key = key_prefix + f"model.layers.{idx}.post_feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                    out_dtype = torch.float,
                ),
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            RMSNorm(
                config = config,
                key = key_prefix + "model.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
                constant_bias = 1.0,
            ),
            Linear(
                config = config,
                key = key_prefix + "lm_head",
                qbits_key = "head_bits",
                alt_key = key_prefix + "model.embed_tokens",
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                softcap = config.final_logit_softcapping,
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
        p = "<bos><start_of_turn>user\n"
        if system_prompt:
            p += "{system_prompt}\n\n"
        p += f"{prompt}\n"
        p += f"<start_of_turn>model\n"
        return p


class Gemma3TextModel(Gemma3Model):
    config_class = Gemma3TextConfig

    def __init__(
        self,
        config: Gemma3TextConfig,
        **kwargs
    ):
        super().__init__(config, key_prefix = "", **kwargs)


class Gemma3MMPool(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        patches_per_image: int,
        tokens_per_side: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Gemma3MMPool"
        self.qmap = qmap

        self.patches_per_image = patches_per_image
        self.tokens_per_side = tokens_per_side

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device: torch.device, **kwargs):
        pass

    @override
    def unload(self):
        pass

    @override
    def weights_numel(self):
        return 0

    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        bsz, _, seq_length = x.shape
        x = (
            x.transpose(1, 2)
            .view(bsz, seq_length, self.patches_per_image, self.patches_per_image)
        )
        kernel_size = self.patches_per_image // self.tokens_per_side
        x = (
            F.avg_pool2d(x, kernel_size = kernel_size, stride = kernel_size)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )
        return x


class Gemma3VisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Gemma3Config) -> dict:
        vlm_tensors = config.stc.list_tensors(prefix = "vision_tower")
        mmp_tensors = config.stc.list_tensors(prefix = "multi_modal_projector")
        return vlm_tensors | mmp_tensors

    def __init__(
        self,
        config: Gemma3Config,
        key_prefix = "vision_tower.",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "fixed_size_image_embeddings": True
        })

        self.modules += [
            Conv(
                config = config,
                key = key_prefix + "vision_model.embeddings.patch_embedding",
                in_channels = config.vision.num_channels,
                out_channels = config.vision.hidden_size,
                kernel_size = (config.vision.patch_size, config.vision.patch_size),
            ),
            PosEmbedding(
                config = config,
                key = key_prefix + "vision_model.embeddings.position_embedding",
                hidden_size = config.vision.hidden_size,
            )
        ]

        self.modules += [
            TransformerBlock(
                config = config,
                key = key_prefix + f"vision_model.encoder.layers.{idx}",
                attn_norm = LayerNorm(
                    config = config,
                    key = key_prefix + f"vision_model.encoder.layers.{idx}.layer_norm1",
                    layernorm_eps = config.vision.layernorm_eps
                ),
                attn = Attention(
                    config = config,
                    key = key_prefix + f"vision_model.encoder.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.vision.hidden_size,
                    head_dim = config.vision.head_dim,
                    num_q_heads = config.vision.num_q_heads,
                    num_kv_heads = config.vision.num_kv_heads,
                    rope_settings = None,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "out_proj",
                    qmap = "block.attn"
                ),
                mlp_norm = LayerNorm(
                    config = config,
                    key = key_prefix + f"vision_model.encoder.layers.{idx}.layer_norm2",
                    layernorm_eps = config.vision.layernorm_eps
                ),
                mlp = MLP(
                    config = config,
                    key = key_prefix + f"vision_model.encoder.layers.{idx}.mlp",
                    hidden_size = config.vision.hidden_size,
                    intermediate_size = config.vision.intermediate_size,
                    key_up = "fc1",
                    key_down = "fc2",
                    activation_fn = "gelu",
                    qmap = "block.mlp",
                    pad_to = 1,
                ),
            )
            for idx in range(config.vision.num_hidden_layers)
        ]

        self.modules += [
            LayerNorm(
                config = config,
                key = key_prefix + f"vision_model.post_layernorm",
                layernorm_eps = config.vision.layernorm_eps
            ),
            Gemma3MMPool(
                config = config,
                key = key_prefix + f"vision_model.mm_pool",
                patches_per_image = int(config.vision.image_size // config.vision.patch_size),
                tokens_per_side = int(config.vision.mm_tokens_per_image ** 0.5)
            ),
            RMSNorm(
                config = config,
                key = "multi_modal_projector.mm_soft_emb_norm",
                rms_norm_eps = config.rms_norm_eps,
                constant_bias = 1.0,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "multi_modal_projector.mm_input_projection",
                in_features = config.vision.hidden_size,
                out_features = config.hidden_size,
                transposed_load = False,
                # qmap = "block",
            )
        ]


    def preprocess(
        self,
        image: Image
    ) -> (torch.Tensor, tuple):
        """
        Convert input image to the standard size and format expected by the Siglip vision tower
        """

        size = tuple(self.config.vision_pp.size[d] for d in ["height", "width"])

        resample = Image.Resampling(self.config.vision_pp.resample)
        image_mean = tuple(self.config.vision_pp.image_mean)
        image_std = tuple(self.config.vision_pp.image_std)
        rescale_factor = self.config.vision_pp.rescale_factor

        # Convert to RGB and resize
        image = convert_to_rgb(image)
        old_size = image.size
        new_size = size
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
                self.config.vision_pp.size["height"],
                self.config.vision_pp.size["width"]
            ),
            torch.half
        )


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image | list[Image],
        text_alias: str | None = None,
    ):
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"
            image_tensor = []
            for i in image:
                t, prep_image_size = self.preprocess(i)
                image_tensor.append(t)
            image_tensor = torch.cat(image_tensor, dim = 0)
            return_batch = True
        else:
            image_tensor, prep_image_size = self.preprocess(image)
            image = [image]
            return_batch = False

        embedding_tensor = self.forward(
            image_tensor,
            params = {"causal": False}
        ).cpu()

        num_emb_tokens = embedding_tensor.shape[1]
        mmes = []
        for i in range(embedding_tensor.shape[0]):
            id_start = tokenizer.single_id("<start_of_image>")
            id_end = tokenizer.single_id("<end_of_image>")
            token_string = torch.tensor([[id_start] + [-1] * num_emb_tokens + [id_end]], dtype = torch.long)

            mme = MMEmbedding(
                embeddings = embedding_tensor[i],
                text_alias = text_alias,
                token_string = token_string
            )

            mme.metadata.update({
                "original_size": image[i].size,
                "preprocessed_size": prep_image_size,
                "model_architecture": self.config.architecture,
            })

            mmes.append(mme)

        return mmes if return_batch else mmes[0]


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids