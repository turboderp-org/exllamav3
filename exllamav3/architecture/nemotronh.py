from __future__ import annotations
from typing_extensions import override
from types import SimpleNamespace
import json
import os
import torch
from PIL import Image
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.file import read_dict
from ..tokenizer import MMEmbedding, Tokenizer
from ..modules.arch_specific.nemotronh_omni import NemotronHOmniPatchEmbedder
from .mm_processing.nemotronh_omni import (
    nemotronh_omni_image_grid,
    nemotronh_omni_video_grid,
    nemotronh_omni_image_to_tensor,
    nemotronh_omni_patchify,
    nemotronh_omni_frame_labels,
    nemotronh_omni_video_retention_mask,
)
from ..modules import (
    RMSNorm,
    LayerNorm,
    PixelUnshuffle,
    Embedding,
    TransformerBlock,
    Attention,
    MLP,
    BlockSparseMLP,
    Linear,
    Mamba2,
    GDNState,
)
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from .nemotronh_mtp import NemotronHMTPModel

class NemotronHConfig(Config):
    arch_string = "NemotronHForCausalLM"
    config_key = None
    tensor_prefix = ""

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": NemotronHModel, "mtp": NemotronHMTPModel},
            config_key = self.config_key,
            **kwargs
        )

        # Layer pattern: M = Mamba2, * = attention, - = MLP, E = MoE
        block_configs = self.read_cfg(list, "block_configs", None)
        if block_configs is not None:
            block_types = self.read_cfg(list, "layers_block_type", None)
            if block_types is None:
                block_types = [layer.get("block_type") for layer in block_configs]
            layer_types = {"mamba": "M", "attention": "*", "mlp": "-", "moe": "E"}
            try:
                self.hybrid_override_pattern = "".join(layer_types[t] for t in block_types)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Unsupported NemotronH block type in {block_types!r}") from exc
            if len(block_types) != len(block_configs):
                raise ValueError("NemotronH block_configs and layers_block_type lengths differ")
            self.layer_moe_configs = block_configs
        else:
            self.hybrid_override_pattern = self.read_cfg(str, "hybrid_override_pattern", no_default)
            self.layer_moe_configs = None

        # Attention params (NoPE: the reference model applies no positional embeddings)
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Mamba2 params
        self.mamba_num_heads = self.read_cfg(int, "mamba_num_heads", no_default)
        self.mamba_head_dim = self.read_cfg(int, "mamba_head_dim", no_default)
        self.ssm_state_size = self.read_cfg(int, "ssm_state_size", no_default)
        self.n_groups = self.read_cfg(int, "n_groups", no_default)
        self.conv_kernel = self.read_cfg(int, "conv_kernel", 4)
        # Non-finite limits are serialized as {"__float__": "Infinity"} by newer Transformers
        self.time_step_limit = [
            float(v["__float__"]) if isinstance(v, dict) else v
            for v in self.read_cfg(list, "time_step_limit", [0.0, float("inf")])
        ]
        self.assert_cfg(str, "mamba_hidden_act", "silu", True)

        # MLP params
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.assert_cfg(str, "mlp_hidden_act", "relu2", True)

        # MoE params (30B-A3B, 120B-A12B): DeepSeek-style sigmoid router with correction bias,
        # non-gated relu2 experts plus one always-on shared expert
        self.num_experts = self.read_cfg(int, ["n_routed_experts", "num_experts"], 0)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", 0)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", 0)
        self.shared_expert_intermediate_size = self.read_cfg(int, "moe_shared_expert_intermediate_size", 0)
        # Latent MoE (Nemotron-3 Super): routed experts run at moe_latent_size behind a pair of
        # projections (fc1_latent_proj / fc2_latent_proj); the shared expert stays full width
        self.moe_latent_size = self.read_cfg(int, "moe_latent_size", None)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 2.5)
        if self.num_experts:
            self.assert_cfg(int, "n_group", 1, True)
            self.assert_cfg(int, "topk_group", 1, True)
            self.assert_cfg(bool, "norm_topk_prob", True, True)

        # Norms
        self.layer_norm_epsilon = self.read_cfg(float, "layer_norm_epsilon", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", len(self.hybrid_override_pattern))
        assert len(self.hybrid_override_pattern) == self.num_hidden_layers, \
            "hybrid_override_pattern length does not match num_hidden_layers"
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # MTP head (Nemotron-3 Super): one DeepSeek-style step (enorm/hnorm/eh_proj) followed by the
        # blocks of mtp_hybrid_override_pattern ("*E": attention + latent MoE) and a final norm, under
        # the mtp.* namespace. The component only exists when the tensors are present (a quant made
        # before MTP support gets them from util/convert_mtp.py)
        self.num_mtp_layers = self.read_cfg(int, "num_nextn_predict_layers", 0)
        self.mtp_hybrid_override_pattern = self.read_cfg(str, "mtp_hybrid_override_pattern", "")
        if self.num_mtp_layers == 0 or not any(
            self.stc.has_tensor(f"mtp.layers.0.eh_proj.{t}") for t in ("weight", "trellis")
        ):
            del self.model_classes["mtp"]



def nemotronh_block(
    config: NemotronHConfig,
    key: str,
    layer_idx: int,
    block_type: str,
    qbits_key: str = "bits",
) -> TransformerBlock:
    """One block of the hybrid pattern: M = Mamba2, * = attention, - = MLP, E = MoE. Shared by the
    trunk (backbone.layers.N) and the MTP head (mtp.layers.N, mtp_bits)"""
    layer_moe_config = config.layer_moe_configs[layer_idx] if config.layer_moe_configs else {}
    norm = RMSNorm(
        config = config,
        key = f"{key}.norm",
        rms_norm_eps = config.layer_norm_epsilon,
    )
    match block_type:
        case "M":
            return TransformerBlock(
                config = config,
                key = key,
                layer_idx = layer_idx,
                attn_norm = norm,
                attn = Mamba2(
                    config = config,
                    key = f"{key}.mixer",
                    layer_idx = layer_idx,
                    hidden_size = config.hidden_size,
                    num_heads = config.mamba_num_heads,
                    head_dim = config.mamba_head_dim,
                    num_groups = config.n_groups,
                    state_size = config.ssm_state_size,
                    rms_norm_eps = config.layer_norm_epsilon,
                    conv_kernel_size = config.conv_kernel,
                    dt_limit = tuple(config.time_step_limit),
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                ),
            )
        case "*":
            return TransformerBlock(
                config = config,
                key = key,
                layer_idx = layer_idx,
                attn_norm = norm,
                attn = Attention(
                    config = config,
                    key = f"{key}.mixer",
                    layer_idx = layer_idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_kv_heads,
                    rope_settings = None,
                    sm_scale = None,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                    qbits_key = qbits_key,
                ),
            )
        case "-":
            return TransformerBlock(
                config = config,
                key = key,
                layer_idx = layer_idx,
                mlp_norm = norm,
                mlp = MLP(
                    config = config,
                    key = f"{key}.mixer",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_down = "down_proj",
                    activation_fn = "relu2",
                    qmap = "block.mlp",
                    out_dtype = torch.float,
                    select_hq_bits = 1,
                    qbits_key = qbits_key,
                ),
            )
        case "E":
            return TransformerBlock(
                config = config,
                key = key,
                layer_idx = layer_idx,
                mlp_norm = norm,
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key}.mixer",
                    hidden_size = config.hidden_size,
                    intermediate_size = layer_moe_config.get("moe_intermediate_size", config.moe_intermediate_size),
                    num_experts = layer_moe_config.get("n_routed_experts", config.num_experts),
                    num_experts_per_tok = layer_moe_config.get("num_experts_per_tok", config.num_experts_per_tok),
                    latent_size = config.moe_latent_size,
                    key_latent_in = "fc1_latent_proj" if config.moe_latent_size else None,
                    key_latent_out = "fc2_latent_proj" if config.moe_latent_size else None,
                    latent_hq_bits = 2,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_routing_gate = "gate",
                    activation_fn = "relu2",
                    router_type = "dots",
                    routed_scaling_factor = config.routed_scaling_factor,
                    n_group = 1,
                    topk_group = 1,
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    qbits_key = qbits_key,
                    shared_experts = MLP(
                        config = config,
                        key = f"{key}.mixer.shared_experts",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.shared_expert_intermediate_size,
                        key_up = "up_proj",
                        key_down = "down_proj",
                        activation_fn = "relu2",
                        qmap = "block.mlp",
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                        qbits_key = qbits_key,
                    ),
                ),
            )
        case _:
            raise ValueError(f"Unknown layer type {block_type!r} in hybrid_override_pattern")


class NemotronHModel(Model):
    config_class = NemotronHConfig

    def __init__(
        self,
        config: NemotronHConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = f"{config.tensor_prefix}backbone.embeddings",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            self.modules.append(nemotronh_block(
                config, f"{config.tensor_prefix}backbone.layers.{idx}", idx, config.hybrid_override_pattern[idx]
            ))

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor(f"{config.tensor_prefix}lm_head"):
            head_alt_key = f"{config.tensor_prefix}backbone.embeddings"

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{config.tensor_prefix}backbone.norm_f",
                rms_norm_eps = config.layer_norm_epsilon,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = f"{config.tensor_prefix}lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        # Mark that we need recurrent cache for generation
        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True,
        })
        self.recurrent_state_cls = GDNState

        self.caps.update({"supports_tp": True})


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n"
            p += f"{system_prompt}<|im_end|>\n"
        p += f"<|im_start|>user\n"
        p += f"{prompt}<|im_end|>\n"
        p += f"<|im_start|>assistant\n"
        return p


class NemotronHOmniConfig(NemotronHConfig):
    """
    Nemotron 3.5 Omni checkpoints (Bolt): the NemotronH Puzzle text backbone under llm_config, a
    C-RADIO vision tower and the mlp1 projector
    """

    arch_string = "NemotronH_Omni_Reasoning_V3"
    config_key = "llm_config"
    tensor_prefix = "language_model."

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, **kwargs)
        self.model_classes["text"] = NemotronHOmniModel
        self.model_classes["vision"] = NemotronHOmniVisionModel

        self.vision = read_nemotronh_omni_vision_config(self.read_cfg(dict, "vision_config", no_default))
        self.vision.video_temporal_patch_size = self.read_cfg(int, "video_temporal_patch_size", 2)
        self.assert_cfg(bool, "vision_final_layernorm", False, True)

        prep_path = os.path.join(directory, "processor_config.json")
        if not os.path.exists(prep_path):
            prep_path = os.path.join(directory, "preprocessor_config.json")
        with open(prep_path, encoding = "utf8") as f:
            self.vision_pp = read_nemotronh_omni_pp_config(json.load(f))

        self.image_token_id = self.read_cfg(int, "img_context_token_id", None)
        self.downsample_ratio = self.read_cfg(float, "downsample_ratio", 0.5)
        self.projector_hidden_size = self.read_cfg(int, "projector_hidden_size", no_default)
        self.ps_version = self.read_cfg(str, "ps_version", "v2")
        self.video_pruning_rate = self.read_cfg(float, "video_pruning_rate", 0.0)


def read_nemotronh_omni_vision_config(config_dict: dict):
    v = SimpleNamespace(**{
        k: read_dict(config_dict, t, k, no_default)
        for k, t in [
            ("hidden_size", int),
            ("num_hidden_layers", int),
            ("num_attention_heads", int),
            ("patch_size", int),
            ("max_img_size", int),
            ("num_cls_tokens", int),
            ("num_registers", int),
            ("layer_norm_eps", float),
            ("mlp_ratio", float),
        ]
    })
    assert read_dict(config_dict, str, "hidden_act", "gelu") == "gelu", \
        "Expected vision_config->hidden_act to be 'gelu'"
    v.num_channels = read_dict(config_dict, int, "num_channels", 3)
    v.head_dim = v.hidden_size // v.num_attention_heads
    v.intermediate_size = int(v.hidden_size * v.mlp_ratio)
    v.patch_dim = v.num_channels * v.patch_size ** 2
    v.num_prefix_tokens = v.num_cls_tokens + v.num_registers
    v.pos_emb_side = v.max_img_size // v.patch_size
    return v


def read_nemotronh_omni_pp_config(config_dict: dict):
    ip = read_dict(config_dict, dict, "image_processor", config_dict)
    pp = SimpleNamespace(**{
        k: read_dict(ip, t, k, no_default)
        for k, t in [
            ("norm_mean", list),
            ("norm_std", list),
            ("patch_size", int),
            ("min_num_patches", int),
            ("max_num_patches", int),
            ("video_target_num_patches", int),
        ]
    })
    pp.max_model_len = read_dict(ip, int, "max_model_len", 16384)
    pp.video_maintain_aspect_ratio = read_dict(ip, bool, "video_maintain_aspect_ratio", True)
    return pp


class NemotronHOmniModel(NemotronHModel):
    config_class = NemotronHOmniConfig


class NemotronHOmniVisionModel(Model):
    """C-RADIO vision tower and the mlp1 projector, for image and video input"""

    config_class = NemotronHOmniConfig

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: NemotronHOmniConfig) -> dict:
        return (
            config.stc.list_tensors(prefix = "vision_model.") |
            config.stc.list_tensors(prefix = "mlp1.")
        )

    def __init__(
        self,
        config: NemotronHOmniConfig,
        key_prefix: str = "vision_model.radio_model.model",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "video_input": True,
            "default_vision_bits": 16,
            "supports_tp": False,
        })
        v = config.vision
        self.merge_size = int(round(1.0 / config.downsample_ratio))

        # Blocks keep an fp32 residual stream; the reference has no LayerScale tensors in the
        # checkpoint (identity scale) and no final encoder norm
        self.modules += [
            NemotronHOmniPatchEmbedder(
                config = config,
                key = f"{key_prefix}.patch_generator",
                hidden_size = v.hidden_size,
                patch_dim = v.patch_dim,
                video_patch_dim = v.video_temporal_patch_size * v.patch_dim,
                table_side = v.pos_emb_side,
                out_dtype = torch.float,
            ),
        ]

        for idx in range(v.num_hidden_layers):
            key = f"{key_prefix}.blocks.{idx}"
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
                        rope_settings = None,
                        key_fused_qkv = "qkv",
                        key_o = "proj",
                        qmap = "block.attn",
                        out_dtype = torch.float,
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
                        out_dtype = torch.float,
                        pad_to = 1,
                    ),
                )
            ]

        self.modules += [
            PixelUnshuffle(
                config = config,
                key = f"{key_prefix}.pixel_shuffle",
                factor = self.merge_size,
                num_prefix_tokens = v.num_prefix_tokens,
                legacy_order = config.ps_version == "v1",
            ),
            RMSNorm(
                config = config,
                key = "mlp1.0",
                rms_norm_eps = 1e-5,
                out_dtype = torch.half,
            ),
            MLP(
                config = config,
                key = "mlp1",
                hidden_size = v.hidden_size * self.merge_size ** 2,
                intermediate_size = config.projector_hidden_size,
                key_up = "1",
                key_down = "3",
                activation_fn = "relu2",
                out_size = config.hidden_size,
                qmap = "block.mlp",
                out_dtype = torch.half,
                pad_to = 1,
            ),
        ]


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids


    def probe_grid(self) -> tuple[int, int]:
        # Largest square grid within the patch budget, a multiple of the merge size on each side
        side = int(self.config.vision_pp.max_num_patches ** 0.5) // self.merge_size * self.merge_size
        return side, side


    def default_load_shape_dtype(self, chunk_size):
        rows, cols = self.probe_grid()
        return (1, rows * cols, self.config.vision.patch_dim), torch.half


    def default_load_params(self, max_chunk_size):
        return self.make_vision_params(self.probe_grid())


    def make_vision_params(self, grid_hw: tuple, video: bool = False) -> dict:
        return {
            "causal": False,
            "grid_hw": grid_hw,
            "video": video,
        }


    def preprocess(
        self,
        image: Image.Image,
        video: bool = False,
    ) -> (torch.Tensor, tuple, tuple):
        """
        Returns the normalized (3, H, W) pixel tensor, the resize target (width, height) and the
        patch grid (rows, cols). Video frames use the fixed-budget grid, images the dynamic one.
        """
        pp = self.config.vision_pp
        if video:
            rows, cols = nemotronh_omni_video_grid(
                image.size,
                pp.video_target_num_patches,
                self.merge_size,
                pp.video_maintain_aspect_ratio,
            )
        else:
            budget = (pp.max_model_len - 4) * self.merge_size ** 2
            budget = min(max(budget, pp.min_num_patches), pp.max_num_patches)
            rows, cols = nemotronh_omni_image_grid(
                image.size,
                pp.patch_size,
                budget,
                pp.min_num_patches,
                self.merge_size,
            )
        size = (rows * pp.patch_size, cols * pp.patch_size)
        pixels = nemotronh_omni_image_to_tensor(image, size, pp.norm_mean, pp.norm_std)
        return pixels, (size[1], size[0]), (rows, cols)


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image.Image | list[Image.Image],
        text_alias: str | None = None,
    ):
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply a single alias to a list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        pixels, prep_image_size, grid_hw = self.preprocess(image)
        patches = nemotronh_omni_patchify(pixels.unsqueeze(0), self.config.vision.patch_size)
        embedding_tensor = self.forward(
            patches.half(),
            params = self.make_vision_params(grid_hw),
        ).cpu()

        id_start = tokenizer.single_id("<img>")
        id_end = tokenizer.single_id("</img>")
        token_string = torch.tensor([[id_start] + [-1] * embedding_tensor.shape[1] + [id_end]], dtype = torch.long)

        mme = MMEmbedding(
            embeddings = embedding_tensor[0],
            text_alias = text_alias,
            token_string = token_string,
        )
        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_image_size,
            "model_architecture": self.config.architecture,
            "media_type": "image",
        })
        return mme


    def get_video_embeddings(
        self,
        tokenizer: Tokenizer,
        frames: list[Image.Image],
        fps: float | None = None,
        frame_indices: list[int] | None = None,
        text_alias: str | None = None,
    ):
        """
        Embed sampled video frames. Frames are packed into temporal patches (tubelets) of
        video_temporal_patch_size, the last one padded by repeating the final frame, and each
        tubelet becomes its own <img>...</img> block preceded by the frame label the reference
        processor writes ("Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.50 seconds: ").
        Timestamps need `fps` (the source frame rate) and, when the frames were subsampled, the
        `frame_indices` they were taken from. Efficient video sampling then prunes the tokens least
        changed from the previous tubelet, so blocks after the first are typically shorter.
        """
        if not frames:
            raise ValueError("Video input requires at least one frame")
        if len({frame.size for frame in frames}) != 1:
            raise ValueError("All video frames must have the same dimensions")

        v = self.config.vision
        temporal = v.video_temporal_patch_size
        prepared = [self.preprocess(frame, video = True) for frame in frames]
        prep_image_size, grid_hw = prepared[0][1], prepared[0][2]
        pixels = torch.stack([p[0] for p in prepared])
        if pixels.shape[0] % temporal:
            pixels = torch.cat((pixels, pixels[-1:].expand(temporal - pixels.shape[0] % temporal, -1, -1, -1)))
        tubelets = pixels.reshape(-1, temporal * 3, pixels.shape[-2], pixels.shape[-1])
        patches = nemotronh_omni_patchify(tubelets, v.patch_size)

        embedding_tensor = self.forward(
            patches.half(),
            params = self.make_vision_params(grid_hw, video = True),
        ).cpu()
        rows, cols = grid_hw[0] // self.merge_size, grid_hw[1] // self.merge_size
        embedding_tensor = embedding_tensor.reshape(-1, rows, cols, embedding_tensor.shape[-1])

        keep_mask = nemotronh_omni_video_retention_mask(embedding_tensor, self.config.video_pruning_rate)
        counts = keep_mask.sum(dim = 1).tolist()
        embedding_tensor = embedding_tensor.reshape(-1, rows * cols, embedding_tensor.shape[-1])[keep_mask]

        id_start = tokenizer.single_id("<img>")
        id_end = tokenizer.single_id("</img>")
        labels = nemotronh_omni_frame_labels(len(frames), temporal, fps, frame_indices)
        token_ids = []
        for g, (label, count) in enumerate(zip(labels, counts)):
            token_ids += tokenizer.encode(("\n" if g > 0 else "") + label)[0].tolist()
            token_ids += [id_start] + [-1] * count + [id_end]
        token_string = torch.tensor([token_ids], dtype = torch.long)

        mme = MMEmbedding(
            embeddings = embedding_tensor,
            text_alias = text_alias,
            token_string = token_string,
        )
        mme.metadata.update({
            "original_size": frames[0].size,
            "preprocessed_size": prep_image_size,
            "frame_count": len(frames),
            "model_architecture": self.config.architecture,
            "media_type": "video",
        })
        return mme
