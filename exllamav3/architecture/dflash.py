from __future__ import annotations
from typing_extensions import override
import torch

from ..cache import Cache
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, RoPE
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.arch_specific.dflash import DFlashInputLayer, DFlashAttention
from ..modules.attn import prepare_for_attn
from ..modules.module import no_p2p_copy
from ..ext import exllamav3_ext as ext
from flash_attn import flash_attn_with_kvcache
import weakref

from ..util.tensor import get_for_device

# TODO: Support DFlash models trained in Speculators (includes lm_head for speculator with limited vocabulary?)

class DFlashConfig(Config):
    arch_string = "DFlashDraftModel"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": DFlashModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        # self.num_target_layers = self.read_cfg(int, "num_target_layers", no_default)
        self.layer_types = self.read_cfg(list, "layer_types", ["full_attention"] * self.num_hidden_layers)
        self.sliding_window = self.read_cfg(int, "sliding_window", 2048)

        # DFlash
        self.mask_token_id = self.read_cfg(int, "dflash_config->mask_token_id", no_default)
        self.target_layer_ids = self.read_cfg(list, "dflash_config->target_layer_ids", no_default)
        assert len(self.target_layer_ids) == self.num_hidden_layers, \
            "Length of target layer list doesn't match num_hidden_layers"
        self.block_size = self.read_cfg(int, "block_size", no_default)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)

        # Vision placeholders
        self.vision = None


class DFlashModel(Model):
    config_class = DFlashConfig

    def __init__(
        self,
        config: DFlashConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.input_layer = DFlashInputLayer(
            config = config,
            key = "fc",
            key_norm = "hidden_norm",
            hidden_size = config.hidden_size,
            target_state_size = config.hidden_size * config.num_hidden_layers,
            mask_token_id = config.mask_token_id,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = config.block_size,
        )
        self.modules += [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.num_hidden_layers):
            is_swa = config.layer_types[idx] == "sliding_attention"

            attn = DFlashAttention(
                config = config,
                key = f"layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_kv_heads,
                rope_settings = config.rope_settings,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                qmap = "block.attn",
                sliding_window = config.sliding_window if is_swa else -1,
                q_norm = RMSNorm(
                    config = config,
                    key = f"layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                out_dtype = torch.float,
            )
            self.attn_modules.append(attn)

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            RMSNorm(
                config = config,
                key = f"norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            )
        ]

        self.logit_layer_idx = None
        self.caps.update({
            "can_quantize": False,
            "supports_tp": False,
            "attach_target": True,
            "dflash_draft": True,
            "default_draft_size": self.config.block_size - 1,
            "autosplit_load_fwd": False,
        })

        self.lm_head = None


    def attach_to(self, target):
        for i, l in enumerate(self.config.target_layer_ids):
            t = target.modules[target.first_block_idx + l]
            t.export_state = True
        self.input_layer.embedding = weakref.ref(target.modules[0])
        self.lm_head = weakref.ref(target.modules[-1])


    def update_kv_from_target(
        self,
        target_hidden: list,
        cache: Cache,
        params: dict,
        lengths: list[int] = None,
    ):
        """
        Update K/V cache with hidden states extracted from target model

        params:
            "block_table": torch.Tensor
            "cache_seqlens": torch.Tensor
        """

        # May update a few redundant tokens when batching, but we'd never draft longer than the cache length
        if lengths is not None:
            max_length = max(lengths)
            target_hidden = [t[:, :max_length] for t in target_hidden]

        # Ensure all state snapshots are on the same device
        device = self.input_layer.device
        for i in range(len(target_hidden)):
            if target_hidden[i].device != device:
                if no_p2p_copy:
                    target_hidden[i] = target_hidden[i].cpu().to(device)
                else:
                    target_hidden[i] = target_hidden[i].to(device)

        # Projection concatenated states to hidden size, once
        target_hidden = torch.cat(target_hidden, dim = -1)
        target_hidden = self.input_layer.proj.forward(target_hidden, {}, out_dtype = torch.half)
        target_hidden = self.input_layer.norm.forward(target_hidden, {}, out_dtype = torch.half)

        bsz, target_seqlen, dim = target_hidden.shape
        params["target_hidden_cc"] = target_hidden

        # Update KV layers
        for layer in self.attn_modules:
            block_table = get_for_device(params, "block_table", layer.device)
            cache_seqlens = get_for_device(params, "cache_seqlens", layer.device)
            target_hidden = get_for_device(params, "target_hidden_cc", layer.device)

            cache_k, cache_v = cache.get_layer(layer.layer_idx, cache_seqlens, block_table, -1)

            # k/v project
            k = layer.k_proj.forward(target_hidden, params)
            v = layer.v_proj.forward(target_hidden, params)
            k = k.view(bsz, target_seqlen, layer.num_kv_heads, layer.head_dim)
            v = v.view(bsz, target_seqlen, layer.num_kv_heads, layer.head_dim)

            # Apply rope and norm to k
            k, _ = layer.rope.apply(
                k, None,
                0,
                cache_seqlens,
                None,
                True,
                layer.k_norm_tensor,
                None,
                layer.norm_eps,
                layer.norm_constant_bias,
                None,
                False,
            )

            # Write k, v to paged cache
            ext.paged_kv_cache_update(
                k, v,
                cache_k, cache_v,
                block_table,
                cache_seqlens,
            )

            cache.update_layer(layer.layer_idx, cache_seqlens, block_table, cache_k, cache_v, target_seqlen)


    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        logits = self.lm_head().prepare_for_device(state, params)
        logits = self.lm_head().forward(logits, params)
        return torch.argmax(logits, dim = -1)


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError()