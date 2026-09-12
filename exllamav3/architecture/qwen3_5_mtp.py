from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..cache import Cache
from ..ext import exllamav3_ext as ext
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear, BlockSparseMLP
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.quant.exl3 import LinearEXL3
from ..modules.attn import prepare_for_attn
from ..util.tensor import get_for_device
from .mtp_hot_vocab import MTPHotVocabConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig


def validate_mtp_hot_blocks(block_ids: list[int], full_vocab: int):
    """Validate packed EXL3 output blocks before constructing a reordered draft head."""
    if not block_ids or block_ids[0] < 0:
        raise ValueError("MTP hot blocks must be a nonempty list of nonnegative IDs")
    if block_ids != sorted(set(block_ids)):
        raise ValueError("MTP hot blocks must be unique and sorted")
    if block_ids[-1] >= (full_vocab + 15) // 16:
        raise ValueError("MTP hot block ID exceeds the model vocabulary")
    # EXL3 applies an independent output Hadamard transform to each 128-token group.
    # Reordering smaller units produces plausible-shaped but invalid logits.
    if len(block_ids) % 8:
        raise ValueError("MTP hot blocks must contain complete 128-token groups")
    for offset in range(0, len(block_ids), 8):
        first = block_ids[offset]
        if first % 8 or block_ids[offset:offset + 8] != list(range(first, first + 8)):
            raise ValueError("MTP hot blocks must be aligned contiguous groups of 8 packed blocks")


class Qwen3_5MTPModel(Model):

    def __init__(
        self,
        config: Qwen3_5Config | Qwen3_5MoeConfig,
        use_moe: bool = False,
        mtp_hot_vocab_config: MTPHotVocabConfig | None = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.use_moe = use_moe
        self.mtp_hot_vocab_config = (
            mtp_hot_vocab_config
            if mtp_hot_vocab_config is not None
            else MTPHotVocabConfig.from_env()
        )

        # Module list: optional embed, then pre_fc norms + fc, then num_mtp_layers * TransformerBlock, then norm
        self.input_layer = Qwen3_5MTPInputLayer(
            config = config,
            key = "mtp",
            key_pre_fc_norm_hidden = "mtp.pre_fc_norm_hidden",
            key_pre_fc_norm_embedding = "mtp.pre_fc_norm_embedding",
            key_fc = "mtp.fc",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        )

        self.modules = [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.mtp_num_hidden_layers):
            attn = Attention(
                config = config,
                key = f"mtp.layers.{idx}.self_attn",
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
                q_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                out_dtype = torch.float,
                interleaved_gate = True,
                qbits_key = "mtp_bits",
            )
            self.attn_modules.append(attn)

            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = f"mtp.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    mlp = (
                        BlockSparseMLP(
                            config = config,
                            key = f"mtp.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.moe_intermediate_size,
                            num_experts = config.num_experts,
                            num_experts_per_tok = config.num_experts_per_tok,
                            key_up = "experts.{expert_idx}.up_proj",
                            key_gate = "experts.{expert_idx}.gate_proj",
                            key_down = "experts.{expert_idx}.down_proj",
                            key_gate_up_split = "experts.gate_up_proj",
                            key_down_split = "experts.down_proj",
                            key_routing_gate = "gate",
                            key_shared_gate = "shared_expert_gate",
                            transpose_fused_weights = False,
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            qbits_key = "mtp_bits",
                            shared_experts = GatedMLP(
                                config = config,
                                key = f"mtp.layers.{idx}.mlp.shared_expert",
                                hidden_size = config.hidden_size,
                                intermediate_size = config.shared_expert_intermediate_size,
                                key_up = "up_proj",
                                key_gate = "gate_proj",
                                key_down = "down_proj",
                                qmap = "block.mlp",
                                interm_dtype = torch.half,
                                out_dtype = torch.float,
                                qbits_key = "mtp_bits",
                                select_hq_bits = 2,
                            )
                        ) if use_moe else
                        GatedMLP(
                            config = config,
                            key = f"mtp.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.intermediate_size,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            qbits_key = "mtp_bits",
                        )
                    ),
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # Final norm
        self.final_norm = RMSNorm(
            config = config,
            key = "mtp.norm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
            constant_bias = 1.0,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 4,  # best measured performance
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None
        self.mtp_sub_lm_head = None
        self.mtp_hot_vocab = 0
        self.mtp_hot_id_map = None
        self.mtp_subhead_validation = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # MTP doesn't take input_ids through Embedding here — embedding is handled in step()
        # But prepare_for_attn still wires up flash-attn params
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and tell target to export hidden state.

        Qwen3.5/3.6 MTP consumes the trunk's post-final-norm hidden state. This differs from
        DeepSeek/GLM-style MTP heads, which consume a pre-norm residual stream.
        """
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)

        # Find the target's embedding (first module of class Embedding)
        target_embed = None
        for m in target.modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        # lm_head is the last module
        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        # Experimental single-GPU MTP fast path. Draft against a selected vocabulary subset and
        # keep its matching input embeddings on GPU. The full target head still verifies every
        # proposed token, so the subset changes draft cost and acceptance rather than target quality.
        hot_config = self.mtp_hot_vocab_config
        hot_blocks_path = hot_config.blocks_path
        full_head = target.modules[-1].inner
        if hot_blocks_path:
            if target.loaded_tp:
                raise ValueError("MTP hot vocabulary currently supports layer-split inference only")
            if not isinstance(full_head, LinearEXL3):
                raise ValueError("MTP hot vocabulary currently requires an EXL3 lm_head")
            full_vocab = target.modules[-1].out_features_unpadded
            with open(hot_blocks_path, "r", encoding = "utf-8") as f:
                block_ids = [int(line) for line in f if line.strip() and not line.lstrip().startswith("#")]
            validate_mtp_hot_blocks(block_ids, full_vocab)
            block_idx = torch.tensor(block_ids, device = full_head.trellis.device, dtype = torch.long)
            token_ids = (
                block_idx[:, None] * 16 +
                torch.arange(16, device = block_idx.device)[None, :]
            ).flatten()
            token_ids = token_ids[token_ids < full_vocab]
            # Keep complete packed blocks. The model's vocabulary is block-aligned today;
            # rejecting a partial final block avoids ambiguous EXL3 output dimensions.
            if token_ids.numel() != len(block_ids) * 16:
                raise ValueError("MTP hot blocks may not include a partial final vocabulary block")
            hot_vocab = token_ids.numel()
            trellis = full_head.trellis.index_select(1, block_idx).contiguous()
            svh = full_head.svh.index_select(0, token_ids).contiguous()
            bias = full_head.bias.index_select(0, token_ids).contiguous() if full_head.bias is not None else None
            self.mtp_hot_id_map = token_ids
            self.mtp_sub_lm_head = LinearEXL3(
                config = target.config,
                in_features = full_head.in_features,
                out_features = hot_vocab,
                suh = full_head.suh,
                svh = svh,
                trellis = trellis,
                mcg = full_head.mcg_tensor,
                mul1 = full_head.mul1_tensor,
                bias = bias,
                out_dtype = full_head.out_dtype,
                key = "mtp.hot_lm_head",
            )
            embed = self.target_embed().embedding.weight.index_select(0, token_ids.cpu())
            embed_dtype = hot_config.embedding_dtype.lower()
            if embed_dtype in ("fp8", "float8", "e4m3"):
                embed_dtype = torch.float8_e4m3fn
            elif embed_dtype in ("fp16", "float16", "half"):
                embed_dtype = torch.float16
            else:
                raise ValueError(f"Unsupported MTP hot-embedding dtype: {embed_dtype!r}")
            self.input_layer.hot_embedding = embed.to(
                device = full_head.trellis.device,
                dtype = embed_dtype,
            ).contiguous()
            inverse = torch.full((full_vocab,), -1, device = token_ids.device, dtype = torch.long)
            inverse[token_ids] = torch.arange(hot_vocab, device = token_ids.device)
            self.input_layer.hot_inverse = inverse
            self.mtp_hot_vocab = hot_vocab

            if hot_config.validate_full_head:
                self.mtp_subhead_validation = {
                    "total": 0,
                    "full_in_hot_vocab": 0,
                    "matches_when_full_in_hot_vocab": 0,
                }

        target_norm = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_norm, RMSNorm), "Expected target final RMSNorm immediately before lm_head"
        self.draft_verifier_params.update({
            "export_state_norm_keys": {target_norm.key},
        })


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        if self.mtp_sub_lm_head is not None:
            logits = self.mtp_sub_lm_head.forward(state, params)
            if params.get("export_draft_conf"):
                conf, sub_argmax = torch.max(logits, dim = -1)
                params["draft_conf"] = conf
            else:
                sub_argmax = torch.argmax(logits, dim = -1)
            sub_argmax = self.mtp_hot_id_map[sub_argmax]
            if self.mtp_subhead_validation is not None:
                ll = self.attached_model().logit_layer_idx
                lm = self.attached_model().modules[ll]
                full_state = lm.prepare_for_device(state, params)
                full_argmax = torch.argmax(lm.forward(full_state, params), dim = -1)
                in_hot = self.input_layer.hot_inverse[full_argmax] >= 0
                matches = sub_argmax == full_argmax
                stats = self.mtp_subhead_validation
                stats["total"] += full_argmax.numel()
                stats["full_in_hot_vocab"] += in_hot.sum().item()
                stats["matches_when_full_in_hot_vocab"] += (matches & in_hot).sum().item()
            return sub_argmax
        elif not self.attached_model().loaded_tp:
            ll = self.attached_model().logit_layer_idx
            lm = self.attached_model().modules[ll]
            logits = lm.prepare_for_device(state, params)
            logits = lm.forward(logits, params)
            if params.get("export_draft_conf"):
                # Per-position confidence for the generator's draft truncation: the argmax logit
                # value, over the unpadded vocabulary
                logits = logits[..., :self.attached_model().config.vocab_size]
                conf, ids = torch.max(logits, dim = -1)
                params["draft_conf"] = conf
                return ids
            return torch.argmax(logits, dim = -1)
        else:
            state = self.attached_model().tp_producer.send(state)
            argmax = self.attached_model().tp_dispatch_lm_head_argmax((state, {}))
            return argmax


class Qwen3_5MoeMTPModel(Qwen3_5MTPModel):

    def __init__(
        self,
        config: Qwen3_5Config | Qwen3_5MoeConfig,
        **kwargs
    ):
        super().__init__(config, use_moe = True, **kwargs)

