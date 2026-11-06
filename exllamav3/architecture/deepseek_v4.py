from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Embedding, RMSNorm, Linear, GatedMLP, BlockSparseMLP, TransformerBlock, \
    HyperConnection, ExpandStreams, HyperHead
from ..modules.dsv4 import DSV4Attention
from ..modules.attn import prepare_for_attn
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from .deepseek_v4_mtp import DeepseekV4MTPModel
from .deepseek_v4_vision import DeepseekV4VisionModel, read_deepseek_v4_vision_config

# DeepSeek-V4: hybrid sparse attention (sliding / CSA / HCA per compress_ratios), mHC
# hyper-connection residual streams, hash-MoE bootstrap layers, sqrt-softplus routing.
# Checkpoint uses DeepSeek's native tensor namespace (layers.N.attn.wq_a, hc_attn_fn,
# embed / head / norm, ...). Reference implementation: transformers models/deepseek_v4.

_RATIO_TO_TYPE = {0: "sliding", 4: "csa", 128: "hca"}


class DeepseekV4Config(Config):
    arch_string = "DeepseekV4ForCausalLM"
    # Where the text settings live in config.json (V4.1 nests them under text_config)
    cfg_prefix = ""
    default_num_hash_layers = 3

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": DeepseekV4Model, "mtp": DeepseekV4MTPModel},
            **kwargs
        )
        p = self.cfg_prefix

        # Attention
        self.hidden_size = self.read_cfg(int, p + "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, p + "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, p + "num_key_value_heads", 1)
        assert self.num_kv_heads == 1, "DeepseekV4: expected shared-KV MQA (num_key_value_heads == 1)"
        self.head_dim = self.read_cfg(int, p + "head_dim", 512)
        self.qk_rope_head_dim = self.read_cfg(int, p + "qk_rope_head_dim", 64)
        self.q_lora_rank = self.read_cfg(int, p + "q_lora_rank", no_default)
        self.o_groups = self.read_cfg(int, p + "o_groups", 8)
        self.o_lora_rank = self.read_cfg(int, p + "o_lora_rank", 1024)
        self.sliding_window = self.read_cfg(int, p + "sliding_window", 128)
        self.index_n_heads = self.read_cfg(int, p + "index_n_heads", 64)
        self.index_head_dim = self.read_cfg(int, p + "index_head_dim", 128)
        self.index_topk = self.read_cfg(int, p + "index_topk", 512)

        # Layer schedule: compress_ratios per layer (0 = sliding, 4 = CSA, 128 = HCA); the
        # list may run past num_hidden_layers to describe MTP blocks (ignored here)
        self.num_hidden_layers = self.read_cfg(int, p + "num_hidden_layers", no_default)
        ratios = self.read_cfg(list, p + "compress_ratios", None)
        self.compress_ratios = ratios[:self.num_hidden_layers] if ratios is not None else None
        if ratios is not None:
            self.layer_types = [self.layer_type(r) for r in self.compress_ratios]
        else:
            inter = ["csa" if i % 2 else "hca" for i in range(max(self.num_hidden_layers - 2, 0))]
            self.layer_types = ["hca"] * min(self.num_hidden_layers, 2) + inter
        self.compress_rate_csa = self.read_cfg(int, p + "compress_rate_csa", 4)
        self.compress_rate_hca = self.read_cfg(int, p + "compress_rate_hca", 128)

        # mHC
        self.hc_mult = self.read_cfg(int, p + "hc_mult", 4)
        self.hc_sinkhorn_iters = self.read_cfg(int, p + "hc_sinkhorn_iters", 20)
        self.hc_eps = self.read_cfg(float, p + "hc_eps", 1e-6)

        # MoE
        self.assert_cfg(str, p + "scoring_func", "sqrtsoftplus", optional = True)
        self.assert_cfg(str, p + "topk_method", "noaux_tc", optional = True)
        self.moe_intermediate_size = self.read_cfg(int, p + "moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, p + "n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, p + "num_experts_per_tok", no_default)
        self.num_shared_experts = self.read_cfg(int, p + "n_shared_experts", 1)
        self.num_hash_layers = self.read_cfg(int, p + "num_hash_layers", self.default_num_hash_layers)
        self.routed_scaling_factor = self.read_cfg(float, p + "routed_scaling_factor", 1.0)
        self.swiglu_limit = self.read_cfg(float, p + "swiglu_limit", 10.0)

        # Norms / rope
        self.rms_norm_eps = self.read_cfg(float, p + "rms_norm_eps", 1e-6)
        self.rope_theta = self.read_cfg(float, p + "rope_theta", 10000.0)
        self.compress_rope_theta = self.read_cfg(float, p + "compress_rope_theta", 160000.0)
        self.rope_scaling = self.read_cfg(dict, p + "rope_scaling", None)

        self.tie_word_embeddings = self.read_cfg(bool, p + "tie_word_embeddings", False)

        # DSpark drafter (mtp.* namespace). num_nextn_predict_layers is unreliable
        # (V4-Flash ships 1 alongside three mtp blocks); the compress_ratios tail past the
        # trunk layers describes the MTP blocks. The component only exists when the
        # checkpoint actually carries the tensors
        self.dspark_block_size = self.read_cfg(int, p + "dspark_block_size", 0)
        self.dspark_noise_token_id = self.read_cfg(int, p + "dspark_noise_token_id", 0)
        self.dspark_markov_rank = self.read_cfg(int, p + "dspark_markov_rank", 256)
        self.dspark_target_layer_ids = self.read_cfg(list, p + "dspark_target_layer_ids", [])
        # Generator drafter convention (dflash flow): block_size counts the seed position
        self.block_size = self.dspark_block_size + 1
        if ratios is not None and len(ratios) > self.num_hidden_layers:
            self.num_mtp_layers = len(ratios) - self.num_hidden_layers
            self.mtp_layer_types = [self.layer_type(r) for r in ratios[self.num_hidden_layers:]]
        else:
            self.num_mtp_layers = 0
            self.mtp_layer_types = []
        if self.num_mtp_layers == 0 or not any(
            self.stc.has_tensor(f"mtp.0.attn.wkv.{t}") for t in ("weight", "trellis")):
            del self.model_classes["mtp"]

        # Vision tower (V4-Flash-Vision-Exp): plain ViT + aligner at the checkpoint root
        self.vision = self.read_vision_config()
        if self.vision is not None and any(
            self.stc.has_tensor(f"vision.patch_embed.proj.{t}") for t in ("weight", "trellis")):
            self.model_classes["vision"] = self.vision_model_class()
        else:
            self.vision = None

    def layer_type(self, ratio: int) -> str:
        return _RATIO_TO_TYPE[ratio]

    def compress_rate(self, idx: int) -> int | None:
        """Pooling rate of a trunk layer: by its type, the config's per-type rates"""
        return {
            "sliding": None,
            "csa": self.compress_rate_csa,
            "hca": self.compress_rate_hca,
        }[self.layer_types[idx]]

    def read_vision_config(self):
        return read_deepseek_v4_vision_config(self)

    def vision_model_class(self):
        return DeepseekV4VisionModel


class DeepseekV4Model(Model):
    config_class = DeepseekV4Config

    def __init__(
        self,
        config: DeepseekV4Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = "embed",
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
            key = f"layers.{idx}"
            self.modules += self.make_pre_block(idx, key)
            attn = self.make_attn(idx, key)
            mlp = self.make_mlp(idx, key)
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(config, f"{key}.attn_norm", config.rms_norm_eps),
                    attn = attn,
                    attn_hc = self.make_hc(key, "attn"),
                    mlp_norm = RMSNorm(config, f"{key}.ffn_norm", config.rms_norm_eps),
                    mlp = mlp,
                    mlp_hc = self.make_hc(key, "ffn"),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("head"):
            head_alt_key = "embed"

        self.modules += [
            HyperHead(
                config = config,
                key = "hc_head",
                hc_mult = config.hc_mult,
                rms_norm_eps = config.rms_norm_eps,
                hc_eps = config.hc_eps,
                carry_pre = self.hc_carry_pre,
            ),
            RMSNorm(
                config = config,
                key = "norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True},
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True

        # All attention state is recurrent-style (rings + pools), no paged KV modules
        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
        })
        if config.vision is not None:
            # Image spans are prefilled as exactly one chunk each (see _prepare_image_chunk)
            self.caps.update({"mm_exact_chunks": True})
        from ..cache.dsa import DSV4State
        self.recurrent_state_cls = DSV4State

    # Every mHC site collapses with the previous site's pre weights (V4.1) instead of its own
    hc_carry_pre = False

    def make_pre_block(self, idx: int, key: str) -> list:
        """Modules that run on the stream stack ahead of block idx (none in V4)"""
        return []

    def make_attn(self, idx: int, key: str, **extra) -> DSV4Attention:
        config = self.config
        return DSV4Attention(
            config = config,
            key = f"{key}.attn",
            layer_idx = idx,
            layer_type = config.layer_types[idx],
            hidden_size = config.hidden_size,
            num_q_heads = config.num_q_heads,
            head_dim = config.head_dim,
            rope_head_dim = config.qk_rope_head_dim,
            q_lora_rank = config.q_lora_rank,
            o_groups = config.o_groups,
            o_lora_rank = config.o_lora_rank,
            sliding_window = config.sliding_window,
            compress_rate = config.compress_rate(idx),
            index_n_heads = config.index_n_heads,
            index_head_dim = config.index_head_dim,
            index_topk = config.index_topk,
            rope_theta = config.rope_theta,
            compress_rope_theta = config.compress_rope_theta,
            rope_scaling = config.rope_scaling,
            rms_norm_eps = config.rms_norm_eps,
            qmap = "block.attn",
            out_dtype = torch.float,
            select_hq_bits = 2,
            **extra,
        )

    def make_mlp(self, idx: int, key: str) -> BlockSparseMLP:
        config = self.config
        is_hash = idx < config.num_hash_layers
        return BlockSparseMLP(
            config = config,
            key = f"{key}.ffn",
            hidden_size = config.hidden_size,
            intermediate_size = config.moe_intermediate_size,
            num_experts = config.num_experts,
            num_experts_per_tok = config.num_experts_per_tok,
            key_up = "experts.{expert_idx}.w3",
            key_gate = "experts.{expert_idx}.w1",
            key_down = "experts.{expert_idx}.w2",
            key_routing_gate = "gate",
            key_e_score_bias = "gate.bias",
            key_e_score_bias_vl = "gate.bias_vl",       # vision variant only (optional)
            key_tid2eid = "gate.tid2eid" if is_hash else None,
            qmap = "block.mlp",
            interm_dtype = torch.half,
            out_dtype = torch.float,
            activation_fn = "silu",
            act_limit = config.swiglu_limit,
            router_type = "sqrtsp_hash" if is_hash else "sqrtsp",
            routed_scaling_factor = config.routed_scaling_factor,
            shared_experts = GatedMLP(
                config = config,
                key = f"{key}.ffn.shared_experts",
                hidden_size = config.hidden_size,
                intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                key_up = "w3",
                key_gate = "w1",
                key_down = "w2",
                qmap = "block.mlp",
                out_dtype = torch.float,
                activation_fn = "silu",
                act_limit = config.swiglu_limit,
                select_hq_bits = 2,
            ),
        )

    def make_hc(self, key: str, tag: str) -> HyperConnection:
        config = self.config
        return HyperConnection(
            config = config,
            key = f"{key}.hc_{tag}",
            hc_mult = config.hc_mult,
            hidden_size = config.hidden_size,
            sinkhorn_iters = config.hc_sinkhorn_iters,
            hc_eps = config.hc_eps,
            rms_norm_eps = config.rms_norm_eps,
            carry_pre = self.hc_carry_pre,
        )

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Hash-MoE layers route by token id (get_for_device copies once per device)
        params["input_ids"] = input_ids
        if self.hc_carry_pre:
            # The first site collapses stream 0 (identity pre weights), see HyperConnection
            params["hc_pre"] = None
        _prepare_image_chunk(input_ids, params)
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"{system_prompt}\n\n"
        p += f"<|User|>{prompt}<|Assistant|>"
        return p


def _prepare_image_chunk(input_ids: torch.Tensor, params: dict):
    """
    Vision (V4-Flash-Vision-Exp): an image span [IMAGE_START .. IMAGE_END] must be prefilled
    as exactly one chunk, which then attends bidirectionally within itself (plus each row's
    ordinary window into the history) -- the reference's per-row "visible" bounds. The
    generator cuts chunks that way (caps mm_exact_chunks); this checks it and sets
    params["nc_chunk"]. The embedding's leading pad rows (before align_lead) are text-like.
    """
    embs = params.get("indexed_embeddings")
    if not embs or input_ids.shape[1] == 1:
        return
    ids = input_ids[0]
    if not bool((ids >= FIRST_MM_EMBEDDING_INDEX).any()):
        return
    for e in embs:
        lo, hi = e.first_index + getattr(e, "align_lead", 0), e.last_index
        n_span = int(((ids >= lo) & (ids < hi)).sum())
        if n_span == 0:
            continue
        span_len = hi - lo
        if input_ids.shape[0] == 1 and n_span == ids.shape[0] == span_len:
            params["nc_chunk"] = True
            return
        raise RuntimeError(
            "DeepSeek-V4 vision: an image span must be prefilled as exactly one chunk "
            f"(chunk of {ids.shape[0]} rows holds {n_span} of a {span_len}-row span)")
