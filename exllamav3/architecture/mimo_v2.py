from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import (
    RMSNorm, Embedding, TransformerBlock, Attention, SlidingAttention, SWAState,
    GatedMLP, BlockSparseMLP, Linear,
)
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from .mimo_v2_mtp import MiMoV2MTPModel
from .mimo_v2_vision import MiMoV2VisionModel, read_mimo_v2_vision_config
from .qwen2_5_vl import read_qwen2_5_vl_pp_config
import os, json

# MiMo-V2's fused qkv_proj is stored as the concatenation of `ckpt_tp` tensor-parallel shards,
# each laid out [q_shard; k_shard; v_shard], and its FP8 weight_scale_inv grid is computed per
# shard with a ceiling (so a GA layer's 13568 rows carry 4 * ceil(3392/128) = 108 scale rows, not
# ceil(13568/128) = 106). Reference: sglang/srt/models/mimo_v2.py, load_mimo_v2_qkv_proj_weight /
# _get_ckpt_qkv_shard_sizes / _deinterleave_qkv_shards / _resolve_deferred_qkv_scale_inv, and
# sglang/srt/configs/model_config.py::get_mimo_v2_fused_qkv_expected_tp_size which fixes
# ckpt_tp = config.num_key_value_heads for every layer, GA and SWA alike.
FP8_BLOCK = 128


def _mimo_v2_qkv_dequant(
    ckpt_tp: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
):
    """Build the fdequant reader for one layer's fused qkv_proj.

    Returns the canonical (out, in) fp16 matrix [Q ; K ; V_padded], where V is widened from
    v_head_dim to head_dim per KV head with zeros so that the attention module's default fused
    ranges (q, k and v all num_heads * head_dim wide) address it directly. The zero lanes are
    dropped again after attention, before o_proj (see Attention.project_o)."""

    q_per = (num_q_heads // ckpt_tp) * head_dim
    k_per = max(1, num_kv_heads // ckpt_tp) * head_dim
    v_per = max(1, num_kv_heads // ckpt_tp) * v_head_dim
    shard = q_per + k_per + v_per

    def fn(stc, fkey: str, device):
        weight = stc.get_tensor(fkey + ".weight", device, no_defer = True)
        scale_inv = stc.get_tensor(fkey + ".weight_scale_inv", device, optional = True, no_defer = True)
        rows, cols = weight.shape
        assert rows == ckpt_tp * shard, \
            f"{fkey}: expected {ckpt_tp} x {shard} = {ckpt_tp * shard} rows, got {rows}"

        qs, ks, vs = [], [], []
        if scale_inv is not None:
            sb = scale_inv.shape[0] // ckpt_tp
            assert sb * ckpt_tp == scale_inv.shape[0] and sb == -(-shard // FP8_BLOCK), \
                f"{fkey}: scale_inv {tuple(scale_inv.shape)} is not {ckpt_tp} x " \
                f"{-(-shard // FP8_BLOCK)} blocks"
        for i in range(ckpt_tp):
            w = weight[i * shard : (i + 1) * shard].float()
            if scale_inv is not None:
                s = scale_inv[i * sb : (i + 1) * sb].float()
                s = s.repeat_interleave(FP8_BLOCK, dim = 0).repeat_interleave(FP8_BLOCK, dim = 1)
                w = w * s[:shard, :cols]
            w = w.half()
            qs.append(w[:q_per])
            ks.append(w[q_per : q_per + k_per])
            vs.append(w[q_per + k_per :])
            del w

        v = torch.cat(vs, dim = 0).view(num_kv_heads, v_head_dim, cols)
        v_pad = torch.zeros((num_kv_heads, head_dim, cols), dtype = torch.half, device = v.device)
        v_pad[:, :v_head_dim, :] = v
        out = torch.cat(qs + ks + [v_pad.view(num_kv_heads * head_dim, cols)], dim = 0)
        return out.contiguous()

    return fn


class MiMoV2Config(Config):
    arch_string = "MiMoV2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": MiMoV2Model, "mtp": MiMoV2MTPModel, "vision": MiMoV2VisionModel},
            **kwargs
        )

        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        # Next-token prediction heads (model.mtp.layers.*), loaded as the "mtp" component
        self.mtp_num_layers = self.read_cfg(int, "num_nextn_predict_layers", 0)
        if self.mtp_num_layers == 0:
            self.model_classes.pop("mtp", None)

        # Vision tower (visual.*), loaded as the "vision" component. The audio encoder and its
        # speech embeddings are not supported
        vision_config = self.read_cfg(dict, "vision_config", None)
        prep_path = os.path.join(self.directory, "preprocessor_config.json")
        if vision_config and os.path.exists(prep_path):
            self.vision = read_mimo_v2_vision_config(vision_config)
            with open(prep_path, encoding = "utf8") as f:
                self.vision_pp = read_qwen2_5_vl_pp_config(json.load(f))
            self.vision_start_token_id = self.read_cfg(int, ["vision_start_token_id", "processor_config->vision_start_token_id"], 151652)
            self.vision_end_token_id = self.read_cfg(int, ["vision_end_token_id", "processor_config->vision_end_token_id"], 151653)
        else:
            self.vision = None
            self.model_classes.pop("vision", None)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # Attention geometry. Q/K run at head_dim (192) while V runs at v_head_dim (128) on every
        # layer; GA and SWA layers differ only in the KV head count (4 vs 8) and the rope theta
        self.head_dim = self.read_cfg(int, "head_dim", None) or \
            self.read_cfg(int, "hidden_size", no_default) // self.read_cfg(int, "num_attention_heads", no_default)
        self.v_head_dim = self.read_cfg(int, "v_head_dim", self.head_dim)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.swa_head_dim = self.read_cfg(int, "swa_head_dim", self.head_dim)
        self.swa_v_head_dim = self.read_cfg(int, "swa_v_head_dim", self.v_head_dim)
        self.swa_num_q_heads = self.read_cfg(int, "swa_num_attention_heads", self.num_q_heads)
        self.swa_num_kv_heads = self.read_cfg(int, "swa_num_key_value_heads", self.num_kv_heads)
        assert self.swa_head_dim == self.head_dim, \
            "MiMoV2: differing GA/SWA Q/K head dims are not supported (one cache geometry per model)"
        assert self.swa_v_head_dim == self.v_head_dim, \
            "MiMoV2: differing GA/SWA V head dims are not supported"
        assert self.swa_num_q_heads == self.num_q_heads, \
            "MiMoV2: differing GA/SWA query head counts are not supported"

        self.assert_cfg(str, "attention_projection_layout", "fused_qkv", True)
        # ckpt_tp for the fused qkv interleave, per sglang get_mimo_v2_fused_qkv_expected_tp_size
        self.qkv_ckpt_tp = self.num_kv_heads

        # hybrid_layer_pattern: 0 = global attention, 1 = sliding-window attention
        self.hybrid_layer_pattern = self.read_cfg(list, "hybrid_layer_pattern", None)
        if self.hybrid_layer_pattern is None:
            self.hybrid_layer_pattern = [0] * self.num_hidden_layers
        assert len(self.hybrid_layer_pattern) == self.num_hidden_layers

        # Only config.sliding_window is read by the reference implementation; sliding_window_size
        # and attention_chunk_size are dead keys carrying the same value. HF's sliding mask keeps
        # `sliding_window` keys including the query, the kernels here keep `window + 1`
        self.sliding_window = self.read_cfg(int, "sliding_window", None)
        if any(p == 1 for p in self.hybrid_layer_pattern):
            assert self.sliding_window, "MiMoV2: sliding_window must be set when the layer pattern uses SWA"

        # Learned per-head attention sinks, gpt-oss style (extra logit column, dropped after softmax)
        self.add_full_attention_sink_bias = self.read_cfg(bool, "add_full_attention_sink_bias", False)
        self.add_swa_attention_sink_bias = self.read_cfg(bool, "add_swa_attention_sink_bias", False)

        # V is scaled before the KV cache write; attention is linear in V, so this folds into o_proj
        self.attention_value_scale = self.read_cfg(float, "attention_value_scale", None) or 1.0

        # MLP / MoE
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", None)
        self.num_experts = self.read_cfg(int, "n_routed_experts", None)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", 8)
        self.moe_layer_freq = self.read_cfg(list, "moe_layer_freq", None)
        if self.moe_layer_freq is None:
            self.moe_layer_freq = [1 if self.num_experts else 0] * self.num_hidden_layers
        assert len(self.moe_layer_freq) == self.num_hidden_layers
        assert self.read_cfg(int, "n_shared_experts", None) in (None, 0), \
            "MiMoV2: shared experts are not present in any published checkpoint"

        # DeepSeek-V3-style noaux_tc routing
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", None) or 1.0
        self.n_group = self.read_cfg(int, "n_group", 1) or 1
        self.topk_group = self.read_cfg(int, "topk_group", 1) or 1
        assert self.n_group == 1 and self.topk_group == 1, \
            f"MiMoV2: group-limited expert routing (n_group = {self.n_group}, topk_group = " \
            f"{self.topk_group}) is not supported"
        self.assert_cfg(str, "scoring_func", "sigmoid", True)
        self.assert_cfg(str, "topk_method", "noaux_tc", True)
        assert self.read_cfg(bool, "norm_topk_prob", True), \
            "MiMoV2: norm_topk_prob = false is not supported"

        # Norms. Note the HF key is layernorm_epsilon, not rms_norm_eps
        self.rms_norm_eps = self.read_cfg(float, ["layernorm_epsilon", "rms_norm_eps"], 1e-6)

        # RoPE: partial (0.334 * 192 = 64 leading dims, NEOX/rotate-half), with a different theta
        # for GA (rope_theta) and SWA (swa_rope_theta) layers
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)
        self.rope_settings_swa = self.read_rope_settings_default(
            RopeStyle.NEOX,
            theta_key = ["swa_rope_theta", "rope_theta", "rope_parameters->rope_theta"],
        )


    def qkv_dequant(self, layer_idx: int):
        return self._qkv_dequant(self.hybrid_layer_pattern[layer_idx] == 1)


    def qkv_dequant_swa(self):
        return self._qkv_dequant(True)


    def _qkv_dequant(self, swa: bool):
        return _mimo_v2_qkv_dequant(
            self.qkv_ckpt_tp,
            self.swa_num_q_heads if swa else self.num_q_heads,
            self.swa_num_kv_heads if swa else self.num_kv_heads,
            self.swa_head_dim if swa else self.head_dim,
            self.swa_v_head_dim if swa else self.v_head_dim,
        )


class MiMoV2Model(Model):
    config_class = MiMoV2Config

    def __init__(
        self,
        config: MiMoV2Config,
        key_prefix: str = "model",
        swa_full: bool = False,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        # swa_full = True keeps every layer on the paged full-length cache (the original port's
        # behaviour, kept for A/B testing); the default routes the 39 sliding-window layers
        # through SlidingAttention's per-slot window ring instead
        self.swa_full = swa_full

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            swa = config.hybrid_layer_pattern[idx] == 1
            num_kv_heads = config.swa_num_kv_heads if swa else config.num_kv_heads
            has_sinks = config.add_swa_attention_sink_bias if swa else config.add_full_attention_sink_bias

            # Shared between both attention flavours. The kernels' left window keeps
            # window + 1 keys including the query, the HF mask keeps sliding_window including
            # the query, so the window passed down is sliding_window - 1
            attn_kwargs = dict(
                config = config,
                key = f"{key_prefix}.layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                v_head_dim = config.v_head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = num_kv_heads,
                rope_settings = config.rope_settings_swa if swa else config.rope_settings,
                # sm_scale follows the Q/K head dim (192 ** -0.5), which is the module default
                sm_scale = None,
                key_fused_qkv = "qkv_proj",
                key_o = "o_proj",
                key_sinks = "attention_sink_bias" if has_sinks else None,
                qmap = "block.attn",
                out_dtype = torch.float,
                select_hq_bits = 2,
            )
            if swa and not swa_full:
                # 39 of 48 layers only ever look 128 tokens back. On the paged full cache they
                # would still cost 8 * 192 * 2 halves per token each (~240 KB/token for the
                # stack); the window ring makes them a fixed per-slot allocation instead.
                attn = SlidingAttention(
                    sliding_window = config.sliding_window - 1,
                    **attn_kwargs,
                )
            else:
                attn = Attention(
                    sliding_window = config.sliding_window - 1 if swa else -1,
                    **attn_kwargs,
                )
            # The fused qkv tensor is TP-shard-interleaved and its FP8 scale grid is per shard;
            # neither is expressible with the generic fused-tensor paths. One reader shared by
            # the three Linears that slice out of it
            qkv_reader = config.qkv_dequant(idx)
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                proj.fdequant = qkv_reader
            # attention_value_scale multiplies V before the cache write; fold it into o_proj
            attn.o_proj.weight_scale = config.attention_value_scale

            if config.moe_layer_freq[idx]:
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_gate = "experts.{expert_idx}.gate_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_routing_gate = "gate",
                    key_e_score_bias = "gate.e_score_correction_bias",
                    qmap = "block.mlp",
                    # Layer 47's routed experts take act(gate) * up to ~84k on some tokens, past
                    # the fp16 max (every other layer peaks under 3k)
                    interm_div = 128.0 if idx == 47 else 1.0,
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    router_type = "dots",
                    routed_scaling_factor = config.routed_scaling_factor,
                    n_group = config.n_group,
                    topk_group = config.topk_group,
                )
            else:
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
                    select_hq_bits = 1,
                )

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = mlp,
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
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        self.caps.update({
            "supports_tp": True,
        })

        # SWA layers keep their KV in a per-slot window ring rather than the paged cache
        self.recurrent_state_cls = None
        if not self.swa_full:
            self.caps.update({
                "recurrent_states": True,
                "default_recurrent_checkpoint_interval": 2048,
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
        # MiMo's template joins turns without a newline after <|im_end|>
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n{system_prompt}<|im_end|>"
        p += f"<|im_start|>user\n{prompt}<|im_end|>"
        p += f"<|im_start|>assistant\n"
        return p
