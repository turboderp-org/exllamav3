from __future__ import annotations
from typing_extensions import override
import os
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    BlockSparseMLP,
    Linear,
    GatedDeltaNet,
    GatedMLP,
    GDNState,
    ExpandStreams,
    GatedResidual,
    PLELayer,
)
from ..modules.gated_rmsnorm import GatedRMSNorm
from ..modules.qsa_indexer import QSAIndexer
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from .qwen3_5 import Qwen3_5VLMoeBaseConfig
from .qwen4_exp_mtp import Qwen4ExpMTPModel

# Qwen3.8-Flash-Next (qwen4_exp): Qwen3.5-MoE lineage with QSA sparse full-attention layers
# (indexer-selected 4-token blocks over standard GQA), low-rank gated-residual hyper-connections
# in place of input/post layernorms (4 fp32 streams, elementwise mixing), a sigmoid-gated GDN
# output norm, and a PLE layer injecting hashed n-gram embeddings (51.2B-row table, streamed
# from disk by default) into the stream stack ahead of one early layer. No final model norm:
# the combine-less hyper_connection_mixer collapses the streams before the head.


class Qwen4ExpConfig(Qwen3_5VLMoeBaseConfig):
    arch_string = "Qwen4ExpForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        from .qwen3_vl import Qwen3VLVisionModel
        # Identical tower to Qwen3.5 (HF: Qwen4ExpVisionModel subclasses Qwen3_5MoeVisionModel
        # with no changes; no deepstack). Text-only checkpoints (converted dirs without a
        # transplanted tower) are detected by the missing preprocessor config
        has_vision = os.path.exists(os.path.join(directory, "preprocessor_config.json"))
        super().__init__(
            directory,
            "text_config",
            Qwen4ExpModel,
            Qwen3VLVisionModel if has_vision else None,
            Qwen4ExpMTPModel,
            **kwargs
        )
        pfx = lambda key: f"text_config->{key}"

        # PLE hashing must see the literal MM placeholder token where the generator substitutes
        # embedding alias ids (HF hashes the placeholder like any other token)
        self.image_token_id = self.read_cfg(int, "image_token_id", None)

        # Gated-residual hyper-connections
        self.hc_mult = self.read_cfg(int, pfx("hc_count"), 4)

        # QSA indexer
        self.indexer_n_heads = self.read_cfg(int, pfx("indexer_n_heads"), no_default)
        self.indexer_kv_heads = self.read_cfg(int, pfx("indexer_kv_heads"), 1)
        self.indexer_head_dim = self.read_cfg(int, pfx("indexer_head_dim"), no_default)
        self.indexer_budget = self.read_cfg(int, pfx("indexer_budget"), no_default)
        self.indexer_compress_ratio = self.read_cfg(int, pfx("indexer_compress_ratio"), 4)

        # GDN output gate
        self.output_gate_type = self.read_cfg(str, pfx("output_gate_type"), "silu")

        # PLE / n-gram embedding
        self.ple_layer_ids = self.read_cfg(list, pfx("ple_layer_ids"), [])   # 1-based
        self.ple_embed_dim = self.read_cfg(int, pfx("ple_embed_dim"), no_default)
        self.ple_conv_kernel_size = self.read_cfg(int, pfx("ple_conv_kernel_size"), 4)
        self.ngram_size = self.read_cfg(int, pfx("ngram_size"), no_default)
        self.heads_per_ngram = self.read_cfg(int, pfx("heads_per_ngram"), no_default)
        self.ple_eos_token_id = self.read_cfg(int, pfx("eos_token_id"), no_default)


def build_qwen4_block(
    config: Qwen4ExpConfig,
    key: str,
    layer_idx: int,
    layer_type: str,
    qbits_key: str = "bits",
) -> TransformerBlock:
    """One qwen4_exp decoder block (shared by the trunk and the MTP head): gated-residual
    hyper-connection sites, GDN (sigmoid-gated norm) or QSA full attention, MoE + shared expert."""
    def _hc(site: str):
        return GatedResidual(
            config = config,
            key = f"{key}.{site}",
            hc_mult = config.hc_mult,
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
        )

    return TransformerBlock(
        config = config,
        key = key,
        layer_idx = layer_idx,
        qbits_key = qbits_key,
        attn_hc = _hc("attn_hyper_connection"),
        mlp_hc = _hc("mlp_hyper_connection"),
        attn = (
            GatedDeltaNet(
                config = config,
                key = f"{key}.linear_attn",
                layer_idx = layer_idx,
                hidden_size = config.hidden_size,
                k_head_dim = config.linear_key_head_dim,
                v_head_dim = config.linear_value_head_dim,
                num_k_heads = config.linear_num_key_heads,
                num_v_heads = config.linear_num_value_heads,
                rms_norm_eps = config.rms_norm_eps,
                conv_kernel_size = config.linear_conv_kernel_dim,
                key_a_log = "A_log",
                key_dt_bias = "dt_bias",
                key_conv1d = "conv1d",
                key_qkv = "in_proj_qkv",
                key_z = "in_proj_z",
                key_b = "in_proj_b",
                key_a = "in_proj_a",
                key_norm = "norm",
                key_o = "out_proj",
                norm = GatedRMSNorm(
                    config, f"{key}.linear_attn.norm",
                    config.rms_norm_eps, out_dtype = torch.half,
                    gate_activation = config.output_gate_type,
                ),
                qmap = "block.attn",
                out_dtype = torch.float,
                select_hq_bits = 2,
            )
            if layer_type == "linear_attention" else
            Attention(
                config = config,
                key = f"{key}.self_attn",
                layer_idx = layer_idx,
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
                qbits_key = qbits_key,
                q_norm = RMSNorm(
                    config = config,
                    key = f"{key}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"{key}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                out_dtype = torch.float,
                interleaved_gate = True,
                select_hq_bits = 2,
                qsa_indexer = QSAIndexer(
                    config = config,
                    key = f"{key}.self_attn.indexer",
                    hidden_size = config.hidden_size,
                    n_heads = config.indexer_n_heads,
                    kv_heads = config.indexer_kv_heads,
                    head_dim = config.indexer_head_dim,
                    token_budget = config.indexer_budget,
                    compress_ratio = config.indexer_compress_ratio,
                    rms_norm_eps = config.rms_norm_eps,
                    qmap = "block.attn",
                    qbits_key = qbits_key,
                ),
            )
        ),
        mlp = BlockSparseMLP(
            config = config,
            key = f"{key}.mlp",
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
            qbits_key = qbits_key,
            interm_dtype = torch.half,
            out_dtype = torch.float,
            shared_experts = GatedMLP(
                config = config,
                key = f"{key}.mlp.shared_expert",
                hidden_size = config.hidden_size,
                intermediate_size = config.shared_expert_intermediate_size,
                key_up = "up_proj",
                key_gate = "gate_proj",
                key_down = "down_proj",
                qmap = "block.mlp",
                qbits_key = qbits_key,
                interm_dtype = torch.half,
                out_dtype = torch.float,
                select_hq_bits = 2,
            )
        ),
    )


class Qwen4ExpModel(Model):
    config_class = Qwen4ExpConfig

    def __init__(
        self,
        config: Qwen4ExpConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        key_prefix = "model.language_model"

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
            ),
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            if (idx + 1) in config.ple_layer_ids:
                self.modules += [
                    PLELayer(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.ple",
                        layer_idx = -(idx + 1),
                        hidden_size = config.hidden_size,
                        hc_mult = config.hc_mult,
                        ple_embed_dim = config.ple_embed_dim,
                        ngram_size = config.ngram_size,
                        heads_per_ngram = config.heads_per_ngram,
                        eos_token_id = config.ple_eos_token_id,
                        conv_kernel_size = config.ple_conv_kernel_size,
                        rms_norm_eps = config.rms_norm_eps,
                        mm_token_id = config.image_token_id,
                    )
                ]

            self.modules += [
                build_qwen4_block(
                    config,
                    f"{key_prefix}.layers.{idx}",
                    idx,
                    config.layer_types[idx],
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        # No final model norm: the combine-less mixer collapses the stream stack
        self.modules += [
            GatedResidual(
                config = config,
                key = f"{key_prefix}.hyper_connection_mixer",
                hc_mult = config.hc_mult,
                hidden_size = config.hidden_size,
                rms_norm_eps = config.rms_norm_eps,
                use_combine = False,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
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
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        p += f"<|im_start|>user\n{prompt}<|im_end|>\n"
        p += f"<|im_start|>assistant\n"
        return p
