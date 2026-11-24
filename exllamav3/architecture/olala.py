from __future__ import annotations

import torch
import torch.nn.functional as F
from typing_extensions import override

from ..cache.recurrent_util import prepare_for_recurrence
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Embedding, Linear
from ..modules.arch_specific.olala import OlalaBlock
from ..modules.attn import prepare_for_attn
from ..modules.recurrent import CommittedRecurrentState


class OlalaConfig(Config):
    arch_string = "OlalaForCausalLM"

    def __init__(self, directory, **kwargs):
        super().__init__(directory, {"text": OlalaModel}, **kwargs)
        self.layer_types = self.read_cfg(str, "layers_config", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        if len(self.layer_types) != self.num_hidden_layers or set(self.layer_types) != {"M", "V"}:
            raise ValueError("Olala requires a layer schedule of Mamba-3 (M) and differential attention (V)")

        enabled = (
            "geodesic_update", "normalize_embeddings_ngpt", "qk_norm", "scalable_softmax",
            "token_shift_attn", "gate_attn", "zero_centered_gate", "zero_centered_gamma",
            "use_completed_p", "moe", "mamba3_add_trapezoid", "mamba3_is_A_dd",
            "mamba3_is_id_rms", "mamba3_remove_conv", "mamba3_rope",
        )
        disabled = (
            "cosnet", "final_norm", "normalize_embeddings", "normalize_lm_head", "patch_level_training",
            "reduce_lm_head", "tie_lm_head", "tie_word_embeddings", "logits_scaling_ngpt", "mixer_gn",
            "use_value_embedding", "mamba3_postgate_norm", "mamba3_remove_BC_bias", "complete_slw",
            "intra_doc_masking", "layer_norm_scaling", "moe_shared_expert_gate", "token_conv1d_attn",
            "xsa", "ngram_embeddings", "mlp_linking", "resformer", "vwn",
        )
        for name in enabled:
            self.assert_cfg(bool, name, True)
        for name in disabled:
            self.assert_cfg(bool, name, False, optional = True)
        self.assert_cfg(str, "gate_type", "elementwise")
        self.assert_cfg(str, "gate_act", "silu")
        self.assert_cfg(str, "normalization_type", "rmsnorm")
        self.assert_cfg(str, "mlp_type", "simple")
        self.assert_cfg(str, "rope_type", "")
        self.assert_cfg(int, "mamba_ngroups", 1)
        for name in ("skip_mixer_layers", "skip_moe_layers", "layers_stem_config"):
            if self.config_dict.get(name):
                raise ValueError(f"Unsupported Olala config field: {name}")
        if self.read_cfg(str, "layers_mlp_config", "") not in ("", "m" * self.num_hidden_layers):
            raise ValueError("Olala requires a mixture-of-experts MLP in every layer")

        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.head_dim = self.read_cfg(int, "head_dim", no_default)
        self.num_attention_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.signal_heads = self.read_cfg(int, "num_signal_heads_diff", no_default)
        noise_heads = self.num_attention_heads - self.signal_heads
        if noise_heads <= 0 or self.signal_heads <= 0 or self.signal_heads % noise_heads:
            raise ValueError("Olala differential attention requires equal groups of signal heads plus one noise head")
        self.attention_rank = self.read_cfg(int, "tpa_rank", no_default)
        self.sliding_window = self.read_cfg(int, "slw_wsize", no_default)
        self.attention_softcap = self.read_cfg(float, "softcap_attn", no_default)
        self.mamba_head_dim = self.read_cfg(int, "mamba_headdim", no_default)
        self.mamba_state_size = self.read_cfg(int, "mamba_d_state", no_default)
        self.mamba_rank = self.read_cfg(int, "mamba_mimo_dim", no_default)
        if (self.mamba_head_dim <= 0 or self.mamba_state_size <= 0 or self.mamba_rank <= 0
                or 2 * self.hidden_size % self.mamba_head_dim or self.mamba_state_size % 4):
            raise ValueError("Unsupported Olala Mamba-3 head or rotary geometry")
        self.num_experts = self.read_cfg(int, "moe_num_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "moe_num_active_experts", no_default)
        self.expert_size = self.read_cfg(int, "moe_routed_intermediate_size", no_default)
        self.latent_size = self.read_cfg(int, "moe_routed_input_dim", no_default)
        self.shared_size = self.read_cfg(int, "moe_shared_intermediate_size", no_default)
        self.expert_scale = self.read_cfg(float, "moe_routed_scaling_factor", no_default)
        self.norm_eps = self.read_cfg(float, "norm_epsilon", no_default)
        if not 1 < self.num_experts_per_tok <= self.num_experts or min(
            self.hidden_size, self.head_dim, self.attention_rank, self.sliding_window, self.attention_softcap,
            self.expert_size, self.latent_size, self.shared_size, self.norm_eps,
        ) <= 0:
            raise ValueError("Invalid Olala attention or mixture-of-experts configuration")
        self.max_position_embeddings = min(self.max_position_embeddings, 32768)


class OlalaEmbedding(Embedding):
    def forward(self, x, params, out_dtype = None):
        return F.normalize(super().forward(x, params, out_dtype = torch.float), dim = -1).to(
            out_dtype or torch.float,
        )


class OlalaHead(Linear):
    def forward(self, x, params, out_dtype = None):
        return super().forward(x.half(), params, out_dtype or torch.float)


class OlalaModel(Model):
    config_class = OlalaConfig

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.modules.append(OlalaEmbedding(
            config = config, key = "model.embedding", vocab_size = config.vocab_size,
            hidden_size = config.hidden_size,
        ))
        self.first_block_idx = len(self.modules)
        self.modules.extend(
            OlalaBlock(config, f"model.layers.{idx}", idx, layer_type)
            for idx, layer_type in enumerate(config.layer_types)
        )
        self.last_kv_module_idx = len(self.modules) - 1
        self.logit_layer_idx = len(self.modules)
        self.modules.append(OlalaHead(
            config = config, key = "lm_head", in_features = config.hidden_size,
            out_features = config.vocab_size, qmap = "block", qbits_key = "head_bits",
            caps = {"logits_output": True},
        ))
        self.calibration_all_experts = True
        self.recurrent_state_cls = CommittedRecurrentState
        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True,
            "supports_tp": False,
            "supports_speculative": False,
        })

    @override
    def prepare_inputs(self, input_ids, params):
        if params.get("recurrent_history"):
            raise ValueError("Olala does not support speculative recurrent history")
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids

    def default_chat_prompt(self, prompt, system_prompt = None):
        from .olala_template import render_olala_chat

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return render_olala_chat(self.config.directory, messages)
