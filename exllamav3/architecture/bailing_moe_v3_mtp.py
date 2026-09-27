from __future__ import annotations
import weakref
import torch
from typing_extensions import override

from ..model.model import Model
from ..modules import Embedding, Linear, RMSNorm, TransformerBlock
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn
from .bailing_moe_v3 import bailing_mla, bailing_mlp


class BailingMoeV3MTPBlock(TransformerBlock):
    """Compile only block children, not the sibling MTP input and final norm."""

    @override
    def get_compile_sizes(self, stc):
        return [size for module in self.modules for size in module.get_compile_sizes(stc)]

    @override
    def get_compile_tensors(self, stc):
        tensors = {}
        for module in self.modules:
            tensors.update(module.get_compile_tensors(stc))
        return tensors


class BailingMoeV3MTPModel(Model):
    """Ling next-token component; target post-norm state, embedding-first merge."""

    def __init__(self, config, key_prefix: str = "model", **kwargs):
        super().__init__(config, **kwargs)
        if config.num_mtp_layers != 1:
            raise ValueError("BailingMoeV3 MTP requires exactly one prediction layer")
        key = f"{key_prefix}.layers.{config.num_hidden_layers}"
        self.input_layer = Qwen3_5MTPInputLayer(
            config = config, key = f"{key}.input",
            key_pre_fc_norm_hidden = f"{key}.hnorm",
            key_pre_fc_norm_embedding = f"{key}.enorm", key_fc = f"{key}.eh_proj",
            hidden_size = config.hidden_size, rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1, out_dtype = torch.float,
            qbits_key = "mtp_bits", constant_bias = 0.0,
        )
        self.modules = [self.input_layer]
        self.first_block_idx = len(self.modules)
        self.modules.append(BailingMoeV3MTPBlock(
            config = config, key = key, layer_idx = 0,
            attn_norm = RMSNorm(config, f"{key}.input_layernorm", config.rms_norm_eps),
            attn = bailing_mla(config, f"{key}.attention", 0, qbits_key = "mtp_bits"),
            mlp_norm = RMSNorm(config, f"{key}.post_attention_layernorm", config.rms_norm_eps),
            mlp = bailing_mlp(config, f"{key}.mlp", config.num_hidden_layers, qbits_key = "mtp_bits"),
        ))
        self.last_kv_module_idx = len(self.modules) - 1
        self.final_norm = RMSNorm(
            config, f"{key}.final_layernorm", config.rms_norm_eps, out_dtype = torch.half,
        )
        self.modules.append(self.final_norm)
        self.caps.update({
            "supports_tp": False, "attach_target": True, "mtp_draft": True,
            "default_draft_size": 1, "autosplit_load_fwd": False,
        })
        self.calibration_all_experts = True
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP uses the target's chat template")

    def attach_to(self, target):
        if target.config is not self.config:
            raise ValueError("Ling target and MTP must share the same Config")
        if not isinstance(target.modules[0], Embedding) or not isinstance(target.modules[-1], Linear):
            raise ValueError("Expected target embedding and output head")
        target_norm = target.modules[target.logit_layer_idx - 1]
        if not isinstance(target_norm, RMSNorm):
            raise ValueError("MTP requires the target post-final-norm hidden state")
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)
        self.target_embed = weakref.ref(target.modules[0])
        self.target_lm_head = weakref.ref(target.modules[-1])
        self.draft_verifier_params = {"export_state_norm_keys": {target_norm.key}}

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, chunk_size):
        return {}

    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        if self.target_lm_head is None or self.target_lm_head() is None:
            raise RuntimeError("Attach the MTP component to its target before inference")
        # The generator supplies the tokenizer domain; physical head rows may include
        # unmapped tokens as well as linear padding. Manual callers must supply it too.
        output_vocab_size = params.get("output_vocab_size")
        if type(output_vocab_size) is not int or not 0 < output_vocab_size <= self.config.vocab_size:
            raise ValueError("Ling MTP requires output_vocab_size from tokenizer.actual_vocab_size "
                             "within the model vocabulary")
        head = self.target_lm_head()
        logits = head.forward(head.prepare_for_device(state, params), params)
        if logits.shape[-1] < output_vocab_size:
            raise ValueError("Ling MTP output head is smaller than output_vocab_size")
        logits = logits[..., :output_vocab_size]
        confidence, ids = logits.max(dim = -1)
        if params.get("export_draft_conf"):
            params["draft_conf"] = confidence
        return ids
