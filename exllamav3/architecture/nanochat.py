from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    MLP,
    Linear,
    ValueEmbeddings,
)
from ..modules.attn import prepare_for_attn
from ..util.tensor import to2

class NanoChatConfig(Config):
    arch_string = "NanoChatForCausalLM"

    def __init__(
        self,
        directory: str,
        derived_model: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            derived_model if derived_model else {"text": NanoChatModel},
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
        self.assert_cfg(str, "hidden_act", "relu2", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NANOCHAT)

        # Output softcap
        self.final_logit_softcapping = self.read_cfg(float, "final_logit_softcapping", 0.0)

        # Auto-detect key format: native (transformer.h.*) vs HF (model.layers.*)
        self.native_keys = self.stc.has_tensor("transformer.wte.weight")

        # Value embeddings on alternating (odd) layers
        self.ve_gate_channels = self.read_cfg(int, "ve_gate_channels", self.num_kv_heads)
        assert self.ve_gate_channels == self.num_kv_heads, \
            "Expected ve_gate_channels to match the number of key/value heads"

        self.has_ve = self.stc.has_tensor("value_embeds.1.weight")

        # Per-layer residual scalars
        self.has_resid = self.stc.has_tensor("resid_lambdas")

        # Backout mechanism
        self.has_backout = self.stc.has_tensor("backout_lambda")


class NanoChatModel(Model):
    config_class = NanoChatConfig

    def __init__(
        self,
        config: NanoChatConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        # Key names: native (transformer.h.*) vs HF (model.layers.*)
        if config.native_keys:
            emb_key = "transformer.wte"
            layer_prefix = "transformer.h"
            kq, kk, kv, ko = "c_q", "c_k", "c_v", "c_proj"
            kup, kdown = "c_fc", "c_proj"
            kattn = "attn"
        else:
            emb_key = "model.embed_tokens"
            layer_prefix = "model.layers"
            kq, kk, kv, ko = "q_proj", "k_proj", "v_proj", "o_proj"
            kup, kdown = "fc1", "fc2"
            kattn = "self_attn"

        # Embedding + norm
        self.modules += [
            Embedding(
                config = config,
                key = emb_key,
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            ),
            RMSNorm(
                config = config,
                key = "_emb_norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.float,
                unweighted = True,
            ),
        ]

        # Residual scalars, save unprocessed tensors below
        if config.has_resid:
            resid_lambdas = config.stc.get_tensor("resid_lambdas", device = "cpu", no_defer = True).float().tolist()
            x0_lambdas = config.stc.get_tensor("x0_lambdas", device = "cpu", no_defer = True).float().tolist()
            assert len(resid_lambdas) == len(x0_lambdas) == config.num_hidden_layers

        if config.has_backout:
            backout_lambda = config.stc.get_tensor("backout_lambda", device = "cpu", no_defer = True).float().tolist()

        # Value embeddings: single CPU module, all lookups at once
        if config.has_ve:
            ve_layers = [2*i + 1 for i in range(config.num_hidden_layers // 2)]
            self.ve_module = ValueEmbeddings(
                config = config,
                key = "value_embeds",
                target_layers = ve_layers,
                vocab_size = config.vocab_size,
                kv_head_dim = config.head_dim,
                num_kv_heads = config.num_kv_heads,
            )
            self.modules += [self.ve_module]
        else:
            ve_layers = []
            self.ve_module = None

        self.first_block_idx = len(self.modules)

        # Transformer blocks
        for idx in range(config.num_hidden_layers):

            ve_gate = Linear(
                config,
                f"{layer_prefix}.{idx}.{kattn}.ve_gate",
                config.num_kv_heads,
                config.num_kv_heads,
                qmap = None,
                out_dtype = torch.half,
                pad_to = 1
            ) if idx in ve_layers else None

            block = TransformerBlock(
                config = config,
                key = f"{layer_prefix}.{idx}",
                ve_gate = ve_gate,
                layer_idx = idx,
                resid_lambda = resid_lambdas[idx] if config.has_resid else None,
                x0_lambda = x0_lambdas[idx] if config.has_resid else None,
                backout_extract = idx == config.num_hidden_layers // 2,
                backout_lambda = backout_lambda[0] if config.has_backout and (idx == config.num_hidden_layers - 1) else None,
                attn_norm = RMSNorm(
                    config = config,
                    key = f"{layer_prefix}.{idx}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    unweighted = True,
                ),
                attn = Attention(
                    config = config,
                    key = f"{layer_prefix}.{idx}.{kattn}",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_kv_heads,
                    rope_settings = config.rope_settings,
                    sm_scale = None,
                    key_q = kq,
                    key_k = kk,
                    key_v = kv,
                    key_o = ko,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    post_rope_norm = True,
                    ve_gate = idx in ve_layers,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{layer_prefix}.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    unweighted = True,
                ),
                mlp = MLP(
                    config = config,
                    key = f"{layer_prefix}.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = kup,
                    key_down = kdown,
                    qmap = "block.mlp",
                    out_dtype = torch.float,
                    interm_dtype = torch.float,
                    activation_fn = "relu2",
                    interm_scale = 50.0,
                ),
                out_dtype = torch.float,
            )
            self.modules += [block]

            # Give ValueEmbeddings a reference to the current attn layer so device can be known at inference time
            if self.ve_module and idx in ve_layers:
                self.ve_module.forward_ref[idx] = block.ve_gate

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = emb_key

        self.modules += [
            RMSNorm(
                config = config,
                key = "model.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
                unweighted = True,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True},
                softcap = config.final_logit_softcapping
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # TP for this architecture is not implemented yet
        self.caps.update({"supports_tp": False})

        # Interrupted quantization cannot resume because this model uses Spaghetti Attention
        self.caps.update({"can_resume_quant": False})


    @staticmethod
    @override
    def get_additional_compiled_tensors(config: NanoChatConfig) -> dict:
        extra = {}
        for k in ["resid_lambdas", "x0_lambdas", "backout_lambda"]:
            e = config.stc.get_tensor_meta(k)
            if e: extra.update(e)
        return extra

    @override
    def prepare_inputs(self, input_ids, params):
        input_ids = prepare_for_attn(input_ids, params)
        params["input_ids"] = input_ids
        return input_ids

    def per_layer_quant_preamble(self, params: dict):
        if self.ve_module and self.ve_module.device is not None:
            self.ve_module.forward(None, params)

    @override
    def default_chat_prompt(self, prompt, system_prompt = None):
        p = "<|bos|>"
        if system_prompt:
            p += system_prompt + "\n\n"
        p += "User: " + prompt + "\n\n"
        p += "Assistant:"
        return p
