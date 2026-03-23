from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, MLP, Linear, Module
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
        self.ve_gate_channels = self.read_cfg(int, "ve_gate_channels", 12)
        self.has_ve = self.stc.has_tensor("value_embeds.1.weight")

        # Per-layer residual scalars
        self.has_resid = self.stc.has_tensor("resid_lambdas")

        # Backout mechanism
        self.has_backout = self.stc.has_tensor("backout_lambda")


def _has_ve(layer_idx):
    return layer_idx % 2 == 1


class ValueEmbeddings(Module):
    """CPU-resident value embeddings. Does all VE lookups at once during forward,
    producing a (num_ve_layers, B, T, kv_dim) tensor stored in params.
    Each attention block indexes its slice. No per-layer CPU-GPU sync."""

    def __init__(self, config, num_layers, kv_dim, ve_gate_channels):
        super().__init__(config, "value_embeds", None)
        self.num_layers = num_layers
        self.kv_dim = kv_dim
        self.ve_gate_channels = ve_gate_channels
        self.ve_layers = {}   # layer_idx -> weight tensor (CPU)
        self.ve_gates = {}    # layer_idx -> gate weight tensor (GPU)

    @override
    def load(self, device, **kwargs):
        stc = self.config.stc
        native = self.config.native_keys
        for i in range(self.num_layers):
            if not _has_ve(i):
                continue
            ve_key = f"value_embeds.{i}.weight"
            if stc.has_tensor(ve_key):
                self.ve_layers[i] = stc.get_tensor(ve_key, torch.device("cpu"))
            vg_key = f"transformer.h.{i}.attn.ve_gate.weight" if native else f"model.layers.{i}.self_attn.ve_gate.weight"
            if stc.has_tensor(vg_key):
                self.ve_gates[i] = stc.get_tensor(vg_key, device)

    @override
    def forward(self, x, params, out_dtype = None):
        input_ids = params.get("_nc_input_ids")
        if input_ids is None or not self.ve_layers:
            return x
        # Batch all VE lookups on CPU, move result to GPU once
        ids_cpu = input_ids.cpu()
        ve_all = {}
        for layer_idx, weight in self.ve_layers.items():
            ve_all[layer_idx] = F.embedding(ids_cpu, weight)
        # Stack and move to GPU in one transfer
        if ve_all:
            params["_nc_ve_all"] = {k: v.to(device = x.device, dtype = x.dtype) for k, v in ve_all.items()}
            params["_nc_ve_gates"] = self.ve_gates
        return x

    @override
    def optimizer_targets(self):
        return []


class ResidualScalars(Module):
    """Load resid_lambdas and x0_lambdas, store in params for blocks."""

    def __init__(self, config):
        super().__init__(config, "resid_scalars", None)
        self.resid_lambdas = None
        self.x0_lambdas = None

    @override
    def load(self, device, **kwargs):
        stc = self.config.stc
        if stc.has_tensor("resid_lambdas"):
            self.resid_lambdas = stc.get_tensor("resid_lambdas", device)
            self.x0_lambdas = stc.get_tensor("x0_lambdas", device)

    @override
    def forward(self, x, params, out_dtype = None):
        params["_nc_x0"] = x.clone()
        params["_nc_resid_lambdas"] = self.resid_lambdas
        params["_nc_x0_lambdas"] = self.x0_lambdas
        return x

    @override
    def optimizer_targets(self):
        return []


class ApplyBackout(Module):
    """Apply backout: x = x - backout_lambda * x_mid."""

    def __init__(self, config):
        super().__init__(config, "apply_backout", None)
        self.backout_lambda = None

    @override
    def load(self, device, **kwargs):
        if self.config.stc.has_tensor("backout_lambda"):
            self.backout_lambda = self.config.stc.get_tensor("backout_lambda", device)

    @override
    def forward(self, x, params, out_dtype = None):
        x_backout = params.pop("_nc_x_backout", None)
        if x_backout is not None and self.backout_lambda is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return x

    @override
    def optimizer_targets(self):
        return []


class NanoChatBlock(Module):
    """Transformer block with residual scaling, VE injection, and backout state."""

    def __init__(
        self,
        config,
        key,
        layer_idx,
        num_layers,
        attn_norm,
        attn,
        mlp_norm,
        mlp,
        has_ve = False,
        out_dtype = None,
    ):
        super().__init__(config, key, None)
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.attn_norm = attn_norm
        self.attn = attn
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.out_dtype = out_dtype
        self._has_ve = has_ve

        self.register_submodule(self.attn_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.mlp_norm)
        self.register_submodule(self.mlp)

        self.num_slices = mlp.num_slices if mlp else 1

    @override
    def forward(self, x, params, out_dtype = None):

        # Residual scaling
        resid_lambdas = params.get("_nc_resid_lambdas")
        if resid_lambdas is not None:
            x0 = params.get("_nc_x0")
            x0_lambdas = params.get("_nc_x0_lambdas")
            if x0 is not None:
                rl = resid_lambdas[self.layer_idx].to(x.dtype)
                xl = x0_lambdas[self.layer_idx].to(x.dtype)
                x = rl * x + xl * x0

        # Store mid-layer for backout
        if self.layer_idx == self.num_layers // 2:
            params["_nc_x_backout"] = x.clone()

        # VE: get pre-computed slice from params, apply gate
        if self._has_ve:
            ve_all = params.get("_nc_ve_all")
            ve_gates = params.get("_nc_ve_gates")
            if ve_all and self.layer_idx in ve_all and ve_gates and self.layer_idx in ve_gates:
                ve = ve_all[self.layer_idx]
                gate_w = ve_gates[self.layer_idx]
                gate = 3 * torch.sigmoid(F.linear(x[..., :gate_w.shape[1]], gate_w))
                kv_dim = ve.shape[-1]
                n_kv_head = gate.shape[-1]
                head_dim = kv_dim // n_kv_head
                params["attn_v_addend"] = (gate.unsqueeze(-1) * ve.view(*ve.shape[:-1], n_kv_head, head_dim)).reshape(ve.shape)
            else:
                params.pop("attn_v_addend", None)
        else:
            params.pop("attn_v_addend", None)

        # Attention
        if self.attn:
            y = self.attn_norm.forward(x, params, out_dtype = torch.half) if self.attn_norm else x.half()
            y = self.attn.forward(y, params)
            if params.get("prefill"):
                return x
            x = x + y

        # MLP
        if self.mlp:
            y = self.mlp_norm.forward(x, params, out_dtype = torch.half) if self.mlp_norm else x.half()
            y = self.mlp.forward(y, params)
            x = x + y

        return to2(x, out_dtype, self.out_dtype)

    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]

    def allocate_q(self, quant_args, surplus_bits):
        from ..conversion.allocation import allocate_transformer
        q = self.attn.q_proj if self.attn else None
        k = self.attn.k_proj if self.attn else None
        v = self.attn.v_proj if self.attn else None
        o = self.attn.o_proj if self.attn else None
        u = self.mlp.ups if self.mlp else None
        d = self.mlp.downs if self.mlp else None
        return allocate_transformer(quant_args.get("bits", 4), surplus_bits, q, k, v, o, None, u, d, None)


class NanoChatModel(Model):
    config_class = NanoChatConfig

    def __init__(
        self,
        config: NanoChatConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        kv_dim = config.num_kv_heads * config.head_dim
        native = config.native_keys

        # Key names: native (transformer.h.*) vs HF (model.layers.*)
        emb_key = "transformer.wte" if native else "model.embed_tokens"
        def lk(idx, suffix):
            prefix = f"transformer.h.{idx}" if native else f"model.layers.{idx}"
            return f"{prefix}.{suffix}"
        kq, kk, kv, ko = ("c_q", "c_k", "c_v", "c_proj") if native else ("q_proj", "k_proj", "v_proj", "o_proj")
        kup, kdown = ("c_fc", "c_proj") if native else ("fc1", "fc2")

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

        # Residual scalars (stores x0 + lambdas in params)
        if config.has_resid:
            self.modules.append(ResidualScalars(config))

        # Value embeddings: single CPU module, all lookups at once
        if config.has_ve:
            self.modules.append(ValueEmbeddings(config, config.num_hidden_layers, kv_dim, config.ve_gate_channels))

        self.first_block_idx = len(self.modules)

        # Transformer blocks
        self.modules += [
            NanoChatBlock(
                config = config,
                key = f"transformer.h.{idx}" if native else f"model.layers.{idx}",
                layer_idx = idx,
                num_layers = config.num_hidden_layers,
                attn_norm = RMSNorm(
                    config = config,
                    key = f"_norm.{idx}.attn",
                    rms_norm_eps = config.rms_norm_eps,
                    unweighted = True,
                ),
                attn = Attention(
                    config = config,
                    key = lk(idx, "attn") if native else lk(idx, "self_attn"),
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
                    post_rope_norm = True
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"_norm.{idx}.mlp",
                    rms_norm_eps = config.rms_norm_eps,
                    unweighted = True,
                ),
                mlp = MLP(
                    config = config,
                    key = lk(idx, "mlp"),
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
                has_ve = _has_ve(idx) and config.has_ve,
                out_dtype = torch.float,
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        # Backout (before final norm)
        if config.has_backout:
            self.modules.append(ApplyBackout(config))

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

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: NanoChatConfig) -> dict:
        """Return VE, scalar, and backout tensors so the quantizer preserves them."""
        extra = config.stc.list_tensors(prefix = "value_embeds")
        # VE gates
        native = config.native_keys
        for i in range(config.num_hidden_layers):
            if _has_ve(i):
                prefix = f"transformer.h.{i}.attn.ve_gate" if native else f"model.layers.{i}.self_attn.ve_gate"
                extra.update(config.stc.list_tensors(prefix = prefix))
        # Scalar tensors (exact keys, no "." children - list_tensors doesn't work for these)
        for key in ["resid_lambdas", "x0_lambdas", "backout_lambda"]:
            if config.stc.has_tensor(key):
                # Build metadata manually since list_tensors fails on leaf keys
                filename = config.stc.tensor_file_map[key]
                h = config.stc.file_headers[filename][key]
                beg, end = h["data_offsets"]
                extra[key] = {"shape": h["shape"], "n_bytes": end - beg, "dtype": h["dtype"]}
        return extra

    @override
    def prepare_inputs(self, input_ids, params):
        input_ids = prepare_for_attn(input_ids, params)
        params["_nc_input_ids"] = input_ids
        return input_ids

    @override
    def default_chat_prompt(self, prompt, system_prompt = None):
        p = "<|bos|>"
        if system_prompt:
            p += system_prompt + "\n\n"
        p += "User: " + prompt + "\n\n"
        p += "Assistant:"
        return p
