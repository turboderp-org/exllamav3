"""Text-only Gemma 4 E-series per-layer inputs and shared K/V attention."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ..module import Module
from ..embedding import Embedding
from ..linear import Linear
from ..rmsnorm import RMSNorm
from ..attn import Attention
from ..attention_fn.dispatch import attn_dispatch
from ...util.rope import RoPE
from ...util.tensor import get_for_device, to2


def preserve(params, key, value):
    params[key] = value
    if "quant_preserve" in params:
        # Calibration forwards one module at a time. Keep side inputs with each row,
        # off the GPU, alongside the ordinary hidden state.
        params["quant_preserve"][key] = value.detach().cpu()


class Gemma4PerLayerEmbedding(Embedding):
    def forward(self, x, params, out_dtype=None):
        ids = params["input_ids"].to(self.device)
        p = super().forward(ids, params, out_dtype=torch.float)
        preserve(params, "gemma4_ple", p)
        return x


class Gemma4PerLayerProjection(Module):
    def __init__(self, config, key):
        super().__init__(config, key + ".per_layer_inputs", None)
        self.num_layers = config.num_hidden_layers
        self.dim = config.hidden_size_per_layer_input
        self.scale = config.hidden_size ** -0.5
        self.proj = Linear(config, key + ".per_layer_model_projection",
                           config.hidden_size, self.num_layers * self.dim,
                           qmap="ple.input", out_dtype=torch.float)
        self.norm = RMSNorm(config, key + ".per_layer_projection_norm",
                            config.rms_norm_eps, out_dtype=torch.float)
        self.register_submodule(self.proj)
        self.register_submodule(self.norm)

    def optimizer_targets(self):
        return [self.proj.optimizer_targets()]

    def get_compile_sizes(self, stc):
        return [n for m in self.modules for n in m.get_compile_sizes(stc)]

    def get_compile_tensors(self, stc):
        return {k: v for m in self.modules for k, v in m.get_compile_tensors(stc).items()}

    def forward(self, x, params, out_dtype=None):
        p = self.proj.forward(x.half(), params) * self.scale
        p = p.view(*x.shape[:-1], self.num_layers, self.dim)
        p = self.norm.forward(p, params)
        e = params["gemma4_ple"].to(p.device).view_as(p)
        preserve(params, "gemma4_ple", ((p + e) * (2 ** -0.5)).half())
        return x


class Gemma4PerLayerInput(Module):
    def __init__(self, config, key, layer_idx):
        super().__init__(config, key + ".ple", None)
        self.ple_idx = layer_idx
        dim = config.hidden_size_per_layer_input
        self.gate = Linear(config, key + ".per_layer_input_gate",
                           config.hidden_size, dim, qmap="ple.gate", out_dtype=torch.half)
        self.proj = Linear(config, key + ".per_layer_projection",
                           dim, config.hidden_size, qmap="ple.proj", out_dtype=torch.float)
        self.norm = RMSNorm(config, key + ".post_per_layer_input_norm",
                            config.rms_norm_eps, out_dtype=torch.float)
        for m in (self.gate, self.proj, self.norm):
            self.register_submodule(m)

    def optimizer_targets(self):
        return [self.gate.optimizer_targets(), self.proj.optimizer_targets()]

    def forward(self, x, params, out_dtype=None):
        p = params["gemma4_ple"].to(x.device)[..., self.ple_idx, :]
        y = F.gelu(self.gate.forward(x.half(), params), approximate="tanh")
        y = self.proj.forward((y * p).contiguous(), params)
        self.norm.forward(y, params, residual=x)
        return x


class Gemma4Attention(Attention):
    store_shared_kv = False

    def publish_kv(self, k, v, params):
        if self.store_shared_kv:
            # Fused projections reuse scratch on the next layer: retain owned copies.
            preserve(params, f"gemma4_k_{self.layer_idx}", k.clone())
            preserve(params, f"gemma4_v_{self.layer_idx}", v.clone())

    def bc_attn_step(self, *args, **kwargs):
        if self.store_shared_kv:
            return None  # The fused path does not export its intermediate K/V yet.
        return super().bc_attn_step(*args, **kwargs)


class Gemma4SharedAttention(Module):
    def __init__(self, config, key, layer_idx, source_idx, head_dim, rope_settings, window):
        super().__init__(config, key, None)
        self.layer_idx = layer_idx
        self.source_idx = source_idx
        self.head_dim = head_dim
        self.num_q_heads = config.num_q_heads
        self.window = window
        self.rope_settings = rope_settings
        self.rope = None
        self.dispatch_cache = {}
        self.q_proj = Linear(config, key + ".q_proj", config.hidden_size,
                             config.num_q_heads * head_dim, qmap="block.attn.input")
        self.o_proj = Linear(config, key + ".o_proj", config.num_q_heads * head_dim,
                             config.hidden_size, qmap="block.attn.output")
        self.q_norm = RMSNorm(config, key + ".q_norm", config.rms_norm_eps,
                              out_dtype=torch.half)
        for m in (self.q_proj, self.o_proj, self.q_norm):
            self.register_submodule(m)

    def optimizer_targets(self):
        return [[self.q_proj.optimizer_targets(), self.o_proj.optimizer_targets()]]

    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        self.rope = RoPE(device, self.rope_settings)

    def unload(self):
        super().unload()
        self.rope = None
        self.dispatch_cache.clear()

    def forward(self, x, params, out_dtype=None):
        b, s, _ = x.shape
        q = self.q_proj.forward(x, params).view(b, s, self.num_q_heads, self.head_dim)
        q, _ = self.rope.apply(q, position=params.get("position", 0),
                              positions=get_for_device(params, "positions", self.device, None),
                              position_ids=get_for_device(params, "position_ids", self.device, None),
                              in_place=True, q_norm=self.q_norm.weight,
                              norm_eps=self.q_norm.rms_norm_eps)
        k = params[f"gemma4_k_{self.source_idx}"].to(x.device)
        v = params[f"gemma4_v_{self.source_idx}"].to(x.device)
        cached = params.get("attn_mode", "flash_attn_nc") == "flash_attn"
        o = attn_dispatch(q, k, v,
                          cache=params.get("cache") if cached else None,
                          cache_idx=self.source_idx,
                          cache_instance=params.get("layer_instance"),
                          block_table=get_for_device(params, "block_table", self.device) if cached else None,
                          cache_seqlens=get_for_device(params, "cache_seqlens", self.device) if cached else None,
                          sm_scale=1.0, window_size=self.window,
                          causal=params.get("causal", True), dispatch_cache=self.dispatch_cache)
        y = self.o_proj.forward(o.reshape(b, s, -1), params)
        return to2(y, out_dtype, None)
