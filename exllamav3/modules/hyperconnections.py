from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from .module import Module
from .rmsnorm import RMSNorm
from ..model.config import Config
from ..ext import exllamav3_ext as ext
from ..util.tensor import g_tensor_cache

# mHC (manifold-constrained hyper-connections, DeepSeek-V4): the residual is carried as
# hc_mult parallel fp32 streams shaped (bsz, seq, hc_mult, hidden). ExpandStreams broadcasts
# the embedding into the streams, each sublayer site mixes them through a HyperConnection
# (sigmoid pre/post weights + Sinkhorn-normalized combine matrix), and HyperHead collapses
# them before the final norm. TransformerBlock consumes HyperConnection via optional
# attn_hc/mlp_hc parameters.


class ExpandStreams(Module):
    """Broadcast the embedding into hc_mult parallel residual streams, fp32."""

    def __init__(self, config: Config, key: str, hc_mult: int):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        return x.float().unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()

    def tp_export(self, plan, producer):
        # Stateless stream broadcast; the residual (and its stream stack) is replicated
        return {
            "cls": ExpandStreams,
            "kwargs": {
                "key": self.key,
                "hc_mult": self.hc_mult,
            },
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        module = ExpandStreams(config = None, **exported["kwargs"])
        module.device = local_context["device"]
        return module


class HyperConnection(Module):
    """mHC mixer for one sublayer site. Owns raw fp32 tensors {key}_fn ((2 + H) * H rows,
    H * hidden cols), {key}_base, {key}_scale. Not a standalone graph module: TransformerBlock
    calls mix() around its attn/mlp sites."""

    def __init__(
        self,
        config: Config | None,
        key: str,                    # e.g. "layers.{idx}.hc_attn"; tensors at "{key}_fn" etc.
        hc_mult: int,
        hidden_size: int,
        sinkhorn_iters: int,
        hc_eps: float,
        rms_norm_eps: float,
    ):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.rms_eps = rms_norm_eps
        self.norm = RMSNorm(config, f"{key}.norm", rms_norm_eps, unweighted = True,
                            out_dtype = torch.float)
        self.register_submodule(self.norm)
        self.fn = None
        self.fn_h = None
        self.base = None
        self.scale = None

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.fn = stc.get_tensor(f"{self.key}_fn", device, no_defer = True).float().contiguous()
        self.base = stc.get_tensor(f"{self.key}_base", device, no_defer = True).float().contiguous()
        self.scale = stc.get_tensor(f"{self.key}_scale", device, no_defer = True).float().contiguous()

    @override
    def unload(self):
        super().unload()
        self.fn = self.fn_h = self.base = self.scale = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}_fn": self.fn.contiguous(),
            f"{self.key}_base": self.base.contiguous(),
            f"{self.key}_scale": self.scale.contiguous(),
        }

    @override
    def weights_numel(self):
        h = self.hc_mult
        return (2 * h + h * h) * (h * self.hidden_size + 1) + 3

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        raise RuntimeError("HyperConnection is not a standalone module; use mix()")

    def mix(self, streams: torch.Tensor, params: dict):
        """streams (b, s, H, D) fp32 -> (post (b,s,H), comb (b,s,H,H), collapsed (b,s,D)).
        Fused ext path (2 kernel launches, see benchmarks/hc_mix/) returns collapsed as HALF
        (both block consumers cast it immediately); the torch fallback keeps fp32."""
        hc = self.hc_mult
        b, s, H, D = streams.shape
        if hc == 4 and streams.dtype == torch.float and D % 4 == 0 and streams.is_contiguous():
            R = b * s
            st = streams.view(R, H, D)
            chunks = ext.hc_mix_num_chunks(R, H * D)
            partials = g_tensor_cache.get(streams.device, (R, chunks, 2 * H + H * H + 1),
                                          torch.float, "hc_mix_partials")
            post = torch.empty((R, H), dtype = torch.float, device = streams.device)
            comb = torch.empty((R, H, H), dtype = torch.float, device = streams.device)
            collapsed = torch.empty((R, D), dtype = torch.half, device = streams.device)
            # Small R (decode): fn in fp16 -- the (M, H * D) matrix is the partials kernel's
            # dominant traffic and the kernel dots it in fp32 either way
            if R <= 32:
                if self.fn_h is None:
                    self.fn_h = self.fn.half()
                fn = self.fn_h
            else:
                fn = self.fn
            ext.hc_mix(st, fn, self.base, self.scale, self.rms_eps, self.hc_eps,
                       self.sinkhorn_iters, partials, post, comb, collapsed)
            return post.view(b, s, H), comb.view(b, s, H, H), collapsed.view(b, s, D)
        flat = self.norm.forward(streams.flatten(2), params)
        mix = F.linear(flat, self.fn)
        pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim = -1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_s, post_s, comb_s = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_s + pre_b) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
        comb = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
        comb = torch.softmax(comb, dim = -1) + self.hc_eps
        comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim = -1, keepdim = True) + self.hc_eps)
            comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * streams).sum(dim = 2)
        return post, comb, collapsed

    def apply_(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        params: dict
    ):
        """Residual update for one sublayer site: x <- post ⊗ y + combᵀ x. Fused ext path
        updates x IN PLACE (each output column depends only on the same column of the H
        stream rows); the torch fallback allocates. Conversion must NOT run the in-place
        path: the capture and advance passes forward the SAME stored input states twice."""
        b, s, H, D = x.shape
        converting = "quant_preserve" in params or "capture" in params
        if not converting and H == 4 and x.dtype == torch.float and x.is_contiguous() and D % 4 == 0 \
                and y.dtype in (torch.float, torch.half) and y.is_contiguous() \
                and post.dtype == torch.float and post.is_contiguous() and comb.is_contiguous():
            R = b * s
            ext.hc_apply(x.view(R, H, D), y.view(R, D), post.view(R, H), comb.view(R, H, H))
            return x
        return post.unsqueeze(-1) * y.float().unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), x)

    def tp_export(self, plan, producer):
        # Streams are replicated across TP workers (like the residual), so plain replication
        return {
            "cls": HyperConnection,
            "kwargs": {
                "key": self.key,
                "hc_mult": self.hc_mult,
                "hidden_size": self.hidden_size,
                "sinkhorn_iters": self.sinkhorn_iters,
                "hc_eps": self.hc_eps,
                "rms_norm_eps": self.rms_eps,
            },
            "fn": producer.send(self.fn),
            "base": producer.send(self.base),
            "scale": producer.send(self.scale),
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = HyperConnection(config = None, **exported["kwargs"])
        module.fn = consumer.recv(exported["fn"], cuda = True)
        module.base = consumer.recv(exported["base"], cuda = True)
        module.scale = consumer.recv(exported["scale"], cuda = True)
        module.device = local_context["device"]
        return module


class HyperHead(Module):
    """Final mHC stream collapse before the model norm. Top-level raw tensors {key}_fn etc."""

    def __init__(self, config: Config, key: str, hc_mult: int, rms_norm_eps: float, hc_eps: float):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult
        self.rms_eps = rms_norm_eps
        self.hc_eps = hc_eps
        self.norm = RMSNorm(config, f"{key}.norm", rms_norm_eps, unweighted = True,
                            out_dtype = torch.float)
        self.register_submodule(self.norm)
        self.fn = None
        self.fn_h = None
        self.base = None
        self.scale = None

    def _tensor_names(self):
        return [f"{self.key}_fn", f"{self.key}_base", f"{self.key}_scale"]

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.fn = stc.get_tensor(f"{self.key}_fn", device, no_defer = True).float().contiguous()
        self.base = stc.get_tensor(f"{self.key}_base", device, no_defer = True).float().contiguous()
        self.scale = stc.get_tensor(f"{self.key}_scale", device, no_defer = True).float().contiguous()

    @override
    def unload(self):
        super().unload()
        self.fn = self.fn_h = self.base = self.scale = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}_fn": self.fn,
            f"{self.key}_base": self.base,
            f"{self.key}_scale": self.scale,
        }

    # The compile step enumerates a module's output tensors by "{key}." prefix; this module's
    # tensor names are underscore-joined at the top level (hc_head_fn etc.), so the prefix trie
    # never matches them and they would be silently dropped from the compiled shards
    @override
    def get_compile_sizes(self, stc):
        return [stc.get_tensor_size(k) for k in self._tensor_names()]

    @override
    def get_compile_tensors(self, stc):
        return {k: stc.get_tensor(k, allow_bf16 = True) for k in self._tensor_names()}

    @override
    def optimizer_targets(self):
        return []

    def tp_export(self, plan, producer):
        # Stream collapse runs on the replicated stream stack: plain replication
        return {
            "cls": HyperHead,
            "kwargs": {
                "key": self.key,
                "hc_mult": self.hc_mult,
                "rms_norm_eps": self.rms_eps,
                "hc_eps": self.hc_eps,
            },
            "fn": producer.send(self.fn),
            "base": producer.send(self.base),
            "scale": producer.send(self.scale),
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = HyperHead(config = None, **exported["kwargs"])
        module.fn = consumer.recv(exported["fn"], cuda = True)
        module.base = consumer.recv(exported["base"], cuda = True)
        module.scale = consumer.recv(exported["scale"], cuda = True)
        module.device = local_context["device"]
        return module

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        b, s, H, D = x.shape
        if H == 4 and x.dtype == torch.float and D % 4 == 0 and x.is_contiguous():
            R = b * s
            chunks = ext.hc_mix_num_chunks(R, H * D)
            partials = g_tensor_cache.get(x.device, (R, chunks, H + 1),
                                          torch.float, "hc_head_partials")
            collapsed = torch.empty((R, D), dtype = torch.float, device = x.device)
            if R <= 32:
                if self.fn_h is None:
                    self.fn_h = self.fn.half()
                fn = self.fn_h
            else:
                fn = self.fn
            ext.hc_head(x.view(R, H, D), fn, self.base, self.scale,
                        self.rms_eps, self.hc_eps, partials, collapsed)
            return collapsed.view(b, s, D)
        flat = self.norm.forward(x.flatten(2), params)
        mixes = F.linear(flat, self.fn)
        pre = torch.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre.unsqueeze(-1) * x).sum(dim = 2)
