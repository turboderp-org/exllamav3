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
            M1 = 2 * H + H * H + 1
            dev = streams.device
            # Decode-class row counts take the static workspaces (allocation latency matters and
            # the graphed callers rely on them); prefill chunks allocate per call so the static
            # cache holds only small buffers
            def ws(numel, dtype, tag):
                if R <= 32:
                    return g_tensor_cache.get_bucketed(dev, numel, dtype, tag)
                return torch.empty((numel,), dtype = dtype, device = dev)
            partials = ws(R * chunks * M1, torch.float, "hc_mix_partials").view(R, chunks, M1)
            post = ws(R * H, torch.float, "hc_post").view(R, H)
            comb = ws(R * H * H, torch.float, "hc_comb").view(R, H, H)
            collapsed = ws(R * D, torch.half, "hc_coll").view(R, D)
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


class GatedResidual(Module):
    """
    Qwen4Exp-style gated residual: the low-rank, elementwise cousin of mHC. The residual is the
    same (bsz, seq, hc_mult, hidden) fp32 stream stack, but mixing is per-channel instead of a
    stream-mixing matrix: per-stream grouped RMSNorm (zero-init weight, applied as 1 + w), a
    low-rank sigmoid gate over the normed stack picks what each stream contributes to the
    elementwise MEAN that feeds the sublayer, and the sublayer output is injected back into the
    raw streams with a per-stream scalar 2*sigmoid gate. No Sinkhorn, no combine matrix.

    Site form (use_combine = True): TransformerBlock calls mix() / apply_() like HyperConnection,
    with comb = None. Final-mixer form (use_combine = False, HF hyper_connection_mixer): a
    standalone module whose forward() collapses the stack.

    Two compute paths sharing the hc_mix.cu machinery: small R (decode) runs the fused
    ext.gr_mix pair (per-stream partial dots on the raw streams + a finalize that derives the
    low-rank gate inline), large R (prefill) runs half GEMMs + a few elementwise ops where
    launch count amortizes and tensor cores carry the FLOPs. apply_() is ext.hc_apply without
    a comb (x[h] += post[h] * y), shared with mHC. _mix_ref() keeps the fp32 torch reference
    the parity tests compare against.

    Tensors: {key}.hc_norm.weight, {key}.input_mix_weight_down.weight,
    {key}.input_mix_weight_up.weight and, for the site form, {key}.block_inject_weight.weight.
    """

    FUSED_MAX_R = 32

    def __init__(
        self,
        config: Config | None,
        key: str,
        hc_mult: int,
        hidden_size: int,
        rms_norm_eps: float,
        use_combine: bool = True,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.rms_eps = rms_norm_eps
        self.use_combine = use_combine
        self.out_dtype = out_dtype
        self.norm_w_raw = None
        self.norm_w = None          # (hc_mult, hidden) fp32, includes the + 1.0 (reference path)
        self.w_h = None             # (hc_mult * hidden) half, includes the + 1.0
        self.down_h = None          # (rank, hc_mult * hidden) half
        self.up_h = None            # (hc_mult * hidden, rank) half, checkpoint orientation
        self.upx_h = None           # (hc_mult, hidden / 4, rank, 4) half (fused-kernel layout)
        self.inject_h = None        # (hc_mult, hc_mult * hidden) half (site form)
        self.proj_h = None          # cat(down, inject) half, unfolded (GEMM path)
        self.fn_h = None            # cat(down, inject) * w half, folded (fused path)
        self.rank = 0

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.norm_w_raw = stc.get_tensor(f"{self.key}.hc_norm.weight", device, no_defer = True)
        down = stc.get_tensor(f"{self.key}.input_mix_weight_down.weight", device, no_defer = True)
        up = stc.get_tensor(f"{self.key}.input_mix_weight_up.weight", device, no_defer = True)
        inject = stc.get_tensor(f"{self.key}.block_inject_weight.weight", device,
                                no_defer = True) if self.use_combine else None
        self._prepare(down, up, inject)

    def _prepare(self, down, up, inject):
        # Derived buffers are deduplicated (down/inject live as views of proj_h; up is kept in
        # its checkpoint orientation and the GEMM path transposes by view), and the fp32 folding
        # intermediates go through a REUSED scratch: load interleaves these preparations with
        # the persistent weight allocations, and per-site transient churn splinters the
        # allocator's segments (measured ~15 GB reserved-not-allocated on the full model)
        from ..util.tensor import g_tensor_cache
        dev = down.device
        H, Dh = self.hc_mult, self.hidden_size
        self.norm_w = (self.norm_w_raw.float() + 1.0).view(H, Dh).contiguous()
        self.w_h = self.norm_w.flatten().half().contiguous()
        self.rank = down.shape[0]
        if inject is None:
            self.proj_h = down.half().contiguous()
            self.inject_h = None
        else:
            self.proj_h = torch.cat((down.half(), inject.half())).contiguous()
            self.inject_h = self.proj_h[self.rank :]
        self.down_h = self.proj_h[: self.rank]
        M = self.proj_h.shape[0]
        tmp = g_tensor_cache.get_bucketed(dev, M * H * Dh, torch.float, "gr_prep_tmp") \
            .view(M, H * Dh)
        tmp.copy_(self.proj_h)
        tmp *= self.w_h.float()
        self.fn_h = tmp.half().contiguous()
        self.up_h = up.half().contiguous()          # (H * D, rank), checkpoint orientation
        # up repacked (H, D/4, rank, 4) so the fused kernel's rank loop reads lane-contiguous
        self.upx_h = self.up_h.view(H, Dh // 4, 4, self.rank) \
            .permute(0, 1, 3, 2).contiguous()

    @override
    def unload(self):
        super().unload()
        self.norm_w_raw = self.norm_w = self.w_h = None
        self.down_h = self.up_h = self.upx_h = self.inject_h = self.proj_h = self.fn_h = None

    @override
    def get_tensors(self):
        t = {
            f"{self.key}.hc_norm.weight": self.norm_w_raw.contiguous(),
            f"{self.key}.input_mix_weight_down.weight": self.down_h.contiguous(),
            f"{self.key}.input_mix_weight_up.weight": self.up_h.contiguous(),
        }
        if self.use_combine:
            t[f"{self.key}.block_inject_weight.weight"] = self.inject_h.contiguous()
        return t

    @override
    def weights_numel(self):
        n = self.hc_mult * self.hidden_size
        return n + 2 * self.rank * n + (self.hc_mult * n if self.use_combine else 0)

    @override
    def optimizer_targets(self):
        return []

    def _mix_ref(self, streams: torch.Tensor):
        """fp32 torch reference of the mix (the parity tests' ground truth): returns
        (post (b, s, H) or None, mixed (b, s, D)), both fp32."""
        x = streams.float()
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.rms_eps) * self.norm_w
        flat = normed.flatten(-2)
        t = F.silu(F.linear(flat, self.down_h.float()) / self.hc_mult)
        w = torch.sigmoid(F.linear(t, self.up_h.float()))
        mixed = (w.unflatten(-1, (self.hc_mult, self.hidden_size)) * normed).mean(dim = -2)
        post = 2.0 * torch.sigmoid(F.linear(flat, self.inject_h.float()) / self.hc_mult) \
            if self.use_combine else None
        return post, mixed

    def _mix(self, streams: torch.Tensor):
        """streams (b, s, H, D) fp32 -> (post (R, H) fp32 or None, mixed (R, D) half)."""
        H, Dh = self.hc_mult, self.hidden_size
        R = streams.shape[0] * streams.shape[1]
        s3 = streams.reshape(R, H, Dh)
        if s3.dtype != torch.float:
            s3 = s3.float()          # MTP sample_from_state passes the half draft stack
        if not s3.is_contiguous():
            s3 = s3.contiguous()
        dev = s3.device
        post = torch.empty((R, H), dtype = torch.float, device = dev) \
            if self.use_combine else None

        if R <= self.FUSED_MAX_R:
            dots = torch.empty((R, self.fn_h.shape[0] + 1, H), dtype = torch.float, device = dev)
            mixed = torch.empty((R, Dh), dtype = torch.half, device = dev)
            ext.gr_mix(s3, self.fn_h, self.upx_h, self.w_h, self.rms_eps, dots, post, mixed)
        else:
            normed = torch.empty((R * H, Dh), dtype = torch.half, device = dev)
            ext.rms_norm(s3.view(R * H, Dh), self.w_h, normed,
                         self.rms_eps, 0.0, 1.0, False, False, H)
            dm = torch.matmul(normed.view(R, H * Dh), self.proj_h.t())     # (R, rank [+ H])
            t = F.silu(dm[:, : self.rank] / H)
            if self.use_combine:
                post.copy_(2.0 * torch.sigmoid(dm[:, self.rank :].float() / H))
            g = torch.matmul(t, self.up_h.t())                             # (R, H * Dh)
            mixed = (torch.sigmoid(g.float()).view(R, H, Dh)
                     * normed.float().view(R, H, Dh)).mean(dim = -2).half()
        return post, mixed

    def mix(self, streams: torch.Tensor, params: dict):
        """(b, s, H, D) fp32 -> (inject gates (b, s, H) fp32, None, collapsed (b, s, D) half)."""
        b, s = streams.shape[:2]
        post, mixed = self._mix(streams)
        return post.view(b, s, self.hc_mult), None, mixed.view(b, s, self.hidden_size)

    def apply_(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor | None,
        params: dict
    ):
        """Residual update for one sublayer site, in place: x <- x + post (x) y (comb unused).
        Conversion must NOT run the in-place path: the capture and advance passes forward the
        SAME stored input states twice (mHC apply_ has the same guard)."""
        if "quant_preserve" in params or "capture" in params:
            return x + post.unsqueeze(-1) * y.float().unsqueeze(-2)
        b, s = x.shape[:2]
        y2 = y.reshape(b * s, self.hidden_size)
        if y2.dtype not in (torch.half, torch.float):
            y2 = y2.half()
        ext.hc_apply(
            x.view(b * s, self.hc_mult, self.hidden_size),
            y2.contiguous(),
            post.reshape(b * s, self.hc_mult).contiguous(),
            None,
        )
        return x

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        """Final-mixer form only: collapse the stream stack."""
        assert not self.use_combine, "site-form GatedResidual is consumed via mix()/apply_()"
        # MTP trunk tap: models without a final norm export the PRE-collapse stream stack here
        # (flattened), the analog of the RMSNorm export hook
        if self.key in params.get("export_state_norm_keys", ()):
            states = params.get("export_states")
            if states is None:
                states = params["export_states"] = []
            states.append(x.flatten(-2).half())
        b, s = x.shape[:2]
        _, mixed = self._mix(x)
        mixed = mixed.view(b, s, self.hidden_size)
        dt = out_dtype or self.out_dtype
        return mixed if dt is None else mixed.to(dt)

    def tp_export(self, plan, producer):
        # Streams are replicated across TP workers (like the residual), so plain replication
        return {
            "cls": GatedResidual,
            "kwargs": {
                "key": self.key,
                "hc_mult": self.hc_mult,
                "hidden_size": self.hidden_size,
                "rms_norm_eps": self.rms_eps,
                "use_combine": self.use_combine,
                "out_dtype": self.out_dtype,
            },
            "norm_w_raw": producer.send(self.norm_w_raw),
            "down": producer.send(self.down_h),
            "up": producer.send(self.up_h),
            "inject": producer.send(self.inject_h) if self.use_combine else None,
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = GatedResidual(config = None, **exported["kwargs"])
        module.norm_w_raw = consumer.recv(exported["norm_w_raw"], cuda = True)
        down = consumer.recv(exported["down"], cuda = True)
        up = consumer.recv(exported["up"], cuda = True)
        inject = consumer.recv(exported["inject"], cuda = True) if module.use_combine else None
        module._prepare(down, up, inject)
        module.device = local_context["device"]
        return module


class HyperHead(Module):
    """Final mHC stream collapse before the model norm. Top-level raw tensors {key}_fn etc.
    mean = True (GLM5.3): parameterless unweighted mean over the streams, no tensors."""

    def __init__(self, config: Config, key: str, hc_mult: int, rms_norm_eps: float, hc_eps: float,
                 mean: bool = False):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult
        self.rms_eps = rms_norm_eps
        self.hc_eps = hc_eps
        self.mean = mean
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
        if self.mean:
            return
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
        if self.mean:
            return {}
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
        if self.mean:
            return []
        return [stc.get_tensor_size(k) for k in self._tensor_names()]

    @override
    def get_compile_tensors(self, stc):
        if self.mean:
            return {}
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
        if self.mean:
            return x.mean(dim = 2)
        b, s, H, D = x.shape
        if H == 4 and x.dtype == torch.float and D % 4 == 0 and x.is_contiguous():
            R = b * s
            chunks = ext.hc_mix_num_chunks(R, H * D)
            partials = g_tensor_cache.get_bucketed(
                x.device, R * chunks * (H + 1), torch.float, "hc_head_partials").view(R, chunks, H + 1)
            collapsed = g_tensor_cache.get_bucketed(
                x.device, R * D, torch.float, "hc_head_coll").view(R, D)
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
