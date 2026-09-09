from __future__ import annotations
from typing_extensions import override
import torch

from ...ext import exllamav3_ext as ext
from ...modules import Module, RMSNorm
from ...util.device_copy import to_device
from ...modules.dsv4 import DSV4Attention
from ...modules.attention_fn.dsa_triton import dsa_attn
from ...util.rope import RopeStyle
from ...util.tensor import get_for_device

import os
dspark_fp8_kv = os.environ.get("EXL3_DSPARK_FP8KV", "0") != "0"

"""
DSpark drafter runtime modules (DeepSeek-V4 MTP component). Reference implementation ships
in the checkpoint: hf/inference/model.py (DSparkAttention / DSparkBlock / forward_spec).

The drafter runs ONE forward per draft call over a block of dspark_block_size tokens
[seed, noise, ...], with each block query attending NON-CAUSALLY over
[the last <=window main-kv rows ++ the whole block] in a single softmax with per-head
sinks, eq. 26 output de-rotation, and the trunk's grouped o_proj. The main-kv history is
derived from the TRUNK's tap states (update_kv_from_target), one 512-wide row per target
position, stored paged so draft and target cache layouts stay aligned. Block sizes are
derived from the trunk's own attention machinery: the fused norm+rope kernel and the
dsa_attn one-shot kernel in NC_BLOCK mode (non-causal chunk + paged window history).
"""


def fp8_fake_quant_(x: torch.Tensor):
    """In-place e4m3 fake-quant of the kv nope part, groups of 64 with ue8m0 (power-of-2)
    scales — matches the reference act_quant(kv[..., :-rd], 64, ...) so the drafter sees
    the kv distribution it was trained with. Experimental (EXL3_DSPARK_FP8KV=1)."""
    g = x.view(*x.shape[:-1], -1, 64).float()
    amax = g.abs().amax(-1, keepdim = True).clamp(min = 1e-12)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (g / scale).to(torch.float8_e4m3fn).float() * scale
    x.copy_(q.view(x.shape).to(x.dtype))


def to_dev(t: torch.Tensor, device) -> torch.Tensor:
    """p2p-safe device move: on some systems torch fails to detect unsupported p2p and
    .to() silently yields an empty tensor; route through system memory when flagged."""
    if t.device == device:
        return t
    return to_device(t, device)


class DSparkAttention(DSV4Attention):
    """
    Compressor-less DSV4 attention with DSpark draft semantics: same projection set and
    tensor keys as a trunk sliding layer (so conversion is identical), but a paged main-kv
    cache layer instead of recurrent ring state, and a block-parallel non-causal forward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Paged main-kv layer, no recurrent ring state
        self.caps["recurrent_cache"] = False
        self.caps["kv_cache"] = True


    def cache_layer_type(self, default, kwargs: dict):
        from ...cache.dsa import CacheLayer_dspark
        return CacheLayer_dspark, {}


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        """Draft-block forward: x (bsz, block, hidden) fp16 (collapsed+normed stream mix).
        Block token positions are cache_seqlens[r] + j; every block query sees the SAME
        range [p0 - hist, p0 + block): the last min(window, p0) main-kv rows (read paged
        via the job's block table) plus the whole block, non-causally, in one softmax with
        sinks. All through the trunk's own kernels (fused norm+rope, dsa_attn NC_BLOCK)."""
        bsz, s, _ = x.shape
        device = x.device
        cache = params["cache"]
        kl = cache.layers[(self.layer_idx, params.get("layer_instance", 0))]
        block_table = get_for_device(params, "block_table", self.device)
        seqlens = params["cache_seqlens"]                       # host values
        H, D, rd = self.num_q_heads, self.head_dim, self.rope_head_dim
        positions = get_for_device(params, "cache_seqlens", device)

        # Projections + fused unweighted q head norm / weighted kv norm / partial rope
        q_res = self.q_norm.forward(self.q_a.forward(x, params), params, out_dtype = torch.half)
        q = self.q_b.forward(q_res, params).view(bsz, s, H, D)
        kv = self.wkv.forward(x, params).view(bsz, s, 1, D)
        ext.rope(
            q, q, kv, kv,
            self._rope_type(), 0, positions, None,
            int(RopeStyle.GPTJ), 1.0, self.q_ones, self.kv_norm_w,
            self.rms_norm_eps, 0.0, 0.0, 0, 1, D - rd,
        )
        kv = kv.view(bsz, s, D)
        if dspark_fp8_kv:
            fp8_fake_quant_(kv[..., : D - rd])

        hpg = H // self.o_groups
        pc0 = x.new_empty((1, D - rd), dtype = torch.half)
        pr0 = x.new_empty((1, rd), dtype = torch.half)
        main_kv = kl.kv.view(-1, D)
        outs = []

        # TODO: Batched DSpark-attn fwd
        for r in range(bsz):
            p0 = int(seqlens[r])
            hist = min(self.sliding_window, p0)
            outs.append(dsa_attn(
                q[r].half().contiguous(),
                pc0, pr0,
                block_table[r : r + 1],
                sinks = self.sinks,
                ring = main_kv,
                kv_chunk = kv[r].contiguous(),
                win_len = hist + s,
                win_floor = p0 - hist,
                ring_beg = 0,
                pool_len = 0,
                q_pos0 = p0,
                compress_rate = 1,
                scale = self.sm_scale,
                derot_inv_freq = self._rope_type_neg(),
                groups = self.o_groups,
                page_size = 256,
                nc_block = True,
                out = torch.empty((self.o_groups, s, hpg * D), dtype = torch.half, device = device),
            ))
        o = torch.stack(outs, dim = 0).permute(0, 2, 1, 3)      # (bsz, s, G, hpg * D)
        o = torch.cat([self.wo_a[g].forward(o[:, :, g].contiguous(), params) for g in range(self.o_groups)], dim = -1)
        return self.wo_b.forward(o, params, out_dtype = out_dtype or self.out_dtype)


    def update_kv_rows(self, main_x: torch.Tensor, params: dict):
        """Write main-kv rows for target positions cache_seqlens[r] .. + s from the
        projected tap state main_x (bsz, s, hidden) fp16. The fused rope kernel applies
        the weighted kv norm (passed in the q-norm slot) and the partial rotation."""
        cache = params["cache"]
        kl = cache.layers[(self.layer_idx, params.get("layer_instance", 0))]
        block_table = get_for_device(params, "block_table", self.device)
        mx = to_dev(main_x, self.device)
        bsz, s, _ = mx.shape
        D, rd = self.head_dim, self.rope_head_dim
        kv = self.wkv.forward(mx, params).view(bsz, s, 1, D)
        positions = get_for_device(params, "cache_seqlens", mx.device)
        ext.rope(
            kv, kv, None, None,
            self._rope_type(), 0, positions, None,
            int(RopeStyle.GPTJ), 1.0, self.kv_norm_w, None,
            self.rms_norm_eps, 0.0, 0.0, 0, 1, D - rd,
        )
        if dspark_fp8_kv:
            fp8_fake_quant_(kv[..., : D - rd])
        kl.write_rows(kv.view(bsz, s, D), positions, block_table)


class DSparkInputLayer(Module):
    """
    Owns main_proj / main_norm (used by update_kv_from_target, quantized by conversion)
    and, in the draft forward, embeds the [seed, noise x (block - 1)] token block via the
    ATTACHED trunk's embedding and expands it to the mHC stream stack.
    """

    def __init__(self, config, key: str, hidden_size: int, n_taps: int, hc_mult: int,
                 noise_token_id: int, block_size: int, rms_norm_eps: float):
        super().__init__(config, key, None)
        from ...modules import Linear
        self.module_name = "DSparkInputLayer"
        self.hc_mult = hc_mult
        self.noise_token_id = noise_token_id
        self.block_size = block_size
        self.main_proj = Linear(
            config = config,
            key = "mtp.0.main_proj",
            in_features = n_taps * hidden_size,
            out_features = hidden_size,
            qmap = "mtp.input",
            qbits_key = "mtp_bits",
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.main_norm = RMSNorm(config, "mtp.0.main_norm", rms_norm_eps)
        self.register_submodule(self.main_proj)
        self.register_submodule(self.main_norm)
        self.attached_model = None
        self.own_embed = None   # set by attach when the target is TP (embedding not borrowable)
        self.caps.update({"x_cpu": True})


    def optimizer_targets(self):
        raise NotImplementedError()


    def project_taps(self, target_hidden: torch.Tensor, params: dict) -> torch.Tensor:
        """(bsz, s, n_taps * hidden) -> normed main_x (bsz, s, hidden) fp16"""
        mx = self.main_proj.forward(target_hidden, params, out_dtype = torch.half)
        return self.main_norm.forward(mx, params, out_dtype = torch.half)


    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x


    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        bsz, seqlen = x.shape
        noise = torch.full((bsz, self.block_size - 1), self.noise_token_id, dtype = torch.long)
        x = torch.cat((x.cpu(), noise), dim = -1)
        embed = self.own_embed if self.own_embed is not None else self.attached_model().modules[0]
        x = embed.forward(x, params)
        # mHC stream stack: broadcast copies of the embedding, fp32
        return x.float().unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
