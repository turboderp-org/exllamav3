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
from ...constants import PAGE_SIZE

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
position, kept in the trunk's sliding-window ring as the drafter's recurrent state. Block sizes are
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
    tensor keys as a trunk sliding layer (so conversion is identical), the trunk's window
    ring as its recurrent state, and a block-parallel non-causal forward. The main-kv rows
    are derived from the target's tap states outside any forward (update_kv_from_target:
    plan_state_rows / write_state_rows / commit_state_rows, the DFlash drafters' pattern),
    so the state stands at the accepted K/V position and the draft forward never moves it.
    The rows ride the generator's recurrent checkpoints with the target's.
    """

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        """Draft-block forward: x (bsz, block, hidden) fp16 (collapsed+normed stream mix).
        Block token positions are cache_seqlens[r] + j; every block query sees the SAME
        range [p0 - hist, p0 + block): the last min(window, p0) main-kv rows from the ring
        plus the whole block, non-causally, in one softmax with sinks. All through the
        trunk's own kernels (fused norm+rope, dsa_attn NC_BLOCK)."""
        bsz, s, _ = x.shape
        device = x.device
        rsg = params["recurrent_states"]
        rsl = self._get_rsl(rsg[0], (self.layer_idx, params.get("layer_instance", 0)))
        seqlens = params["cache_seqlens"]                       # host values
        H, D, rd = self.num_q_heads, self.head_dim, self.rope_head_dim
        positions = get_for_device(params, "cache_seqlens", device)

        # Projections + fused unweighted q head norm / weighted kv norm / partial rope
        _, q, kv = self._project_qkv(x, params, 0, positions)
        if dspark_fp8_kv:
            fp8_fake_quant_(kv[..., : D - rd])

        hpg = H // self.o_groups
        pc0 = x.new_empty((1, D - rd), dtype = torch.half)
        pr0 = x.new_empty((1, rd), dtype = torch.half)
        bt0 = torch.zeros((1, 1), dtype = torch.int32, device = device)
        outs = []

        # TODO: Batched DSpark-attn fwd
        for r in range(bsz):
            rs = rsg[r]
            p0 = int(seqlens[r])
            assert rs.position == p0, f"{self.key}: draft state at {rs.position}, K/V position {p0}"
            hist = min(self.sliding_window, p0)
            assert p0 - rs.window_beg >= hist, f"{self.key}: window rows not resident"
            outs.append(dsa_attn(
                q[r].half().contiguous(),
                pc0, pr0, bt0,
                sinks = self.sinks,
                ring = rsl.ring[rs.slot],
                kv_chunk = kv[r].contiguous(),
                win_len = hist + s,
                win_floor = p0 - hist,
                ring_beg = rs.window_beg,
                pool_len = 0,
                q_pos0 = p0,
                compress_rate = 1,
                scale = self.sm_scale,
                derot_inv_freq = self._rope_type_neg(),
                groups = self.o_groups,
                nc_block = True,
                out = torch.empty((self.o_groups, s, hpg * D), dtype = torch.half, device = device),
            ))
        o = torch.stack(outs, dim = 0).permute(0, 2, 1, 3)      # (bsz, s, G, hpg * D)
        o = torch.cat([self.wo_a[g].forward(o[:, :, g].contiguous(), params) for g in range(self.o_groups)], dim = -1)
        return self.wo_b.forward(o, params, out_dtype = out_dtype or self.out_dtype)


    def project_rows(self, main_x: torch.Tensor, params: dict) -> torch.Tensor:
        """Main-kv rows (bsz, s, head_dim) for target positions cache_seqlens[r] .. + s from
        the projected tap state main_x (bsz, s, hidden) fp16. The fused rope kernel applies
        the weighted kv norm (passed in the q-norm slot) and the partial rotation."""
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
        return kv.view(bsz, s, D)


    def plan_state_rows(self, rsg, positions: list[int], lengths: list[int] | None, n_default: int):
        """Ring geometry for the rows written outside a forward: row i's lengths[i] rows (default
        n_default) go at absolute position positions[i], where its state must stand. A run past
        the end of the ring rebases the window: the ring keeps the last `window` rows below the
        new position (page-aligned start), moved to the front. Every DSpark layer shares one state,
        so the plan is taken before any layer writes and the states move once (commit_state_rows).
        Returns per row (slot, pos0, n, offset, shift, n_keep, n_from_ring), or None when nothing
        is written"""
        rsl = self._get_rsl(rsg[0], (self.layer_idx, 0))
        rows, w = rsl.ring_rows, self.sliding_window
        plan = []
        for i, rs in enumerate(rsg):
            n = n_default if lengths is None else lengths[i]
            if n <= 0:
                plan.append(None)
                continue
            pos0 = positions[i]
            assert rs.position == pos0, f"{self.key}: draft state at {rs.position}, rows written at {pos0}"
            offset = pos0 - rs.window_beg
            pos_end = pos0 + n
            if offset + n <= rows:
                plan.append((rs.slot, pos0, n, offset, 0, 0, 0))
                continue
            new_beg = max(pos_end - w, 0) // PAGE_SIZE * PAGE_SIZE
            n_keep = pos_end - new_beg
            n_from_ring = max(n_keep - n, 0)
            assert 0 < n_keep <= rows and n_from_ring <= offset
            plan.append((rs.slot, pos0, n, offset, new_beg - rs.window_beg, n_keep, n_from_ring))
        return plan


    def write_state_rows(self, rsg, plan: list, kv: torch.Tensor):
        """Write this layer's rows (bsz, s, head_dim) per plan_state_rows into its ring"""
        rsl = self._get_rsl(rsg[0], (self.layer_idx, 0))
        for i, p in enumerate(plan):
            if p is None:
                continue
            slot, pos0, n, offset, shift, n_keep, n_from_ring = p
            ring = rsl.ring[slot]
            if shift == 0:
                ring[offset : offset + n].copy_(kv[i, :n])
                continue
            if n_from_ring > 0:
                ring[:n_from_ring].copy_(ring[offset - n_from_ring : offset].clone())
            ring[n_from_ring : n_keep].copy_(kv[i, n - (n_keep - n_from_ring) : n])


    @staticmethod
    def commit_state_rows(rsg, plan: list):
        """Move the states past the rows every layer wrote"""
        for rs, p in zip(rsg, plan):
            if p is None:
                continue
            _, pos0, n, _, shift, _, _ = p
            rs.window_beg += shift
            rs.position = pos0 + n
            rs.last_history = 0
            rs.wshift = 0


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
            select_hq_bits = 1,
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
