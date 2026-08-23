import os

import torch

from ...ext import exllamav3_ext as ext
from ...constants import PAGE_SIZE
from ...util.tensor import g_tensor_cache
from .bc_attn import _compile_kernel
from .dsa_triton import _dsa_attn_split_kernel, _dsa_attn_combine_kernel, _dsa_indexer_fewq_kernel

"""
Whole-attention-step graphs for DeepSeek-V4 DSA decode (BC_DSV4Attention): projections,
fused-norm rope, both compressors, [indexer + capture-safe top-k], AOT split/combine
attention, grouped o_proj and the ring append run as one C++ call, captured per
(seq 1..16, regime dense/topk) slot and replayed with a handful of patched scalars plus the
input pointer. One BC per (module, cache layer state, job slot): the slot's rings/pools are
baked into the graphs. pos/ring_beg flow through device scalars shared by all layers of a
job (position is layer-invariant, so steady-state decode needs one 8-byte host write per
step for the whole model).

On by default; EXL3_BC_DSA=0 falls back to the eager path. Ineligible configurations
decline per-layer; build failures raise with EXL3_BC_DSA_DEBUG=1, else decline.
"""

bc_dsa_enable = os.environ.get("EXL3_BC_DSA", "1") != "0"
_bc_debug = os.environ.get("EXL3_BC_DSA_DEBUG", "0") != "0"

MAX_QLEN = 16
N_SPLITS = 16
BLOCK_H = 16


def _exl3_bc(lin):
    if lin is None or lin.quant_type != "exl3":
        return None
    return lin.inner.bc


class BCDsa:

    def __init__(self, module, rs, rsl, kl):
        m = module
        self.module = m
        self.device = torch.device(m.device) if isinstance(m.device, int) else torch.device(m.device)
        self.hidden = m.hidden_size
        self.head_dim = m.head_dim
        self.rd = m.rope_head_dim
        self.H = m.num_q_heads
        self.G = m.o_groups
        self.o_lora = m.wo_a[0].out_features
        self.m_rate = m.compress_rate if m.compressor is not None else 1
        self.window = m.sliding_window
        self.has_comp = m.compressor is not None
        self.has_idx = m.indexer is not None
        self.kl = kl
        self.epp = kl.epp if self.has_comp else PAGE_SIZE
        self.slot = rs.slot
        slot = rs.slot

        # Companions must exist (deferred-safe: built at first fused forward, which has
        # already happened by the time decode reaches here)
        if self.has_comp and m.compressor.bc is None:
            raise RuntimeError("compressor BC missing")
        if self.has_idx and m.indexer.bc is None:
            raise RuntimeError("indexer BC missing")
        if m.wo_a_multi is None:
            raise RuntimeError("wo_a multilinear missing")
        for lin in [m.q_a, m.q_b, m.wkv, m.wo_b] + ([m.idx_wq_b] if self.has_idx else []):
            if _exl3_bc(lin) is None:
                raise RuntimeError(f"non-exl3 projection {lin.key}")
        if m.q_a.out_features != m.q_a.out_features_unpadded or \
                m.q_b.out_features != m.q_b.out_features_unpadded or \
                m.wkv.out_features != m.wkv.out_features_unpadded:
            raise RuntimeError("padded projections")

        idx_w = None
        if self.has_idx:
            iw = m.idx_weights.inner
            if getattr(iw, "weight", None) is None or iw.weight.dtype != torch.half:
                raise RuntimeError("indexer weights not fp16")
            idx_w = iw.weight

        # Device position scalars + per-job block-table static, shared across layers per
        # (state, device). The graphs bake the bt pointer; its CONTENT is refreshed once
        # per job step (the job's table row can gain valid columns as the batch table
        # widens, and a new job on the slot brings a new table)
        num_pages = rs.cache.max_num_tokens // PAGE_SIZE
        key = ("bc_dsa_pos", self.device)
        store = rs.__dict__.setdefault("_bc_dsa_pos", {})
        if key not in store:
            store[key] = (
                torch.zeros(1, dtype = torch.int32, device = self.device),
                torch.zeros(1, dtype = torch.int32, device = self.device),
                torch.zeros(2, dtype = torch.int32, pin_memory = True),
                [-1, -1],   # host mirror (pos, ring_beg)
                torch.zeros(num_pages, dtype = torch.int32, device = self.device),
                [-1, -1],   # bt mirror (state serial, pos)
            )
        self.pos_dev, self.rb_dev, self.pos_pin, self.pos_mirror, self.bt_st, self.bt_mirror \
            = store[key]

        mu = m.wo_a_multi
        q_lora = m.q_a.out_features
        self.q_lora = q_lora

        # x-side projection fan: q_a/wkv/comp/idx as one per-matrix-N mgemm, when the whole
        # group shares bits/format and q_a is the widest output (the dtype/locks carrier)
        fan_lins = [m.q_a, m.wkv]
        if self.has_comp:
            fan_lins += [m.compressor.wkv, m.compressor.wgate]
        if self.has_idx:
            fan_lins += [m.indexer.wkv, m.indexer.wgate]
        fan_inner = [l.inner for l in fan_lins]
        self.fan_ns = [l.out_features for l in fan_lins]
        uniform = (
            all(l.quant_type == "exl3" for l in fan_lins)
            and all(l.out_features == l.out_features_unpadded for l in fan_lins)
            and len({(i.K, i.mcg, i.mul1) for i in fan_inner}) == 1
            and max(self.fan_ns) <= q_lora
        )
        self.fan = None
        if uniform:
            dev = self.device
            self.fan = dict(
                trellis = torch.tensor([i.trellis.data_ptr() for i in fan_inner],
                                       dtype = torch.long, device = dev),
                suh = torch.tensor([i.suh.data_ptr() for i in fan_inner],
                                   dtype = torch.long, device = dev),
                svh = torch.tensor([i.svh.data_ptr() for i in fan_inner],
                                   dtype = torch.long, device = dev),
                n = torch.tensor(self.fan_ns, dtype = torch.int32, device = dev),
                idx = torch.arange(len(fan_lins), dtype = torch.long, device = dev).unsqueeze(0),
            )
        cap = kl.capacity if self.has_comp else PAGE_SIZE
        bt = self.bt_st.view(1, -1) if self.has_comp else \
            torch.zeros((1, 1), dtype = torch.int32, device = self.device)

        self.bc = ext.BC_DSV4Attention(
            _exl3_bc(m.q_a), _exl3_bc(m.q_b), _exl3_bc(m.wkv), _exl3_bc(m.wo_b),
            _exl3_bc(m.idx_wq_b) if self.has_idx else _exl3_bc(m.q_a), idx_w,
            m.compressor.bc if self.has_comp else None,
            m.indexer.bc if self.has_idx else None,
            mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh, m.woa_indices,
            mu.K, mu.mcg, mu.mul1,
            m.q_norm.weight.data, m.q_ones, m.kv_norm_w,
            m._rope_type(), m._rope_type_neg(), m.sinks,
            rsl.ring[slot],
            rsl.comp_buf_kv[slot] if self.has_comp else None,
            rsl.comp_buf_gate[slot] if self.has_comp else None,
            rsl.comp_ovl[slot] if (self.has_comp and rsl.comp_ovl is not None) else None,
            kl.pool_c.view(-1, kl.D_c) if self.has_comp else None,
            kl.pool_r.view(-1, kl.D_r) if self.has_comp else None,
            rsl.idx_buf_kv[slot] if self.has_idx else None,
            rsl.idx_buf_gate[slot] if self.has_idx else None,
            rsl.idx_ovl[slot] if self.has_idx else None,
            kl.pool_idx.view(-1, kl.D_i) if self.has_idx else None,
            bt, self.pos_dev, self.rb_dev,
            self.hidden, self.H, self.head_dim, self.rd, q_lora,
            self.G, self.o_lora,
            m.index_n_heads if self.has_idx else 1,
            m.index_head_dim if self.has_idx else 1,
            m.index_topk if self.has_idx else 1,
            self.window, self.m_rate, self.epp, m.sm_scale, m.rms_norm_eps,
            N_SPLITS, BLOCK_H,
            self.fan["trellis"] if self.fan else None,
            self.fan["suh"] if self.fan else None,
            self.fan["svh"] if self.fan else None,
            self.fan["n"] if self.fan else None,
            self.fan["idx"] if self.fan else None,
        )
        self.scores_max = -(-cap // 128) * 128

    def _configure(self, seq, regime):
        dev = self.device
        gtc = g_tensor_cache
        hd, H, G = self.head_dim, self.H, self.G
        hpg = H // G
        D = hd

        def st(shape, dtype, tag):
            return gtc.get(dev, shape, dtype, tag)

        # x_st/wts_st padded to >= 8 rows: cuBLASLt's kernel choice for the wts GEMM at
        # M = 2..7 is ~13x slower than its M >= 8 choice, so the hgemm runs full-height
        # over zero-padded rows (fewq never reads the padded wts rows)
        seq_p = max(seq, 8)
        x_st = st((seq_p, self.hidden), torch.half, "bcd_x")
        x_st.zero_()
        qa_st = st((seq, self.q_lora), torch.half, "bcd_qa")
        qres_st = st((seq, self.q_lora), torch.half, "bcd_qres")
        q_st = st((seq, H * hd), torch.half, "bcd_q")
        kv_st = st((seq, hd), torch.half, "bcd_kv")
        xh_a = st((seq, self.hidden), torch.half, "bcd_xha")
        xh_b = st((seq, self.q_lora), torch.half, "bcd_xhb")
        m_mod = self.module
        mgc_comp = mgc_idx = None
        if self.has_comp:
            mgc_comp = st((2, seq, m_mod.compressor.wkv.out_features), torch.half, "bcd_mgc")
        if self.has_idx:
            mgc_idx = st((2, seq, m_mod.indexer.wkv.out_features), torch.half, "bcd_mgi")
        qidx_st = wts_st = scores_st = indices_st = None
        k_fewq = None
        if regime == 1:
            Hi, Di = m_mod.index_n_heads, m_mod.index_head_dim
            qidx_st = st((seq, Hi * Di), torch.half, "bcd_qidx")
            wts_st = st((seq_p, Hi), torch.half, "bcd_wts")
            scores_st = st((seq, self.scores_max), torch.half, "bcd_scores")
            scores_st.fill_(-float("inf"))   # stale region must stay -inf (topk scans full width)
            kp = -(-m_mod.index_topk // 32) * 32
            indices_st = st((seq, kp), torch.int32, "bcd_indices")
            sig = {
                "q_idx": "*fp16:16", "w": "*fp16:16", "k_idx": "*fp16:16", "scores": "*fp16:16",
                "T": "i32", "R": "i32", "q_pos0": "i32", "bound_max": "i32",
                "block_table": "*i32:16", "num_pages_per_row": "i32",
            } | {n: "constexpr" for n in (
                "H_i", "H_pad", "D_i", "S_stride", "compress_rate", "scale", "BLOCK_N",
                "SEQ", "MULTIROW", "EPP", "DEBUG_BOUNDS", "DEBUG_PAGES")}
            consts = dict(
                H_i = Hi, H_pad = max(16, 1 << (Hi - 1).bit_length()), D_i = Di,
                S_stride = self.scores_max, compress_rate = self.m_rate,
                scale = Di ** -0.5 * Hi ** -0.5, BLOCK_N = 128,
                SEQ = 1, MULTIROW = 0, EPP = self.epp, DEBUG_BOUNDS = 0, DEBUG_PAGES = 0,
            )
            k_fewq = _compile_kernel(dev, _dsa_indexer_fewq_kernel, sig, consts, 8, 2)

        hb = -(-H // BLOCK_H)
        ws_ml = st((seq * hb * N_SPLITS * BLOCK_H * 2,), torch.float, "bcd_wsml")
        ws_acc = st((seq * hb * N_SPLITS * BLOCK_H * D,), torch.float, "bcd_wsacc")
        attn_out = st((G, seq, hpg * hd), torch.half, "bcd_aout")
        woa_c = st((G, seq, self.o_lora), torch.half, "bcd_woac")
        woa_t = st((seq, G * self.o_lora), torch.half, "bcd_woat")
        woa_xh = st((G * seq, hpg * hd), torch.half, "bcd_woaxh")
        y_st = st((seq, self.hidden), torch.float, "bcd_y")

        D_c = hd - self.rd
        kp = -(-self.module.index_topk // 32) * 32 if self.has_idx else 32
        sig_s = {
            "q": "*fp16:16", "ring": "*fp16:16", "kv_chunk": "*fp16:16", "pool_c": "*fp16:16",
            "pool_r": "*fp16:16", "block_table": "*i32:16", "indices": "*i32:16",
            "ws_ml": "*fp32:16", "ws_acc": "*fp32:16",
            "k_len": "i32", "win_len": "i32", "pool_len": "i32",
            "num_pages_per_row": "i32", "q_pos0": "i32", "win_floor": "i32", "ring_beg": "i32",
            "slot_ids": "i32", "ring_stride": "i32",
        } | {n: "constexpr" for n in (
            "H", "page_size", "D_c", "D_c_pad", "D_r", "K_pad", "compress_rate", "scale",
            "HAS_WINDOW", "DENSE_POOL", "BLOCK_H", "BLOCK_N", "BLOCK_W", "SEQ", "MULTIROW", "DEBUG_BOUNDS", "DEBUG_PAGES", "Q_SPLIT", "OUT_LATENT")}
        consts_s = dict(
            H = H, page_size = self.epp, D_c = D_c,
            D_c_pad = 1 << (D_c - 1).bit_length(), D_r = self.rd, K_pad = kp,
            compress_rate = self.m_rate, scale = self.module.sm_scale,
            HAS_WINDOW = True, DENSE_POOL = regime == 0,
            BLOCK_H = BLOCK_H, BLOCK_N = 32, BLOCK_W = 16,
            SEQ = 1, MULTIROW = 0, DEBUG_BOUNDS = 0, DEBUG_PAGES = 0,
            Q_SPLIT = 0, OUT_LATENT = 0,
        )
        k_split = _compile_kernel(dev, _dsa_attn_split_kernel, sig_s, consts_s, 4, 2)

        sig_c = {
            "ws_ml": "*fp32:16", "ws_acc": "*fp32:16", "sinks": "*fp32:16",
            "derot_inv_freq": "*fp32:16",
            "out": "*fp16:16", "q_pos0": "i32", "R": "i32", "n_splits": "i32",
        } | {n: "constexpr" for n in (
            "H", "D_c", "D_r", "HAS_SINKS", "DEROTATE", "HPG", "BLOCK_H", "BLOCK_D",
            "SEQ", "MULTIROW", "OUT_LATENT")}
        consts_c = dict(
            H = H, D_c = D_c, D_r = self.rd, HAS_SINKS = True, DEROTATE = True,
            HPG = hpg, BLOCK_H = BLOCK_H, BLOCK_D = 128,
            SEQ = 1, MULTIROW = 0, OUT_LATENT = 0,
        )
        k_combine = _compile_kernel(dev, _dsa_attn_combine_kernel, sig_c, consts_c, 4, 2)

        fan_c = fan_ah = None
        if self.fan is not None:
            outs = [qa_st, kv_st]
            if self.has_comp:
                outs += [mgc_comp[0], mgc_comp[1]]
            if self.has_idx:
                outs += [mgc_idx[0], mgc_idx[1]]
            fan_c = torch.tensor([o.data_ptr() for o in outs], dtype = torch.long, device = dev)
            fan_ah = st((len(outs), seq, self.hidden), torch.half, "bcd_fanah")
        self.bc.configure_slot(
            seq, regime, x_st, qa_st, qres_st, q_st, kv_st, xh_a, xh_b,
            mgc_comp, mgc_idx, qidx_st, wts_st, scores_st, indices_st,
            ws_ml, ws_acc, attn_out, woa_c, woa_t, woa_xh, y_st,
            k_split, k_combine, k_fewq, fan_c, fan_ah)

    def run(self, x, rs, rsl, bt_row = None):
        """x (1, seq, hidden) fp16 contiguous, bt_row (1, npr) i32 device block-table row of
        this job (paged pools). Returns y (1, seq, hidden) fp32 (a static: consume before
        the next BC call), or None to decline (host-side ring maintenance needed this
        step)."""
        seq = x.shape[1]
        pos = rs.position
        win_beg = rs.window_beg

        # Ring shift/rebase must not happen inside the graph; decline to the eager path for
        # that step (page-granular, rare)
        if pos - win_beg + seq > rsl.ring_rows:
            return None
        ec = (pos + seq) // self.m_rate if self.has_comp else 0
        regime = 1 if (self.has_idx and ec > self.module.index_topk) else 0
        if self.has_comp and (bt_row is None or ec > bt_row.shape[1] * self.epp):
            return None

        if self.bc.needs_configure(seq, regime):
            self._configure(seq, regime)

        # Refresh the block-table static once per job step (any compressor layer may be the
        # one to do it; sliding layers never touch it, so this is tracked separately from
        # the position mirror)
        if self.has_comp:
            serial = getattr(rs, "serial", -1)
            if self.bt_mirror[0] != serial or self.bt_mirror[1] != pos:
                npr = min(bt_row.shape[1], self.bt_st.shape[0])
                self.bt_st[:npr].copy_(bt_row[0, :npr])
                self.bt_mirror[0] = serial
                self.bt_mirror[1] = pos

        # One 8-byte host write per step per job, shared by every layer
        if self.pos_mirror[0] != pos or self.pos_mirror[1] != win_beg:
            self.pos_pin[0] = pos
            self.pos_pin[1] = win_beg
            self.pos_dev.copy_(self.pos_pin[0:1], non_blocking = True)
            self.rb_dev.copy_(self.pos_pin[1:2], non_blocking = True)
            self.pos_mirror[0] = pos
            self.pos_mirror[1] = win_beg

        y = self.bc.run(x, pos, win_beg, regime)
        return y.view(1, seq, self.hidden)


def build_bc_dsa(module, rs, rsl, kl):
    try:
        return BCDsa(module, rs, rsl, kl)
    except Exception:
        if _bc_debug:
            raise
        return None


class BCDsaBatch:
    """
    Batched whole-step graphs (BC_DSV4BatchAttention): B jobs x S tokens per transition,
    graphs keyed (B, S). One instance per (module, cache layer state); all per-job state
    flows through the (6, MAX_B) device array + the (MAX_B, num_pages) block-table static,
    both refreshed once per step by run(). The x-side projection fan is required.
    """

    MAX_B = 8

    def __init__(self, module, rsl, kl):
        m = module
        self.module = m
        self.device = torch.device(m.device) if isinstance(m.device, int) else torch.device(m.device)
        self.hidden = m.hidden_size
        self.head_dim = m.head_dim
        self.rd = m.rope_head_dim
        self.H = m.num_q_heads
        self.G = m.o_groups
        self.o_lora = m.wo_a[0].out_features
        self.m_rate = m.compress_rate if m.compressor is not None else 1
        self.window = m.sliding_window
        self.has_comp = m.compressor is not None
        self.has_idx = m.indexer is not None
        self.kl = kl
        self.epp = kl.epp if self.has_comp else PAGE_SIZE
        self.kp = -(-m.index_topk // 32) * 32 if self.has_idx else 32

        if m.wo_a_multi is None:
            raise RuntimeError("wo_a multilinear missing")
        for lin in [m.q_b, m.wo_b] + ([m.idx_wq_b] if self.has_idx else []):
            if _exl3_bc(lin) is None:
                raise RuntimeError(f"non-exl3 projection {lin.key}")
        if m.q_b.out_features != m.q_b.out_features_unpadded:
            raise RuntimeError("padded projections")

        idx_w = None
        if self.has_idx:
            iw = m.idx_weights.inner
            if getattr(iw, "weight", None) is None or iw.weight.dtype != torch.half:
                raise RuntimeError("indexer weights not fp16")
            idx_w = iw.weight

        # x-side fan (required: the whole batched body assumes one projection mgemm)
        fan_lins = [m.q_a, m.wkv]
        if self.has_comp:
            if not m.compressor.fused_ready:
                m.compressor._build_fused()
            fan_lins += [m.compressor.wkv, m.compressor.wgate]
        if self.has_idx:
            if not m.indexer.fused_ready:
                m.indexer._build_fused()
            fan_lins += [m.indexer.wkv, m.indexer.wgate]
        fan_inner = [l.inner for l in fan_lins]
        self.fan_ns = [l.out_features for l in fan_lins]
        q_lora = m.q_a.out_features
        self.q_lora = q_lora
        if not (
            all(l.quant_type == "exl3" for l in fan_lins)
            and all(l.out_features == l.out_features_unpadded for l in fan_lins)
            and len({(i.K, i.mcg, i.mul1) for i in fan_inner}) == 1
            and max(self.fan_ns) <= q_lora
        ):
            raise RuntimeError("x-side fan unavailable")
        dev = self.device
        self.fan = dict(
            trellis = torch.tensor([i.trellis.data_ptr() for i in fan_inner],
                                   dtype = torch.long, device = dev),
            suh = torch.tensor([i.suh.data_ptr() for i in fan_inner],
                               dtype = torch.long, device = dev),
            svh = torch.tensor([i.svh.data_ptr() for i in fan_inner],
                               dtype = torch.long, device = dev),
            n = torch.tensor(self.fan_ns, dtype = torch.int32, device = dev),
            idx = torch.arange(len(fan_lins), dtype = torch.long, device = dev).unsqueeze(0),
        )
        fk = fan_inner[0]

        # q-side fan (q_b + idx_wq_b as one mgemm over q_res), when the module built one
        if not m.x_fan_ready:
            m._build_x_fan()
        self.fan2 = m.q_fan if self.has_idx else None

        # Per-step device state: (6, MAX_B) [pos, floor, beg, ec, slot, klen] + block
        # tables. Pinned staging is a ring: the host runs ahead of the replay stream
        num_pages = kl.num_pages if self.has_comp else 1
        self.arr = torch.zeros((6, self.MAX_B), dtype = torch.int32, device = dev)
        self.bt_st = torch.zeros((self.MAX_B, num_pages), dtype = torch.int32, device = dev)
        self.pins = [torch.zeros((6, self.MAX_B), dtype = torch.int32, pin_memory = True)
                     for _ in range(8)]
        self.pin_i = 0

        mu = m.wo_a_multi
        comp, idx = m.compressor, m.indexer
        self.bc = ext.BC_DSV4BatchAttention(
            _exl3_bc(m.q_b), _exl3_bc(m.wo_b),
            _exl3_bc(m.idx_wq_b) if self.has_idx else _exl3_bc(m.q_b), idx_w,
            mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh, m.woa_indices,
            mu.K, mu.mcg, mu.mul1,
            self.fan["trellis"], self.fan["suh"], self.fan["svh"],
            self.fan["n"], self.fan["idx"], fk.K, fk.mcg, fk.mul1,
            m.q_norm.weight.data, m.q_ones, m.kv_norm_w,
            m._rope_type(), m._rope_type_neg(), m.sinks,
            comp.ape if self.has_comp else None,
            comp.fused_norm_w if self.has_comp else None,
            comp.fused_inv_freq if self.has_comp else None,
            comp.norm.rms_norm_eps if self.has_comp else 0.0,
            idx.ape if self.has_idx else None,
            idx.fused_norm_w if self.has_idx else None,
            idx.fused_inv_freq if self.has_idx else None,
            idx.norm.rms_norm_eps if self.has_idx else 0.0,
            rsl.ring,
            rsl.comp_buf_kv if self.has_comp else None,
            rsl.comp_buf_gate if self.has_comp else None,
            rsl.comp_ovl if (self.has_comp and rsl.comp_ovl is not None) else None,
            kl.pool_c.view(-1, kl.D_c) if self.has_comp else None,
            kl.pool_r.view(-1, kl.D_r) if self.has_comp else None,
            rsl.idx_buf_kv if self.has_idx else None,
            rsl.idx_buf_gate if self.has_idx else None,
            rsl.idx_ovl if self.has_idx else None,
            kl.pool_idx.view(-1, kl.D_i) if self.has_idx else None,
            self.arr, self.bt_st,
            self.hidden, self.H, self.head_dim, self.rd, q_lora,
            self.G, self.o_lora,
            m.index_n_heads if self.has_idx else 1,
            m.index_head_dim if self.has_idx else 1,
            m.index_topk if self.has_idx else 1,
            self.window, self.m_rate, self.epp, m.sm_scale, m.rms_norm_eps,
            8, BLOCK_H,
            self.fan2["trellis"] if self.fan2 else None,
            self.fan2["suh"] if self.fan2 else None,
            self.fan2["svh"] if self.fan2 else None,
            self.fan2["n"] if self.fan2 else None,
            self.fan2["idx"] if self.fan2 else None,
            self.fan2["K"] if self.fan2 else 0,
            self.fan2["mcg"] if self.fan2 else False,
            self.fan2["mul1"] if self.fan2 else False,
        )
        cap = kl.capacity if self.has_comp else PAGE_SIZE
        self.scores_max = -(-cap // 128) * 128

    def _configure(self, B, S):
        dev = self.device
        gtc = g_tensor_cache
        R = B * S
        hd, H, G = self.head_dim, self.H, self.G
        hpg = H // G
        m_mod = self.module

        def st(shape, dtype, tag):
            return gtc.get(dev, shape, dtype, tag)

        # Padded like BCDsa: the wts hgemm runs full-height so cuBLASLt picks its
        # M >= 8 kernel; everything else reads the R-row view
        R_p = max(R, 8)
        x_st = st((R_p, self.hidden), torch.half, "bcdb_x")
        x_st.zero_()
        qa_st = st((R, self.q_lora), torch.half, "bcdb_qa")
        qres_st = st((R, self.q_lora), torch.half, "bcdb_qres")
        q_st = st((R, H * hd), torch.half, "bcdb_q")
        kv_st = st((R, hd), torch.half, "bcdb_kv")
        xh_b = st((R, self.q_lora), torch.half, "bcdb_xhb")
        mgc_comp = mgc_idx = None
        if self.has_comp:
            mgc_comp = st((2, R, m_mod.compressor.wkv.out_features), torch.half, "bcdb_mgc")
        if self.has_idx:
            mgc_idx = st((2, R, m_mod.indexer.wkv.out_features), torch.half, "bcdb_mgi")

        qidx_st = wts_st = scores_st = indices_st = None
        k_fewq = None
        if self.has_idx:
            Hi, Di = m_mod.index_n_heads, m_mod.index_head_dim
            qidx_st = st((R, Hi * Di), torch.half, "bcdb_qidx")
            wts_st = st((R_p, Hi), torch.half, "bcdb_wts")
            scores_st = st((R, self.scores_max), torch.half, "bcdb_scores")
            indices_st = st((R, self.kp), torch.int32, "bcdb_indices")
            sig = {
                "q_idx": "*fp16:16", "w": "*fp16:16", "k_idx": "*fp16:16", "scores": "*fp16:16",
                "T": "*i32:16", "R": "i32", "q_pos0": "*i32:16", "bound_max": "*i32:16",
                "block_table": "*i32:16", "num_pages_per_row": "i32",
            } | {n: "constexpr" for n in (
                "H_i", "H_pad", "D_i", "S_stride", "compress_rate", "scale", "BLOCK_N",
                "SEQ", "MULTIROW", "EPP", "DEBUG_BOUNDS", "DEBUG_PAGES")}
            consts = dict(
                H_i = Hi, H_pad = max(16, 1 << (Hi - 1).bit_length()), D_i = Di,
                S_stride = self.scores_max, compress_rate = self.m_rate,
                scale = Di ** -0.5 * Hi ** -0.5, BLOCK_N = 128,
                SEQ = S, MULTIROW = 1, EPP = self.epp, DEBUG_BOUNDS = 0, DEBUG_PAGES = 0,
            )
            k_fewq = _compile_kernel(dev, _dsa_indexer_fewq_kernel, sig, consts, 8, 2)

        hb = -(-H // BLOCK_H)
        n_splits = 8
        ws_ml = st((R * hb * n_splits * BLOCK_H * 2,), torch.float, "bcdb_wsml")
        ws_acc = st((R * hb * n_splits * BLOCK_H * hd,), torch.float, "bcdb_wsacc")
        attn_out = st((G, R, hpg * hd), torch.half, "bcdb_aout")
        woa_c = st((G, R, self.o_lora), torch.half, "bcdb_woac")
        woa_t = st((R, G * self.o_lora), torch.half, "bcdb_woat")
        woa_xh = st((G * R, hpg * hd), torch.half, "bcdb_woaxh")
        y_st = st((R, self.hidden), torch.float, "bcdb_y")

        D_c = hd - self.rd
        sig_s = {
            "q": "*fp16:16", "ring": "*fp16:16", "kv_chunk": "*fp16:16", "pool_c": "*fp16:16",
            "pool_r": "*fp16:16", "block_table": "*i32:16", "indices": "*i32:16",
            "ws_ml": "*fp32:16", "ws_acc": "*fp32:16",
            "k_len": "*i32:16", "win_len": "i32", "pool_len": "*i32:16",
            "num_pages_per_row": "i32", "q_pos0": "*i32:16", "win_floor": "*i32:16",
            "ring_beg": "*i32:16", "slot_ids": "*i32:16", "ring_stride": "i32",
        } | {n: "constexpr" for n in (
            "H", "page_size", "D_c", "D_c_pad", "D_r", "K_pad", "compress_rate", "scale",
            "HAS_WINDOW", "DENSE_POOL", "BLOCK_H", "BLOCK_N", "BLOCK_W", "SEQ", "MULTIROW", "DEBUG_BOUNDS", "DEBUG_PAGES", "Q_SPLIT", "OUT_LATENT")}
        consts_s = dict(
            H = H, page_size = self.epp, D_c = D_c,
            D_c_pad = 1 << (D_c - 1).bit_length(), D_r = self.rd, K_pad = self.kp,
            compress_rate = self.m_rate, scale = self.module.sm_scale,
            HAS_WINDOW = True, DENSE_POOL = not self.has_idx,
            BLOCK_H = BLOCK_H, BLOCK_N = 32, BLOCK_W = 16,
            SEQ = S, MULTIROW = 1, DEBUG_BOUNDS = 0, DEBUG_PAGES = 0,
            Q_SPLIT = 0, OUT_LATENT = 0,
        )
        k_split = _compile_kernel(dev, _dsa_attn_split_kernel, sig_s, consts_s, 4, 2)

        sig_c = {
            "ws_ml": "*fp32:16", "ws_acc": "*fp32:16", "sinks": "*fp32:16",
            "derot_inv_freq": "*fp32:16",
            "out": "*fp16:16", "q_pos0": "*i32:16", "R": "i32", "n_splits": "i32",
        } | {n: "constexpr" for n in (
            "H", "D_c", "D_r", "HAS_SINKS", "DEROTATE", "HPG", "BLOCK_H", "BLOCK_D",
            "SEQ", "MULTIROW", "OUT_LATENT")}
        consts_c = dict(
            H = H, D_c = D_c, D_r = self.rd, HAS_SINKS = True, DEROTATE = True,
            HPG = hpg, BLOCK_H = BLOCK_H, BLOCK_D = 128,
            SEQ = S, MULTIROW = 1, OUT_LATENT = 0,
        )
        k_combine = _compile_kernel(dev, _dsa_attn_combine_kernel, sig_c, consts_c, 4, 2)

        outs = [qa_st, kv_st]
        if self.has_comp:
            outs += [mgc_comp[0], mgc_comp[1]]
        if self.has_idx:
            outs += [mgc_idx[0], mgc_idx[1]]
        fan_c = torch.tensor([o.data_ptr() for o in outs], dtype = torch.long, device = dev)
        fan_ah = st((len(outs), R, self.hidden), torch.half, "bcdb_fanah")
        fan2_c = fan2_ah = None
        if self.fan2:
            fan2_c = torch.tensor([q_st.data_ptr(), qidx_st.data_ptr()],
                                  dtype = torch.long, device = dev)
            fan2_ah = st((2, R, self.q_lora), torch.half, "bcdb_fan2ah")
        self.bc.configure_slot(
            B, S, x_st, qa_st, qres_st, q_st, kv_st, xh_b,
            mgc_comp, mgc_idx, qidx_st, wts_st, scores_st, indices_st,
            ws_ml, ws_acc, attn_out, woa_c, woa_t, woa_xh, y_st,
            fan_c, fan_ah, k_split, k_combine, k_fewq, fan2_c, fan2_ah)

    def run(self, x, B, S, pos_l, floor_l, beg_l, ec_l, slot_l, bt):
        """x (B, S, hidden) fp16 contiguous, bt (rows, npr) i32 device batch block table.
        Returns y (B, S, hidden) fp32 (a static: consume before the next BC call)."""
        if self.bc.needs_configure(B, S):
            self._configure(B, S)

        pin = self.pins[self.pin_i]
        self.pin_i = (self.pin_i + 1) % len(self.pins)
        pin[:5, :B].copy_(torch.tensor([pos_l, floor_l, beg_l, ec_l, slot_l],
                                       dtype = torch.int32))
        pin[5, :B] = self.kp
        self.arr.copy_(pin, non_blocking = True)
        if self.has_comp:
            npr = min(bt.shape[1], self.bt_st.shape[1])
            self.bt_st[:B, :npr].copy_(bt[:B, :npr])

        y = self.bc.run(x, B, S)
        return y.view(B, S, self.hidden)


def build_bc_dsa_batch(module, rsl, kl):
    try:
        return BCDsaBatch(module, rsl, kl)
    except Exception:
        if _bc_debug:
            raise
        return None
