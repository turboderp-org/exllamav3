"""
Fused DSv4 compressor kernel (ext.dsv4_compress, exllamav3_ext/dsv4_compress.cu) against an all-at-once torch
reference of the DSV4Compressor math (softmax-gated window pooling, RMS norm, partial rope). The fused path keeps
ring / snapshot state across calls and writes each emitted entry either straight into the pool (row = absolute
entry index, optionally split into a nope / rope pair of destinations) or, with stage_rel, into a per-call
staging buffer at rows [0, n_windows) that the packed-pool quantizer scatters afterwards. Covers CSA-shaped
(overlapping, m = 4), indexer-shaped (overlapping, narrow) and HCA-shaped (non-overlapping, m = 128) compressors,
uneven chunk schedules, and bitwise chunked-vs-whole and staged-vs-direct agreement of the fused path itself.

Empty inputs (test_empty_*) of dsv4_compress and dsv4_pool_quant_scatter: no tokens or no jobs is a no-op (state
and pools untouched); a zero window size m (softmax pooling over nothing), a zero head dim (RMS norm over
nothing), an empty ring or snapshot ring, and batched destinations without rows raise. No CUDA error is left
pending.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def torch_reference(kv_rows, gate_rows, ape, norm_w, eps, inv_freq, m, overlapping, hd):
    """All-at-once reference over the full row history, replicating DSV4Compressor.forward stateless math + norm +
    rope. Returns (n_windows, hd) fp32"""
    total = kv_rows.shape[0]
    nw = total // m
    if nw == 0:
        return torch.zeros((0, hd), dtype = torch.float, device = kv_rows.device)
    W = kv_rows.shape[-1]
    kv = kv_rows[:nw * m].float().view(nw, m, W)
    gate = gate_rows[:nw * m].float().view(nw, m, W) + ape.unsqueeze(0)
    if overlapping:
        new_kv = kv.new_zeros((nw, 2 * m, hd))
        new_gate = gate.new_full((nw, 2 * m, hd), -float("inf"))
        new_kv[:, m:] = kv[..., hd:]
        new_gate[:, m:] = gate[..., hd:]
        if nw > 1:
            new_kv[1:, :m] = kv[:-1, :, :hd]
            new_gate[1:, :m] = gate[:-1, :, :hd]
        kv, gate = new_kv, new_gate
    comp = (kv * gate.softmax(dim = 1)).sum(dim = 1)
    comp = comp * torch.rsqrt(comp.square().mean(-1, keepdim = True) + eps) * norm_w.float()
    rd = inv_freq.shape[0] * 2
    wpos = torch.arange(nw, device = kv.device).float() * m
    theta = wpos[:, None] * inv_freq[None, :]
    cos, sin = theta.cos(), theta.sin()
    rope = comp[:, hd - rd:]
    e, o = rope[:, 0::2], rope[:, 1::2]
    comp[:, hd - rd:] = torch.stack((e * cos - o * sin, o * cos + e * sin), dim = -1).flatten(-2)
    return comp


def run_fused(kv_rows, gate_rows, ape, norm_w, eps, inv_freq, m, overlapping, hd, chunks,
              buf_rows, ovl_depth, split, stage_rel):
    """Drive ext.dsv4_compress chunk by chunk with fresh ring state. Returns the emitted entries (n_windows, hd)"""
    device = kv_rows.device
    total = kv_rows.shape[0]
    W = kv_rows.shape[-1]
    cap = total // m + 8
    ring_kv = torch.zeros((buf_rows, W), dtype = torch.half, device = device)
    ring_gate = torch.zeros((buf_rows, W), dtype = torch.half, device = device)
    ovl = torch.zeros((ovl_depth, 2, m, hd), dtype = torch.float, device = device) if overlapping else None
    if stage_rel:
        # Whole-width staging rows (no split destination), as the packed-pool path passes them
        pool = torch.full((cap, hd), float("nan"), dtype = torch.half, device = device)
        dest_b = None
    elif split:
        wa = hd - inv_freq.shape[0] * 2
        dest_a = torch.zeros((cap, wa), dtype = torch.half, device = device)
        dest_b = torch.zeros((cap, hd - wa), dtype = torch.half, device = device)
    else:
        dest_a = torch.zeros((cap, hd), dtype = torch.half, device = device)
        dest_b = None
    pos = 0
    for c in chunks:
        if stage_rel:
            dest_a = torch.full((c // m + 1, hd), float("nan"), dtype = torch.half, device = device)
        ext.dsv4_compress(
            kv_rows[pos:pos + c], gate_rows[pos:pos + c], ring_kv, ring_gate, ovl,
            ape, norm_w, eps, inv_freq, dest_a, dest_b, pos, None, m, None, None, 0, stage_rel)
        if stage_rel:
            ec0, nw_c = pos // m, (pos + c) // m - pos // m
            pool[ec0:ec0 + nw_c] = dest_a[:nw_c]
        pos += c
    nw = total // m
    if stage_rel:
        return pool[:nw]
    return torch.cat([dest_a[:nw], dest_b[:nw]], dim = -1) if split else dest_a[:nw].clone()


#           tag               hd   W     m    ovl    total  chunks                  split
CASES = [
    ("csa split",      512, 1024, 4,   True,  317, [37, 1, 1, 128, 150], True),
    ("csa whole",      512, 1024, 4,   True,  64,  [64], True),
    ("csa singles",    512, 1024, 4,   True,  23,  [1] * 23, True),
    ("csa tiny-first", 512, 1024, 4,   True,  9,   [2, 1, 6], True),
    ("idx",            128, 256,  4,   True,  317, [37, 1, 1, 128, 150], False),
    ("idx singles",    128, 256,  4,   True,  17,  [1] * 17, False),
    ("hca split",      512, 512,  128, False, 517, [200, 56, 1, 260], True),
    ("hca whole",      512, 512,  128, False, 384, [384], True),
    ("big chunk",      512, 1024, 4,   True,  600, [600], True),        # > buf_rows
    ("big + tail",     512, 1024, 4,   True,  700, [650, 50], True),    # store clamp
]


@pytest.mark.parametrize("stage_rel", [False, True], ids = ["direct", "staged"])
@pytest.mark.parametrize("i,tag,hd,W,m,ovl,total,chunks,split",
                         [pytest.param(i, *c, id = c[0].replace(" ", "_")) for i, c in enumerate(CASES)])
def test_dsv4_compress(device, i, tag, hd, W, m, ovl, total, chunks, split, stage_rel):
    torch.manual_seed(400 + i)
    kv_rows = (torch.randn((total, W), device = device) * 0.7).half()
    gate_rows = (torch.randn((total, W), device = device) * 1.5).half()
    ape = (torch.randn((m, W), device = device) * 0.8).float()
    norm_w = (torch.randn((hd,), device = device) * 0.3 + 1.0).half()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, device = device).float() / 64))
    if hd < 64:
        inv_freq = inv_freq[:hd // 2]
    eps = 1e-6
    args = (kv_rows, gate_rows, ape, norm_w, eps, inv_freq, m, ovl, hd)

    ref = torch_reference(*args)
    got = run_fused(*args, chunks, buf_rows = 256 + m, ovl_depth = 256 // m + 2, split = split,
                    stage_rel = stage_rel)
    got_whole = run_fused(*args, [total], buf_rows = max(256 + m, total + m),
                          ovl_depth = max(256 // m + 2, total // m + 2), split = split, stage_rel = stage_rel)

    assert got.shape[0] == ref.shape[0] > 0, f"{got.shape[0]} windows vs ref {ref.shape[0]}"
    err = (got.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert err < 2e-2, f"rel err {err:.2e}"
    if len(chunks) > 1:
        assert torch.equal(got, got_whole), "chunked != whole"
    if stage_rel:
        direct = run_fused(*args, chunks, buf_rows = 256 + m, ovl_depth = 256 // m + 2, split = split,
                           stage_rel = False)
        assert torch.equal(got, direct), "staged entries differ from the direct pool store"


# --------------------------------------------------------------------------------------------------------------
# Empty inputs

def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("case, error", [
    ("seq0", None),
    ("seq0_pos_tensor", None),
    ("jobs0", None),
    ("m0", "empty compression window"),
    ("hd0", "empty head dim"),
    ("ring0", "empty ring"),
    ("ovl_depth0", "empty snapshot ring"),
    ("dest_rows0", "empty destination"),
    ("pool_bt0", "no pages in block table"),
])
@torch.inference_mode()
def test_empty_dsv4_compress(device, case, error):
    hd, m, slots, cap = 128, 4, 3, 16
    overlap = case == "ovl_depth0"
    W = 2 * hd if overlap else hd
    if case == "hd0":
        hd = W = 0
    if case == "m0":
        m = 0
    batched = case in ("jobs0", "dest_rows0", "pool_bt0")
    jobs = 0 if case == "jobs0" else 2
    seq = 0 if case.startswith("seq0") or case == "jobs0" else 5
    buf_rows = 0 if case == "ring0" else 256 + 4
    rows = jobs * seq if batched else seq
    kv = torch.randn((rows, W), device = device).half()
    gate = torch.randn((rows, W), device = device).half()
    ring_shape = (slots, buf_rows, W) if batched else (buf_rows, W)
    ring_kv = torch.full(ring_shape, 7.0, dtype = torch.half, device = device)
    ring_gate = torch.full(ring_shape, 7.0, dtype = torch.half, device = device)
    ovl = None
    if overlap:
        ovl = torch.full((0, 2, m, hd), 7.0, device = device)
    ape = torch.randn((max(m, 1), W), device = device)
    norm_w = torch.ones((hd,), dtype = torch.half, device = device)
    inv_freq = torch.rand((min(32, hd // 2),), device = device)
    if batched:
        dest_a = torch.full((0 if case == "dest_rows0" else slots, cap, hd), 7.0, dtype = torch.half, device = device)
    else:
        dest_a = torch.full((cap, hd), 7.0, dtype = torch.half, device = device)
    pos_t = torch.full((max(jobs, 1),), 5, dtype = torch.int32, device = device) \
        if batched or case == "seq0_pos_tensor" else None
    slot_ids = torch.arange(jobs, dtype = torch.int32, device = device) if batched else None
    refs = [t.clone() for t in (ring_kv, ring_gate, dest_a)]
    pool_bt = torch.zeros((jobs, 0), dtype = torch.int32, device = device) if case == "pool_bt0" else None
    args = (kv, gate, ring_kv, ring_gate, ovl, ape, norm_w, 1e-6, inv_freq, dest_a, None, 5, pos_t, m,
            slot_ids, pool_bt, 8 if pool_bt is not None else 0, False)
    if error:
        with pytest.raises(RuntimeError, match = f"dsv4_compress: .*{error}"):
            ext.dsv4_compress(*args)
    else:
        ext.dsv4_compress(*args)
    for t, r in zip((ring_kv, ring_gate, dest_a), refs):
        assert torch.equal(t, r)
    _device_still_works(device)


@pytest.mark.parametrize("jobs, seq, nw_max, pos_tensor", [(1, 0, 4, False), (1, 0, 0, True), (2, 0, 0, True), (0, 5, 4, True)])
@torch.inference_mode()
def test_empty_dsv4_pool_quant_scatter(device, jobs, seq, nw_max, pos_tensor):
    D_c, D_r, bits, m, epp, rows = 128, 64, 4, 4, 8, 32
    G = D_c // 32
    stage = torch.randn((jobs, nw_max, D_c + D_r), device = device).half()
    if jobs == 1:
        stage = stage[0]
    pool_q = torch.full((rows, G * bits), 7, dtype = torch.int32, device = device)
    pool_s = torch.full((rows, G), 7.0, dtype = torch.half, device = device)
    pool_r = torch.full((rows, D_r), 7.0, dtype = torch.half, device = device)
    pool_bt = torch.zeros((max(jobs, 1), rows // epp), dtype = torch.int32, device = device)
    pos_t = torch.full((max(jobs, 1),), 3, dtype = torch.int32, device = device) if pos_tensor else None
    refs = [t.clone() for t in (pool_q, pool_s, pool_r)]
    ext.dsv4_pool_quant_scatter(stage, pool_q, pool_s, pool_r, pool_bt, 3, pos_t, m, seq, epp)
    for t, r in zip((pool_q, pool_s, pool_r), refs):
        assert torch.equal(t, r)
    _device_still_works(device)
    with pytest.raises(RuntimeError, match = "bad epp / m"):
        ext.dsv4_pool_quant_scatter(stage, pool_q, pool_s, pool_r, pool_bt, 3, pos_t, 0, seq, epp)


@pytest.mark.parametrize("jobs, pos_tensor", [(1, False), (1, True), (2, True)])
@torch.inference_mode()
def test_dsv4_pool_quant_scatter_rejects_empty_block_table(device, jobs, pos_tensor):
    D_c, D_r, bits, m, epp, rows, seq = 128, 64, 4, 4, 8, 32, 5
    G = D_c // 32
    stage = torch.randn((jobs, seq // m + 1, D_c + D_r), device = device).half()
    if jobs == 1:
        stage = stage[0]
    pool_q = torch.full((rows, G * bits), 7, dtype = torch.int32, device = device)
    pool_s = torch.full((rows, G), 7.0, dtype = torch.half, device = device)
    pool_r = torch.full((rows, D_r), 7.0, dtype = torch.half, device = device)
    pool_bt = torch.zeros((jobs, 0), dtype = torch.int32, device = device)
    pos_t = torch.full((jobs,), 3, dtype = torch.int32, device = device) if pos_tensor else None
    refs = [t.clone() for t in (pool_q, pool_s, pool_r)]
    with pytest.raises(RuntimeError, match = "dsv4_pool_quant_scatter: .*no pages in block table"):
        ext.dsv4_pool_quant_scatter(stage, pool_q, pool_s, pool_r, pool_bt, 3, pos_t, m, seq, epp)
    for t, r in zip((pool_q, pool_s, pool_r), refs):
        assert torch.equal(t, r)
    _device_still_works(device)
