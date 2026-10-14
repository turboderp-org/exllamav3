"""
MLAttention runs attention in absorbed form: per-head K and V are never built, only the latent and the shared
rope key. These tests check that against a direct transcription of the reference (DeepSeek-V2/V3) forward
(testlib.mla.ref_forward), which does build them, over the paged cache the generator would use, and the paths
of the module against each other (chunked/whole prefill, decode, cache-less, quantized latent cache, decode
graph, MHA-form prefill).

RoPE is applied by the module's own RoPE object on both sides, so a failure here is MLA math - absorption, cache
layout, kernels - and not rope conventions, which modules/rope covers.
"""

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.ext import exllamav3_ext as ext

from testlib.compare import rel_err
from testlib.exl3 import checkpoint_tensors, generator
from testlib.mla import KEY, build_mla, load_mla, make_cache, make_qcache, ref_forward, run_module


def block_table(bsz, pages_per_seq, device):
    return torch.arange(pages_per_seq * bsz, dtype = torch.int32, device = device).view(bsz, pages_per_seq)


@pytest.mark.parametrize("q_lora", [None, 256])
@pytest.mark.parametrize("H", [8, 16])
@pytest.mark.parametrize("S", [1, 17, 300])
def test_mla_vs_reference(tmp_path, device, q_lora, H, S):
    """Absorbed path against the explicit per-head reference, single chunk."""
    module, t, key = build_mla(tmp_path, device, H = H, q_lora = q_lora, seed = H + S)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    layer = make_cache(module, 4 * PAGE_SIZE * bsz)
    bt = block_table(bsz, 4, device)

    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)
    ref = ref_forward(module, t, key, x, positions)
    out = run_module(module, x, layer, bt)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"


def test_mla_nope(tmp_path, device):
    """Kimi-Linear style: MLA layers with no RoPE at all."""
    module, t, key = build_mla(tmp_path, device, H = 8, nope_only = True, seed = 7)
    bsz, S = 1, 200
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    layer = make_cache(module, 4 * PAGE_SIZE)
    bt = block_table(1, 4, device)
    ref = ref_forward(module, t, key, x, torch.zeros((bsz,), dtype = torch.int32, device = device))
    out = run_module(module, x, layer, bt)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"


def build_exl3_nope(directory, device, H = 16, hidden = 1024, kv_lora = 512, nope = 128, rope_dim = 64,
                    v_head = 128, seed = 0, K = 4):
    """Kimi Linear shaped MLA with EXL3 (random trellis, mul1) q / kv_a / o projections and no rope instance:
    the decode graph only admits EXL3 projections, so this is the only way to reach it from a unit test. kv_b
    stays fp16 (the module reads it as the absorbed W_UK / W_UV)."""
    g = generator(seed)
    t = {}
    for name, k, n in (
        ("q_proj", hidden, H * (nope + rope_dim)),
        ("kv_a_proj_with_mqa", hidden, kv_lora + rope_dim),
        ("o_proj", H * v_head, hidden),
    ):
        n_pad = (n + 127) // 128 * 128
        t.update(checkpoint_tensors(f"{KEY}.{name}", k, n_pad, K, g, codebook = "mul1"))
    t[f"{KEY}.kv_a_layernorm.weight"] = (torch.randn(kv_lora, generator = g) * 0.1 + 1).half()
    t[f"{KEY}.kv_b_proj.weight"] = (torch.randn(H * (nope + v_head), kv_lora, generator = g) * 0.085).half()

    module = load_mla(t, directory, device, H = H, hidden = hidden, kv_lora = kv_lora, nope = nope,
                      rope_dim = rope_dim, v_head = v_head, nope_only = True)
    assert module.q_proj.quant_type == "exl3"
    return module


def test_mla_nope_decode_matches_prefill(tmp_path, device):
    """Kimi Linear: pe dims present but never rotated. The decode graph used to require a rope instance whenever
    qk_rope_head_dim > 0 (and its C++ ran the rope stage unconditionally, which raises on a rope-less module), so
    this configuration silently fell back to the dispatch path. Now the graph must run, agree with the dispatch
    decode path at fp16 noise, and both must agree with the prefill path."""
    import exllamav3.modules.attention_fn.bc_mla as bcm
    module = build_exl3_nope(tmp_path, device, seed = 13)
    bsz, S = 2, 300
    # Random trellis weights decode to O(1) entries, so a small input keeps the scores O(1)
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.03).half()
    bt = block_table(bsz, 4, device)

    whole = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt)
    assert torch.isfinite(whole).all() and whole.abs().max() > 0
    step_graph = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt, chunk = 1)
    if bcm.bc_attn_enable:
        assert any(bool(v) for v in module.dispatch_cache.values()), \
            "decode graph declined the rope-less MLA configuration"

    enable = bcm.bc_attn_enable
    bcm.bc_attn_enable = False
    module.dispatch_cache.clear()
    try:
        step_dispatch = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt, chunk = 1)
    finally:
        bcm.bc_attn_enable = enable
        module.dispatch_cache.clear()

    # Graph vs dispatch decode: same kernels' worth of fp16 rounding, tight. Decode vs prefill crosses GEMV/GEMM
    # and attention kernels, looser (the fp16-weight variant above sees the same)
    assert rel_err(step_graph, step_dispatch) < 3e-3, \
        f"graph vs dispatch rel err {rel_err(step_graph, step_dispatch):.3e}"
    assert rel_err(step_graph, whole) < 2e-2, f"decode vs prefill rel err {rel_err(step_graph, whole):.3e}"


@pytest.mark.parametrize("chunk", [PAGE_SIZE, 128, 64])
def test_mla_chunked_prefill(tmp_path, device, chunk):
    """Chunked prefill must reproduce the single-shot result: the cache carries the context."""
    module, t, key = build_mla(tmp_path, device, H = 8, q_lora = 256, seed = 3)
    bsz, S = 2, 512
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = block_table(bsz, 4, device)

    whole = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt)
    parts = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt, chunk = chunk)
    assert rel_err(parts, whole) < 5e-3, f"rel err {rel_err(parts, whole):.3e}"


def test_mla_decode_matches_prefill(tmp_path, device):
    """Token-by-token decode must match the prefill result for the same sequence - this is the path that crosses
    from the long-query kernel to the flash-decoding kernel."""
    module, t, key = build_mla(tmp_path, device, H = 16, q_lora = 256, seed = 11)
    bsz, S = 2, 300
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = block_table(bsz, 4, device)

    whole = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt)
    step = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt, chunk = 1)
    assert rel_err(step, whole) < 5e-3, f"rel err {rel_err(step, whole):.3e}"


def test_mla_scrambled_pages(tmp_path, device):
    """Page order in the block table must not matter."""
    module, t, key = build_mla(tmp_path, device, H = 8, seed = 5)
    bsz, S = 1, 700
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    npages = 8
    ordered = block_table(1, npages, device)
    scrambled = torch.tensor([[5, 2, 7, 0, 3, 6, 1, 4]], dtype = torch.int32, device = device)

    a = run_module(module, x, make_cache(module, npages * PAGE_SIZE), ordered)
    b = run_module(module, x, make_cache(module, npages * PAGE_SIZE), scrambled)
    assert rel_err(a, b) == 0.0, f"page order changed the result: {rel_err(a, b):.3e}"


def test_mla_cache_is_latent(tmp_path, device):
    """The cache must hold only the latent and the rope key - no per-head K/V anywhere."""
    module, t, key = build_mla(tmp_path, device, H = 128, seed = 1)
    layer = make_cache(module, 4 * PAGE_SIZE)
    assert layer.k.shape == (4, PAGE_SIZE, 1, module.kv_lora_rank)
    assert layer.v.shape == (4, PAGE_SIZE, 1, module.qk_rope_head_dim)
    per_token = layer.storage_size() / (4 * PAGE_SIZE)
    assert per_token == (module.kv_lora_rank + module.qk_rope_head_dim) * 2
    # vs. what expanded per-head K/V would have cost
    expanded = module.num_q_heads * (module.qk_head_dim + module.v_head_dim) * 2
    assert per_token * 40 < expanded


def test_mla_no_context_sized_temporaries(tmp_path, device):
    """The forward must not allocate anything that scales with context length.

    This is what separates real MLA from an implementation that quietly up-projects the cached latents back into
    per-head K/V for every forward pass: such a path would allocate ctx * H * (qk_head_dim + v_head_dim) * 2 bytes
    here and give up the whole point of MLA."""
    H = 32
    module, t, key = build_mla(tmp_path, device, H = H, seed = 2)
    npages = 64
    layer = make_cache(module, npages * PAGE_SIZE)
    bt = block_table(1, npages, device)
    x = (torch.randn((1, 1, module.hidden_size), device = device) * 0.5).half()

    def peak_for(ctx):
        seqlens = torch.full((1,), ctx, dtype = torch.int32, device = device)
        params = {
            "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
            "cache_seqlens": seqlens, "positions": seqlens.clone(),
        }
        module.forward(x, params)          # warm: kernel compile, scratch, block-table upload
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        module.forward(x, params)
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) - base

    small, large = peak_for(1024), peak_for(8192)
    expanded = (8192 - 1024) * H * (module.qk_head_dim + module.v_head_dim) * 2
    assert large - small < expanded * 0.01, (
        f"working set grew {large - small} bytes from 1k to 8k context; an expanded-K/V path "
        f"would grow by {expanded}"
    )


@pytest.mark.parametrize("q_lora", [None, 256])
@pytest.mark.parametrize("S", [1, 200, 600])
def test_mla_nocache(tmp_path, device, q_lora, S):
    """Cache-less path (attn_mode flash_attn_nc), which is what the quantization calibration forward uses. Must
    match both the reference and the cached path."""
    module, t, key = build_mla(tmp_path, device, H = 8, q_lora = q_lora, seed = S)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)
    ref = ref_forward(module, t, key, x, positions)

    nc = module.forward(x, {"attn_mode": "flash_attn_nc", "positions": positions})
    assert rel_err(nc, ref) < 5e-3, f"vs reference: {rel_err(nc, ref):.3e}"

    npages = (S + PAGE_SIZE - 1) // PAGE_SIZE
    bt = block_table(bsz, npages, device)
    cached = run_module(module, x, make_cache(module, npages * PAGE_SIZE * bsz), bt)
    assert rel_err(nc, cached) < 2e-3, f"vs cached path: {rel_err(nc, cached):.3e}"


# Quantized latent cache

@pytest.mark.parametrize("bits", [2, 4, 5, 8])
@pytest.mark.parametrize("q_len,kv_len,splits", [(1, 1000, None), (1, 4096, 1), (16, 2048, 3), (300, 2048, None)])
def test_mla_qc_kernel_vs_dequant_reference(device, bits, q_len, kv_len, splits):
    """The qc kernels must reproduce the fp16 kernels running on the exact values the packed cache represents
    (quant->dequant roundtrip through the same CUDA kernels). This isolates the loaders, the H32 fold and the
    scatter from quantization error itself, which cancels."""
    from exllamav3.modules.attention_fn.mla_triton import (
        mla_attn_triton_decode, mla_attn_triton_prefill, mla_kv_quant_append, mla_kv_append,
    )
    torch.manual_seed(bits * 1000 + kv_len)
    H, D_c, D_r = 16, 512, 64
    bsz = 2
    groups = D_c // 32
    npages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = npages * bsz
    dev = device

    # Scrambled pages: the scatter and the loaders must agree through the block table
    bt = torch.randperm(total_pages, dtype = torch.int32, device = dev).view(bsz, npages)
    seqlens = torch.full((bsz,), kv_len - q_len, dtype = torch.int32, device = dev)

    ckv_rows = (torch.randn((bsz, kv_len, D_c), device = dev) * 0.1).half()
    kpe_rows = (torch.randn((bsz, kv_len, D_r), device = dev) * 0.1).half()

    # Packed cache via the production append
    qk = torch.zeros((total_pages, PAGE_SIZE, groups * bits), dtype = torch.int, device = dev)
    sk = torch.zeros((total_pages, PAGE_SIZE, groups), dtype = torch.half, device = dev)
    kpe_q = torch.zeros((total_pages, PAGE_SIZE, 1, D_r), dtype = torch.half, device = dev)
    zero = torch.zeros((bsz,), dtype = torch.int32, device = dev)
    mla_kv_quant_append(ckv_rows, kpe_rows, qk, sk, kpe_q, bt, zero, bits)

    # fp16 cache holding the values the packed cache represents
    tmp_q = torch.empty((bsz * kv_len, groups * bits), dtype = torch.int, device = dev)
    tmp_s = torch.empty((bsz * kv_len, groups), dtype = torch.half, device = dev)
    ext.quant_cache_cont(ckv_rows.reshape(-1, D_c).contiguous(), tmp_q, tmp_s, 0.0)
    deq = torch.empty((bsz * kv_len, D_c), dtype = torch.half, device = dev)
    ext.dequant_cache_cont(tmp_q, tmp_s, deq, 0.0)
    ckv_f = torch.zeros((total_pages, PAGE_SIZE, 1, D_c), dtype = torch.half, device = dev)
    kpe_f = torch.zeros((total_pages, PAGE_SIZE, 1, D_r), dtype = torch.half, device = dev)
    mla_kv_append(deq.view(bsz, kv_len, D_c), kpe_rows, ckv_f, kpe_f, bt, zero)

    R = bsz * q_len
    q_lat = (torch.randn((H, R, D_c), device = dev) * 0.1).half()
    q_pe = (torch.randn((H, R, D_r), device = dev) * 0.1).half()

    if q_len <= 16:
        kw = dict(num_splits = splits)
        o_ref = mla_attn_triton_decode(q_lat, q_pe, ckv_f, kpe_f, bt, seqlens, bsz, q_len,
                                       pre_appended_len = q_len, **kw)
        o_qc = mla_attn_triton_decode(q_lat, q_pe, qk, kpe_q, bt, seqlens, bsz, q_len,
                                      pre_appended_len = q_len, qc = (sk, bits), **kw)
    else:
        o_ref = mla_attn_triton_prefill(q_lat, q_pe, ckv_f, kpe_f, bt, seqlens, bsz, q_len,
                                        pre_appended_len = q_len)
        o_qc = mla_attn_triton_prefill(q_lat, q_pe, qk, kpe_q, bt, seqlens, bsz, q_len,
                                       pre_appended_len = q_len, qc = (sk, bits))
    err = rel_err(o_qc, o_ref)
    assert err < 5e-3, f"bits {bits} q {q_len} kv {kv_len}: rel err {err:.3e}"


def test_mla_quant_cache_q8_close_to_fp16(tmp_path, device):
    """Q8 latent quantization is near-lossless; the module output should track the fp16 cache."""
    module, t, key = build_mla(tmp_path, device, H = 8, q_lora = 256, seed = 21)
    bsz, S = 2, 300
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = block_table(bsz, 4, device)
    fp = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt)
    q8 = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, 8), bt)
    assert rel_err(q8, fp) < 3e-2, f"rel err {rel_err(q8, fp):.3e}"


@pytest.mark.parametrize("bits,tol", [(4, 8e-2), (6, 4e-2)])
def test_mla_quant_cache_consistency(tmp_path, device, bits, tol):
    """Chunked prefill, whole-shot prefill and token-by-token decode must stay in the same quantization-error
    band of each other.

    They are NOT near-identical the way the fp16 cache is: the projection GEMMs tile differently per chunk shape,
    so the pre-quantization rows differ at ulp level, and the quantizer turns a near-boundary ulp into a full
    discrete code flip (~1% of packed words at Q4). The fp16 rope pages differing at ulp level across chunkings
    confirms the diff originates upstream of the cache. The bitwise-level correctness bar is
    test_mla_qc_kernel_vs_dequant_reference, where the cache content is fixed by construction."""
    module, t, key = build_mla(tmp_path, device, H = 8, seed = 23)
    bsz, S = 2, 300
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = block_table(bsz, 4, device)
    whole = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, bits), bt)
    parts = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, bits), bt, chunk = 128)
    step = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, bits), bt, chunk = 1)
    assert rel_err(parts, whole) < tol, f"chunked vs whole: {rel_err(parts, whole):.3e}"
    assert rel_err(step, whole) < tol, f"decode vs whole: {rel_err(step, whole):.3e}"


@pytest.mark.parametrize("bits,tol", [(8, 2e-2), (6, 6e-2), (4, 1.5e-1)])
def test_mla_quant_cache_vs_reference(tmp_path, device, bits, tol):
    """Sanity bound against the exact reference: quantization error should shrink with bits and stay in the
    expected band. (The tight correctness bar is the dequant-reference kernel test.)"""
    module, t, key = build_mla(tmp_path, device, H = 8, seed = 25)
    bsz, S = 2, 300
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = block_table(bsz, 4, device)
    ref = ref_forward(module, t, key, x, torch.zeros((bsz,), dtype = torch.int32, device = device))
    out = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, bits), bt)
    err = rel_err(out, ref)
    assert err < tol, f"Q{bits}: rel err {err:.3e}"


def test_mla_prefill_mode_equivalence(tmp_path, device):
    """The MHA-form prefill (up-projected past tiles) and the absorbed prefill must agree; they compute the same
    attention in different factorizations."""
    import exllamav3.modules.mla_attn as M
    module, t, key = build_mla(tmp_path, device, H = 8, q_lora = 256, seed = 31)
    bsz, S = 2, 600
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = block_table(bsz, 4, device)
    saved = M._prefill_mode
    try:
        M._prefill_mode = "mha"
        o_mha = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt)
        M._prefill_mode = "absorbed"
        o_abs = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt)
    finally:
        M._prefill_mode = saved
    assert rel_err(o_mha, o_abs) < 2e-3, f"mha vs absorbed: {rel_err(o_mha, o_abs):.3e}"


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_mla_gather_tile(device, bits):
    """The tile gather kernel must reproduce the exact latent/rope rows the cache holds (via dequant_cache_cont
    for the packed case, including the inverse H32 rotation)."""
    from exllamav3.modules.attention_fn.mla_triton import (
        _mla_gather_tile_kernel, mla_kv_append, mla_kv_quant_append,
    )
    from exllamav3.modules.attention_fn.triton_paged import _get_h32
    import triton
    torch.manual_seed(bits + 41)
    D_c, D_r, PS = 512, 64, PAGE_SIZE
    kv_len, npages = 900, 4
    dev = device
    bt = torch.randperm(npages, dtype = torch.int32, device = dev).view(1, npages)
    zero = torch.zeros((1,), dtype = torch.int32, device = dev)
    ckv_rows = (torch.randn((1, kv_len, D_c), device = dev) * 0.1).half()
    kpe_rows = (torch.randn((1, kv_len, D_r), device = dev) * 0.1).half()

    if bits == 0:
        ck = torch.zeros((npages, PS, 1, D_c), dtype = torch.half, device = dev)
        kp = torch.zeros((npages, PS, 1, D_r), dtype = torch.half, device = dev)
        mla_kv_append(ckv_rows, kpe_rows, ck, kp, bt, zero)
        want = ckv_rows[0]
        sk, h32, qbits = ck, ck, 0
    else:
        groups = D_c // 32
        ck = torch.zeros((npages, PS, groups * bits), dtype = torch.int, device = dev)
        sk = torch.zeros((npages, PS, groups), dtype = torch.half, device = dev)
        kp = torch.zeros((npages, PS, 1, D_r), dtype = torch.half, device = dev)
        mla_kv_quant_append(ckv_rows, kpe_rows, ck, sk, kp, bt, zero, bits)
        tmp_q = torch.empty((kv_len, groups * bits), dtype = torch.int, device = dev)
        tmp_s = torch.empty((kv_len, groups), dtype = torch.half, device = dev)
        ext.quant_cache_cont(ckv_rows[0].contiguous(), tmp_q, tmp_s, 0.0)
        want = torch.empty((kv_len, D_c), dtype = torch.half, device = dev)
        ext.dequant_cache_cont(tmp_q, tmp_s, want, 0.0)
        h32, qbits = _get_h32(dev), bits

    a, e = 100, 800   # tile crossing page boundaries
    out_c = torch.empty((e - a, D_c), dtype = torch.half, device = dev)
    out_r = torch.empty((e - a, D_r), dtype = torch.half, device = dev)
    _mla_gather_tile_kernel[(triton.cdiv(e - a, 64),)](
        ck, kp, sk, h32, bt, out_c, out_r, a, e - a, npages, 0,
        qbits, PS, D_c, D_r, 64, num_warps = 4, num_stages = 2,
    )
    err_c = (out_c.float() - want[a:e].float()).abs().max().item()
    err_r = (out_r.float() - kpe_rows[0, a:e].float()).abs().max().item()
    tol = 1e-3 if bits else 0.0   # unrotation is a small fp16 dot; fp16 gather must be exact
    assert err_c <= tol, f"latent gather err {err_c:.3e}"
    assert err_r == 0.0, f"kpe gather err {err_r:.3e}"


def test_mla_kv_b_export_roundtrip(tmp_path, device):
    """get_tensors must reconstruct kv_b_proj.weight in the exact checkpoint layout from the flat storage (the
    conversion pipeline carries it into quantized models verbatim)."""
    module, t, key = build_mla(tmp_path, device, H = 8, seed = 51)
    got = module.get_tensors()[f"{key}.kv_b_proj.weight"]
    want = t[f"{key}.kv_b_proj.weight"]
    assert got.shape == want.shape
    assert torch.equal(got.cpu().half(), want.cpu().half()), "kv_b reconstruction differs"
