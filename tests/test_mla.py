import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import pytest
import torch

from exllamav3.modules import MLAttention
from exllamav3.cache import CacheLayer_MLA_fp16
from exllamav3.constants import PAGE_SIZE
from exllamav3.util.rope import RopeSettings, RopeStyle
from exllamav3.modules.attention_fn.mla_triton import has_triton

# MLAttention runs attention in absorbed form: per-head K and V are never built, only the latent
# and the shared rope key. These tests check that against a direct transcription of the reference
# (DeepSeek-V2/V3) forward, which does build them, over the paged cache the generator would use.
#
# RoPE is applied by the module's own RoPE object on both sides, so a failure here is MLA math -
# absorption, cache layout, kernels - and not rope conventions, which test_rope.py covers.

device = "cuda:0"
pytestmark = pytest.mark.skipif(not has_triton, reason = "requires Triton")


class FakeSTC:
    """Minimal stand-in for the safetensors collection, serving tensors from a dict."""

    def __init__(self, tensors):
        self.tensors = tensors

    def has_tensor(self, key):
        return key in self.tensors

    def has_tensor_group(self, key, subkeys):
        if isinstance(key, list):
            return all(self.has_tensor_group(k, subkeys) for k in key)
        return all(
            (f"{key}.{sk}" in self.tensors if isinstance(sk, str)
             else any(f"{key}.{s}" in self.tensors for s in sk))
            for sk in subkeys
        )

    def get_tensor(self, key, device = None, optional = False, allow_bf16 = False,
                   float2half = False, no_defer = False, transpose = False, pad_to = None,
                   fidx = None):
        if key not in self.tensors:
            if optional:
                return None
            raise ValueError(f"Required tensor {key} not found")
        x = self.tensors[key].to(device if device is not None else "cpu")
        if float2half and x.dtype in (torch.float32, torch.float64, torch.bfloat16):
            x = x.half()
        if transpose:
            x = x.T.contiguous()
        if pad_to is not None:
            pad = []
            for i in range(len(pad_to) - 1, -1, -1):
                pad += [0, max(0, pad_to[i] - x.shape[i])]
            if any(pad):
                x = torch.nn.functional.pad(x, pad)
        return x.contiguous()


class FakeConfig:
    def __init__(self, tensors):
        self.stc = FakeSTC(tensors)


def build(H = 8, hidden = 1024, kv_lora = 512, nope = 128, rope_dim = 64, v_head = 128,
          q_lora = None, nope_only = False, seed = 0, wscale = 0.085):
    """Random-weight MLAttention plus the raw weights, for the reference forward.

    wscale is chosen so pre-softmax scores land around unit standard deviation. The q and k paths
    both pass through an RMSNorm, so the input scale is normalized away and it is the weight scale
    alone that sets the score magnitude - at the obvious 0.02 the scores have std 0.12, softmax is
    nearly uniform, and the comparison goes blind to score errors (a 1% W_UK perturbation moved the
    output by less than the fp16 noise floor)."""
    g = torch.Generator(device = "cpu").manual_seed(seed)

    def rnd(*shape, scale = wscale):
        return (torch.randn(*shape, generator = g) * scale).half()

    key = "model.layers.0.self_attn"
    qk_head = nope + rope_dim
    t = {
        f"{key}.kv_a_proj_with_mqa.weight": rnd(kv_lora + rope_dim, hidden),
        f"{key}.kv_a_layernorm.weight": (torch.randn(kv_lora, generator = g) * 0.1 + 1).half(),
        f"{key}.kv_b_proj.weight": rnd(H * (nope + v_head), kv_lora),
        f"{key}.o_proj.weight": rnd(hidden, H * v_head),
    }
    if q_lora is None:
        t[f"{key}.q_proj.weight"] = rnd(H * qk_head, hidden)
    else:
        t[f"{key}.q_a_proj.weight"] = rnd(q_lora, hidden)
        t[f"{key}.q_a_layernorm.weight"] = (torch.randn(q_lora, generator = g) * 0.1 + 1).half()
        t[f"{key}.q_b_proj.weight"] = rnd(H * qk_head, q_lora)

    rope_settings = None if nope_only else RopeSettings(
        head_dim = rope_dim, rope_theta = 10000.0, rope_style = RopeStyle.NEOX,
    )
    module = MLAttention(
        config = FakeConfig(t), key = key, layer_idx = 0, hidden_size = hidden,
        num_q_heads = H, kv_lora_rank = kv_lora, qk_nope_head_dim = nope,
        qk_rope_head_dim = rope_dim, v_head_dim = v_head, rope_settings = rope_settings,
        q_lora_rank = q_lora, rms_norm_eps = 1e-6,
    )
    module.load(torch.device(device))
    return module, {k: v.to(device) for k, v in t.items()}, key


def rms_norm(x, w, eps):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + eps) * w.float())


def ref_forward(module, t, key, x, positions):
    """Reference MLA: build per-head K and V explicitly, then plain MHA. This is the form the
    absorbed path must reproduce."""
    m = module
    bsz, S, _ = x.shape
    H, nope, rope_dim, v_head = m.num_q_heads, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim

    xf = x.float()
    if m.q_lora_rank is None:
        q = xf @ t[f"{key}.q_proj.weight"].float().T
    else:
        q = xf @ t[f"{key}.q_a_proj.weight"].float().T
        q = rms_norm(q, t[f"{key}.q_a_layernorm.weight"], m.norm_eps)
        q = q @ t[f"{key}.q_b_proj.weight"].float().T
    q = q.view(bsz, S, H, m.qk_head_dim)
    q_nope, q_pe = q[..., :nope], q[..., nope:]

    ckv_kpe = xf @ t[f"{key}.kv_a_proj_with_mqa.weight"].float().T
    ckv = rms_norm(ckv_kpe[..., :m.kv_lora_rank], t[f"{key}.kv_a_layernorm.weight"], m.norm_eps)
    k_pe = ckv_kpe[..., m.kv_lora_rank:].view(bsz, S, 1, rope_dim)

    # Same RoPE object as the module, so this isolates the MLA math
    if m.rope is not None:
        q_pe, k_pe = m.rope.apply(
            q_pe.half().contiguous(), k_pe.half().contiguous(),
            0, positions, None, False, None, None, m.norm_eps, 0.0, None, False,
        )
    q_pe, k_pe = q_pe.float(), k_pe.float()

    kv = (ckv.half().float() @ t[f"{key}.kv_b_proj.weight"].float().T).view(bsz, S, H, nope + v_head)
    k_nope, v = kv[..., :nope], kv[..., nope:]

    k = torch.cat([k_nope, k_pe.expand(bsz, S, H, rope_dim)], dim = -1)
    q_full = torch.cat([q_nope, q_pe], dim = -1)

    scores = torch.einsum("bqhd,bkhd->bhqk", q_full, k) * m.sm_scale
    mask = torch.arange(S, device = x.device)[None, :] <= torch.arange(S, device = x.device)[:, None]
    scores = scores.masked_fill(~mask[None, None], -float("inf"))
    p = torch.softmax(scores, dim = -1)
    o = torch.einsum("bhqk,bkhd->bqhd", p, v).reshape(bsz, S, H * v_head)
    return o @ t[f"{key}.o_proj.weight"].float().T


def make_cache(module, max_tokens):
    layer = CacheLayer_MLA_fp16(None, module, 0, max_tokens)
    layer.alloc(torch.device(device))
    return layer


def run_module(module, x, layer, bt, chunk = None):
    """Feed x through the module in one or more chunks, as the generator would."""
    bsz, S, _ = x.shape
    chunk = chunk or S
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = device)
    outs = []
    for a in range(0, S, chunk):
        b = min(a + chunk, S)
        params = {
            "attn_mode": "flash_attn",
            "cache": layer,
            "block_table": bt,
            "cache_seqlens": seqlens,
            "positions": seqlens.clone(),
        }
        outs.append(module.forward(x[:, a:b].contiguous(), params))
        seqlens = seqlens + (b - a)
    return torch.cat(outs, dim = 1)


def rel_err(a, b):
    return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-6)


@pytest.mark.parametrize("q_lora", [None, 256])
@pytest.mark.parametrize("H", [8, 16])
@pytest.mark.parametrize("S", [1, 17, 300])
def test_mla_vs_reference(q_lora, H, S):
    """Absorbed path against the explicit per-head reference, single chunk."""
    module, t, key = build(H = H, q_lora = q_lora, seed = H + S)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    layer = make_cache(module, 4 * PAGE_SIZE * bsz)
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)

    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)
    ref = ref_forward(module, t, key, x, positions)
    out = run_module(module, x, layer, bt)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"


def test_mla_nope():
    """Kimi-Linear style: MLA layers with no RoPE at all."""
    module, t, key = build(H = 8, nope_only = True, seed = 7)
    bsz, S = 1, 200
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    layer = make_cache(module, 4 * PAGE_SIZE)
    bt = torch.arange(4, dtype = torch.int32, device = device).view(1, 4)
    ref = ref_forward(module, t, key, x, torch.zeros((bsz,), dtype = torch.int32, device = device))
    out = run_module(module, x, layer, bt)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"


@pytest.mark.parametrize("chunk", [PAGE_SIZE, 128, 64])
def test_mla_chunked_prefill(chunk):
    """Chunked prefill must reproduce the single-shot result: the cache carries the context."""
    module, t, key = build(H = 8, q_lora = 256, seed = 3)
    bsz, S = 2, 512
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)

    whole = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt)
    parts = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt, chunk = chunk)
    assert rel_err(parts, whole) < 5e-3, f"rel err {rel_err(parts, whole):.3e}"


def test_mla_decode_matches_prefill():
    """Token-by-token decode must match the prefill result for the same sequence - this is the
    path that crosses from the long-query kernel to the flash-decoding kernel."""
    module, t, key = build(H = 16, q_lora = 256, seed = 11)
    bsz, S = 2, 300
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)

    whole = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt)
    step = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt, chunk = 1)
    assert rel_err(step, whole) < 5e-3, f"rel err {rel_err(step, whole):.3e}"


def test_mla_scrambled_pages():
    """Page order in the block table must not matter."""
    module, t, key = build(H = 8, seed = 5)
    bsz, S = 1, 700
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    npages = 8
    ordered = torch.arange(npages, dtype = torch.int32, device = device).view(1, npages)
    scrambled = torch.tensor([[5, 2, 7, 0, 3, 6, 1, 4]], dtype = torch.int32, device = device)

    a = run_module(module, x, make_cache(module, npages * PAGE_SIZE), ordered)
    b = run_module(module, x, make_cache(module, npages * PAGE_SIZE), scrambled)
    assert rel_err(a, b) == 0.0, f"page order changed the result: {rel_err(a, b):.3e}"


def test_mla_cache_is_latent():
    """The cache must hold only the latent and the rope key - no per-head K/V anywhere."""
    module, t, key = build(H = 128, seed = 1)
    layer = make_cache(module, 4 * PAGE_SIZE)
    assert layer.k.shape == (4, PAGE_SIZE, 1, module.kv_lora_rank)
    assert layer.v.shape == (4, PAGE_SIZE, 1, module.qk_rope_head_dim)
    per_token = layer.storage_size() / (4 * PAGE_SIZE)
    assert per_token == (module.kv_lora_rank + module.qk_rope_head_dim) * 2
    # vs. what expanded per-head K/V would have cost
    expanded = module.num_q_heads * (module.qk_head_dim + module.v_head_dim) * 2
    assert per_token * 40 < expanded


def test_mla_no_context_sized_temporaries():
    """The forward must not allocate anything that scales with context length.

    This is what separates real MLA from an implementation that quietly up-projects the cached
    latents back into per-head K/V for every forward pass: such a path would allocate
    ctx * H * (qk_head_dim + v_head_dim) * 2 bytes here and give up the whole point of MLA.
    """
    H = 32
    module, t, key = build(H = H, seed = 2)
    npages = 64
    layer = make_cache(module, npages * PAGE_SIZE)
    bt = torch.arange(npages, dtype = torch.int32, device = device).view(1, npages)
    x = (torch.randn((1, 1, module.hidden_size), device = device) * 0.5).half()

    def peak_for(ctx):
        seqlens = torch.full((1,), ctx, dtype = torch.int32, device = device)
        params = {
            "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
            "cache_seqlens": seqlens, "positions": seqlens.clone(),
        }
        module.forward(x, params)          # warm: kernel compile, scratch, block-table upload
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        module.forward(x, params)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) - base

    small, large = peak_for(1024), peak_for(8192)
    expanded = (8192 - 1024) * H * (module.qk_head_dim + module.v_head_dim) * 2
    assert large - small < expanded * 0.01, (
        f"working set grew {large - small} bytes from 1k to 8k context; an expanded-K/V path "
        f"would grow by {expanded}"
    )


@pytest.mark.parametrize("q_lora", [None, 256])
@pytest.mark.parametrize("S", [1, 200, 600])
def test_mla_nocache(q_lora, S):
    """Cache-less path (attn_mode flash_attn_nc), which is what the quantization calibration
    forward uses. Must match both the reference and the cached path."""
    module, t, key = build(H = 8, q_lora = q_lora, seed = S)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)
    ref = ref_forward(module, t, key, x, positions)

    nc = module.forward(x, {"attn_mode": "flash_attn_nc", "positions": positions})
    assert rel_err(nc, ref) < 5e-3, f"vs reference: {rel_err(nc, ref):.3e}"

    npages = (S + PAGE_SIZE - 1) // PAGE_SIZE
    bt = torch.arange(npages * bsz, dtype = torch.int32, device = device).view(bsz, npages)
    cached = run_module(module, x, make_cache(module, npages * PAGE_SIZE * bsz), bt)
    assert rel_err(nc, cached) < 2e-3, f"vs cached path: {rel_err(nc, cached):.3e}"


# ---- quantized latent cache ----------------------------------------------------------------

from exllamav3.cache import CacheLayer_MLA_quant
from exllamav3.ext import exllamav3_ext as ext


def make_qcache(module, max_tokens, bits):
    layer = CacheLayer_MLA_quant(None, module, 0, max_tokens, k_bits = bits)
    layer.alloc(torch.device(device))
    return layer


@pytest.mark.parametrize("bits", [2, 4, 5, 8])
@pytest.mark.parametrize("q_len,kv_len,splits", [(1, 1000, None), (1, 4096, 1), (16, 2048, 3), (300, 2048, None)])
def test_mla_qc_kernel_vs_dequant_reference(bits, q_len, kv_len, splits):
    """The qc kernels must reproduce the fp16 kernels running on the exact values the packed
    cache represents (quant->dequant roundtrip through the same CUDA kernels). This isolates the
    loaders, the H32 fold and the scatter from quantization error itself, which cancels."""
    from exllamav3.modules.attention_fn.mla_triton import (
        mla_attn_triton_decode, mla_attn_triton_prefill, mla_kv_quant_append, mla_kv_append,
    )
    torch.manual_seed(bits * 1000 + kv_len)
    H, D_c, D_r = 16, 512, 64
    bsz = 2
    groups = D_c // 32
    npages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = npages * bsz
    dev = torch.device(device)

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


def test_mla_quant_cache_q8_close_to_fp16():
    """Q8 latent quantization is near-lossless; the module output should track the fp16 cache."""
    module, t, key = build(H = 8, q_lora = 256, seed = 21)
    bsz, S = 2, 300
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)
    fp = run_module(module, x, make_cache(module, 4 * PAGE_SIZE * bsz), bt)
    q8 = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, 8), bt)
    assert rel_err(q8, fp) < 3e-2, f"rel err {rel_err(q8, fp):.3e}"


@pytest.mark.parametrize("bits,tol", [(4, 8e-2), (6, 4e-2)])
def test_mla_quant_cache_consistency(bits, tol):
    """Chunked prefill, whole-shot prefill and token-by-token decode must stay in the same
    quantization-error band of each other.

    They are NOT near-identical the way the fp16 cache is: the projection GEMMs tile differently
    per chunk shape, so the pre-quantization rows differ at ulp level, and the quantizer turns a
    near-boundary ulp into a full discrete code flip (~1% of packed words at Q4). The fp16 rope
    pages differing at ulp level across chunkings confirms the diff originates upstream of the
    cache. The bitwise-level correctness bar is test_mla_qc_kernel_vs_dequant_reference, where
    the cache content is fixed by construction."""
    module, t, key = build(H = 8, seed = 23)
    bsz, S = 2, 300
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)
    whole = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, bits), bt)
    parts = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, bits), bt, chunk = 128)
    step = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, bits), bt, chunk = 1)
    assert rel_err(parts, whole) < tol, f"chunked vs whole: {rel_err(parts, whole):.3e}"
    assert rel_err(step, whole) < tol, f"decode vs whole: {rel_err(step, whole):.3e}"


@pytest.mark.parametrize("bits,tol", [(8, 2e-2), (6, 6e-2), (4, 1.5e-1)])
def test_mla_quant_cache_vs_reference(bits, tol):
    """Sanity bound against the exact reference: quantization error should shrink with bits and
    stay in the expected band. (The tight correctness bar is the dequant-reference kernel test.)"""
    module, t, key = build(H = 8, seed = 25)
    bsz, S = 2, 300
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)
    ref = ref_forward(module, t, key, x, torch.zeros((bsz,), dtype = torch.int32, device = device))
    out = run_module(module, x, make_qcache(module, 4 * PAGE_SIZE * bsz, bits), bt)
    err = rel_err(out, ref)
    print(f"  Q{bits} vs reference: rel err {err:.3e}")
    assert err < tol, f"Q{bits}: rel err {err:.3e}"


def test_mla_prefill_mode_equivalence():
    """The MHA-form prefill (up-projected past tiles) and the absorbed prefill must agree; they
    compute the same attention in different factorizations."""
    import exllamav3.modules.mla_attn as M
    module, t, key = build(H = 8, q_lora = 256, seed = 31)
    bsz, S = 2, 600
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)
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
def test_mla_gather_tile(bits):
    """The tile gather kernel must reproduce the exact latent/rope rows the cache holds (via
    dequant_cache_cont for the packed case, including the inverse H32 rotation)."""
    from exllamav3.modules.attention_fn.mla_triton import (
        _mla_gather_tile_kernel, mla_kv_append, mla_kv_quant_append,
    )
    from exllamav3.modules.attention_fn.triton_paged import _get_h32
    import triton
    torch.manual_seed(bits + 41)
    D_c, D_r, PS = 512, 64, PAGE_SIZE
    kv_len, npages = 900, 4
    dev = torch.device(device)
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


def test_mla_kv_b_export_roundtrip():
    """get_tensors must reconstruct kv_b_proj.weight in the exact checkpoint layout from the
    flat storage (the conversion pipeline carries it into quantized models verbatim)."""
    module, t, key = build(H = 8, seed = 51)
    got = module.get_tensors()[f"{key}.kv_b_proj.weight"]
    want = t[f"{key}.kv_b_proj.weight"]
    assert got.shape == want.shape
    assert torch.equal(got.cpu().half(), want.cpu().half()), "kv_b reconstruction differs"
