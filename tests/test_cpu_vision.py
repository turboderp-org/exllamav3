# Fixture-free primitive coverage for the CPU vision execution path:
#
#   - cpu_matmul (half in / float out vs FP32 reference, overflow case)
#   - Conv flat-3D CPU forward vs CUDA ext.hgemm
#   - LinearFP16 CPU forward vs CUDA ext.hgemm
#   - RoPE.apply with external 2D-grid frequencies on CPU vs direct torch reference
#   - attn_dispatch on CPU tensors vs direct torch SDPA
#   - reconstruct_had_slice_cpu vs CUDA reconstruct_had_slice for every K and codebook
#   - LinearEXL3 CPU forwards vs the CUDA reconstructed-HGEMM path
#
# CUDA-reference cases run on cuda:0 when available and are skipped otherwise; pure-CPU cases
# always run.

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.tensor import cpu_matmul
from exllamav3.util.rope import RopeSettings, RopeStyle, RoPE
from exllamav3.modules.attention_fn.dispatch import attn_dispatch
from exllamav3.modules.conv import Conv
from exllamav3.modules.quant.fp16 import LinearFP16
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.model.config import NullConfig

torch.manual_seed(0)

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if HAS_CUDA else "cpu"


def sylvester(n):
    h = torch.ones(1, 1, dtype = torch.float)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


H128 = sylvester(128) / 128 ** 0.5


def rel_err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


# ------------------------------------------------------------------------------------------------
# cpu_matmul: mixed half in / float out. The FP32 reference is finite but exceeds 65504, so a
# half-accumulating or half-rounded implementation produces inf where this expects finite values.
# ------------------------------------------------------------------------------------------------

def test_cpu_matmul_mixed_overflow():
    k, n, m = 256, 384, 4
    x = (torch.randn(m, k) * 40).half()
    w = (torch.randn(k, n) * 40).half()
    y = cpu_matmul(x, w, torch.float)
    ref = x.float() @ w.float()
    assert y.dtype == torch.float
    assert torch.isfinite(y).all(), "float output must stay finite"
    assert rel_err(y, ref) < 3e-3


def test_cpu_matmul_same_dtype():
    k, n, m = 128, 64, 8
    x = torch.randn(m, k).half()
    w = torch.randn(k, n).half()
    y = cpu_matmul(x, w)
    assert y.dtype == torch.half
    assert rel_err(y, (x.float() @ w.float())) < 2e-2


# ------------------------------------------------------------------------------------------------
# Conv (flat 3D patch embedding shape): CPU forward vs CUDA kernel
# ------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUDA, reason = "requires CUDA")
def test_conv_flat_cpu_vs_cuda():
    bsz, seq, dim_in, dim_out = 1, 1024, 1176, 1152

    def make_conv(device):
        conv = Conv(
            config = NullConfig(),
            key = "test.conv",
            in_channels = 3,
            out_channels = dim_out,
            kernel_size = (2, 14, 14),
            flat = True,
            out_dtype = torch.float,
        )
        conv.device = torch.device(device)
        conv._numel = 0
        return conv

    weight = (torch.randn(dim_out, 3, 2, 14, 14) * 0.05).half()
    bias = (torch.randn(dim_out) * 0.02).half()
    conv_cpu = make_conv("cpu")
    conv_cpu.weight = weight
    conv_cpu.bias = bias
    x = (torch.randn(bsz, seq, dim_in) * 0.5).half()
    y_cpu = conv_cpu.forward(x, params = {})

    conv_cuda = make_conv(DEVICE)
    conv_cuda.weight = weight.to(DEVICE)
    conv_cuda.bias = bias.to(DEVICE)
    y_cuda = conv_cuda.forward(x.to(DEVICE), params = {})
    assert y_cpu.dtype == torch.float
    assert rel_err(y_cpu, y_cuda.cpu()) < 3e-3


# ------------------------------------------------------------------------------------------------
# LinearFP16: CPU forward vs CUDA kernel, both output dtypes
# ------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUDA, reason = "requires CUDA")
@pytest.mark.parametrize("out_dtype", [torch.half, torch.float])
def test_linear_fp16_cpu_vs_cuda(out_dtype):
    m, k, n = 64, 1280, 3420
    w = (torch.randn(k, n) * 0.05).half()
    b = (torch.randn(n) * 0.02).half()
    lin_cpu = LinearFP16(k, n, w, b, k, n, key = "t.cpu")
    lin_cuda = LinearFP16(k, n, w.to(DEVICE), b.to(DEVICE), k, n, key = "t.cuda")

    lin_cpu.out_dtype = out_dtype
    lin_cuda.out_dtype = out_dtype
    assert lin_cpu.bc is None and lin_cuda.bc is not None

    x = (torch.randn(m, k) * 0.5).half()
    y_cpu = lin_cpu.forward(x, params = {}, out_dtype = out_dtype)
    y_cuda = lin_cuda.forward(x.to(DEVICE), params = {}, out_dtype = out_dtype)
    assert rel_err(y_cpu, y_cuda.cpu()) < 3e-3


# ------------------------------------------------------------------------------------------------
# RoPE with external precomputed angles: CPU path vs direct torch reference
# ------------------------------------------------------------------------------------------------

def test_rope_external_inv_freq_cpu():
    head_dim, num_heads, seq = 80, 16, 512
    rope = RoPE("cpu", RopeSettings(head_dim = head_dim, rope_style = RopeStyle.NEOX))
    q = torch.randn(seq, num_heads, head_dim).half()
    k = torch.randn(seq, num_heads, head_dim).half()

    # Qwen-style external angle rows: [seq, head_dim // 2]
    ids = torch.stack([
        torch.arange(seq) % 37,
        torch.arange(seq) % 53,
    ], dim = -1)
    base_freqs = 1.0 / (
        20000.0 ** (torch.arange(0, head_dim // 2, 2, dtype = torch.float) / (head_dim // 2))
    )
    grid = torch.outer(torch.arange(64, dtype = torch.float), base_freqs)
    inv_freq = grid[ids].flatten(1)

    q_rot, k_rot = rope.apply(q, k, inv_freq = inv_freq)

    sin = torch.cat((inv_freq.sin(), inv_freq.sin()), dim = -1)
    cos = torch.cat((inv_freq.cos(), inv_freq.cos()), dim = -1)

    def rotate(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return x * cos.unsqueeze(1) + torch.cat((-x2, x1), dim = -1) * sin.unsqueeze(1)

    assert rel_err(q_rot, rotate(q.float()).half()) < 2e-3
    assert rel_err(k_rot, rotate(k.float()).half()) < 2e-3


# ------------------------------------------------------------------------------------------------
# attn_dispatch on CPU tensors: non-causal cacheless call must select SDPA and match reference
# ------------------------------------------------------------------------------------------------

def test_attn_dispatch_cpu_nocache():
    bsz, heads, kv_heads, seq, dim = 1, 12, 12, 256, 64
    q = torch.randn(bsz, seq, heads, dim).half()
    k = torch.randn(bsz, seq, kv_heads, dim).half()
    v = torch.randn(bsz, seq, kv_heads, dim).half()

    o = attn_dispatch(q = q, k = k, v = v, causal = False)

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal = False
    ).transpose(1, 2)
    assert o.shape == ref.shape
    assert rel_err(o, ref) < 2e-3


def test_attn_dispatch_cpu_unsupported_raises():
    bsz, heads, seq, dim = 1, 4, 32, 32
    q = torch.randn(bsz, seq, heads, dim).half()
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(ValueError):
        attn_dispatch(q = q, k = k, v = v, causal = False, window_size = 8)


# ------------------------------------------------------------------------------------------------
# reconstruct_had_slice_cpu vs CUDA reference: every K, all codebooks, full and sliced
# ------------------------------------------------------------------------------------------------

RECONSTRUCT_CASES = [
    (256, 128, 3, False, False),
    (512, 384, 2, False, False),
    (1024, 512, 5, False, False),
    (384, 256, 4, True, False),
    (256, 512, 3, False, True),
    (128, 128, 1, False, False),
    (128, 128, 8, False, False),
    (256, 256, 6, True, False),
    (256, 256, 7, False, True),
]


@pytest.mark.skipif(not HAS_CUDA, reason = "requires CUDA")
@pytest.mark.parametrize("k_dim,n_dim,K,mcg,mul1", RECONSTRUCT_CASES)
def test_reconstruct_cpu_vs_cuda(k_dim, n_dim, K, mcg, mul1):
    torch.manual_seed(k_dim * 7 + n_dim + K)
    trellis = torch.randint(
        0, 65536, (k_dim // 16, n_dim // 16, 256 * K // 16),
        dtype = torch.int32, device = DEVICE,
    ).to(torch.short)
    suh = torch.sign(torch.randn(k_dim, device = DEVICE)).half()
    svh = torch.sign(torch.randn(n_dim, device = DEVICE)).half()

    w_cuda = torch.empty(k_dim, n_dim, dtype = torch.half, device = DEVICE)
    ext.reconstruct_had_slice(w_cuda, trellis, suh, svh, K, mcg, mul1, 0)

    w_cpu = torch.empty(k_dim, n_dim, dtype = torch.half)
    ext.reconstruct_had_slice_cpu(
        w_cpu, trellis.cpu(), suh.cpu(), svh.cpu(), K, mcg, mul1, 0
    )
    scale = w_cuda.float().abs().max().item()
    assert (w_cpu.float() - w_cuda.float().cpu()).abs().max().item() / scale < 2e-3

    if n_dim >= 384:
        ws_cpu = torch.empty(k_dim, 128, dtype = torch.half)
        ext.reconstruct_had_slice_cpu(
            ws_cpu, trellis.cpu(), suh.cpu(), svh[128:].cpu(), K, mcg, mul1, 128
        )
        errs = (ws_cpu.float() - w_cuda[:, 128:256].float().cpu()).abs().max().item()
        assert errs / scale < 2e-3


def test_reconstruct_cpu_invalid_shapes_raise():
    K = 4
    trellis = torch.randint(-32768, 32768, (8, 8, 256 * K // 16), dtype = torch.int16)
    ones = torch.ones(128, dtype = torch.half)

    # unaligned n_offset
    w = torch.empty(128, 128, dtype = torch.half)
    with pytest.raises(RuntimeError):
        ext.reconstruct_had_slice_cpu(w, trellis, ones, ones, K, False, False, 7)

    # wrong packed depth for K
    trellis_bad = torch.randint(-32768, 32768, (8, 8, 256 * (K + 1) // 16), dtype = torch.int16)
    with pytest.raises(RuntimeError):
        ext.reconstruct_had_slice_cpu(w, trellis_bad, ones, ones, K, False, False, 0)

    # undersized svh
    with pytest.raises(RuntimeError):
        ext.reconstruct_had_slice_cpu(w, trellis, ones, torch.ones(127, dtype = torch.half), K, False, False, 0)


# ------------------------------------------------------------------------------------------------
# LinearEXL3 CPU forward vs CUDA reconstructed-HGEMM path
# ------------------------------------------------------------------------------------------------

def _exl3_tensors(k_dim, n_dim, K, mcg, mul1, bias = True):
    torch.manual_seed(k_dim * 13 + n_dim + K)
    trellis = torch.randint(
        0, 65536, (k_dim // 16, n_dim // 16, 256 * K // 16),
        dtype = torch.int32,
    ).to(torch.short)
    suh = torch.sign(torch.randn(k_dim)).half()
    svh = torch.sign(torch.randn(n_dim)).half()
    b = torch.randn(n_dim).half() * 0.01 if bias else None
    return (k_dim, n_dim, trellis, suh, svh, b, mcg, mul1)


def _make_exl3_linear(tensors, device):
    k_dim, n_dim, trellis, suh, svh, b, mcg, mul1 = tensors
    return LinearEXL3(
        NullConfig(),
        k_dim,
        n_dim,
        None,
        None,
        None,
        suh.to(device),
        svh.to(device),
        trellis.to(device),
        torch.tensor(1, dtype = torch.int8) if mcg else None,
        torch.tensor(1, dtype = torch.int8) if mul1 else None,
        b.to(device) if b is not None else None,
        key = f"test.linear.{device}",
    )


@pytest.mark.skipif(not HAS_CUDA, reason = "requires CUDA")
@pytest.mark.parametrize("rows", [1, 256])
def test_linear_exl3_cpu_vs_cuda(rows):
    tensors = _exl3_tensors(256, 512, 4, False, True)
    lin_cpu = _make_exl3_linear(tensors, "cpu")
    lin_cuda = _make_exl3_linear(tensors, DEVICE)
    assert lin_cpu.bc is None and lin_cuda.bc is not None

    x = (torch.randn(rows, 256) * 0.2).half()
    y_cpu = lin_cpu.forward(x, params = {})
    y_cuda = lin_cuda.forward(x.to(DEVICE), params = {"reconstruct": True})
    assert rel_err(y_cpu, y_cuda.cpu()) < 2e-2


@pytest.mark.skipif(not HAS_CUDA, reason = "requires CUDA")
def test_linear_exl3_cpu_float_output():
    tensors = _exl3_tensors(128, 128, 2, False, True)
    lin_cpu = _make_exl3_linear(tensors, "cpu")
    x = (torch.randn(4, 128) * 0.2).half()
    y_float = lin_cpu.forward(x, params = {}, out_dtype = torch.float)
    y_half = lin_cpu.forward(x, params = {})
    assert y_float.dtype == torch.float
    assert rel_err(y_float, y_half) < 2e-2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ------------------------------------------------------------------------------------------------
# Full Qwen3VLVisionModel tower executed on CPU with synthetic weights: proves the exact module
# graph the Qwen 3.5 vision component instantiates (flat conv, interpolated position embedding,
# NEOX rope + non-causal SDPA blocks, deepstack mergers) runs without CUDA end to end.
# ------------------------------------------------------------------------------------------------

class StubSTC:
    """Minimal stand-in for SafetensorsCollection serving deterministic CPU tensors."""

    def __init__(self, tensors):
        self.tensors = tensors

    def has_tensor(self, key):
        return key in self.tensors

    def has_tensor_group(self, key, names):
        def has(n):
            alts = n if isinstance(n, (list, tuple)) else [n]
            return any(f"{key}.{a}" in self.tensors for a in alts)
        return all(has(n) for n in names)

    def get_tensor(self, key, device, optional = False, **kwargs):
        t = self.tensors.get(key)
        if t is None:
            if optional:
                return None
            raise KeyError(key)
        if kwargs.get("transpose"):
            t = t.t()
        return t.to(device)

    def begin_deferred_load(self):
        pass

    def end_deferred_load(self):
        pass


def _qwen35_vision_weights():
    """Deterministic fp16 weights matching the Qwen3.5 vision tensor layout."""
    g = torch.Generator().manual_seed(7)
    def rnd(*shape, scale = 0.02):
        return (torch.randn(*shape, generator = g) * scale).half()
    H, NH, INTER, OUT = 128, 8, 256, 64
    POS = 144  # 12x12 position grid
    t = {}
    t["model.visual.patch_embed.proj.weight"] = rnd(H, 3, 2, 14, 14)
    t["model.visual.patch_embed.proj.bias"] = rnd(H)
    t["model.visual.pos_embed.weight"] = rnd(POS, H)
    for b in range(2):
        p = f"model.visual.blocks.{b}"
        t[f"{p}.norm1.weight"] = torch.ones(H).half()
        # Checkpoint (HF) orientation: [out, in]; the loader transposes at fetch time
        t[f"{p}.attn.qkv.weight"] = rnd(3 * H, H)
        t[f"{p}.attn.proj.weight"] = rnd(H, H)
        t[f"{p}.norm2.weight"] = torch.ones(H).half()
        t[f"{p}.mlp.linear_fc1.weight"] = rnd(INTER, H)
        t[f"{p}.mlp.linear_fc2.weight"] = rnd(H, INTER)
    MERGE = 4  # spatial_merge_size ** 2
    t["model.visual.deepstack_merger_list.0.norm.weight"] = torch.ones(H * MERGE).half()
    t["model.visual.deepstack_merger_list.0.linear_fc1.weight"] = rnd(H * MERGE, H * MERGE)
    t["model.visual.deepstack_merger_list.0.linear_fc2.weight"] = rnd(OUT, H * MERGE)
    t["model.visual.merger.norm.weight"] = torch.ones(H).half()
    t["model.visual.merger.linear_fc1.weight"] = rnd(H * MERGE, H * MERGE)
    t["model.visual.merger.linear_fc2.weight"] = rnd(OUT, H * MERGE)
    return t


def _make_qwen35_vision_model():
    from types import SimpleNamespace
    from exllamav3.architecture.qwen3_vl import Qwen3VLVisionModel
    from exllamav3.model.config import InferParams

    H, NH = 128, 8
    v = SimpleNamespace(
        deepstack_visual_indexes = [0],
        depth = 2,
        hidden_act = "gelu",
        hidden_size = H,
        intermediate_size = 256,
        num_heads = NH,
        num_position_embeddings = 144,
        out_hidden_size = 64,
        patch_size = 14,
        spatial_merge_size = 2,
        temporal_patch_size = 2,
        model_type = "qwen3_5_vision",
    )
    v.num_channels = 3
    v.head_dim = H // NH
    v.rope_theta = 10000.0
    v.layernorm_eps = 1e-6
    pp = SimpleNamespace(
        resample = 3,
        rescale_factor = 1 / 255,
        image_mean = [0.5, 0.5, 0.5],
        image_std = [0.5, 0.5, 0.5],
        temporal_patch_size = 2,
        patch_size = 14,
        merge_size = 2,
        min_pixels = 56 * 56,
        max_pixels = 56 * 56,
    )
    cfg = SimpleNamespace(vision = v, vision_pp = pp, layer_map = None,
                          vision_start_token_id = 151652, vision_end_token_id = 151653,
                          architecture = "Qwen3_5ForConditionalGeneration",
                          infer_params = InferParams(), stc = StubSTC(_qwen35_vision_weights()))
    vm = Qwen3VLVisionModel(cfg)
    vm.fwd_modules = [(m, 0, i) for i, m in enumerate(vm.modules)]
    return vm


def test_qwen35_vision_tower_cpu_end_to_end():
    from PIL import Image as PILImage

    vm = _make_qwen35_vision_model()
    vm.load("cpu")

    image = PILImage.new("RGB", (56, 56), (120, 40, 200))
    mme = vm.get_image_embeddings(None, image)

    # 56x56 -> 4x4 patches -> merged into a 2x2 token grid
    assert mme.mm_length == 4, f"unexpected mm_length {mme.mm_length}"
    assert mme.embeddings.shape == (4, 64)
    assert len(mme.deepstack_embeddings) == 1
    assert mme.deepstack_embeddings[0].shape == (4, 64)
    assert torch.isfinite(mme.embeddings).all()
    vm.unload()


# ------------------------------------------------------------------------------------------------
# fn_torch_sdpa_fallback_nocache with cu_seqlens: segmented execution must match per-segment
# dense SDPA; the dense (non-varlen) path must be unchanged
# ------------------------------------------------------------------------------------------------

def _sdpa_args(q, k, v, cu_seqlens = None, max_seqlen = None):
    from exllamav3.modules.attention_fn.common import AttnArgs
    bsz, seq, heads, dim = q.shape
    return AttnArgs(
        bsz = bsz,
        q_len = seq,
        num_q_heads = heads,
        dim = dim,
        kv_len = k.shape[1],
        num_kv_heads = k.shape[2],
        q = q,
        k = k,
        v = v,
        k_cache = None,
        v_cache = None,
        causal = False,
        sm_scale = dim ** -0.5,
        cu_seqlens = cu_seqlens,
        max_seqlen = max_seqlen,
        window_size = None,
        softcap = 0.0,
        block_table = None,
        cache_seqlens = None,
    )


def test_sdpa_varlen_parity():
    from exllamav3.modules.attention_fn.torch import fn_torch_sdpa_fallback_nocache

    g = torch.Generator().manual_seed(11)
    q = torch.randn(1, 37, 4, 64, generator = g).half()
    k = torch.randn(1, 37, 4, 64, generator = g).half()
    v = torch.randn(1, 37, 4, 64, generator = g).half()
    bounds = [0, 5, 6, 20, 37]

    ref = torch.cat([
        fn_torch_sdpa_fallback_nocache(_sdpa_args(q[:, s:e], k[:, s:e], v[:, s:e]))
        for s, e in zip(bounds[:-1], bounds[1:])
    ], dim = 1)

    out = fn_torch_sdpa_fallback_nocache(
        _sdpa_args(q, k, v, cu_seqlens = torch.tensor(bounds, dtype = torch.int32), max_seqlen = 15)
    )
    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), rtol = 0, atol = 1e-3), \
        f"varlen mismatch, max diff {(out.float() - ref.float()).abs().max().item()}"


def test_sdpa_fallback_dense_path_unchanged():
    import torch.nn.functional as F
    from exllamav3.modules.attention_fn.torch import fn_torch_sdpa_fallback_nocache

    g = torch.Generator().manual_seed(12)
    q = torch.randn(2, 64, 8, 64, generator = g).half()
    k = torch.randn(2, 64, 8, 64, generator = g).half()
    v = torch.randn(2, 64, 8, 64, generator = g).half()

    out = fn_torch_sdpa_fallback_nocache(_sdpa_args(q, k, v)).float()
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        is_causal = False,
        scale = 64 ** -0.5,
    ).transpose(1, 2)
    assert rel_err(out, ref) < 2e-3


# ------------------------------------------------------------------------------------------------
# Qwen3VLVisionModel.get_image_embeddings with multiple different-size images: packed varlen
# forward must reproduce per-image standalone results exactly enough on CPU
# ------------------------------------------------------------------------------------------------

def test_qwen35_vision_multi_image_packing():
    from PIL import Image as PILImage

    vm = _make_qwen35_vision_model()
    # Allow a second, larger image: min/max pixels bound qwen2_smart_resize
    vm.config.vision_pp.max_pixels = 200 * 200
    vm.load("cpu")

    img_a = PILImage.new("RGB", (56, 56), (120, 40, 200))
    img_b = PILImage.new("RGB", (112, 140), (30, 200, 90))  # -> 8x10 patches

    mmes = vm.get_image_embeddings(None, [img_a, img_b])
    assert len(mmes) == 2

    solo_a = vm.get_image_embeddings(None, img_a)
    solo_b = vm.get_image_embeddings(None, img_b)

    assert mmes[0].mm_length == 4 and mmes[1].mm_length == 20
    assert mmes[0].grid_thw == solo_a.grid_thw and mmes[1].grid_thw == solo_b.grid_thw
    assert mmes[0].metadata["original_size"] == (56, 56)
    assert mmes[1].metadata["original_size"] == (112, 140)

    for packed, solo in [(mmes[0], solo_a), (mmes[1], solo_b)]:
        assert torch.isfinite(packed.embeddings).all()
        assert torch.allclose(packed.embeddings.float(), solo.embeddings.float(), rtol = 0, atol = 1e-3), \
            f"packing mismatch, max diff {(packed.embeddings.float() - solo.embeddings.float()).abs().max().item()}"
        for de_p, de_s in zip(packed.deepstack_embeddings, solo.deepstack_embeddings):
            assert torch.allclose(de_p.float(), de_s.float(), rtol = 0, atol = 1e-3)

    vm.unload()


# ------------------------------------------------------------------------------------------------
# Qwen3VLVisionModel.get_video_embeddings: frame list packs into one clip with one attention
# segment per frame (temporal_patch_size truncation applies); single MMEmbedding returned
# ------------------------------------------------------------------------------------------------

def test_qwen35_vision_video_frames():
    from PIL import Image as PILImage

    vm = _make_qwen35_vision_model()
    vm.load("cpu")

    frames = [PILImage.new("RGB", (56, 56), (n * 20, 100, 255 - n * 20)) for n in range(8)]
    mme = vm.get_video_embeddings(None, frames)

    # 8 frames / temporal_patch_size 2 -> t = 4; 4x4 patches per frame -> grid (4, 4, 4)
    assert isinstance(mme.mm_length, int)
    assert mme.grid_thw == (4, 4, 4)
    assert mme.mm_length == 4 * 4 * 4 // 4
    assert mme.embeddings.shape == (16, 64)
    assert len(mme.deepstack_embeddings) == 1
    assert mme.deepstack_embeddings[0].shape == (16, 64)
    assert torch.isfinite(mme.embeddings).all()
    assert mme.metadata["original_size"] == (56, 56)

    vm.unload()
