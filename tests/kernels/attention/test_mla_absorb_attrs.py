"""
The AOT compile of _mla_absorb_kernel must carry the pointer-alignment specialization that
mla_absorb_signature() tags into the BC signature. On gfx1100 (RDNA3, triton 3.7.1, ROCm 7.14)
the unspecialized compile of this kernel at GLM-5.3-style wide NoPE shapes is a genuine
miscompile: a 65536-byte-LDS 277-spill binary whose output is NaN on some launches and wrong
by ~1e3 on others, while the aligned compile (32768 B, 41 spills) is exact. That garbage
propagated through the BC-MLA decode graph into all-NaN logits, and argmax over NaN returns
token id 0 ('!' in the GLM-5.3 tokenizer), so the model streamed an endless "!!!!!" loop.

This compiles the kernel exactly as the BC path does -- signature from mla_absorb_signature(),
attrs derived by signature_to_ast, launched through the CompiledKernel handle -- and checks it
against a per-head fp32 reference. If the ":16" tagging is dropped from the signature helper,
the compile reverts to the broken variant and this test fails on the affected parts (and merely
re-validates correctness elsewhere).
"""

import pytest
import torch
import triton

from exllamav3.modules.attention_fn.bc_attn import signature_to_ast
from exllamav3.modules.attention_fn.bc_mla import mla_absorb_signature
from exllamav3.modules.attention_fn.mla_triton import _mla_absorb_kernel

pytestmark = pytest.mark.attention


# GLM-5.3-Flash MLA decode shapes: 64 heads, NoPE-only heads of 256, 512-wide latent
H, QK_DIM, D_NOPE, D_C = 64, 256, 256, 512
W_Q = H * QK_DIM
NUM_WARPS, NUM_STAGES = 4, 2
TOL = 5e-2    # healthy variant measures ~1.2e-4 against the fp32 reference


def compile_aligned():
    sig, pointer_range_32 = mla_absorb_signature()
    if torch.version.hip:
        # The compile contract on AMD: ":16" tags AND pointer_range_32. Missing either
        # compiles a variant that is wrong on gfx1100 (NaN/garbage without tags, rare
        # ~0.4-wrong elements without the range)
        assert pointer_range_32, "mla_absorb_signature() dropped pointer_range_32 on HIP"
        assert all(v.endswith(":16") for k, v in sig.items() if v.startswith("*")), \
            "mla_absorb_signature() dropped the ':16' alignment tags on HIP"
        # The compile contract on AMD: ":16" tags AND pointer_range_32. Missing either
        # compiles a variant that is wrong on gfx1100 (NaN/garbage without tags, rare
        # ~0.4-wrong elements without the range)
        assert all(v.endswith(":16") for k, v in sig.items() if v.startswith("*")), \
            "mla_absorb_signature() dropped the ':16' alignment tags on HIP"
    sig = sig | {n: "constexpr" for n in (
        "n_q_heads", "QK_DIM", "Q_STRIDE", "D_nope", "NOPE_PAD", "D_c", "BLOCK_M", "BLOCK_N")}
    plain, attrs = signature_to_ast(_mla_absorb_kernel, sig, pointer_range_32 = pointer_range_32)
    assert attrs, "no specialization attrs derived for this backend"
    consts = dict(n_q_heads = H, QK_DIM = QK_DIM, Q_STRIDE = W_Q, D_nope = D_NOPE,
                  NOPE_PAD = triton.next_power_of_2(D_NOPE), D_c = D_C,
                  BLOCK_M = 16, BLOCK_N = 128)
    from triton.compiler import ASTSource
    ck = triton.compile(
        ASTSource(fn = _mla_absorb_kernel, signature = plain, constexprs = consts, attrs = attrs),
        options = {"num_warps": NUM_WARPS, "num_stages": NUM_STAGES})
    ck._init_handles()
    return ck


@pytest.mark.parametrize("block_m", [16, 64])     # decode slot and the (8,16) batched slot
@torch.inference_mode()
def test_absorb_signature_attrs(device, block_m):
    torch.manual_seed(0)
    R = 1 if block_m == 16 else 8
    q = (torch.randn(R, W_Q, device = device) * 0.1).half()
    w = (torch.randn(D_C, H * D_NOPE, device = device) * 0.05).half()
    out = torch.empty(H, R, D_C, dtype = torch.half, device = device)
    ref = torch.einsum("rhd,hdc->hrc",
                       q.view(R, H, QK_DIM)[..., :D_NOPE].float(),
                       w.view(D_C, H, D_NOPE).float().permute(1, 2, 0))

    ck = compile_aligned()
    grid_x = triton.cdiv(R, block_m)
    stream = torch.cuda.current_stream(device).cuda_stream
    # The miscompile is nondeterministic per launch; take the worst of a few runs
    for _ in range(4):
        out.zero_()
        ck.run(grid_x, D_C // 128, H, stream, ck.function, ck.packed_metadata,
               None, None, None, q, w, out, R)
        torch.cuda.synchronize(device)
        err = (out.float() - ref).abs().max().item()
        assert torch.isfinite(out.float()).all(), "NaN in absorb output (miscompiled variant?)"
        assert err <= TOL, f"absorb output wrong by {err:.3f} (miscompiled variant?)"
