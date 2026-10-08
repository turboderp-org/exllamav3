"""
EXL3 Linear forward paths: the quantized GEMM/GEMV (LinearEXL3.bc.run_alloc) and the reconstruct + fp16 GEMM path
(reconstruct_hgemm: unfused had/reconstruct below 1024 rows, fused had-folding reconstruct from 1024 rows, column
slices past MAX_RECONSTRUCT_SLICE_N) must agree with each other and with the dequantize-then-matmul reference
(testlib.exl3.linear_ref) over row counts spanning the GEMV, multi-row and GEMM tiers, every integer bitrate with
all three codebooks, the half-integer bitrates (mul1 only), with and without bias. Modules are built from random
trellis tensors through the real loader (Linear(...).load).

params {"reconstruct": False} only selects the quantized kernels up to AUTO_RECONSTRUCT_THRESHOLD rows unless
infer_params.no_reconstruct is set; the module under test sets it so every row count really takes the quantized
path, and test_auto_reconstruct_dispatch checks the threshold rule itself.
"""

import pytest
import torch

from exllamav3.modules import Linear
from exllamav3.modules.quant.exl3 import AUTO_RECONSTRUCT_THRESHOLD, MAX_RECONSTRUCT_SLICE_N

from testlib.checkpoint import module_config
from testlib.exl3 import CODEBOOKS, checkpoint_tensors, generator, linear_ref

KEY = "model.layers.0.mlp.up_proj"
ROWS = [1, 2, 8, 16, 17, 31, 32, 33, 256, 2048]
BITRATES = [(K, cb) for K in range(1, 9) for cb in CODEBOOKS] + [(K, "mul1") for K in (1.5, 2.5, 3.5)]


def rel_rms(a, b) -> float:
    """Relative RMS error: fp16 accumulation-order differences stay small, a broken tile or slice is O(1)"""
    return ((a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt()).item()


def build(directory, device, k, n, K, codebook, bias, seed, **infer_params):
    t = checkpoint_tensors(KEY, k, n, K, generator(seed), codebook = codebook, bias = bias)
    module = Linear(module_config(t, directory, **infer_params), KEY, k, n)
    module.load(device)
    assert module.quant_type == "exl3"
    return module, {name: v.to(device) for name, v in t.items()}


def reference(t, x, K, codebook):
    ref = linear_ref(x.view(-1, x.shape[-1]), t[f"{KEY}.trellis"], t[f"{KEY}.suh"], t[f"{KEY}.svh"], K, codebook)
    if f"{KEY}.bias" in t:
        ref = (ref.float() + t[f"{KEY}.bias"].float()).half()
    return ref.view(*x.shape[:-1], -1)


class ReconstructSpy:
    """Counts calls into LinearEXL3.reconstruct_hgemm, so a test can assert which path a forward took"""

    def __init__(self, inner):
        self.calls = 0
        orig = inner.reconstruct_hgemm

        def spy(*args, **kwargs):
            self.calls += 1
            return orig(*args, **kwargs)

        inner.reconstruct_hgemm = spy

    def took_reconstruct(self, fn):
        before = self.calls
        out = fn()
        return out, self.calls > before


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("k, n", [(2048, 512), (512, 1536)])
@pytest.mark.parametrize("K, codebook", BITRATES)
@torch.inference_mode()
def test_quant_vs_reconstruct(tmp_path, device, K, codebook, k, n, bias):
    module, t = build(tmp_path, device, k, n, K, codebook, bias, seed = int(K * 10) + k + n + bias,
                      no_reconstruct = True)
    spy = ReconstructSpy(module.inner)
    gen = generator(int(K * 10) + 1)
    for rows in ROWS:
        x = (torch.randn((1, rows, k), generator = gen) * 0.5).half().to(device)
        ref = reference(t, x, K, codebook)
        y_q, recon = spy.took_reconstruct(lambda: module.forward(x, {"reconstruct": False}))
        assert not recon, f"rows {rows}: no_reconstruct forward took the reconstruct path"
        y_r, recon = spy.took_reconstruct(lambda: module.forward(x, {"reconstruct": True}))
        assert recon, f"rows {rows}: reconstruct forward took the quantized path"
        for name, y in (("quantized", y_q), ("reconstruct", y_r)):
            assert y.shape == (1, rows, n) and torch.isfinite(y).all(), f"rows {rows} {name}: bad output"
            err = rel_rms(y, ref)
            assert err < 0.02, f"rows {rows} {name} vs reference: relative RMS error {err:.4f}"
        err = rel_rms(y_q, y_r)
        assert err < 0.02, f"rows {rows} quantized vs reconstruct: relative RMS error {err:.4f}"


@pytest.mark.parametrize("K, codebook", [(2, "3inst"), (4, "mcg"), (3.5, "mul1")])
@torch.inference_mode()
def test_wide_output_slices(tmp_path, device, K, codebook):
    """Output width past MAX_RECONSTRUCT_SLICE_N: the reconstruct path runs in column slices (unfused and fused)"""
    k, n = 256, MAX_RECONSTRUCT_SLICE_N + 256
    module, t = build(tmp_path, device, k, n, K, codebook, True, seed = 99, no_reconstruct = True)
    gen = generator(5)
    for rows in (1, 33, 1024):
        x = (torch.randn((1, rows, k), generator = gen) * 0.5).half().to(device)
        ref = reference(t, x, K, codebook)
        for params in ({"reconstruct": True}, {"reconstruct": False}):
            err = rel_rms(module.forward(x, params), ref)
            assert err < 0.02, f"rows {rows} {params}: relative RMS error {err:.4f}"


@torch.inference_mode()
def test_auto_reconstruct_dispatch(tmp_path, device):
    """{"reconstruct": False} takes the quantized kernels up to AUTO_RECONSTRUCT_THRESHOLD rows and reconstructs
    past it, unless infer_params.no_reconstruct; {"reconstruct": True} always reconstructs"""
    k, n = 512, 512
    auto, _ = build(tmp_path / "auto", device, k, n, 4, "mul1", False, seed = 1)
    forced, _ = build(tmp_path / "forced", device, k, n, 4, "mul1", False, seed = 1, no_reconstruct = True)
    spies = {"auto": ReconstructSpy(auto.inner), "forced": ReconstructSpy(forced.inner)}
    for rows in (1, AUTO_RECONSTRUCT_THRESHOLD, AUTO_RECONSTRUCT_THRESHOLD + 1, 1024):
        x = torch.randn((1, rows, k), device = device).half()
        for name, module in (("auto", auto), ("forced", forced)):
            _, recon = spies[name].took_reconstruct(lambda: module.forward(x, {"reconstruct": False}))
            expect = name == "auto" and rows > AUTO_RECONSTRUCT_THRESHOLD
            assert recon == expect, f"{name}, rows {rows}: reconstruct path taken = {recon}"
            _, recon = spies[name].took_reconstruct(lambda: module.forward(x, {"reconstruct": True}))
            assert recon, f"{name}, rows {rows}: reconstruct requested but not taken"
