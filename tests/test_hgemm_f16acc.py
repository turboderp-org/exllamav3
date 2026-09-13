"""Dense FP16-partial GEMM regressions, without a model checkpoint."""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="requires Ampere or later",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("batch,M,N,K", [
    (1, 1, 128, 64), (2, 129, 256, 128), (1, 513, 1024, 1024),
    (1, 1024, 1024, 512), (8, 128, 2048, 2048), (1, 512, 4096, 512),
])
@torch.inference_mode()
def test_f16acc_padded_output(dtype, batch, M, N, K):
    torch.manual_seed(12345)
    sa, sb, ldc = M * K + 8, K * N + 8, N + 16
    sc = M * ldc + 2
    a = torch.empty(batch * sa, device="cuda", dtype=torch.half).as_strided((batch, M, K), (sa, K, 1))
    b = torch.empty(batch * sb, device="cuda", dtype=torch.half).as_strided((batch, K, N), (sb, N, 1))
    a.normal_(std=0.25)
    b.normal_(std=0.25)
    storage = torch.full((batch * sc + 8,), 123, device="cuda", dtype=dtype)
    c = storage.as_strided((batch, M, N), (sc, ldc, 1))
    mask = torch.zeros_like(storage, dtype=torch.bool)
    mask.as_strided((batch, M, N), (sc, ldc, 1)).fill_(True)
    ext.hgemm_f16acc(a, b, c)
    assert bool((storage[~mask] == 123).all())

    ref = torch.empty((batch, M, N), device="cuda", dtype=torch.float32)
    for i in range(batch):
        ext.hgemm(a[i], b[i], ref[i])
    relative_rms = ((c.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert float(relative_rms) < 0.001

    # Identical inputs must produce identical bits, including with reused scratch/output storage.
    before = c.clone()
    ext.hgemm_f16acc(a, b, c)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(c.view(bits), before.view(bits))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_f16acc_exact_row_selection(dtype):
    M, N, K = 129, 256, 128
    a = torch.zeros(M, K, device="cuda", dtype=torch.half)
    indices = torch.arange(M, device="cuda") % K
    a[torch.arange(M, device="cuda"), indices] = 1
    b = torch.randn(K, N, device="cuda", dtype=torch.half)
    c = torch.empty(M, N, device="cuda", dtype=dtype)
    ext.hgemm_f16acc(a, b, c)
    assert torch.equal(c, b[indices].to(dtype))


@pytest.mark.parametrize("case", ["float_alignment", "half_alignment", "batch_input", "batch_output", "row_overlap", "batch_overlap", "cpu"])
@torch.inference_mode()
def test_f16acc_rejects_unsafe_views(case):
    batch, M, N, K = 2, 32, 128, 64
    a = torch.randn(batch, M, K, device="cuda", dtype=torch.half)
    b = torch.randn(batch, K, N, device="cuda", dtype=torch.half)
    c = torch.empty(batch, M, N, device="cuda", dtype=torch.float32)
    if case.endswith("alignment"):
        dtype = torch.float32 if case == "float_alignment" else torch.float16
        c = torch.empty(batch * M * N + 1, device="cuda", dtype=dtype)[1:].view(batch, M, N)
    elif case == "batch_input":
        a = torch.empty(batch * M * K + 8, device="cuda", dtype=torch.half).as_strided(a.shape, (M * K + 1, K, 1))
    elif case == "batch_output":
        c = torch.empty(batch * M * N + 8, device="cuda").as_strided(c.shape, (M * N + 1, N, 1))
    elif case == "row_overlap":
        c = c.as_strided(c.shape, (M * N, N - 2, 1))
    elif case == "batch_overlap":
        c = c.as_strided(c.shape, (0, N, 1))
    else:
        b = b.cpu()
    with pytest.raises(RuntimeError, match="unsupported"):
        ext.hgemm_f16acc(a, b, c)


@pytest.mark.parametrize("batch,M,N,K", [(1, 512, 4096, 4096), (8, 128, 2048, 2048)])
@torch.inference_mode()
def test_f16acc_dispatch_and_graph(batch, M, N, K):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("expanded dispatch was tuned for GeForce Blackwell")
    if not ext.hgemm_f16acc_status(torch.cuda.current_device()):
        pytest.skip("FP16 MMA dispatch is disabled")
    torch.manual_seed(98765)
    shape_a = (M, K) if batch == 1 else (batch, M, K)
    shape_b = (K, N) if batch == 1 else (batch, K, N)
    shape_c = (M, N) if batch == 1 else (batch, M, N)
    a = torch.randn(shape_a, device="cuda", dtype=torch.half)
    b = torch.randn(shape_b, device="cuda", dtype=torch.half)
    expected = torch.empty(shape_c, device="cuda")
    actual = torch.empty_like(expected)
    ext.hgemm_f16acc(a, b, expected)
    run = ext.hgemm_recon if batch == 1 else ext.hgemm_batched
    run(a, b, actual)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(a, b, actual)
    graph.replay()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
