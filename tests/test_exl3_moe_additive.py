import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "CUDA is required"
)


def test_additive_moe_abi_version():
    version = ext.EXL3_MOE_ADDITIVE_ABI_VERSION
    assert type(version) is int
    assert version == 1


def _metadata(device: torch.device):
    dim = 128
    trellis = torch.zeros((dim // 16, dim // 16, 16),
                          dtype = torch.int16, device = device)
    suh = torch.ones(dim, dtype = torch.float16, device = device)
    svh = torch.ones(dim, dtype = torch.float16, device = device)
    pointer = lambda tensor: torch.tensor(
        [tensor.data_ptr()], dtype = torch.int64, device = device
    )

    base = (
        pointer(trellis), pointer(suh), pointer(svh),
        pointer(trellis), pointer(suh), pointer(svh),
        pointer(trellis), pointer(suh), pointer(svh),
    )
    # A second sparse stage has a null pointer. A zero scale must prevent the
    # kernel from dereferencing it.
    residual_ptrs = torch.tensor(
        [[trellis.data_ptr()], [0]], dtype = torch.int64, device = device
    )
    residual_scales = torch.tensor(
        [[0.125], [0.0]], dtype = torch.float32, device = device
    )
    residual_k = torch.tensor([1, 1], dtype = torch.int32, device = device)
    residual = (
        residual_ptrs, residual_ptrs, residual_ptrs,
        residual_scales, residual_scales, residual_scales,
        residual_k, residual_k, residual_k,
    )
    # Keep every pointee alive for the asynchronous launches below.
    retained = (trellis, suh, svh)
    return base, residual, retained


def _run(rows: int, capacity: int, id_dtype: torch.dtype,
         weight_dtype: torch.dtype, base, residual, topk: int = 1):
    device = torch.device("cuda", torch.cuda.current_device())
    dim = 128
    concurrency = ext.exl3_moe_max_concurrency(device.index)
    hidden = (
        torch.arange(rows * dim, dtype = torch.float32, device = device)
        .reshape(rows, dim).remainder(17).sub_(8).mul_(1e-3).half()
    )
    output = torch.zeros_like(hidden, dtype = torch.float32)
    topk_ids = torch.zeros((rows, topk), dtype = id_dtype, device = device)
    topk_weights = torch.linspace(
        0.25, 0.75, rows * topk, dtype = torch.float32, device = device
    ).to(weight_dtype).reshape(rows, topk)
    expert_map = torch.zeros(1, dtype = torch.int64, device = device)
    expert_count = torch.empty(2, dtype = torch.int64, device = device)
    expert_offsets = torch.empty_like(expert_count)
    route_count = rows * topk
    token_sorted = torch.empty(route_count, dtype = torch.int64, device = device)
    weight_sorted = torch.empty(
        route_count, dtype = torch.float16, device = device
    )
    temp_state_g = torch.empty(
        (concurrency, capacity, dim), dtype = torch.float16, device = device
    )
    temp_state_u = torch.empty_like(temp_state_g)
    temp_intermediate_g = torch.empty_like(temp_state_g)
    temp_intermediate_u = torch.empty_like(temp_state_g)

    ext.exl3_moe_additive_fused(
        hidden, output, topk_ids, topk_weights, expert_map,
        expert_count, expert_offsets, token_sorted, weight_sorted,
        temp_state_g, temp_state_u, temp_intermediate_g,
        temp_intermediate_u, 0, 1, 1, 1, *base, *residual, 1,
        True, False, True, False, True, False, 0.0, 1,
    )
    torch.cuda.synchronize()
    return (
        output, expert_count, expert_offsets, token_sorted, weight_sorted,
        topk_weights,
    )


@pytest.mark.parametrize("rows", [3, 5])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "weight_dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@torch.inference_mode()
def test_additive_fused_tiles_overflow_and_skips_sparse_null_stage(
    rows, id_dtype, weight_dtype
):
    device = torch.device("cuda", torch.cuda.current_device())
    base, residual, retained = _metadata(device)

    tiled = _run(rows, 2, id_dtype, weight_dtype, base, residual)
    reference = _run(rows, rows, id_dtype, weight_dtype, base, residual)

    torch.testing.assert_close(tiled[0], reference[0], rtol = 1e-3, atol = 1e-3)
    assert tiled[0].abs().max().item() > 0
    assert tiled[1].tolist() == [rows, 0]
    assert tiled[2].tolist() == [0, rows]
    assert tiled[3].tolist() == list(range(rows))
    torch.testing.assert_close(
        tiled[4], tiled[5].reshape(-1).half(), rtol = 0, atol = 0
    )
    assert retained


@torch.inference_mode()
def test_additive_fused_topk_routes_do_not_require_topk_times_workspace():
    device = torch.device("cuda", torch.cuda.current_device())
    rows = 5
    base, residual, retained = _metadata(device)

    tiled = _run(
        rows, rows, torch.int64, torch.float32, base, residual, topk = 2
    )
    reference = _run(
        rows, 2 * rows, torch.int64, torch.float32, base, residual, topk = 2
    )

    torch.testing.assert_close(tiled[0], reference[0], rtol = 1e-3, atol = 1e-3)
    assert tiled[0].abs().max().item() > 0
    assert tiled[1].tolist() == [2 * rows, 0]
    assert retained


@torch.inference_mode()
def test_legacy_moe_preserves_oversized_expert_fallback_contract():
    device = torch.device("cuda", torch.cuda.current_device())
    rows = 3
    capacity = 2
    dim = 128
    concurrency = ext.exl3_moe_max_concurrency(device.index)
    base, _residual, retained = _metadata(device)
    hidden = torch.ones((rows, dim), dtype = torch.float16, device = device)
    output = torch.zeros_like(hidden, dtype = torch.float32)
    expert_count = torch.tensor([rows, 0], dtype = torch.int64, device = device)
    token_sorted = torch.arange(rows, dtype = torch.int64, device = device)
    weight_sorted = torch.ones(rows, dtype = torch.float16, device = device)
    temp_state_g = torch.empty(
        (concurrency, capacity, dim), dtype = torch.float16, device = device
    )
    temp_state_u = torch.empty_like(temp_state_g)
    temp_intermediate_g = torch.empty_like(temp_state_g)
    temp_intermediate_u = torch.empty_like(temp_state_g)

    ext.exl3_moe(
        hidden, output, expert_count, token_sorted, weight_sorted,
        temp_state_g, temp_state_u, temp_intermediate_g,
        temp_intermediate_u, 0, 1, 1, 1, *base,
        True, False, True, False, True, False, 0.0, -1,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(output).item() == 0
    assert retained
