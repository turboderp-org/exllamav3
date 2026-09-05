import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "EXL3 BF16 I/O tests require CUDA",
)


def require_supported_device():
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if properties.major < 8:
        pytest.skip("EXL3 BF16 I/O requires compute capability 8.0 or newer")
    optin_smem = getattr(
        properties,
        "shared_memory_per_block_optin",
        properties.shared_memory_per_block,
    )
    if optin_smem < 90 * 1024:
        pytest.skip("EXL3 BF16 I/O requires 90 KiB of dynamic shared memory")
    torch.cuda.set_device(device)
    return device


def make_trellis(k, n, bits, device, seed):
    generator = torch.Generator(device = device)
    generator.manual_seed(seed)
    return torch.randint(
        0,
        65536,
        (k // 16, n // 16, bits * 16),
        dtype = torch.int32,
        device = device,
        generator = generator,
    ).to(torch.int16)


def make_signs(size, device, seed):
    generator = torch.Generator(device = device)
    generator.manual_seed(seed)
    values = torch.randint(
        0,
        2,
        (size,),
        dtype = torch.int32,
        device = device,
        generator = generator,
    )
    return (values * 2 - 1).to(torch.float16)


def relative_rms(actual, expected):
    actual_f = actual.float()
    expected_f = expected.float()
    return float(
        torch.sqrt(torch.mean((actual_f - expected_f) ** 2))
        / torch.sqrt(torch.mean(expected_f ** 2))
    )


class GemmCase:
    def __init__(self, m, k, n, bits, device):
        self.a_bf16 = torch.randn((m, k), dtype = torch.bfloat16, device = device)
        self.b = make_trellis(k, n, bits, device, 1000 + bits)
        self.suh = make_signs(k, device, 2000 + bits)
        self.svh = make_signs(n, device, 3000 + bits)
        self.a_had = torch.empty((m, k), dtype = torch.float16, device = device)
        self.scratch = torch.empty((m, n), dtype = torch.float16, device = device)
        self.output = torch.empty((m, n), dtype = torch.bfloat16, device = device)

    def run(self, force_num_sms = 0, final_grid_sync = False):
        ext.exl3_gemm_bf16_io(
            self.a_bf16,
            self.b,
            self.scratch,
            self.output,
            self.suh,
            self.a_had,
            self.svh,
            2,
            force_num_sms,
            True,
            final_grid_sync,
        )

    def reference(self):
        a_fp16 = self.a_bf16.to(torch.float16)
        a_had = torch.empty_like(a_fp16)
        output_fp16 = torch.empty_like(self.scratch)
        ext.exl3_gemm(
            a_fp16,
            self.b,
            output_fp16,
            self.suh,
            a_had,
            self.svh,
            2,
            True,
            False,
            0,
        )
        return output_fp16.to(torch.bfloat16)


class MgemmCase:
    def __init__(self, m, k, n, bits, count, device):
        self.m = m
        self.k = k
        self.n = n
        self.count = count
        self.a_bf16 = torch.randn((m, k), dtype = torch.bfloat16, device = device)
        self.bs = [
            make_trellis(k, n, bits, device, 4000 + bits * 10 + index)
            for index in range(count)
        ]
        self.shared_suh = make_signs(k, device, 5000 + bits)
        self.svhs = [make_signs(n, device, 6000 + index) for index in range(count)]
        self.b_ptrs = self.pointer_tensor(self.bs, device)
        self.suh_ptrs = self.pointer_tensor([self.shared_suh] * count, device)
        self.unique_suh_ptrs = self.pointer_tensor([self.shared_suh], device)
        self.svh_ptrs = self.pointer_tensor(self.svhs, device)
        self.had_group_ids = torch.zeros(count, dtype = torch.int32)
        self.a_had = torch.empty(
            (count, m, k), dtype = torch.float16, device = device
        )
        self.a_had_grouped = torch.empty(
            (1, m, k), dtype = torch.float16, device = device
        )
        self.scratch = torch.empty(
            (count, m, n), dtype = torch.float16, device = device
        )
        self.output = torch.empty(
            (m, count * n), dtype = torch.bfloat16, device = device
        )

    @staticmethod
    def pointer_tensor(tensors, device):
        return torch.tensor(
            [tensor.data_ptr() for tensor in tensors],
            dtype = torch.int64,
            device = device,
        )

    def run(
        self,
        force_num_sms = 0,
        direct_output = False,
        final_group_barrier = False,
    ):
        ext.exl3_mgemm_bf16_io(
            self.a_bf16,
            self.b_ptrs,
            self.scratch,
            self.output,
            self.suh_ptrs,
            self.a_had,
            self.svh_ptrs,
            self.bs[0].size(2) // 16,
            force_num_sms,
            self.output.size(1),
            True,
            direct_output,
            final_group_barrier,
        )

    def run_grouped(
        self,
        force_num_sms = 0,
        direct_output = False,
        final_group_barrier = False,
    ):
        ext.exl3_mgemm_bf16_io_grouped_had(
            self.a_bf16,
            self.b_ptrs,
            self.scratch,
            self.output,
            self.unique_suh_ptrs,
            self.a_had_grouped,
            self.svh_ptrs,
            self.had_group_ids,
            self.bs[0].size(2) // 16,
            force_num_sms,
            self.output.size(1),
            True,
            direct_output,
            final_group_barrier,
        )

    def reference(self):
        a_fp16 = self.a_bf16.to(torch.float16)
        outputs = []
        for b, svh in zip(self.bs, self.svhs):
            a_had = torch.empty_like(a_fp16)
            output = torch.empty(
                (self.m, self.n), dtype = torch.float16, device = a_fp16.device
            )
            ext.exl3_gemm(
                a_fp16,
                b,
                output,
                self.shared_suh,
                a_had,
                svh,
                2,
                True,
                False,
                0,
            )
            outputs.append(output.to(torch.bfloat16))
        return torch.cat(outputs, dim = 1)


@pytest.mark.parametrize("bits", (5, 6))
@pytest.mark.parametrize("m", (4, 16))
@torch.inference_mode()
def test_exl3_gemm_bf16_io_matches_fp16_boundary(bits, m):
    device = require_supported_device()
    case = GemmCase(m, 128, 128, bits, device)
    reference = case.reference()
    case.run(final_grid_sync = False)
    torch.cuda.synchronize()
    without_final_sync = case.output.clone()
    case.run(final_grid_sync = True)
    torch.cuda.synchronize()
    with_final_sync = case.output.clone()
    torch.testing.assert_close(with_final_sync, without_final_sync, rtol = 0, atol = 0)
    assert relative_rms(without_final_sync, reference) < 0.003
    assert float(torch.nn.functional.cosine_similarity(
        without_final_sync.float().flatten(), reference.float().flatten(), dim = 0
    )) > 0.99999


@pytest.mark.parametrize("bits", (5, 6))
@pytest.mark.parametrize("m", (4, 16))
@torch.inference_mode()
def test_exl3_mgemm_bf16_io_modes_and_graph(bits, m):
    device = require_supported_device()
    case = MgemmCase(m, 128, 128, bits, 2, device)
    reference = case.reference()

    case.run(direct_output = False, final_group_barrier = True)
    torch.cuda.synchronize()
    ordinary = case.output.clone()
    assert relative_rms(ordinary, reference) < 0.003
    assert float(torch.nn.functional.cosine_similarity(
        ordinary.float().flatten(), reference.float().flatten(), dim = 0
    )) > 0.99999

    case.run(force_num_sms = 1, direct_output = False, final_group_barrier = True)
    torch.cuda.synchronize()
    forced_one_sm = case.output.clone()
    assert relative_rms(forced_one_sm, reference) < 0.003
    assert float(torch.nn.functional.cosine_similarity(
        forced_one_sm.float().flatten(), reference.float().flatten(), dim = 0
    )) > 0.99999

    case.run(direct_output = False, final_group_barrier = False)
    torch.cuda.synchronize()
    ordinary_no_barrier = case.output.clone()
    torch.testing.assert_close(ordinary_no_barrier, ordinary, rtol = 0, atol = 0)

    case.run(direct_output = True, final_group_barrier = False)
    torch.cuda.synchronize()
    ordinary_direct = case.output.clone()
    torch.testing.assert_close(ordinary_direct, ordinary, rtol = 0, atol = 0)

    case.run_grouped(direct_output = False, final_group_barrier = True)
    torch.cuda.synchronize()
    grouped_scratch = case.output.clone()
    torch.testing.assert_close(grouped_scratch, ordinary, rtol = 0, atol = 0)

    case.run_grouped(direct_output = True, final_group_barrier = False)
    torch.cuda.synchronize()
    grouped_direct = case.output.clone()
    torch.testing.assert_close(grouped_direct, grouped_scratch, rtol = 0, atol = 0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run_grouped(direct_output = True, final_group_barrier = False)
    graph.replay()
    torch.cuda.synchronize()
    before = case.output.clone()
    case.a_bf16.copy_(torch.randn_like(case.a_bf16))
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(case.output, before)


@torch.inference_mode()
def test_exl3_bf16_io_rejects_invalid_layouts():
    device = require_supported_device()
    case = GemmCase(4, 128, 128, 5, device)

    with pytest.raises(RuntimeError, match = "1 <= m <= 16"):
        bad_a = torch.empty((0, 128), dtype = torch.bfloat16, device = device)
        ext.exl3_gemm_bf16_io(
            bad_a, case.b, case.scratch[:0], case.output[:0], case.suh,
            case.a_had[:0], case.svh, 2, 0, True, False,
        )

    with pytest.raises(RuntimeError, match = "128-aligned k"):
        bad_a = torch.empty((4, 64), dtype = torch.bfloat16, device = device)
        bad_b = make_trellis(64, 128, 5, device, 7001)
        bad_suh = torch.empty(64, dtype = torch.float16, device = device)
        bad_a_had = torch.empty_like(bad_a, dtype = torch.float16)
        ext.exl3_gemm_bf16_io(
            bad_a, bad_b, case.scratch, case.output, bad_suh, bad_a_had,
            case.svh, 2, 0, True, False,
        )

    with pytest.raises(RuntimeError, match = "exact K5 or K6"):
        bad_b = torch.empty(
            (8, 8, 81), dtype = torch.int16, device = device
        )
        ext.exl3_gemm_bf16_io(
            case.a_bf16, bad_b, case.scratch, case.output, case.suh,
            case.a_had, case.svh, 2, 0, True, False,
        )

    with pytest.raises(RuntimeError, match = "nonnegative"):
        case.run(force_num_sms = -1)

    properties = torch.cuda.get_device_properties(device)
    with pytest.raises(RuntimeError, match = "invalid EXL3 BF16 I/O SM count"):
        case.run(force_num_sms = properties.multi_processor_count + 1)


@torch.inference_mode()
def test_exl3_grouped_had_rejects_invalid_metadata_before_launch():
    device = require_supported_device()
    case = MgemmCase(4, 128, 128, 5, 2, device)

    case.had_group_ids = case.had_group_ids.to(device)
    with pytest.raises(RuntimeError, match = "CPU tensor"):
        case.run_grouped()

    case.had_group_ids = torch.tensor((0, -1), dtype = torch.int32)
    with pytest.raises(RuntimeError, match = "outside"):
        case.run_grouped()

    case.had_group_ids = torch.tensor((0, 1), dtype = torch.int32)
    with pytest.raises(RuntimeError, match = "outside"):
        case.run_grouped()

    case.had_group_ids = torch.zeros((1, 2), dtype = torch.int32)
    with pytest.raises(RuntimeError):
        case.run_grouped()

    case.had_group_ids = torch.zeros(2, dtype = torch.int32)
    case.run_grouped()
    torch.cuda.synchronize()


@torch.inference_mode()
def test_exl3_mgemm_bf16_io_rejects_invalid_layouts():
    device = require_supported_device()
    case = MgemmCase(4, 128, 128, 5, 2, device)

    def run(a, b_ptrs, scratch, output, suh_ptrs, a_had, svh_ptrs):
        ext.exl3_mgemm_bf16_io(
            a,
            b_ptrs,
            scratch,
            output,
            suh_ptrs,
            a_had,
            svh_ptrs,
            5,
            0,
            output.size(1),
            True,
            False,
            False,
        )

    with pytest.raises(RuntimeError, match = "one-dimensional|1"):
        run(
            case.a_bf16,
            case.b_ptrs.view(1, 2),
            case.scratch,
            case.output,
            case.suh_ptrs,
            case.a_had,
            case.svh_ptrs,
        )

    with pytest.raises(RuntimeError, match = "matrix count"):
        run(
            case.a_bf16,
            case.b_ptrs[:0],
            case.scratch[:0],
            case.output[:, :0],
            case.suh_ptrs[:0],
            case.a_had[:0],
            case.svh_ptrs[:0],
        )

    with pytest.raises(RuntimeError, match = "128-aligned k"):
        bad_a = torch.empty((4, 64), dtype = torch.bfloat16, device = device)
        bad_a_had = torch.empty(
            (2, 4, 64), dtype = torch.float16, device = device
        )
        run(
            bad_a,
            case.b_ptrs,
            case.scratch,
            case.output,
            case.suh_ptrs,
            bad_a_had,
            case.svh_ptrs,
        )

    with pytest.raises(RuntimeError, match = "128-aligned n"):
        bad_scratch = torch.empty(
            (2, 4, 64), dtype = torch.float16, device = device
        )
        bad_output = torch.empty(
            (4, 128), dtype = torch.bfloat16, device = device
        )
        run(
            case.a_bf16,
            case.b_ptrs,
            bad_scratch,
            bad_output,
            case.suh_ptrs,
            case.a_had,
            case.svh_ptrs,
        )

    with pytest.raises(RuntimeError, match = "does not hold every matrix"):
        bad_output = torch.empty(
            (4, 128), dtype = torch.bfloat16, device = device
        )
        run(
            case.a_bf16,
            case.b_ptrs,
            case.scratch,
            bad_output,
            case.suh_ptrs,
            case.a_had,
            case.svh_ptrs,
        )
