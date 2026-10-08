"""
Half-integer bitrates (1.5 / 2.5 / 3.5 bpw, mul1) in the fused MoE prefill kernel (ext.exl3_moe), whose instances
are compiled per rate: single 16-row launches, per-range 16 / 32 / 64-row tiles and the mixed-rate runtime-switch
instance, against the dense reconstruct path (testlib.moe.expert_mlp_ref). The CPU expert kernels at these rates
are covered in tests/moe_cpu/test_half_rates_cpu.py.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_experts, rand_scale, rand_trellis
from testlib.moe import contiguous_layout, expert_mlp_ref, expert_ptr_tables, experts_to, moe_buffers

HALF_RATES = (1.5, 2.5, 3.5)
CAP = 128


@pytest.mark.parametrize("K", HALF_RATES)
@pytest.mark.parametrize("dims", [(256, 128), (512, 256)])   # N = 128 and N = 256 tile shapes
def test_fused_prefill_instances(device, K, dims):
    """The compiled half-rate instances of exl3_moe, 16 / 32 / 64-row tiles, against the reconstruct path"""
    H, I = dims
    counts = [1, 5, 16, 17, 32, 33, 48, 64, 100]
    E = len(counts)
    gen = torch.Generator().manual_seed(11)
    exd = experts_to(rand_experts(E, H, I, K, gen), device)
    tabs = expert_ptr_tables(exd, device)
    A = sum(counts)
    y = (torch.randn((A, H), generator = gen) * 0.05).half().to(device)
    weight_sorted = (torch.rand((A,), generator = gen) * 0.5 + 0.5).half().to(device)
    token_sorted, expert_count, starts = contiguous_layout(counts, device)
    bufs = moe_buffers(H, I, CAP, device)

    def ref_of(ex_, K_down):
        ref = torch.empty((A, H), dtype = torch.float, device = device)
        for e in range(E):
            a = int(starts[e])
            rows = slice(a, a + counts[e])
            ref[rows] = expert_mlp_ref(ex_, e, y[rows], K, K_down) * weight_sorted[rows].float().unsqueeze(1)
        return ref

    def launch(tabs_, K_down, lo, hi, m_tile, out):
        ext.exl3_moe(y, out, expert_count, token_sorted, weight_sorted, *bufs, 0, K, K, K_down,
                     *tabs_["g"], *tabs_["u"], *tabs_["d"], False, True, False, True, False, True, 0.0, -1,
                     None, None, lo, hi, m_tile, 0, 0, 0, 0, 0, 0)

    ref = ref_of(exd, K)
    scale = ref.abs().max().item()
    single = torch.zeros((A, H), dtype = torch.float, device = device)
    launch(tabs, K, 1, CAP, 16, single)
    tiered = torch.zeros((A, H), dtype = torch.float, device = device)
    launch(tabs, K, 33, CAP, 64, tiered)
    launch(tabs, K, 17, 32, 32, tiered)
    launch(tabs, K, 1, 16, 16, tiered)
    torch.cuda.synchronize()
    assert (single - ref).abs().max().item() <= 3e-3 * scale, "single 16-row launch"
    assert (tiered - ref).abs().max().item() <= 3e-3 * scale, "per-range row tiles"

    # A mixed-rate launch takes the runtime-switch instance: same arithmetic, different kernel
    ex2 = rand_experts(E, H, I, K, gen)
    ex2["d"] = [(rand_trellis(I, H, 4, gen), rand_scale(I, gen), rand_scale(H, gen)) for _ in range(E)]
    exd2 = experts_to(ex2, device)
    mixed = torch.zeros((A, H), dtype = torch.float, device = device)
    launch(expert_ptr_tables(exd2, device), 4, 1, CAP, 16, mixed)
    torch.cuda.synchronize()
    ref2 = ref_of(exd2, 4)
    assert (mixed - ref2).abs().max().item() <= 3e-3 * ref2.abs().max().item(), "mixed-rate launch"
