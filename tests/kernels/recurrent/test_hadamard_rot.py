"""
The packed int pool's per-32-wide-k-slice rotation (gdn.cu, had32_warp) is probed through the scan
kernel: a zeroed initial state, one step, g = 0 (no decay), beta = 1 and v = 1 at v-column 0 make the
pool slice (head 0, v column 0, k-block 0) the rotated basis vector H @ e_j. The probe quantizes
exactly to +/-Q for every width W, so the stored scale is the slice's exact absmax and decoding by it
leaves the pure sign pattern sgn(H @ e_j). Since sgn(H @ e_j) = sqrt(32) * H @ e_j and
H @ sgn(H @ e_j) = sqrt(32) * e_j, the recovered column pins the literal Sylvester pattern; a
different-but-orthogonal H32 (Paley) must fail it.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.hadamard import get_hadamard, paley_torch
from testlib.gdn import state_from_pool

K_HEAD_DIM = V_HEAD_DIM = 128
SUBK = K_HEAD_DIM // 32


def _get_h32(dev):
    # Sylvester H32, normalized so H^2 = I: get_hadamard(32) is the recursive construction the
    # kernel's butterfly emits, H[i,j] = (-1)^popcount(i & j)
    return (get_hadamard(32).float() * 32 ** -0.5).to(device = dev)


@pytest.mark.parametrize("bits", [8, 9, 10, 12], ids = ["w8", "w9", "w10", "w12"])
@pytest.mark.parametrize("j", [0, 1, 5, 17, 31])
@torch.inference_mode()
def test_int_state_rotates_the_update_by_sylvester_h32(device, bits, j):
    nk = nv = 1
    seqlen = 1
    mixed_qkv = torch.zeros((1, seqlen, 2 * nk * K_HEAD_DIM + nv * V_HEAD_DIM), device = device).bfloat16()
    mixed_qkv[0, 0, nk * K_HEAD_DIM + j] = 1.0        # k = e_j (unit vector: the l2 norm is a no-op)
    mixed_qkv[0, 0, 2 * nk * K_HEAD_DIM] = 1.0        # v = 1 at v-column 0
    g = torch.zeros((1, seqlen, nv), device = device)  # exp(0) = 1: the zero state decays to zero
    beta = torch.ones((1, seqlen, nv), device = device).bfloat16()
    pool = torch.zeros((2, 1, nv, V_HEAD_DIM, K_HEAD_DIM * bits // 8), dtype = torch.int8, device = device)
    scales = torch.zeros((2, 1, nv, V_HEAD_DIM, SUBK), dtype = torch.float16, device = device)
    out = torch.empty((1, seqlen, nv, V_HEAD_DIM), dtype = torch.bfloat16, device = device)

    ext.cuda_recurrent_gated_delta_rule(mixed_qkv, g, beta, pool, out, nk, nv, K_HEAD_DIM, V_HEAD_DIM,
                                        None, False, None, scales, bits)
    torch.cuda.synchronize(device)

    amax = scales[0, 0, 0, 0, 0].float()
    assert amax > 0
    # The probe value is exactly +/-Q at every width, so the stored scale is the slice's exact
    # absmax: decoding and dividing by it leaves the pure sign pattern sgn(H @ e_j)
    slice_ = state_from_pool(pool, scales, bits)[0, 0, 0, :32, 0] / amax
    h32 = _get_h32(device)
    eye = torch.eye(32, device = device)
    assert torch.allclose(h32 @ h32, eye, rtol = 2e-2, atol = 2e-2)   # normalized: H^2 = I
    # H @ sgn(H @ e_j) = sqrt(32) * e_j: the recovered unit column pins the literal Sylvester
    # pattern, fp16 scale rounding plus fp arithmetic stay within 2e-2
    torch.testing.assert_close(h32 @ slice_, 32 ** 0.5 * eye[:, j], rtol = 2e-2, atol = 2e-2)
    # Negative control: an orthogonal but different H32 (Paley) does not match the probe, so the
    # assertion above pins the literal Sylvester pattern, not just orthogonality
    paley_m = (paley_torch(32).float() * 32 ** -0.5).to(device = device)
    assert torch.allclose(paley_m @ paley_m.T, eye, rtol = 2e-2, atol = 2e-2)   # orthogonal, not symmetric
    assert (h32 @ slice_ - paley_m[:, j]).abs().max().item() > 2e-2
