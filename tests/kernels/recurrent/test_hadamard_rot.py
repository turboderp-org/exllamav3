"""
The int8 pool's per-32-wide-k-slice rotation (gdn.cu, had32_warp) is probed through the scan kernel:
a zeroed initial state, one step, g = 0 (no decay), beta = 1 and v = 1 at v-column 0 make the
dequantized pool slice (head 0, v column 0, k-block 0) the rotated basis vector H @ e_j. The
comparison is against the literal Sylvester pattern; a different-but-orthogonal H32 (Paley) must
fail it.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.hadamard import get_hadamard, paley_torch

K_HEAD_DIM = V_HEAD_DIM = 128
SUBK = K_HEAD_DIM // 32


def _get_h32(dev):
    # Sylvester H32, normalized so H^2 = I: get_hadamard(32) is the recursive construction the
    # kernel's butterfly emits, H[i,j] = (-1)^popcount(i & j)
    return (get_hadamard(32).float() * 32 ** -0.5).to(device = dev)


@pytest.mark.parametrize("j", [0, 1, 5, 17, 31])
@torch.inference_mode()
def test_int8_state_rotates_the_update_by_sylvester_h32(device, j):
    nk = nv = 1
    seqlen = 1
    mixed_qkv = torch.zeros((1, seqlen, 2 * nk * K_HEAD_DIM + nv * V_HEAD_DIM), device = device).bfloat16()
    mixed_qkv[0, 0, nk * K_HEAD_DIM + j] = 1.0        # k = e_j (unit vector: the l2 norm is a no-op)
    mixed_qkv[0, 0, 2 * nk * K_HEAD_DIM] = 1.0        # v = 1 at v-column 0
    g = torch.zeros((1, seqlen, nv), device = device)  # exp(0) = 1: the zero state decays to zero
    beta = torch.ones((1, seqlen, nv), device = device).bfloat16()
    pool = torch.zeros((2, 1, nv, K_HEAD_DIM, V_HEAD_DIM), dtype = torch.int8, device = device)
    scales = torch.zeros((2, 1, nv, V_HEAD_DIM, SUBK), dtype = torch.float16, device = device)
    out = torch.empty((1, seqlen, nv, V_HEAD_DIM), dtype = torch.bfloat16, device = device)

    ext.cuda_recurrent_gated_delta_rule(mixed_qkv, g, beta, pool, out, nk, nv, K_HEAD_DIM, V_HEAD_DIM,
                                        None, False, None, scales)
    torch.cuda.synchronize(device)

    amax = scales[0, 0, 0, 0, 0].float()
    assert amax > 0
    # The stored scale is the slice's absmax: step = amax / 127
    slice_ = pool[0, 0, 0, :32, 0].float() * amax / 127.0
    h32 = _get_h32(device)
    eye = torch.eye(32, device = device)
    assert torch.allclose(h32 @ h32, eye, rtol = 2e-2, atol = 2e-2)   # normalized: H^2 = I
    # int8 rounds once, so the pattern is within one quantization step (~1e-4 here) of the
    # Sylvester column -- a conservative 2e-2
    torch.testing.assert_close(slice_, h32[:, j], rtol = 2e-2, atol = 2e-2)
    # Negative control: an orthogonal but different H32 (Paley) does not match the probe, so the
    # assertion above pins the literal Sylvester pattern, not just orthogonality
    paley_m = (paley_torch(32).float() * 32 ** -0.5).to(device = device)
    assert torch.allclose(paley_m @ paley_m.T, eye, rtol = 2e-2, atol = 2e-2)   # orthogonal, not symmetric
    assert (slice_ - paley_m[:, j]).abs().max().item() > 2e-2
