import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.rope import RoPE, RopeStyle, RopeSettings
import torch.testing

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

device = "cuda:2"

# ((bsz, seq_len, num_heads_q, head_dim), (bsz, seq_len, num_heads_k, head_dim))
qk_dims = [
    ((1, 1, 8, 128), (1, 1, 8, 128)),
    ((1, 864, 8, 128), (1, 864, 8, 128)),
    ((1, 864, 128, 128), (1, 864, 8, 128)),
    ((1, 864, 8, 64), (1, 864, 8, 64)),
    ((1, 64, 8, 64), (1, 64, 2, 64)),
    ((1, 10, 80, 256), (1, 10, 10, 256)),
    ((1, 600, 80, 256), (1, 600, 10, 256)),
    ((5, 47, 80, 128), (5, 47, 10, 128)),
    ((17, 1, 32, 256), (17, 1, 10, 256)),
    ((1, 1, 28, 64), (1, 1, 7, 64)),
    ((1, 1, 28, 96), (1, 1, 7, 96)),
    ((1, 1, 28, 80), (1, 1, 7, 80)),
    ((1, 1, 28, 32), (1, 1, 7, 32)),
]

rope_styles = {RopeStyle.GPTJ, RopeStyle.NEOX}
# rope_styles = [RopeStyle.NEOX]
# rope_styles = [RopeStyle.GPTJ]

norm_opt = [False, True]

@pytest.mark.parametrize("qk_dim", qk_dims)
@pytest.mark.parametrize("rope_style", rope_styles)
@pytest.mark.parametrize("use_norm", norm_opt)
@torch.inference_mode()
def test_rope(qk_dim, rope_style, use_norm):

    def qk():
        torch.manual_seed(0)
        q_pr = torch.randn(qk_dim[0], dtype = torch.half, device = device)
        k_pr = torch.randn(qk_dim[1], dtype = torch.half, device = device) if qk_dim[1] else None
        return q_pr, k_pr

    bsz, seq_len, _, head_dim = qk_dim[0]

    rope_layer = RoPE(
        device = device,
        rope_settings = RopeSettings(
            rope_theta = 1.0,
            head_dim = head_dim,
            rope_scaling = None,
            max_position_embeddings = 32768,
            partial_rotary_factor = 1.0,
            rope_style = rope_style,
        )
    )

    def apply_norm(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
        constant_bias: float
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim = -1, keepdim = True) + eps
        x = x * torch.rsqrt(var)
        x = x.to(dtype)
        x = x * (w + constant_bias)
        return x

    def run(position, positions, position_ids):
        q, k = qk()
        eps = 1e-6
        constant_bias = 0.0
        if use_norm:
            norm_q = torch.randn(head_dim, device = q.device, dtype = torch.half) / 2.0
            norm_k = torch.randn(head_dim, device = k.device, dtype = torch.half) / 2.0
            q = apply_norm(q, norm_q, eps, constant_bias)
            k = apply_norm(k, norm_k, eps, constant_bias)
        else:
            norm_q = None
            norm_k = None
        q_ref, k_ref = rope_layer.apply_torch(q, k, position, positions, position_ids)
        q, k = qk()
        q, k = rope_layer.apply(q, k, position, positions, position_ids, True, norm_q, norm_k, eps, constant_bias)
        torch.testing.assert_close(q, q_ref, rtol = 3e-3, atol = 3e-3)
        if k is not None:
            torch.testing.assert_close(k, k_ref, rtol = 3e-3, atol = 3e-3)

    # No offset
    run(0, None, None)

    # Some offset
    run(19, None, None)

    # Batched offset
    run(0, torch.randint(size = (bsz,), low = 0, high = 49, dtype = torch.int, device = device), None)

    # Batched position ids
    run(0, None, torch.randint(size = (bsz, seq_len), low = 0, high = 117, dtype = torch.int, device = device))