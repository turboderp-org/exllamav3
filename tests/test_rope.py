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
    ((1, 64, 32, 128), None),
    ((1, 64, 32, 128), (1, 64, 8, 128)),
    ((1, 64, 8, 64), (1, 64, 2, 64)),
    ((1, 810, 80, 256), (1, 810, 10, 256)),
    ((5, 47, 80, 128), (5, 47, 10, 128)),
    ((17, 1, 32, 256), None),
    ((17, 1, 32, 256), (17, 1, 10, 256)),
    ((1, 1, 28, 64), (1, 1, 7, 64)),
    ((1, 1, 28, 96), (1, 1, 7, 96)),
    ((1, 1, 28, 32), (1, 1, 7, 32)),
    ((5, 512, 13, 48), None),
    ((1, 64, 16, 100), None),
]

# rope_styles = [RopeStyle.GPTJ, RopeStyle.NEOX]
rope_styles = [RopeStyle.NEOX]

@pytest.mark.parametrize("qk_dim", qk_dims)
@pytest.mark.parametrize("rope_style", rope_styles)
@torch.inference_mode()
def test_rope(qk_dim, rope_style):

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

    def run(position, positions, position_ids):
        q, k = qk()
        rope_layer.apply_torch(q, k, position, positions, position_ids)
        q_ref, k_ref = q, k
        q, k = qk()
        rope_layer.apply(q, k, position, positions, position_ids)
        torch.testing.assert_close(q, q_ref, rtol = 1e-6, atol = 1e-6)
        if k is not None:
            torch.testing.assert_close(k, k_ref, rtol = 1e-6, atol = 1e-6)

    # No offset
    run(0, None, None)

    # Some offset
    run(19, None, None)

    # Batched offset
    run(0, torch.randint(size = (bsz,), low = 0, high = 49, dtype = torch.int, device = device), None)

    # Batched position ids
    run(0, None, torch.randint(size = (bsz, seq_len), low = 0, high = 117, dtype = torch.int, device = device))