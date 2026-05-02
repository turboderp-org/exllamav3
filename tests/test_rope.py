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


@pytest.mark.parametrize("rope_style", rope_styles)
@pytest.mark.parametrize("use_norm", norm_opt)
@pytest.mark.parametrize("in_place", [False, True])
@torch.inference_mode()
def test_rope_multidim(rope_style, use_norm, in_place):

    bsz = 2
    seq_len = 280
    num_heads = 16
    head_dim = 72
    rotate_dims = 2
    partial_head_dim = head_dim // rotate_dims

    rope_layer = RoPE(
        device = device,
        rope_settings = RopeSettings(
            rope_theta = 100.0,
            head_dim = partial_head_dim,
            rope_scaling = None,
            max_position_embeddings = 131072,
            partial_rotary_factor = 1.0,
            rope_style = rope_style,
            rotate_dims = rotate_dims,
        )
    )

    def qk():
        torch.manual_seed(0)
        q_pr = torch.randn((bsz, seq_len, num_heads, head_dim), dtype = torch.half, device = device)
        k_pr = torch.randn((bsz, seq_len, num_heads, head_dim), dtype = torch.half, device = device)
        return q_pr, k_pr

    def apply_norm(
        x: torch.Tensor,
        w: torch.Tensor | None,
        eps: float,
        constant_bias: float
    ) -> torch.Tensor:
        x = x.float()
        var = x.pow(2).mean(dim = -1, keepdim = True) + eps
        x = x * torch.rsqrt(var)
        if w is not None:
            x = x * (w.float() + constant_bias)
        return x.half()

    def apply_rope_embed(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)
        if rope_style == RopeStyle.NEOX:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            xr = torch.cat((-x2, x1), dim = -1)
        else:
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            xr = torch.stack((-x2, x1), dim = -1).flatten(-2)
        return (x * cos + xr * sin).transpose(1, 2).half()

    def apply_multidim_ref(
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        q_norm: torch.Tensor | None,
        k_norm: torch.Tensor | None,
        eps: float,
        constant_bias: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q_norm is not None:
            q = apply_norm(q, q_norm, eps, constant_bias)
            k = apply_norm(k, k_norm, eps, constant_bias)

        out_q = []
        out_k = []
        for rdim in range(rotate_dims):
            start = partial_head_dim * rdim
            end = start + partial_head_dim
            pos = position_ids[:, :, rdim].float()
            freqs = torch.einsum("bi,j->bij", pos, rope_layer.inv_freq.float())
            sin = freqs.sin() * rope_layer.attn_factor
            cos = freqs.cos() * rope_layer.attn_factor
            if rope_style == RopeStyle.NEOX:
                sin = torch.cat((sin, sin), dim = -1)
                cos = torch.cat((cos, cos), dim = -1)
            else:
                sin = torch.repeat_interleave(sin, 2, dim = -1)
                cos = torch.repeat_interleave(cos, 2, dim = -1)
            out_q.append(apply_rope_embed(q[..., start : end], sin, cos))
            out_k.append(apply_rope_embed(k[..., start : end], sin, cos))

        return torch.cat(out_q, dim = -1), torch.cat(out_k, dim = -1)

    base = torch.arange(seq_len, dtype = torch.int, device = device)
    position_ids = torch.stack((base % 20, base // 20), dim = -1).unsqueeze(0).repeat(bsz, 1, 1)
    position_ids[1, :, 0] += 3
    position_ids[1, :, 1] += 5

    q, k = qk()
    eps = 1e-6
    constant_bias = 0.0
    if use_norm:
        torch.manual_seed(1)
        norm_q = torch.randn(head_dim, device = device, dtype = torch.half) / 2.0
        norm_k = torch.randn(head_dim, device = device, dtype = torch.half) / 2.0
    else:
        norm_q = None
        norm_k = None

    q_ref, k_ref = apply_multidim_ref(q, k, position_ids, norm_q, norm_k, eps, constant_bias)
    q, k = qk()
    q_pre = q.clone()
    k_pre = k.clone()
    q_out, k_out = rope_layer.apply(q, k, 0, None, position_ids, in_place, norm_q, norm_k, eps, constant_bias)
    torch.testing.assert_close(q_out, q_ref, rtol = 3e-3, atol = 3e-3)
    torch.testing.assert_close(k_out, k_ref, rtol = 3e-3, atol = 3e-3)
    if not in_place:
        torch.testing.assert_close(q, q_pre, rtol = 0, atol = 0)
        torch.testing.assert_close(k, k_pre, rtol = 0, atol = 0)
