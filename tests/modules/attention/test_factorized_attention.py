import torch

from exllamav3.modules.attention_fn.factorized_triton import factorized_attention_decode, subtract_noise
from exllamav3.modules.factorized_attention import shift_kv
from exllamav3.modules.rmsnorm import zero_centered_rms_norm


def test_factorized_attention_decode_matches_torch(device):
    rng = torch.Generator(device = device).manual_seed(79)
    batch, q_heads, kv_heads, dim, rank = 2, 12, 3, 128, 4
    window, eps = 32768, 1e-6

    def rand(*shape):
        return torch.randn(*shape, generator = rng, device = device)

    q = rand(batch, 1, q_heads, dim)
    ak, av = (rand(batch, 1, kv_heads, rank) for _ in range(2))
    bk, bv = (rand(batch, 1, rank, dim) for _ in range(2))
    shift_k, shift_v = (rand(batch, 1, kv_heads) for _ in range(2))
    params = {
        "k_norm.norm.weight": rand(dim).bfloat16(),
        "q_norm.norm.weight": rand(dim).bfloat16(),
        "softmax_scaler": rand(q_heads),
    }
    positions = torch.tensor([[0], [window + 1]], dtype = torch.int32, device = device)
    slots = torch.tensor([2, 0], dtype = torch.int32, device = device)
    state = (rand(3, kv_heads, dim), rand(3, kv_heads, dim))
    untouched = [tensor[1].clone() for tensor in state]
    raw_k, raw_v = (ak @ bk) / rank, (av @ bv) / rank
    k = shift_kv(raw_k, state[0][slots], shift_k, positions)
    k = zero_centered_rms_norm(k, params["k_norm.norm.weight"], eps)
    v = shift_kv(raw_v, state[1][slots], shift_v, positions)
    pos_scale = (positions.float() + 1).clamp_max(window).log()
    query = zero_centered_rms_norm(q, params["q_norm.norm.weight"], eps)
    query = query * pos_scale[..., None, None] * params["softmax_scaler"][None, None, :, None]
    actual = factorized_attention_decode(
        q, ak, bk, av, bv, shift_k, shift_v, params, slots, positions, state,
        q_heads, kv_heads, dim, rank, window, eps,
    )

    for output, expected in zip(actual, (query, k, v), strict = True):
        torch.testing.assert_close(output, expected.half(), atol = 1e-3, rtol = 1e-3)
    for tensor, raw, unchanged in zip(state, (raw_k, raw_v), untouched, strict = True):
        torch.testing.assert_close(tensor[slots], raw.squeeze(1), atol = 2e-6, rtol = 2e-6)
        assert torch.equal(tensor[1], unchanged)


def test_subtract_noise_preserves_fp32_rounding(device):
    rng = torch.Generator(device = device).manual_seed(31)
    attn = torch.randn((2, 7, 12, 128), device = device, dtype = torch.float16, generator = rng)
    weights = torch.rand((2, 7, 3), device = device, generator = rng)
    for x in (attn, attn[:, ::2]):
        w = weights if x.is_contiguous() else weights[:, ::2]
        groups = x.float().unflatten(-2, (3, 4))
        expected = (groups[..., :-1, :] - w[..., None, None] * groups[..., -1:, :]).flatten(-3)
        torch.testing.assert_close(subtract_noise(x, w, torch.float32), expected, atol = 0, rtol = 0)
