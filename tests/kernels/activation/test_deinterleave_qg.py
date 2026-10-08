"""
ext.deinterleave_qg(qg, q, g, head_dim): splits a per-head-interleaved projection output (the interleaved attention
output gate of attn.py: each head stores head_dim q values followed by head_dim gate values) into contiguous q and g.

    qg viewed as (n, 2, head_dim):  q.view(n, head_dim) = qg[:, 0],  g.view(n, head_dim) = qg[:, 1]

where n = q.numel() / head_dim, regardless of how the leading dims are shaped (callers pass q as (B, S, H, D) and
g as (B, S, H * D)). A pure copy: bit-exact, NaN/inf payloads preserved, qg untouched, nothing written outside q
and g. Rejects (TORCH_CHECK): non-fp16 dtypes, head_dim % 8 != 0, non-contiguous tensors, q.numel() != g.numel()
or qg.numel() != 2 * q.numel().

Reference: torch view / indexing.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

GUARD = 64


def guarded_empty(shape, device):
    n = 1
    for s in shape:
        n *= s
    buf = torch.full((n + 2 * GUARD,), -7.0, dtype = torch.half, device = device)
    return buf, buf[GUARD:GUARD + n].view(shape)


def assert_guards_intact(buf, n):
    for part in (buf[:GUARD], buf[GUARD + n:]):
        assert (part == -7.0).all(), "write outside the output tensor"


def random_bits_half(shape, device):
    # Arbitrary bit patterns (incl. NaN / inf / subnormals): a copy must preserve them exactly
    return torch.randint(-32768, 32768, shape, dtype = torch.int16, device = device).view(torch.half)


@pytest.mark.parametrize("bsz, q_len", [(1, 1), (1, 7), (2, 33), (1, 512)])
@pytest.mark.parametrize("heads", [1, 3, 32])
@pytest.mark.parametrize("head_dim", [8, 64, 72, 128, 136, 256])
@torch.inference_mode()
def test_deinterleave_qg(device, bsz, q_len, heads, head_dim):
    qg = random_bits_half((bsz, q_len, heads * 2 * head_dim), device)
    qg0 = qg.clone()
    qbuf, q = guarded_empty((bsz, q_len, heads, head_dim), device)
    gbuf, g = guarded_empty((bsz, q_len, heads * head_dim), device)
    ext.deinterleave_qg(qg, q, g, head_dim)
    split = qg0.view(-1, 2, head_dim)
    bits = lambda t: t.view(torch.int16)
    assert torch.equal(bits(q).view(-1, head_dim), bits(split[:, 0]))
    assert torch.equal(bits(g).view(-1, head_dim), bits(split[:, 1]))
    assert torch.equal(bits(qg), bits(qg0)), "qg modified"
    assert_guards_intact(qbuf, q.numel())
    assert_guards_intact(gbuf, g.numel())


@torch.inference_mode()
def test_deinterleave_qg_rejects(device):
    qg = torch.randn(2, 4 * 2 * 64, device = device).half()
    q = torch.empty(2, 4 * 64, device = device).half()
    g = torch.empty(2, 4 * 64, device = device).half()
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.deinterleave_qg(qg.float(), q, g, 64)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.deinterleave_qg(qg, q.float(), g, 64)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.deinterleave_qg(qg, q, g.float(), 64)
    with pytest.raises(RuntimeError, match = "multiple of 8"):
        ext.deinterleave_qg(qg, q, g, 60)
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.deinterleave_qg(torch.randn(512, 2, device = device).half().t(), q, g, 64)
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.deinterleave_qg(qg, torch.empty(256, 2, device = device).half().t(), g, 64)
    with pytest.raises(RuntimeError, match = "size mismatch"):
        ext.deinterleave_qg(qg, q[:1], g, 64)
    with pytest.raises(RuntimeError, match = "size mismatch"):
        ext.deinterleave_qg(qg[:1], q, g, 64)
