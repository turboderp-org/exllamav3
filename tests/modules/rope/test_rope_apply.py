"""
RoPE.apply / RoPE.apply_torch without keys (k = None, as for query-only rotations): the query comes out the same
as when rotated together with a key, for 3-D (seq, heads, dim) and 4-D (bsz, seq, heads, dim) inputs, in and out
of place.
"""

import pytest
import torch

from exllamav3.util.rope import RoPE, RopeSettings


def make_rope(device):
    return RoPE(device, RopeSettings(head_dim = 64, rope_theta = 10000.0, rope_scaling = None,
                                     max_position_embeddings = 4096))


@pytest.mark.parametrize("fn", ["apply", "apply_torch"])
@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("in_place", [False, True])
@torch.inference_mode()
def test_query_only(device, fn, rank, in_place):
    rope = make_rope(device)
    shape = (5, 4, 64) if rank == 3 else (2, 5, 4, 64)
    torch.manual_seed(0)
    q = torch.randn(shape, dtype = torch.half, device = device)
    k = torch.randn(shape, dtype = torch.half, device = device)
    apply = getattr(rope, fn)
    want, _ = apply(q.clone(), k.clone(), 7, in_place = in_place)
    got, got_k = apply(q.clone(), None, 7, in_place = in_place)
    assert got_k is None
    assert torch.equal(got, want)
