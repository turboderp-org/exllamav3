"""
MRoPE frequency layouts of RoPE.get_mrope_freqs against the two HF formulas: contiguous sections
(Qwen2.5-VL, GLM-4V: `freq.split(mrope_section)`, chunk i from component i % 3) when the config has no
mrope_interleaved flag, and the interleaved layout (Qwen3-VL, Qwen3.5, Qwen3.8: `apply_interleaved_mrope`)
when it does. The 3D position ids come from the same generator in both; only the assignment of (t, h, w)
positions to rotary frequencies is under test.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.rope import RoPE, RopeSettings, RopeStyle

# CPU only (RoPE on "cpu", the host-side ext.gen_mrope_pos_ids); the extension must be built but no device is used
pytestmark = pytest.mark.nogpu


def _reference(inv_freq, pos_ids, section, interleaved):
    """HF: freqs (3, L, half) from the 3D positions, then one of the two layouts over the half dim"""
    freqs = pos_ids.float()[:, :, None] * inv_freq[None, None, :]
    if interleaved:
        out = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start = 1):
            idx = slice(offset, section[dim] * 3, 3)
            out[:, idx] = freqs[dim, :, idx]
        return out
    chunks = freqs.split(section, dim = -1)
    return torch.cat([chunks[i][i % 3] for i in range(3)], dim = -1)


@pytest.mark.parametrize("name, head_dim, partial, section, interleaved", [
    ("glm4v", 128, 0.5, [8, 12, 12], False),
    ("qwen2_5_vl", 128, 1.0, [16, 24, 24], False),
    ("qwen3_vl", 128, 1.0, [24, 20, 20], True),
    ("qwen3_5", 256, 0.25, [11, 11, 10], True),
])
def test_mrope_layout(name, head_dim, partial, section, interleaved):
    scaling = {"rope_type": "default", "mrope_section": section}
    if interleaved:
        scaling["mrope_interleaved"] = True
    rs = RopeSettings(head_dim = head_dim, rope_theta = 10000.0, rope_scaling = scaling,
                      partial_rotary_factor = partial, rope_style = RopeStyle.NEOX)
    rope = RoPE("cpu", rs)
    assert rope.inv_freq.shape[-1] == sum(section), "mrope_section must cover the rotary half dim"

    # 7 text tokens, a 1 x 6 x 10 image (merged grid 3 x 5 at merge size 2) occupying the embedding's
    # reserved token-id range, 5 more text tokens. Spans are token-id ranges, not positions
    grid = (1, 6, 10); merge = 2
    n_img = grid[0] * (grid[1] // merge) * (grid[2] // merge)
    first = 1_000_000
    ids = torch.tensor([[100] * 7 + list(range(first, first + n_img)) + [100] * 5], dtype = torch.long)
    emb = SimpleNamespace(first_index = first, last_index = first + n_img, grid_thw = grid, mrope_merge_size = merge)
    freqs, next_pos = rope.get_mrope_freqs(ids, [emb], ids.shape[-1])

    pos_ids = torch.zeros((3, ids.shape[-1]), dtype = torch.long)
    ext.gen_mrope_pos_ids(pos_ids, ids.squeeze(0).contiguous(), merge, [(first, first + n_img)], [grid])
    assert not torch.equal(pos_ids[1], pos_ids[0]), "image rows must give the H component its own positions"
    ref = _reference(rope.inv_freq.float(), pos_ids, section, interleaved)
    assert torch.allclose(freqs.view(ids.shape[-1], -1), ref, rtol = 1e-5, atol = 1e-5), name

    # The other layout must differ on the image span (so the flag is what selects it)
    other = _reference(rope.inv_freq.float(), pos_ids, section, not interleaved)
    assert not torch.allclose(freqs.view(ids.shape[-1], -1), other, rtol = 1e-5, atol = 1e-5)
