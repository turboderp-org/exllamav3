"""
DFlash2 sampled selector walk (speculative sampling): the CUDA kernel (ext.dflash2_selector_walk_sample) must draw
the same path as the torch formulation for the same Gumbel noise and return the same proposal distribution q.

    python -m pytest tests/test_dflash2_walk_sample.py -v
"""
import pytest
import torch

from exllamav3.modules.arch_specific.dflash2 import DFlash2Selector

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda:0"


class _Proj:
    def __init__(self, w):
        self.w = w
    def forward(self, x, params = None):
        return (x.float() @ self.w).half()


class _Sel:
    walk_sample = DFlash2Selector.walk_sample
    def __init__(self, vocab, hidden, rank, top_k, dtype):
        g = torch.Generator(device = DEV).manual_seed(3)
        self.pred_codebook = (torch.randn((vocab, rank), device = DEV, generator = g) * 0.3).to(dtype)
        self.succ_codebook = (torch.randn((vocab, rank), device = DEV, generator = g) * 0.3).to(dtype)
        self.hidden_proj = _Proj(torch.randn((hidden, rank), device = DEV, generator = g) * 0.05)
        self.top_k = top_k


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.6, 1.0, 1.5])
def test_walk_sample_cuda_matches_torch(dtype, temperature):
    vocab, hidden, rank, top_k, bsz, rows = 4096, 256, 64, 16, 2, 7
    sel = _Sel(vocab, hidden, rank, top_k, dtype)
    g = torch.Generator(device = DEV).manual_seed(11)
    h = torch.randn((bsz, rows, hidden), device = DEV, generator = g).half()
    logits = (torch.randn((bsz, rows, vocab), device = DEV, generator = g) * 3.0).half()
    anchor = torch.randint(0, vocab, (bsz,), device = DEV, generator = g)
    out_c, q_c, c_c = sel.walk_sample(h, logits, anchor, temperature, generator = torch.Generator(device = DEV).manual_seed(5))
    # torch path: same noise (same generator seed), force the fallback with a CPU-typed check
    orig = sel.pred_codebook
    sel.pred_codebook = orig.float()           # dtype outside (half, bf16) -> torch formulation
    sel.succ_codebook = sel.succ_codebook.float()
    out_t, q_t, c_t = sel.walk_sample(h, logits, anchor, temperature, generator = torch.Generator(device = DEV).manual_seed(5))
    assert torch.equal(c_c, c_t)
    assert torch.equal(out_c, out_t), (out_c, out_t)
    assert (q_c - q_t).abs().max().item() < 2e-3
    assert torch.allclose(q_c.sum(-1), torch.ones_like(q_c.sum(-1)), atol = 1e-4)
