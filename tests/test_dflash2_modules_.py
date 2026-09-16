"""
Parity tests for the DFlash2 modules against the reference implementation
(z-lab/dflash, dflash/model.py, MIT). The pure math functions are compared on
identical seeded tensors, so tolerances are tight.
"""
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    pytest.importorskip("dflash.model", reason="reference dflash package not installed") is None,
    reason="reference dflash package not installed",
)

from dflash.model import (  # noqa: E402
    CandidateSelector as RefSelector,
    GroupedDynamicCausalConv as RefConv,
)

from exllamav3.modules.arch_specific.dflash2 import (  # noqa: E402
    conv_finish,
    conv_prepare,
    selector_select,
)


def _mk_conv(hidden=256, k=2, g=8, seed=0):
    torch.manual_seed(seed)
    base = torch.randn(2, k, hidden) * 0.05
    proj_w = torch.randn(2 * k * (hidden // g), hidden) * 0.05
    ref = RefConv(hidden, k, g)
    with torch.no_grad():
        ref.base_kernel.copy_(base)
        ref.kernel_projection.weight.copy_(proj_w)
    return base, proj_w, ref, hidden, k, g


def test_conv_prepare_parity():
    base, proj_w, ref, hidden, k, g = _mk_conv()
    x = torch.randn(1, 7, hidden)
    out_ref, dyn_ref = ref.prepare(x)
    out_ours, dyn_ours = conv_prepare(x, base[0], proj_w, k, g)
    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dyn_ours, dyn_ref, atol=1e-5, rtol=1e-5)


def test_conv_finish_parity():
    base, proj_w, ref, hidden, k, g = _mk_conv(seed=1)
    x = torch.randn(1, 7, hidden)
    _, dyn = ref.prepare(x)
    y = torch.randn(1, 7, hidden)
    out_ref = ref.finish(y, dyn)
    out_ours = conv_finish(y, dyn, base[1], g)
    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)


def _mk_selector(vocab=500, rank=32, k=8, hidden=128, seed=2):
    torch.manual_seed(seed)
    cfg = SimpleNamespace(
        vocab_size=vocab,
        hidden_size=hidden,
        dflash_config={"selector_rank": rank, "selector_top_k": k},
    )
    ref = RefSelector(cfg)
    A = torch.randn(vocab, rank) * 0.05
    B = torch.randn(vocab, rank) * 0.05
    P = torch.randn(rank, hidden) * 0.05
    with torch.no_grad():
        ref.predecessor_codebook.weight.copy_(A)
        ref.successor_codebook.weight.copy_(B)
        ref.hidden_projection.weight.copy_(P)
    return A, B, P, ref, vocab, rank, k, hidden


def _fixed_inputs(vocab, hidden, T=5, seed=3):
    torch.manual_seed(seed)
    h = torch.randn(1, T, hidden)
    logits = torch.randn(1, T, vocab)
    anchor = torch.tensor([7])
    return h, logits, anchor


def test_selector_greedy_parity():
    A, B, P, ref, vocab, rank, k, hidden = _mk_selector()
    h, logits, anchor = _fixed_inputs(vocab, hidden)
    path_ref, cand_ref, q_ref = ref.select(h, logits, anchor, 0.0)
    path_ours, cand_ours, q_ours = selector_select(h, logits, anchor, A, B, P, 0.0, k)
    assert q_ref is None and q_ours is None
    # topk(sorted=False) order can differ between implementations only if scores tie;
    # the walked path must agree
    assert torch.equal(path_ref, path_ours)
    assert torch.equal(cand_ref, cand_ours)


def test_selector_sampled_q_parity():
    A, B, P, ref, vocab, rank, k, hidden = _mk_selector()
    h, logits, anchor = _fixed_inputs(vocab, hidden, seed=4)
    torch.manual_seed(123)
    path_ref, cand_ref, q_ref = ref.select(h, logits, anchor, 0.8)
    torch.manual_seed(123)
    path_ours, cand_ours, q_ours = selector_select(h, logits, anchor, A, B, P, 0.8, k)
    assert q_ref is not None and q_ours is not None
    assert torch.allclose(q_ours, q_ref, atol=1e-6, rtol=1e-6)
    assert torch.equal(path_ref, path_ours)
    assert torch.equal(cand_ref, cand_ours)


def test_selector_sampled_q_is_distribution_over_candidates():
    A, B, P, _, vocab, rank, k, hidden = _mk_selector(seed=5)
    h, logits, anchor = _fixed_inputs(vocab, hidden, seed=6)
    _, candidates, q = selector_select(h, logits, anchor, A, B, P, 1.0, k)
    assert q is not None
    assert torch.allclose(q.sum(-1), torch.ones(1, h.shape[1]), atol=1e-6)
    # scatter q into a vocab-wide tensor at the candidate indices and read it back:
    # if q's support is exactly the candidates, the round trip is lossless
    full = torch.zeros(1, h.shape[1], vocab)
    full.scatter_(-1, candidates, q)
    assert torch.allclose(torch.gather(full, -1, candidates), q, atol=1e-6)
    assert full.sum(-1).allclose(torch.ones(1, h.shape[1]), atol=1e-6)


def test_selector_greedy_returns_no_q():
    A, B, P, _, vocab, rank, k, hidden = _mk_selector(seed=7)
    h, logits, anchor = _fixed_inputs(vocab, hidden, seed=8)
    path, _, q = selector_select(h, logits, anchor, A, B, P, 0.0, k)
    assert q is None
    assert path.shape == (1, h.shape[1])
