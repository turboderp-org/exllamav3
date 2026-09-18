"""
Parity tests for the DFlash2 modules against the reference implementation
(z-lab/dflash, dflash/model.py, MIT). The pure math functions are compared on
identical seeded tensors, so tolerances are tight. Only the reference-parity
tests need the optional dflash package; the oracle, confidence, and Triton
tests run in the default suite.
"""
from types import SimpleNamespace

import pytest
import torch

try:
    from dflash.model import (
        CandidateSelector as RefSelector,
        GroupedDynamicCausalConv as RefConv,
    )
    has_ref = True
except ImportError:
    has_ref = False

requires_ref = pytest.mark.skipif(
    not has_ref,
    reason = "reference dflash package not installed",
)

from exllamav3.modules.arch_specific.dflash2 import (  # noqa: E402
    _grouped_dynamic_convolve,
    conv_finish,
    conv_prepare,
    grouped_dynamic_convolve,
    selector_select,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "CUDA device required",
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


@requires_ref
def test_conv_prepare_parity():
    base, proj_w, ref, hidden, k, g = _mk_conv()
    x = torch.randn(1, 7, hidden)
    out_ref, dyn_ref = ref.prepare(x)
    out_ours, dyn_ours = conv_prepare(x, base[0], proj_w, k, g)
    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dyn_ours, dyn_ref, atol=1e-5, rtol=1e-5)


@requires_ref
def test_conv_finish_parity():
    base, proj_w, ref, hidden, k, g = _mk_conv(seed=1)
    x = torch.randn(1, 7, hidden)
    _, dyn = ref.prepare(x)
    y = torch.randn(1, 7, hidden)
    out_ref = ref.finish(y, dyn)
    out_ours = conv_finish(y, dyn, base[1], g)
    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)


def _mk_codebooks(vocab=500, rank=32, k=8, hidden=128, seed=2):
    torch.manual_seed(seed)
    A = torch.randn(vocab, rank) * 0.05
    B = torch.randn(vocab, rank) * 0.05
    P = torch.randn(rank, hidden) * 0.05
    return A, B, P, vocab, rank, k, hidden


def _mk_selector(vocab=500, rank=32, k=8, hidden=128, seed=2):
    A, B, P, vocab, rank, k, hidden = _mk_codebooks(vocab, rank, k, hidden, seed)
    cfg = SimpleNamespace(
        vocab_size=vocab,
        hidden_size=hidden,
        dflash_config={"selector_rank": rank, "selector_top_k": k},
    )
    ref = RefSelector(cfg)
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


@requires_ref
def test_selector_greedy_parity():
    A, B, P, ref, vocab, rank, k, hidden = _mk_selector()
    h, logits, anchor = _fixed_inputs(vocab, hidden)
    path_ref, cand_ref, q_ref = ref.select(h, logits, anchor, 0.0)
    path_ours, cand_ours, q_ours, _ = selector_select(h, logits, anchor, A, B, P, 0.0, k)
    assert q_ref is None and q_ours is None
    # topk(sorted=False) order can differ between implementations only if scores tie;
    # the walked path must agree
    assert torch.equal(path_ref, path_ours)
    assert torch.equal(cand_ref, cand_ours)


@requires_ref
def test_selector_sampled_q_parity():
    A, B, P, ref, vocab, rank, k, hidden = _mk_selector()
    h, logits, anchor = _fixed_inputs(vocab, hidden, seed=4)
    torch.manual_seed(123)
    path_ref, cand_ref, q_ref = ref.select(h, logits, anchor, 0.8)
    torch.manual_seed(123)
    path_ours, cand_ours, q_ours, _ = selector_select(h, logits, anchor, A, B, P, 0.8, k)
    assert q_ref is not None and q_ours is not None
    assert torch.allclose(q_ours, q_ref, atol=1e-6, rtol=1e-6)
    assert torch.equal(path_ref, path_ours)
    assert torch.equal(cand_ref, cand_ours)


def test_selector_sampled_q_is_distribution_over_candidates():
    A, B, P, vocab, rank, k, hidden = _mk_codebooks(seed=5)
    h, logits, anchor = _fixed_inputs(vocab, hidden, seed=6)
    _, candidates, q, _ = selector_select(h, logits, anchor, A, B, P, 1.0, k)
    assert q is not None
    assert torch.allclose(q.sum(-1), torch.ones(1, h.shape[1]), atol=1e-6)
    # scatter q into a vocab-wide tensor at the candidate indices and read it back:
    # if q's support is exactly the candidates, the round trip is lossless
    full = torch.zeros(1, h.shape[1], vocab)
    full.scatter_(-1, candidates, q)
    assert torch.allclose(torch.gather(full, -1, candidates), q, atol=1e-6)
    assert full.sum(-1).allclose(torch.ones(1, h.shape[1]), atol=1e-6)


def test_selector_greedy_returns_no_q():
    A, B, P, vocab, rank, k, hidden = _mk_codebooks(seed=7)
    h, logits, anchor = _fixed_inputs(vocab, hidden, seed=8)
    path, _, q, _ = selector_select(h, logits, anchor, A, B, P, 0.0, k)
    assert q is None
    assert path.shape == (1, h.shape[1])


def test_selector_confidence_default_is_none():
    A, B, P, vocab, rank, k, hidden = _mk_codebooks(seed=9)
    h, logits, anchor = _fixed_inputs(vocab, hidden, seed=10)
    _, _, _, conf = selector_select(h, logits, anchor, A, B, P, 0.0, k)
    assert conf is None


def test_selector_confidence_is_chosen_transition_score():
    # return_confidence=True must report, per position, the walked transition's
    # score (unary + bilinear edge), i.e. the max over the candidate list on the
    # greedy walk. propose() exports these as draft_conf for the -dds/-dc crop.
    A, B, P, vocab, rank, k, hidden = _mk_codebooks(seed=11)
    h, logits, anchor = _fixed_inputs(vocab, hidden, seed=12)
    path, cands, _, conf = selector_select(h, logits, anchor, A, B, P, 0.0, k, True)
    assert conf is not None
    assert conf.shape == (1, h.shape[1])
    import torch.nn.functional as Fn
    unary, _ = torch.topk(logits, k, dim = -1, sorted = False)
    hp = Fn.linear(h, P).to(A.dtype)
    pred = anchor.to(A.device)
    for pos in range(h.shape[1]):
        scores = unary[:, pos] + torch.einsum(
            "br,bkr->bk",
            Fn.embedding(pred, A) * hp[:, pos],
            Fn.embedding(cands[:, pos], B),
        )
        best = scores.max(dim = -1).values
        assert torch.allclose(conf[:, pos].float(), best.float(), atol = 1e-4, rtol = 1e-4)
        pred = path[:, pos]


# ---------------------------------------------------------------------------
# Triton serve path vs the eager oracle (CUDA only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch,length,hidden,kernel,group", [
    (1, 7, 256, 2, 8),
    (2, 8, 512, 2, 16),
    (1, 16, 1024, 3, 32),
    (3, 5, 128, 2, 4),
])
@requires_cuda
def test_triton_conv_matches_eager(batch, length, hidden, kernel, group):
    torch.manual_seed(0)
    groups = hidden // group
    dev = "cuda"
    h = torch.randn(batch, length, hidden, device = dev, dtype = torch.float16)
    dyn = torch.randn(batch, length, kernel, groups, device = dev, dtype = torch.float16)
    base = torch.randn(kernel, hidden, device = dev, dtype = torch.float16) * 0.05
    ref = grouped_dynamic_convolve(h, dyn, base, group)
    out = _grouped_dynamic_convolve(h, dyn, base, group)
    assert out.dtype == h.dtype
    # fp16 accumulation: the Triton kernel and the eager oracle agree to ~1e-3 rel
    assert torch.allclose(out.float(), ref.float(), atol = 2e-2, rtol = 2e-3)


@requires_cuda
def test_triton_conv_serve_dtypes_and_view():
    # Serve-realistic inputs: prepare() passes fp16 hidden with an fp32 dynamic
    # *view* sliced from the projection output (non-contiguous); finish() passes
    # fp32 hidden with fp32 dynamic. Both must match the eager oracle.
    torch.manual_seed(0)
    dev = "cuda"
    batch, length, hidden, kernel, group = 1, 8, 512, 2, 16
    groups = hidden // group
    base = torch.randn(kernel, hidden, device = dev, dtype = torch.float16) * 0.05

    h16 = torch.randn(batch, length, hidden, device = dev, dtype = torch.float16)
    dyn_full = torch.randn(batch, length, 2, kernel, groups, device = dev, dtype = torch.float32)
    dyn_view = dyn_full[..., 0, :, :]
    assert not dyn_view.is_contiguous()
    ref = grouped_dynamic_convolve(h16, dyn_view, base, group)
    out = _grouped_dynamic_convolve(h16, dyn_view, base, group)
    assert out.dtype == h16.dtype
    assert torch.allclose(out.float(), ref.float(), atol = 2e-2, rtol = 2e-3)

    h32 = torch.randn(batch, length, hidden, device = dev, dtype = torch.float32)
    dyn32 = torch.randn(batch, length, kernel, groups, device = dev, dtype = torch.float32)
    ref32 = grouped_dynamic_convolve(h32, dyn32, base, group)
    out32 = _grouped_dynamic_convolve(h32, dyn32, base, group)
    assert out32.dtype == torch.float32
    assert torch.allclose(out32, ref32, atol = 1e-5, rtol = 1e-5)
