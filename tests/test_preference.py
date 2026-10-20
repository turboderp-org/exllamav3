"""
CPU tests for preference training (DPO / KTO / SimPO) support:

  * the fused CE heads' ``reduction="none"`` per-token mode (the streaming
    building block for per-sequence logprobs) -- parity vs
    ``F.cross_entropy(reduction='none')``, vector-grad_output backward parity,
    gradcheck, and single-shot vs vocab-chunked agreement;
  * per-row completion logprob sums (the ``compute_logps`` core);
  * ``dpo_loss`` / ``kto_loss`` / ``simpo_loss`` formulas vs hand-written
    references (TRL semantics: sigmoid + label smoothing, hinge, ipo; KTO KL
    clamp, weights, apo_zero_unpaired; SimPO gamma margin, length
    normalization/invariance) and the mismatched-pair KL rotation;
  * ``DiffLinear.adapter_enabled`` -- the reference-model view is the PURE
    frozen base (LoRA term AND pissa offset off) -- plus the
    ``adapters_disabled()`` context manager's flag handling.

No GPU / compiled extension / real model needed. Run:
    python tests/test_preference.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"exl3train.{name}", os.path.join(_TRAIN_DIR, f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = m
    spec.loader.exec_module(m)
    return m


_qll = _load("qlora_linear")
_fce = _load("fused_ce")
_nl = _load("native_llama")
_pref = _load("preference")

FusedLinearCrossEntropy = _fce.FusedLinearCrossEntropy
FusedLinearCrossEntropyVocabChunked = _fce.FusedLinearCrossEntropyVocabChunked
fused_linear_cross_entropy = _fce.fused_linear_cross_entropy
fused_linear_cross_entropy_vocab_chunked = _fce.fused_linear_cross_entropy_vocab_chunked
IGNORE_INDEX = _fce.IGNORE_INDEX
DiffLinear = _nl.DiffLinear
NativeLlamaQLoRA = _nl.NativeLlamaQLoRA
dpo_loss = _pref.dpo_loss
kto_loss = _pref.kto_loss
simpo_loss = _pref.simpo_loss
orpo_loss = _pref.orpo_loss
log1mexp = _pref.log1mexp
mismatched_kl_shift = _pref.mismatched_kl_shift


def _slice_fn(weight):
    def fn(start, n):
        return weight[:, start:start + n]
    return fn


# ----------------------------------------------------------------------------
# reduction="none" on the fused heads
# ----------------------------------------------------------------------------

def test_reduction_none_parity():
    """Per-token NLL from both fused heads must match
    F.cross_entropy(reduction='none') exactly (zeros at ignored positions), and
    a weighted sum of the per-token outputs must backprop identically."""
    torch.manual_seed(0)
    n, d, v = 40, 16, 50
    weight = torch.randn(d, v, dtype=torch.float64)
    labels = torch.randint(0, v, (n,))
    labels[::4] = IGNORE_INDEX
    coef = torch.randn(n, dtype=torch.float64)   # arbitrary per-token weights

    h_ref = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    logits = h_ref @ weight
    nll_ref = F.cross_entropy(logits, labels, reduction="none",
                              ignore_index=IGNORE_INDEX)
    (nll_ref * coef).sum().backward()

    for tag, run in (
        ("single-shot", lambda h: FusedLinearCrossEntropy.apply(
            h, labels, lambda: weight, 7, IGNORE_INDEX, 0.0, "none")),
        ("vocab-chunked", lambda h: FusedLinearCrossEntropyVocabChunked.apply(
            h, labels, _slice_fn(weight), v, 16, 7, IGNORE_INDEX, 1, 0.0, "none")),
    ):
        h = h_ref.detach().clone().requires_grad_(True)
        nll = run(h)
        assert nll.shape == (n,), f"{tag}: wrong shape {nll.shape}"
        assert torch.allclose(nll, nll_ref, atol=1e-10), f"{tag}: per-token NLL mismatch"
        assert torch.all(nll[::4] == 0), f"{tag}: ignored tokens not zero"
        (nll * coef).sum().backward()
        assert torch.allclose(h.grad, h_ref.grad, atol=1e-9), \
            f"{tag}: vector-grad backward mismatch"
    print("[pref] reduction='none' parity (both heads) PASSED")


def test_reduction_none_softcap_parity():
    """Per-token mode composes with the Gemma tanh softcap on both heads."""
    torch.manual_seed(1)
    n, d, v, cap = 24, 12, 30, 10.0
    weight = torch.randn(d, v, dtype=torch.float64) * 3
    labels = torch.randint(0, v, (n,))
    labels[::5] = IGNORE_INDEX
    coef = torch.randn(n, dtype=torch.float64)

    h_ref = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    logits = cap * torch.tanh((h_ref @ weight) / cap)
    nll_ref = F.cross_entropy(logits, labels, reduction="none",
                              ignore_index=IGNORE_INDEX)
    (nll_ref * coef).sum().backward()

    for tag, run in (
        ("single-shot", lambda h: FusedLinearCrossEntropy.apply(
            h, labels, lambda: weight, 5, IGNORE_INDEX, cap, "none")),
        ("vocab-chunked", lambda h: FusedLinearCrossEntropyVocabChunked.apply(
            h, labels, _slice_fn(weight), v, 8, 5, IGNORE_INDEX, 1, cap, "none")),
    ):
        h = h_ref.detach().clone().requires_grad_(True)
        nll = run(h)
        assert torch.allclose(nll, nll_ref, atol=1e-10), f"{tag}: softcap NLL mismatch"
        (nll * coef).sum().backward()
        assert torch.allclose(h.grad, h_ref.grad, atol=1e-9), \
            f"{tag}: softcap vector-grad mismatch"
    print("[pref] reduction='none' softcap parity PASSED")


def test_reduction_none_gradcheck():
    torch.manual_seed(2)
    n, d, v = 10, 8, 20
    weight = torch.randn(d, v, dtype=torch.float64)
    labels = torch.randint(0, v, (n,))
    labels[3] = IGNORE_INDEX
    h = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda h_: FusedLinearCrossEntropy.apply(
            h_, labels, lambda: weight, 4, IGNORE_INDEX, 0.0, "none"),
        (h,), eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    assert ok
    ok = torch.autograd.gradcheck(
        lambda h_: FusedLinearCrossEntropyVocabChunked.apply(
            h_, labels, _slice_fn(weight), v, 8, 4, IGNORE_INDEX, 1, 0.0, "none"),
        (h,), eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    assert ok
    print("[pref] reduction='none' gradcheck (both heads) PASSED")


def test_row_logps_sums():
    """The compute_logps core: -sum of shifted per-token NLL per batch row must
    equal a hand-built per-row completion logprob (prompt masked -100)."""
    torch.manual_seed(3)
    b, t, d, v = 3, 12, 8, 25
    weight = torch.randn(d, v, dtype=torch.float64)
    hidden = torch.randn(b, t, d, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, v, (b, t))
    labels[:, :5] = IGNORE_INDEX            # "prompt" span
    labels[2, 5:9] = IGNORE_INDEX           # ragged supervision

    nll = fused_linear_cross_entropy(hidden, lambda: weight, labels,
                                     chunk=4, shift=True, reduction="none")
    logps = -nll.view(b, t - 1).sum(dim=-1)

    # Reference: full log_softmax gather over the supervised shifted positions.
    logp = F.log_softmax(hidden[:, :-1] @ weight, dim=-1)      # [b, t-1, v]
    lbl = labels[:, 1:]
    ref = torch.zeros(b, dtype=torch.float64)
    for i in range(b):
        for j in range(t - 1):
            if lbl[i, j] != IGNORE_INDEX:
                ref[i] += logp[i, j, lbl[i, j]]
    assert torch.allclose(logps, ref, atol=1e-10), "per-row logps mismatch"

    counts = (labels[:, 1:] != IGNORE_INDEX).sum(dim=-1)
    assert counts.tolist() == [7, 7, 3]
    logps.sum().backward()   # graph reaches hidden
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    print("[pref] per-row completion logps PASSED")


# ----------------------------------------------------------------------------
# DPO loss
# ----------------------------------------------------------------------------

def test_dpo_sigmoid():
    torch.manual_seed(4)
    b, beta = 8, 0.15
    pc, pr = torch.randn(b), torch.randn(b)
    rc, rr = torch.randn(b), torch.randn(b)

    losses, cr, rj = dpo_loss(pc, pr, rc, rr, beta=beta)
    delta = (pc - rc) - (pr - rr)
    ref = -F.logsigmoid(beta * delta)
    assert torch.allclose(losses, ref, atol=1e-7), "sigmoid DPO mismatch"
    assert torch.allclose(cr, beta * (pc - rc)) and torch.allclose(rj, beta * (pr - rr))
    assert not cr.requires_grad and not rj.requires_grad

    # Label smoothing (cDPO): epsilon-weighted flipped term.
    eps = 0.1
    losses_ls, _, _ = dpo_loss(pc, pr, rc, rr, beta=beta, label_smoothing=eps)
    ref_ls = -(1 - eps) * F.logsigmoid(beta * delta) - eps * F.logsigmoid(-beta * delta)
    assert torch.allclose(losses_ls, ref_ls, atol=1e-7), "cDPO mismatch"

    # At policy == reference the loss is exactly log 2 (the step-0 anchor).
    l0, _, _ = dpo_loss(rc, rr, rc, rr, beta=beta)
    assert torch.allclose(l0, torch.full_like(l0, float(torch.log(torch.tensor(2.0)))))
    print("[pref] DPO sigmoid + label smoothing PASSED")


def test_dpo_hinge_ipo():
    torch.manual_seed(5)
    b, beta = 6, 0.2
    pc, pr = torch.randn(b), torch.randn(b)
    rc, rr = torch.randn(b), torch.randn(b)
    nc = torch.randint(3, 30, (b,))
    nr = torch.randint(3, 30, (b,))

    delta = (pc - rc) - (pr - rr)
    losses, _, _ = dpo_loss(pc, pr, rc, rr, beta=beta, loss_type="hinge")
    assert torch.allclose(losses, torch.relu(1 - beta * delta), atol=1e-7)

    losses, _, _ = dpo_loss(pc, pr, rc, rr, beta=beta, loss_type="ipo",
                            chosen_counts=nc, rejected_counts=nr)
    davg = (pc - rc) / nc - (pr - rr) / nr
    assert torch.allclose(losses, (davg - 1 / (2 * beta)) ** 2, atol=1e-7)

    try:
        dpo_loss(pc, pr, rc, rr, loss_type="ipo")
        assert False, "ipo without counts should raise"
    except ValueError:
        pass
    try:
        dpo_loss(pc, pr, rc, rr, loss_type="nope")
        assert False, "unknown loss_type should raise"
    except ValueError:
        pass
    print("[pref] DPO hinge + ipo PASSED")


def test_dpo_gradient_direction():
    """Sanity: the sigmoid DPO gradient pushes chosen logps UP and rejected
    logps DOWN (the whole point of the objective)."""
    pc = torch.zeros(4, requires_grad=True)
    pr = torch.zeros(4, requires_grad=True)
    rc = torch.zeros(4)
    rr = torch.zeros(4)
    losses, _, _ = dpo_loss(pc, pr, rc, rr, beta=0.1)
    losses.mean().backward()
    assert torch.all(pc.grad < 0), "chosen grad should be negative (ascent on logp)"
    assert torch.all(pr.grad > 0), "rejected grad should be positive"
    print("[pref] DPO gradient direction PASSED")


# ----------------------------------------------------------------------------
# SimPO loss
# ----------------------------------------------------------------------------

def test_simpo_formula():
    """simpo_loss vs the hand-written TRL CPOTrainer(loss_type='simpo') form:
    -log sigmoid(beta * (avg_c - avg_r) - gamma), rewards = beta * avg logp
    (detached), cDPO-style label smoothing."""
    torch.manual_seed(10)
    b, beta, gamma = 8, 2.0, 0.7
    nc = torch.randint(3, 30, (b,))
    nr = torch.randint(3, 30, (b,))
    pc = -torch.rand(b) * nc          # plausible summed logps (negative)
    pr = -torch.rand(b) * nr

    losses, cr, rr = simpo_loss(pc, pr, nc, nr, beta=beta, gamma=gamma)
    avg_c, avg_r = pc / nc, pr / nr
    logits = beta * (avg_c - avg_r) - gamma
    assert torch.allclose(losses, -F.logsigmoid(logits), atol=1e-7), \
        "SimPO loss mismatch"
    assert torch.allclose(cr, beta * avg_c) and torch.allclose(rr, beta * avg_r)
    assert not cr.requires_grad and not rr.requires_grad

    # Label smoothing: epsilon-weighted flipped term.
    eps = 0.1
    losses_ls, _, _ = simpo_loss(pc, pr, nc, nr, beta=beta, gamma=gamma,
                                 label_smoothing=eps)
    ref_ls = -(1 - eps) * F.logsigmoid(logits) - eps * F.logsigmoid(-logits)
    assert torch.allclose(losses_ls, ref_ls, atol=1e-7), "smoothed SimPO mismatch"

    # Zero avg-logp margin -> loss is exactly -log sigmoid(-gamma) (the only
    # fixed anchor SimPO has; note it is NOT ln 2 unless gamma == 0).
    l0, _, _ = simpo_loss(pc, pc, nc, nc, beta=beta, gamma=gamma)
    anchor = -F.logsigmoid(torch.tensor(-gamma))
    assert torch.allclose(l0, anchor.expand_as(l0), atol=1e-7), "gamma anchor"
    lg0, _, _ = simpo_loss(pc, pc, nc, nc, beta=beta, gamma=0.0)
    ln2 = float(torch.log(torch.tensor(2.0)))
    assert torch.allclose(lg0, torch.full_like(lg0, ln2), atol=1e-7)

    # Zero counts must clamp, not divide by zero.
    z = torch.zeros(b, dtype=torch.long)
    lz, _, _ = simpo_loss(pc, pr, z, z, beta=beta, gamma=gamma)
    assert torch.isfinite(lz).all(), "zero-count clamp failed"
    print("[pref] SimPO formula + smoothing + gamma anchor PASSED")


def test_simpo_length_invariance():
    """The whole point of SimPO's normalization: scaling a completion's summed
    logp and token count by the same factor leaves the loss unchanged (a
    longer completion with the same per-token quality scores the same)."""
    torch.manual_seed(11)
    b = 6
    nc = torch.randint(3, 30, (b,))
    nr = torch.randint(3, 30, (b,))
    pc, pr = -torch.rand(b) * nc, -torch.rand(b) * nr
    l1, _, _ = simpo_loss(pc, pr, nc, nr, beta=2.0, gamma=0.5)
    l2, _, _ = simpo_loss(3 * pc, 5 * pr, 3 * nc, 5 * nr, beta=2.0, gamma=0.5)
    assert torch.allclose(l1, l2, atol=1e-6), "length invariance broken"
    print("[pref] SimPO length invariance PASSED")


def test_simpo_gradient_direction():
    """The gradient pushes chosen logps UP and rejected logps DOWN; no
    reference tensors are involved anywhere in the graph."""
    pc = torch.zeros(4, requires_grad=True)
    pr = torch.zeros(4, requires_grad=True)
    counts = torch.full((4,), 10)
    losses, _, _ = simpo_loss(pc, pr, counts, counts, beta=2.0, gamma=0.5)
    losses.mean().backward()
    assert torch.all(pc.grad < 0), "chosen grad should be negative (ascent on logp)"
    assert torch.all(pr.grad > 0), "rejected grad should be positive"
    print("[pref] SimPO gradient direction PASSED")


# ----------------------------------------------------------------------------
# ORPO loss
# ----------------------------------------------------------------------------

def test_log1mexp():
    """log(1 - exp(x)) on both branches vs float64 direct evaluation; finite
    with a finite gradient at x = 0 (p = 1 exactly)."""
    x = -torch.logspace(-6, 2, 200, dtype=torch.float64)      # -1e-6 .. -100
    ref = torch.log1p(-torch.exp(x))
    assert torch.allclose(log1mexp(x), ref, rtol=1e-9, atol=0), "log1mexp"
    # fp32 near zero: the naive form loses everything, the stable one doesn't
    x32 = torch.tensor([-1e-5], dtype=torch.float32)
    assert abs(log1mexp(x32).item() - float(torch.log1p(-torch.exp(x32.double())))) < 1e-5
    z = torch.zeros(1, requires_grad=True)
    y = log1mexp(z)
    y.backward()
    assert torch.isfinite(y).all() and torch.isfinite(z.grad).all()
    print("[pref] log1mexp PASSED")


def test_orpo_formula():
    """orpo_loss vs the hand-written TRL ORPOTrainer.odds_ratio_loss form on
    length-normalized logps: log_odds = (lc - lr) - (log(1-e^lc) -
    log(1-e^lr)), loss = -beta * log sigmoid(log_odds), rewards = beta *
    mean logp (detached)."""
    torch.manual_seed(12)
    b, beta = 8, 0.1
    nc = torch.randint(3, 30, (b,))
    nr = torch.randint(3, 30, (b,))
    pc = -torch.rand(b) * nc * 2          # summed logps (negative)
    pr = -torch.rand(b) * nr * 2
    losses, cr, rr, lo = orpo_loss(pc, pr, nc, nr, beta=beta)
    lc, lr = pc / nc, pr / nr
    log_odds = (lc - lr) - (torch.log1p(-torch.exp(lc)) - torch.log1p(-torch.exp(lr)))
    assert torch.allclose(lo, log_odds, atol=1e-6), "log odds mismatch"
    assert torch.allclose(losses, -beta * F.logsigmoid(log_odds), atol=1e-7), \
        "ORPO loss mismatch"
    assert torch.allclose(cr, beta * lc) and torch.allclose(rr, beta * lr)
    assert not cr.requires_grad and not rr.requires_grad and not lo.requires_grad
    # Equal mean logps -> log odds 0 -> loss exactly beta * ln 2.
    l0, _, _, lo0 = orpo_loss(pc, pc, nc, nc, beta=beta)
    ln2 = float(torch.log(torch.tensor(2.0)))
    assert torch.allclose(l0, torch.full_like(l0, beta * ln2), atol=1e-7)
    assert torch.allclose(lo0, torch.zeros_like(lo0), atol=1e-7)
    # Odds, not probabilities: the same logp gap counts for more at high p.
    # lc - lr = 0.1 at p ~ 0.9 vs p ~ 0.05 -> larger log odds ratio.
    hi, _, _, lo_hi = orpo_loss(torch.tensor([-0.1]), torch.tensor([-0.2]),
                                torch.ones(1), torch.ones(1), beta=beta)
    low, _, _, lo_low = orpo_loss(torch.tensor([-3.0]), torch.tensor([-3.1]),
                                  torch.ones(1), torch.ones(1), beta=beta)
    assert lo_hi > lo_low and hi < low
    # Zero counts clamp, not divide by zero.
    z = torch.zeros(b, dtype=torch.long)
    lz, _, _, _ = orpo_loss(pc, pr, z, z, beta=beta)
    assert torch.isfinite(lz).all()
    print("[pref] ORPO formula + anchors PASSED")


def test_orpo_gradient_direction():
    """The odds-ratio term pushes chosen logps UP and rejected DOWN (no
    reference tensors anywhere). Its gradient carries the paper's fade factor
    sigma(-log_odds) = 1 / (1 + odds_w/odds_l) (sec. 4.3): with rejected held
    fixed, the push on rejected shrinks as chosen pulls ahead. (The chosen
    side also carries d log_odds / d logp_w = 1 / (1 - P_w), which GROWS with
    P_w, so no such monotone statement holds there.)"""
    pc = torch.full((4,), -5.0, requires_grad=True)
    pr = torch.full((4,), -5.0, requires_grad=True)
    counts = torch.full((4,), 10)
    losses, _, _, _ = orpo_loss(pc, pr, counts, counts, beta=0.1)
    losses.mean().backward()
    assert torch.all(pc.grad < 0) and torch.all(pr.grad > 0)
    g_even = pr.grad.clone()
    pr2 = torch.full((4,), -5.0, requires_grad=True)
    pc2 = torch.full((4,), -2.0)                        # chosen well ahead
    losses, _, _, _ = orpo_loss(pc2, pr2, counts, counts, beta=0.1)
    losses.mean().backward()
    assert torch.all(pr2.grad > 0) and torch.all(pr2.grad < g_even), \
        "rejected-side push should fade as chosen pulls ahead"
    print("[pref] ORPO gradient direction PASSED")


# ----------------------------------------------------------------------------
# KTO loss
# ----------------------------------------------------------------------------

def test_kto_formula():
    torch.manual_seed(6)
    nd, nu, nk, beta = 5, 4, 9, 0.1
    pc, pu = torch.randn(nd), torch.randn(nu)
    rc, ru = torch.randn(nd), torch.randn(nu)
    pkl, rkl = torch.randn(nk) + 0.5, torch.randn(nk)   # positive-mean KL

    losses, cr, rr, kl = kto_loss(pc, pu, pkl, rc, ru, rkl, beta=beta,
                                  desirable_weight=1.5, undesirable_weight=0.75)
    kl_ref = (pkl - rkl).mean().clamp(min=0)
    assert torch.allclose(kl, kl_ref, atol=1e-7), "KL estimate mismatch"
    ref_c = 1.5 * (1 - torch.sigmoid(beta * ((pc - rc) - kl_ref)))
    ref_u = 0.75 * (1 - torch.sigmoid(beta * (kl_ref - (pu - ru))))
    assert torch.allclose(losses, torch.cat([ref_c, ref_u]), atol=1e-7), \
        "KTO losses mismatch"
    assert torch.allclose(cr, beta * (pc - rc)) and torch.allclose(rr, beta * (pu - ru))

    # Negative-mean KL must clamp to exactly 0.
    _, _, _, kl0 = kto_loss(pc, pu, rkl - 5.0, rc, ru, rkl, beta=beta)
    assert kl0.item() == 0.0, "KL clamp failed"

    # No KL batch (None) -> KL term 0.
    _, _, _, kln = kto_loss(pc, pu, None, rc, ru, None, beta=beta)
    assert kln.item() == 0.0
    print("[pref] KTO formula + KL clamp + weights PASSED")


def test_kto_apo_and_empty_subsets():
    torch.manual_seed(7)
    beta = 0.2
    pc, rc = torch.randn(4), torch.randn(4)
    pu, ru = torch.randn(3), torch.randn(3)
    pkl, rkl = torch.randn(5), torch.randn(5)

    losses, _, _, kl = kto_loss(pc, pu, pkl, rc, ru, rkl, beta=beta,
                                loss_type="apo_zero_unpaired")
    assert kl.item() == 0.0, "apo variant must not use the KL term"
    ref = torch.cat([1 - torch.sigmoid(beta * (pc - rc)),
                     torch.sigmoid(beta * (pu - ru))])
    assert torch.allclose(losses, ref, atol=1e-7), "apo_zero_unpaired mismatch"

    # A micro-batch with only desirable (or only undesirable) rows must work.
    empty = torch.zeros(0)
    losses, cr, rr, _ = kto_loss(pc, empty, pkl, rc, empty, rkl, beta=beta)
    assert losses.shape == (4,) and rr.numel() == 0
    losses, cr, rr, _ = kto_loss(empty, pu, pkl, empty, ru, rkl, beta=beta)
    assert losses.shape == (3,) and cr.numel() == 0

    try:
        kto_loss(pc, pu, pkl, rc, ru, rkl, loss_type="nope")
        assert False, "unknown loss_type should raise"
    except ValueError:
        pass
    print("[pref] KTO apo variant + empty subsets PASSED")


def test_mismatched_kl_shift():
    assert mismatched_kl_shift(0) == []
    assert mismatched_kl_shift(1) == [0]
    for n in (2, 3, 7):
        s = mismatched_kl_shift(n)
        assert sorted(s) == list(range(n)), "not a permutation"
        assert all(s[i] != i for i in range(n)), "an index maps to itself"
        assert s == [n - 1] + list(range(n - 1)), "not TRL's +1 rotation"
    print("[pref] mismatched KL shift PASSED")


# ----------------------------------------------------------------------------
# Adapter disable (the reference-model view)
# ----------------------------------------------------------------------------

class _MockInner:
    def __init__(self, weight):
        self._w = weight
        self.trellis = weight
        self.bias = None

    def get_weight_tensor(self):
        return self._w

    def get_bias_tensor(self):
        return None


class MockLinear(nn.Module):
    def __init__(self, in_features, out_features, key, scale=0.05,
                 dtype=torch.float64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.key = key
        self.device = torch.device("cpu")
        w = torch.randn(in_features, out_features, dtype=dtype) * scale
        self.register_buffer("frozen_weight", w)
        self.inner = _MockInner(self.frozen_weight)
        self.lora_a_tensors = {}
        self.lora_b_tensors = {}


def test_diff_linear_adapter_disable():
    torch.manual_seed(8)
    lin = MockLinear(16, 12, "model.layers.0.self_attn.q_proj")
    w = DiffLinear(lin, r=4, alpha=4.0, compute_dtype=torch.float64)
    with torch.no_grad():
        w.lora_b.normal_()          # make the adapter a real delta
    x = torch.randn(5, 16, dtype=torch.float64)
    base = x @ lin.frozen_weight

    y_on = w(x)
    assert not torch.allclose(y_on, base, atol=1e-9), "adapter should change output"
    w.adapter_enabled = False
    assert torch.allclose(w(x), base, atol=1e-12), "disabled != pure base"
    w.adapter_enabled = True
    assert torch.allclose(w(x), y_on, atol=1e-12), "re-enable didn't restore"

    # With a pissa offset installed: enabled = residual base + adapter;
    # disabled must STILL be the pure base (offset dropped with the adapter).
    a0 = torch.randn(16, 4, dtype=torch.float64) * 0.1
    b0 = torch.randn(4, 12, dtype=torch.float64) * 0.1
    w.set_init_offset(a0, b0)
    y_pissa = w(x)
    # lora_a/b are fp32 master weights (the Function casts them to x's dtype);
    # mirror that cast in the reference.
    la = w.lora_a.detach().to(torch.float64)
    lb = w.lora_b.detach().to(torch.float64)
    ref_pissa = (x @ (lin.frozen_weight - w.scale * (a0 @ b0))
                 + w.scale * ((x @ la) @ lb))
    assert torch.allclose(y_pissa, ref_pissa, atol=1e-9), "pissa-enabled mismatch"
    w.adapter_enabled = False
    assert torch.allclose(w(x), base, atol=1e-12), \
        "disabled with pissa offset != pure base"
    print("[pref] DiffLinear adapter disable (incl. pissa offset) PASSED")


def test_adapters_disabled_context():
    torch.manual_seed(9)
    net = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)   # headless (no __init__)
    nn.Module.__init__(net)
    w1 = DiffLinear(MockLinear(8, 8, "l.0.q_proj"), r=2, alpha=2.0,
                    compute_dtype=torch.float64)
    w2 = DiffLinear(MockLinear(8, 8, "l.0.v_proj"), r=0, alpha=2.0,
                    compute_dtype=torch.float64)
    net._wrappers = [w1, w2]

    assert not net._adapters_off and w1.adapter_enabled
    with net.adapters_disabled():
        assert net._adapters_off
        assert not w1.adapter_enabled and not w2.adapter_enabled
        # Reentrant: nested enter/exit keeps the outer state.
        with net.adapters_disabled():
            assert net._adapters_off
        assert net._adapters_off and not w1.adapter_enabled
    assert not net._adapters_off and w1.adapter_enabled and w2.adapter_enabled

    # Exception-safe restore.
    try:
        with net.adapters_disabled():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not net._adapters_off and w1.adapter_enabled
    print("[pref] adapters_disabled context manager PASSED")


def main():
    from util import run_timed
    run_timed([
        test_reduction_none_parity,
        test_reduction_none_softcap_parity,
        test_reduction_none_gradcheck,
        test_row_logps_sums,
        test_dpo_sigmoid,
        test_dpo_hinge_ipo,
        test_dpo_gradient_direction,
        test_simpo_formula,
        test_simpo_length_invariance,
        test_simpo_gradient_direction,
        test_kto_formula,
        test_kto_apo_and_empty_subsets,
        test_mismatched_kl_shift,
        test_diff_linear_adapter_disable,
        test_adapters_disabled_context,
    ], "preference")
    print("\nAll preference-training checks passed.")


if __name__ == "__main__":
    main()
