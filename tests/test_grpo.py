"""
CPU tests for the GRPO math (exllamav3/training/grpo.py) and the built-in
rewards (rewards.py): advantages, the clipped surrogate and its reductions,
the k3 KL term, the reward plug-in loader, and the exact stop-token sampler
on a stub net. No model, no CUDA.

Run:  python -m pytest -q tests/test_grpo.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

if "exl3train" not in sys.modules:
    _pkg = types.ModuleType("exl3train")
    _pkg.__path__ = [_TRAIN_DIR]
    sys.modules["exl3train"] = _pkg


def _load(name: str):
    full = f"exl3train.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(_TRAIN_DIR, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[full] = m
    spec.loader.exec_module(m)
    return m


rewards_mod = _load("rewards")
grpo = _load("grpo")

group_advantages = grpo.group_advantages
combine_rewards = grpo.combine_rewards
grpo_loss = grpo.grpo_loss
load_reward_func = grpo.load_reward_func
call_reward_funcs = grpo.call_reward_funcs
sample_completions_exact = grpo.sample_completions_exact


# ----------------------------------------------------------------------------
# advantages
# ----------------------------------------------------------------------------

def test_group_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0,   # group 0: mean .5, std sqrt(1/3)
                      2.0, 2.0, 2.0, 2.0])  # group 1: all equal -> zero std
    adv, info = group_advantages(r, 4, scale="group")
    s = math.sqrt(1.0 / 3.0)
    exp0 = torch.tensor([0.5, -0.5, -0.5, 0.5]) / (s + 1e-4)
    assert torch.allclose(adv[:4], exp0, atol=1e-6)
    assert torch.all(adv[4:] == 0)
    assert abs(info["frac_zero_std"] - 0.5) < 1e-9
    assert abs(info["reward_mean"] - 1.25) < 1e-9
    # scale none: plain centering; batch: one std for everyone
    adv_n, _ = group_advantages(r, 4, scale="none")
    assert torch.allclose(adv_n[:4], torch.tensor([0.5, -0.5, -0.5, 0.5]))
    adv_b, _ = group_advantages(r, 4, scale="batch")
    bstd = r.std(unbiased=True)
    assert torch.allclose(adv_b[:4], torch.tensor([0.5, -0.5, -0.5, 0.5]) / (bstd + 1e-4))
    # NaN = abstained reward: left out of the stats, advantage 0
    rn = torch.tensor([1.0, float("nan"), 0.0, 1.0])
    adv_nan, info = group_advantages(rn, 4)
    m = 2.0 / 3.0
    std = math.sqrt(((1 - m) ** 2 + (0 - m) ** 2 + (1 - m) ** 2) / 2)
    assert adv_nan[1] == 0.0
    assert torch.allclose(adv_nan[[0, 2, 3]],
                          torch.tensor([1 - m, 0 - m, 1 - m]) / (std + 1e-4), atol=1e-6)
    # shape / option guards
    for bad in (lambda: group_advantages(r, 3), lambda: group_advantages(r, 4, scale="x")):
        try:
            bad()
            assert False, "expected ValueError"
        except ValueError:
            pass
    print("[grpo] group advantages PASSED")


def test_combine_rewards():
    nan = float("nan")
    pf = torch.tensor([[1.0, nan], [nan, nan], [0.5, 2.0]])
    rew, all_nan = combine_rewards(pf, [1.0, 0.5])
    assert torch.allclose(rew, torch.tensor([1.0, 0.0, 1.5]))
    assert all_nan.tolist() == [False, True, False]
    rew1, _ = combine_rewards(pf)
    assert torch.allclose(rew1, torch.tensor([1.0, 0.0, 2.5]))
    try:
        combine_rewards(pf, [1.0])
        assert False
    except ValueError:
        pass
    print("[grpo] combine rewards PASSED")


# ----------------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------------

def _rows(R=3, T=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    logps = -torch.rand(R, T, generator=g) * 2
    mask = torch.ones(R, T)
    mask[0, 3:] = 0                        # ragged completions
    mask[2, 4:] = 0
    logps = logps * mask
    adv = torch.tensor([1.0, -0.5, 0.25])
    return logps, mask, adv


def test_loss_reinforce_equivalence():
    """With old == policy (ratio 1, one optimizer step per generation batch)
    the clipped surrogate's gradient is REINFORCE's: for loss_type grpo
    d loss / d logp_t = -A_i / |y_i| / R."""
    logps, mask, adv = _rows()
    lp = logps.clone().requires_grad_(True)
    loss, m = grpo_loss(lp, mask, adv, loss_type="grpo")
    loss.backward()
    R = lp.shape[0]
    expect = -(adv.unsqueeze(1) / mask.sum(1, keepdim=True)) * mask / R
    assert torch.allclose(lp.grad, expect, atol=1e-6)
    assert m["clip_frac"] == 0.0 and m["kl"] == 0.0 and abs(m["ratio_mean"] - 1) < 1e-6
    # value: with ratio 1 every token's surrogate is -A_i, so the loss VALUE
    # is -mean_i A_i (the gradient is where the logps enter)
    assert torch.allclose(loss.detach(), -adv.mean(), atol=1e-6)
    print("[grpo] REINFORCE equivalence PASSED")


def test_loss_clipping():
    """Ratios 1.5 under A > 0 clip to 1 + eps_high; ratios 0.5 under A < 0
    clip to 1 - eps; the metrics count exactly those tokens."""
    T = 4
    logps = torch.zeros(2, T)
    mask = torch.ones(2, T)
    adv = torch.tensor([1.0, -1.0])
    old = torch.zeros(2, T)
    old[0] = -math.log(1.5)        # policy/old = 1.5 on row 0
    old[1] = -math.log(0.5)        # 0.5 on row 1
    loss, m = grpo_loss(logps, mask, adv, old_token_logps=old,
                        epsilon=0.2, epsilon_high=0.28, loss_type="bnpo")
    # row 0: -min(1.5*1, 1.28*1) = -1.28 ; row 1: -min(0.5*-1, 0.8*-1) = 0.8
    assert torch.allclose(loss, torch.tensor((-1.28 * T + 0.8 * T) / (2 * T)), atol=1e-6)
    assert abs(m["clip_high"] - 0.5) < 1e-9 and abs(m["clip_low"] - 0.5) < 1e-9
    assert abs(m["clip_frac"] - 1.0) < 1e-9
    # flipping the advantages puts every ratio on the unclipped side
    _, m2 = grpo_loss(logps, mask, -adv, old_token_logps=old, epsilon=0.2,
                      epsilon_high=0.28, loss_type="bnpo")
    assert m2["clip_frac"] == 0.0
    print("[grpo] clipping PASSED")


def test_loss_kl():
    logps, mask, adv = _rows()
    base, _ = grpo_loss(logps, mask, adv, loss_type="bnpo")
    same, m = grpo_loss(logps, mask, adv, ref_token_logps=logps.clone(),
                        beta=0.1, loss_type="bnpo")
    assert torch.allclose(base, same, atol=1e-7) and m["kl"] == 0.0
    ref = logps - 0.3 * mask
    with_kl, m = grpo_loss(logps, mask, adv, ref_token_logps=ref, beta=0.1,
                           loss_type="bnpo")
    d = -0.3
    k3 = math.exp(d) - d - 1          # > 0, identical on every masked token
    assert abs(m["kl"] - k3) < 1e-6
    assert torch.allclose(with_kl, base + 0.1 * k3, atol=1e-6)   # ratio 1 -> no IS factor
    try:
        grpo_loss(logps, mask, adv, beta=0.1)
        assert False
    except ValueError:
        pass
    print("[grpo] k3 KL PASSED")


def test_loss_reductions_and_is_level():
    logps, mask, adv = _rows()
    tok = (logps * mask)
    per_tok_sum = -(adv.unsqueeze(1) * tok).sum()      # ratio 1 -> -A * logp per token... value check below
    # with ratio 1, per-token loss = -A (coef 1 * A); its token sum = -sum_i A_i |y_i|
    s = -(adv * mask.sum(1)).sum()
    l_bnpo, _ = grpo_loss(logps, mask, adv, loss_type="bnpo")
    assert torch.allclose(l_bnpo, s / mask.sum(), atol=1e-6)
    l_dapo, _ = grpo_loss(logps, mask, adv, loss_type="dapo", total_tokens=100)
    assert torch.allclose(l_dapo, s / 100, atol=1e-6)
    l_dapo_local, _ = grpo_loss(logps, mask, adv, loss_type="dapo")
    assert torch.allclose(l_dapo_local, l_bnpo, atol=1e-6)
    l_dr, _ = grpo_loss(logps, mask, adv, loss_type="dr_grpo", max_completion_length=8)
    assert torch.allclose(l_dr, s / (3 * 8), atol=1e-6)
    try:
        grpo_loss(logps, mask, adv, loss_type="dr_grpo")
        assert False
    except ValueError:
        pass
    # sequence-level IS: one ratio per row = exp(mean log ratio)
    old = logps - 0.1 * mask
    _, mt = grpo_loss(logps, mask, adv, old_token_logps=old, importance_sampling_level="token")
    _, ms = grpo_loss(logps, mask, adv, old_token_logps=old, importance_sampling_level="sequence")
    assert abs(mt["ratio_mean"] - math.exp(0.1)) < 1e-5
    assert abs(ms["ratio_mean"] - math.exp(0.1)) < 1e-5
    del per_tok_sum
    print("[grpo] reductions + IS level PASSED")


# ----------------------------------------------------------------------------
# reward plug-ins
# ----------------------------------------------------------------------------

def test_reward_loader_and_call(tmp_path):
    fn, name = load_reward_func("think_format")
    assert fn is rewards_mod.think_format and name == "think_format"
    fn2, _ = load_reward_func("builtin:exact_answer")
    assert fn2 is rewards_mod.exact_answer
    p = tmp_path / "my_rewards.py"
    p.write_text("def shout(prompts, completions, completion_ids, bonus=None, **kw):\n"
                 "    return [ (1.0 if c.isupper() else 0.0) + (bonus[i] or 0) "
                 "for i, c in enumerate(completions)]\n"
                 "def abstain(prompts, completions, completion_ids, **kw):\n"
                 "    return [None] * len(completions)\n")
    f3, n3 = load_reward_func(f"{p}:shout")
    assert n3 == "my_rewards.py:shout"
    f4, n4 = load_reward_func("exl3train.rewards:think_format")
    assert f4 is rewards_mod.think_format
    f5, _ = load_reward_func(f"{p}:abstain")
    for bad in ("nope", f"{p}:missing", "exl3train.rewards:missing"):
        try:
            load_reward_func(bad)
            assert False, bad
        except (ValueError, AttributeError):
            pass
    out = call_reward_funcs([(f3, n3), (f5, "abstain")],
                            prompts=["q", "q"], completions=["HI", "hi"],
                            completion_ids=[[1], [2, 3]],
                            extras={"bonus": [0.5, None]})
    assert out.shape == (2, 2)
    assert out[0, 0] == 1.5 and out[1, 0] == 0.0
    assert torch.isnan(out[:, 1]).all()
    try:
        call_reward_funcs([(lambda **k: [1.0], "short")], ["q", "q"], ["a", "b"], [[1], [2]])
        assert False
    except ValueError:
        pass
    print("[grpo] reward loader + call PASSED")


def test_builtin_rewards():
    ea = rewards_mod.exact_answer
    comps = ["so the answer is \\boxed{42}.", "x = 7\n#### 7", "I think 3, no 4", "", "foo"]
    out = ea(["p"] * 5, comps, [[0]] * 5, answer=["42", "7.0", "4", "1", None])
    assert out == [1.0, 1.0, 1.0, 0.0, None]
    assert rewards_mod.extract_answer("Total: 1,234 dollars") == "1234"
    tf = rewards_mod.think_format
    assert tf(["p"] * 3, ["<think>hm</think> 4", "<think>x</think>", "4"], None) == [1.0, 0.0, 0.0]
    lt = rewards_mod.length_target
    r = lt(["p"] * 2, ["a", "b"], [[0] * 10, [0] * 30], target_len=[10, 10])
    assert r[0] == 1.0 and abs(r[1] - 0.0) < 1e-9
    print("[grpo] built-in rewards PASSED")


# ----------------------------------------------------------------------------
# exact sampler
# ----------------------------------------------------------------------------

class ChainNet(nn.Module):
    """Vocab of 6: the policy deterministically continues x -> x+1 up to 4,
    and 4 -> 5 (= EOS). Hidden = sharp one-hot of the next token; the LM
    head is the identity, so sampling at T=1 is effectively greedy."""
    V, EOS = 6, 5

    def __init__(self):
        super().__init__()
        self.final_softcap = 0.0
        self.device = torch.device("cpu")
        self.calls = 0

    def forward(self, ids, attention_mask=None):
        self.calls += 1
        nxt = torch.clamp(ids + 1, max=self.EOS)
        return F.one_hot(nxt, self.V).float() * 40.0

    def lm_head_weight_fn(self):
        return lambda: torch.eye(self.V)


def test_sample_completions_exact():
    net = ChainNet()
    net.train()
    out = sample_completions_exact(net, [[0], [3], [2, 3]], 6, temperature=1.0,
                                   stop_ids=[ChainNet.EOS], pad_id=0)
    assert out[0] == ([1, 2, 3, 4, 5], False)
    assert out[1] == ([4, 5], False)
    assert out[2] == ([4, 5], False)
    assert net.training                       # mode restored
    assert net.calls == 5                     # stopped once every row was done
    out = sample_completions_exact(net, [[0], [3]], 3, stop_ids=[ChainNet.EOS])
    assert out[0] == ([1, 2, 3], True) and out[1] == ([4, 5], False)
    out = sample_completions_exact(net, [[3]], 3, stop_ids=())
    assert out[0] == ([4, 5, 5], True)        # no stop ids: runs to the cap
    print("[grpo] exact sampler PASSED")


if __name__ == "__main__":
    import tempfile, pathlib
    test_group_advantages()
    test_combine_rewards()
    test_loss_reinforce_equivalence()
    test_loss_clipping()
    test_loss_kl()
    test_loss_reductions_and_is_level()
    test_reward_loader_and_call(pathlib.Path(tempfile.mkdtemp()))
    test_builtin_rewards()
    test_sample_completions_exact()
    print("ALL GRPO TESTS PASSED")
