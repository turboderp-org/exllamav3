"""
CPU tests for training/qlora_train_grpo.py's orchestration: the native
sampler's token collection, generate_and_score (rollout -> decode -> reward
plug-ins -> advantages) and policy_pass (micro-batched per-token logps ->
GRPO loss -> backward, with the reference / old-policy caches), all on stub
net / tokenizer / generator objects. No model, no CUDA.

The trainer imports ``exllamav3`` for Config/Model/Tokenizer/Cache/Generator/
Job and the sampler; on a machine without the built extension those names are
stubbed here (the ``exllamav3.training`` package is pointed at the real
directory, so native_llama / grpo / rewards load for real).

Run:  python -m pytest -q tests/test_grpo_trainer.py
"""

from __future__ import annotations
import contextlib
import os
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")
_SCRIPTS = os.path.join(_ROOT, "training")

try:
    import exllamav3  # noqa: F401  (the real package, when the extension is built)
    from exllamav3 import Job  # noqa: F401
except Exception:
    ex = types.ModuleType("exllamav3")
    ex.__path__ = []
    for _n in ("Config", "Model", "Tokenizer", "Cache", "Generator"):
        setattr(ex, _n, type(_n, (), {}))

    class Job:  # the kwargs sample_completions_native passes
        def __init__(self, input_ids, max_new_tokens=None, sampler=None, seed=None,
                     stop_conditions=None, **kw):
            self.input_ids = input_ids
            self.max_new_tokens = max_new_tokens
            self.sampler = sampler
            self.seed = seed
            self.stop_conditions = set(stop_conditions or [])

    ex.Job = Job
    gen_pkg = types.ModuleType("exllamav3.generator")
    gen_pkg.__path__ = []
    samp = types.ModuleType("exllamav3.generator.sampler")

    class ComboSampler:
        def __init__(self, **kw):
            self.kw = kw

    samp.ComboSampler = ComboSampler
    gen_pkg.sampler = samp
    tr = types.ModuleType("exllamav3.training")
    tr.__path__ = [_TRAIN_DIR]
    ex.generator = gen_pkg
    ex.training = tr
    sys.modules.update({"exllamav3": ex, "exllamav3.generator": gen_pkg,
                        "exllamav3.generator.sampler": samp,
                        "exllamav3.training": tr})

if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import qlora_train_grpo as T  # noqa: E402


# ----------------------------------------------------------------------------
# stubs
# ----------------------------------------------------------------------------

class StubNet(nn.Module):
    """Vocab 6, EOS 5, deterministic chain x -> x+1 (the exact sampler's
    policy); compute_logps scores every supervised token at -1 (base) or
    -1 + p (policy) like test_realtime's stub, so the GRPO gradient lands on
    p with a sign set by the reward/length correlation."""
    V, EOS = 6, 5

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.device = torch.device("cpu")
        self.final_softcap = 0.0
        self.adapters_off = False
        self.applied = self.removed = 0
        self.logps_calls = []
        self.target_modules = ["q_proj"]
        self._wrappers = []

    def forward(self, ids, attention_mask=None):
        return F.one_hot(torch.clamp(ids + 1, max=self.EOS), self.V).float() * 40.0

    def lm_head_weight_fn(self):
        return lambda: torch.eye(self.V)

    def compute_logps(self, input_ids, labels, attention_mask=None, chunk=0,
                      return_per_token=False):
        self.logps_calls.append((input_ids, labels, self.adapters_off))
        mask = labels[:, 1:] != -100
        per_tok = (-1.0 + (0.0 if self.adapters_off else self.p)) * mask.float()
        if return_per_token:
            return per_tok, mask
        return per_tok.sum(-1), mask.sum(-1)

    @contextlib.contextmanager
    def adapters_disabled(self):
        prev = self.adapters_off
        self.adapters_off = True
        try:
            yield self
        finally:
            self.adapters_off = prev

    def apply_to_native(self, scaling=1.0):
        self.applied += 1

    def remove_from_native(self):
        self.removed += 1

    def trainable_parameters(self):
        return [self.p]


class StubTokenizer:
    eos_token_id = 5

    def decode(self, ids, decode_special_tokens=False):
        return [" ".join(str(int(t)) for t in row) for row in ids]

    def encode(self, text, add_bos=False, add_eos=False, encode_special_tokens=False):
        return torch.tensor([[7]]) if text == "<eot>" else torch.tensor([[8, 9]])


class StubGenerator:
    """Runs every enqueued job to completion on the chain policy, streaming
    the tokens across the result dict and the eos 'held' blob like the real
    generator. ``drop_stop`` leaves the stop token OUT of the token stream to
    exercise the sampler's fix-up."""

    def __init__(self, drop_stop=False):
        self.jobs = []
        self.drop_stop = drop_stop

    def enqueue(self, job):
        self.jobs.append(job)

    def num_remaining_jobs(self):
        return len(self.jobs)

    def iterate(self):
        results = []
        for job in self.jobs:
            last = int(job.input_ids[0, -1])
            toks, reason = [], "max_new_tokens"
            for _ in range(job.max_new_tokens - 1):
                last = min(last + 1, StubNet.EOS)
                toks.append(last)
                if last in job.stop_conditions:
                    reason = "stop_token"
                    break
            r = {"job": job, "eos": True, "eos_reason": reason}
            if reason == "stop_token":
                r["eos_triggering_token_id"] = toks[-1]
                if self.drop_stop:
                    toks = toks[:-1]
            k = len(toks) // 2
            if toks[:k]:
                r["token_ids"] = torch.tensor(toks[:k])
            if toks[k:]:
                r["held"] = {"token_ids": torch.tensor(toks[k:])}
            results.append(r)
        self.jobs = []
        return results


def make_args(**over):
    a = types.SimpleNamespace(
        temperature=1.0, top_k=0, top_p=1.0, max_completion_length=6,
        num_generations=2, scale_rewards="group", mask_truncated_completions=False,
        batch=2, beta=0.0, epsilon=0.2, epsilon_high=None, loss_type="dapo",
        importance_sampling_level="token", ce_chunk=0, num_iterations=1)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def alternating_reward(prompts, completions, completion_ids, **columns):
    """0/1 alternating so every group of 2 has a +/- advantage; asserts the
    extras arrive as per-row lists aligned with the completions."""
    assert columns["answer"] == ["x", "x", "y", "y"], columns
    assert prompts == ["p0", "p0", "p1", "p1"]
    return [float(i % 2) for i in range(len(completions))]


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------

def test_native_sampler_collection():
    args = make_args(max_completion_length=3)
    for drop in (False, True):
        gen = StubGenerator(drop_stop=drop)
        out = T.sample_completions_native(gen, [[0], [3], [4]], 3, args, [5], seed=1)
        assert out[0] == ([1, 2, 3], True)       # ran into the cap
        assert out[1] == ([4, 5], False)         # stop token kept as last token
        assert out[2] == ([5], False)            # immediate stop -> 1-token completion
    print("[grpo-trainer] native sampler collection OK")


def test_stop_token_ids():
    assert T.stop_token_ids(StubTokenizer(), "<eot>") == [5, 7]
    assert T.stop_token_ids(StubTokenizer(), "multi") == [5]      # 2-token eot ignored
    assert T.stop_token_ids(StubTokenizer(), "") == [5]
    print("[grpo-trainer] stop token ids OK")


def _batch():
    return [{"prompt_ids": [0], "prompt": "p0", "extras": {"answer": "x"}},
            {"prompt_ids": [3], "prompt": "p1", "extras": {"answer": "y"}}]


def test_generate_and_score_exact_and_native():
    import random
    net, tok = StubNet(), StubTokenizer()
    args = make_args()
    funcs = [(alternating_reward, "alt")]
    for gen in (None, StubGenerator()):
        s = T.generate_and_score(net, _batch(), args, gen, funcs, None,
                                 random.Random(0), 0, [5], tok)
        comps = [c for _, c in s["rows"]]
        assert comps == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [4, 5], [4, 5]]
        assert s["lengths"] == [5, 5, 2, 2] and s["truncated"] == [False] * 4
        assert s["texts"][0] == "1 2 3 4 5" and s["texts"][2] == "4 5"
        assert s["rewards"].tolist() == [0.0, 1.0, 0.0, 1.0]
        # group std of {0,1} = sqrt(.5); advantages +-0.5/(std+eps)
        a = 0.5 / (0.5 ** 0.5 + 1e-4)
        assert torch.allclose(s["adv"], torch.tensor([-a, a, -a, a]), atol=1e-6)
        assert s["info"]["frac_zero_std"] == 0.0 and s["per_func"].shape == (4, 1)
    assert net.applied == 1 and net.removed == 1        # native path pushed + removed the adapter
    print("[grpo-trainer] generate_and_score OK")


def test_policy_pass_paths():
    import random
    net, tok = StubNet(), StubTokenizer()
    args = make_args()
    s = T.generate_and_score(net, _batch(), args, None, [(alternating_reward, "alt")],
                             None, random.Random(0), 0, [5], tok)
    # plain pass: ratio 1, dapo; the stub's logp is -1 + p on every token, so
    # d loss / dp = -sum_i A_i |y_i| / total_tokens; here A is +-a with equal
    # lengths inside each group -> the derivative is exactly 0 and the loss
    # value is -sum(A_i |y_i|)/N = 0 too. Perturb one advantage to get signal.
    s["adv"][1] = s["adv"][1] * 2
    loss, m = T.policy_pass(net, s, args, 0)
    assert net.p.grad is not None and net.p.grad.abs().item() > 0
    assert m["clip_frac"] == 0.0 and m["kl"] == 0.0 and m["sup"] == 14
    tot = 14.0
    expect_loss = -float((s["adv"] * torch.tensor([5., 5., 2., 2.])).sum()) / tot
    assert abs(loss - expect_loss) < 1e-5
    assert len(net.logps_calls) == 2                      # 4 rows / batch 2
    net.p.grad = None
    net.logps_calls.clear()

    # masked truncated rows drop out of the token count and the loss
    s["truncated"] = [True, False, False, False]
    args_m = make_args(mask_truncated_completions=True)
    loss_m, m_m = T.policy_pass(net, s, args_m, 0)
    assert m_m["sup"] == 9
    net.p.grad = None
    s["truncated"] = [False] * 4

    # beta > 0: reference logps come from the adapters-disabled forward
    args_b = make_args(beta=0.1)
    mb = [range(0, 2), range(2, 4)]
    ref = [T._token_logps(net, [s["rows"][i] for i in r], 0, args_b,
                          truncated=[False, False], reference=True) for r in mb]
    assert all(call[2] for call in net.logps_calls[-2:])  # adapters were off
    assert not net.adapters_off                            # and restored
    assert all(not t.requires_grad for t, _ in ref)
    loss_b, m_b = T.policy_pass(net, s, args_b, 0, ref_cache=ref)
    assert m_b["kl"] == 0.0                                # policy == reference at p = 0
    assert abs(loss_b - loss) < 1e-6
    net.p.grad = None

    # num_iterations > 1: old logps captured before the pass, ratio still 1
    args_i = make_args(num_iterations=2, loss_type="grpo")
    with torch.no_grad():
        old = [T._token_logps(net, [s["rows"][i] for i in r], 0, args_i,
                              truncated=[False, False]) for r in mb]
    loss_i, m_i = T.policy_pass(net, s, args_i, 0, old_cache=old)
    assert abs(m_i["ratio_mean"] - 1.0) < 1e-6 and net.p.grad is not None
    print("[grpo-trainer] policy_pass paths OK")


if __name__ == "__main__":
    test_native_sampler_collection()
    test_stop_token_ids()
    test_generate_and_score_exact_and_native()
    test_policy_pass_paths()
    print("ALL GRPO TRAINER TESTS PASSED")
