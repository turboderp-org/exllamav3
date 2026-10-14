"""
Group Relative Policy Optimization (GRPO) for QLoRA-on-EXL3: the math.

Pure-torch pieces of an on-policy GRPO step, CPU-testable
(``tests/test_grpo.py``); the trainer (``training/qlora_train_grpo.py``)
owns rollouts through the native generator, the reward plug-ins and the
optimizer loop.

GRPO (Shao et al. 2024, DeepSeekMath, arXiv:2402.03300): for each prompt
sample a GROUP of ``G`` completions from the current policy, score them with
a reward function, and use the group-relative advantage
``A_i = (r_i - mean(r)) / std(r)`` in a PPO-style clipped surrogate

    L = -1/|y| sum_t min( rho_t A_i, clip(rho_t, 1-eps, 1+eps) A_i )
        + beta * KL_t,     rho_t = pi_theta(y_t) / pi_old(y_t)

with the per-token k3 KL estimator ``exp(ref - logp) - (ref - logp) - 1``
(Schulman 2020) against a frozen reference. Semantics follow HuggingFace
TRL's ``GRPOTrainer`` (1.15.0) so hyperparameters transfer:

  * advantages: ``scale_rewards`` = ``group`` (divide by the group std,
    the paper) | ``batch`` (batch std, Dr. GRPO-style scale) | ``none``;
    an ``eps`` of 1e-4 on the std; NaN rewards (a reward function that
    returned ``None``) are ignored in the statistics and get advantage 0;
  * loss reductions (``loss_type``): ``grpo`` (per-sequence token mean, then
    batch mean -- the paper), ``bnpo`` (token mean over the micro-batch),
    ``dr_grpo`` (token sum / (rows * max_completion_length)), ``dapo`` (token
    sum / the GENERATION batch's completion tokens -- TRL's default);
  * two-sided clipping ``epsilon`` / ``epsilon_high`` (DAPO's clip-higher);
    ``importance_sampling_level`` = ``token`` | ``sequence`` (GSPO);
  * ``beta`` = 0 (TRL's default) drops the KL term and the reference
    forward entirely; ``beta > 0`` scores the reference as the frozen base
    (``NativeLlamaQLoRA.adapters_disabled()`` -- no second model copy).

The "old policy" of the ratio is the policy that produced the rollouts: a
DETACHED copy of the current per-token logps when a generation batch is
trained for one optimizer step (``num_iterations == 1``, rho == 1, clipping
a no-op), or the logps captured right after rollout when it is reused for
several (``num_iterations > 1``). Never a model.

Credit: formulations adapted from HuggingFace TRL (Apache-2.0, Copyright The
HuggingFace Team); an independent reimplementation against the exllamav3
native training path, not a copy of TRL code. See ``doc/orpo_grpo_plan.md``
for the TRL / axolotl / paper comparison the semantics were checked against.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Callable, Optional, Sequence

import torch

GRPO_LOSS_TYPES = ("grpo", "bnpo", "dr_grpo", "dapo")
SCALE_REWARDS = ("group", "batch", "none")
IS_LEVELS = ("token", "sequence")


# ---------------------------------------------------------------------------
# Rewards -> advantages
# ---------------------------------------------------------------------------

def _nanstd(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Bessel-corrected std ignoring NaN (TRL's ``nanstd``); 0 where fewer
    than two values are present."""
    mean = torch.nanmean(x, dim=dim, keepdim=True)
    var = torch.nanmean((x - mean) ** 2, dim=dim)
    count = (~torch.isnan(x)).sum(dim=dim)
    var = var * count / (count - 1).clamp(min=1)
    var = torch.where(count > 1, var, torch.zeros_like(var))
    return torch.sqrt(var)


def combine_rewards(per_func: torch.Tensor,
                    weights: Optional[Sequence[float]] = None,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted sum of the reward functions' outputs per row.

    ``per_func``: ``[R, F]`` with NaN where a function returned ``None`` for
    that row (skipped: TRL's multi-task convention). ``weights``: ``[F]``
    (default all 1). Returns ``(rewards [R], all_nan [R] bool)``; a row where
    EVERY function abstained sums to 0 (TRL's ``nansum``) and is flagged so
    the caller can warn."""
    R, F = per_func.shape
    w = (torch.ones(F, dtype=torch.float32) if weights is None
         else torch.as_tensor(list(weights), dtype=torch.float32))
    if w.numel() != F:
        raise ValueError(f"{F} reward functions but {w.numel()} weights")
    pf = per_func.float()
    all_nan = torch.isnan(pf).all(dim=1)
    return torch.nansum(pf * w.to(pf.device).unsqueeze(0), dim=1), all_nan


def group_advantages(rewards: torch.Tensor, num_generations: int,
                     scale: str = "group", eps: float = 1e-4,
                     ) -> tuple[torch.Tensor, dict]:
    """Group-relative advantages (GRPO eq. for outcome supervision; TRL
    ``scale_rewards`` semantics).

    ``rewards``: ``[P * G]`` with the ``G`` completions of each prompt
    CONSECUTIVE (NaN = no reward -> advantage 0, ignored in the stats).
    Returns ``(advantages [P * G] fp32, info)`` where ``info`` carries
    ``reward_mean`` / ``reward_std`` over the batch and ``frac_zero_std``,
    the fraction of groups whose rewards were all equal (no learning signal
    from that prompt -- TRL's ``frac_reward_zero_std``)."""
    if scale not in SCALE_REWARDS:
        raise ValueError(f"scale must be one of {SCALE_REWARDS}, got {scale!r}")
    if num_generations < 1:
        raise ValueError("num_generations must be >= 1")
    r = rewards.float()
    if r.numel() % num_generations:
        raise ValueError(f"{r.numel()} rewards is not a multiple of "
                         f"num_generations={num_generations}")
    g = r.view(-1, num_generations)
    mean = torch.nanmean(g, dim=1, keepdim=True)
    gstd = _nanstd(g, dim=1)                               # [P]
    adv = g - mean
    if scale == "group":
        adv = adv / (gstd.unsqueeze(1) + eps)
    elif scale == "batch":
        adv = adv / (_nanstd(r, dim=0) + eps)
    adv = torch.nan_to_num(adv.reshape(-1), nan=0.0)
    info = {
        "reward_mean": float(torch.nanmean(r)) if (~torch.isnan(r)).any() else 0.0,
        "reward_std": float(_nanstd(r, dim=0)),
        "frac_zero_std": float(torch.isclose(gstd, torch.zeros_like(gstd))
                               .float().mean()),
    }
    return adv, info


# ---------------------------------------------------------------------------
# The clipped surrogate
# ---------------------------------------------------------------------------

def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x.expand_as(mask) if x.shape != mask.shape else x
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def grpo_loss(
    token_logps: torch.Tensor,            # [R, T] policy log p(y_t), 0 off-mask
    mask: torch.Tensor,                   # [R, T] 1 on completion tokens
    advantages: torch.Tensor,             # [R]
    *,
    old_token_logps: Optional[torch.Tensor] = None,   # [R, T]; None = detach(policy)
    ref_token_logps: Optional[torch.Tensor] = None,   # [R, T]; needed when beta > 0
    beta: float = 0.0,
    epsilon: float = 0.2,
    epsilon_high: Optional[float] = None,
    loss_type: str = "dapo",
    importance_sampling_level: str = "token",
    max_completion_length: Optional[int] = None,
    total_tokens: Optional[float] = None,
    bias_correction_kl: bool = True,
) -> tuple[torch.Tensor, dict]:
    """GRPO loss for one micro-batch of ``R`` completion rows.

    ``token_logps`` / ``mask`` come from ``compute_logps(...,
    return_per_token=True)`` with the prompt masked; ``advantages`` from
    :func:`group_advantages`. Returns ``(loss, metrics)`` with detached
    ``kl`` (masked-mean k3, 0 when ``beta == 0``), ``clip_frac`` (fraction of
    tokens whose ratio hit a clip bound in the direction that matters:
    ``rho < 1 - eps`` with ``A < 0`` or ``rho > 1 + eps_high`` with ``A > 0``),
    ``clip_low`` / ``clip_high`` and ``ratio_mean``.

    Reductions: ``grpo`` and ``bnpo`` normalize within the micro-batch (the
    trainer averages micro-batch losses over the accumulation window, TRL's
    ``loss / gradient_accumulation_steps``); ``dr_grpo`` divides the token sum
    by ``R * max_completion_length``; ``dapo`` divides by ``total_tokens``,
    the completion-token count of the whole generation batch (defaults to
    this micro-batch's tokens when not given).
    """
    if loss_type not in GRPO_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {GRPO_LOSS_TYPES}, got {loss_type!r}")
    if importance_sampling_level not in IS_LEVELS:
        raise ValueError(f"importance_sampling_level must be one of {IS_LEVELS}")
    if beta != 0.0 and ref_token_logps is None:
        raise ValueError("beta > 0 needs ref_token_logps")
    eps_low = epsilon
    eps_high = epsilon if epsilon_high is None else epsilon_high
    mask = mask.to(token_logps.dtype)
    adv = advantages.to(token_logps.device, token_logps.dtype).unsqueeze(1)   # [R, 1]
    old = token_logps.detach() if old_token_logps is None else old_token_logps

    log_ratio = token_logps - old
    if importance_sampling_level == "sequence":
        log_w = ((log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
    else:
        log_w = log_ratio
    coef_1 = torch.exp(log_w)
    coef_2 = torch.clamp(coef_1, 1.0 - eps_low, 1.0 + eps_high)
    per_token = -torch.min(coef_1 * adv, coef_2 * adv)

    kl_mean = torch.zeros((), device=token_logps.device)
    if beta != 0.0:
        d = ref_token_logps - token_logps
        per_token_kl = torch.exp(d) - d - 1.0
        if bias_correction_kl:
            per_token_kl = per_token_kl * coef_1
        per_token = per_token + beta * per_token_kl
        kl_mean = _masked_mean(torch.exp(d) - d - 1.0, mask).detach()

    if loss_type == "grpo":
        loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        loss = (per_token * mask).sum() / mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        if not max_completion_length:
            raise ValueError("loss_type='dr_grpo' needs max_completion_length")
        loss = (per_token * mask).sum() / (mask.shape[0] * max_completion_length)
    else:  # dapo
        norm = float(total_tokens) if total_tokens else float(mask.sum())
        loss = (per_token * mask).sum() / max(norm, 1.0)

    with torch.no_grad():
        c1 = coef_1.expand_as(mask) if coef_1.shape != mask.shape else coef_1
        a = adv.expand_as(mask)
        low = ((c1 < 1.0 - eps_low) & (a < 0)).to(mask.dtype)
        high = ((c1 > 1.0 + eps_high) & (a > 0)).to(mask.dtype)
        metrics = {
            "kl": float(kl_mean),
            "clip_low": float(_masked_mean(low, mask)),
            "clip_high": float(_masked_mean(high, mask)),
            "clip_frac": float(_masked_mean(torch.clamp(low + high, max=1.0), mask)),
            "ratio_mean": float(_masked_mean(c1, mask)),
        }
    return loss, metrics


# ---------------------------------------------------------------------------
# Exact (differentiable-forward) sampler with stop tokens
# ---------------------------------------------------------------------------

def apply_sampling_filters(logits: torch.Tensor, temperature: float,
                           top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """temperature -> top-k -> top-p on a ``[R, V]`` logits row block (the
    EBFT exact sampler's stack, shared here)."""
    logits = logits.float() / temperature
    if top_k and top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sp = sorted_logits.softmax(dim=-1)
        drop = sp.cumsum(dim=-1) - sp >= top_p          # top token always survives
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits)
    return logits


def sample_completions_exact(
    net, prompt_rows: Sequence[Sequence[int]], max_new_tokens: int, *,
    temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
    stop_ids: Sequence[int] = (), pad_id: int = 0,
    generator: Optional[torch.Generator] = None,
) -> list[tuple[list[int], bool]]:
    """On-policy completions through the DIFFERENTIABLE forward (no KV cache):
    ``max_new_tokens`` full re-forwards at most, each row stopping at the
    first token in ``stop_ids`` (kept as the completion's last token, so the
    policy is trained to emit it). Exactly the policy the loss is computed
    under -- zero sampling/scoring mismatch -- at the cost of speed; the
    trainer's default is the KV-cached native generator.

    Returns ``[(completion_ids, truncated)]`` per prompt row, ``truncated``
    = the row hit ``max_new_tokens`` without a stop token."""
    assert temperature > 0.0, "GRPO needs temperature > 0 (a group of identical greedy completions has no advantage)"
    R = len(prompt_rows)
    lens = torch.tensor([len(p) for p in prompt_rows], dtype=torch.long)
    L = int(lens.max())
    rows = torch.full((R, L + max_new_tokens), pad_id, dtype=torch.long)
    for i, p in enumerate(prompt_rows):
        rows[i, :len(p)] = torch.as_tensor(list(p), dtype=torch.long)
    device = getattr(net, "device", rows.device)
    rows, lens = rows.to(device), lens.to(device)
    stop = torch.tensor(sorted(set(int(s) for s in stop_ids)), dtype=torch.long,
                        device=device)
    done = torch.zeros(R, dtype=torch.bool, device=device)
    out: list[list[int]] = [[] for _ in range(R)]
    head_w = net.lm_head_weight_fn()
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if bool(done.all()):
                    break
                t = int(lens.max())
                attn = (torch.arange(t, device=device).unsqueeze(0)
                        < lens.unsqueeze(1)).long()
                hidden = net.forward(rows[:, :t], attention_mask=attn)
                idx = (lens - 1).to(hidden.device)
                last = hidden[torch.arange(R, device=hidden.device), idx]
                w = head_w()
                logits = last.to(w.device).to(w.dtype) @ w
                softcap = getattr(net, "final_softcap", 0.0)
                if softcap:
                    logits = softcap * torch.tanh(logits / softcap)
                logits = apply_sampling_filters(logits, temperature, top_k, top_p)
                nxt = torch.multinomial(logits.softmax(dim=-1), 1,
                                        generator=generator).squeeze(-1).to(device)
                active = ~done
                ar = torch.arange(R, device=device)
                rows[ar[active], lens[active]] = nxt[active]
                for i in ar[active].tolist():
                    out[i].append(int(nxt[i]))
                lens = lens + active.long()
                if stop.numel():
                    done = done | (active & torch.isin(nxt, stop))
    finally:
        if was_training:
            net.train()
    truncated = (~done).tolist()
    return [(out[i], bool(truncated[i])) for i in range(R)]


# ---------------------------------------------------------------------------
# Reward plug-ins
# ---------------------------------------------------------------------------

RewardFunc = Callable[..., Sequence[Optional[float]]]


def load_reward_func(spec: str) -> tuple[RewardFunc, str]:
    """Resolve a reward-function spec to ``(callable, display_name)``.

    Accepted forms (TRL's custom-reward contract either way -- the callable
    takes ``prompts=``, ``completions=``, ``completion_ids=`` and every other
    dataset column as a keyword list, and returns one ``float | None`` per
    completion):

      * ``exact_answer`` / ``builtin:exact_answer`` -- a built-in from
        ``exllamav3.training.rewards`` (see ``BUILTIN_REWARDS`` there);
      * ``path/to/rewards.py:my_reward`` -- a function in a Python FILE
        (loaded by path; the file is imported once per process);
      * ``my_package.rewards:my_reward`` -- a function in an importable
        module.
    """
    from . import rewards as builtin
    s = spec.strip()
    if s.startswith("builtin:"):
        s = s[len("builtin:"):]
    if ":" not in s:
        fn = getattr(builtin, s, None)
        if fn is None or not callable(fn) or s not in builtin.BUILTIN_REWARDS:
            raise ValueError(f"unknown built-in reward {spec!r}; built-ins: "
                             f"{', '.join(builtin.BUILTIN_REWARDS)}; or pass "
                             f"path/to/file.py:func or package.module:func")
        return fn, s
    target, func = s.rsplit(":", 1)
    if target.endswith(".py") or os.sep in target or os.path.exists(target):
        path = os.path.abspath(os.path.expanduser(target))
        if not os.path.exists(path):
            raise ValueError(f"reward file not found: {path}")
        name = f"_grpo_reward_{abs(hash(path)) & 0xffffffff:x}"
        mod = sys.modules.get(name)
        if mod is None:
            mspec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(mspec)
            sys.modules[name] = mod
            mspec.loader.exec_module(mod)
        display = f"{os.path.basename(path)}:{func}"
    else:
        mod = importlib.import_module(target)
        display = f"{target}:{func}"
    fn = getattr(mod, func, None)
    if fn is None or not callable(fn):
        raise ValueError(f"{spec!r}: no callable {func!r} in {target}")
    return fn, display


def call_reward_funcs(funcs: Sequence[tuple[RewardFunc, str]],
                      prompts: Sequence, completions: Sequence[str],
                      completion_ids: Sequence[Sequence[int]],
                      extras: Optional[dict] = None) -> torch.Tensor:
    """Call every reward function on the batch and stack the results into
    ``[R, F]`` (NaN where a function returned ``None``). ``extras`` maps each
    other dataset column to its per-row list (TRL passes them as kwargs, so a
    reward can read e.g. ``answer=[...]``)."""
    R = len(completions)
    out = torch.full((R, len(funcs)), float("nan"), dtype=torch.float32)
    kw = dict(extras or {})
    for j, (fn, name) in enumerate(funcs):
        res = fn(prompts=list(prompts), completions=list(completions),
                 completion_ids=[list(c) for c in completion_ids], **kw)
        res = list(res)
        if len(res) != R:
            raise ValueError(f"reward {name} returned {len(res)} values for "
                             f"{R} completions")
        out[:, j] = torch.tensor([float("nan") if v is None else float(v)
                                  for v in res], dtype=torch.float32)
    return out
