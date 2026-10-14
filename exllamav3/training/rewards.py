"""
Built-in GRPO reward functions (``--reward exact_answer`` etc.).

Every reward follows TRL's custom-reward contract so the same functions
drop into ``trl.GRPOTrainer`` unchanged, and so any TRL-style reward file
drops into ours (``--reward path/to/file.py:func``):

    def my_reward(prompts, completions, completion_ids, **columns) -> list[float | None]

``prompts`` are the dataset's prompt values (strings or message lists, one
per completion -- a prompt is repeated for each of its ``G`` completions),
``completions`` the decoded completion texts, ``completion_ids`` their token
ids, and every other dataset column arrives as a keyword list aligned with
the completions (``answer=[...]``). Return one number per completion, or
``None`` to abstain on a row (that function is then left out of the row's
reward; a row every function abstains on gets reward 0).

These three cover the usual verifiable-reward demos; a judge-model reward or
anything task-specific is a user file with the same signature.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

BUILTIN_REWARDS = ("exact_answer", "think_format", "length_target")

_BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_THINK = re.compile(r"^\s*<think>.*?</think>\s*\S", re.DOTALL)


def extract_answer(text: str) -> str:
    """Final-answer extraction for :func:`exact_answer`: the last
    ``\\boxed{...}``, else the text after the last ``####`` (GSM8K style),
    else the last number in the text, else the last non-empty line."""
    if not text:
        return ""
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    if "####" in text:
        return text.rsplit("####", 1)[1].strip().splitlines()[0].strip() if \
            text.rsplit("####", 1)[1].strip() else ""
    nums = _NUMBER.findall(text.replace(",", ""))
    if nums:
        return nums[-1]
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _normalize(s) -> str:
    s = str(s).strip().strip(".").replace(",", "").replace("$", "").lower()
    try:
        f = float(s)
        return f"{f:.6g}"
    except ValueError:
        return " ".join(s.split())


def exact_answer(prompts: Sequence, completions: Sequence[str],
                 completion_ids: Optional[Sequence] = None,
                 answer: Optional[Sequence] = None, **columns
                 ) -> list[Optional[float]]:
    """1.0 when the completion's extracted final answer matches the row's
    ``answer`` column (numeric or normalized-string equality), else 0.0;
    ``None`` (abstain) for rows with no answer."""
    out: list[Optional[float]] = []
    for i, c in enumerate(completions):
        a = None if answer is None else answer[i]
        if a is None or (isinstance(a, str) and not a.strip()):
            out.append(None)
            continue
        out.append(1.0 if _normalize(extract_answer(c)) == _normalize(a) else 0.0)
    return out


def think_format(prompts: Sequence, completions: Sequence[str],
                 completion_ids: Optional[Sequence] = None, **columns
                 ) -> list[float]:
    """1.0 when the completion opens with a ``<think>...</think>`` block
    followed by a non-empty answer, else 0.0 (the common reasoning-format
    reward)."""
    return [1.0 if _THINK.match(c or "") else 0.0 for c in completions]


def length_target(prompts: Sequence, completions: Sequence[str],
                  completion_ids: Optional[Sequence] = None,
                  target_len: Optional[Sequence] = None, **columns
                  ) -> list[float]:
    """``max(0, 1 - |len - target| / target)`` on the completion's token
    count, ``target`` from a per-row ``target_len`` column (default 256
    tokens). A soft length prior to mix with a task reward."""
    out = []
    for i, c in enumerate(completions):
        t = None if target_len is None else target_len[i]
        t = float(t) if t else 256.0
        n = len(completion_ids[i]) if completion_ids is not None else len((c or "").split())
        out.append(max(0.0, 1.0 - abs(n - t) / t))
    return out
