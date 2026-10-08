import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.modules.attention_fn import dispatch


def _call(q_len, hint):
    q = torch.empty((1, q_len, 8, 128), dtype = torch.float16)
    k = torch.empty((1, q_len, 1, 128), dtype = torch.float16)
    return dispatch.attn_dispatch(q, k, k, dispatch_cache = hint)


@pytest.mark.parametrize("q_lengths", [(32, 64), (64, 24, 64)])
def test_prefill_hint_does_not_hide_newly_eligible_backend(monkeypatch, q_lengths):
    served = []

    def preferred(args):
        if args.q_len < 33:
            return None
        served.append("preferred")
        return args.q

    def fallback(args):
        served.append("fallback")
        return args.q

    monkeypatch.setattr(dispatch, "attn_fns", [preferred, fallback])
    hint = {}
    for q_len in q_lengths:
        _call(q_len, hint)
        assert hint["fn"] is (preferred if q_len >= 33 else fallback)

    assert served == ["preferred" if q_len >= 33 else "fallback" for q_len in q_lengths]


def test_highest_priority_hint_runs_once(monkeypatch):
    calls = []

    def preferred(args):
        calls.append("preferred")
        return args.q

    def fallback(args):
        raise AssertionError("Compatible preferred backend should win")

    monkeypatch.setattr(dispatch, "attn_fns", [preferred, fallback])
    _call(64, {"fn": preferred})
    assert calls == ["preferred"]


def test_removed_backend_hint_is_not_called(monkeypatch):
    def removed(args):
        raise AssertionError("A backend outside the candidate list must not run")

    def preferred(args):
        return args.q

    monkeypatch.setattr(dispatch, "attn_fns", [preferred])
    hint = {"fn": removed}
    _call(64, hint)
    assert hint["fn"] is preferred


def test_lower_priority_hint_skips_scan_when_preferred_declines(monkeypatch):
    calls = []

    def preferred(args):
        calls.append("preferred")
        return None

    def fallback(args):
        calls.append("fallback")
        return args.q

    def no_scan(self):
        raise AssertionError("A compatible hint should skip the sanity check and scan")

    monkeypatch.setattr(dispatch, "attn_fns", [preferred, fallback])
    monkeypatch.setattr(dispatch.AttnArgs, "sanity_check", no_scan)
    hint = {"fn": fallback}
    _call(24, hint)
    assert calls == ["preferred", "fallback"]
    assert hint["fn"] is fallback
