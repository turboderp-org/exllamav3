"""
The generator's wait for a pass's sampled tokens: while the GPU still runs the queued pass, the CPU MoE workers' hot
polling moves from the main model to the MTP head and the promote-on-miss decide runs; then only the compute stream
is waited on (the decide's transfers queue on the promotion stream). Exercised on a stand-in generator.
"""

from types import SimpleNamespace

import pytest

import exllamav3.generator.generator as gen_mod
from exllamav3.generator.generator import Generator


def _record(monkeypatch):
    calls = []
    monkeypatch.setattr(gen_mod, "set_moe_cpu_hot", lambda config, text, draft: calls.append(("hot", text, draft)))
    monkeypatch.setattr(gen_mod, "promote_tick", lambda ip: calls.append("tick"))
    return calls


@pytest.mark.parametrize("mtp", [True, False])
def test_verify_sync_ticks_before_the_wait(monkeypatch, device, mtp):
    calls = _record(monkeypatch)
    stand_in = SimpleNamespace(model = SimpleNamespace(config = SimpleNamespace(infer_params = None)), mtp_draft = mtp)
    Generator._sync_verify_pass(stand_in, device)
    if mtp:
        assert calls == [("hot", False, False), "tick", ("hot", False, True)]
    else:
        assert calls == ["tick"]
