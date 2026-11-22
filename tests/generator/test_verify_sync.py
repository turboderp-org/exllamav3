"""
The generator's wait for a pass's sampled tokens: while the GPU still runs the queued pass, the CPU MoE workers' hot
polling moves from the main model to the MTP head and the promote-on-miss decide runs; then only the compute stream
is waited on (the decide's transfers queue on the promotion stream). Exercised on a stand-in generator.
"""

from types import SimpleNamespace

import pytest
import torch

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


class _Reached(Exception):
    pass


def _speculative_generator(monkeypatch, job):
    """A stand-in generator whose iterate_gen runs up to the verify pass's readback (a draft window of 2)"""
    g = Generator.__new__(Generator)
    g.active_jobs = [job]
    g.model = SimpleNamespace(caps = (), forward = lambda input_ids, params: torch.zeros(1, 3, 8))
    g.draft_model = g.cache = None
    g.target_recurrent = False
    g.prev_ids = 0
    g.draft_sample_pinned = torch.empty((32, 8), dtype = torch.long)
    g._stage_arena = lambda rows, width, with_positions: (None, (None, None, None))
    g._upload_arena = lambda arena, rows, width, with_positions, params: None
    g._staging = lambda name, rows, width = None, dtype = torch.int32: torch.empty(rows, width, dtype = dtype)
    monkeypatch.setattr(gen_mod.ext, "generator_stage_decode", lambda *args: None)
    def reached(device):
        raise _Reached
    g._sync_verify_pass = reached
    return g


def _serial_job(**overrides):
    seq = SimpleNamespace(kv_position = 4, block_index_tensor = None, sequence_ids = [0] * 9)
    job = SimpleNamespace(
        sequences = [seq], embeddings = [], new_tokens = 5, min_new_tokens = 0, filters = [], filter_futures = [],
        logit_masks = [], checkpoint = None, forced_ids = None, time_first_token = 0.0,
        sampler = SimpleNamespace(reqs_past_ids = False),
        is_prefill_done = lambda: True, get_max_seq_len = lambda: 10,
        prepare_logit_mask = lambda: None, prepare_sampling_past_ids = lambda: None,
        get_input_ids_list = lambda draft_tokens, idx, add_to_cache: [torch.zeros(1, 3, dtype = torch.long)],
    )
    for k, v in overrides.items():
        setattr(job, k, v)
    return job


# Jobs the batched readback skips (each position is sampled after the previous one is accepted): the verify pass
# still gets its tick and hot handoff ahead of the serial walk
@pytest.mark.parametrize("overrides", [
    dict(checkpoint = object()),
    dict(sampler = SimpleNamespace(reqs_past_ids = True)),
    dict(min_new_tokens = 10),
], ids = ["banned_strings", "past_ids", "min_new_tokens"])
def test_serial_readback_still_syncs_the_verify_pass(monkeypatch, overrides):
    g = _speculative_generator(monkeypatch, _serial_job(**overrides))
    with pytest.raises(_Reached):
        g.iterate_gen([], torch.zeros(1, 2, dtype = torch.long))


class _PrefilledJob:
    sequences = [None]
    def is_prefill_done(self): return True
    def prefill(self, results, chunk_size = None): pass


def _mtp_generator(monkeypatch, calls, verify):
    """A stand-in MTP generator for one iterate(): `verify` stands in for iterate_gen"""
    g = Generator.__new__(Generator)
    g.pipeline = g.draft_cache = g.recurrent_cache = g.visualizer = g.sparse_head = g.disk_page_cache = None
    g.cache = SimpleNamespace(initialized = True)
    g.pagetable = SimpleNamespace(num_unreferenced_pages = lambda: 0)
    g.pending_jobs, g.active_jobs = [], [_PrefilledJob()]
    g.max_batch_size, g.max_chunk_size, g.shared_chunk_size = 16, 2048, 2048
    g.prefill_cooldown = g.prefill_cooldown_left = g.prefill_rotation = 0
    g.draft_model, g.mtp_draft, g.dflash_draft = object(), True, False
    g.num_draft_tokens, g.draft_max_rows, g.hybrid_draft_tokens, g.disk_cache_write_interval = 3, 0, 0, 0
    g.model = SimpleNamespace(config = SimpleNamespace())
    g.iterate_draftmodel_mtp_gen = lambda results, window: None
    g.iterate_gen = lambda results, draft_tokens = None: verify(g)
    monkeypatch.setattr(gen_mod, "set_moe_cpu_hot", lambda config, text, draft: calls.append((text, draft)))
    return g


def _verify(g, finishes):
    gen_mod.set_moe_cpu_hot(None, text = False, draft = True)       # the verify sync's handoff
    if finishes:
        g.active_jobs.clear()


def test_idle_generator_releases_hot_workers(monkeypatch):
    """The last verify sync leaves the MTP head's worker hot for a draft that never comes when the job ends there"""
    calls = []
    _mtp_generator(monkeypatch, calls, lambda g: _verify(g, finishes = True)).iterate()
    assert calls[-1] == (False, False)


def test_busy_generator_keeps_the_handoff(monkeypatch):
    calls = []
    _mtp_generator(monkeypatch, calls, lambda g: _verify(g, finishes = False)).iterate()
    assert calls[-1] == (False, True)
