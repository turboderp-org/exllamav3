"""Request lifetime, compact batch ordering and target-only routing offsets, checked without a GPU."""
import math
import random
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from exllamav3.generator.job import Job
from exllamav3.generator.generator import Generator
from exllamav3.model.model_tp import Model_TPMixin
from exllamav3.modules import BlockSparseMLP
from exllamav3.modules import block_sparse_mlp_routing as routing
from testlib.routing import make_cfg

pytestmark = pytest.mark.nogpu


def generator():
    gen = Generator.__new__(Generator)
    gen.expert_jitter_layout = {"layer0": 0, "layer1": 3}
    gen.expert_jitter_size = 7
    gen.pagetable = SimpleNamespace(max_pages = 16, all_pages = [])
    gen.max_draft_tokens = 0
    gen.max_chunk_size = 256
    gen.max_batch_size = 8
    gen.padded_vocab_size = 32
    gen.mtp_draft = gen.dflash_draft = False
    gen.ngram_match_min = gen.hybrid_draft_tokens = 0
    gen.ngram_corpus = gen.recurrent_cache = gen.draft_model = None
    gen.cache = gen.draft_cache = None
    gen.model = MagicMock(caps = {})
    return gen


def job(gen, j = 2.0, seed = 123):
    job = Job(torch.tensor([[1, 2, 3]]), max_new_tokens = 8, seed = seed, expert_jitter = j)
    job.prepare_for_queue(gen, 0)
    return job


@pytest.mark.parametrize("j", [0, 0.9, -1, float("inf"), float("-inf"), float("nan")])
def test_invalid_strength(j):
    with pytest.raises(ValueError, match = "expert_jitter"):
        Job(torch.tensor([[1]]), expert_jitter = j)


def test_seed_rng_and_disabled():
    gen = generator()
    a, b, c = job(gen), job(gen), job(gen, seed = 124)
    assert torch.equal(a.expert_jitter_offsets, b.expert_jitter_offsets)
    assert not torch.equal(a.expert_jitter_offsets, c.expert_jitter_offsets)
    assert a.rng.getstate() == random.Random(123).getstate()
    assert a.expert_jitter_offsets.shape == (1, 7)
    assert a.expert_jitter_offsets.dtype == torch.half
    assert (a.expert_jitter_offsets.float().abs() <= math.log(2) + 1e-3).all()
    off = job(gen, j = 1)
    assert off.expert_jitter_offsets is None
    assert gen.expert_jitter_params([off]) == {}
    assert torch.isfinite(job(gen, j = 1e300).expert_jitter_offsets).all()
    gen.expert_jitter_size = 0
    assert job(gen).expert_jitter_offsets is None  # dense target


def test_requeue_keeps_matrix():
    gen = generator()
    a = job(gen)
    offsets, state = a.expert_jitter_offsets, a.rng.getstate()
    a.sequences[0].sequence_ids.append(torch.tensor([[4]]))
    a.new_tokens = 1
    a.cached_pages = a.cached_tokens = 0
    a = a.prepare_for_requeue()
    assert a.expert_jitter == 2
    assert a.expert_jitter_offsets is offsets
    assert a.rng.getstate() == state


def test_layout_counts_full_expert_sets_once():
    gen = generator()
    gen.expert_jitter_layout, gen.expert_jitter_size = None, 0
    layers = [BlockSparseMLP.__new__(BlockSparseMLP) for _ in range(2)]
    for layer, key, count in zip(layers, ("layer0", "layer1"), (3, 4)):
        layer.key, layer.num_experts = key, count
    gen.model = [SimpleNamespace(), *layers, layers[0]]
    gen.prepare_expert_jitter_layout()
    gen.prepare_expert_jitter_layout()
    assert gen.expert_jitter_layout == {"layer0": 0, "layer1": 3}
    assert gen.expert_jitter_size == 7


@pytest.mark.parametrize("batch,tokens", [(1, 4), (3, 1), (3, 4)])
def test_layer_and_token_broadcast(monkeypatch, batch, tokens):
    gen = generator()
    enabled, disabled = job(gen), job(gen, j = 1)
    jobs = [enabled] if batch == 1 else [job(gen, seed = 124), disabled, enabled]
    params = gen.expert_jitter_params(jobs)
    cfg = make_cfg(16, 4, 2, "cpu")
    monkeypatch.setattr(routing, "_gate_t", lambda cfg: None)
    def project(y, gate, logits, *args):
        logits.zero_()
    monkeypatch.setattr(routing, "ext", SimpleNamespace(routing_project = project))
    logits, _, _ = routing._jitter_logits(batch * tokens, cfg,
                                          torch.zeros(batch * tokens, 16).half(), params["expert_jitter_offsets"][:, 3:])
    expected = torch.cat([torch.zeros(1, 4).half() if j is disabled else
                          j.expert_jitter_offsets[:, 3:] for j in jobs]).repeat_interleave(tokens, dim = 0)
    assert torch.equal(logits, expected)
    if batch == 3:
        assert not logits[tokens:2 * tokens].any()


@pytest.mark.parametrize("mtp", [False, True])
def test_prefill_target_only(mtp):
    gen = generator()
    gen.mtp_draft = mtp
    gen.draft_model = MagicMock(draft_verifier_params = {})
    def target_forward(input_ids, params):
        params["export_states"] = [torch.zeros(1, input_ids.shape[1], 4)]
    gen.model.forward.side_effect = target_forward
    a = job(gen)
    seq = a.sequences[0]
    seq.allocated_pages = [SimpleNamespace(kv_position = 0, sequence = torch.zeros(1, 256).long())]
    seq.block_index_tensor = torch.zeros(1, 1).int()
    a.cached_pages = a.cached_tokens = 0
    a.prefill([])
    target = gen.model.forward if mtp else gen.model.prefill
    assert target.call_args.kwargs["params"]["expert_jitter_offsets"] is a.expert_jitter_offsets
    assert "expert_jitter_offsets" not in gen.draft_model.prefill.call_args.kwargs["params"]


def test_tp_tensor_uses_arena():
    gen = generator()
    params = gen.expert_jitter_params([job(gen)])
    offsets = params["expert_jitter_offsets"]
    sender = SimpleNamespace(tp_producer = MagicMock())
    with patch.object(sender.tp_producer, "send", side_effect = lambda t: ("arena", id(t))):
        Model_TPMixin.prepare_inputs_for_tp(sender, torch.tensor([[1]]), params)
    assert params["expert_jitter_offsets"] == ("arena", id(offsets))
    assert params["expert_jitter_layout"] == gen.expert_jitter_layout


@pytest.mark.parametrize("speculative", [False, True])
def test_decode_compact_batch(speculative):
    gen = generator()
    gen.num_draft_tokens = 0
    gen.target_recurrent = False
    gen.draft_model = MagicMock(draft_verifier_params = {})
    ready, skipped, disabled, last = job(gen, seed = 124), job(gen, seed = 125), job(gen, j = 1), job(gen)
    gen.active_jobs = [ready, skipped, disabled, last]
    for a in (ready, disabled, last):
        a.time_first_token = 1.0  # avoid GPU timing synchronization
        seq = a.sequences[0]
        seq.prefill_complete = True
        seq.kv_position = 2
        seq.block_index_tensor = torch.zeros(1, 1).int()
        seq.allocated_pages = [SimpleNamespace(kv_position = 2, ref_count = 1,
                                              sequence = torch.zeros(1, 256).long())]
    gen._staging = lambda name, rows, width = None, dtype = torch.int32: torch.zeros(
        (rows,) if width is None else (rows, width), dtype = dtype)
    # Stop at the model boundary: exercise real input assembly without sampling or GPU execution.
    gen.model.forward.side_effect = RuntimeError("model boundary")
    with pytest.raises(RuntimeError, match = "model boundary"):
        gen.iterate_gen([], torch.tensor([[4, 5], [6, 7], [8, 9]]) if speculative else None)
    inputs = gen.model.forward.call_args.kwargs
    assert inputs["input_ids"].shape == (3, 3 if speculative else 1)
    offsets = inputs["params"]["expert_jitter_offsets"]
    assert torch.equal(offsets[0:1], ready.expert_jitter_offsets)
    assert not offsets[1].any()
    assert torch.equal(offsets[2:3], last.expert_jitter_offsets)
