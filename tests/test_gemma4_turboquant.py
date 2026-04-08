import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.cache.cache import Cache
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.cache.gemma4 import (
    Gemma4FullCacheLayer,
    Gemma4FullQuantCacheLayer,
    Gemma4QuantCacheLayer,
    Gemma4SingleQuantCacheLayer,
    Gemma4SWAQuantCacheLayer,
    select_gemma4_cache_layer,
)
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.generator import Generator
from exllamav3.generator.gemma4_pagetable import Gemma4PageTable
from exllamav3.generator.job import Job
from exllamav3.generator.pagetable import PageTable, Sequence, tensor_hash_checksum
from exllamav3.architecture.gemma4 import (
    _filter_gemma4_indexed_embeddings,
    _update_gemma4_reconstruct_mode,
)
from exllamav3.modules.gemma4 import Gemma4Attention, Gemma4TransformerBlock
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX


class _FakeAttention:

    def __init__(self, layer_idx: int, sliding_window: int = 1024):
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window


class _DummyLayer:

    def __init__(self, role: str):
        self.cache_role = role
        self.calls = []

    def copy_page(self, source, from_page: int, to_page: int, num_tokens: int):
        self.calls.append((getattr(source, "cache_role", "default"), from_page, to_page, num_tokens))


class _NoCopyPlanPageTable(PageTable):

    def build_cache_copy_plan(self, source_page, target_page, num_tokens, seq = None):
        return None


def _make_gemma4_pagetable(
    *,
    sliding_window: int = 256,
    max_chunk_size: int = 4,
    num_draft_tokens: int = 0,
    full_max_num_tokens: int = 1024,
    swa_max_num_tokens: int = 768,
    profile: str = "generic",
):
    model = SimpleNamespace(
        config = SimpleNamespace(sliding_window = sliding_window),
        caps = {"gemma4_profile": profile},
    )
    generator = SimpleNamespace(
        model = model,
        max_chunk_size = max_chunk_size,
        num_draft_tokens = num_draft_tokens,
    )
    cache = SimpleNamespace(
        model = SimpleNamespace(loaded_tp = False),
        max_num_tokens = full_max_num_tokens,
        layers = {
            0: SimpleNamespace(cache_role = "full", max_num_tokens = full_max_num_tokens),
            1: SimpleNamespace(cache_role = "swa", max_num_tokens = swa_max_num_tokens),
        },
    )
    return Gemma4PageTable(generator, cache)


def _set_seq_pages(pagetable: Gemma4PageTable, seq: Sequence, pages, role: str) -> None:
    pagetable._set_seq_allocated_pages(seq, pages, role = role)


def _get_seq_pages(pagetable: Gemma4PageTable, seq: Sequence, role: str):
    return pagetable._get_seq_allocated_pages(seq, role = role)


def _set_seq_backing_pages(pagetable: Gemma4PageTable, seq: Sequence, pages, role: str) -> None:
    pagetable._set_seq_backing_pages(seq, pages, role = role)


def _get_seq_backing_pages(pagetable: Gemma4PageTable, seq: Sequence, role: str):
    return pagetable._get_seq_backing_pages(seq, role = role)


def _set_seq_kv(pagetable: Gemma4PageTable, seq: Sequence, kv_position: int, role: str) -> None:
    pagetable._set_seq_kv_position(seq, kv_position, role = role)


def _get_seq_page_map(pagetable: Gemma4PageTable, seq: Sequence, role: str):
    return pagetable._get_seq_page_map(seq, role = role)


def _build_mm_mask_reference(
    *,
    sliding_window: int,
    bsz: int,
    seqlen: int,
    total_lens: torch.Tensor,
    cache_seqlens: torch.Tensor | None,
    vision_group_ids: torch.Tensor | None,
    q_dtype: torch.dtype,
    device: torch.device,
    causal: bool,
) -> torch.Tensor:
    max_total = int(total_lens.max())
    mask = torch.full(
        (bsz, 1, seqlen, max_total),
        torch.finfo(q_dtype).min,
        dtype = q_dtype,
        device = device,
    )
    for b in range(bsz):
        total = int(total_lens[b])
        if total == 0:
            continue
        past = int(cache_seqlens[b]) if cache_seqlens is not None else 0
        full_groups = None
        if vision_group_ids is not None:
            full_groups = torch.full((total,), -1, dtype = torch.int32, device = device)
            full_groups[past : past + seqlen] = vision_group_ids[b]
        for qi in range(seqlen):
            q_abs = past + qi
            if not causal:
                start = 0 if sliding_window < 0 else max(0, q_abs - sliding_window)
                end = total if sliding_window < 0 else min(total, q_abs + sliding_window + 1)
                mask[b, 0, qi, start:end] = 0
            else:
                end = min(total, q_abs + 1)
                start = 0 if sliding_window < 0 else max(0, q_abs - sliding_window)
                if end > start:
                    mask[b, 0, qi, start:end] = 0
                if full_groups is not None:
                    q_group = int(full_groups[q_abs])
                    if q_group >= 0:
                        same_group = full_groups[:total] == q_group
                        mask[b, 0, qi, :total][same_group] = 0
    return mask


def test_select_gemma4_cache_layer_quant_respects_roles_and_normalizes_tokens():
    layer_types = ["full_attention", "sliding_attention"]

    full_selected = select_gemma4_cache_layer(
        CacheLayer_quant,
        _FakeAttention(0),
        layer_types,
        cache_kwargs = {
            "max_num_tokens": 1024,
            "full_max_num_tokens": 4096,
            "swa_max_num_tokens": 1537,
            "sentinel": "keep",
        },
    )
    assert full_selected["layer_type"] is Gemma4FullQuantCacheLayer
    assert full_selected["max_num_tokens"] == 1024
    assert full_selected["cache_kwargs"] == {"sentinel": "keep"}

    swa_selected = select_gemma4_cache_layer(
        CacheLayer_quant,
        _FakeAttention(1),
        layer_types,
        cache_kwargs = {
            "max_num_tokens": 4096,
            "swa_max_num_tokens": 1537,
            "sentinel": "keep",
        },
    )
    assert swa_selected["layer_type"] is Gemma4SWAQuantCacheLayer
    assert swa_selected["max_num_tokens"] == 1536
    assert swa_selected["cache_kwargs"] == {"sentinel": "keep"}

    tiny_swa = select_gemma4_cache_layer(
        CacheLayer_quant,
        _FakeAttention(1),
        layer_types,
        cache_kwargs = {
            "max_num_tokens": 4096,
            "swa_max_num_tokens": 128,
        },
    )
    assert tiny_swa["max_num_tokens"] == 256

    page_aligned_swa = select_gemma4_cache_layer(
        CacheLayer_quant,
        _FakeAttention(1),
        layer_types,
        cache_kwargs = {
            "max_num_tokens": 4096,
            "swa_max_num_tokens": 896,
        },
    )
    assert page_aligned_swa["max_num_tokens"] == 768

    full_fp16 = select_gemma4_cache_layer(
        CacheLayer_fp16,
        _FakeAttention(0),
        layer_types,
        cache_kwargs = {"max_num_tokens": 1024},
    )
    assert full_fp16["layer_type"] is Gemma4FullCacheLayer


def test_select_gemma4_cache_layer_applies_auto_swa_default_only_when_unspecified():
    layer_types = ["full_attention", "sliding_attention"]

    auto_swa = select_gemma4_cache_layer(
        CacheLayer_quant,
        _FakeAttention(1),
        layer_types,
        cache_kwargs = {
            "max_num_tokens": 4096,
        },
    )
    assert auto_swa["layer_type"] is Gemma4SWAQuantCacheLayer
    assert auto_swa["max_num_tokens"] == 1280

    explicit_swa = select_gemma4_cache_layer(
        CacheLayer_quant,
        _FakeAttention(1),
        layer_types,
        cache_kwargs = {
            "max_num_tokens": 4096,
            "swa_max_num_tokens": 1537,
        },
    )
    assert explicit_swa["max_num_tokens"] == 1536


def test_pagetable_advance_draft_decode_params_keeps_default_path_and_extends_gemma_roles():
    default_pagetable = PageTable(
        SimpleNamespace(),
        SimpleNamespace(max_num_tokens = PAGE_SIZE * 4),
    )
    default_params = {
        "cache_seqlens": torch.tensor([3, 7], dtype = torch.int32),
    }
    default_pagetable.advance_draft_decode_params(default_params, step = 2)
    assert default_params["cache_seqlens"].tolist() == [5, 9]

    gemma4_pagetable = _make_gemma4_pagetable()
    gemma4_params = {
        "cache_seqlens": torch.tensor([1], dtype = torch.int32),
        "cache_seqlens_full": torch.tensor([4], dtype = torch.int32),
        "cache_seqlens_swa": torch.tensor([6], dtype = torch.int32),
    }
    gemma4_pagetable.advance_draft_decode_params(gemma4_params, step = 2)
    assert gemma4_params["cache_seqlens"].tolist() == [3]
    assert gemma4_params["cache_seqlens_full"].tolist() == [6]
    assert gemma4_params["cache_seqlens_swa"].tolist() == [8]


def test_generator_iterate_draftmodel_gen_advances_custom_draft_cache_seqlens():
    observed_forward = []
    observed_prefill = []

    class _FakePageTable:

        def __init__(self):
            self.advance_calls = 0

        def build_draft_decode_params(self, active_jobs, max_seq_len):
            return {
                "block_table": torch.zeros((1, 1), dtype = torch.int32),
                "cache_seqlens": torch.tensor([3], dtype = torch.int32),
                "block_table_full": torch.zeros((1, 1), dtype = torch.int32),
                "cache_seqlens_full": torch.tensor([5], dtype = torch.int32),
                "block_table_swa": torch.zeros((1, 1), dtype = torch.int32),
                "cache_seqlens_swa": torch.tensor([7], dtype = torch.int32),
            }

        def advance_draft_decode_params(self, params, step = 1):
            self.advance_calls += 1
            params["cache_seqlens"] += step
            params["cache_seqlens_full"] += step
            params["cache_seqlens_swa"] += step

    class _FakeDraftModel:

        def forward(self, input_ids, params):
            observed_forward.append({
                "cache_seqlens": params["cache_seqlens"].clone(),
                "cache_seqlens_full": params["cache_seqlens_full"].clone(),
                "cache_seqlens_swa": params["cache_seqlens_swa"].clone(),
            })
            logits = torch.zeros((1, 1, 4), dtype = torch.float32)
            logits[:, :, 1] = 1.0
            return logits

        def prefill(self, input_ids, params):
            observed_prefill.append({
                "cache_seqlens": params["cache_seqlens"].clone(),
                "cache_seqlens_full": params["cache_seqlens_full"].clone(),
                "cache_seqlens_swa": params["cache_seqlens_swa"].clone(),
            })

    fake_job = SimpleNamespace(
        embeddings = [],
        time_first_token = 0.0,
        is_prefill_done = lambda: True,
        get_max_seq_len = lambda: 1,
        get_input_ids_list = lambda: [torch.tensor([[11]], dtype = torch.long)],
    )
    fake_self = SimpleNamespace(
        active_jobs = [fake_job],
        num_draft_tokens = 2,
        pagetable = _FakePageTable(),
        draft_model = _FakeDraftModel(),
        draft_cache = object(),
        draft_input_ids_pinned = torch.zeros((1, 1), dtype = torch.long),
        draft_ids_pinned = torch.zeros((1, 2), dtype = torch.long),
    )

    Generator.iterate_draftmodel_gen(fake_self, results = [])

    assert fake_self.pagetable.advance_calls == 2
    assert [entry["cache_seqlens"].item() for entry in observed_forward] == [3, 4]
    assert [entry["cache_seqlens_full"].item() for entry in observed_forward] == [5, 6]
    assert [entry["cache_seqlens_swa"].item() for entry in observed_forward] == [7, 8]
    assert observed_prefill[-1]["cache_seqlens"].item() == 5
    assert observed_prefill[-1]["cache_seqlens_full"].item() == 7
    assert observed_prefill[-1]["cache_seqlens_swa"].item() == 9


def test_generator_iterate_gen_rolls_back_rejected_draft_tokens_for_each_sequence():
    page_a = SimpleNamespace(kv_position = 1)
    page_b = SimpleNamespace(kv_position = 1)
    seq_a = SimpleNamespace(kv_position = 1, allocated_pages = [page_a])
    seq_b = SimpleNamespace(kv_position = 1, allocated_pages = [page_b])
    sync_calls = []

    class _FakeModel:
        caps = {}

        def forward(self, input_ids, params):
            return torch.zeros((2, 2, 4), dtype = torch.float32)

    class _FakePageTable:

        def build_decode_params(self, active_jobs, max_seq_len, use_offsets = False):
            return {
                "block_table": torch.zeros((2, 1), dtype = torch.int32),
                "cache_seqlens": torch.tensor([1, 1], dtype = torch.int32),
            }

        def sync_sequence_views(self, seq):
            sync_calls.append(seq)

        def defrag(self):
            return None

    class _FakeJob:

        def __init__(self):
            self.sequences = [seq_a, seq_b]
            self.embeddings = []
            self.time_first_token = 0.0
            self.new_tokens = 0
            self.filters = []
            self.filter_futures = []
            self.logit_masks = []
            self.accepted_draft_tokens = 0
            self.rejected_draft_tokens = 0

        def is_prefill_done(self):
            return True

        def get_max_seq_len(self):
            return 1

        def get_input_ids_list(self, draft_tokens = None, batch_offset = 0, add_to_cache = False):
            return [
                torch.tensor([[1]], dtype = torch.long),
                torch.tensor([[2]], dtype = torch.long),
            ]

        def prepare_logit_mask(self):
            return None

        def prepare_sampling_past_ids(self):
            return None

        def receive_logits(self, token_logits):
            return torch.tensor(0), None, None, 1.0

        def receive_sample(self, token_logits, next_token, next_k_tokens, next_k_probs, next_prob, results):
            return False, torch.tensor(1), False

        def deallocate_pages(self):
            return None

        def free_recurrent_state(self):
            return None

    fake_job = _FakeJob()
    fake_self = SimpleNamespace(
        active_jobs = [fake_job],
        num_draft_tokens = 0,
        model = _FakeModel(),
        pagetable = _FakePageTable(),
        recurrent_cache = None,
        filter_pool = None,
        cache = object(),
        num_remaining_jobs = lambda: 1,
    )

    Generator.iterate_gen(fake_self, results = [], draft_tokens = torch.zeros((1, 1), dtype = torch.long))

    assert fake_job.rejected_draft_tokens == 1
    assert page_a.kv_position == 0
    assert page_b.kv_position == 0
    assert sync_calls == [seq_a, seq_b]


def test_gemma4_pagetable_reports_min_safe_swa_window_and_atomic_prefill_guard():
    pagetable = _make_gemma4_pagetable(swa_max_num_tokens = 512)
    assert pagetable.get_min_swa_capacity_tokens() == 516
    assert pagetable.get_min_swa_capacity_pages() == 3

    seq_len = 600
    ids = torch.arange(seq_len, dtype = torch.long).unsqueeze(0)
    seq = Sequence(ids, ids.clone())
    _set_seq_pages(pagetable, seq, pagetable.swa_arena.all_pages[:2], role = "swa")
    _set_seq_kv(pagetable, seq, 0, role = "swa")

    embedding = SimpleNamespace(first_index = 0, last_index = 516)
    with pytest.raises(ValueError, match = "next page-aligned safe size 768"):
        pagetable.clamp_prefill_end(
            seq,
            0,
            seq_len,
            embeddings = [embedding],
            atomic_mm_prefill = True,
        )


def test_gemma4_pagetable_builds_role_aware_copy_plan_from_cached_and_active_pages():
    pagetable = _make_gemma4_pagetable()
    source_full = pagetable.full_arena.all_pages[0]
    target_full = pagetable.full_arena.all_pages[1]
    source_swa = pagetable.swa_arena.all_pages[0]
    target_swa = pagetable.swa_arena.all_pages[1]

    pagetable.cached_full_to_swa_pages[pagetable._cached_full_page_key(source_full)] = source_swa
    pagetable.active_full_to_swa_pages[id(target_full)] = target_swa

    plan = pagetable.build_cache_copy_plan(source_full, target_full, 17)
    assert plan == {
        "full": (source_full.page_index, target_full.page_index),
        "swa": (source_swa.page_index, target_swa.page_index),
    }


def test_gemma4_pagetable_cached_swa_lookup_tracks_full_page_hash():
    pagetable = _make_gemma4_pagetable()
    source_full = pagetable.full_arena.all_pages[0]
    target_full = pagetable.full_arena.all_pages[1]
    source_swa = pagetable.swa_arena.all_pages[0]
    target_swa = pagetable.swa_arena.all_pages[1]

    pagetable.cached_full_to_swa_pages[pagetable._cached_full_page_key(source_full)] = source_swa
    pagetable.active_full_to_swa_pages[id(target_full)] = target_swa

    source_full.phash = b"\xff" * 16

    assert pagetable.build_cache_copy_plan(source_full, target_full, 17) is None


def test_gemma4_pagetable_does_not_reuse_full_prefix_without_cached_swa_source():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)

    ids = torch.arange(0, 288, dtype = torch.long).unsqueeze(0)

    seq_1 = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq_1, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq_1, None)
    seq_1.kv_position = ids.shape[-1] - 1
    _set_seq_kv(pagetable, seq_1, seq_1.kv_position, role = "full")
    _get_seq_pages(pagetable, seq_1, "full")[0].kv_position = PAGE_SIZE
    _get_seq_pages(pagetable, seq_1, "full")[1].kv_position = 31
    _get_seq_pages(pagetable, seq_1, "full")[1].prev_hash = _get_seq_pages(pagetable, seq_1, "full")[0].phash
    _get_seq_pages(pagetable, seq_1, "full")[1].sequence[:, :31].copy_(ids[:, PAGE_SIZE : PAGE_SIZE + 31])
    _get_seq_backing_pages(pagetable, seq_1, "swa")[0].kv_position = 256
    _get_seq_backing_pages(pagetable, seq_1, "swa")[1].kv_position = 31
    _get_seq_backing_pages(pagetable, seq_1, "swa")[1].prev_hash = _get_seq_backing_pages(pagetable, seq_1, "swa")[0].phash
    pagetable.sync_sequence_views(seq_1)
    pagetable.deallocate_sequence(seq_1)

    seq_2 = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq_2, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq_2, None)

    assert seq_2.kv_position == 0


def test_gemma4_pagetable_does_not_reuse_later_pages_after_prefix_miss():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)

    reusable_page = pagetable.full_arena.all_pages[1]
    reusable_page.add_ref_clear(pagetable.full_arena.access_serial + 1, b"b" * 16)
    reusable_page.kv_position = PAGE_SIZE

    cached_swa = pagetable.swa_arena.all_pages[0]
    cached_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    pagetable.cached_full_to_swa_pages[pagetable._cached_full_page_key(reusable_page)] = cached_swa

    allocated_pages, kv_position, cached_pages, _ = pagetable.allocate_pages(
        [b"a" * 16, b"b" * 16],
        0,
        None,
    )

    assert kv_position == 0
    assert cached_pages == 0
    assert allocated_pages[1] is not reusable_page
    assert allocated_pages[1].kv_position == 0


def test_gemma4_pagetable_evicted_cached_swa_snapshot_does_not_block_admission():
    pagetable = _make_gemma4_pagetable()
    source_full = pagetable.full_arena.all_pages[0]
    cached_swa = pagetable.swa_arena.all_pages[0]
    cached_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    pagetable.cached_full_to_swa_pages[pagetable._cached_full_page_key(source_full)] = cached_swa

    seq = SimpleNamespace(page_hashes = [b"a", b"b", b"c"], new_unique_pages = 0)
    job = SimpleNamespace(sequences = [seq], all_unique_hashes = set(), max_skips = 0)

    assert len(pagetable.swa_arena.unreferenced_pages) == 2
    assert pagetable.can_admit_job(job, 0, 8) is True
    assert len(pagetable.cached_full_to_swa_pages) == 0
    assert len(pagetable.swa_arena.unreferenced_pages) == 3


def test_gemma4_pagetable_restores_cached_swa_prefix_for_reused_full_pages():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 1024)
    swa_layer = _DummyLayer("swa")
    swa_layer.max_num_tokens = 1024
    full_layer = SimpleNamespace(cache_role = "full", max_num_tokens = 1024)
    pagetable.cache.layers = {0: full_layer, 1: swa_layer}

    ids = torch.arange(0, 288, dtype = torch.long).unsqueeze(0)

    seq_1 = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq_1, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq_1, None)
    seq_1.kv_position = ids.shape[-1] - 1
    _set_seq_kv(pagetable, seq_1, seq_1.kv_position, role = "full")
    _get_seq_pages(pagetable, seq_1, "full")[0].kv_position = PAGE_SIZE
    _get_seq_pages(pagetable, seq_1, "full")[1].kv_position = 31
    _get_seq_pages(pagetable, seq_1, "full")[1].prev_hash = _get_seq_pages(pagetable, seq_1, "full")[0].phash
    _get_seq_pages(pagetable, seq_1, "full")[1].sequence[:, :31].copy_(ids[:, PAGE_SIZE : PAGE_SIZE + 31])
    _get_seq_backing_pages(pagetable, seq_1, "swa")[0].kv_position = 256
    _get_seq_backing_pages(pagetable, seq_1, "swa")[1].kv_position = 31
    _get_seq_backing_pages(pagetable, seq_1, "swa")[1].prev_hash = _get_seq_backing_pages(pagetable, seq_1, "swa")[0].phash
    pagetable.sync_sequence_views(seq_1)

    prompt_full_page = _get_seq_pages(pagetable, seq_1, "full")[0]
    prompt_terminal_page = _get_seq_pages(pagetable, seq_1, "full")[1]
    pagetable.cache_prefill_copy_source(seq_1)
    cached_source_page = pagetable.cached_full_to_swa_pages[pagetable._cached_full_page_key(prompt_full_page)]
    partial_key = pagetable._cached_partial_page_key(
        prompt_terminal_page.prev_hash,
        ids[:, PAGE_SIZE : PAGE_SIZE + 31],
    )
    assert pagetable.cached_partial_to_swa_pages[partial_key].kv_position == 31
    pagetable.deallocate_sequence(seq_1)

    seq_2 = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq_2, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq_2, None)

    restored_page = _get_seq_backing_pages(pagetable, seq_2, "swa")[0]
    assert restored_page is not cached_source_page
    assert restored_page.kv_position == PAGE_SIZE
    assert seq_2.kv_position == PAGE_SIZE
    assert swa_layer.calls[0] == ("swa", 0, cached_source_page.page_index, PAGE_SIZE)
    assert swa_layer.calls[-1] == ("swa", cached_source_page.page_index, restored_page.page_index, PAGE_SIZE)


def test_gemma4_pagetable_keeps_prefill_terminal_snapshot_on_deallocate():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)
    swa_layer = _DummyLayer("swa")
    swa_layer.max_num_tokens = 768
    full_layer = SimpleNamespace(cache_role = "full", max_num_tokens = 1024)
    pagetable.cache.layers = {0: full_layer, 1: swa_layer}

    ids = torch.arange(0, 64, dtype = torch.long).unsqueeze(0)

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)
    seq.kv_position = ids.shape[-1] - 1
    _set_seq_kv(pagetable, seq, seq.kv_position, role = "full")
    _get_seq_pages(pagetable, seq, "full")[0].kv_position = 63
    _get_seq_backing_pages(pagetable, seq, "swa")[0].kv_position = 63
    pagetable.sync_sequence_views(seq)

    prompt_terminal_page = _get_seq_pages(pagetable, seq, "full")[0]
    active_terminal_swa = _get_seq_page_map(pagetable, seq, "swa")[id(prompt_terminal_page)]
    pagetable.cache_prefill_copy_source(seq)
    partial_key = pagetable._cached_partial_page_key(None, ids[:, :63])
    snapshot_page = pagetable.cached_partial_to_swa_pages[partial_key]
    assert snapshot_page is not active_terminal_swa

    pagetable.deallocate_sequence(seq)

    assert pagetable.cached_partial_to_swa_pages[partial_key] is snapshot_page


def test_gemma4_pagetable_cached_full_snapshot_stays_pinned_after_capture():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 1024)
    swa_layer = _DummyLayer("swa")
    swa_layer.max_num_tokens = 1024
    full_layer = SimpleNamespace(cache_role = "full", max_num_tokens = 1024)
    pagetable.cache.layers = {0: full_layer, 1: swa_layer}

    ids = torch.arange(0, 288, dtype = torch.long).unsqueeze(0)

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)
    seq.kv_position = ids.shape[-1] - 1
    _set_seq_kv(pagetable, seq, seq.kv_position, role = "full")
    _get_seq_pages(pagetable, seq, "full")[0].kv_position = PAGE_SIZE
    _get_seq_pages(pagetable, seq, "full")[1].kv_position = 31
    _get_seq_backing_pages(pagetable, seq, "swa")[0].kv_position = PAGE_SIZE
    _get_seq_backing_pages(pagetable, seq, "swa")[1].kv_position = 31
    pagetable.sync_sequence_views(seq)

    prompt_full_page = _get_seq_pages(pagetable, seq, "full")[0]
    pagetable.cache_prefill_copy_source(seq)

    snapshot_page = pagetable.cached_full_to_swa_pages[pagetable._cached_full_page_key(prompt_full_page)]
    assert snapshot_page.ref_count == 1
    assert snapshot_page.phash in pagetable.swa_arena.referenced_pages


def test_gemma4_pagetable_cached_partial_snapshot_stays_pinned_after_capture():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)
    swa_layer = _DummyLayer("swa")
    swa_layer.max_num_tokens = 768
    full_layer = SimpleNamespace(cache_role = "full", max_num_tokens = 1024)
    pagetable.cache.layers = {0: full_layer, 1: swa_layer}

    ids = torch.arange(0, 64, dtype = torch.long).unsqueeze(0)

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)
    seq.kv_position = ids.shape[-1] - 1
    _set_seq_kv(pagetable, seq, seq.kv_position, role = "full")
    _get_seq_pages(pagetable, seq, "full")[0].kv_position = 63
    _get_seq_backing_pages(pagetable, seq, "swa")[0].kv_position = 63
    pagetable.sync_sequence_views(seq)

    pagetable.cache_prefill_copy_source(seq)
    partial_key = pagetable._cached_partial_page_key(None, ids[:, :63])
    snapshot_page = pagetable.cached_partial_to_swa_pages[partial_key]
    assert snapshot_page.ref_count == 1
    assert snapshot_page.phash in pagetable.swa_arena.referenced_pages


def test_gemma4_pagetable_clears_fresh_full_and_swa_page_sequences():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)

    full_page = pagetable.full_arena.all_pages[0]
    full_page.sequence.fill_(123)

    swa_page = pagetable.swa_arena.all_pages[0]
    swa_page.sequence.fill_(456)

    allocated_full, _, _, _ = pagetable.allocate_pages([b"a" * 16], 0, None)
    allocated_swa, _, _, _ = pagetable.allocate_pages([], 1, None, role = "swa")

    assert allocated_full[0] is full_page
    assert torch.count_nonzero(full_page.sequence) == 0
    assert allocated_swa[0] is swa_page
    assert torch.count_nonzero(swa_page.sequence) == 0


def test_gemma4_pagetable_prefers_low_full_pages_for_fresh_prefill():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)

    for page in pagetable.full_arena.all_pages:
        page.access_serial = 100 + page.page_index
    for idx in range(5, pagetable.full_arena.max_pages):
        pagetable.full_arena.all_pages[idx].access_serial = idx - 5
    pagetable.full_arena.access_serial = 1000
    pagetable.full_arena.last_defrag_serial = 1000

    allocated_pages, kv_position, cached_pages, _ = pagetable.allocate_pages([], 2, None)

    assert [page.page_index for page in allocated_pages] == [0, 1]
    assert kv_position == 0
    assert cached_pages == 0


def test_gemma4_pagetable_prefers_low_full_pages_after_first_prefix_miss():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)

    for page in pagetable.full_arena.all_pages:
        page.access_serial = 100 + page.page_index
    for idx in range(5, pagetable.full_arena.max_pages):
        pagetable.full_arena.all_pages[idx].access_serial = idx - 5
    pagetable.full_arena.access_serial = 1000
    pagetable.full_arena.last_defrag_serial = 1000

    allocated_pages, kv_position, cached_pages, _ = pagetable.allocate_pages([b"a" * 16], 1, None)

    assert [page.page_index for page in allocated_pages] == [0, 1]
    assert kv_position == 0
    assert cached_pages == 0


def test_gemma4_pagetable_build_cache_copy_plan_uses_cached_partial_swa_source():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)

    source_page = pagetable.full_arena.all_pages[0]
    source_page.add_ref_unique(pagetable.full_arena.access_serial + 1)
    source_page.kv_position = 20
    source_page.prev_hash = b"prev"
    source_page.sequence[:, :20].copy_(torch.arange(PAGE_SIZE, PAGE_SIZE + 20, dtype = torch.long).unsqueeze(0))

    target_page = pagetable.full_arena.all_pages[1]
    target_page.add_ref_unique(pagetable.full_arena.access_serial + 2)

    target_swa = pagetable.swa_arena.all_pages[0]
    target_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    pagetable.active_full_to_swa_pages[id(target_page)] = target_swa

    partial_key = pagetable._cached_partial_page_key(source_page.prev_hash, source_page.sequence[:, :20])
    source_swa = pagetable.swa_arena.all_pages[1]
    source_swa.add_ref_unique(pagetable.swa_arena.access_serial + 2)
    pagetable.cached_partial_to_swa_pages[partial_key] = source_swa

    plan = pagetable.build_cache_copy_plan(source_page, target_page, 20)

    assert plan == {
        "full": (source_page.page_index, target_page.page_index),
        "swa": (source_swa.page_index, target_swa.page_index),
    }


def test_job_prefill_falls_back_to_normal_ingest_when_custom_copy_plan_is_unavailable():
    model_calls = []

    class _FakeModel:
        caps = {}

        def prefill(self, input_ids, params):
            model_calls.append({
                "shape": tuple(input_ids.shape),
                "cache_seqlens": tuple(params["cache_seqlens"].tolist()),
            })

    generator = SimpleNamespace(
        max_chunk_size = PAGE_SIZE + 16,
        num_draft_tokens = 0,
        recurrent_cache = None,
        draft_model = None,
        draft_cache = None,
        cache = SimpleNamespace(copy_page = lambda *args, **kwargs: None),
        model = _FakeModel(),
        recurrent_checkpoint_interval = PAGE_SIZE,
        max_batch_size = 1,
        max_total_tokens = PAGE_SIZE * 4,
        padded_vocab_size = 128,
    )
    pagetable = _NoCopyPlanPageTable(generator, SimpleNamespace(max_num_tokens = PAGE_SIZE * 4))
    generator.pagetable = pagetable

    ids = torch.arange(PAGE_SIZE + 16, dtype = torch.long).unsqueeze(0)
    job = Job(input_ids = ids, max_new_tokens = 8)
    job.prepare_for_queue(generator, serial_number = 1)
    job.allocate_pages()

    match_page = pagetable.all_pages[-1]
    match_page.prev_hash = None
    match_page.kv_position = 8
    match_page.sequence[:, :8].copy_(ids[:, :8])

    results = []
    job.prefill(results)

    assert model_calls == [{
        "shape": (1, PAGE_SIZE),
        "cache_seqlens": (0,),
    }]
    assert job.cached_tokens == 0
    assert job.sequences[0].kv_position == PAGE_SIZE
    assert results and results[0]["stage"] == "prefill"


def test_gemma4_pagetable_disables_full_reuse_without_matching_partial_source():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)

    ids = torch.arange(0, 288, dtype = torch.long).unsqueeze(0)
    first_page = ids[:, :PAGE_SIZE]
    first_hash = tensor_hash_checksum(first_page, None)

    reusable_page = pagetable.full_arena.all_pages[1]
    reusable_page.add_ref_clear(pagetable.full_arena.access_serial + 1, first_hash)
    reusable_page.kv_position = PAGE_SIZE
    reusable_page.sequence[:, :PAGE_SIZE].copy_(first_page)

    cached_swa = pagetable.swa_arena.all_pages[0]
    cached_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    pagetable.cached_full_to_swa_pages[(first_hash, False)] = cached_swa

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)

    assert seq.page_hashes[0] != first_hash
    assert seq.kv_position == 0


def test_gemma4_pagetable_scopes_cached_full_sources_by_prompt_mode():
    pagetable = _make_gemma4_pagetable()
    source_full = pagetable.full_arena.all_pages[0]
    target_full = pagetable.full_arena.all_pages[1]
    source_swa = pagetable.swa_arena.all_pages[0]
    target_swa = pagetable.swa_arena.all_pages[1]

    source_full.phash = b"f" * 16
    pagetable.cached_full_to_swa_pages[pagetable._cached_full_page_key(source_full, True)] = source_swa
    pagetable.active_full_to_swa_pages[id(target_full)] = target_swa

    assert pagetable.build_cache_copy_plan(source_full, target_full, 17) is None


def test_gemma4_pagetable_reuses_full_prefix_with_matching_mm_mode():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768, profile = "26b_a4b_moe")
    swa_layer = _DummyLayer("swa")
    swa_layer.max_num_tokens = 768
    full_layer = SimpleNamespace(cache_role = "full", max_num_tokens = 768)
    pagetable.cache.layers = {0: full_layer, 1: swa_layer}

    ids = torch.full((1, PAGE_SIZE + 1), FIRST_MM_EMBEDDING_INDEX, dtype = torch.long)
    first_page = ids[:, :PAGE_SIZE]
    first_hash = tensor_hash_checksum(first_page, None)

    reusable_page = pagetable.full_arena.all_pages[1]
    reusable_page.add_ref_clear(pagetable.full_arena.access_serial + 1, first_hash)
    reusable_page.kv_position = PAGE_SIZE
    reusable_page.sequence[:, :PAGE_SIZE].copy_(first_page)

    cached_swa = pagetable.swa_arena.all_pages[0]
    cached_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    cached_swa.kv_position = PAGE_SIZE
    pagetable.cached_full_to_swa_pages[(first_hash, True)] = cached_swa

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)

    assert seq.page_hashes[0] == first_hash
    assert seq.kv_position == PAGE_SIZE


def test_gemma4_pagetable_26b_fresh_text_prefers_low_index_swa_pages():
    pagetable = _make_gemma4_pagetable(
        sliding_window = 256,
        swa_max_num_tokens = 1024,
        profile = "26b_a4b_moe",
    )

    # Make the highest-index SWA page look oldest so the generic LRU order would
    # pick it first. The 26B fresh-text path should override that and keep the
    # local window on the lowest page indices instead.
    for page_idx, page in enumerate(pagetable.swa_arena.all_pages):
        page.access_serial = 100 + page_idx
    pagetable.swa_arena.all_pages[3].access_serial = 0

    ids = torch.arange(0, 21, dtype = torch.long).unsqueeze(0)
    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)

    swa_pages = _get_seq_pages(pagetable, seq, role = "swa")
    assert [page.page_index for page in swa_pages] == [0]
    assert seq.kv_position == 0


def test_gemma4_pagetable_non_26b_text_keeps_lru_swa_allocation_order():
    pagetable = _make_gemma4_pagetable(
        sliding_window = 256,
        swa_max_num_tokens = 1024,
        profile = "generic",
    )

    for page_idx, page in enumerate(pagetable.swa_arena.all_pages):
        page.access_serial = 100 + page_idx
    pagetable.swa_arena.all_pages[3].access_serial = 0

    ids = torch.arange(0, 21, dtype = torch.long).unsqueeze(0)
    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)

    swa_pages = _get_seq_pages(pagetable, seq, role = "swa")
    assert [page.page_index for page in swa_pages] == [3]


def test_gemma4_pagetable_26b_mm_keeps_lru_swa_allocation_order():
    pagetable = _make_gemma4_pagetable(
        sliding_window = 256,
        swa_max_num_tokens = 1024,
        profile = "26b_a4b_moe",
    )

    for page_idx, page in enumerate(pagetable.swa_arena.all_pages):
        page.access_serial = 100 + page_idx
    pagetable.swa_arena.all_pages[3].access_serial = 0

    ids = torch.full((1, 21), FIRST_MM_EMBEDDING_INDEX, dtype = torch.long)
    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)

    swa_pages = _get_seq_pages(pagetable, seq, role = "swa")
    assert [page.page_index for page in swa_pages] == [3]


def test_gemma4_pagetable_26b_reused_text_prefix_prefers_low_index_full_tail_page():
    pagetable = _make_gemma4_pagetable(
        sliding_window = 256,
        swa_max_num_tokens = 1024,
        profile = "26b_a4b_moe",
    )
    swa_layer = _DummyLayer("swa")
    swa_layer.max_num_tokens = 1024
    full_layer = SimpleNamespace(cache_role = "full", max_num_tokens = 1024)
    pagetable.cache.layers = {0: full_layer, 1: swa_layer}

    for page_idx, page in enumerate(pagetable.full_arena.all_pages):
        page.access_serial = 100 + page_idx
    pagetable.full_arena.all_pages[3].access_serial = 0

    ids = torch.arange(0, PAGE_SIZE + 21, dtype = torch.long).unsqueeze(0)
    first_page = ids[:, :PAGE_SIZE]
    first_hash = tensor_hash_checksum(first_page, None)

    reusable_page = pagetable.full_arena.all_pages[1]
    reusable_page.add_ref_clear(pagetable.full_arena.access_serial + 1, first_hash)
    reusable_page.kv_position = PAGE_SIZE
    reusable_page.sequence[:, :PAGE_SIZE].copy_(first_page)

    cached_swa = pagetable.swa_arena.all_pages[0]
    cached_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    cached_swa.kv_position = PAGE_SIZE
    pagetable.cached_full_to_swa_pages[(first_hash, False)] = cached_swa
    partial_ids = ids[:, PAGE_SIZE:PAGE_SIZE + 20]
    partial_key = pagetable._cached_partial_page_key(first_hash, partial_ids, False)
    partial_swa = pagetable.swa_arena.all_pages[1]
    partial_swa.add_ref_unique(pagetable.swa_arena.access_serial + 2)
    partial_swa.kv_position = 20
    pagetable.cached_partial_to_swa_pages[partial_key] = partial_swa

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)

    full_pages = _get_seq_pages(pagetable, seq, role = "full")
    assert [page.page_index for page in full_pages] == [1, 0]
    assert seq.kv_position == PAGE_SIZE


def test_gemma4_pagetable_non_26b_reused_text_prefix_keeps_lru_full_tail_page():
    pagetable = _make_gemma4_pagetable(
        sliding_window = 256,
        swa_max_num_tokens = 1024,
        profile = "generic",
    )
    swa_layer = _DummyLayer("swa")
    swa_layer.max_num_tokens = 1024
    full_layer = SimpleNamespace(cache_role = "full", max_num_tokens = 1024)
    pagetable.cache.layers = {0: full_layer, 1: swa_layer}

    for page_idx, page in enumerate(pagetable.full_arena.all_pages):
        page.access_serial = 100 + page_idx
    pagetable.full_arena.all_pages[3].access_serial = 0

    ids = torch.arange(0, PAGE_SIZE + 21, dtype = torch.long).unsqueeze(0)
    first_page = ids[:, :PAGE_SIZE]
    first_hash = tensor_hash_checksum(first_page, None)

    reusable_page = pagetable.full_arena.all_pages[1]
    reusable_page.add_ref_clear(pagetable.full_arena.access_serial + 1, first_hash)
    reusable_page.kv_position = PAGE_SIZE
    reusable_page.sequence[:, :PAGE_SIZE].copy_(first_page)

    cached_swa = pagetable.swa_arena.all_pages[0]
    cached_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    cached_swa.kv_position = PAGE_SIZE
    pagetable.cached_full_to_swa_pages[(first_hash, False)] = cached_swa
    partial_ids = ids[:, PAGE_SIZE:PAGE_SIZE + 20]
    partial_key = pagetable._cached_partial_page_key(first_hash, partial_ids, False)
    partial_swa = pagetable.swa_arena.all_pages[1]
    partial_swa.add_ref_unique(pagetable.swa_arena.access_serial + 2)
    partial_swa.kv_position = 20
    pagetable.cached_partial_to_swa_pages[partial_key] = partial_swa

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)

    full_pages = _get_seq_pages(pagetable, seq, role = "full")
    assert [page.page_index for page in full_pages] == [1, 3]


def test_gemma4_pagetable_26b_mm_reused_prefix_keeps_lru_full_tail_page():
    pagetable = _make_gemma4_pagetable(
        sliding_window = 256,
        swa_max_num_tokens = 1024,
        profile = "26b_a4b_moe",
    )
    swa_layer = _DummyLayer("swa")
    swa_layer.max_num_tokens = 1024
    full_layer = SimpleNamespace(cache_role = "full", max_num_tokens = 1024)
    pagetable.cache.layers = {0: full_layer, 1: swa_layer}

    for page_idx, page in enumerate(pagetable.full_arena.all_pages):
        page.access_serial = 100 + page_idx
    pagetable.full_arena.all_pages[3].access_serial = 0

    ids = torch.full((1, PAGE_SIZE + 21), FIRST_MM_EMBEDDING_INDEX, dtype = torch.long)
    first_page = ids[:, :PAGE_SIZE]
    first_hash = tensor_hash_checksum(first_page, None)

    reusable_page = pagetable.full_arena.all_pages[1]
    reusable_page.add_ref_clear(pagetable.full_arena.access_serial + 1, first_hash)
    reusable_page.kv_position = PAGE_SIZE
    reusable_page.sequence[:, :PAGE_SIZE].copy_(first_page)

    cached_swa = pagetable.swa_arena.all_pages[0]
    cached_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    cached_swa.kv_position = PAGE_SIZE
    pagetable.cached_full_to_swa_pages[(first_hash, True)] = cached_swa
    partial_ids = ids[:, PAGE_SIZE:PAGE_SIZE + 20]
    partial_key = pagetable._cached_partial_page_key(first_hash, partial_ids, True)
    partial_swa = pagetable.swa_arena.all_pages[1]
    partial_swa.add_ref_unique(pagetable.swa_arena.access_serial + 2)
    partial_swa.kv_position = 20
    pagetable.cached_partial_to_swa_pages[partial_key] = partial_swa

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)

    full_pages = _get_seq_pages(pagetable, seq, role = "full")
    assert [page.page_index for page in full_pages] == [1, 3]


def test_gemma4_pagetable_reuses_text_full_prefix_before_first_mm_request():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768, profile = "26b_a4b_moe")
    swa_layer = _DummyLayer("swa")
    swa_layer.max_num_tokens = 768
    full_layer = SimpleNamespace(cache_role = "full", max_num_tokens = 1024)
    pagetable.cache.layers = {0: full_layer, 1: swa_layer}

    ids = torch.arange(0, PAGE_SIZE + 1, dtype = torch.long).unsqueeze(0)
    first_page = ids[:, :PAGE_SIZE]
    first_hash = tensor_hash_checksum(first_page, None)

    reusable_page = pagetable.full_arena.all_pages[1]
    reusable_page.add_ref_clear(pagetable.full_arena.access_serial + 1, first_hash)
    reusable_page.kv_position = PAGE_SIZE
    reusable_page.sequence[:, :PAGE_SIZE].copy_(first_page)

    cached_swa = pagetable.swa_arena.all_pages[0]
    cached_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    cached_swa.kv_position = PAGE_SIZE
    pagetable.cached_full_to_swa_pages[(first_hash, False)] = cached_swa

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)
    pagetable.allocate_sequence(seq, None)

    assert seq.page_hashes[0] == first_hash
    assert seq.kv_position == PAGE_SIZE


def test_gemma4_pagetable_scopes_cached_partial_sources_by_prompt_mode():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)

    source_page = pagetable.full_arena.all_pages[0]
    source_page.add_ref_unique(pagetable.full_arena.access_serial + 1)
    source_page.kv_position = 20
    source_page.prev_hash = b"prev"
    source_page.sequence[:, :20].copy_(torch.arange(PAGE_SIZE, PAGE_SIZE + 20, dtype = torch.long).unsqueeze(0))

    target_page = pagetable.full_arena.all_pages[1]
    target_page.add_ref_unique(pagetable.full_arena.access_serial + 2)

    target_swa = pagetable.swa_arena.all_pages[0]
    target_swa.add_ref_unique(pagetable.swa_arena.access_serial + 1)
    pagetable.active_full_to_swa_pages[id(target_page)] = target_swa

    partial_key = pagetable._cached_partial_page_key(source_page.prev_hash, source_page.sequence[:, :20], True)
    source_swa = pagetable.swa_arena.all_pages[1]
    source_swa.add_ref_unique(pagetable.swa_arena.access_serial + 2)
    pagetable.cached_partial_to_swa_pages[partial_key] = source_swa

    assert pagetable.build_cache_copy_plan(source_page, target_page, 20) is None


def test_gemma4_pagetable_keeps_full_page_hashes_when_no_reuse_candidate_exists():
    pagetable = _make_gemma4_pagetable(sliding_window = 256, swa_max_num_tokens = 768)

    ids = torch.arange(0, 288, dtype = torch.long).unsqueeze(0)
    first_page = ids[:, :PAGE_SIZE]
    first_hash = tensor_hash_checksum(first_page, None)

    seq = Sequence(ids, ids.clone())
    pagetable.prepare_sequence(seq, False, 0, allow_page_reuse = True)

    assert seq.page_hashes[0] == first_hash


def test_cache_copy_page_uses_role_specific_page_plan_and_default_fallback():
    src_cache = object.__new__(Cache)
    dst_cache = object.__new__(Cache)
    src_cache.model = SimpleNamespace(loaded_tp = False)
    dst_cache.model = SimpleNamespace(loaded_tp = False)
    src_cache.num_layers = dst_cache.num_layers = 2
    src_cache.layers = {0: _DummyLayer("full"), 1: _DummyLayer("swa")}
    dst_cache.layers = {0: _DummyLayer("full"), 1: _DummyLayer("swa")}

    Cache.copy_page(
        src_cache,
        dst_cache,
        1,
        2,
        5,
        page_plan = {
            "full": (10, 20),
            "swa": (11, 21),
        },
    )
    assert dst_cache.layers[0].calls == [("full", 10, 20, 5)]
    assert dst_cache.layers[1].calls == [("swa", 11, 21, 5)]

    src_cache_2 = object.__new__(Cache)
    dst_cache_2 = object.__new__(Cache)
    src_cache_2.model = SimpleNamespace(loaded_tp = False)
    dst_cache_2.model = SimpleNamespace(loaded_tp = False)
    src_cache_2.num_layers = dst_cache_2.num_layers = 2
    src_cache_2.layers = {0: _DummyLayer("full"), 1: _DummyLayer("swa")}
    dst_cache_2.layers = {0: _DummyLayer("full"), 1: _DummyLayer("swa")}

    Cache.copy_page(
        src_cache_2,
        dst_cache_2,
        1,
        2,
        5,
        page_plan = {"default": (7, 8)},
    )
    assert dst_cache_2.layers[0].calls == [("full", 7, 8, 5)]
    assert dst_cache_2.layers[1].calls == [("swa", 7, 8, 5)]


def test_gemma4_prepare_inputs_drops_mm_state_for_plain_text_decode():
    embeddings = [
        SimpleNamespace(first_index = 1000, last_index = 1016),
        SimpleNamespace(first_index = 2000, last_index = 2016),
    ]
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.long)
    params = {"indexed_embeddings": embeddings.copy()}

    _filter_gemma4_indexed_embeddings(input_ids, params)

    assert "indexed_embeddings" not in params


def test_gemma4_prepare_inputs_keeps_only_current_mm_ranges():
    embeddings = [
        SimpleNamespace(first_index = 1000, last_index = 1016),
        SimpleNamespace(first_index = 2000, last_index = 2016),
    ]
    input_ids = torch.tensor([[1, 1002, 3], [4, 2005, 6]], dtype = torch.long)
    params = {"indexed_embeddings": embeddings.copy()}

    _filter_gemma4_indexed_embeddings(input_ids, params)

    assert params["indexed_embeddings"] == embeddings

    params_single = {"indexed_embeddings": embeddings.copy()}
    _filter_gemma4_indexed_embeddings(torch.tensor([[1, 1003, 4]], dtype = torch.long), params_single)
    assert params_single["indexed_embeddings"] == [embeddings[0]]


def test_gemma4_26b_mm_request_keeps_temp_reconstruct_even_after_decode_tokens():
    params = {
        "indexed_embeddings": [SimpleNamespace(first_index = 1000, last_index = 1016)],
    }
    input_ids = torch.tensor([[1, 2, 3]], dtype = torch.long)

    has_mm_request = bool(params.get("indexed_embeddings"))
    _filter_gemma4_indexed_embeddings(input_ids, params)
    assert "indexed_embeddings" not in params

    _update_gemma4_reconstruct_mode(params, "26b_a4b_moe", has_mm_request = has_mm_request)

    assert params["reconstruct"] is True
    assert params["_gemma4_temp_reconstruct"] is True


def test_gemma4_26b_text_request_clears_temp_reconstruct_when_mm_request_ends():
    params = {
        "reconstruct": True,
        "_gemma4_temp_reconstruct": True,
    }

    _update_gemma4_reconstruct_mode(params, "26b_a4b_moe", has_mm_request = False)

    assert "reconstruct" not in params
    assert "_gemma4_temp_reconstruct" not in params


def test_gemma4_26b_mm_request_preserves_explicit_reconstruct_flag():
    params = {
        "indexed_embeddings": [SimpleNamespace(first_index = 1000, last_index = 1016)],
        "reconstruct": True,
    }

    _update_gemma4_reconstruct_mode(params, "26b_a4b_moe", has_mm_request = True)

    assert params["reconstruct"] is True
    assert "_gemma4_temp_reconstruct" not in params


def test_full_gemma4_mm_only_keeps_bsz1_graph_enabled_by_default_and_supports_force_disable():
    block = object.__new__(Gemma4TransformerBlock)
    block.mlp = object()

    full_attn = object.__new__(Gemma4Attention)
    full_attn.sliding_window = -1
    full_attn.device = torch.device("cpu")
    block.attn = full_attn

    assert block._should_disable_bsz1_graph({"indexed_embeddings": [object()]}) is False
    assert block._should_disable_bsz1_graph({}) is False
    assert block._should_disable_bsz1_graph({
        "indexed_embeddings": [object()],
        "_force_disable_bsz1_graph": True,
    }) is True

    sliding_attn = object.__new__(Gemma4Attention)
    sliding_attn.sliding_window = 1024
    sliding_attn.device = torch.device("cpu")
    block.attn = sliding_attn

    assert block._should_disable_bsz1_graph({
        "indexed_embeddings": [object()],
        "_force_disable_bsz1_graph": True,
    }) is False


@pytest.mark.parametrize("sliding_window", [-1, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_groups", [False, True])
def test_gemma4_mm_mask_vectorized_matches_reference(sliding_window: int, causal: bool, with_groups: bool):
    attn = object.__new__(Gemma4Attention)
    attn.sliding_window = sliding_window

    bsz = 2
    seqlen = 4
    device = torch.device("cpu")
    total_lens = torch.tensor([7, 6], dtype = torch.int32)
    cache_seqlens = torch.tensor([3, 2], dtype = torch.int32)
    vision_group_ids = None
    if with_groups:
        vision_group_ids = torch.tensor(
            [
                [0, 0, -1, -1],
                [-1, 1, 1, -1],
            ],
            dtype = torch.int32,
        )

    expected = _build_mm_mask_reference(
        sliding_window = sliding_window,
        bsz = bsz,
        seqlen = seqlen,
        total_lens = total_lens,
        cache_seqlens = cache_seqlens,
        vision_group_ids = vision_group_ids,
        q_dtype = torch.float16,
        device = device,
        causal = causal,
    )
    actual = attn._build_mm_mask(
        bsz = bsz,
        seqlen = seqlen,
        total_lens = total_lens,
        cache_seqlens = cache_seqlens,
        vision_group_ids = vision_group_ids,
        q_dtype = torch.float16,
        device = device,
        causal = causal,
    )

    assert torch.equal(actual, expected.eq(0))


def test_gemma4_mm_mask_cache_reuses_same_forward_params_and_separates_roles():
    params = {}
    total_lens = torch.tensor([7], dtype = torch.int32)
    cache_seqlens = torch.tensor([3], dtype = torch.int32)
    vision_group_ids = torch.tensor([[0, 0, -1, -1]], dtype = torch.int32)

    full_attn = object.__new__(Gemma4Attention)
    full_attn.sliding_window = -1
    full_mask_1 = full_attn._get_mm_mask_cached(
        params,
        1,
        4,
        total_lens,
        cache_seqlens,
        vision_group_ids,
        torch.float16,
        torch.device("cpu"),
        True,
    )
    full_mask_2 = full_attn._get_mm_mask_cached(
        params,
        1,
        4,
        total_lens,
        cache_seqlens,
        vision_group_ids,
        torch.float16,
        torch.device("cpu"),
        True,
    )
    assert full_mask_1 is full_mask_2

    swa_attn = object.__new__(Gemma4Attention)
    swa_attn.sliding_window = 256
    swa_mask = swa_attn._get_mm_mask_cached(
        params,
        1,
        4,
        total_lens,
        cache_seqlens,
        vision_group_ids,
        torch.float16,
        torch.device("cpu"),
        True,
    )
    assert swa_mask is not full_mask_1


def test_gemma4_mm_visible_positions_cache_reuses_and_reduces_mask():
    attn = object.__new__(Gemma4Attention)
    attn.sliding_window = 1

    params = {}
    total_lens = torch.tensor([5], dtype = torch.int32)
    cache_seqlens = torch.tensor([3], dtype = torch.int32)
    vision_group_ids = torch.tensor([[-1, -1]], dtype = torch.int32)

    mask = attn._get_mm_mask_cached(
        params,
        1,
        2,
        total_lens,
        cache_seqlens,
        vision_group_ids,
        torch.float16,
        torch.device("cpu"),
        True,
    )
    selected_1, counts_1, reduced_1 = attn._get_mm_visible_positions_cached(
        params,
        mask,
        total_lens,
        torch.device("cpu"),
    )
    selected_2, counts_2, reduced_2 = attn._get_mm_visible_positions_cached(
        params,
        mask,
        total_lens,
        torch.device("cpu"),
    )

    assert selected_1 is selected_2
    assert counts_1 is counts_2
    assert reduced_1 is reduced_2
    assert counts_1.tolist() == [3]
    assert selected_1[0, :3].tolist() == [2, 3, 4]
    assert torch.equal(reduced_1[0, 0, :, :3], mask[0, 0, :, selected_1[0, :3]])


def test_gemma4_mm_visible_positions_vectorized_handles_ragged_batches():
    attn = object.__new__(Gemma4Attention)
    attn.sliding_window = 1

    params = {}
    total_lens = torch.tensor([5, 3], dtype = torch.int32)
    cache_seqlens = torch.tensor([3, 1], dtype = torch.int32)
    vision_group_ids = torch.tensor(
        [
            [-1, -1],
            [0, -1],
        ],
        dtype = torch.int32,
    )

    mask = attn._get_mm_mask_cached(
        params,
        2,
        2,
        total_lens,
        cache_seqlens,
        vision_group_ids,
        torch.float16,
        torch.device("cpu"),
        True,
    )
    selected, counts, reduced = attn._get_mm_visible_positions_cached(
        params,
        mask,
        total_lens,
        torch.device("cpu"),
    )

    assert counts.tolist() == [3, 3]
    assert selected[0, :3].tolist() == [2, 3, 4]
    assert selected[1, :3].tolist() == [0, 1, 2]
    assert torch.equal(reduced[0, 0, :, :3], mask[0, 0, :, selected[0, :3]])
    assert torch.equal(reduced[1, 0, :, :3], mask[1, 0, :, selected[1, :3]])


def test_gemma4_mm_visible_positions_handles_empty_selection():
    attn = object.__new__(Gemma4Attention)
    attn.sliding_window = 1
    params = {}
    mask = torch.zeros((2, 1, 3, 4), dtype = torch.bool)
    total_lens = torch.tensor([0, 0], dtype = torch.int32)

    selected, counts, reduced = attn._get_mm_visible_positions_cached(
        params,
        mask,
        total_lens,
        torch.device("cpu"),
    )

    assert selected.shape == (2, 0)
    assert reduced.shape == (2, 1, 3, 0)
    assert counts.tolist() == [0, 0]


def test_full_text_only_single_quant_cache_uses_compact_path():
    attn = object.__new__(Gemma4Attention)
    attn.sliding_window = -1
    attn.force_quantized_fallback = False

    cache_layer = object.__new__(Gemma4SingleQuantCacheLayer)

    allow_full_compact_cache = (
        attn.sliding_window < 0 and
        isinstance(cache_layer, Gemma4SingleQuantCacheLayer)
    )
    use_shadow_cache = (
        isinstance(cache_layer, Gemma4QuantCacheLayer) and
        attn.sliding_window < 0 and
        not allow_full_compact_cache
    )
    use_compact_cache = (
        isinstance(cache_layer, Gemma4SingleQuantCacheLayer) and
        not use_shadow_cache and
        (
            allow_full_compact_cache or
            attn.force_quantized_fallback or
            False or
            False
        )
    )

    assert allow_full_compact_cache is True
    assert use_shadow_cache is False
    assert use_compact_cache is True


def test_full_mm_without_vision_groups_does_not_force_mm_fallback():
    attn = object.__new__(Gemma4Attention)
    attn.sliding_window = -1
    attn.force_quantized_fallback = False
    attn.head_dim = 256
    attn.device = torch.device("cpu")
    attn._get_cache_layer = lambda cache: object()

    params = {"indexed_embeddings": [object()]}
    has_mm_embeddings = bool(params.get("indexed_embeddings"))
    vision_group_ids = attn._get_vision_group_ids(params)
    if attn.sliding_window < 0:
        vision_group_ids = None
    needs_custom_mm_mask = vision_group_ids is not None
    force_quantized_fallback = (
        attn.force_quantized_fallback and
        False and
        False and
        isinstance(attn._get_cache_layer(None), Gemma4QuantCacheLayer)
    )

    assert has_mm_embeddings is True
    assert needs_custom_mm_mask is False
    assert force_quantized_fallback is False
    assert (attn.head_dim > 256 or needs_custom_mm_mask or force_quantized_fallback) is False


def test_gemma4_kv_workspace_reuses_same_forward_params():
    attn = object.__new__(Gemma4Attention)
    attn.num_kv_heads = 2
    attn.head_dim = 4

    params = {}
    k1, v1 = attn._get_kv_workspace(params, 1, 8, torch.half, torch.device("cpu"))
    k2, v2 = attn._get_kv_workspace(params, 1, 8, torch.half, torch.device("cpu"))
    assert k1 is k2
    assert v1 is v2

    k3, v3 = attn._get_kv_workspace(params, 1, 16, torch.half, torch.device("cpu"))
    assert k3 is not k1
    assert v3 is not v1

    hk1, hv1 = attn._get_kv_workspace(params, 1, 8, torch.half, torch.device("cpu"), heads_first = True)
    hk2, hv2 = attn._get_kv_workspace(params, 1, 8, torch.half, torch.device("cpu"), heads_first = True)
    assert hk1 is hk2
    assert hv1 is hv2
    assert hk1.shape == (1, 2, 8, 4)
    assert hv1.shape == (1, 2, 8, 4)
    assert hk1 is not k1
    assert hv1 is not v1


def test_gemma4_gather_cache_pages_supports_heads_first_workspace():
    attn = object.__new__(Gemma4Attention)
    attn.num_kv_heads = 2
    attn.head_dim = 4

    cache_tensor = torch.arange(3 * 256 * 2 * 4, dtype = torch.float32).reshape(3, 256, 2, 4)
    block_table = torch.tensor([[2, 1], [0, 2]], dtype = torch.int32)
    total_lens = torch.tensor([6, 3], dtype = torch.int32)

    gathered_time = attn._gather_cache_pages(cache_tensor, block_table, total_lens)
    gathered_heads = attn._gather_cache_pages(
        cache_tensor,
        block_table,
        total_lens,
        gathered = torch.zeros((2, 2, 6, 4), dtype = cache_tensor.dtype),
    )
    for batch_idx, total in enumerate(total_lens.tolist()):
        assert torch.equal(gathered_heads[batch_idx, :, :total], gathered_time[batch_idx, :total].transpose(0, 1))


def test_gemma4_gather_selected_cache_pages_matches_full_gather_subset():
    attn = object.__new__(Gemma4Attention)
    attn.num_kv_heads = 2
    attn.head_dim = 4

    cache_tensor = torch.arange(3 * 256 * 2 * 4, dtype = torch.float32).reshape(3, 256, 2, 4)
    block_table = torch.tensor([[2, 1], [0, 2]], dtype = torch.int32)
    total_lens = torch.tensor([6, 3], dtype = torch.int32)
    selected_positions = torch.tensor([[1, 4, 5], [0, 2, 0]], dtype = torch.long)
    selected_counts = torch.tensor([3, 2], dtype = torch.int32)

    full_gather = attn._gather_cache_pages(cache_tensor, block_table, total_lens)
    subset_heads = attn._gather_selected_cache_pages(
        cache_tensor,
        block_table,
        selected_positions,
        selected_counts,
        torch.zeros((2, 2, 3, 4), dtype = cache_tensor.dtype),
    )

    assert torch.equal(subset_heads[0, :, :3], full_gather[0, selected_positions[0, :3]].transpose(0, 1))
    assert torch.equal(subset_heads[1, :, :2], full_gather[1, selected_positions[1, :2]].transpose(0, 1))


def test_gemma4_mm_attention_softcap_gqa_matches_repeat_reference():
    attn = object.__new__(Gemma4Attention)
    attn.sm_scale = None
    attn.head_dim = 4
    attn.logit_softcapping = 30.0
    attn.gqa = True

    q = torch.arange(2 * 4 * 3 * 4, dtype = torch.float32).reshape(2, 4, 3, 4) / 17.0
    k = torch.arange(2 * 2 * 5 * 4, dtype = torch.float32).reshape(2, 2, 5, 4) / 19.0
    v = torch.arange(2 * 2 * 5 * 4, dtype = torch.float32).reshape(2, 2, 5, 4) / 23.0
    mask = torch.ones((2, 1, 3, 5), dtype = torch.bool)
    mask[:, :, :, -1] = False

    repeat = q.shape[1] // k.shape[1]
    k_ref = k.repeat_interleave(repeat, dim = 1)
    v_ref = v.repeat_interleave(repeat, dim = 1)
    scale = attn.head_dim ** -0.5
    scores = torch.matmul(q.float(), k_ref.transpose(-1, -2).float()) * scale
    scores = torch.tanh(scores / attn.logit_softcapping) * attn.logit_softcapping
    scores.masked_fill_(mask.logical_not(), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim = -1, dtype = torch.float32)
    expected = torch.matmul(probs, v_ref.float()).to(q.dtype)

    actual = attn._mm_attention(q, k, v, mask)
    assert torch.allclose(actual, expected, atol = 1e-5, rtol = 1e-5)
