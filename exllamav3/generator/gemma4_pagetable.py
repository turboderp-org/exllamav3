from __future__ import annotations

from collections import OrderedDict, deque

import torch

from ..constants import PAGE_SIZE
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from .pagetable import CachePage, PageArena, PageTable, tensor_hash_checksum


class Gemma4PageTable(PageTable):

    def __init__(self, generator, cache):
        super().__init__(generator, cache)
        self.sliding_window = generator.model.config.sliding_window
        self.full_arena = self.default_arena
        self.swa_arena = PageArena(self, self._get_role_max_pages("swa"), role = "swa")
        self._allocate_has_mm_tokens = False
        self._prefer_low_index_swa = False
        self.active_full_to_swa_pages: dict[int, CachePage] = {}
        self.cached_full_to_swa_pages: OrderedDict[tuple[bytes, bool], CachePage] = OrderedDict()
        self.cached_partial_to_swa_pages: OrderedDict[tuple[bytes, bool], CachePage] = OrderedDict()
        sliding_window_pages = 0 if self.sliding_window < 0 else (self.sliding_window + PAGE_SIZE - 1) // PAGE_SIZE
        self.max_cached_copy_sources = max(1, sliding_window_pages + 1)

    @staticmethod
    def _ensure_seq_role_state(seq) -> dict[str, dict]:
        state = getattr(seq, "_gemma4_role_state", None)
        if state is None:
            state = {
                "pages": {},
                "backing_pages": {},
                "block_index_tensors": {},
                "kv_positions": {},
                "page_maps": {},
            }
            setattr(seq, "_gemma4_role_state", state)
        return state

    def _get_seq_allocated_pages(self, seq, role: str = "default"):
        if role == "default":
            return seq.allocated_pages
        return self._ensure_seq_role_state(seq)["pages"].get(role)

    def _set_seq_allocated_pages(self, seq, pages, role: str = "default"):
        if role == "default":
            seq.allocated_pages = pages
            return
        self._ensure_seq_role_state(seq)["pages"][role] = pages

    def _get_seq_backing_pages(self, seq, role: str = "default"):
        if role == "default":
            return seq.allocated_pages
        return self._ensure_seq_role_state(seq)["backing_pages"].get(role)

    def _set_seq_backing_pages(self, seq, pages, role: str = "default"):
        if role == "default":
            seq.allocated_pages = pages
            return
        state = self._ensure_seq_role_state(seq)["backing_pages"]
        if pages is None:
            state.pop(role, None)
        else:
            state[role] = pages

    def _get_seq_block_index_tensor(self, seq, role: str = "default"):
        if role == "default":
            return seq.block_index_tensor
        return self._ensure_seq_role_state(seq)["block_index_tensors"].get(role)

    def _get_seq_kv_position(self, seq, role: str = "default"):
        if role == "default":
            return seq.kv_position
        return self._ensure_seq_role_state(seq)["kv_positions"].get(role, seq.kv_position)

    def _set_seq_kv_position(self, seq, kv_position: int, role: str = "default"):
        if role == "default":
            seq.kv_position = kv_position
            return
        self._ensure_seq_role_state(seq)["kv_positions"][role] = kv_position

    def _clear_seq_role_views(self, seq):
        if hasattr(seq, "_gemma4_role_state"):
            delattr(seq, "_gemma4_role_state")

    def _get_seq_page_map(self, seq, role: str = "default"):
        if role == "default":
            return {}
        return self._ensure_seq_role_state(seq)["page_maps"].get(role, {})

    def _set_seq_page_map(self, seq, page_map: dict[int, CachePage], role: str = "default"):
        if role == "default":
            return
        self._ensure_seq_role_state(seq)["page_maps"][role] = page_map

    def _build_seq_block_index_tensor(self, seq, role: str = "default"):
        if role == "default":
            return seq.build_block_index_tensor()
        pages = self._get_seq_allocated_pages(seq, role = role)
        tensor = None if pages is None else torch.tensor(
            [[page.page_index for page in pages]],
            dtype = torch.int32,
        )
        self._ensure_seq_role_state(seq)["block_index_tensors"][role] = tensor
        return tensor

    def build_batch_block_index(
        self,
        active_jobs: list,
        max_seq_len: int,
        role: str = "default",
        single_sequence_per_job: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = 0
        for job in active_jobs:
            if not job.is_prefill_done():
                continue
            batch_size += 1 if single_sequence_per_job else len(job.sequences)

        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_index = torch.zeros((batch_size, max_pages_batch), dtype = torch.int32)
        cache_seqlens = torch.zeros((batch_size,), dtype = torch.int32)

        batch = 0
        for job in active_jobs:
            if not job.is_prefill_done():
                continue
            for seq in job.sequences:
                seq_block_index = self._get_seq_block_index_tensor(seq, role = role)
                if seq_block_index is None:
                    continue
                seq_block_index = seq_block_index[:, :max_pages_batch]
                block_index[batch : batch + 1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = self._get_seq_kv_position(seq, role = role)
                batch += 1
                if single_sequence_per_job:
                    break

        return block_index, cache_seqlens

    @staticmethod
    def _sequence_has_mm_tokens(seq) -> bool:
        cached = getattr(seq, "_gemma4_has_mm_tokens", None)
        if cached is None:
            cached = bool((seq.input_ids.torch() >= FIRST_MM_EMBEDDING_INDEX).any().item())
            setattr(seq, "_gemma4_has_mm_tokens", cached)
        return cached

    @staticmethod
    def _cached_full_page_key(full_page: CachePage, has_mm_tokens: bool = False) -> tuple[bytes, bool]:
        return full_page.phash, has_mm_tokens

    @staticmethod
    def _cached_partial_page_key(
        prev_hash: bytes | None,
        partial_ids: torch.Tensor,
        has_mm_tokens: bool = False,
    ) -> tuple[bytes, bool]:
        return tensor_hash_checksum(partial_ids, prev_hash), has_mm_tokens

    def _get_prompt_window_start_page(self, prompt_cached_tokens: int) -> int:
        if prompt_cached_tokens <= 0 or self.sliding_window < 0:
            return 0
        window_start = max(0, prompt_cached_tokens - self.sliding_window)
        return window_start // PAGE_SIZE

    def _has_matching_partial_copy_source(self, seq) -> bool:
        prompt_cached_tokens = len(seq.input_ids) - 1
        prompt_tail_tokens = prompt_cached_tokens % PAGE_SIZE
        if prompt_tail_tokens <= 0:
            return True

        page_idx = prompt_cached_tokens // PAGE_SIZE
        prefill_ids = seq.sequence_ids.torch_slice(page_idx * PAGE_SIZE, prompt_cached_tokens)
        prev_hash = None if page_idx == 0 else seq.page_hashes[page_idx - 1]
        partial_key = self._cached_partial_page_key(prev_hash, prefill_ids, self._sequence_has_mm_tokens(seq))
        return self.cached_partial_to_swa_pages.get(partial_key) is not None

    def _has_reusable_full_prefix(self, seq) -> bool:
        prompt_cached_tokens = len(seq.input_ids) - 1
        prompt_full_pages = prompt_cached_tokens // PAGE_SIZE
        if prompt_full_pages <= 0:
            return False

        has_mm_tokens = self._sequence_has_mm_tokens(seq)
        for page_hash in seq.page_hashes[:prompt_full_pages]:
            if self.cached_full_to_swa_pages.get((page_hash, has_mm_tokens)) is None:
                return False
            if (
                self.find_referenced_page(page_hash, role = "full") is not None or
                self.find_unreferenced_page(page_hash, role = "full") is not None
            ):
                return True
            return False
        return False

    def _iter_prompt_window_full_pages(self, seq, full_pages: list[CachePage]) -> list[tuple[CachePage, CachePage]]:
        old_swa_map = self._get_seq_page_map(seq, role = "swa")
        prompt_cached_tokens = len(seq.input_ids) - 1
        prompt_full_pages = prompt_cached_tokens // PAGE_SIZE
        if prompt_full_pages <= 0:
            return []
        start_page = self._get_prompt_window_start_page(prompt_cached_tokens)
        cached_pairs = []
        for page_idx in range(start_page, min(prompt_full_pages, len(full_pages))):
            full_page = full_pages[page_idx]
            swa_page = old_swa_map.get(id(full_page))
            if swa_page is not None:
                cached_pairs.append((full_page, swa_page))
        return cached_pairs

    def _copy_cached_swa_page(self, source_page: CachePage, target_page: CachePage, num_tokens: int) -> None:
        if num_tokens <= 0:
            return
        for layer in self.cache.layers.values():
            if getattr(layer, "cache_role", "default") != "swa":
                continue
            layer.copy_page(layer, source_page.page_index, target_page.page_index, num_tokens)
        target_page.kv_position = num_tokens
        target_page.prev_hash = source_page.prev_hash
        target_page.can_revert = False

    def _restore_cached_swa_prefix(self, seq, swa_pages: list[CachePage]) -> None:
        full_pages = self._get_seq_allocated_pages(seq, role = "full") or seq.allocated_pages or []
        if not full_pages or not swa_pages:
            return

        cached_full_pages = seq.kv_position // PAGE_SIZE
        if cached_full_pages <= 0:
            return

        has_mm_tokens = self._sequence_has_mm_tokens(seq)
        start_page = self._get_prompt_window_start_page(seq.kv_position)
        local_cached_pages = max(0, cached_full_pages - start_page)
        for local_idx in range(min(local_cached_pages, len(swa_pages))):
            full_page = full_pages[start_page + local_idx]
            full_page_key = self._cached_full_page_key(full_page, has_mm_tokens)
            cached_source = self.cached_full_to_swa_pages.get(full_page_key)
            if cached_source is None:
                continue
            self.cached_full_to_swa_pages.move_to_end(full_page_key)
            self._copy_cached_swa_page(cached_source, swa_pages[local_idx], PAGE_SIZE)

    def _snapshot_prompt_window_sources(self, seq, full_pages: list[CachePage]) -> None:
        has_mm_tokens = self._sequence_has_mm_tokens(seq)
        prompt_window_pairs = [
            (full_page, active_swa)
            for full_page, active_swa in self._iter_prompt_window_full_pages(seq, full_pages)
            if self.cached_full_to_swa_pages.get(self._cached_full_page_key(full_page, has_mm_tokens)) is None
        ]
        if not prompt_window_pairs:
            return

        self._evict_cached_swa_for_admission(len(prompt_window_pairs))
        if not self.swa_arena.unreferenced_pages:
            return

        snapshot_count = min(len(prompt_window_pairs), len(self.swa_arena.unreferenced_pages))
        snapshot_pages, _, _, _ = self.allocate_pages([], snapshot_count, None, role = "swa")
        for (full_page, active_swa), snapshot_page in zip(prompt_window_pairs[:snapshot_count], snapshot_pages):
            self._copy_cached_swa_page(active_swa, snapshot_page, PAGE_SIZE)
            self._cache_copy_candidate(full_page, snapshot_page, has_mm_tokens)

    def _get_role_max_pages(self, role: str) -> int:
        role_max_tokens = [
            layer.max_num_tokens
            for layer in self.cache.layers.values()
            if getattr(layer, "cache_role", "default") == role
        ]
        if not role_max_tokens:
            return self.default_arena.max_pages
        return min(role_max_tokens) // PAGE_SIZE

    def _get_swa_capacity_pages(self, seq) -> int:
        total_pages = len(seq.page_hashes or []) + seq.new_unique_pages
        if total_pages <= 0:
            return 0
        if self.sliding_window < 0:
            return min(total_pages, self.swa_arena.max_pages)
        capacity_pages = self.get_min_swa_capacity_pages()
        return min(total_pages, capacity_pages, self.swa_arena.max_pages)

    def get_min_swa_capacity_tokens(self) -> int:
        if self.sliding_window < 0:
            return self.full_arena.max_pages * PAGE_SIZE
        max_write_tokens = max(self.generator.max_chunk_size, self.generator.num_draft_tokens + 1)
        return self.sliding_window + PAGE_SIZE + max_write_tokens

    def get_min_swa_capacity_pages(self) -> int:
        capacity_tokens = self.get_min_swa_capacity_tokens()
        return (capacity_tokens + PAGE_SIZE - 1) // PAGE_SIZE

    def prepare_sequence(
        self,
        seq,
        has_prefix_token: bool,
        max_new_tokens: int,
        allow_page_reuse: bool = True,
    ):
        unique_hashes, unique_pages = super().prepare_sequence(
            seq,
            has_prefix_token,
            max_new_tokens,
            allow_page_reuse = allow_page_reuse,
        )
        if not allow_page_reuse:
            return unique_hashes, unique_pages
        if not self._has_reusable_full_prefix(seq):
            return unique_hashes, unique_pages
        if self._has_matching_partial_copy_source(seq):
            return unique_hashes, unique_pages
        return super().prepare_sequence(
            seq,
            has_prefix_token,
            max_new_tokens,
            allow_page_reuse = False,
        )

    def get_arena(self, role: str = "default") -> PageArena:
        if role == "swa":
            return self.swa_arena
        if role in ("default", "full"):
            return self.full_arena
        raise KeyError(f"Unknown page arena role: {role}")

    def num_unreferenced_pages_by_role(self) -> dict[str, int]:
        full_available = len(self.full_arena.unreferenced_pages)
        return {
            "default": full_available,
            "full": full_available,
            "swa": len(self.swa_arena.unreferenced_pages),
        }

    def current_new_pages_required_by_role(self, job) -> dict[str, int]:
        full_required = self.current_new_pages_required(job)
        swa_required = sum(self._get_swa_capacity_pages(seq) for seq in job.sequences)
        return {
            "default": full_required,
            "full": full_required,
            "swa": swa_required,
        }

    def _evict_cached_swa_for_admission(self, required_pages: int) -> None:
        if required_pages <= len(self.swa_arena.unreferenced_pages):
            return
        while self.cached_partial_to_swa_pages and required_pages > len(self.swa_arena.unreferenced_pages):
            self._evict_oldest_cached_partial_swa_page()
        while self.cached_full_to_swa_pages and required_pages > len(self.swa_arena.unreferenced_pages):
            self._evict_oldest_cached_swa_page()

    def can_admit_job(self, job, current_batch_size: int, max_batch_size: int) -> bool:
        if len(job.sequences) + current_batch_size > max_batch_size:
            return False
        required = self.current_new_pages_required_by_role(job)
        swa_required = required.get("swa", 0)
        if swa_required:
            self._evict_cached_swa_for_admission(swa_required)
        available = self.num_unreferenced_pages_by_role()
        for role, pages in required.items():
            if pages > available.get(role, 0):
                return False
        return True

    def _release_cached_swa_page(self, full_page: CachePage) -> None:
        for has_mm_tokens in (False, True):
            cached_swa = self.cached_full_to_swa_pages.pop(
                self._cached_full_page_key(full_page, has_mm_tokens),
                None,
            )
            if cached_swa is not None and cached_swa.ref_count > 0:
                self.deallocate_pages([cached_swa])

    @staticmethod
    def _clear_page_sequence(page: CachePage) -> None:
        # Gemma4 reuses physical cache pages across heterogeneous text/MM requests.
        # Clear the token bookkeeping buffer whenever a page is freshly assigned to a
        # new logical sequence so stale ids cannot influence later partial-page logic.
        page.sequence.zero_()

    def _evict_oldest_cached_swa_page(self) -> None:
        if not self.cached_full_to_swa_pages:
            return
        _, cached_swa = self.cached_full_to_swa_pages.popitem(last = False)
        if cached_swa.ref_count > 0:
            self.deallocate_pages([cached_swa])

    def _evict_oldest_cached_partial_swa_page(self) -> None:
        if not self.cached_partial_to_swa_pages:
            return
        _, cached_swa = self.cached_partial_to_swa_pages.popitem(last = False)
        if cached_swa.ref_count > 0:
            self.deallocate_pages([cached_swa])

    def _cache_copy_candidate(self, full_page: CachePage, swa_page: CachePage, has_mm_tokens: bool) -> None:
        full_page_key = self._cached_full_page_key(full_page, has_mm_tokens)
        existing = self.cached_full_to_swa_pages.get(full_page_key)
        if existing is swa_page:
            self.cached_full_to_swa_pages.move_to_end(full_page_key)
            return
        if existing is not None and existing.ref_count > 0:
            self.deallocate_pages([existing])
        self.cached_full_to_swa_pages[full_page_key] = swa_page
        self.cached_full_to_swa_pages.move_to_end(full_page_key)
        while len(self.cached_full_to_swa_pages) > self.max_cached_copy_sources:
            self._evict_oldest_cached_swa_page()

    def _cache_partial_copy_candidate(self, partial_key: tuple[bytes, bool], swa_page: CachePage) -> None:
        existing = self.cached_partial_to_swa_pages.get(partial_key)
        if existing is swa_page:
            self.cached_partial_to_swa_pages.move_to_end(partial_key)
            return
        if existing is not None and existing.ref_count > 0:
            self.deallocate_pages([existing])
        self.cached_partial_to_swa_pages[partial_key] = swa_page
        self.cached_partial_to_swa_pages.move_to_end(partial_key)
        while len(self.cached_partial_to_swa_pages) > self.max_cached_copy_sources:
            self._evict_oldest_cached_partial_swa_page()

    def cache_prefill_copy_source(self, seq) -> None:
        # In TP mode the cache tensors live on split worker-local cache layers, so the
        # main-process cache objects do not own backing qk/qv/sk/sv tensors. This
        # snapshot is an optimization only, so skip it and keep the deallocate-time pin
        # path as the correctness-preserving fallback.
        if self.cache.model.loaded_tp:
            return

        full_pages = self._get_seq_allocated_pages(seq, role = "full") or seq.allocated_pages or []
        has_mm_tokens = self._sequence_has_mm_tokens(seq)
        if full_pages:
            self._snapshot_prompt_window_sources(seq, full_pages)

        prompt_cached_tokens = len(seq.input_ids) - 1
        prompt_tail_tokens = prompt_cached_tokens % PAGE_SIZE
        if prompt_tail_tokens <= 0:
            return

        prompt_terminal_page_idx = len(seq.page_hashes or [])
        if not (0 <= prompt_terminal_page_idx < len(full_pages)):
            return

        prompt_terminal_page = full_pages[prompt_terminal_page_idx]
        active_swa = self._get_seq_page_map(seq, role = "swa").get(id(prompt_terminal_page))
        if active_swa is None:
            return

        prev_hash = None if prompt_terminal_page_idx == 0 else seq.page_hashes[prompt_terminal_page_idx - 1]
        partial_ids = seq.sequence_ids.torch_slice(prompt_terminal_page_idx * PAGE_SIZE, prompt_cached_tokens)
        partial_key = self._cached_partial_page_key(prev_hash, partial_ids, has_mm_tokens)
        if self.cached_partial_to_swa_pages.get(partial_key) is not None:
            return

        # Snapshotting the prompt-terminal local page is an optimization, not a correctness
        # requirement. If the SWA arena is currently full, fall back to the existing
        # deallocate-time pinning path instead of overcommitting local pages.
        if not self.swa_arena.unreferenced_pages:
            return

        snapshot_pages, _, _, _ = self.allocate_pages([], 1, None, role = "swa")
        if not snapshot_pages:
            return
        snapshot_page = snapshot_pages[0]

        for layer in self.cache.layers.values():
            if getattr(layer, "cache_role", "default") != "swa":
                continue
            layer.copy_page(layer, active_swa.page_index, snapshot_page.page_index, prompt_tail_tokens)

        snapshot_page.kv_position = prompt_tail_tokens
        snapshot_page.prev_hash = active_swa.prev_hash
        snapshot_page.can_revert = False
        self._cache_partial_copy_candidate(partial_key, snapshot_page)

    def _sync_role_views(self, seq) -> None:
        old_swa_map = self._get_seq_page_map(seq, role = "swa")
        for full_page_id, swa_page in old_swa_map.items():
            if self.active_full_to_swa_pages.get(full_page_id) is swa_page:
                del self.active_full_to_swa_pages[full_page_id]

        old_full_kv = self._get_seq_kv_position(seq, role = "full")
        old_swa_kv = self._get_seq_kv_position(seq, role = "swa")
        full_pages = self._get_seq_backing_pages(seq, role = "full") or seq.allocated_pages
        full_kv = seq.kv_position
        base_page = 0

        self._set_seq_allocated_pages(seq, full_pages, role = "full")
        self._set_seq_kv_position(seq, full_kv, role = "full")
        self._build_seq_block_index_tensor(seq, role = "full")

        swa_backing = self._get_seq_backing_pages(seq, role = "swa")
        if swa_backing is not None:
            swa_pages = list(self._get_seq_allocated_pages(seq, role = "swa") or swa_backing)
            if len(swa_pages) != len(swa_backing):
                swa_pages = list(swa_backing)

            if self.sliding_window < 0:
                old_base_page = 0
                new_base_page = 0
            else:
                old_window_start = max(0, old_full_kv - self.sliding_window)
                new_window_start = max(0, full_kv - self.sliding_window)
                old_base_page = old_window_start // PAGE_SIZE
                new_base_page = new_window_start // PAGE_SIZE
            base_page = new_base_page

            old_valid_pages = min(len(swa_pages), (old_swa_kv + PAGE_SIZE - 1) // PAGE_SIZE) if old_swa_kv else 0
            pages_to_drop = max(0, new_base_page - old_base_page)
            if full_kv >= old_full_kv and old_valid_pages and pages_to_drop:
                drop = min(pages_to_drop, old_valid_pages)
                swa_pages = (
                    swa_pages[drop:old_valid_pages] +
                    swa_pages[:drop] +
                    swa_pages[old_valid_pages:]
                )

            swa_kv = min(full_kv - new_base_page * PAGE_SIZE, len(swa_pages) * PAGE_SIZE)
        elif full_pages is None:
            swa_pages = None
            swa_kv = 0
        elif self.sliding_window < 0:
            swa_pages = full_pages
            swa_kv = full_kv
        else:
            window_start = max(0, full_kv - self.sliding_window)
            base_page = window_start // PAGE_SIZE
            swa_pages = full_pages[base_page:]
            swa_kv = full_kv - base_page * PAGE_SIZE

        self._set_seq_allocated_pages(seq, swa_pages, role = "swa")
        self._set_seq_kv_position(seq, swa_kv, role = "swa")
        self._build_seq_block_index_tensor(seq, role = "swa")
        swa_page_map = {}
        if full_pages is not None and swa_pages is not None:
            max_local_pages = min(len(swa_pages), max(len(full_pages) - base_page, 0))
            for local_idx in range(max_local_pages):
                full_page = full_pages[base_page + local_idx]
                swa_page = swa_pages[local_idx]
                swa_page_map[id(full_page)] = swa_page
                self.active_full_to_swa_pages[id(full_page)] = swa_page
        self._set_seq_page_map(seq, swa_page_map, role = "swa")

    def clear_page(self, page: CachePage, role: str = "default"):
        if role == "default":
            self._release_cached_swa_page(page)
        super().clear_page(page, role = role)

    def reset_page_table(self):
        self.active_full_to_swa_pages.clear()
        self.cached_full_to_swa_pages.clear()
        self.cached_partial_to_swa_pages.clear()
        super().reset_page_table()
        self.swa_arena.reset()

    def allocate_pages(
        self,
        page_hashes: list,
        new_unique_pages: int,
        recurrent_pages: list[int] | None,
        role: str = "default",
    ):
        if role != "default":
            if (
                role == "swa" and
                self._prefer_low_index_swa and
                not page_hashes and
                recurrent_pages is None
            ):
                arena = self.get_arena(role)
                allocated_pages = []
                available_pages = list(arena.unreferenced_pages.values())
                available_pages.sort(key = lambda page: page.page_index)
                available_pages = deque(available_pages)

                for _ in range(new_unique_pages):
                    arena.access_serial += 1
                    while available_pages[0].ref_count:
                        available_pages.popleft()
                    page = available_pages.popleft()
                    page.add_ref_unique(arena.access_serial)
                    self._clear_page_sequence(page)
                    allocated_pages.append(page)

                cached_pages = 0
                kv_position = 0
                non_sequential_pages = 0
                for page_a, page_b in zip(allocated_pages, allocated_pages[1:]):
                    if page_b.page_index != page_a.page_index + 1:
                        non_sequential_pages += 1

                return allocated_pages, kv_position, cached_pages, non_sequential_pages

            allocated_pages, kv_position, cached_pages, non_sequential_pages = super().allocate_pages(
                page_hashes,
                new_unique_pages,
                recurrent_pages,
                role = role,
            )
            for page in allocated_pages:
                if page.kv_position == 0:
                    self._clear_page_sequence(page)
            return allocated_pages, kv_position, cached_pages, non_sequential_pages

        arena = self.get_arena(role)
        allocated_pages = []
        available_pages = None
        reuse_prefix_valid = True
        profile = getattr(getattr(self.generator, "model", None), "caps", {}).get("gemma4_profile", "generic")
        prefer_low_index_pages = (
            len(page_hashes) == 0 or
            (
                profile == "26b_a4b_moe" and
                not self._allocate_has_mm_tokens
            )
        )

        def get_available_pages() -> deque[CachePage]:
            pages = list(arena.unreferenced_pages.values())
            # Gemma4 26B text requests are sensitive to full-page placement after multimodal
            # churn, including the "reuse one full page, append one new tail page" case.
            # Keep the generic LRU path for every other model/request type, but make 26B
            # non-MM text full-page allocation deterministic by preferring the lowest page
            # indices across the whole request, not only for a completely fresh prefix.
            pages.sort(key = (lambda page: page.page_index) if prefer_low_index_pages else (lambda page: page.access_serial))
            return deque(pages)

        for h in page_hashes:
            arena.access_serial += 1

            cached_swa = (
                self.cached_full_to_swa_pages.get((h, self._allocate_has_mm_tokens))
                if reuse_prefix_valid else None
            )

            rp = arena.referenced_pages.get(h)
            if rp and cached_swa is not None:
                rp.add_ref(arena.access_serial)
                allocated_pages.append(rp)
                continue

            up = arena.unreferenced_pages.get(h)
            if up and cached_swa is not None:
                up.add_ref(arena.access_serial)
                allocated_pages.append(up)
                continue

            reuse_prefix_valid = False
            if not allocated_pages:
                prefer_low_index_pages = True

            if available_pages is None:
                available_pages = get_available_pages()
            else:
                while available_pages[0].ref_count:
                    available_pages.popleft()

            op = available_pages.popleft()
            self._release_cached_swa_page(op)
            if (
                not cached_swa and
                (h in arena.referenced_pages or h in arena.unreferenced_pages)
            ):
                op.add_ref_unique(arena.access_serial)
                self._clear_page_sequence(op)
            else:
                op.add_ref_clear(arena.access_serial, h)
                self._clear_page_sequence(op)
            allocated_pages.append(op)

        for _ in range(new_unique_pages):
            arena.access_serial += 1

            if available_pages is None:
                available_pages = get_available_pages()
            else:
                while available_pages[0].ref_count:
                    available_pages.popleft()

            op = available_pages.popleft()
            self._release_cached_swa_page(op)
            op.add_ref_unique(arena.access_serial)
            self._clear_page_sequence(op)
            allocated_pages.append(op)

        cached_pages = 0
        for page in allocated_pages:
            if page.kv_position == PAGE_SIZE:
                cached_pages += 1
            else:
                break

        if recurrent_pages is not None:
            max_recur = 0
            for rp in recurrent_pages:
                if rp < cached_pages:
                    max_recur = rp + 1
            cached_pages = max_recur

        kv_position = cached_pages * PAGE_SIZE

        non_sequential_pages = 0
        for page_a, page_b in zip(allocated_pages, allocated_pages[1:]):
            if page_b.page_index != page_a.page_index + 1:
                non_sequential_pages += 1

        return allocated_pages, kv_position, cached_pages, non_sequential_pages

    def allocate_sequence(self, seq, recurrent_cache):
        self._allocate_has_mm_tokens = self._sequence_has_mm_tokens(seq)
        try:
            result = super().allocate_sequence(seq, recurrent_cache)
        finally:
            self._allocate_has_mm_tokens = False
        swa_capacity_pages = self._get_swa_capacity_pages(seq)
        if swa_capacity_pages:
            self._evict_cached_swa_for_admission(swa_capacity_pages)
            profile = getattr(getattr(self.generator, "model", None), "caps", {}).get("gemma4_profile", "generic")
            # 26B text requests can start from a completely fresh local SWA view after
            # multimodal churn. In that case, keep the local window on the lowest SWA
            # page indices so the initial block-table mapping is deterministic instead
            # of depending on prior SWA eviction order.
            self._prefer_low_index_swa = (
                profile == "26b_a4b_moe" and
                not self._sequence_has_mm_tokens(seq) and
                seq.kv_position == 0
            )
            try:
                swa_pages, _, _, _ = self.allocate_pages([], swa_capacity_pages, None, role = "swa")
            finally:
                self._prefer_low_index_swa = False
            self._set_seq_backing_pages(seq, swa_pages, role = "swa")
            self._restore_cached_swa_prefix(seq, swa_pages)
        self.sync_sequence_views(seq)
        return result

    def deallocate_sequence(self, seq):
        old_swa_map = self._get_seq_page_map(seq, role = "swa")
        for full_page_id, swa_page in old_swa_map.items():
            if self.active_full_to_swa_pages.get(full_page_id) is swa_page:
                del self.active_full_to_swa_pages[full_page_id]

        swa_backing = self._get_seq_backing_pages(seq, role = "swa")
        if swa_backing:
            self.deallocate_pages(swa_backing)
            self._set_seq_backing_pages(seq, None, role = "swa")
        super().deallocate_sequence(seq)

    def sync_sequence_views(self, seq):
        self._sync_role_views(seq)

    def clamp_prefill_end(
        self,
        seq,
        prefill_start: int,
        prefill_end: int,
        embeddings = None,
        atomic_mm_prefill: bool = False,
    ) -> int:
        swa_pages = self._get_seq_allocated_pages(seq, role = "swa")
        if not swa_pages:
            return prefill_end

        local_capacity = len(swa_pages) * PAGE_SIZE
        local_past = self._get_seq_kv_position(seq, role = "swa")
        local_headroom = local_capacity - local_past
        if local_headroom <= 0:
            raise ValueError(
                "Gemma4 sliding-window cache has no local headroom left for prefill "
                f"(capacity {local_capacity}, local past {local_past}). "
                "Increase swa_cache_size or reduce the effective local write span."
            )

        max_prefill_end = min(prefill_end, prefill_start + local_headroom)
        if max_prefill_end >= prefill_end or not atomic_mm_prefill or not embeddings:
            return max_prefill_end

        seq_tokens = seq.sequence_ids.torch().view(-1)
        for embedding in embeddings:
            mask = (seq_tokens >= embedding.first_index) & (seq_tokens < embedding.last_index)
            if not mask.any():
                continue
            mm_positions = torch.nonzero(mask, as_tuple = False).flatten()
            mm_start = int(mm_positions[0])
            mm_end = int(mm_positions[-1]) + 1
            if prefill_start < mm_end and mm_start < max_prefill_end < mm_end:
                required_tokens = mm_end - prefill_start
                recommended_tokens = ((required_tokens + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
                raise ValueError(
                    "Gemma4 atomic MM prefill span exceeds the current local SWA cache view "
                    f"(need {required_tokens} local tokens, but only {local_headroom} are available; "
                    f"capacity {local_capacity}, local past {local_past}; "
                    f"next page-aligned safe size {recommended_tokens}). "
                    "Increase swa_cache_size or reduce max_chunk_size."
                )

        return max_prefill_end

    def build_cache_copy_plan(self, source_page, target_page, num_tokens, seq = None):
        has_mm_tokens = self._sequence_has_mm_tokens(seq) if seq is not None else False
        source_partial = None
        if source_page.kv_position < PAGE_SIZE and source_page.prev_hash is not None and num_tokens > 0:
            partial_ids = source_page.sequence[:, :num_tokens]
            partial_key = self._cached_partial_page_key(source_page.prev_hash, partial_ids, has_mm_tokens)
            source_partial = self.cached_partial_to_swa_pages.get(partial_key)
        source_swa = (
            self.active_full_to_swa_pages.get(id(source_page)) or
            self.cached_full_to_swa_pages.get(self._cached_full_page_key(source_page, has_mm_tokens)) or
            source_partial
        )
        target_swa = (
            self.active_full_to_swa_pages.get(id(target_page)) or
            self.cached_full_to_swa_pages.get(self._cached_full_page_key(target_page, has_mm_tokens))
        )
        if source_swa is None or target_swa is None:
            return None
        source_page_key = self._cached_full_page_key(source_page, has_mm_tokens)
        if source_page_key in self.cached_full_to_swa_pages:
            self.cached_full_to_swa_pages.move_to_end(source_page_key)
        if source_partial is not None:
            self.cached_partial_to_swa_pages.move_to_end(partial_key)
        return {
            "full": (source_page.page_index, target_page.page_index),
            "swa": (source_swa.page_index, target_swa.page_index),
        }

    def build_draft_decode_params(self, active_jobs: list, max_seq_len: int) -> dict:
        params = super().build_draft_decode_params(active_jobs, max_seq_len)
        for job in active_jobs:
            if not job.is_prefill_done():
                continue
            for seq in job.sequences:
                self.sync_sequence_views(seq)
                break
        params["block_table_full"], params["cache_seqlens_full"] = self.build_batch_block_index(
            active_jobs,
            max_seq_len,
            role = "full",
            single_sequence_per_job = True,
        )
        params["block_table_swa"], params["cache_seqlens_swa"] = self.build_batch_block_index(
            active_jobs,
            max_seq_len,
            role = "swa",
            single_sequence_per_job = True,
        )
        return params

    def build_decode_params(self, active_jobs: list, max_seq_len: int, use_offsets: bool = False) -> dict:
        params = super().build_decode_params(active_jobs, max_seq_len, use_offsets = use_offsets)
        for job in active_jobs:
            if not job.is_prefill_done():
                continue
            for seq in job.sequences:
                self.sync_sequence_views(seq)
        params["block_table_full"], params["cache_seqlens_full"] = self.build_batch_block_index(
            active_jobs,
            max_seq_len,
            role = "full",
            single_sequence_per_job = False,
        )
        params["block_table_swa"], params["cache_seqlens_swa"] = self.build_batch_block_index(
            active_jobs,
            max_seq_len,
            role = "swa",
            single_sequence_per_job = False,
        )
        return params

    def build_prefill_params(self, seq, prefill_start: int) -> dict:
        params = super().build_prefill_params(seq, prefill_start)
        self.sync_sequence_views(seq)
        params["block_table_full"] = self._get_seq_block_index_tensor(seq, role = "full")
        params["block_table_swa"] = self._get_seq_block_index_tensor(seq, role = "swa")
        params["cache_seqlens_full"] = torch.tensor([self._get_seq_kv_position(seq, role = "full")], dtype = torch.int32)
        params["cache_seqlens_swa"] = torch.tensor([self._get_seq_kv_position(seq, role = "swa")], dtype = torch.int32)
        return params
