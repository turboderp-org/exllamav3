from __future__ import annotations
from functools import lru_cache
import torch
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from ..cache.cache import Cache
if TYPE_CHECKING:
    from .generator import Generator
from ..constants import PAGE_SIZE
from collections import deque, defaultdict
from itertools import pairwise
from ..util.tensor import SeqTensor
from exllamav3.ext import exllamav3_ext as ext
import time
from ..cache import RecurrentCache
from ..util import profile_opt


def _tensor_blake2b_checksum(tensor: torch.Tensor, prev_hash: bytes | None) -> bytes:
    hasher = hashlib.blake2b(digest_size = 16)
    if prev_hash is not None:
        hasher.update(prev_hash)
    hasher.update(tensor.numpy().tobytes())
    return hasher.digest()

_uniquehash = 0
def _randomhash():
    global _uniquehash
    _uniquehash += 1
    return _uniquehash.to_bytes(16, byteorder = 'big')

tensor_hash_checksum = _tensor_blake2b_checksum
random_hash = _randomhash


@dataclass
class CachePage:

    arena: PageArena
    page_index: int

    # Hash of this page if kv_position == PAGE_SIZE, else random hash. Also used to index (un)referenced_pages
    phash: bytes
    phash_revert: bytes

    # Hash of previous page in chain
    prev_hash: bytes | None
    prev_hash_revert: bytes | None

    # Number of active jobs referencing page
    ref_count: int

    # Last time this page was assigned to a job
    access_serial: int
    access_serial_revert: int

    # Number of tokens in page for which KV is valid assuming prev_hash
    kv_position: int
    kv_position_revert: int

    # Specific tokens for which KV is valid assuming prev_hash
    sequence: torch.Tensor
    can_revert: bool

    # Used by defragmenter
    new_page_index: int
    children: list[CachePage]
    longest_chain: int

    def __repr__(self):
        return (
            f"CachePage: idx = {self.page_index}, ref_count = {self.ref_count}, "
            f"phash: ..{str(self.phash)[8:24]}.., prev_hash: ..{str(self.prev_hash)[8:24]}.., "
            f"kvp {self.kv_position}"
        )

    # Copy page state so page can be reverted
    def backup(self):
        self.phash_revert = self.phash
        self.prev_hash_revert = self.prev_hash
        self.access_serial_revert = self.access_serial
        self.kv_position_revert = self.kv_position
        self.can_revert = True

    # Reuse unreferenced page
    def revert(self):
        assert self.can_revert
        self.phash = self.phash_revert
        self.prev_hash = self.prev_hash_revert
        self.access_serial = self.access_serial_revert
        self.kv_position = self.kv_position_revert
        self.can_revert = False

    # Increase reference count
    def add_ref(self, serial):
        if self.ref_count == 0:
            del self.arena.unreferenced_pages[self.phash]
            assert self.phash not in self.arena.referenced_pages
            self.arena.referenced_pages[self.phash] = self
        self.ref_count += 1
        self.access_serial = max(serial, self.access_serial)
        self.can_revert = False

    # Increase reference count and clear page
    def add_ref_clear(self, serial, newhash):
        assert self.ref_count == 0
        del self.arena.unreferenced_pages[self.phash]
        self.phash = newhash
        assert self.phash not in self.arena.referenced_pages
        self.arena.referenced_pages[self.phash] = self
        self.ref_count += 1
        self.access_serial = serial
        self.prev_hash = None
        self.can_revert = False
        self.kv_position = 0

    # Add reference to (currently) unique page
    def add_ref_unique(self, serial):
        self.backup()
        assert self.ref_count == 0
        del self.arena.unreferenced_pages[self.phash]
        self.phash = _randomhash()
        assert self.phash not in self.arena.referenced_pages
        self.arena.referenced_pages[self.phash] = self
        self.ref_count += 1
        self.access_serial = serial
        self.prev_hash = None
        self.kv_position = 0

    # Decrease reference count
    def sub_ref(self):
        self.ref_count -= 1
        if self.ref_count == 0:
            del self.arena.referenced_pages[self.phash]
            if self.can_revert:
                self.revert()
            if self.phash in self.arena.referenced_pages or self.phash in self.arena.unreferenced_pages:
                self.phash = _randomhash()
                self.prev_hash = None
            assert self.phash not in self.arena.unreferenced_pages
            self.arena.unreferenced_pages[self.phash] = self

    # Clear page
    def clear(self):
        assert self.ref_count == 0
        del self.arena.unreferenced_pages[self.phash]
        self.phash = _randomhash()
        self.prev_hash = None
        self.kv_position = 0
        self.can_revert = False
        self.sequence[:, :] = 0
        assert self.phash not in self.arena.unreferenced_pages
        self.arena.unreferenced_pages[self.phash] = self

    # Update hash
    def update_hash(self, newhash = None):
        if newhash is None:
            newhash = tensor_hash_checksum(self.sequence, self.prev_hash)
        assert self.ref_count > 0
        assert self.kv_position == PAGE_SIZE
        del self.arena.referenced_pages[self.phash]
        self.phash = newhash
        self.can_revert = False
        assert self.phash not in self.arena.referenced_pages
        self.arena.referenced_pages[self.phash] = self

    # Clear allocated page to repeat prefill
    def make_unique(self):
        assert self.ref_count > 0
        del self.arena.referenced_pages[self.phash]
        self.phash = _randomhash()
        assert self.phash not in self.arena.referenced_pages
        self.arena.referenced_pages[self.phash] = self
        self.prev_hash = None
        self.kv_position = 0


class PageArena:

    def __init__(self, pagetable: PageTable, max_pages: int, role: str = "default"):
        self.pagetable = pagetable
        self.role = role
        self.max_pages = max_pages
        self.access_serial = max_pages
        self.referenced_pages = {}
        self.unreferenced_pages = {}
        self.all_pages = []
        self.last_defrag_serial = max_pages
        self.reset()

    def reset(self):
        self.referenced_pages = {}
        self.unreferenced_pages = {}
        self.all_pages = []
        for idx in range(self.max_pages):
            h = _randomhash()
            cp = CachePage(
                arena = self,
                page_index = idx,
                phash = h,
                phash_revert = h,
                prev_hash = None,
                prev_hash_revert = None,
                sequence = torch.empty((1, PAGE_SIZE), dtype = torch.long),
                ref_count = 0,
                access_serial = idx,
                access_serial_revert = idx,
                kv_position = 0,
                kv_position_revert = 0,
                can_revert = False,
                new_page_index = 0,
                children = [],
                longest_chain = 1,
            )
            self.all_pages.append(cp)
            self.unreferenced_pages[h] = cp
        self.access_serial = self.max_pages
        self.last_defrag_serial = self.access_serial


class Sequence:

    def __init__(self, ids: torch.Tensor, seq_ids: torch.Tensor):
        self.input_ids = SeqTensor.from_tensor(ids, seq_dim = -1)
        self.sequence_ids = SeqTensor.from_tensor(seq_ids, seq_dim = -1)
        self.kv_position = 0
        self.page_hashes = None
        self.new_unique_pages = 0
        self.allocated_pages = None
        self.block_index_tensor = None
        self.live = True
        self.prefill_complete = False


    def prepare(self, has_prefix_token: bool, max_new_tokens: int, allow_page_reuse: bool = True):
        self.page_hashes = []
        unique_hashes = set()

        max_len = len(self.sequence_ids) + max_new_tokens
        if has_prefix_token: max_len += 1
        context_pages = (len(self.sequence_ids) - 1) // PAGE_SIZE
        total_pages = (max_len + PAGE_SIZE - 1) // PAGE_SIZE

        r_hash = None
        for i in range(context_pages):
            # TODO: profile/optimize hash function
            if allow_page_reuse:
                page_ids = self.sequence_ids.torch_slice(i * PAGE_SIZE, (i + 1) * PAGE_SIZE)
                assert page_ids.shape[-1] == PAGE_SIZE
                r_hash = tensor_hash_checksum(page_ids, r_hash)
            else:
                r_hash = _randomhash()
            self.page_hashes.append(r_hash)
            unique_hashes.add(r_hash)

        self.new_unique_pages = total_pages - context_pages
        return unique_hashes, self.new_unique_pages

    def build_block_index_tensor(self):
        if self.allocated_pages is None:
            self.block_index_tensor = None
        else:
            self.block_index_tensor = torch.tensor(
                [[page.page_index for page in self.allocated_pages]],
                dtype = torch.int32,
            )
        return self.block_index_tensor

    def allocate_pages(self, pagetable: PageTable, recurrent_cache: None | RecurrentCache):
        # If recurrent model, find logest recurrent prefix
        recurrent_pages = None
        if recurrent_cache is not None:
            recurrent_pages = []
            for pi, ph in enumerate(self.page_hashes):
                rs = recurrent_cache.get(ph)
                if rs:
                    recurrent_pages.append(pi)

        # Allocate pages in KV cache, limit prefix caching to available recurrent states
        self.allocated_pages, self.kv_position, cached_pages, non_sequential_pages = \
            pagetable.allocate_pages(self.page_hashes, self.new_unique_pages, recurrent_pages)

        # Prepare block index
        self.build_block_index_tensor()

        # If recurrent model, grab cached state for prefix length
        recurrent_state = None
        if recurrent_cache is not None:
            if cached_pages > 0:
                recurrent_state = recurrent_cache.get(self.page_hashes[cached_pages - 1])
                assert recurrent_state is not None, "Failed to get cached recurrent state"

        return len(self.allocated_pages), cached_pages, non_sequential_pages, recurrent_state


class PageTable:

    def __init__(
        self,
        generator: Generator,
        cache: Cache
    ):
        self.generator = generator
        self.cache = cache
        self.default_arena = PageArena(self, cache.max_num_tokens // PAGE_SIZE, role = "default")

    @property
    def max_pages(self):
        return self.default_arena.max_pages

    @property
    def access_serial(self):
        return self.default_arena.access_serial

    @access_serial.setter
    def access_serial(self, value):
        self.default_arena.access_serial = value

    @property
    def referenced_pages(self):
        return self.default_arena.referenced_pages

    @referenced_pages.setter
    def referenced_pages(self, value):
        self.default_arena.referenced_pages = value

    @property
    def unreferenced_pages(self):
        return self.default_arena.unreferenced_pages

    @unreferenced_pages.setter
    def unreferenced_pages(self, value):
        self.default_arena.unreferenced_pages = value

    @property
    def all_pages(self):
        return self.default_arena.all_pages

    @all_pages.setter
    def all_pages(self, value):
        self.default_arena.all_pages = value

    @property
    def last_defrag_serial(self):
        return self.default_arena.last_defrag_serial

    @last_defrag_serial.setter
    def last_defrag_serial(self, value):
        self.default_arena.last_defrag_serial = value

    def build_batch_block_index(
        self,
        active_jobs: list,
        max_seq_len: int,
        role: str = "default",
        single_sequence_per_job: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if role != "default":
            raise KeyError(f"Unknown page arena role: {role}")
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
                seq_block_index = seq.block_index_tensor
                if seq_block_index is None:
                    continue
                seq_block_index = seq_block_index[:, :max_pages_batch]
                block_index[batch : batch + 1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1
                if single_sequence_per_job:
                    break

        return block_index, cache_seqlens

    def build_draft_decode_params(self, active_jobs: list, max_seq_len: int) -> dict:
        # The default pagetable exposes the same single-cache decode metadata the
        # generator used inline before custom pagetables were introduced.
        block_index, cache_seqlens = self.build_batch_block_index(
            active_jobs,
            max_seq_len,
            role = "default",
            single_sequence_per_job = True,
        )
        return {
            "block_table": block_index,
            "cache_seqlens": cache_seqlens,
        }

    def advance_draft_decode_params(self, params: dict, step: int = 1) -> None:
        # Default pagetable still advances the original single cache-seqlens
        # tensor only. Custom pagetables can extend this when they expose
        # additional role-aware seqlens tensors.
        params["cache_seqlens"] += step

    def build_decode_params(self, active_jobs: list, max_seq_len: int, use_offsets: bool = False) -> dict:
        # Non-Gemma models stay on this original single-cache path. Custom
        # pagetables can extend it with extra role-aware tables while keeping the
        # default contract intact for every other architecture.
        block_index, cache_seqlens = self.build_batch_block_index(
            active_jobs,
            max_seq_len,
            role = "default",
            single_sequence_per_job = False,
        )
        positions = torch.zeros_like(cache_seqlens) if use_offsets else None

        batch = 0
        for job in active_jobs:
            if not job.is_prefill_done():
                continue
            for seq in job.sequences:
                if positions is not None:
                    positions[batch] = seq.kv_position + job.alt_rope_offset
                batch += 1

        params = {
            "block_table": block_index,
            "cache_seqlens": cache_seqlens,
        }
        if positions is not None:
            params["positions"] = positions
        return params

    def build_prefill_params(self, seq: Sequence, prefill_start: int) -> dict:
        # Default prefill still uses one block table and one cache_seqlens tensor.
        return {
            "block_table": seq.block_index_tensor,
            "cache_seqlens": torch.tensor([prefill_start], dtype = torch.int32),
        }

    def build_cache_copy_plan(
        self,
        source_page: CachePage,
        target_page: CachePage,
        num_tokens: int,
        seq: Sequence | None = None,
    ) -> dict[str, tuple[int, int]] | bool | None:
        # The default pagetable keeps the original single-cache copy behavior. Custom
        # pagetables can return a role-aware plan when they need extra copy targets.
        return False

    def get_arena(self, role: str = "default") -> PageArena:
        if role != "default":
            raise KeyError(f"Unknown page arena role: {role}")
        return self.default_arena

    def num_unreferenced_pages_by_role(self) -> dict[str, int]:
        return { "default": self.num_unreferenced_pages() }

    def current_new_pages_required_by_role(self, job) -> dict[str, int]:
        return { "default": self.current_new_pages_required(job) }

    def can_admit_job(self, job, current_batch_size: int, max_batch_size: int) -> bool:
        if len(job.sequences) + current_batch_size > max_batch_size:
            return False
        required = self.current_new_pages_required_by_role(job)
        available = self.num_unreferenced_pages_by_role()
        for role, pages in required.items():
            if pages > available.get(role, 0):
                return False
        return True

    def get_visualizer_num_pages(self) -> int:
        return self.get_arena("default").max_pages

    def get_visualizer_state(self, active_jobs: list) -> tuple[list[tuple[int, list[int]]], list[float]]:
        chains = []
        for job in active_jobs:
            for seq in job.sequences:
                idx = job.serial_number
                pages = seq.allocated_pages or []
                chain = [page.page_index for page in pages]
                chains.append((idx, chain))
        usage = [0.0] * self.get_visualizer_num_pages()
        for page in self.all_pages:
            usage[page.page_index] = page.kv_position / PAGE_SIZE
        return chains, usage

    def get_max_pages_for_job(self, job) -> int:
        return self.get_arena("default").max_pages

    def find_referenced_page(self, page_hash: bytes, role: str = "default"):
        return self.get_arena(role).referenced_pages.get(page_hash)

    def find_unreferenced_page(self, page_hash: bytes, role: str = "default"):
        return self.get_arena(role).unreferenced_pages.get(page_hash)

    def iter_pages(self, role: str = "default"):
        return iter(self.get_arena(role).all_pages)

    def prepare_sequence(
        self,
        seq: Sequence,
        has_prefix_token: bool,
        max_new_tokens: int,
        allow_page_reuse: bool = True,
    ):
        return seq.prepare(has_prefix_token, max_new_tokens, allow_page_reuse = allow_page_reuse)

    def clamp_prefill_end(
        self,
        seq: Sequence,
        prefill_start: int,
        prefill_end: int,
        embeddings = None,
        atomic_mm_prefill: bool = False,
    ) -> int:
        return prefill_end

    def sync_sequence_views(self, seq: Sequence):
        # Default pagetable has only one logical cache view, so there is nothing to
        # synchronize after page mutations.
        pass

    def cache_prefill_copy_source(self, seq: Sequence):
        # Default pagetable does not snapshot role-specific local pages after prefill.
        pass

    def clear_page(self, page: CachePage, role: str = "default"):
        page.clear()

    def allocate_sequence(self, seq: Sequence, recurrent_cache: None | RecurrentCache):
        return seq.allocate_pages(self, recurrent_cache)

    def deallocate_sequence(self, seq: Sequence):
        if seq.allocated_pages is not None:
            self.deallocate_pages(seq.allocated_pages)
            seq.allocated_pages = []

    def current_new_pages_required(self, job) -> int:
        arena = self.get_arena("default")
        new_pages = 0
        for h in job.all_unique_hashes:
            if h not in arena.referenced_pages:
                new_pages += 1
        for s in job.sequences:
            new_pages += s.new_unique_pages
        return new_pages


    def reset_page_table(self):
        """
        Reset the page table.
        """
        self.default_arena.reset()


    def print_page_list(self, short: bool = True):
        arena = self.get_arena("default")
        for cp in arena.all_pages:
            if cp.phash in arena.referenced_pages:
                assert cp.ref_count > 0
                ref = str(cp.ref_count) if cp.ref_count < 10 else "+"
            elif cp.phash in arena.unreferenced_pages:
                assert cp.ref_count == 0
                ref = "."
            else:
                ref = "#"
            if short: print(ref, end = "")
            else: print(str(cp) + f", ref {ref}")
        print()


    def allocate_pages(
        self,
        page_hashes: list,
        new_unique_pages: int,
        recurrent_pages: list[int] | None,
        role: str = "default",
    ):
        arena = self.get_arena(role)
        allocated_pages = []
        available_pages = None

        # Allocate whole pages
        for lp, h in enumerate(page_hashes):
            arena.access_serial += 1

            # Find matching referenced page
            rp = arena.referenced_pages.get(h)
            if rp:
                rp.add_ref(arena.access_serial)
                allocated_pages.append(rp)

            # If possible, reuse an unreferenced page with matching hash
            else:
                up = arena.unreferenced_pages.get(h)
                if up:
                    up.add_ref(arena.access_serial)
                    allocated_pages.append(up)

                # No matching pages
                else:

                    # Get list of unreferenced pages in order of oldest to newest
                    if available_pages is None:
                        available_pages = list(arena.unreferenced_pages.values())
                        available_pages.sort(key = lambda x: x.access_serial)
                        available_pages = deque(available_pages)
                    else:
                        while available_pages[0].ref_count:
                            available_pages.popleft()

                    # Allocate oldest unreferenced page
                    op = available_pages.popleft()
                    op.add_ref_clear(arena.access_serial, h)
                    allocated_pages.append(op)

        # Allocate unique pages
        for npi in range(new_unique_pages):
            arena.access_serial += 1

            # Get list of unreferenced pages in order of oldest to newest
            if available_pages is None:
                available_pages = list(arena.unreferenced_pages.values())
                available_pages.sort(key = lambda x: x.access_serial)
                available_pages = deque(available_pages)
            else:
                while available_pages[0].ref_count:
                    available_pages.popleft()

            op = available_pages.popleft()
            op.add_ref_unique(arena.access_serial)
            allocated_pages.append(op)

        # List prefilled pages
        cached_pages = 0
        for page in allocated_pages:
            if page.kv_position == PAGE_SIZE:
                cached_pages += 1
            else:
                break

        # If recurrent cache used, roll back to longest prefix, clear subsequent pages
        if recurrent_pages is not None:
            max_recur = 0
            for rp in recurrent_pages:
                if rp < cached_pages:
                    max_recur = rp + 1
            # for cpi in range(max_recur, cached_pages):
            #     allocated_pages[cpi].make_unique()
            cached_pages = max_recur

        # Advance cache over prefilled pages
        kv_position = cached_pages * PAGE_SIZE

        non_sequential_pages = 0
        for page_a, page_b in pairwise(allocated_pages):
            if page_b.page_index != page_a.page_index + 1:
                non_sequential_pages += 1

        return allocated_pages, kv_position, cached_pages, non_sequential_pages


    def deallocate_pages(self, allocated_pages: list):
        for page in allocated_pages:
            page.sub_ref()


    def num_unreferenced_pages(self):
        return len(self.get_arena("default").unreferenced_pages)


    def validate_pagetable(self, active_jobs):
        arena = self.get_arena("default")

        def p_assert(exp):
            # assert exp
            if not exp:
                xx = 0

        # Check page collections
        ids = set()
        for p in arena.referenced_pages.values():
            p_assert(p.ref_count > 0)
            ids.add(id(p))
        for p in arena.unreferenced_pages.values():
            p_assert(p.ref_count == 0)
            ids.add(id(p))
        p_assert(len(ids) == arena.max_pages)
        p_assert(len(arena.all_pages) == arena.max_pages)

        # Check job reference counts
        refcounts = [0] * arena.max_pages
        for job in active_jobs:
            for seq in job.sequences:
                for page in seq.allocated_pages:
                    refcounts[page.page_index] += 1

        for page in arena.all_pages:
            p_assert(page.ref_count == refcounts[page.page_index])

        # Check that all hashes are unique
        hashes = set()
        for page in arena.all_pages:
            p_assert(page.phash not in hashes)
            hashes.add(page.phash)
        p_assert(len(hashes) == arena.max_pages)

        # Check individual hashes
        for page in arena.all_pages:
            if page.kv_position == PAGE_SIZE and page.phash[:8] != b'\x00\x00\x00\x00\x00\x00\x00\x00':
                h = tensor_hash_checksum(page.sequence, page.prev_hash)
                p_assert(page.phash == h)

        # Check job sequences
        for job in active_jobs:
            for seq in job.sequences:
                k, j = 0, 0
                while j < seq.kv_position:
                    i, j = j, min(j + PAGE_SIZE, seq.kv_position)
                    jobt = seq.sequence_ids.torch()[:, i : j]
                    paget = seq.allocated_pages[k].sequence[:, 0 : j - i]
                    p_assert(torch.equal(jobt, paget))
                    k += 1


    def defrag(self, debug = False):
        arena = self.get_arena("default")

        if not self.generator.enable_defrag:
            return

        # Defragment once job queue is empty and all pages have been touched at least once
        if arena.access_serial < arena.last_defrag_serial + arena.max_pages * 8:
            return
        arena.last_defrag_serial = arena.access_serial

        assert not arena.referenced_pages

        if debug:
            torch.cuda.synchronize()
            time_begin = time.time()

        # Build page index
        page_index = {}
        def build_page_index():
            nonlocal page_index
            page_index = {}
            for page in arena.all_pages:
                page_index[page.phash] = page
                page.children = []
                page.longest_chain = 1
        build_page_index()

        # Find cached sequences that can be recovered
        root_pages = []
        def build_root_pages():
            nonlocal root_pages
            root_pages = []
            for page in arena.all_pages:
                if page.prev_hash is None:
                    root_pages.append(page)
                else:
                    parent = page_index.get(page.prev_hash)
                    if parent is not None:
                        parent.children.append(page)
        build_root_pages()

        # Measure recoverable sequence length
        def measure(p):
            p.longest_chain = 1
            if p.children:
                p.longest_chain += max([measure(pc) for pc in p.children])
            return p.longest_chain

        def measure_iterative(root):
            stack = [(root, False)]
            while stack:
                node, visited = stack.pop()
                if not visited:
                    stack.append((node, True))
                    for child in node.children:
                        stack.append((child, False))
                else:
                    if node.children:
                        node.longest_chain = 1 + max(child.longest_chain for child in node.children)
                    else:
                        node.longest_chain = 1
            return root.longest_chain

        for page in root_pages:
            measure_iterative(page)

        # Recursively sort branches by length
        def sort_seq(p):
            if len(p.children) > 1:
                p.children = sorted(p.children, key = lambda x: x.longest_chain, reverse = True)
            for pc in p.children:
                sort_seq(pc)

        def sort_seq_iterative(root):
            stack = [root]
            while stack:
                node = stack.pop()
                if len(node.children) > 1:
                    node.children = sorted(node.children, key = lambda x: x.longest_chain, reverse = True)
                stack.extend(node.children)

        for page in root_pages:
            sort_seq_iterative(page)

        # Process roots in order of increasing age
        root_pages = sorted(root_pages, key = lambda x: x.access_serial)

        # Maintain the longest sequence for each tree and create new root nodes from trimmed branches
        index = 0
        while index < len(root_pages):
            page = root_pages[index]
            while page.children:
                root_pages += page.children[1:]
                page.children = page.children[:1]
                page = page.children[0]
            index += 1

        # Reorder partial sequences into the longest possible contiguous strings
        new_page_index = 0
        shift_counts = defaultdict(int)
        non_orphaned_pages = []
        orphans = page_index
        for page in root_pages:
            while True:
                non_orphaned_pages.append(page)
                del orphans[page.phash]
                page.new_page_index = new_page_index
                shift = page.new_page_index - page.page_index
                shift_counts[shift] += 1
                new_page_index += 1
                if not page.children:
                    break
                page = page.children[0]

        # Move orphans to end of cache, ordered by last access
        if orphans:
            orphans = list(orphans.values())
            orphans = sorted(orphans, key = lambda x: x.page_index)
            access_serials = [page.access_serial for page in orphans]
            access_serials = sorted(access_serials)
            for page, access_serial in zip(orphans, access_serials):
                page.access_serial = access_serial
                page.new_page_index = new_page_index
                shift = page.new_page_index - page.page_index
                shift_counts[shift] += 1
                new_page_index += 1

        assert new_page_index == arena.max_pages

        # Adjust overall shift to minimize page copies
        shift_adjust = max(shift_counts, key = shift_counts.get)

        # Order of operations
        if debug:
            print("Page shifts")

        defrag_map = {}
        for page in arena.all_pages:
            page.new_page_index = (page.new_page_index - shift_adjust + arena.max_pages) % arena.max_pages
            if page.page_index != page.new_page_index:
                defrag_map[page.new_page_index] = page.page_index
                if debug:
                    print(f"{page.new_page_index:2} ← {page.page_index:2}")

        # Don't bother if less than 10% of cache is fragmented
        if len(defrag_map) <= max(arena.max_pages // 10, 2):
            return

        # Find page rotations
        if debug:
            print("Page rotations")

        all_rotations = []
        while defrag_map:

            # Get first dst,src pair in new loop
            dst = next(iter(defrag_map))
            src = defrag_map[dst]
            del defrag_map[dst]
            rotation = [dst, src]

            # Walk around loop
            while True:
                if src == rotation[0]:
                    rotation = [-1, src] + rotation[:-1] + [-1]
                    all_rotations += rotation
                    break
                dst = src
                src = defrag_map[dst]
                del defrag_map[dst]
                rotation += [dst, src]

            if debug:
                print(" ← ".join([".."] + [f"{rotation[i + 1]:2}" for i in range(0, len(rotation) - 2, 2)] + [".."]))

        # Rotate pages
        all_rotations_cpu = torch.tensor(all_rotations, dtype = torch.int)
        @lru_cache
        def get_all_rotations(device):
            nonlocal all_rotations_cpu
            return all_rotations_cpu.to(device)

        @lru_cache
        def get_buffer(shape, device, dtype):
            return torch.empty(shape, device = device, dtype = dtype)

        if self.generator.model.loaded_tp:
            self.generator.model.tp_rotate_cache_pages(id(self.cache), all_rotations_cpu)
        else:
            cache_tensors = self.cache.get_all_tensors()
            for cache in cache_tensors:
                buffer = get_buffer(cache[0].shape, cache.device, cache.dtype)
                all_rotations = get_all_rotations(cache.device)
                ext.cache_rotate(cache, all_rotations, buffer)

        # Write new page indices
        for page in arena.all_pages:
            page.page_index = page.new_page_index

        # Debug stuff
        if debug:
            build_page_index()
            build_root_pages()

            def dbg_walk(l, p):
                nonlocal walks
                l = l + [p]
                if not p.children:
                    walks.append(l)
                else:
                    for p in p.children:
                        dbg_walk(l, p)

            print("Cache seqs")
            for page in root_pages:
                walks = []
                dbg_walk([], page)
                for pp in walks:
                    print(" → ".join([f"{p.page_index:2}" for p in pp]))

            torch.cuda.synchronize()
            elapsed = time.time() - time_begin
            print(f"Defrag latency: {elapsed:.5f} s")
