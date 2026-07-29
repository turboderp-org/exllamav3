from __future__ import annotations
from functools import lru_cache
import heapq
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
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
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

def is_content_hash(h: bytes) -> bool:
    # Random (placeholder) hashes are counter values whose top eight bytes stay zero; a blake2b content
    # hash matches that pattern with vanishing probability
    return h[:8] != bytes(8)


@dataclass
class CachePage:

    pagetable: PageTable
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

    # Format page for debug output (steady state only)
    def format(self):
        pagetable = self.pagetable
        tokenizer = pagetable.generator.tokenizer
        f = f"{self.page_index:3}"
        if self.prev_hash is None:
            f += f" - prev: rt "
        else:
            prev_page = self.pagetable.referenced_pages.get(self.prev_hash) or self.pagetable.unreferenced_pages.get(self.prev_hash)
            if prev_page:
                f += f" - prev: {prev_page.page_index:3}"
            else:
                f += f" - prev:    "
        f += f" - kvpr: {self.kv_position:3}"
        text = repr(tokenizer.decode(self.sequence[0, :self.kv_position], decode_special_tokens = True))
        if len(text) <= 100:
            f += " " + text + " " * (100 - len(text))
        else:
            f += " " + text[:47] + " .... " + text[-47:]
        f += f" - ser: {self.access_serial:5}"
        return f

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
            del self.pagetable.unreferenced_pages[self.phash]
            assert self.phash not in self.pagetable.referenced_pages
            self.pagetable.referenced_pages[self.phash] = self
        self.ref_count += 1
        self.access_serial = max(serial, self.access_serial)
        self.can_revert = False

    # Increase reference count and clear page
    def add_ref_clear(self, serial, newhash):
        assert self.ref_count == 0
        del self.pagetable.unreferenced_pages[self.phash]
        self.phash = newhash
        assert self.phash not in self.pagetable.referenced_pages
        self.pagetable.referenced_pages[self.phash] = self
        self.ref_count += 1
        self.access_serial = serial
        self.prev_hash = None
        self.can_revert = False
        self.kv_position = 0

    # Add reference to (currently) unique page
    def add_ref_unique(self, serial):
        self.backup()
        assert self.ref_count == 0
        del self.pagetable.unreferenced_pages[self.phash]
        self.phash = _randomhash()
        assert self.phash not in self.pagetable.referenced_pages
        self.pagetable.referenced_pages[self.phash] = self
        self.ref_count += 1
        self.access_serial = serial
        self.prev_hash = None
        self.kv_position = 0

    # Decrease reference count
    def sub_ref(self):
        self.ref_count -= 1
        if self.ref_count == 0:
            del self.pagetable.referenced_pages[self.phash]
            if self.can_revert:
                self.revert()
            if self.phash in self.pagetable.referenced_pages or self.phash in self.pagetable.unreferenced_pages:
                self.phash = _randomhash()
                self.prev_hash = None
            assert self.phash not in self.pagetable.unreferenced_pages
            self.pagetable.unreferenced_pages[self.phash] = self

    # Clear page
    def clear(self):
        assert self.ref_count == 0
        del self.pagetable.unreferenced_pages[self.phash]
        self.phash = _randomhash()
        self.prev_hash = None
        self.kv_position = 0
        self.can_revert = False
        self.sequence[:, :] = 0
        assert self.phash not in self.pagetable.unreferenced_pages
        self.pagetable.unreferenced_pages[self.phash] = self

    # Update hash
    def update_hash(self, newhash = None):
        if newhash is None:
            newhash = tensor_hash_checksum(self.sequence, self.prev_hash)
        assert self.ref_count > 0
        assert self.kv_position == PAGE_SIZE
        del self.pagetable.referenced_pages[self.phash]
        self.phash = newhash
        self.can_revert = False
        assert self.phash not in self.pagetable.referenced_pages
        self.pagetable.referenced_pages[self.phash] = self

    # Clear allocated page to repeat prefill
    def make_unique(self):
        assert self.ref_count > 0
        del self.pagetable.referenced_pages[self.phash]
        self.phash = _randomhash()
        assert self.phash not in self.pagetable.referenced_pages
        self.pagetable.referenced_pages[self.phash] = self
        self.prev_hash = None
        self.kv_position = 0


class Sequence:

    def __init__(self, ids: torch.Tensor, seq_ids: torch.Tensor):
        self.input_ids = SeqTensor.from_tensor(ids, seq_dim = -1)
        self.sequence_ids = SeqTensor.from_tensor(seq_ids, seq_dim = -1)
        self.kv_position = 0
        self.page_hashes = None
        self.max_cached_pages = None
        self.new_unique_pages = 0
        self.allocated_pages = None
        self.block_index_tensor = None
        self.live = True
        self.prefill_complete = False

        # Multimodal token spans
        self.multimodal_mask = ids[0] >= FIRST_MM_EMBEDDING_INDEX

        # MTP carry hidden — last hidden state of the previous prefill chunk, used to seed
        # the prev_hidden shift in update_kv_from_target. None until first prefill chunk runs.
        self.mtp_carry_hidden = None


    def prepare(self, has_prefix_token: bool, max_new_tokens: int):
        self.page_hashes = []
        unique_hashes = set()

        max_len = len(self.sequence_ids) + max_new_tokens
        if has_prefix_token: max_len += 1
        context_pages = (len(self.sequence_ids) - 1) // PAGE_SIZE
        total_pages = (max_len + PAGE_SIZE - 1) // PAGE_SIZE

        r_hash = None
        for i in range(context_pages):
            # TODO: profile/optimize hash function
            page_ids = self.sequence_ids.torch_slice(i * PAGE_SIZE, (i + 1) * PAGE_SIZE)
            assert page_ids.shape[-1] == PAGE_SIZE
            r_hash = tensor_hash_checksum(page_ids, r_hash)
            self.page_hashes.append(r_hash)
            unique_hashes.add(r_hash)

        self.new_unique_pages = total_pages - context_pages
        return unique_hashes, self.new_unique_pages

    def build_block_index_tensor(self):
        self.block_index_tensor = torch.tensor(
            [[page.page_index for page in self.allocated_pages]],
            dtype = torch.int32,
        )

    def allocate_pages(
        self,
        pagetable: PageTable,
        recurrent_cache: None | RecurrentCache,
        protected_hashes: set | None = None,
    ):
        if self.max_cached_pages is None:
            page_hashes = self.page_hashes
        else:
            page_hashes = self.page_hashes[:self.max_cached_pages]
        new_unique_pages = self.new_unique_pages + len(self.page_hashes) - len(page_hashes)

        # If recurrent model, find logest recurrent prefix
        recurrent_pages = None
        restore_limit = None
        if recurrent_cache is not None:
            recurrent_pages = []
            for pi, ph in enumerate(page_hashes):
                rs = recurrent_cache.get(ph)
                if rs:
                    recurrent_pages.append(pi)
            # CPU-tier restores past the last checkpoint can't advance the resume point; replay prefill
            # rewrites those pages anyway
            restore_limit = (max(recurrent_pages) + 1) if recurrent_pages else 0

        # Allocate pages in KV cache, limit prefix caching to available recurrent states
        self.allocated_pages, self.kv_position, cached_pages, non_sequential_pages = \
            pagetable.allocate_pages(page_hashes, new_unique_pages, recurrent_pages, protected_hashes, restore_limit)

        # Prepare block index
        self.build_block_index_tensor()

        # If recurrent model, grab cached state for prefix length
        stashed_recurrent_state = None
        if recurrent_cache is not None:
            if cached_pages > 0:
                stashed_recurrent_state = recurrent_cache.get_stashed(page_hashes[cached_pages - 1])
                assert stashed_recurrent_state is not None, "Failed to get cached recurrent state"

        return len(self.allocated_pages), cached_pages, non_sequential_pages, stashed_recurrent_state


class PageTable:

    def __init__(
        self,
        generator: Generator,
        cache: Cache
    ):
        """
        Manage the physical cache pages backing prompt and generation state.

        Completed pages are keyed by a chained hash of their token contents and previous page hash, forming an
        implicit prefix tree: sequences with the same prefix resolve to the same chain of CachePage objects and can
        share K/V storage during batched inference. Hash collisions are deliberately treated as impossible in
        practice for this purpose; checking full token contents on every lookup would cost more than the vanishing
        collision risk justifies. When a page becomes unreferenced after inference it remains indexed by its hash,
        so later jobs can revive it for prompt-cache reuse until eviction or defragmentation overwrites it.
        Eviction consumes empty and orphaned pages first, then prunes cached sequences tail-first, least recently
        used sequence first (see build_eviction_order).
        """
        self.generator = generator
        self.cache = cache
        self.max_pages = cache.max_num_tokens // PAGE_SIZE

        self.access_serial = self.max_pages
        self.referenced_pages = {}
        self.unreferenced_pages = {}
        self.all_pages = []
        self.reset_page_table()
        self.last_defrag_serial = self.max_pages

        # "tree" = prune cached sequences tail-first, oldest first; "lru" = legacy oldest-page-first policy,
        # kept for comparison/debug
        self.eviction_policy = "tree"

        # Optional second-tier page cache in system memory (CPUPageCache), set by the Generator. Complete pages
        # are pushed there on eviction and restored from there on allocation
        self.cpu_tier = None

        # Cheap always-on counters for cache efficiency analysis
        self.metrics = {
            "evictions": 0,               # unreferenced pages repurposed for new sequences
            "evictions_live": 0,          # of those, complete hashed pages (i.e. lost prompt-cache entries)
            "stashes_stranded": 0,        # live evictions that anchored a recurrent checkpoint, making it unusable
            "alloc_pages": 0,             # pages claimed by starting jobs
            "alloc_cached_pages": 0,      # of those, reused as part of a resumable cached prefix
            "alloc_tier_pages": 0,        # of those, restored from the CPU tier rather than found in VRAM
            "alloc_kv_only_pages": 0,     # cached KV pages not resumable because no recurrent stash covers them
        }


    def reset_page_table(self):
        """
        Reset the page table.
        """
        self.referenced_pages = {}
        self.unreferenced_pages = {}
        self.all_pages = []
        for idx in range(self.max_pages):
            h = _randomhash()
            cp = CachePage(
                pagetable = self,
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


    def dump_page_list(self, short: bool = True):
        return "\n".join([cp.format() for cp in self.all_pages])


    def build_eviction_order(self, protected_hashes: set | None = None) -> deque:
        """
        Order the currently unreferenced pages by eviction priority, least valuable first:

          1. Empty pages (kv_position == 0), oldest first: they hold no reusable K/V.
          2. Orphaned trees: chains whose ancestry is broken (prev_hash no longer resolves to a live page).
             Individual pages remain reachable by exact hash lookup, but the broken chain makes them the least
             valuable intact data in the cache.
          3. Rooted trees, i.e. complete cached sequences.

        Trees are consumed one at a time, least recently used root first; since any reuse of a cached prefix
        refreshes the root's access serial, root age reflects the recency of the whole tree. Within a tree, pages
        are pruned leaf-first, oldest leaf first, so a cached sequence is eaten from its tail and the longest
        possible prefix survives. When one job reclaims pages from a paused long sequence, the paused job loses
        only as many trailing pages as were actually taken; under the old oldest-page-first policy the root was
        typically taken first, making the entire chain unrecoverable and forcing a full prefill on resume.

        Pages whose hash appears in protected_hashes are pages the current allocation is itself about to claim,
        and are excluded so that serving early cache misses cannot cannibalize later hits of the same sequence.
        Anything the tree walk cannot emit (pages below a referenced or protected ancestor, in principle hash
        cycles) is appended at the end, protected pages dead last, so allocation never starves while unreferenced
        pages exist.
        """
        if protected_hashes is None:
            protected_hashes = set()

        if self.eviction_policy == "lru":
            order = [p for p in self.unreferenced_pages.values() if p.phash not in protected_hashes]
            order.sort(key = lambda p: p.access_serial)
            order += sorted(
                (p for p in self.unreferenced_pages.values() if p.phash in protected_hashes),
                key = lambda p: p.access_serial
            )
            return deque(order)

        order = []
        emitted = set()

        # Class 1: empty pages
        empty = [
            p for p in self.unreferenced_pages.values()
            if p.kv_position == 0 and p.phash not in protected_hashes
        ]
        empty.sort(key = lambda p: p.access_serial)
        for p in empty:
            order.append(p)
            emitted.add(id(p))

        # Link pages into trees. Referenced pages participate as (non-evictable) internal nodes so that the
        # unreferenced extensions of a live prefix are still pruned tail-first.
        page_index = {}
        for p in self.all_pages:
            p.children = []
            page_index[p.phash] = p

        roots = []
        orphan_roots = []
        parent_of = {}
        for p in self.all_pages:
            if p.prev_hash is None:
                roots.append(p)
                continue
            parent = page_index.get(p.prev_hash)
            if parent is None or parent is p:
                orphan_roots.append(p)
            else:
                parent.children.append(p)
                parent_of[id(p)] = parent

        roots.sort(key = lambda p: p.access_serial)
        orphan_roots.sort(key = lambda p: p.access_serial)

        def prune(root):
            # Emit the tree's evictable pages in the order given by repeatedly removing the oldest current leaf.
            # A page that cannot be evicted (referenced or protected) blocks its ancestors, since evicting an
            # ancestor would orphan a chain that is still in use.
            remaining = {}
            heap = []
            stack = [root]
            while stack:
                p = stack.pop()
                remaining[id(p)] = len(p.children)
                if not p.children:
                    heapq.heappush(heap, (p.access_serial, p.page_index, p))
                stack.extend(p.children)
            while heap:
                _, _, p = heapq.heappop(heap)
                if id(p) not in emitted:
                    if p.ref_count > 0 or p.phash in protected_hashes:
                        continue
                    order.append(p)
                    emitted.add(id(p))
                parent = parent_of.get(id(p))
                if parent is not None:
                    remaining[id(parent)] -= 1
                    if remaining[id(parent)] == 0:
                        heapq.heappush(heap, (parent.access_serial, parent.page_index, parent))

        # Classes 2 and 3
        for root in orphan_roots:
            prune(root)
        for root in roots:
            prune(root)

        # Last resort
        leftovers = [p for p in self.unreferenced_pages.values() if id(p) not in emitted]
        leftovers.sort(key = lambda p: (p.phash in protected_hashes, p.access_serial))
        order += leftovers

        return deque(order)


    def evict(self, page: CachePage, protect: set | None = None):
        """
        An unreferenced page is about to be repurposed and its current identity destroyed. At this point
        page.phash/page.sequence still describe the dying contents and the K/V data is still intact in the cache
        tensors at page.page_index, so complete hashed pages are pushed to the CPU tier here (asynchronously, but
        stream-ordered before anything that could overwrite them). A page taken for new generation output may
        still be reverted unwritten, so the tier can receive pushes for pages that survive; deduplication by hash
        makes that harmless.
        """
        self.metrics["evictions"] += 1
        if page.kv_position == PAGE_SIZE and is_content_hash(page.phash):
            self.metrics["evictions_live"] += 1
            if self.cpu_tier is not None:
                self.cpu_tier.store(page, self.access_serial, protect)
            else:
                rc = self.generator.recurrent_cache
                if rc is not None and page.phash in rc:
                    self.metrics["stashes_stranded"] += 1


    def get_live_page(self, phash: bytes) -> CachePage | None:
        """
        Return the complete page currently holding phash, if any.
        """
        page = self.referenced_pages.get(phash) or self.unreferenced_pages.get(phash)
        if page is not None and page.kv_position == PAGE_SIZE:
            return page
        return None


    def allocate_pages(
        self,
        page_hashes: list,
        new_unique_pages: int,
        recurrent_pages: list[int] | None,
        protected_hashes: set | None = None,
        restore_limit: int | None = None
    ):
        """
        Allocate physical cache pages for one sequence.

        Existing full prompt pages are resolved by hash first, reusing referenced pages for shared prefixes or
        unreferenced pages for prompt-cache hits. A page hash missing from VRAM but present in the CPU tier is
        restored into the allocated page instead of leaving it for prefill; restore_limit optionally caps how many
        leading pages this is attempted for (recurrent models gain nothing from restoring pages past the last
        usable checkpoint, since replay prefill rewrites them anyway). Remaining misses, plus unique pages needed
        for new generation, are taken from unreferenced pages in the order given by build_eviction_order. For
        hybrid recurrent/KV models, recurrent_pages marks which hashed pages also have stashed recurrent
        checkpoints; the usable cached prefix is capped to the longest page prefix that has both valid K/V pages
        and the matching recurrent state.
        """
        allocated_pages = []
        available_pages = None

        def next_evictable():
            # Deferred so allocations served entirely by hash matches never build the eviction order. Pages
            # claimed by hash after the order was built are skipped by their reference count.
            nonlocal available_pages
            if available_pages is None:
                available_pages = self.build_eviction_order(protected_hashes)
            else:
                while available_pages[0].ref_count:
                    available_pages.popleft()
            page = available_pages.popleft()
            self.evict(page, protected_hashes)
            return page

        # Allocate whole pages
        for lp, h in enumerate(page_hashes):
            self.access_serial += 1

            # Find matching referenced page
            rp = self.referenced_pages.get(h)
            if rp:
                rp.add_ref(self.access_serial)
                allocated_pages.append(rp)

            # If possible, reuse an unreferenced page with matching hash
            else:
                up = self.unreferenced_pages.get(h)
                if up:
                    up.add_ref(self.access_serial)
                    allocated_pages.append(up)

                # No matching page, allocate the best eviction candidate. If the missing page is stored in the
                # CPU tier, restore it into the new page; the eviction above may itself push to the tier, and in
                # the (protected against, but possible) event that this dropped the entry we were after, the page
                # simply falls through to normal prefill.
                else:
                    op = next_evictable()
                    op.add_ref_clear(self.access_serial, h)
                    if (
                        self.cpu_tier is not None and
                        (restore_limit is None or lp < restore_limit) and
                        h in self.cpu_tier
                    ):
                        entry = self.cpu_tier.fetch(h, op.page_index, self.access_serial)
                        op.sequence.copy_(entry["tokens"])
                        op.prev_hash = page_hashes[lp - 1] if lp > 0 else None
                        op.kv_position = PAGE_SIZE
                        self.metrics["alloc_tier_pages"] += 1
                    allocated_pages.append(op)

        # Allocate unique pages
        prev = allocated_pages[-1] if allocated_pages else None
        for npi in range(new_unique_pages):
            self.access_serial += 1
            op = next_evictable()
            op.add_ref_unique(self.access_serial)
            # Link the first fresh page to a complete cached prefix. Normally the link is written when the
            # preceding page completes during generation, but when a requeued job resumes on a fully cached
            # prompt that moment lies in the previous round, and without the link every requeue round would
            # start a new root, fragmenting the sequence's chain for eviction and defragmentation purposes.
            if prev is not None and prev.kv_position == PAGE_SIZE:
                op.prev_hash = prev.phash
            allocated_pages.append(op)
            prev = op

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
            self.metrics["alloc_kv_only_pages"] += cached_pages - max_recur
            cached_pages = max_recur

        self.metrics["alloc_pages"] += len(allocated_pages)
        self.metrics["alloc_cached_pages"] += cached_pages

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
        return len(self.unreferenced_pages)


    def is_resumable(self, phash: bytes) -> bool:
        """
        Whether a job claiming a prompt that ends in the page identified by phash could resume from it: an
        unbroken chain of pages down to the root, each either complete in VRAM or restorable from the CPU tier.
        This is the anchor condition for a recurrent checkpoint stashed under phash.
        """
        h = phash
        steps = 0
        while True:
            page = self.get_live_page(h)
            if page is not None:
                prev = page.prev_hash
            elif self.cpu_tier is not None and h in self.cpu_tier:
                prev = self.cpu_tier.entries[h]["prev_hash"]
            else:
                return False
            if prev is None:
                return True
            h = prev
            steps += 1
            if steps > self.max_pages + (len(self.cpu_tier) if self.cpu_tier is not None else 0):
                return False


    def audit_recurrent_sync(self, recurrent_cache) -> dict:
        """
        Measure two-way staleness between the KV page cache and the recurrent checkpoint cache on hybrid models.

        A stash keyed by page hash h is usable only if a job with a matching prompt can claim complete KV pages
        for the entire chain up to and including h ("anchored"); if any ancestor page was evicted, the stash is
        stranded and occupies system RAM without ever being restorable. Conversely, a complete KV page is only
        worth keeping on a hybrid model if some anchored stash exists at or below it in its subtree: prefill must
        replay all tokens past the resume point to advance the recurrent state, rewriting the KV data anyway.

        Returns counts and byte sizes for both kinds of stranded entry. Read-only; intended for periodic
        instrumentation and to evaluate a two-way pruning pass.
        """
        complete = {}
        for p in self.all_pages:
            if p.kv_position == PAGE_SIZE:
                complete[p.phash] = p

        def prev_of(h):
            # A chain link is intact if the page is complete in VRAM or restorable from the CPU tier
            p = complete.get(h)
            if p is not None:
                return True, p.prev_hash
            if self.cpu_tier is not None and h in self.cpu_tier:
                return True, self.cpu_tier.entries[h]["prev_hash"]
            return False, None

        chain_ok = {}
        def chain_complete(h):
            # Walk to the root, verifying every ancestor is present
            walk = []
            walked = set()
            ok = True
            while True:
                r = chain_ok.get(h)
                if r is not None:
                    ok = r
                    break
                if h in walked:
                    ok = False
                    break
                walk.append(h)
                walked.add(h)
                present, prev = prev_of(h)
                if not present:
                    ok = False
                    break
                if prev is None:
                    break
                h = prev
            for hh in walk:
                chain_ok[hh] = ok
            return ok

        anchored_pages = set()
        stashes_anchored, stashes_stranded = 0, 0
        stash_bytes_anchored, stash_bytes_stranded = 0, 0
        for h, stash in recurrent_cache.items():
            size = stash.get("checkpoint_size", 0)
            if chain_complete(h):
                stashes_anchored += 1
                stash_bytes_anchored += size
                # VRAM pages on the path from an anchored stash to the root are resumable (the chain is known
                # intact here, so the walk terminates at the root)
                hh = h
                while hh is not None:
                    p = complete.get(hh)
                    if p is not None:
                        if id(p) in anchored_pages:
                            break
                        anchored_pages.add(id(p))
                    _, hh = prev_of(hh)
            else:
                stashes_stranded += 1
                stash_bytes_stranded += size

        kv_pages_complete = len(complete)
        kv_pages_resumable = len(anchored_pages)

        return {
            "stashes_anchored": stashes_anchored,
            "stashes_stranded": stashes_stranded,
            "stash_bytes_anchored": stash_bytes_anchored,
            "stash_bytes_stranded": stash_bytes_stranded,
            "kv_pages_complete": kv_pages_complete,
            # Complete KV pages above the deepest anchored stash of their chain; on a hybrid model these are
            # rewritten by replay prefill and save nothing
            "kv_pages_unresumable": kv_pages_complete - kv_pages_resumable,
        }


    def validate_pagetable(self, active_jobs):

        def p_assert(exp):
            assert exp, "Page table validation failed"

        # Check page collections
        ids = set()
        for p in self.referenced_pages.values():
            p_assert(p.ref_count > 0)
            ids.add(id(p))
        for p in self.unreferenced_pages.values():
            p_assert(p.ref_count == 0)
            ids.add(id(p))
        p_assert(len(ids) == self.max_pages)
        p_assert(len(self.all_pages) == self.max_pages)

        # Check job reference counts
        refcounts = [0] * self.max_pages
        for job in active_jobs:
            for seq in job.sequences:
                for page in seq.allocated_pages:
                    refcounts[page.page_index] += 1

        for page in self.all_pages:
            p_assert(page.ref_count == refcounts[page.page_index])

        # Check that all hashes are unique
        hashes = set()
        for page in self.all_pages:
            p_assert(page.phash not in hashes)
            hashes.add(page.phash)
        p_assert(len(hashes) == self.max_pages)

        # Check individual hashes
        for page in self.all_pages:
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

        if not self.generator.enable_defrag:
            return

        # Defragment once job queue is empty and all pages have been touched at least once
        if self.access_serial < self.last_defrag_serial + self.max_pages * 8:
            return
        self.last_defrag_serial = self.access_serial

        assert not self.referenced_pages

        if debug:
            torch.cuda.synchronize()
            time_begin = time.time()

        # Build page index
        page_index = {}
        def build_page_index():
            nonlocal page_index
            page_index = {}
            for page in self.all_pages:
                page_index[page.phash] = page
                page.children = []
                page.longest_chain = 1
        build_page_index()

        # Find cached sequences that can be recovered
        root_pages = []
        def build_root_pages():
            nonlocal root_pages
            root_pages = []
            for page in self.all_pages:
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

        assert new_page_index == self.max_pages

        # Adjust overall shift to minimize page copies
        shift_adjust = max(shift_counts, key = shift_counts.get)

        # Order of operations
        if debug:
            print("Page shifts")

        defrag_map = {}
        for page in self.all_pages:
            page.new_page_index = (page.new_page_index - shift_adjust + self.max_pages) % self.max_pages
            if page.page_index != page.new_page_index:
                defrag_map[page.new_page_index] = page.page_index
                if debug:
                    print(f"{page.new_page_index:2} ← {page.page_index:2}")

        # Don't bother if less than 10% of cache is fragmented
        if len(defrag_map) <= max(self.max_pages // 10, 2):
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
        for page in self.all_pages:
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
