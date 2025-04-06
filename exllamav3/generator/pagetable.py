from __future__ import annotations
import torch
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from ..cache.cache import Cache
if TYPE_CHECKING:
    from .generator import Generator
from ..constants import PAGE_SIZE
from collections import deque
from itertools import pairwise
from ..util.tensor import SeqTensor


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

    def __repr__(self):
        return (
            f"CachePage: idx = {self.page_index}, ref_count = {self.ref_count}, "
            f"phash: ..{str(self.phash)[8:24]}.., prev_hash: ..{str(self.prev_hash)[8:24]}.., "
            f"kvp {self.kv_position}"
        )

    # Copy page state so page can be reverted even
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
    def update_hash(self, newhash):
        assert self.ref_count > 0
        assert self.kv_position == PAGE_SIZE
        del self.pagetable.referenced_pages[self.phash]
        self.phash = newhash
        self.can_revert = False
        assert self.phash not in self.pagetable.referenced_pages
        self.pagetable.referenced_pages[self.phash] = self


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

    def allocate_pages(self, pagetable: PageTable):
        self.allocated_pages, self.kv_position, cached_pages, non_sequential_pages = \
            pagetable.allocate_pages(self.page_hashes, self.new_unique_pages)
        self.build_block_index_tensor()
        return len(self.allocated_pages), cached_pages, non_sequential_pages


class PageTable:

    def __init__(
        self,
        generator: Generator,
        cache: Cache
    ):
        self.generator = generator
        self.cache = cache
        self.max_pages = cache.max_num_tokens // PAGE_SIZE

        self.access_serial = self.max_pages
        self.referenced_pages = {}
        self.unreferenced_pages = {}
        self.all_pages = []
        self.reset_page_table()


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
                access_serial_revert = 0,
                kv_position = 0,
                kv_position_revert = 0,
                can_revert = False,
                new_page_index = 0
            )
            self.all_pages.append(cp)
            self.unreferenced_pages[h] = cp
        self.access_serial = self.max_pages
        # TODO: (defrag)
        # self.last_defrag_serial = self.access_serial


    def print_page_list(self, short: bool = True):
        for cp in self.all_pages:
            if cp.phash in self.referenced_pages:
                assert cp.ref_count > 0
                ref = str(cp.ref_count) if cp.ref_count < 10 else "+"
            elif cp.phash in self.unreferenced_pages:
                assert cp.ref_count == 0
                ref = "."
            else:
                ref = "#"
            if short: print(ref, end = "")
            else: print(str(cp) + f", ref {ref}")
        print()


    def defrag(self):
        # TODO: (defrag)
        pass


    def allocate_pages(
        self,
        page_hashes: list,
        new_unique_pages: int
    ):
        allocated_pages = []
        available_pages = None

        # Allocate whole pages
        for h in page_hashes:
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

                # No matching pages
                else:

                    # Get list of unreferenced pages in order of oldest to newest
                    if available_pages is None:
                        available_pages = list(self.unreferenced_pages.values())
                        available_pages.sort(key = lambda x: x.access_serial)
                        available_pages = deque(available_pages)
                    else:
                        while available_pages[0].ref_count:
                            available_pages.popleft()

                    # Allocate oldest unreferenced page
                    op = available_pages.popleft()
                    op.add_ref_clear(self.access_serial, h)
                    allocated_pages.append(op)

        # Allocate unique pages
        for npi in range(new_unique_pages):
            self.access_serial += 1

            # Get list of unreferenced pages in order of oldest to newest
            if available_pages is None:
                available_pages = list(self.unreferenced_pages.values())
                available_pages.sort(key = lambda x: x.access_serial)
                available_pages = deque(available_pages)
            else:
                while available_pages[0].ref_count:
                    available_pages.popleft()

            op = available_pages.popleft()
            op.add_ref_unique(self.access_serial)
            allocated_pages.append(op)

        # Advance cache over prefilled pages
        kv_position = 0
        cached_pages = 0
        for page in allocated_pages:
            if page.kv_position == PAGE_SIZE:
                kv_position += PAGE_SIZE
                cached_pages += 1
            else:
                break

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