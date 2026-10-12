"""
Stand-ins for driving the CPU page tier (exllamav3/generator/cpu_cache.py) without a model or generator.

The tier only needs, per cache, a `layers` dict whose values expose get_tensors() (page-major tensors, None
entries skipped) and, per stored page, the page table's (phash, prev_hash, page_index, sequence) record.

    FakeCacheLayer([tensor, None])      one cache layer over the given tensors
    page(tag, page_index, prev = None)  a page record whose hash is phash(tag) and whose tokens are all `tag`
    phash(tag)                          the hash page(tag, ...) carries
"""

from collections import namedtuple

import torch

PAGE_SIZE = 32


class FakeCacheLayer:

    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensors(self):
        return self.tensors


Page = namedtuple("Page", ["phash", "prev_hash", "page_index", "sequence"])


def phash(tag: int) -> bytes:
    return tag.to_bytes(2, "big")


def page(tag: int, page_index: int, prev: bytes | None = None, page_size: int = PAGE_SIZE) -> Page:
    return Page(phash(tag), prev, page_index, torch.full((page_size,), tag, dtype = torch.long))
