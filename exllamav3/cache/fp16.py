from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..constants import PAGE_SIZE
from ..models import Model, Config
from .cache import CacheLayer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import Attention

class CacheLayer_fp16(CacheLayer):

    def __init__(
        self,
        config: Config,
        attention: Attention,
        max_num_tokens: int,
    ):
        super().__init__(config, attention, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."

        self.shape = (
            (max_num_tokens // PAGE_SIZE, PAGE_SIZE, attention.num_kv_heads, attention.head_dim)
            if attention else None
        )
        self.k = None
        self.v = None
        self.device = None


    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.k = torch.zeros(self.shape, dtype = torch.half, device = device) if self.shape else None
        self.v = torch.zeros(self.shape, dtype = torch.half, device = device) if self.shape else None


    @override
    def free(self):
        self.device = None
        self.k = None
        self.v = None


    @override
    def get_kv(self):
        return self.k, self.v


    @override
    def copy_page(self, source: CacheLayer_fp16, from_page: int, to_page: int, num_tokens: int):
        assert self.shape == source.shape
        kd = self.k[to_page, :num_tokens, :, :]
        vd = self.v[to_page, :num_tokens, :, :]
        ks = source.k[from_page, :num_tokens, :, :]
        vs = source.v[from_page, :num_tokens, :, :]
        kd.copy_(ks, non_blocking = True)
        vd.copy_(vs, non_blocking = True)