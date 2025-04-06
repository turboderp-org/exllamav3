from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..constants import PAGE_SIZE
from ..models import Model, Config
from .cache import CacheLayer

class CacheLayer_fp16(CacheLayer):

    def __init__(
        self,
        config: Config,
        max_num_tokens: int,
    ):
        super().__init__(config, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."

        self.shape = (max_num_tokens // PAGE_SIZE, PAGE_SIZE, config.num_kv_heads, config.head_dim)
        self.k = None
        self.v = None
        self.device = None


    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.k = torch.zeros(self.shape, dtype = torch.half, device = device)
        self.v = torch.zeros(self.shape, dtype = torch.half, device = device)


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
        kd = self.k[to_page, :num_tokens, :, :]
        vd = self.v[to_page, :num_tokens, :, :]
        ks = source.k[from_page, :num_tokens, :, :]
        vs = source.v[from_page, :num_tokens, :, :]
        kd.copy_(ks, non_blocking = True)
        vd.copy_(vs, non_blocking = True)