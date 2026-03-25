from __future__ import annotations
from typing_extensions import override
import torch
from torch.nn import functional as F
from torch import nn
from ..model.config import Config
from ..util.tensor import to2
from . import Module, Linear
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from ..model.model_tp_alloc import TPAllocation

class ValueEmbeddings(Module):
    """CPU-resident value embeddings. Does all VE lookups at once during forward,
    producing a (num_ve_layers, B, T, kv_dim) tensor stored in params.
    Each attention block indexes its slice. No per-layer CPU-GPU sync."""

    def __init__(
        self,
        config: Config,
        key: str,
        target_layers: list[int],
        vocab_size: int,
        kv_head_dim: int,
        num_kv_heads: int,
    ):
        super().__init__(config, key, None)
        self.target_layers = target_layers
        self.vocab_size = vocab_size
        self.kv_head_dim = kv_head_dim
        self.num_kv_heads = num_kv_heads
        self.weight = {}
        self.forward_ref = {}

        self.caps.update({
            "prefer_cpu": True,
            "retain_during_quant": True,
        })

    @override
    def load(self, device, **kwargs):
        self.device = device
        self.weight = {}
        for layer_idx in self.target_layers:
            self.weight[layer_idx] = self.config.stc.get_tensor(f"{self.key}.{layer_idx}.weight", self.device, float2half = True)

    @override
    def unload(self):
        self.weight = {}

    @override
    def get_tensors(self):
        return {
            f"{self.key}.{layer_idx}.weight": self.weight[layer_idx].contiguous()
            for layer_idx in self.target_layers
        }

    @override
    def forward(self, x, params, out_dtype = None):
        input_ids = params["input_ids"]
        # Move all embeddings to target device(s) here
        params.update({
            f"_nc_ve.{layer_idx}": F.embedding(input_ids, weight).unflatten(-1, (self.num_kv_heads, self.kv_head_dim))
            .to(self.forward_ref[layer_idx].device, non_blocking = True).contiguous()
            for layer_idx, weight in self.weight.items()
        })
        return x

    @override
    def optimizer_targets(self):
        return []

    def weights_numel(self):
        return len(self.target_layers) * self.vocab_size * self.num_kv_heads * self.kv_head_dim
