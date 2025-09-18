from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type
import torch
import torch.nn.functional as F
from torch import nn
from ..model import Model, Config
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import Attention

class CacheLayer(ABC):

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        **kwargs
    ):
        self.config = config
        self.attention = attention
        self.cache_id = cache_id
        self.max_num_tokens = max_num_tokens

    @abstractmethod
    def alloc(self, device: torch.device):
        pass

    @abstractmethod
    def free(self):
        pass

    @abstractmethod
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor) -> tuple:
        pass

    @abstractmethod
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        pass

    @abstractmethod
    def copy_page(self, source: CacheLayer, from_page: int, to_page: int, num_tokens: int):
        pass

    @abstractmethod
    def get_tensors(self):
        pass

    @abstractmethod
    def storage_size(self):
        pass

    @abstractmethod
    def overhead_size(self):
        pass

    @abstractmethod
    def tp_export(self, plan):
        pass

    @abstractmethod
    def get_kv_alloc_placeholder(self):
        # Used by layersplit loader to simulate dequant overhead, if any. Returns a reference to hold while
        # inference is simulated, or None for unquantized cache
        pass


class Cache:

    def __init__(
        self,
        model: Model,
        max_num_tokens: int,
        layer_type: Type[CacheLayer] | None = None,
        **kwargs
    ):
        """
        Create cache for model

        :param model:
            Model for which to create the cache. Once created, the cache is tied to the model. Loading the model
            will create cache tensors and unloading the model will destroy them. To delete the cache itself without
            deleting the reference to the model, use detach_from_model

        :param layer_type:
            Cache layer class, CacheLayer_fp16 or CacheLayer_quant

        :param max_num_tokens:
            Max number of total tokens in the cache. Must be a multiple of the page size (256). For use with the
            dynamic generator, this is the total number of tokens that can be allocated across concurrent jobs. For
            batched inference, seq_len * batch_size <= max_num_tokens

        :param k_bits:
            If layer_type == CacheLayer_quant, bits per element of the quantized keys tensor

        :param v_bits:
            If layer_type == CacheLayer_quant, bits per element of the quantized values tensor

        """
        self.model = model
        self.config = model.config
        self.max_num_tokens = max_num_tokens

        from .fp16 import CacheLayer_fp16
        self.layer_type = layer_type or CacheLayer_fp16
        # self.recurrent_layer_type = recurrent_layer_type or RecurrentLayer_fp16

        cl = self.model.get_cache_layers()
        self.num_layers = len(cl)
        self.layers = {
            attn.layer_idx: self.layer_type(self.config, attn, id(self), self.max_num_tokens, **kwargs)
            for attn in cl
        }

        self.attach_to_model()


    def attach_to_model(self, model: Model | None = None):
        """
        Attach cache to model. Registering the cache with the model (done automatically by the Cache constructor)
        is necessary in order to tie loading of the model to allocation of cache tensors. Multiple caches can be
        attached to the same model.
        """
        if model is None:
            model = self.model

        cl = model.get_cache_layers()
        for module in cl:
            layer = self.layers[module.layer_idx]
            assert layer not in module.cache_layers, "Cannot attach cache twice to the same model."
            module.cache_layers.append(layer)


    def detach_from_model(self, model: Model | None = None):
        """
        Detach cache from model. Must be called if you want to delete a cache without deleting the model.
        """
        if model is None:
            model = self.model

        cl = model.get_cache_layers()
        for module in cl:
            layer = self.layers[module.layer_idx]
            module.cache_layers.remove(layer)


    def get_layer(self, idx: int, cache_seqlens: torch.Tensor, block_table: torch.Tensor) -> tuple:
        return self.layers[idx].get_kv(cache_seqlens, block_table)


    def update_layer(
        self,
        idx: int,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        return self.layers[idx].update_kv(cache_seqlens, block_table, k, v, length)


    def copy_page(
        self,
        target: Cache,
        from_page: int,
        to_page: int,
        num_tokens: int,
    ):
        assert target == self or (not target.model.loaded_tp and not self.model.loaded_tp), \
            "Cannot copy pages between TP and non-TP caches, or between distinct TP caches."
        assert target.num_layers == self.num_layers
        if not self.model.loaded_tp:
            for idx, src in self.layers.items():
                dst = target.layers[idx]
                assert type(src) is type(dst)
                dst.copy_page(src, from_page, to_page, num_tokens)
        else:
            self.model.tp_cache_page_copy(id(self), from_page, to_page, num_tokens)


    def get_all_tensors(self):
        tensors = []
        for layer in self.layers.values():
            tensors += layer.get_tensors()
        return tensors


    def new_recurrent_state(self):
        rl = self.model.get_recurrent_layers()
        state = {
            attn.layer_idx: attn.new_recurrent_state()
            for attn in rl
        }
        return state