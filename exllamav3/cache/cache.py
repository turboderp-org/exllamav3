from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type
import torch
import torch.nn.functional as F
from torch import nn
from ..models import Model, Config
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import Attention

class CacheLayer(ABC):

    def __init__(
        self,
        config: Config,
        attention: Attention,
        max_num_tokens: int,
        **kwargs
    ):
        self.config = config
        self.attention = attention
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

        self.num_layers = len(self.model.get_cache_layers())
        self.layers = [
            self.layer_type(self.config, attn, self.max_num_tokens, **kwargs)
            for attn in self.model.get_cache_layers()
        ]
        self.attach_to_model()


    def attach_to_model(self, model: Model | None = None):
        """
        Attach cache to model. Registering the cache with the model (done automatically by the Cache constructor)
        is necessary in order to tie loading of the model to allocation of cache tensors. Multiple caches can be
        attached to the same model.
        """
        if model is None:
            model = self.model
        model_num_layers = len(model.get_cache_layers())
        assert model_num_layers == self.num_layers, \
            f"Cannot attach cache with {self.num_layers} layers to model with {model_num_layers} layers."
        for layer, module in zip(self.layers, model.get_cache_layers()):
            assert layer not in module.cache_layers, \
                "Cannot attach cache twice to the same model."
            module.cache_layers.append(layer)


    def detach_from_model(self, model: Model | None = None):
        """
        Detach cache from model. Must be called if you want to delete a cache without deleting the model.
        """
        if model is None:
            model = self.model
        model_num_layers = len(model.get_cache_layers())
        assert model_num_layers == self.num_layers, \
            f"Cannot detach cache with {self.num_layers} layers from model with {model_num_layers()} layers."
        for layer, module in zip(self.layers, model.get_cache_layers()):
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
        assert target.num_layers == self.num_layers
        for src, dst in zip(target.layers, self.layers):
            assert type(src) is type(dst)
            dst.copy_page(src, from_page, to_page, num_tokens)
