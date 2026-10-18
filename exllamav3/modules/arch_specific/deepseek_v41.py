from __future__ import annotations
from typing_extensions import override
from functools import lru_cache
import os
import torch
from tokenizers import Tokenizer, Regex, normalizers
from ...model.config import Config
from ...modules import Module, Linear, RMSNorm, NGramEmbedding
from ...modules.hyperconnections import hc_flush
from ...ext import exllamav3_ext as ext
from ...cache.recurrent import host_copy
from ...constants import PAGE_SIZE
from ...tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX


@lru_cache(maxsize = 1)
def engram_token_map(path: str) -> torch.Tensor:
    tokenizer = Tokenizer.from_file(path)
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), "\ue000"),
        normalizers.Strip(),
        normalizers.Replace("\ue000", " "),
    ])
    keys, ids = {}, []
    for i in range(tokenizer.get_vocab_size(with_added_tokens = True)):
        text = tokenizer.decode([i], skip_special_tokens = False)
        key = tokenizer.id_to_token(i) if "\ufffd" in text else normalizer.normalize_str(text) or text
        ids.append(keys.setdefault(key, len(keys)))
    return torch.tensor(ids, dtype = torch.long)


class EngramLayerState:

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        self.module = module
        self.max_history = max_history
        self.cache_id = cache_id
        self.ids = torch.zeros((max_batch_size, module.ring_rows), dtype = torch.long)

    def tp_export(self, plan):
        # Every rank sees the same input ids, so each keeps its own identical ring
        return {
            "cls": EngramLayerState,
            "args": {"max_batch_size": self.ids.shape[0], "max_history": self.max_history, "cache_id": self.cache_id},
        }

    def get_checkpoint_size(self):
        return self.ids.shape[1] * 8

    def clear(self, idx):
        pass

    def stash(self, slot, position):
        return host_copy(self.ids[slot])

    def unstash(self, slot, stashed, position):
        self.ids[slot].copy_(stashed)


class EngramLayer(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        hc_mult: int,
        ngram_size: int,
        heads_per_ngram: int,
        head_dim: int,
        head_vocab_sizes: list,
        layer_multipliers: list,
        compressed_vocab_size: int,
        pad_token_id: int,
        sliding_window: int,
        rms_norm_eps: float,
        embed: NGramEmbedding | None = None,
        wkv: Linear | None = None,
        norm_key: RMSNorm | None = None,
        norm_query: RMSNorm | None = None,
    ):
        super().__init__(config = config, key = key, qmap = None)
        self.tp_kwargs = dict(
            layer_idx = layer_idx, hidden_size = hidden_size, hc_mult = hc_mult, ngram_size = ngram_size,
            heads_per_ngram = heads_per_ngram, head_dim = head_dim, head_vocab_sizes = head_vocab_sizes,
            layer_multipliers = layer_multipliers, compressed_vocab_size = compressed_vocab_size,
            pad_token_id = pad_token_id, sliding_window = sliding_window, rms_norm_eps = rms_norm_eps,
        )
        embed_dim = (ngram_size - 1) * heads_per_ngram * head_dim
        self.embed = embed if embed is not None else NGramEmbedding(
            config = config,
            key = f"{key}.embed",
            ngram_size = ngram_size,
            heads_per_ngram = heads_per_ngram,
            ple_embed_dim = embed_dim,
            eos_token_id = -1,
            head_vocab_sizes = head_vocab_sizes,
            layer_multipliers = layer_multipliers,
        )
        self.wkv = wkv if wkv is not None else Linear(
            config = config,
            key = f"{key}.wkv",
            in_features = embed_dim,
            out_features = (hc_mult + 1) * hidden_size,
            out_dtype = torch.half,
        )
        def norm(name):
            return RMSNorm(config, f"{key}.{name}", rms_norm_eps, tensor_weight_suffix = False, groups = hc_mult)
        self.norm_key = norm_key if norm_key is not None else norm("k_weight")
        self.norm_query = norm_query if norm_query is not None else norm("q_weight")
        for m in [self.embed, self.wkv, self.norm_key, self.norm_query]:
            self.register_submodule(m)
        self.compressed_vocab_size = compressed_vocab_size
        self.pad_token_id = pad_token_id
        self.ring_rows = -(-(sliding_window + 3 * PAGE_SIZE) // PAGE_SIZE) * PAGE_SIZE

        assert layer_idx < 0
        self.layer_idx = layer_idx
        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = EngramLayerState
        self.recurrent_layers = []
        self.tp_mode = False
        self.tp_recurrent_lookup = {}

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.token_map = engram_token_map(os.path.join(self.config.directory, "tokenizer.json"))
        assert int(self.token_map.max()) + 1 == self.compressed_vocab_size, \
            f"{self.key}: tokenizer does not match engram_compressed_vocab_size"
        self.pad_id = int(self.token_map[self.pad_token_id])

    @override
    def optimizer_targets(self):
        return self.wkv.optimizer_targets()

    @override
    def make_tp_allocation(self, options: dict) -> list:
        # Replicated on every rank: a few MB of projection and norms, the tables stay on the host
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        return {
            "cls": EngramLayer,
            "kwargs": {"key": self.key, **self.tp_kwargs},
            **{name: getattr(self, name).tp_export(plan, producer)
               for name in ("embed", "wkv", "norm_key", "norm_query")},
            "token_map": producer.send(self.token_map),
            "pad_id": self.pad_id,
            "recurrent_layers": [rl.tp_export(plan) for rl in self.recurrent_layers],
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        sub = {name: exported[name]["cls"].tp_import(local_context, exported[name], plan)
               for name in ("embed", "wkv", "norm_key", "norm_query")}
        module = EngramLayer(config = None, **exported["kwargs"], **sub)
        module.token_map = local_context["consumer"].recv(exported["token_map"], cuda = False)
        module.pad_id = exported["pad_id"]
        module.device = local_context["device"]
        module.tp_mode = True
        for rl in exported["recurrent_layers"]:
            rli = rl["cls"](module, **rl["args"])
            module.recurrent_layers.append(rli)
            module.tp_recurrent_lookup[rl["args"]["cache_id"]] = rli
        return module

    @override
    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        hc_flush(params)
        return super().prepare_for_device(x, params)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        hc_flush(params)
        bsz, seq, H, D = x.shape
        ctx = self.embed.context_len
        ids = params["input_ids"].to("cpu", torch.long)
        mm = ids >= FIRST_MM_EMBEDDING_INDEX if params.get("indexed_embeddings") else None
        if mm is not None:
            ids = ids.masked_fill(mm, self.pad_token_id)
        ids = self.token_map[ids]
        hist = torch.cat((ids.new_full((bsz, ctx), self.pad_id), ids), dim = 1)
        rsg = params.get("recurrent_states")
        if rsg:
            # TP workers receive the cache's id in place of the cache and look the ring up by it
            ring = self.tp_recurrent_lookup[rsg[0].cache] if self.tp_mode else \
                rsg[0].cache.get_recurrent_layer((self.layer_idx, params.get("layer_instance", 0)))
            ring = ring.ids
            R = ring.shape[1]
            for i, rs in enumerate(rsg[:bsz]):
                n = min(rs.position, ctx)
                hist[i, ctx - n:ctx] = ring[rs.slot, torch.arange(rs.position - n, rs.position) % R]
                p = torch.arange(max(rs.position, rs.position + seq - R), rs.position + seq)
                ring[rs.slot, p % R] = ids[i, -p.numel():]
        else:
            assert params.get("position", 0) == 0, \
                "EngramLayer requires recurrent states for forwards past position 0"
        kv = self.wkv.forward(self.embed.forward(hist, params), params)
        key = self.norm_key.forward(kv[..., :H * D].view(bsz, seq, H, D), params, out_dtype = torch.float)
        query = self.norm_query.forward(x, params, out_dtype = torch.float)
        gate = torch.bmm(query.view(-1, 1, D), key.reshape(-1, D, 1)).view(bsz, seq, H)
        gated = torch.empty((bsz, seq, H, D), dtype = torch.float, device = x.device)
        ext.ple_gate(gate, kv[..., H * D:].contiguous(), gated, D ** -0.5)
        if mm is not None and mm.any():
            gated.masked_fill_(mm.to(x.device)[..., None, None], 0)
        return gated.add_(x)
