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
        self.ids = torch.zeros((max_batch_size, module.ring_rows), dtype = torch.long)

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
    ):
        super().__init__(config = config, key = key, qmap = None)
        embed_dim = (ngram_size - 1) * heads_per_ngram * head_dim
        self.embed = NGramEmbedding(
            config = config,
            key = f"{key}.embed",
            ngram_size = ngram_size,
            heads_per_ngram = heads_per_ngram,
            ple_embed_dim = embed_dim,
            eos_token_id = -1,
            head_vocab_sizes = head_vocab_sizes,
            layer_multipliers = layer_multipliers,
        )
        self.wkv = Linear(
            config = config,
            key = f"{key}.wkv",
            in_features = embed_dim,
            out_features = (hc_mult + 1) * hidden_size,
            out_dtype = torch.half,
        )
        def norm(name):
            return RMSNorm(config, f"{key}.{name}", rms_norm_eps, tensor_weight_suffix = False, groups = hc_mult)
        self.norm_key = norm("k_weight")
        self.norm_query = norm("q_weight")
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
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        hc_flush(params)
        bsz, seq, H, D = x.shape
        ctx = self.embed.context_len
        ids = self.token_map[params["input_ids"].to("cpu", torch.long)]
        hist = torch.cat((ids.new_full((bsz, ctx), self.pad_id), ids), dim = 1)
        rsg = params.get("recurrent_states")
        if rsg:
            ring = rsg[0].cache.get_recurrent_layer((self.layer_idx, params.get("layer_instance", 0))).ids
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
        return gated.add_(x)
