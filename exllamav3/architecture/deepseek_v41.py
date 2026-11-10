from __future__ import annotations
from typing_extensions import override
from functools import lru_cache
import os
import numpy as np
import torch
from ..model.config import no_default
from ..modules import PLELayer
from ..modules.ngram_embedding import _find_nth_prime_after
from ..modules.attn import prepare_for_attn
from .deepseek_v4 import DeepseekV4Config, DeepseekV4Model

# DeepSeek-V4.1: DeepSeek-V4 blocks with the settings nested under text_config, where runs of layers
# read one compressed KV pool (kv_source_layer_ids) and one index selection (index_source_layer_ids,
# two-level through the candidate blocks of candidate_source_layer_id), the index keys come from the
# pooled latent, queries carry no per-head norm, every mHC site collapses with the previous site's
# pre weights, and Engram layers add gated n-gram rows to the streams. Reference implementation ships
# in the checkpoint: inference/model.py and inference/engram.py.


@lru_cache(maxsize = 2)
def engram_token_map(path: str) -> torch.Tensor:
    """Compressed token ids for the Engram hash (inference/engram.py build_compressed_token_map): tokens
    whose decoded text normalizes alike share one id"""
    from tokenizers import Tokenizer, Regex, normalizers
    tokenizer = Tokenizer.from_file(path)
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), ""),
        normalizers.Strip(),
        normalizers.Replace("", " "),
    ])
    keys, ids = {}, []
    for i in range(tokenizer.get_vocab_size(with_added_tokens = True)):
        text = tokenizer.decode([i], skip_special_tokens = False)
        key = tokenizer.id_to_token(i) if "�" in text else normalizer.normalize_str(text) or text
        ids.append(keys.setdefault(key, len(keys)))
    return torch.tensor(ids, dtype = torch.long)


class DeepseekV41Config(DeepseekV4Config):
    arch_string = "DeepseekV41ForCausalLM"
    cfg_prefix = "text_config->"
    default_num_hash_layers = 0

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        from .deepseek_v41_mtp import DeepseekV41MTPModel
        super().__init__(
            directory,
            {"text": DeepseekV41Model, "mtp": DeepseekV41MTPModel},
            **kwargs
        )
        p = self.cfg_prefix
        self.kv_source_layer_ids = self.read_cfg(list, p + "kv_source_layer_ids", no_default)
        self.index_source_layer_ids = self.read_cfg(list, p + "index_source_layer_ids", no_default)
        self.candidate_source_layer_id = self.read_cfg(int, p + "candidate_source_layer_id", -1)
        self.candidate_topk_blocks = self.read_cfg(int, p + "candidate_topk_blocks", 0)
        self.candidate_block_size = self.read_cfg(int, p + "candidate_block_size", 1)
        # Below index_topk entries every query attends densely (a top-k over fewer entries keeps them
        # all); that shortcut is exact only while the candidate blocks also keep every entry there
        assert self.candidate_source_layer_id < 0 or self.candidate_topk_blocks * self.candidate_block_size >= self.index_topk, \
            "DeepseekV41: candidate_topk_blocks * candidate_block_size must cover index_topk entries"
        for idx, ratio in enumerate(self.compress_ratios):
            src = self.kv_source(idx)
            assert not ratio or self.compress_ratios[src] == ratio, \
                f"DeepseekV41: layer {idx} cannot read the pool of layer {src}"

        # Engram: the hash layout (inference/engram.py EngramLayout) derives from the config alone,
        # one prime-sized bucket range per (n-gram size, head) in order, never reused
        self.engram_layer_ids = self.read_cfg(list, p + "engram_layer_ids", [])
        self.engram_ngram_size = self.read_cfg(int, p + "engram_max_ngram_size", 4)
        self.engram_n_heads = self.read_cfg(int, p + "engram_n_heads", 8)
        self.engram_head_dim = self.read_cfg(int, p + "engram_head_dim", 256)
        self.engram_pad_token_id = self.read_cfg(int, p + "engram_pad_token_id", 2)
        self.engram_compressed_vocab_size = self.read_cfg(int, p + "engram_compressed_vocab_size", 0)
        prime = self.read_cfg(int, p + "engram_vocab_size", 0) - 1
        rows = self.read_cfg(list, p + "engram_num_embeddings", [])
        self.engram_hash_params = {}
        for idx, n in zip(self.engram_layer_ids, rows):
            sizes = []
            for _ in range((self.engram_ngram_size - 1) * self.engram_n_heads):
                prime = _find_nth_prime_after(prime, 1)
                sizes.append(prime)
            assert sum(sizes) == n, f"DeepseekV41: unexpected engram table size for layer {idx}"
            bound = (2 ** 63 - 1) // self.engram_compressed_vocab_size // 2
            mult = np.random.default_rng(10007 * idx).integers(0, bound, size = self.engram_ngram_size, dtype = np.int64)
            self.engram_hash_params[idx] = (sizes, (mult * 2 + 1).tolist())

        # DSpark draft head: its own routed expert count
        self.dspark_num_experts = self.read_cfg(int, p + "dspark_n_routed_experts", 0)
        self.dspark_num_experts_per_tok = self.read_cfg(int, p + "dspark_num_experts_per_tok", 0)

    def kv_source(self, idx: int) -> int | None:
        """The layer whose compressed pool layer idx attends over (itself when it compresses)"""
        if not self.compress_ratios[idx]:
            return None
        return max((s for s in self.kv_source_layer_ids if s <= idx), default = idx)

    @override
    def layer_type(self, ratio: int) -> str:
        return "hca" if ratio else "sliding"

    @override
    def compress_rate(self, idx: int) -> int | None:
        return self.compress_ratios[idx] or None

    @override
    def read_vision_config(self):
        from .deepseek_v41_vision import read_deepseek_v41_vision_config
        return read_deepseek_v41_vision_config(self)

    @override
    def vision_model_class(self):
        from .deepseek_v41_vision import DeepseekV41VisionModel
        return DeepseekV41VisionModel

    def get_tensor_name_fixes(self):
        return {
            ".markov_head.embed.weight": ".markov_head.markov_w1.weight",
            ".markov_head.head.weight": ".markov_head.markov_w2.weight",
        }


class DeepseekV41Model(DeepseekV4Model):
    config_class = DeepseekV41Config
    hc_carry_pre = True

    def __init__(
        self,
        config: DeepseekV41Config,
        **kwargs
    ):
        self.kv_owners = {}
        super().__init__(config, **kwargs)
        # Tensor parallel: every rank holds every attention layer with the pools replicated, so a
        # reader finds its owner on its own rank (DSV4Attention.tp_import); the Engram layers are
        # replicated like the Qwen3.8 PLE layers
        self.caps.update({"supports_tp": True})

    @override
    def make_attn(self, idx: int, key: str, **extra):
        config = self.config
        src = config.kv_source(idx)
        full = idx in config.index_source_layer_ids
        cand = config.candidate_source_layer_id
        attn = super().make_attn(
            idx, key,
            kv_source = src,
            kv_owner = self.kv_owners.get(src),
            indexer_mode = ("full" if full else "shared") if src is not None else None,
            q_head_norm = False,
            candidate_mode = "source" if idx == cand else "use" if full and 0 <= cand < idx else None,
            candidate_topk_blocks = config.candidate_topk_blocks,
            candidate_block_size = config.candidate_block_size,
            **extra,
        )
        # Pools and selections are read across layers (every layer past 20 reads layer 20's), so a
        # split loads all attention layers on one device (Model.load attn_device)
        attn.caps["attn_device"] = True
        self.kv_owners[idx] = attn
        return attn

    @override
    def make_pre_block(self, idx: int, key: str) -> list:
        config = self.config
        if idx not in config.engram_layer_ids:
            return []
        token_map = engram_token_map(os.path.join(config.directory, "tokenizer.json"))
        assert int(token_map.max()) + 1 == config.engram_compressed_vocab_size, \
            f"DeepseekV41: tokenizer does not match engram_compressed_vocab_size"
        pad_id = int(token_map[config.engram_pad_token_id])
        return [
            PLELayer(
                config = config,
                key = f"{key}.engram",
                layer_idx = -(idx + 1),
                hidden_size = config.hidden_size,
                hc_mult = config.hc_mult,
                ple_embed_dim = (config.engram_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim,
                ngram_size = config.engram_ngram_size,
                heads_per_ngram = config.engram_n_heads,
                eos_token_id = pad_id,
                conv_kernel_size = 0,
                rms_norm_eps = config.rms_norm_eps,
                # Image spans hash as the pad token (no n-gram spans one)
                mm_token_id = config.engram_pad_token_id,
                mm_gate_off = True,
                ngram_key = f"{key}.engram.embed",
                kv_proj_key = "wkv",
                norm_keys = ("k_weight", "q_weight", "norm_conv"),
                norm_constant_bias = 0.0,
                norm_weight_suffix = False,
                token_map = token_map,
                hash_params = config.engram_hash_params[idx],
            )
        ]

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # V4.1 attends causally inside an image span (the V4 vision-exp non-causal image chunk does not
        # apply), so the span needs no chunk check and no nc_chunk flag
        params["input_ids"] = input_ids
        if self.hc_carry_pre:
            params["hc_pre"] = None
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # The checkpoint's template in chat mode (no reasoning header): the assistant turn opens
        # with a closed think tag; without it the model ends its turn at once
        p = "<｜begin▁of▁sentence｜>"
        if system_prompt:
            p += f"<｜System｜>{system_prompt}"
        p += f"<｜User｜>{prompt}<｜Assistant｜></think>"
        return p
