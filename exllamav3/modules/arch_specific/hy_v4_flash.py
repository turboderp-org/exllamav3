from __future__ import annotations
from typing_extensions import override
import math
import numpy as np
import torch
import torch.nn.functional as F
from ...model.config import Config
from ..module import Module
from ..linear import Linear
from ..rmsnorm import RMSNorm
from ..row_table import RowTable
from ..quant.exl3_lib.ngram_codec import mul1_codebook, dequant_rows, words_per_row, ring_dim

"""
Hy4-flash (hy_v4_flash) modules: gated multi-head latent attention with a learned sink, read
either through a 512-token window or through the compressed indexer's group selection; identity
hyper-connections over four residual streams; the Engram n-gram memory; the stream collapse
before the head.

Reference: transformers models/hy_v4_flash (pull request 50903 at b865850d), checked per layer
against it on the checkpoint's own weights. Everything here is the reference arithmetic in plain
torch over the whole row: no KV cache and no decode kernels. That is what the conversion's
calibration forwards and a cache-less check forward need; serving kernels are out of scope.

Tensors that the reference keeps in fp32 (hyper-connection fn/base/scale, sinks, the indexer's
pooling projections and position bias, the router's score bias) are carried as stored.
"""

RAW_KW = dict(allow_bf16 = True, no_defer = True)


class RawTensors(Module):
    """Named checkpoint tensors carried as stored (dtype and shape unchanged) into the output."""

    def __init__(self, config: Config | None, key: str, names: list[str], optional: list[str] | None = None):
        super().__init__(config, key, None)
        self.names = list(names)
        self.optional = set(optional or [])
        self.t = {}
        self._numel = 0

    def full(self, name: str) -> str:
        return f"{self.key}.{name}" if self.key else name

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        stc = self.config.stc
        self.t = {}
        for n in self.names:
            t = stc.get_tensor(self.full(n), device, optional = n in self.optional, **RAW_KW)
            if t is not None:
                self.t[n] = t
        self._numel = sum(t.numel() for t in self.t.values())

    @override
    def unload(self):
        self.device = None
        self.t = {}

    @override
    def get_tensors(self):
        return {self.full(n): t.contiguous() for n, t in self.t.items()}

    @override
    def weights_numel(self):
        return self._numel

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x, params, out_dtype = None):
        raise RuntimeError("RawTensors is a tensor holder, not a graph module")

    def f(self, name: str) -> torch.Tensor:
        return self.t[name].float()


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) over the last dimension, fp32"""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(dim = -1, keepdim = True) + eps)


def rope_angles(positions: torch.Tensor, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """positions (..., n) -> cos, sin (..., n, dim): one angle per adjacent pair (2i, 2i + 1),
    inv_freq[i] = theta ** (-2i / dim). Text positions only (the three M-RoPE planes are equal)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.float, device = positions.device) / dim))
    freqs = positions[..., None].float() * inv_freq
    return freqs.cos().repeat_interleave(2, dim = -1), freqs.sin().repeat_interleave(2, dim = -1)


def rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs of the last dimension: (a, b) -> (a cos - b sin, b cos + a sin)"""
    x = x.float()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), dim = -1).flatten(-2)
    return x * cos + rot * sin


class HyV4HyperConnection(RawTensors):
    """Identity hyper-connection site (no stream-to-stream combine). mix() collapses the streams
    for the sublayer and returns the weights its output is written back with."""

    def __init__(self, config: Config | None, key: str, hc_mult: int, hc_eps: float, rms_norm_eps: float,
                 magnitude: float):
        super().__init__(config, key, ["hc_fn", "hc_base", "hc_scale"])
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.rms_eps = rms_norm_eps
        self.magnitude = magnitude

    def mix(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """streams (b, s, H, D) fp32 -> (post (b, s, H), collapsed (b, s, D)), both fp32"""
        H = self.hc_mult
        flat = streams.flatten(2).float()
        inv = torch.rsqrt(flat.square().mean(dim = -1, keepdim = True) + self.rms_eps)
        mixes = F.linear(flat, self.f("hc_fn")) * inv
        base, scale = self.f("hc_base"), self.f("hc_scale")
        pre = torch.sigmoid(mixes[..., :H] * scale[0] + base[:H]) + self.hc_eps
        post = self.magnitude * torch.sigmoid(mixes[..., H:] * scale[1] + base[H:]) + self.hc_eps
        return post, (pre.unsqueeze(-1) * streams).sum(dim = 2)

    @staticmethod
    def apply(streams: torch.Tensor, y: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return post.unsqueeze(-1) * y.float().unsqueeze(-2) + streams


class HyV4HyperHead(RawTensors):
    """Collapse of the stream stack before the final norm."""

    def __init__(self, config: Config | None, key: str, hc_mult: int, hc_eps: float, rms_norm_eps: float):
        super().__init__(config, key, ["hc_head_fn", "hc_head_base", "hc_head_scale"])
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.rms_eps = rms_norm_eps

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        flat = x.flatten(2).float()
        inv = torch.rsqrt(flat.square().mean(dim = -1, keepdim = True) + self.rms_eps)
        mixes = F.linear(flat, self.f("hc_head_fn")) * inv
        pre = torch.sigmoid(mixes * self.f("hc_head_scale") + self.f("hc_head_base")) + self.hc_eps
        return (pre.unsqueeze(-1) * x).sum(dim = 2)


class HyV4Attention(Module):
    """
    Gated MLA. layer_type "sliding": every query reads the last `sliding_window` keys.
    layer_type "sparse": every query reads all keys up to index_topk keys of context, and past
    that the tokens of the index_topk / index_kpool - 1 best complete key groups the compressed
    indexer scores, plus its own group.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        layer_type: str,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
        sliding_window: int = 0,
        index_n_heads: int = 0,
        index_head_dim: int = 0,
        index_topk: int = 0,
        index_kpool: int = 0,
        qmap: str | None = "block.attn",
        qbits_key: str = "bits",
        select_hq_bits: int = 2,
        indexer_hq_bits: int = 0,
        out_dtype: torch.dtype | None = torch.float,
        query_chunk: int = 1024,
    ):
        super().__init__(config, key, None)
        assert layer_type in ("sliding", "sparse")
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope = qk_nope_head_dim
        self.rope = qk_rope_head_dim
        self.qk = qk_nope_head_dim + qk_rope_head_dim
        self.v = v_head_dim
        self.rope_theta = rope_theta
        self.rms_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.sm_scale = self.qk ** -0.5
        self.out_dtype = out_dtype
        self.query_chunk = query_chunk
        self.q_priority = 2 + select_hq_bits

        def lin(name, i, o, qm, hq = select_hq_bits, out = torch.half, **kw):
            m = Linear(
                config, f"{key}.{name}", i, o,
                qmap = (qmap + qm) if (qmap is not None and qm is not None) else None,
                out_dtype = out, trim_padded_out = True, select_hq_bits = hq, qbits_key = qbits_key, **kw
            )
            self.register_submodule(m)
            return m

        def norm(name):
            m = RMSNorm(config, f"{key}.{name}", rms_norm_eps, out_dtype = torch.half)
            self.register_submodule(m)
            return m

        H = num_heads
        self.q_a_proj = lin("q_a_proj", hidden_size, q_lora_rank, ".input")
        self.q_a_layernorm = norm("q_a_layernorm")
        self.q_b_proj = lin("q_b_proj", q_lora_rank, H * self.qk, ".q_a")
        self.kv_a_proj = lin("kv_a_proj_with_mqa", hidden_size, kv_lora_rank + qk_rope_head_dim, ".input")
        self.kv_a_layernorm = norm("kv_a_layernorm")
        self.kv_b_proj = lin("kv_b_proj", kv_lora_rank, H * (self.nope + self.v), ".kv_a")
        self.linear_gate = lin("linear_gate", hidden_size, H * self.v, ".input")
        self.o_proj = lin("o_proj", H * self.v, hidden_size, ".o", out = torch.float)
        self.post_proj_norm = RMSNorm(config, f"{key}.post_proj_norm", rms_norm_eps, out_dtype = torch.float)
        self.register_submodule(self.post_proj_norm)
        self.sink = RawTensors(config, key, ["learnable_sink_param"])
        self.register_submodule(self.sink)

        self.indexer = None
        if layer_type == "sparse":
            ik = f"{key}.compressed_indexer"
            # The indexer's query projection is budgeted at the base rate (not promoted by --hq).
            # Its key side is small and its selection noise is coherent across every reader, so
            # the key projection, the head weights and the fp32 pooling stay as stored
            self.idx_query = Linear(
                config, f"{ik}.query_projection", q_lora_rank, index_n_heads * index_head_dim,
                qmap = (qmap + ".q_a") if qmap is not None else None,
                out_dtype = torch.half, trim_padded_out = True, select_hq_bits = indexer_hq_bits,
                qbits_key = qbits_key,
            )
            self.register_submodule(self.idx_query)
            self.indexer = RawTensors(config, ik, [
                "token_key_projection.weight", "token_key_norm.weight", "head_weight_projection.weight",
                "key_value_projection.weight", "key_gate_projection.weight", "key_norm.weight",
                "intra_group_ape",
            ])
            self.register_submodule(self.indexer)

    @override
    def optimizer_targets(self):
        t = [
            [self.q_a_proj.optimizer_targets(), self.kv_a_proj.optimizer_targets(),
             self.linear_gate.optimizer_targets()],
            [self.q_b_proj.optimizer_targets(), self.kv_b_proj.optimizer_targets()],
            [self.o_proj.optimizer_targets()],
        ]
        if self.indexer is not None:
            t[1].append(self.idx_query.optimizer_targets())
        return t

    @override
    def weights_numel(self):
        return sum(m.weights_numel() for m in self.modules)

    # ---- compressed indexer ------------------------------------------------------------------

    def pool_groups(self, token_keys: torch.Tensor) -> torch.Tensor:
        """(b, s, d) normed token keys -> (b, s // kpool, d) group keys (fp32 inside)"""
        b, s, d = token_keys.shape
        r = self.index_kpool
        g = s // r
        grouped = token_keys[:, :g * r].reshape(b, g, r, d).float()
        ix = self.indexer
        values = F.linear(grouped, ix.f("key_value_projection.weight"))
        gates = F.linear(grouped, ix.f("key_gate_projection.weight")) + ix.f("intra_group_ape")
        pooled = (values * gates.softmax(dim = 2)).sum(dim = 2)
        return rms(pooled, self.rms_eps) * ix.f("key_norm.weight")

    def select(self, x: torch.Tensor, q_resid: torch.Tensor, positions: torch.Tensor, params: dict) -> torch.Tensor:
        """(b, s, kept) token indices every query may read, own group last. Keys are the row's own
        tokens (cache-less), so slots and positions coincide."""
        b, s, _ = x.shape
        r = self.index_kpool
        ix = self.indexer
        tk = F.linear(x.float(), ix.f("token_key_projection.weight"))
        tk = rms(tk, self.rms_eps).to(x.dtype).float() * ix.f("token_key_norm.weight")
        tk = tk.to(x.dtype)
        gk = self.pool_groups(tk).to(x.dtype).float()
        g = gk.shape[1]
        nope = self.index_head_dim - self.rope
        cos, sin = rope_angles(positions, self.rope, self.rope_theta)                       # (s, rope)
        query = self.idx_query.forward(q_resid, params).view(b, s, self.index_n_heads, self.index_head_dim).float()
        query = torch.cat((query[..., :nope], rope_rotate(query[..., nope:], cos[None, :, None], sin[None, :, None])),
                          dim = -1)
        gpos = positions[: g * r : r]
        gcos, gsin = rope_angles(gpos, self.rope, self.rope_theta)
        keys = torch.cat((gk[..., :nope], rope_rotate(gk[..., nope:], gcos[None], gsin[None])), dim = -1)
        hw = F.linear(x.float(), ix.f("head_weight_projection.weight")) \
            * (self.index_n_heads ** -0.5 * self.index_head_dim ** -0.5)                     # (b, s, heads)
        own = (torch.arange(s, device = x.device) // r)
        group_ids = torch.arange(g, device = x.device)
        k = min(self.index_topk // r - 1, g)
        picks = []
        for t0 in range(0, s, self.query_chunk):
            t1 = min(t0 + self.query_chunk, s)
            scores = F.relu(torch.matmul(query[:, t0:t1], keys.unsqueeze(1).transpose(-1, -2)))   # (b, q, heads, g)
            logits = torch.matmul(hw[:, t0:t1].unsqueeze(-2), scores).squeeze(-2)                 # (b, q, g)
            logits = logits.masked_fill(group_ids >= own[t0:t1, None], float("-inf"))
            top = logits.topk(k, dim = -1).indices
            picks.append(torch.cat((top, own[t0:t1].expand(b, -1).unsqueeze(-1)), dim = -1))
        groups = torch.cat(picks, dim = 1)                                                   # (b, s, k + 1)
        offs = torch.arange(r, device = x.device)
        return (groups.unsqueeze(-1) * r + offs).flatten(-2).clamp(max = s - 1)

    # ---- forward -----------------------------------------------------------------------------

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        b, s, _ = x.shape
        H = self.num_heads
        position = int(params.get("position", 0) or 0)
        positions = torch.arange(s, device = x.device) + position
        gate = self.linear_gate.forward(x, params).view(b, s, H, self.v)
        q_resid = self.q_a_layernorm.forward_torch(self.q_a_proj.forward(x, params), params, torch.half)
        q = self.q_b_proj.forward(q_resid, params).view(b, s, H, self.qk).transpose(1, 2).float()
        ckv = self.kv_a_proj.forward(x, params)
        k_lat = self.kv_a_layernorm.forward_torch(ckv[..., :self.kv_lora_rank].contiguous(), params, torch.half)
        cos, sin = rope_angles(positions, self.rope, self.rope_theta)
        k_rot = rope_rotate(ckv[..., self.kv_lora_rank:], cos[None], sin[None]).to(x.dtype).float()   # (b, s, rope)
        q_rot = rope_rotate(q[..., self.nope:], cos[None, None], sin[None, None]).to(x.dtype).float()
        q = torch.cat((q[..., :self.nope], q_rot), dim = -1)                                 # (b, H, s, qk)
        kv = self.kv_b_proj.forward(k_lat, params).view(b, s, H, self.nope + self.v).transpose(1, 2)
        k = torch.cat((kv[..., :self.nope].float(), k_rot[:, None].expand(-1, H, -1, -1)), dim = -1)
        v = kv[..., self.nope:].float()

        recalled = None
        if self.indexer is not None and s > self.index_topk:
            recalled = self.select(x, q_resid, positions, params)

        sink = self.sink.f("learnable_sink_param").view(1, H, 1, 1)
        out = torch.empty((b, H, s, self.v), dtype = torch.float, device = x.device)
        for t0 in range(0, s, self.query_chunk):
            t1 = min(t0 + self.query_chunk, s)
            k0 = max(0, t0 - self.sliding_window + 1) if self.layer_type == "sliding" else 0
            w = torch.matmul(q[:, :, t0:t1], k[:, :, k0:t1].transpose(-1, -2)) * self.sm_scale   # (b, H, q, keys)
            qi = torch.arange(t0, t1, device = x.device)[:, None]
            kj = torch.arange(k0, t1, device = x.device)[None, :]
            dead = kj > qi
            if self.layer_type == "sliding":
                dead = dead | (qi - kj >= self.sliding_window)
            dead = dead[None].expand(b, -1, -1)
            if recalled is not None:
                keep = torch.zeros((b, t1 - t0, t1), dtype = torch.bool, device = x.device)
                keep.scatter_(-1, recalled[:, t0:t1].clamp(max = t1 - 1), True)
                dead = dead | ~keep
            w = w.masked_fill(dead[:, None], float("-inf"))
            w = torch.cat((w, sink.expand(b, -1, t1 - t0, -1)), dim = -1)
            p = torch.softmax(w, dim = -1)[..., :-1]
            out[:, :, t0:t1] = torch.matmul(p, v[:, :, k0:t1])
        out = out.transpose(1, 2) * torch.sigmoid(gate.float())                             # (b, s, H, v)
        y = self.o_proj.forward(out.reshape(b, s, H * self.v).half(), params)
        return self.post_proj_norm.forward_torch(y, params, out_dtype or self.out_dtype)


class HyV4EngramTable(Module):
    """
    One Engram hash table: (ngram sizes 2..max) x heads_per_ngram hash heads, one prime-sized
    bucket range per head in one flat table of `row_dim`-wide rows.

    Stored either as the checkpoint's BF16 tensor <key>.weight, read through the fp8 round trip
    the reference applies to every looked-up row (e4m3, one amax / 448 scale per row), or as
    <key>.trellis (exl3_ngram_trellis, one mul1 tail-biting ring per row plus an fp16 scale and a
    per-head bias), which decodes to an approximation of those fp8-rounded rows directly.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_id: int,
        max_ngram_size: int,
        heads_per_ngram: int,
        row_dim: int,
        head_vocab_sizes: list[int],
        pad_id: int,
        seed: int,
        fp8_rows: bool,
        lookup_key: str,
        stream_from_disk: bool | None = None,
    ):
        super().__init__(config, key, None)
        self.layer_id = layer_id
        self.max_ngram_size = max_ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.num_heads = (max_ngram_size - 1) * heads_per_ngram
        assert len(head_vocab_sizes) == self.num_heads
        self.row_dim = row_dim
        self.head_vocab_sizes = torch.tensor(head_vocab_sizes, dtype = torch.long)
        self.head_offsets = self.head_vocab_sizes.cumsum(0) - self.head_vocab_sizes
        self.num_rows = int(self.head_vocab_sizes.sum())
        self.pad_id = pad_id
        self.seed = seed
        self.fp8_rows = fp8_rows
        self.lookup_key = lookup_key
        self.stream_from_disk = stream_from_disk
        self.table = None
        self.K = None
        self.head_bias = None
        self.codebook = None
        self.lookup_table = None
        self.multipliers = None
        self.caps.update({"prefer_cpu": True})

    @staticmethod
    def draw_multipliers(seed: int, layer_id: int, max_ngram_size: int, compressed_vocab_size: int) -> list[int]:
        half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab_size // 2)
        gen = np.random.default_rng(seed + 10007 * layer_id)
        drawn = gen.integers(0, half_bound, size = max_ngram_size, dtype = np.int64)
        return (drawn * 2 + 1).tolist()

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        stc = self.config.stc
        self.lookup_table = stc.get_tensor(self.lookup_key, "cpu", **RAW_KW)
        self.multipliers = torch.tensor(self.draw_multipliers(
            self.seed, self.layer_id, self.max_ngram_size, int(self.lookup_table.max()) + 1), dtype = torch.long)
        table = RowTable.find(stc, self.key, "trellis")
        if table is not None:
            bias = stc.get_tensor(f"{self.key}.head_bias", "cpu", **RAW_KW)
            for name, mine in (("head_offsets", self.head_offsets), ("head_vocab_sizes", self.head_vocab_sizes),
                               ("layer_multipliers", self.multipliers)):
                stored = stc.get_tensor(f"{self.key}.{name}", "cpu", **RAW_KW).long()
                assert torch.equal(stored, mine), f"{self.key}.{name} does not match the config's hash layout"
            self.head_bias = bias.float()
            assert self.head_bias.shape == (self.num_heads, self.row_dim)
            self.codebook = mul1_codebook("cpu")
        else:
            table = RowTable.find(stc, self.key, "weight")
            if table is None:
                raise ValueError(f"No .trellis or .weight tensor found for {self.key}")
        infer_params = getattr(self.config, "infer_params", None)
        stream = self.stream_from_disk
        if stream is None:
            stream = infer_params.ngram_stream_from_disk if infer_params is not None else True
        table.open(stc, stream, allow_bf16 = self.head_bias is None, what = f"Engram table {self.key} held in RAM")
        self.table = table
        assert table.num_rows == self.num_rows, f"{self.key}: {table.num_rows} rows, config says {self.num_rows}"
        if self.head_bias is not None:
            self.K = (table.row_words - 1) * 16 // self.row_dim
            assert table.row_words == words_per_row(self.K, self.row_dim)
        else:
            assert table.row_words == self.row_dim

    @override
    def unload(self):
        if self.table is not None:
            self.table.close()
        self.table = None
        self.device = None
        self.K = None
        self.head_bias = None
        self.codebook = None
        self.lookup_table = None
        self.multipliers = None

    @override
    def get_tensors(self):
        # the table lives out of line (engram table files); the token map stays with the layer
        return {self.lookup_key: self.lookup_table.contiguous()} if self.lookup_table is not None else {}

    @override
    def weights_numel(self):
        return self.num_rows * self.row_dim

    def row_ids(self, input_ids: torch.Tensor, position: int = 0, history: torch.Tensor | None = None) -> torch.Tensor:
        """(b, s) token ids -> (b, s, heads) table rows. Tokens before the row's start read as the
        pad id; `history` (b, max_ngram_size - 1) supplies them for a row that continues one."""
        ids = input_ids.to("cpu", torch.long)
        b, s = ids.shape
        n = self.lookup_table.shape[0]
        lut = self.lookup_table.long()
        ctx = self.max_ngram_size - 1
        if history is None:
            assert position == 0, "a row past position 0 needs its token history"
            history = ids.new_full((b, ctx), self.pad_id)
        full = torch.cat((history.to("cpu", torch.long), ids), dim = 1)
        comp = torch.where(full < n, lut[full.clamp(max = n - 1)], lut[self.pad_id])
        mixed = [comp[:, ctx - k : ctx - k + s] * self.multipliers[k] for k in range(self.max_ngram_size)]
        out = []
        for size in range(2, self.max_ngram_size + 1):
            mix = mixed[0]
            for k in range(1, size):
                mix = torch.bitwise_xor(mix, mixed[k])
            lo = (size - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            out.append(torch.remainder(mix.unsqueeze(-1), self.head_vocab_sizes[lo:hi]) + self.head_offsets[lo:hi])
        return torch.cat(out, dim = -1)

    @staticmethod
    def fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
        """Per-row e4m3 round trip of the reference: scale = amax / 448, values through float8_e4m3fn"""
        x = x.float()
        fmax = torch.finfo(torch.float8_e4m3fn).max
        scale = x.abs().amax(dim = -1, keepdim = True).clamp(min = 1e-12) / fmax
        return (x / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn).float() * scale

    def fetch(self, rows: torch.Tensor, device) -> torch.Tensor:
        """(...,) table rows -> (..., row_dim) fp32 on device"""
        flat = rows.reshape(-1).to("cpu", torch.long)
        uids, inverse = torch.unique(flat, return_inverse = True)
        raw = self.table.fetch(uids)
        if self.K:
            heads = (torch.searchsorted(self.head_offsets, uids, right = True) - 1).clamp(0, self.num_heads - 1)
            dec = dequant_rows(raw, self.K, self.codebook, self.head_bias[heads])
        else:
            dec = raw.float()
            if self.fp8_rows:
                dec = self.fp8_round_trip(dec)
        return dec[inverse].view(*rows.shape, self.row_dim).to(device)

    @override
    def forward(self, x, params, out_dtype = None):
        raise RuntimeError("HyV4EngramTable is read through HyV4Engram")


class HyV4Engram(Module):
    """
    Engram layer: adds gated n-gram rows to the stream stack ahead of a decoder block.
    The norms of this layer are torch RMSNorms built without an eps. torch then takes the machine
    epsilon of the dtype the norm computes in, and it computes bf16 and fp16 inputs in fp32: the
    eps is 1.1920929e-07 whatever the model dtype (measured on torch 2.13: a bf16 input matches
    that formula to 1.6e-3, and is 100 % off the bf16 epsilon 0.0078125). vLLM's serving code
    writes the same value out for the row and value norms. Stream energies entering the first
    Engram layer are about 3e-3, so a wrong eps here changes the gates completely.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        layer_id: int,
        hidden_size: int,
        hc_mult: int,
        max_ngram_size: int,
        heads_per_ngram: int,
        row_dim: int,
        head_vocab_sizes: list[int],
        pad_id: int,
        seed: int,
        fp8_rows: bool,
        norm_eps: float,
        stream_from_disk: bool | None = None,
        no_ngram_ids: tuple = (),
    ):
        super().__init__(config, key, None)
        # positions holding one of these ids own no n-gram and get no residual (image and video
        # placeholders, the pad id); they still count as context for the positions after them
        self.no_ngram_ids = tuple(i for i in no_ngram_ids if i is not None)
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.num_heads = (max_ngram_size - 1) * heads_per_ngram
        self.row_dim = row_dim
        self.norm_eps = norm_eps
        self.table = HyV4EngramTable(
            config, f"{key}.multi_head_embedding.embedding", layer_id, max_ngram_size, heads_per_ngram, row_dim,
            head_vocab_sizes, pad_id, seed, fp8_rows, f"{key}.lookup_table", stream_from_disk,
        )
        self.register_submodule(self.table)
        names = ["value_proj.weight", "value_proj.bias", "unified_pre_gamma"]
        for i in range(hc_mult):
            names += [f"key_projs.{i}.weight", f"key_projs.{i}.bias", f"norm_key.{i}.weight",
                      f"norm_query.{i}.weight", f"unified_per_hc_scales.{i}"]
        names += [f"lookup_per_head_norms.{h}.weight" for h in range(self.num_heads)]
        self.w = RawTensors(config, key, names)
        self.register_submodule(self.w)
        self.caps.update({"prefer_cpu": False})

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        self.table.load(torch.device("cpu"), **kwargs)
        self.w.load(device, **kwargs)

    @override
    def optimizer_targets(self):
        return []

    @override
    def weights_numel(self):
        # the table accounts for itself out of line
        return self.w.weights_numel()

    def delta(self, streams: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """streams (b, s, H, D) fp32, rows (b, s, heads, row_dim) fp32 -> the residual to add"""
        w, eps, D = self.w, self.norm_eps, self.hidden_size
        gamma = torch.stack([w.f(f"lookup_per_head_norms.{h}.weight") for h in range(self.num_heads)])
        e = (rms(rows, eps) * gamma).flatten(-2)                                          # (b, s, heads * row_dim)
        gates = []
        for i in range(self.hc_mult):
            key = F.linear(e, w.f(f"key_projs.{i}.weight"), w.f(f"key_projs.{i}.bias"))
            key = rms(key, eps) * w.f(f"norm_key.{i}.weight")
            query = rms(streams[:, :, i], eps) * w.f(f"norm_query.{i}.weight")
            g = (key * query).sum(dim = -1) * D ** -0.5
            g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
            gates.append(g.sigmoid())
        gates = torch.stack(gates, dim = 2).unsqueeze(-1)                                  # (b, s, H, 1)
        value = F.linear(e, w.f("value_proj.weight"), w.f("value_proj.bias"))
        signal = w.f("unified_pre_gamma") * rms(value, eps)
        scales = torch.stack([w.f(f"unified_per_hc_scales.{i}") for i in range(self.hc_mult)])   # (H, D)
        return (gates + scales) * signal.unsqueeze(2)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        ids = params.get("input_ids")
        assert ids is not None, "HyV4Engram needs params['input_ids']"
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        rows = self.table.row_ids(ids, int(params.get("position", 0) or 0), params.get("engram_history"))
        emb = self.table.fetch(rows, x.device)
        d = self.delta(x, emb)
        if self.no_ngram_ids:
            own = torch.ones_like(ids, dtype = torch.bool)
            for i in self.no_ngram_ids:
                own &= ids != i
            if not bool(own.all()):
                d = d * own.to(d.device)[:, :, None, None]
        return x + d


class HyV4Block(Module):
    """Decoder block: hyper-connection site, norm, gated MLA, sandwich norm (inside the
    attention), write back; the same around the MLP, whose sandwich norm is `mlp_post_norm`."""

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        attn_hc: HyV4HyperConnection,
        attn_norm: RMSNorm,
        attn: Module,
        mlp_hc: HyV4HyperConnection,
        mlp_norm: RMSNorm,
        mlp: Module,
        mlp_post_norm: RMSNorm,
        qbits_key: str = "bits",
    ):
        super().__init__(config, key, None)
        self.layer_idx = layer_idx
        self.attn_hc, self.attn_norm, self.attn = attn_hc, attn_norm, attn
        self.mlp_hc, self.mlp_norm, self.mlp, self.mlp_post_norm = mlp_hc, mlp_norm, mlp, mlp_post_norm
        self.qbits_key = qbits_key
        for m in (attn_hc, attn_norm, attn, mlp_hc, mlp_norm, mlp, mlp_post_norm):
            self.register_submodule(m)
        self.num_slices = getattr(mlp, "num_slices", 1)

    @override
    def optimizer_targets(self):
        return [self.attn.optimizer_targets(), self.mlp.optimizer_targets()]

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        post, y = self.attn_hc.mix(x)
        y = self.attn_norm.forward_torch(y, params, torch.half)
        y = self.attn.forward(y, params)
        x = HyV4HyperConnection.apply(x, y, post)

        post, y = self.mlp_hc.mix(x)
        y = self.mlp_norm.forward_torch(y, params, torch.half)
        y = self.mlp.forward(y, params)
        y = self.mlp_post_norm.forward_torch(y, params, torch.float)
        return HyV4HyperConnection.apply(x, y, post)


class HyV4SideLayer(Module):
    """A group of side-model tensors (DFly2 drafter, vision tower) for the uncalibrated side
    quantization: Linears that take a bitrate, and everything else as stored. No forward."""

    def __init__(self, config: Config | None, key: str, linears: list[Linear], raw: RawTensors | None):
        super().__init__(config, key, None)
        self.linears = linears
        self.raw = raw
        for m in linears:
            self.register_submodule(m)
        self.register_submodule(raw)

    @override
    def optimizer_targets(self):
        return [m.optimizer_targets() for m in self.linears]

    @override
    def forward(self, x, params, out_dtype = None):
        raise RuntimeError("side-model layers are quantized without a forward pass")
