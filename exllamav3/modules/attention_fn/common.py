from typing import NamedTuple, Protocol
import torch

class AttnArgs(NamedTuple):
    bsz: int
    q_len: int
    num_q_heads: int
    dim: int
    kv_len: int
    num_kv_heads: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    k_cache: torch.Tensor | None
    v_cache: torch.Tensor | None
    causal: bool
    sm_scale: float
    cu_seqlens: torch.Tensor | None
    max_seqlen: int | None
    window_size: int | None
    softcap: float
    block_table: torch.Tensor | None
    cache_seqlens: torch.Tensor | None
    non_causal_spans: list | None = None
    q_cache: tuple | None = None    # (qk, sk, qv, sv, k_bits, v_bits): packed quantized cache
    sinks: torch.Tensor | None = None    # learned per-q-head sink logits (gpt-oss style)

    def sanity_check(self):
        # Cache must be paged
        if self.q_cache is not None:
            assert self.block_table is not None and self.cache_seqlens is not None
            assert self.k_cache is None and self.v_cache is None
        else:
            assert (
                (self.k_cache is not None) ==
                (self.v_cache is not None) ==
                (self.block_table is not None) ==
                (self.cache_seqlens is not None)
            )
        # cu_seqlens requires max seqlen
        assert (
            (self.cu_seqlens is not None) ==
            (self.max_seqlen is not None)
        )
        # GQA must divide q_heads evenly
        assert self.num_q_heads % self.num_kv_heads == 0
        # Must have queries
        assert self.bsz >= 1
        assert self.q_len >= 1
        # Sane head dim
        assert 16 <= self.dim <= 1024

    def has_kv_cache(self) -> bool:
        return self.k_cache is not None

    def is_gqa(self):
        return self.num_q_heads != self.num_kv_heads

    def is_varlen(self):
        return self.cu_seqlens is not None

    def get_window_size(self):
        if self.window_size is None or self.window_size == -1:
            return -1, -1
        return self.window_size, 0

    def is_swa(self):
        return self.window_size is not None and self.window_size != -1


class AttnFn(Protocol):
    def __call__(self, args: AttnArgs) -> torch.Tensor | None: ...


def get_non_causal_span_arglist(args: AttnArgs):
    arglist = []
    for span in args.non_causal_spans:
        a, b, nc = span[:3]
        pre = span[3] if len(span) > 3 else 0
        l = b - a
        window_size = (
            (max(args.window_size, l + pre), l - 1 if nc else 0)
            if args.window_size is not None and args.window_size > 0 and nc else
            args.get_window_size()
        )
        if args.q_cache is not None:
            # Quant-direct: the whole chunk's K/V was quantized into the paged cache before
            # dispatch, so each span is a pure read over the packed cache up to kv position
            # cache_seqlens + b. Rows past b are already written but sit above the length the
            # kernel derives from cache_seqlens + pre_appended_len, so they are never read
            qk, sk, qv, sv, k_bits, v_bits = args.q_cache
            arglist.append(dict(
                q = args.q[:, a: b].contiguous(),
                k = None,
                v = None,
                k_cache = qk,
                v_cache = qv,
                block_table = args.block_table,
                cache_seqlens = args.cache_seqlens + a,
                causal = not nc,
                softmax_scale = args.sm_scale,
                window_size = window_size,
                softcap = args.softcap,
                sinks = args.sinks,
                qc = (sk, sv, k_bits, v_bits),
                pre_appended_len = l,
                n_kv_heads_override = args.num_kv_heads,
            ))
            continue
        # Only the Triton wrappers take a sinks argument; backends without support decline
        # sinked calls before expanding the spans, so the key is omitted when unused
        extra = dict(sinks = args.sinks) if args.sinks is not None else {}
        arglist.append(dict(
            q = args.q[:, a: b],
            k = args.k[:, a: b],
            v = args.v[:, a: b],
            k_cache = args.k_cache,
            v_cache = args.v_cache,
            block_table = args.block_table,
            cache_seqlens = args.cache_seqlens + a,
            causal = not nc,
            softmax_scale = args.sm_scale,
            window_size = window_size,
            softcap = args.softcap,
            **extra,
        ))
    return arglist