"""
Trellis quantization of token-embedding tables (the exl3_trellis_embed format consumed by the
Embedding module's trellis branch and util/convert_embedding.py).

One packed row is (G + D*K/16) little-endian uint16 words (stored as int16): the G = D/256
per-256-column-group fp16 pre-scales followed by the D*K-bit tail-biting ring bitstream
(codec: modules/quant/exl3_lib/embed_trellis.py, bit-exact reference for the fused GPU
kernel). The table is quantized in the QTIP-transformed space: column scales (per-column RMS
over ALL rows, fp16) -> per-element signs from a seeded LCG keyed by the table row id ->
256-D Sylvester Hadamard groups, then per-group pre-scale + trellis VQ (the ring spans the
full row; the fp16 pre-scales are LS-refit against the source rows after encoding and the
refit values are the stored scale words: the decode is linear in the scale word, so the fit
s* = (w.q)/(q.q) minimizes the per-group source-space squared error for the fixed codes
(measured +0.01..0.02 dB SQNR over K = 4..8 on real tables).

PROGRAMMATIC ENTRY POINT (stable signature; external tooling - e.g. vocab-prune suites -
imports this and never re-implements the encoder or codec):

    quantize_trellis_table(
        source,                      # str: HF model dir or plain .safetensors dir, OR
                                     # torch.Tensor (V, D) source rows (any float dtype)
        out_path,                    # output .safetensors file
        K,                           # bits per position, 4..8 (fused kernel range)
        seed = 0,                    # LCG sign-stream seed (pinned in metadata)
        key = "model.embed_tokens.weight",   # tensor key when source is a directory
        chunk_rows = 8192,           # rows per work chunk
        limit_rows = None,           # quantize only the first N (kept) rows (testing)
        resume = False,              # continue an interrupted run (parameters must match)
        threads = 0,                 # encoder threads (0 = all cores, capped so the per-thread
                                     # history scratch stays under ~512 MiB)
        devices = None,              # CUDA device indices for the transform/quality passes
        quality_sample = None,       # report quality on a deterministic sample of at most N
                                     # encoded rows across the whole run, allocated
                                     # proportionally per chunk (minimum 1 per chunk;
                                     # 0 = skip the quality report)
        verbose = True,
    ) -> dict                        # stats: rows, K, seed, encoder, rfn, sqnr_db, elapsed...

Determinism: identical (source rows, K, seed, encoder, chunk_rows) produces a byte-identical output file - the AVX2 encoder is per-row
deterministic and thread-count independent, the transform has no reductions, and the column
scales accumulate in fp64 in row order (chunk_rows participates because it fixes the
reduction tree).

Encoder decision (measured 2026-09-23): the production quantize_tiles CUDA kernel hardcodes
tile lengths 256/160 (exllamav3_ext/quant/quantize.cu TORCH_CHECK), so it does not accept the
D = 5120 full-row rings this format needs; the encoder here is the AVX2 CPU port of the same
tail-biting Viterbi (quantize_tiles_kernel.cuh two-pass scheme, parameterized tile length).
"""

from __future__ import annotations
import glob
import json
import math
import os
import platform
import struct
import threading
import time
import torch

from ..modules.quant.exl3_lib.embed_trellis import (  # noqa: F401  (re-exported)
    GROUP, FORMAT, FORMAT_VERSION, CS_HEURISTIC, CS_MIN,
    words_per_row, mul1_codebook, pack_rows, unpack_rows, dequant_rows,
    lcg_signs, hadamard_butterfly, dequant_rows_transformed, forward_transform,
    quantize_rows_grouped, ls_refit_scales,
)
from .ngram import StreamingSafetensorsWriter, read_table_tensor

DEFAULT_TENSOR_KEY = "model.embed_tokens.weight"

# --------------------------------------------------------------------------- AVX2 encoder
#
# CPU port of exllamav3_ext/quant/quantize_tiles_kernel.cuh with the tile length L
# parameterized (the CUDA kernel hardcodes 256/160 instances) and no bias (bias = 0 for the
# token table). Same two-pass tail-biting scheme as the kernel, except pass 2's constraint
# placement:
#   pass 1: D DP steps in ring order starting at D/2, free start; history for natural
#           positions < D/2 only.
#   trace 1: from the final argmin edge back to natural position 0 -> end_state.
#   pass 2: D DP steps in natural order, position 0 constrained to in_edge == end_state
#           (the kernel instead masks pre_state at every step; constraining only the ring
#           start is sufficient for the traced path to be the min-cost ring-closed path).
#   trace 2: states[i] = (prev_edge << K) | edge.
# fp32 costs (the CUDA kernel uses fp16 costs; fp32 only breaks ties differently, marginally
# closer to the true optimum). The inner loop blends on a single LT mask so the first k on
# exact ties wins, bit-identical to the scalar reference `if (t < best) ...`.

_CPP_ENCODER = r"""
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>

#define INF 3.402823466e+38f

// One DP step over all out_edges for position ri.
//   wv        : the (pre-scaled) weight at this position
//   cin       : incoming cost vector (edges floats); nullptr = first step (cost 0)
//   cout      : outgoing cost vector
//   hist      : (D, edges) uint16 history; written iff write_hist
//   pre_state : -1 = unconstrained; else position-0 constraint in_edge == pre_state
static inline void dp_step(
    const float* cb,
    int ri, float wv,
    const float* cin, float* cout,
    uint16_t* hist, bool write_hist,
    int pre_state, int K, int edges, int kshift)
{
    const __m256 w1 = _mm256_set1_ps(wv);
    for (int e0 = 0; e0 < edges; e0 += 8)
    {
        const int eh = e0 >> K;
        __m256 best = _mm256_set1_ps(INF);
        __m256i bestk = _mm256_setzero_si256();

        if (pre_state >= 0)
        {
            // Only k = pre_state >> kshift survives, and only if this group's
            // e >> K matches pre_state's low kshift bits.
            const int k = pre_state >> kshift;
            if (eh == (pre_state & ((1 << kshift) - 1)))
            {
                const __m256 c8 = _mm256_loadu_ps(cb + (size_t) k * edges + e0);
                __m256 d = _mm256_sub_ps(c8, w1);
                best = _mm256_mul_ps(d, d);
                bestk = _mm256_set1_epi32(k);
            }
        }
        else
        {
            for (int k = 0; k < (1 << K); ++k)
            {
                const __m256 c8 = _mm256_loadu_ps(cb + (size_t) k * edges + e0);
                __m256 d = _mm256_sub_ps(c8, w1);
                __m256 t = _mm256_mul_ps(d, d);
                t = _mm256_add_ps(t, _mm256_set1_ps(cin ? cin[(k << kshift) | eh] : 0.0f));
                const __m256 lt = _mm256_cmp_ps(t, best, _CMP_LT_OQ);
                best = _mm256_blendv_ps(best, t, lt);
                bestk = _mm256_blendv_epi8(bestk, _mm256_set1_epi32(k), _mm256_castps_si256(lt));
            }
        }
        _mm256_storeu_ps(cout + e0, best);
        if (write_hist)
        {
            int bk[8];
            _mm256_storeu_si256((__m256i*) bk, bestk);
            uint16_t* h = hist + (size_t) ri * edges + e0;
            for (int q = 0; q < 8; ++q)
                h[q] = (uint16_t) ((bk[q] << kshift) | eh);
        }
    }
}

void trellis_encode(
    at::Tensor w,        // (E, D) fp32 pre-scaled rows
    at::Tensor cb,       // (65536,) fp32 codebook
    at::Tensor states,   // (E, D) int16 out
    int64_t K, int64_t D, int64_t nthreads)
{
    const int E = (int) w.size(0);
    const int KRS = 16 - (int) K;          // state = (k << KRS) | e
    const int edges = 1 << KRS;
    const int kshift = KRS - (int) K;      // in_edge = (k << kshift) | (e >> K)
    const float* wpp = w.data_ptr<float>();
    const float* cbp = cb.data_ptr<float>();
    uint16_t* stp = (uint16_t*) states.data_ptr<int16_t>();

    const int nt = E < 256 ? 1 : (int) std::min<int64_t>(nthreads, E);
    const int per = (E + nt - 1) / nt;
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t)
    {
        const int r0 = t * per;
        const int r1 = std::min(E, r0 + per);
        if (r0 >= r1) continue;
        pool.emplace_back([&, r0, r1]() {
            // per-thread scratch: two cost vectors + history (hist is row-reused,
            // so it must not be shared across threads)
            std::vector<float> ca(edges), cbu(edges);
            std::vector<uint16_t> histbuf((size_t) D * edges);
            uint16_t* hisp = histbuf.data();
            for (int r = r0; r < r1; ++r)
            {
                const float* wrow = wpp + (size_t) r * D;
                // ---- pass 1: ring order from D/2, free start ----
                {
                    int ri = (int) (D / 2);
                    dp_step(cbp, ri, wrow[ri], nullptr, ca.data(),
                            hisp, ri < D / 2, -1, (int) K, edges, kshift);
                    for (int i = 1; i < D; ++i)
                    {
                        ri = (int) ((i + D / 2) % D);
                        dp_step(cbp, ri, wrow[ri], ca.data(), cbu.data(),
                                hisp, ri < D / 2, -1, (int) K, edges, kshift);
                        std::swap(ca, cbu);
                    }
                }
                // argmin (deterministic: smallest edge on ties)
                int e_star = 0;
                float bestv = ca[0];
                for (int e = 1; e < edges; ++e)
                    if (ca[e] < bestv) { bestv = ca[e]; e_star = e; }
                // ---- trace 1: back to natural position 0 ----
                int edge = e_star;
                for (int i = D - 1; i >= 0; --i)
                {
                    const int ri = (int) ((i + D / 2) % D);
                    edge = hisp[(size_t) ri * edges + edge];
                    if (ri == 0) break;
                }
                const int end_state = edge;
                // ---- pass 2: natural order, position 0 constrained ----
                {
                    dp_step(cbp, 0, wrow[0], nullptr, ca.data(),
                            hisp, true, end_state, (int) K, edges, kshift);
                    for (int i = 1; i < D; ++i)
                    {
                        dp_step(cbp, i, wrow[i], ca.data(), cbu.data(),
                                hisp, true, -1, (int) K, edges, kshift);
                        std::swap(ca, cbu);
                    }
                }
                // trace 2: start from end_state (position 0's constraint forces
                // e_{D-1} = end_state; the traced path is the min-cost path ending there)
                edge = end_state;
                uint16_t* strow = stp + (size_t) r * D;
                for (int i = D - 1; i >= 0; --i)
                {
                    const int prev = hisp[(size_t) i * edges + edge];
                    strow[i] = (uint16_t) ((prev << (int) K) | edge);
                    edge = prev;
                }
            }
        });
    }
    for (auto& th : pool) th.join();
}
"""

_encoder_mod = None


def _encoder_build():
    global _encoder_mod
    if _encoder_mod is None:
        machine = platform.machine()
        if machine not in {"x86_64", "AMD64"}:
            raise RuntimeError(
                f"the trellis embed encoder requires an x86-64 AVX2 host (it compiles with "
                f"-march=x86-64-v3), but this machine reports architecture {machine!r}")
        from torch.utils.cpp_extension import load_inline
        _encoder_mod = load_inline(
            name = "trellis_embed_encode_ext",
            cpp_sources = _CPP_ENCODER,
            functions = ["trellis_encode"],
            extra_cflags = ["-O3", "-march=x86-64-v3"],
            with_cuda = False,
            verbose = False,
        )
    return _encoder_mod


_cb_f32_cache = {}


def _cb_f32():
    if "cpu" not in _cb_f32_cache:
        _cb_f32_cache["cpu"] = mul1_codebook().float().contiguous()
    return _cb_f32_cache["cpu"]


def encode_rows(w_pre: torch.Tensor, K: int, D: int, threads: int = 0, chunk: int = 0) -> torch.Tensor:
    """Viterbi states (N, D) int16 for pre-scaled transformed rows w_pre (N, D) fp32 CPU."""
    mod = _encoder_build()
    if threads <= 0:
        # per-thread history scratch is D * 2^(16-K) uint16 (10 MiB at K = 6, D = 5120);
        # cap the default (0 = all cores) so the transient scratch stays under ~512 MiB.
        # An explicitly requested thread count is the user's choice and is not capped.
        per_thread_bytes = D * 2 ** (16 - K) * 2
        threads = min(os.cpu_count() or 4, max(1, (512 * 2**20) // per_thread_bytes))
    N = w_pre.shape[0]
    if chunk <= 0:
        chunk = max(64, min(N, 2048))
    states = torch.empty(N, D, dtype = torch.int16)
    cb = _cb_f32()
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        mod.trellis_encode(w_pre[c0:c1].contiguous(), cb, states[c0:c1], K, D, threads)
    return states

class EmbedSource:
    """
    Row-addressable view of a source embedding tensor inside a directory of .safetensors
    files (an HF model dir or a plain dump): resolves the tensor key once (exact or last-
    two-segments suffix match; the match must be unique), then serves contiguous row ranges
    and indexed ranges without loading the table.
    """

    def __init__(self, directory: str, key: str = DEFAULT_TENSOR_KEY):
        self.directory = directory
        suf = ".".join(key.split(".")[-2:])
        files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
        assert files, f"no .safetensors files in {directory}"
        from safetensors import safe_open
        self.file = None
        self.key = None
        for f in files:
            try:
                with safe_open(f, framework = "pt") as h:
                    keys = list(h.keys())
            except Exception:
                continue
            match = [k for k in keys if k == key or k.endswith("." + suf) or k == suf]
            if match:
                assert len(match) == 1, \
                    f"multiple embedding tensors match {key} in {f}: {match}; pass an unambiguous tensor key"
                self.file, self.key = f, match[0]
                break
        assert self.file is not None, \
            f"embedding tensor {key} not found in any .safetensors file in {directory}"
        with safe_open(self.file, framework = "pt") as h:
            sl = h.get_slice(self.key)
            shape = sl.get_shape()
            self.dtype = sl.get_dtype()
        assert len(shape) == 2, f"{self.key}: expected a 2-D table, got shape {shape}"
        self.num_rows, self.D = int(shape[0]), int(shape[1])
        self._local = threading.local()

    def _open(self):
        h = getattr(self._local, "h", None)
        if h is None:
            from safetensors import safe_open
            h = safe_open(self.file, framework = "pt")
            self._local.h = h
        return h

    def read_rows(self, start: int, end: int) -> torch.Tensor:
        return self._open().get_slice(self.key)[start:end]

    def read_rows_indexed(self, idx: torch.Tensor) -> torch.Tensor:
        """Rows at global indices (sorted, duplicates allowed), coalesced into runs."""
        idx = idx.to(torch.int64)
        assert idx.dim() == 1
        iv = idx.tolist()
        parts = []
        h = self._open()
        n = len(iv)
        i = 0
        while i < n:
            j = i + 1
            while j < n and iv[j] == iv[j - 1] + 1:
                j += 1
            parts.append(h.get_slice(self.key)[iv[i]:iv[j - 1] + 1])
            i = j
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def close(self):
        # Release the thread-local safe_open handle (this safetensors version exposes no
        # close() on the handle: dropping the reference frees the file descriptor via the
        # refcounted Rust object)
        self._local.h = None


# --------------------------------------------------------------------------- engine

def compute_column_scales(source_rows_iter, D: int) -> torch.Tensor:
    """
    Per-column RMS over all source rows, fp16. Accumulated in fp64 in row order: identical
    (rows, chunking) always produce identical scales (reduction tree fixed by how the
    caller chunks the iterator). source_rows_iter yields (<=chunk_rows, D) float tensors.
    """
    acc = torch.zeros(D, dtype = torch.float64)
    n = 0
    for rows in source_rows_iter:
        # same sanitization as the encode path (quantize_rows_grouped's nan_to_num): a raw
        # NaN/inf would poison the fp64 accumulator and fp16-clamp does not repair it, so
        # the stored scale word would go non-finite and corrupt every decoded row while
        # the encoder still emits a "valid" packed table
        r = torch.nan_to_num(rows.float(), nan = 0.0, posinf = 0.0, neginf = 0.0)
        acc += r.square().sum(dim = 0, dtype = torch.float64)
        n += r.shape[0]
    assert n > 0, "no source rows"
    rms = (acc / n).sqrt()
    # floor = fp16 min subnormal: a zero (or fp16-underflowed) column RMS would otherwise
    # become a 0.0 divisor in forward_transform, and the Hadamard butterfly would spread the
    # resulting inf/nan across the entire 256-column group (silent whole-group corruption)
    return rms.clamp(min = 5.960464477539063e-8).to(torch.float16)


def quantize_trellis_table(
    source,
    out_path: str,
    K: int,
    seed: int = 0,
    key: str = DEFAULT_TENSOR_KEY,
    chunk_rows: int = 8192,
    limit_rows: int | None = None,
    resume: bool = False,
    threads: int = 0,
    devices: list[int] | None = None,
    quality_sample: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Quantize a token-embedding table to the exl3_trellis_embed format. See the module
    docstring for the contract of this stable entry point (source = directory or rows
tensor). Returns a stats dict with
    rows/processed_rows/measured_rows/K/seed/encoder/rfn/sqnr_db/refit_moved_frac/
    refit_mean_rel/refit_gain_db/refit_nonfinite/refit_degenerate/elapsed/bytes. The
    quantization quality (SQNR dB + rfn) is measured against the source rows over
    measured_rows encoded rows (all of them unless quality_sample limits the measurement);
    the refit_* fields size the LS scale-word refit (analytic, see ls_refit_scales).
    """
    assert 4 <= K <= 8, "the fused dequant kernel supports K in 4..8"
    assert chunk_rows > 0
    assert os.path.isdir(str(source)) if isinstance(source, str) else torch.is_tensor(source), \
        "source must be a directory of .safetensors files or a (V, D) rows tensor"
    # create the output directory: this is the stable programmatic entry point, and a
    # bare FileNotFoundError from the writer's open() would give external callers no hint
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok = True)


    if isinstance(source, str):
        src = EmbedSource(source, key)
        src_D, src_name = src.D, f"{src.key} in {src.file}"
        row_source = src
    else:
        w = source
        assert w.dim() == 2, "rows tensor source must be (V, D)"
        class _RowsSource:
            def __init__(self, w):
                self.w = w
                self.num_rows, self.D = int(w.shape[0]), int(w.shape[1])
                self.key = "<tensor>"
            def read_rows(self, start, end):
                return self.w[start:end]
        row_source = _RowsSource(w)
        src_D, src_name = row_source.D, "rows tensor"
    D = src_D
    assert D % GROUP == 0, f"hidden size {D} must be a multiple of {GROUP}"
    G = D // GROUP
    # the output stem must match the RUNTIME module key - the RESOLVED source key, not
    # the raw argument: EmbedSource suffix-matches, so key "embed_tokens.weight" may
    # resolve to "model.embed_tokens.weight"; deriving the stem from the argument would
    # emit a file the loader silently ignores (fp16 fallback)
    base_key = row_source.key if row_source.key != "<tensor>" else key
    tensor_key = base_key.rsplit(".", 1)[0] if base_key.endswith(".weight") else base_key

    total_out = row_source.num_rows
    if limit_rows is not None:
        total_out = min(total_out, limit_rows)

    if verbose:
        print(f" -- source table: {src_name} {row_source.num_rows} x {D} ({row_source.dtype if hasattr(row_source, 'dtype') else source.dtype}), "
              f"quantizing {total_out} rows at K = {K}, seed = {seed}")
        print(f" -- output tensors: {tensor_key}.weight_trellis, {tensor_key}.col_scales")

    # ---------------------------------------------------------------- column scales
    col_scales = read_table_tensor(out_path, f"{tensor_key}.col_scales", torch.float16) if resume else None
    if col_scales is None:
        if resume and os.path.exists(out_path):
            # the writer's resume branch keeps the small-tensor bytes (col_scales) written
            # before the stream region: refitting fresh while the file keeps the stale
            # bytes would encode rows against scales that never get stored, and every
            # later decode would use mismatched scales
            raise ValueError(
                f"cannot resume {out_path}: the stored column scales are unreadable "
                f"(truncated or corrupt); start a fresh run without resume")
        t0 = time.time()
        def rows_iter():
            for lo in range(0, row_source.num_rows, chunk_rows):
                yield row_source.read_rows(lo, min(lo + chunk_rows, row_source.num_rows))
        col_scales = compute_column_scales(rows_iter(), D)
        if verbose:
            print(f" -- column scales fitted in {time.time() - t0:.0f} s")
    elif verbose:
        print(f" -- resuming with stored column scales")

    # ---------------------------------------------------------------- encode + write
    W = words_per_row(D, K, G)
    writer = StreamingSafetensorsWriter(
        out_path,
        stream_tensors = [(f"{tensor_key}.weight_trellis", (total_out, W))],
        stream_dtype_str = "I16",
        small_tensors = {f"{tensor_key}.col_scales": col_scales},
        metadata = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "K": str(K),
            "G": str(G),
            "seed": str(seed),
            "codebook": "mul1",
            "transform": "qtip1",
            "rows": str(total_out),
            "hidden": str(D),
        },
        resume = resume,
        chunk_rows = chunk_rows,
    )
    start_row = writer.resume_rows
    if verbose and start_row:
        print(f" -- resuming at row {start_row} of {total_out} ({start_row / total_out * 100:.1f}% done)")

    dev = f"cuda:{devices[0]}" if devices and torch.cuda.is_available() else "cpu"
    cs_f32_dev = col_scales.float().to(dev)
    # quality (SQNR/rfn) is measured on the decoder output vs the source rows; a sample
    # (deterministic generator seed 0) limits the measured rows without touching the encode
    # (only the sampled rows are decoded: the chain is row-independent, so the sampled
    # rows decode bit-identically to a full decode followed by the same pick)
    qsample_all = quality_sample is None or quality_sample >= total_out
    qgen = torch.Generator().manual_seed(0)
    acc = {"err_sq": 0.0, "src_sq": 0.0}
    # LS scale-word refit diagnostics (analytic, from the refit pass itself; the shipped
    # sqnr_db/rfn stay decoder-output measures)
    racc = {"groups": 0, "moved": 0, "rel_shift": 0.0, "nonfinite": 0, "degenerate": 0,
            "err_sq_s0": 0.0, "err_sq_refit": 0.0}
    measured_rows = 0
    t0 = time.time()
    enc_rows = 0
    for lo in range(start_row, total_out, chunk_rows):
        hi = min(lo + chunk_rows, total_out)
        rows = row_source.read_rows(lo, hi)
        w = torch.nan_to_num(rows.float(), nan = 0.0, posinf = 0.0, neginf = 0.0)
        out_ids = torch.arange(lo, hi, dtype = torch.int64)   # sign stream keys = OUTPUT rows
        y = forward_transform(w.to(dev), cs_f32_dev, seed, out_ids.to(dev)).cpu()
        # the LS refit needs the SOURCE rows (the encoder only ever sees the transformed
        # ones), so it is handed to the codec-level encoder as a callback: it replaces the
        # heuristic pre-scales as the stored scale words and leaves the code words untouched.
        # It costs one extra unit-scale decode of the chunk (a few hundred MiB of transients
        # and a few seconds per chunk, against a ~30 min encode) for that fractional dB
        chunk_rinfo = []
        def refit(states, scale0):
            scales, rinfo = ls_refit_scales(states, w, scale0, col_scales, seed, out_ids, K, D, G)
            chunk_rinfo.append(rinfo)
            return scales, rinfo
        packed, _ = quantize_rows_grouped(y, K, D,
                                          lambda wp, KK, DD: encode_rows(wp, KK, DD, threads),
                                          group = GROUP, refit = refit)
        for rinfo in chunk_rinfo:
            for k in racc:
                racc[k] += rinfo[k]
        writer.write_chunk(packed)
        if quality_sample == 0:
            pass
        else:
            if not qsample_all:
                pick = torch.randperm(hi - lo, generator = qgen)[:max(1, quality_sample * (hi - lo) // total_out)]
                # decode only the sampled rows (row-independent chain -> bit-identical
                # to decoding the whole chunk and then picking)
                xhat = dequant_rows_transformed(packed[pick], col_scales, K, seed, D, G,
                                                row_ids = out_ids[pick])
                w_q = w[pick]
            else:
                xhat = dequant_rows_transformed(packed, col_scales, K, seed, D, G, row_ids = out_ids)
                w_q = w
            measured_rows += xhat.shape[0]
            acc["err_sq"] += (xhat - w_q).double().square().sum().item()
            acc["src_sq"] += w_q.double().square().sum().item()
        enc_rows += hi - lo
        if verbose:
            el = time.time() - t0
            sq = (f", sqnr so far {10 * math.log10(max(acc['src_sq'], 1e-30) / max(acc['err_sq'], 1e-30)):.2f} dB"
                  if quality_sample != 0 else "")
            print(f"   encoded {hi}/{total_out} rows ({enc_rows / max(el, 1e-9):.0f} rows/s{sq})",
                  flush = True)
    writer.finalize()

    elapsed = time.time() - t0
    measured = quality_sample != 0 and acc["src_sq"] > 0
    rfn = math.sqrt(acc["err_sq"] / acc["src_sq"]) if measured else None
    # a non-finite err_sq must not read as a perfect inf SQNR: it means the decode
    # produced NaN/inf, i.e. a corrupted table, not a lossless one (only an exact-zero
    # error is genuinely lossless)
    if not measured:
        sqnr = None
    elif acc["err_sq"] == 0:
        sqnr = float("inf")
    elif math.isfinite(acc["err_sq"]):
        sqnr = 10 * math.log10(acc["src_sq"] / acc["err_sq"])
    else:
        sqnr = None
    bpw = (D * K + 16 * G) / D
    # LS scale-word refit report: the refit is a real (provable) error reduction but a small
    # one, so its size belongs in the run record - without it a +0.01 dB refinement is
    # indistinguishable from a no-op after the fact
    refit_groups = int(racc["groups"])
    refit_gain = (10 * math.log10(racc["err_sq_s0"] / racc["err_sq_refit"])
                  if refit_groups > 0 and racc["err_sq_s0"] > 0 and racc["err_sq_refit"] > 0 and
                  math.isfinite(racc["err_sq_s0"]) and math.isfinite(racc["err_sq_refit"]) else None)
    stats = {
        "rows": total_out,
        "processed_rows": enc_rows,
        "measured_rows": measured_rows,
        "K": K,
        "seed": seed,
        "encoder": "cpu_avx2_viterbi",
        "rfn": rfn,
        "sqnr_db": sqnr,
        "refit_moved_frac": (racc["moved"] / refit_groups) if refit_groups else None,
        "refit_mean_rel": (racc["rel_shift"] / refit_groups) if refit_groups else None,
        "refit_gain_db": refit_gain,
        "refit_nonfinite": racc["nonfinite"],
        "refit_degenerate": racc["degenerate"],
        "bpw": bpw,
        "elapsed": elapsed,
        "rows_per_s": enc_rows / max(elapsed, 1e-9),
        "bytes": os.path.getsize(out_path),
    }
    if verbose:
        q = f"SQNR {sqnr:.2f} dB, rfn {rfn:.5f}" if measured and sqnr is not None else \
            ("decode produced non-finite values (corrupted table)" if measured else "quality report skipped")
        rf = "no scale refit"
        if refit_groups:
            gain = f", {refit_gain:+.3f} dB" if refit_gain is not None else ""
            rf = (f"scale words refit for {stats['refit_moved_frac'] * 100:.1f}% of groups "
                  f"({stats['refit_mean_rel'] * 100:.3f}% mean shift{gain})")
        print(f" -- done: {enc_rows} rows in {elapsed:.0f} s ({stats['rows_per_s']:.0f} rows/s), "
              f"{q}, {rf}, {bpw:.3f} bpw, "
              f"{stats['bytes'] / 2**20:.0f} MiB -> {out_path}")
    if racc["nonfinite"]:
        print(f" [warn] {racc['nonfinite']} of {refit_groups} LS-optimal scale words left the fp16 "
              f"range; those groups kept the heuristic pre-scale (the table stays finite)")
    return stats





# --------------------------------------------------------------------------- reader

class TrellisEmbedTableReader:
    """
    On-demand reader for an exl3_trellis_embed table file: parses the safetensors header,
    validates the format metadata and serves arbitrary row indices (dequantized via the
    bit-exact CPU reference codec). This is the converter's verification gate and the
    programmatic post-prune verification path.
    """

    def __init__(self, path: str):
        self.path = path
        self.fd = None   # set last: an init failure must not leave a dangling fd
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        self.metadata = header.pop("__metadata__", {})
        assert self.metadata.get("format") == FORMAT, f"{path}: not an {FORMAT} file"
        assert self.metadata.get("version") == FORMAT_VERSION, f"{path}: unsupported version {self.metadata.get('version')}"
        self.K = int(self.metadata["K"])
        self.G = int(self.metadata["G"])
        self.seed = int(self.metadata["seed"])
        self.rows = int(self.metadata["rows"])
        self.hidden = int(self.metadata["hidden"])
        assert self.G == self.hidden // GROUP

        def resolve(suffix):
            keys = [k for k in header if k.endswith(suffix)]
            assert len(keys) == 1, f"expected one *{suffix} tensor in {path}, found {len(keys)}"
            return keys[0]

        info = header[resolve(".weight_trellis")]
        assert info["dtype"] == "I16" and len(info["shape"]) == 2
        self.num_rows, self.row_words = info["shape"]
        assert self.num_rows == self.rows, f"{path}: metadata rows {self.rows} != tensor rows {self.num_rows}"
        assert self.row_words == words_per_row(self.hidden, self.K, self.G), \
            f"{path}: tensor width {self.row_words} inconsistent with K {self.K} hidden {self.hidden}"
        self.data_base = 8 + hlen
        self.table_offset = self.data_base + info["data_offsets"][0]
        cinfo = header[resolve(".col_scales")]
        assert cinfo["dtype"] == "F16", f"{path}: col_scales dtype {cinfo['dtype']!r} != F16"
        with open(path, "rb") as f:
            f.seek(self.data_base + cinfo["data_offsets"][0])
            raw = f.read(cinfo["data_offsets"][1] - cinfo["data_offsets"][0])
        self.col_scales = torch.frombuffer(bytearray(raw), dtype = torch.float16).view(*cinfo["shape"])
        assert self.col_scales.shape == (self.hidden,)
        self.fd = os.open(path, os.O_RDONLY)

    def words_per_row(self) -> int:
        return words_per_row(self.hidden, self.K, self.G)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def read_rows_packed(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: 1-D global row indices (CPU). Returns (N, row_words) int16 (CPU).
        Consecutive indices are coalesced into single preads."""
        idx = indices.cpu().to(torch.int64)
        assert bool((idx >= 0).all()) and bool((idx < self.num_rows).all()), \
            f"{self.path}: row index out of range [0, {self.num_rows}) " \
            f"(min {int(idx.min())}, max {int(idx.max())}): reading unvalidated indices " \
            f"would silently return other bytes as a packed row"
        row_bytes = self.row_words * 2
        buf = bytearray(idx.numel() * row_bytes)
        mv = memoryview(buf)
        iv = idx.tolist()
        n = len(iv)
        i = 0
        while i < n:
            j = i + 1
            while j < n and iv[j] == iv[j - 1] + 1:
                j += 1
            data = os.pread(self.fd, (j - i) * row_bytes, self.table_offset + iv[i] * row_bytes)
            mv[i * row_bytes : j * row_bytes] = data
            i = j
        return torch.frombuffer(buf, dtype = torch.int16).view(idx.numel(), self.row_words)

    def dequant(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather + decode rows via the bit-exact reference codec: (N, hidden) fp32."""
        idx = indices.cpu().to(torch.int64)
        packed = self.read_rows_packed(idx)
        return dequant_rows_transformed(packed, self.col_scales, self.K, self.seed,
                                        self.hidden, self.G, row_ids = idx)