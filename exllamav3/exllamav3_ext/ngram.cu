#include <cuda_fp16.h>
#include "ngram.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <algorithm>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "util.h"
#include "util.cuh"

namespace py = pybind11;

/*

Embedding table fast path (Qwen3.8-Flash-Next PLE table, quantized token embeddings): the hot
loop stays on the CPU where the token ids already live (no device round trips), the scattered
table rows are gathered with threaded preads into a pinned staging buffer, and the trellis rows
are decoded by a memory-bound GPU kernel after one non-blocking H2D copy.

  ngram_hash_cpu:   eos-segmented n-gram hashing + global dedup: emits sorted unique table row
                    ids, the inverse map for the output gather, and each unique row's hash head.
                    Exact port of the torch reference (NGramEmbedding.compute_ngram_ids +
                    torch.unique), which the HF parity test pins down bitwise.
  ngram_gather_cpu: threaded pread of the unique rows (run-coalesced: unique ids are sorted, so
                    adjacent rows merge into single reads) into a caller buffer. This file holds
                    the Linux version; ngram_gather_win.cpp has the overlapped-ReadFile port.
  ngram_dequant:    one block per ring (an n-gram row, or one group of a token embedding row):
                    unpack the tail-biting ring bitstream, decode the mul1 codebook value per
                    element and apply the fp16 scale, then either add the per-head bias or
                    rotate the ring back and apply its sign vector. Matches dequant_rows() in
                    ngram_codec.py (fp16 codebook rounding included).

*/

#define MAX_RING_DIM 256
#define MUL1 0x83DCD12Du

int64_t ngram_hash_cpu
(
    const at::Tensor& ids,           // (bsz, ctx + seq) int64 CPU
    int64_t seq_len,
    const at::Tensor& multipliers,   // (ngram_size) int64 CPU
    const at::Tensor& offsets,       // (num_heads) int64 CPU
    const at::Tensor& sizes,         // (num_heads) int64 CPU
    int64_t heads_per_ngram,
    int64_t eos_token,
    at::Tensor uids,                 // (cap) int64 CPU out, cap >= bsz * seq * num_heads
    at::Tensor inverse,              // (bsz * seq * num_heads) int64 CPU out
    at::Tensor heads                 // (cap) int32 CPU out
)
{
    TORCH_CHECK(ids.device().is_cpu() && ids.dtype() == at::kLong && ids.is_contiguous(),
                "ngram_hash_cpu: ids must be contiguous int64 CPU");
    TORCH_CHECK(ids.dim() == 2, "ngram_hash_cpu: ids must be (bsz, ctx + seq)");
    // Host code reads and writes these through raw pointers
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*> { &multipliers, &offsets, &sizes, &uids, &inverse })
        TORCH_CHECK(t->device().is_cpu() && t->dtype() == at::kLong && t->is_contiguous(),
                    "ngram_hash_cpu: multipliers, offsets, sizes, uids and inverse must be contiguous int64 CPU");
    TORCH_CHECK(heads.device().is_cpu() && heads.dtype() == at::kInt && heads.is_contiguous(),
                "ngram_hash_cpu: heads must be contiguous int32 CPU");
    int64_t bsz = ids.size(0);
    int64_t T = ids.size(1);
    int64_t ctx = T - seq_len;
    int64_t ngram_size = multipliers.numel();
    int64_t H = offsets.numel();
    TORCH_CHECK(seq_len >= 0 && ctx >= 0 && H == (ngram_size - 1) * heads_per_ngram, "ngram_hash_cpu: dims");
    TORCH_CHECK(sizes.numel() == H, "ngram_hash_cpu: sizes must have one entry per head");
    int64_t n = bsz * seq_len * H;
    TORCH_CHECK(uids.numel() >= n && inverse.numel() >= n && heads.numel() >= n,
                "ngram_hash_cpu: output buffers too small");

    const int64_t* idp = (const int64_t*) ids.data_ptr();
    const int64_t* mult = (const int64_t*) multipliers.data_ptr();
    const int64_t* offs = (const int64_t*) offsets.data_ptr();
    const int64_t* szs = (const int64_t*) sizes.data_ptr();
    int64_t* uids_p = (int64_t*) uids.data_ptr();
    int64_t* inv_p = (int64_t*) inverse.data_ptr();
    int32_t* heads_p = (int32_t*) heads.data_ptr();
    for (int64_t h = 0; h < H; ++h)
        TORCH_CHECK(szs[h] > 0, "ngram_hash_cpu: empty hash table for head ", h);
    if (!n) return 0;

    py::gil_scoped_release release;

    struct HK { int64_t hash; int64_t idx; };
    std::vector<HK> hk((size_t) n);

    for (int64_t b = 0; b < bsz; ++b)
    {
        const int64_t* row = idp + b * T;
        int64_t prev_eos = -1;   // last eos position strictly before p
        for (int64_t p = 0; p < T; ++p)
        {
            if (p > 0 && row[p - 1] == eos_token) prev_eos = p - 1;
            if (p >= ctx)
            {
                int64_t seg_start = prev_eos + 1;
                uint64_t mixed = (uint64_t) row[p] * (uint64_t) mult[0];
                int64_t* out = inv_p + (b * seq_len + (p - ctx)) * H;
                for (int64_t s = 1; s < ngram_size; ++s)
                {
                    int64_t src = (p - s >= seg_start && p - s >= 0) ? row[p - s] : eos_token;
                    mixed ^= (uint64_t) src * (uint64_t) mult[s];
                    int64_t lo = (s - 1) * heads_per_ngram;
                    for (int64_t h = lo; h < lo + heads_per_ngram; ++h)
                    {
                        int64_t m = (int64_t) mixed % szs[h];
                        if (m < 0) m += szs[h];
                        int64_t idx = (b * seq_len + (p - ctx)) * H + h;
                        hk[(size_t) idx] = { m + offs[h], idx };
                        (void) out;
                    }
                }
            }
        }
    }

    std::sort(hk.begin(), hk.end(), [](const HK& a, const HK& b) { return a.hash < b.hash; });

    int64_t u = -1;
    int64_t last = -1;
    for (int64_t i = 0; i < n; ++i)
    {
        if (u < 0 || hk[(size_t) i].hash != last)
        {
            last = hk[(size_t) i].hash;
            uids_p[++u] = last;
            // hash head: the offsets are ascending starts of disjoint per-head ranges
            int64_t h = (int64_t) (std::upper_bound(offs, offs + H, last) - offs) - 1;
            heads_p[u] = (int32_t) h;
        }
        inv_p[hk[(size_t) i].idx] = u;
    }
    return u + 1;
}

// Persistent gather pool. Decode-time gathers are a handful of scattered rows whose COLD
// latency dominates: they must all be in flight concurrently (a sequential loop pays one full
// disk latency per row -- measured 16 x ~78us = 1.25ms/token cold). Spawning threads per call
// would cost more than the reads, so lazily started workers pull (offset, dst, bytes) tasks
// from a queue: one task per coalesced run for small gathers, chunked run groups for prefill-
// sized ones. The pool object is intentionally leaked (never destructed) -- joining or
// destroying a static condvar at process exit deadlocks (see the stloader shutdown notes).

#ifndef _WIN32

namespace
{

struct GatherCtx
{
    int fd;
    int64_t base_offset;
    int64_t row_bytes;
    int64_t uid_base;
    const int64_t* up;
    uint8_t* op;
    std::atomic<bool> failed { false };
};

struct GatherPool
{
    std::mutex mx;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    GatherCtx* ctx = nullptr;
    std::vector<std::pair<int64_t, int64_t>> tasks;   // uid-index spans [i0, i1)
    size_t next = 0;
    size_t pending = 0;
    bool started = false;
    std::mutex call_mx;

    static void read_span(GatherCtx* c, int64_t i0, int64_t i1)
    {
        int64_t i = i0;
        while (i < i1)
        {
            // coalesce a run of consecutive rows into one read
            int64_t j = i + 1;
            while (j < i1 && c->up[j] == c->up[j - 1] + 1) ++j;
            int64_t off = c->base_offset + (c->up[i] - c->uid_base) * c->row_bytes;
            int64_t nbytes = (j - i) * c->row_bytes;
            int64_t pos = 0;
            while (pos < nbytes)
            {
                ssize_t got = pread(c->fd, c->op + i * c->row_bytes + pos,
                                    (size_t) (nbytes - pos), (off_t) (off + pos));
                if (got <= 0) { c->failed.store(true); return; }
                pos += got;
            }
            i = j;
        }
    }

    void worker()
    {
        std::unique_lock<std::mutex> lk(mx);
        while (true)
        {
            cv_work.wait(lk, [&] { return next < tasks.size(); });
            auto span = tasks[next++];
            GatherCtx* c = ctx;
            lk.unlock();
            read_span(c, span.first, span.second);
            lk.lock();
            if (--pending == 0)
                cv_done.notify_all();
        }
    }

    void run(GatherCtx* c, std::vector<std::pair<int64_t, int64_t>>& new_tasks)
    {
        std::lock_guard<std::mutex> call_lock(call_mx);
        std::unique_lock<std::mutex> lk(mx);
        if (!started)
        {
            started = true;
            int n = (int) std::min(32u, std::max(4u, std::thread::hardware_concurrency()));
            for (int i = 0; i < n; ++i)
                std::thread(&GatherPool::worker, this).detach();
        }
        ctx = c;
        tasks.swap(new_tasks);
        next = 0;
        pending = tasks.size();
        cv_work.notify_all();
        // The caller drains the queue too: page-cache-warm gathers (~us per read) then finish
        // in-line instead of stalling ~100us on worker wake/sleep latency, while cold gathers
        // still fan out across whichever workers wake in time
        while (next < tasks.size())
        {
            auto span = tasks[next++];
            lk.unlock();
            read_span(c, span.first, span.second);
            lk.lock();
            if (--pending == 0)
                cv_done.notify_all();
        }
        cv_done.wait(lk, [&] { return pending == 0; });
        tasks.clear();
        ctx = nullptr;
    }
};

GatherPool* gather_pool()
{
    static GatherPool* pool = new GatherPool();   // leaked: see note above
    return pool;
}

}  // namespace

void ngram_gather_cpu
(
    int64_t fd,
    int64_t base_offset,
    int64_t row_bytes,
    const at::Tensor& uids,          // (U) int64 CPU, sorted ascending
    int64_t uid_base,                // subtracted from uids (shard-local row index)
    at::Tensor out                   // (U, row_bytes / itemsize) CPU, contiguous
)
{
    TORCH_CHECK(uids.device().is_cpu() && uids.dtype() == at::kLong && uids.is_contiguous(),
                "ngram_gather_cpu: uids must be contiguous int64 CPU");
    TORCH_CHECK(out.device().is_cpu() && out.is_contiguous(), "ngram_gather_cpu: bad out");
    int64_t U = uids.numel();
    TORCH_CHECK(out.size(0) >= U && out.size(1) * out.element_size() == row_bytes,
                "ngram_gather_cpu: out shape");
    if (!U || !row_bytes) return;
    const int64_t* up = (const int64_t*) uids.data_ptr();
    uint8_t* op = (uint8_t*) out.data_ptr();

    py::gil_scoped_release release;

    GatherCtx ctx { (int) fd, base_offset, row_bytes, uid_base, up, op };

    // Coalesce consecutive rows into runs
    std::vector<std::pair<int64_t, int64_t>> runs;   // (first index, count)
    runs.reserve(64);
    for (int64_t i = 0; i < U; )
    {
        int64_t j = i + 1;
        while (j < U && up[j] == up[j - 1] + 1) ++j;
        runs.emplace_back(i, j - i);
        i = j;
    }
    size_t n_runs = runs.size();

    if (n_runs == 1)
    {
        // single contiguous read: no pool round trip
        GatherPool::read_span(&ctx, 0, U);
        TORCH_CHECK(!ctx.failed.load(), "ngram_gather_cpu: short read");
        return;
    }

    // One span task per run keeps every cold read in flight at once; past ~64 runs, group
    // whole runs into span tasks so the queue overhead stays negligible (a worker issues its
    // group's runs sequentially -- pool-width reads still in flight)
    std::vector<std::pair<int64_t, int64_t>> tasks;
    size_t group = (n_runs + 63) / 64;
    tasks.reserve((n_runs + group - 1) / group);
    for (size_t r = 0; r < n_runs; r += group)
    {
        size_t r1 = std::min(n_runs, r + group);
        tasks.emplace_back(runs[r].first, runs[r1 - 1].first + runs[r1 - 1].second);
    }
    gather_pool()->run(&ctx, tasks);
    TORCH_CHECK(!ctx.failed.load(), "ngram_gather_cpu: short read");
}

#endif  // ^ Linux (pread); the Windows version (overlapped ReadFile) is in ngram_gather_win.cpp

// The 16-bit state of position i is the (q-1) preceding K-bit symbols below its own in the
// ring: a window of q*K <= 21 bits starting at stream bit (((i - (q - 1)) mod dim) * K),
// symbols read low-to-high = positions i-(q-1) ... i, stacked into the state with position i
// in the LOW bits. Word 0 holds the scale; the window spans at most three stream words
// (two words carry 32 - off bits, which covers 21 bits only when off <= 11).
__device__ __forceinline__ float ring_decode
(
    const uint16_t* __restrict__ row,
    int i,
    int dim,
    int K,
    int words,
    float scale
)
{
    int q = (15 + K) / K;
    int p = i - (q - 1);
    if (p < 0) p += dim;
    int start = p * K;
    int off = start & 15;
    int w0 = 1 + (start >> 4);
    uint32_t win = (uint32_t) row[w0] >> off;
    if (q * K > 16 - off)
    {
        int w1 = w0 + 1;
        if (w1 >= words) w1 = 1;
        win |= (uint32_t) row[w1] << (16 - off);
        if (q * K > 32 - off)
        {
            int w2 = w1 + 1;
            if (w2 >= words) w2 = 1;
            win |= (uint32_t) row[w2] << (32 - off);
        }
    }
    uint32_t mask = (1u << K) - 1u;
    uint32_t state = 0;
    #pragma unroll 16
    for (int t = 0; t < q; ++t)
        state |= ((win >> ((q - 1 - t) * K)) & mask) << (t * K);
    state &= 0xFFFFu;

    // mul1 codebook value, rounded to fp16 like the cached table
    uint32_t prod = state * MUL1;
    float h = 1024.0f + (float) ((prod & 0xff) + ((prod >> 8) & 0xff) +
                                 ((prod >> 16) & 0xff) + ((prod >> 24) & 0xff));
    float k_inv = __half2float(__ushort_as_half(0x1eee));
    float k_bias = __half2float(__ushort_as_half(0xc931));
    return __half2float(__float2half_rn(h * k_inv + k_bias)) * scale;
}

// Decode four consecutive elements of a K == 8 ring. K is not an argument: the host dispatch
// (ngram_dequant) launches both fast kernels only under its K == 8 gate, the single source of
// truth, so a wrong K cannot reach this decoder. ring_decode stays the general path for other K.
__device__ __forceinline__ void ring_decode4
(
    const uint16_t* __restrict__ row,
    int i0,               // first of four consecutive elements, i0 % 4 == 0
    int dim,
    int words,
    float scale,
    float out4[4]
)
{
    // Element i's state is stream bytes (i-1, i) = row bytes (i+1, i+2), so with
    // w0 = row bytes (i0, i0+1), w1 = (i0+2, i0+3), w2 = (i0+4, i0+5) the four
    // (sym_{i-1}, sym_i) pairs are w0hi|w1lo, w1, w1hi|w2lo, w2. Three aligned
    // u16 loads (row stride is 2 mod 4, so u32 loads are forbidden) replace the
    // two per state of the scalar decoder; the third is the row's final word at
    // i0 == dim - 4, in range because the dispatch checked words == 1 + dim * K / 16.
    const uint32_t w0 = row[i0 >> 1], w1 = row[(i0 >> 1) + 1], w2 = row[(i0 >> 1) + 2];
    uint32_t pair[4] = { (w0 >> 8) | (w1 << 8), w1, (w1 >> 8) | (w2 << 8), w2 };
    float k_inv = __half2float(__ushort_as_half(0x1eee));
    float k_bias = __half2float(__ushort_as_half(0xc931));
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        if (i0 == 0 && j == 0)
            out4[j] = ring_decode(row, 0, dim, 8, words, scale);      // element 0's pair wraps
        else
        {
            uint32_t p = pair[j] & 0xffffu;
            uint32_t state = ((p & 0xffu) << 8) | (p >> 8);   // sym_i in the low byte
            uint32_t prod = state * MUL1;
            float h = 1024.0f + (float) ((prod & 0xff) + ((prod >> 8) & 0xff) +
                                         ((prod >> 16) & 0xff) + ((prod >> 24) & 0xff));
            out4[j] = __half2float(__float2half_rn(h * k_inv + k_bias)) * scale;
        }
    }
}

// dim == 256 rotated rings (token embedding groups), K == 8: 64 lanes x float4.
// H-256 keeps the generic kernel's stage order and pairing (s = 1, 2, ..., 128,
// element with the bit set gets old[lo] - old[hi]): s = 1, 2 on the lane's four
// registers, s = 4..64 as component shuffles (distance s >> 2), s = 128 as one
// smem round trip, so the result is bit-exact with the generic kernel.
__global__ __launch_bounds__(64)
void ngram_dequant_rot4
(
    const int16_t* __restrict__ packed,
    const int32_t* __restrict__ heads,
    const half* __restrict__ aux,
    void* __restrict__ out,
    const int words,
    const int num_heads,
    const bool f32
)
{
    const int r = blockIdx.x;
    const int i = threadIdx.x;
    const uint16_t* row = (const uint16_t*) packed + (size_t) r * words;
    const float scale = __half2float(__ushort_as_half(row[0]));
    float v[4];
    ring_decode4(row, 4 * i, 256, words, scale, v);

    float a[4] = { 0.0f };
    if (aux)
    {
        // issue #525 bound as in the generic kernel
        int head = heads ? heads[r] : r % num_heads;
        head = max(0, min(head, num_heads - 1));
        const half* ar = aux + (size_t) head * 256;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            a[j] = __half2float(ar[4 * i + j]);
    }

    {   // s = 1
        float t0 = v[0] + v[1], t1 = v[0] - v[1];
        float t2 = v[2] + v[3], t3 = v[2] - v[3];
        v[0] = t0; v[1] = t1; v[2] = t2; v[3] = t3;
    }
    {   // s = 2
        float t0 = v[0] + v[2], t2 = v[0] - v[2];
        float t1 = v[1] + v[3], t3 = v[1] - v[3];
        v[0] = t0; v[1] = t1; v[2] = t2; v[3] = t3;
    }
    #pragma unroll
    for (int d = 1; d <= 16; d <<= 1)     // s = 4, 8, 16, 32, 64
    {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float o = __shfl_xor_sync(0xffffffffu, v[j], d);
            v[j] = (i & d) ? o - v[j] : v[j] + o;
        }
    }
    __shared__ float4 sf[64];             // s = 128: across the two warps
    sf[i] = make_float4(v[0], v[1], v[2], v[3]);
    __syncthreads();
    const float4 p = sf[i ^ 32];
    if (i < 32)
    {
        v[0] += p.x; v[1] += p.y; v[2] += p.z; v[3] += p.w;
    }
    else
    {
        v[0] = p.x - v[0]; v[1] = p.y - v[1]; v[2] = p.z - v[2]; v[3] = p.w - v[3];
    }

    const float norm = 1.0f / sqrtf(256.0f);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        v[j] = aux ? v[j] * norm * a[j] : v[j] * norm;

    if (f32)
        ((float4*) out)[(size_t) r * 64 + i] = make_float4(v[0], v[1], v[2], v[3]);
    else
    {
        union { uint2 u; half h[4]; } pk;
        pk.h[0] = __float2half_rn(v[0]); pk.h[1] = __float2half_rn(v[1]);
        pk.h[2] = __float2half_rn(v[2]); pk.h[3] = __float2half_rn(v[3]);
        ((uint2*) out)[(size_t) r * 64 + i] = pk.u;
    }
}

// dim == 160 non-rotated rings (PLE rows) with per-head bias, K == 8, fp16 out:
// grid-striding over U * 40 lanes of four consecutive elements. Same decode as
// the generic kernel, same bias add in fp32, so bit-exact with it.
__global__ __launch_bounds__(256)
void ngram_dequant_bias4
(
    const int16_t* __restrict__ packed,
    const int32_t* __restrict__ heads,
    const half* __restrict__ aux,
    half* __restrict__ out,
    const int U,
    const int words,
    const int num_heads
)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < U * 40; idx += gridDim.x * blockDim.x)
    {
        const int r = idx / 40;
        const int l = idx - r * 40;
        const int i0 = 4 * l;
        const uint16_t* row = (const uint16_t*) packed + (size_t) r * words;
        float v[4];
        ring_decode4(row, i0, 160, words, __half2float(__ushort_as_half(row[0])), v);

        float a[4] = { 0.0f };
        if (aux)
        {
            int head = heads ? heads[r] : r % num_heads;
            head = max(0, min(head, num_heads - 1));
            const half* ar = aux + (size_t) head * 160;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                a[j] = __half2float(ar[i0 + j]);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] += a[j];
        union { uint2 u; half h[4]; } pk;
        pk.h[0] = __float2half_rn(v[0]); pk.h[1] = __float2half_rn(v[1]);
        pk.h[2] = __float2half_rn(v[2]); pk.h[3] = __float2half_rn(v[3]);
        ((uint2*) out)[(size_t) r * 40 + l] = pk.u;
    }
}

__global__ __launch_bounds__(MAX_RING_DIM)
void ngram_dequant_kernel
(
    const int16_t* __restrict__ packed,   // (U, 1 + dim * K / 16)
    const int32_t* __restrict__ heads,    // (U) row of aux per ring, or null for ring index mod num_heads
    const half* __restrict__ aux,         // (num_heads, dim) bias, or signs when rotating; may be null
    void* __restrict__ out,               // (U, dim), half or float: element type must match f32
    const int K,
    const int words,                      // 1 + dim * K / 16
    const int num_heads,
    const bool rotate,
    const bool f32
)
{
    const int r = blockIdx.x;
    const int i = threadIdx.x;
    const int dim = blockDim.x;
    const uint16_t* row = (const uint16_t*) packed + (size_t) r * words;
    float v = ring_decode(row, i, dim, K, words, __half2float(__ushort_as_half(row[0])));

    float a = 0.0f;
    if (aux)
    {
        // The loader validates the bias row count against the head count; the bound here keeps a
        // bad head id from reading outside aux regardless (issue #525)
        int head = heads ? heads[r] : r % num_heads;
        head = max(0, min(head, num_heads - 1));
        a = __half2float(aux[(size_t) head * dim + i]);
    }

    if (rotate)
    {
        __shared__ float sf[MAX_RING_DIM];
        if (dim >= 32)
        {
            // Walsh-Hadamard, stage order s = 1, 2, 4, ..., dim/2: inside each warp the stages
            // up to s = 16 run on shuffles (no smem, no syncs); across the warps, one smem
            // round trip exchanges the warp results and a register butterfly per lane finishes
            // the remaining stages. Same op order and pairing as the torch reference, so the
            // result is bit-exact with it (and with the older smem butterfly below).
            const int lane = i & 31;
            #pragma unroll
            for (int s = 1; s < 32; s <<= 1)
            {
                float o = __shfl_xor_sync(0xffffffffu, v, s);
                v = (lane & s) ? o - v : v + o;
            }
            const int w = dim >> 5;
            if (w > 1)
            {
                sf[i] = v;
                __syncthreads();
                float w0 = sf[lane], w1 = sf[32 + lane], w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0, w7 = 0;
                if (w >= 4)
                {
                    w2 = sf[64 + lane];
                    w3 = sf[96 + lane];
                }
                if (w >= 8)
                {
                    w4 = sf[128 + lane];
                    w5 = sf[160 + lane];
                    w6 = sf[192 + lane];
                    w7 = sf[224 + lane];
                }
                { float t = w1; w1 = w0 - t; w0 = w0 + t; }
                if (w >= 4)
                {
                    { float t = w3; w3 = w2 - t; w2 = w2 + t; }
                    { float t = w5; w5 = w4 - t; w4 = w4 + t; }
                    { float t = w7; w7 = w6 - t; w6 = w6 + t; }
                    { float t = w2; w2 = w0 - t; w0 = w0 + t; }
                    { float t = w3; w3 = w1 - t; w1 = w1 + t; }
                    { float t = w6; w6 = w4 - t; w4 = w4 + t; }
                    { float t = w7; w7 = w5 - t; w5 = w5 + t; }
                    if (w >= 8)
                    {
                        { float t = w4; w4 = w0 - t; w0 = w0 + t; }
                        { float t = w5; w5 = w1 - t; w1 = w1 + t; }
                        { float t = w6; w6 = w2 - t; w2 = w2 + t; }
                        { float t = w7; w7 = w3 - t; w3 = w3 + t; }
                    }
                }
                const int warp = i >> 5;
                v = (w == 2) ? (warp ? w1 : w0)
                    : (warp == 0 ? w0 : warp == 1 ? w1 : warp == 2 ? w2 : warp == 3 ? w3 :
                       warp == 4 ? w4 : warp == 5 ? w5 : warp == 6 ? w6 : w7);
            }
        }
        else
        {
            // Rings below one warp (dim = 16) keep the smem butterfly
            sf[i] = v;
            __syncthreads();
            for (int s = 1; s < dim; s <<= 1)
            {
                float lo = sf[i & ~s];
                float hi = sf[i | s];
                __syncthreads();
                sf[i] = (i & s) ? lo - hi : lo + hi;
                __syncthreads();
            }
            v = sf[i];
        }
        v *= (1.0f / sqrtf((float) dim));
        if (aux) v *= a;
    }
    else v += a;

    if (f32)
        ((float*) out)[(size_t) r * dim + i] = v;
    else
        ((half*) out)[(size_t) r * dim + i] = __float2half_rn(v);
}

void ngram_dequant
(
    const at::Tensor& packed,                   // (U, 1 + dim * K / 16) int16 CUDA
    int64_t K,
    const c10::optional<at::Tensor>& heads,     // (U) int32 CUDA
    const c10::optional<at::Tensor>& aux,       // (num_heads, dim) half CUDA
    at::Tensor out,                             // (U, dim) half or float CUDA
    bool rotate
)
{
    const at::cuda::OptionalCUDAGuard device_guard(packed.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(packed, kShort);
    TORCH_CHECK(out.scalar_type() == at::kHalf || out.scalar_type() == at::kFloat, "ngram_dequant: out must be half or float");
    TORCH_CHECK(packed.is_contiguous() && out.is_contiguous(), "ngram_dequant: contiguous inputs required");
    int64_t U = packed.size(0);
    int words = (int) packed.size(1);
    int dim = (int) out.size(1);
    TORCH_CHECK(1 <= K && K <= 8 && words == 1 + dim * (int) K / 16, "ngram_dequant: bad K/words");
    TORCH_CHECK(dim <= MAX_RING_DIM && dim % 16 == 0 && out.size(0) >= U, "ngram_dequant: shapes");
    TORCH_CHECK(!rotate || (dim & (dim - 1)) == 0, "ngram_dequant: rotation needs a power of two");
    if (heads)
    {
        TORCH_CHECK_DTYPE((*heads), kInt);
        TORCH_CHECK(aux && heads->is_contiguous() && heads->numel() >= U, "ngram_dequant: heads");
    }
    if (aux)
    {
        TORCH_CHECK_DTYPE((*aux), kHalf);
        TORCH_CHECK(aux->is_contiguous() && aux->dim() == 2 && aux->size(1) == dim, "ngram_dequant: aux shape");
        TORCH_CHECK(!U || aux->size(0) > 0, "ngram_dequant: aux is empty");
    }
    if (!U || !dim) return;

    const bool f32 = out.scalar_type() == at::kFloat;
    const int16_t* packed_ptr = (const int16_t*) packed.data_ptr();
    const int32_t* heads_ptr = heads ? (const int32_t*) heads->data_ptr() : nullptr;
    const half* aux_ptr = aux ? (const half*) aux->data_ptr() : nullptr;
    const int heads_per_row = aux ? (int) aux->size(0) : 1;

    if (K == 8 && rotate && dim == 256)
        ngram_dequant_rot4<<<(unsigned int) U, 64, 0, stream>>>
        (
            packed_ptr, heads_ptr, aux_ptr, out.data_ptr(),
            words, heads_per_row, f32
        );
    else if (K == 8 && !rotate && !f32 && dim == 160)
        ngram_dequant_bias4<<<((unsigned int) U * 40 + 255) / 256, 256, 0, stream>>>
        (
            packed_ptr, heads_ptr, aux_ptr, (half*) out.data_ptr(),
            (int) U, words, heads_per_row
        );
    else
        ngram_dequant_kernel<<<(unsigned int) U, dim, 0, stream>>>
        (
            packed_ptr, heads_ptr, aux_ptr, out.data_ptr(),
            (int) K, words, heads_per_row, rotate, f32
        );
    cuda_check(cudaPeekAtLastError());
}
