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

Hashed n-gram embedding fast path (Qwen3.8-Flash-Next PLE table): the hot loop stays on the CPU
where the token ids already live (no device round trips), the scattered table rows are gathered
with threaded preads into a pinned staging buffer, and the trellis rows are decoded by a
memory-bound GPU kernel after one non-blocking H2D copy.

  ngram_hash_cpu:   eos-segmented n-gram hashing + global dedup: emits sorted unique table row
                    ids, the inverse map for the output gather, and each unique row's hash head.
                    Exact port of the torch reference (NGramEmbedding.compute_ngram_ids +
                    torch.unique), which the HF parity test pins down bitwise.
  ngram_gather_cpu: threaded pread of the unique rows (run-coalesced: unique ids are sorted, so
                    adjacent rows merge into single reads) into a caller buffer.
  ngram_dequant:    one block per row: unpack the tail-biting ring bitstream, decode the mul1
                    codebook value per element, apply the fp16 row scale and the per-head bias.
                    Matches dequant_rows() in ngram_codec.py (fp16 codebook rounding included).

*/

#define ROW_DIM 160
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
    int64_t bsz = ids.size(0);
    int64_t T = ids.size(1);
    int64_t ctx = T - seq_len;
    int64_t ngram_size = multipliers.numel();
    int64_t H = offsets.numel();
    TORCH_CHECK(ctx >= 0 && H == (ngram_size - 1) * heads_per_ngram, "ngram_hash_cpu: dims");
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
    if (!U) return;
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

#else

// No pread on Windows; the python side forces the RAM table modes there until a
// ReadFile/OVERLAPPED implementation exists
void ngram_gather_cpu
(
    int64_t fd,
    int64_t base_offset,
    int64_t row_bytes,
    const at::Tensor& uids,
    int64_t uid_base,
    at::Tensor out
)
{
    TORCH_CHECK(false, "ngram_gather_cpu: n-gram table streaming is not implemented on Windows");
}

#endif

__global__ __launch_bounds__(ROW_DIM)
void ngram_dequant_kernel
(
    const int16_t* __restrict__ packed,   // (U, 1 + ROW_DIM * K / 16)
    const int32_t* __restrict__ heads,    // (U)
    const half* __restrict__ bias,        // (num_heads, ROW_DIM)
    half* __restrict__ out,               // (U, ROW_DIM)
    const int K,
    const int words                       // 1 + ROW_DIM * K / 16
)
{
    const int r = blockIdx.x;
    const int i = threadIdx.x;

    extern __shared__ uint16_t sw[];
    if (i < words) sw[i] = (uint16_t) packed[(size_t) r * words + i];
    __syncthreads();

    float scale = __half2float(__ushort_as_half(sw[0]));

    // stream bit m of element i lives at ring position ((i - m / K) mod ROW_DIM) * K + m % K
    uint32_t state = 0;
    #pragma unroll 4
    for (int m = 0; m < 16; ++m)
    {
        int pos = i - m / K;
        if (pos < 0) pos += ROW_DIM;
        int sb = pos * K + m % K;
        uint32_t bit = (sw[1 + (sb >> 4)] >> (sb & 15)) & 1;
        state |= bit << m;
    }

    // mul1 codebook value, rounded to fp16 like the cached table
    uint32_t prod = state * MUL1;
    float h = 1024.0f + (float) ((prod & 0xff) + ((prod >> 8) & 0xff) +
                                 ((prod >> 16) & 0xff) + ((prod >> 24) & 0xff));
    float k_inv = __half2float(__ushort_as_half(0x1eee));
    float k_bias = __half2float(__ushort_as_half(0xc931));
    float cb = __half2float(__float2half_rn(h * k_inv + k_bias));

    float b = __half2float(bias[(size_t) heads[r] * ROW_DIM + i]);
    out[(size_t) r * ROW_DIM + i] = __float2half_rn(cb * scale + b);
}

void ngram_dequant
(
    const at::Tensor& packed,        // (U, 1 + ROW_DIM * K / 16) int16 CUDA
    int64_t K,
    const at::Tensor& heads,         // (U) int32 CUDA
    const at::Tensor& bias,          // (num_heads, ROW_DIM) half CUDA
    at::Tensor out                   // (U, ROW_DIM) half CUDA
)
{
    const at::cuda::OptionalCUDAGuard device_guard(packed.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(packed, kShort);
    TORCH_CHECK_DTYPE(heads, kInt);
    TORCH_CHECK_DTYPE(bias, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(packed.is_contiguous() && heads.is_contiguous() && bias.is_contiguous() &&
                out.is_contiguous(), "ngram_dequant: contiguous inputs required");
    int64_t U = packed.size(0);
    int words = (int) packed.size(1);
    TORCH_CHECK(1 <= K && K <= 8 && words == 1 + ROW_DIM * (int) K / 16, "ngram_dequant: bad K/words");
    TORCH_CHECK(out.size(0) >= U && out.size(1) == ROW_DIM && heads.numel() >= U,
                "ngram_dequant: shapes");
    TORCH_CHECK(bias.size(1) == ROW_DIM, "ngram_dequant: bias shape");
    if (!U) return;

    ngram_dequant_kernel<<<(unsigned int) U, ROW_DIM, words * sizeof(uint16_t), stream>>>
    (
        (const int16_t*) packed.data_ptr(),
        (const int32_t*) heads.data_ptr(),
        (const half*) bias.data_ptr(),
        (half*) out.data_ptr(),
        (int) K,
        words
    );
    cuda_check(cudaPeekAtLastError());
}
