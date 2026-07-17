#include <immintrin.h>
#include "all_reduce_cpu_avx2.h"
#include "all_reduce_cpu_avx512.h"
#include "../avx2_target.h"
#include "../avx512_target.h"
#include "../util.h"
#include <thread>
#include <atomic>
#include <chrono>

#ifndef __linux__
#include <intrin.h>
#endif

// ---------------------------------------------------------------------------------------------
// Worker pool for sliced accumulates. Workers are spawned lazily in the CPU-reduce process,
// spin-park between generations (pause loop backing off to 50 us naps) and each executes its
// own task slot when the generation counter bumps. Only one reduce runs at a time, so a single
// generation/done pair suffices.

#define CPUREDUCE_MAX_WORKERS 3

struct SliceTask
{
    void (*fn3)(uint16_t*, const uint16_t*, const uint16_t*, size_t);
    void (*fn2)(uint16_t*, const uint16_t*, size_t);
    uint16_t* dst;
    const uint16_t* a;
    const uint16_t* b;
    size_t count;
};

static SliceTask g_slice_tasks[CPUREDUCE_MAX_WORKERS];
static std::atomic<uint64_t> g_slice_gen{0};
static std::atomic<uint64_t> g_slice_done{0};
static std::atomic<int> g_slice_workers{0};

static void slice_worker(int idx)
{
    enable_fast_fp();  // MXCSR is per-thread
    uint64_t seen = 0;
    int idle = 0;
    while (true)
    {
        uint64_t gen = g_slice_gen.load(std::memory_order_acquire);
        if (gen == seen)
        {
            if (++idle < 8192)
            {
                _mm_pause();
                continue;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        idle = 0;
        seen = gen;
        SliceTask t = g_slice_tasks[idx];
        if (t.count)
        {
            if (t.fn3) t.fn3(t.dst, t.a, t.b, t.count);
            else       t.fn2(t.dst, t.b, t.count);
        }
        g_slice_done.fetch_add(1, std::memory_order_release);
    }
}

void cpu_reduce_parallel
(
    void (*fn3)(uint16_t*, const uint16_t*, const uint16_t*, size_t),
    void (*fn2)(uint16_t*, const uint16_t*, size_t),
    uint16_t* dst,
    const uint16_t* a,
    const uint16_t* b,
    size_t count,
    int threads
)
{
    threads = MIN(threads, CPUREDUCE_MAX_WORKERS + 1);
    if (threads <= 1 || count < 8192)
    {
        if (fn3) fn3(dst, a, b, count);
        else     fn2(dst, b, count);
        return;
    }

    // Lazy spawn up to threads - 1 workers (pool only grows)
    int need = threads - 1;
    int cur = g_slice_workers.load(std::memory_order_relaxed);
    while (cur < need)
    {
        if (g_slice_workers.compare_exchange_strong(cur, cur + 1))
            std::thread(slice_worker, cur).detach();
        cur = g_slice_workers.load(std::memory_order_relaxed);
    }
    int nworkers = g_slice_workers.load(std::memory_order_relaxed);

    // 128-element-aligned slices; the add kernels need multiples of 64 and count is guaranteed
    // to be one (the last slice takes the remainder)
    size_t per = ((count / (size_t)threads) + 127) & ~(size_t)127;
    if (per == 0) per = count;
    uint64_t done0 = g_slice_done.load(std::memory_order_acquire);

    size_t off = MIN(per, count);   // slice 0 runs on this thread
    for (int w = 0; w < nworkers; ++w)
    {
        size_t n = off < count ? MIN(per, count - off) : 0;
        g_slice_tasks[w] = SliceTask{ fn3, fn2, dst + off, a ? a + off : nullptr, b + off, n };
        off += n;
    }
    g_slice_gen.fetch_add(1, std::memory_order_release);

    if (fn3) fn3(dst, a, b, MIN(per, count));
    else     fn2(dst, b, MIN(per, count));

    // Every worker acknowledges every generation (zero-count tasks included)
    while ((int64_t)(g_slice_done.load(std::memory_order_acquire) - done0) < nworkers)
        _mm_pause();
}

AVX2_TARGET
inline void do16(uint16_t* __restrict ap, const uint16_t* __restrict bp)
{
    // Load 16 BF16 from A and B
    __m256i va16 = _mm256_loadu_si256((const __m256i*)ap);
    __m256i vb16 = _mm256_loadu_si256((const __m256i*)bp);

    __m128i a_lo16 = _mm256_castsi256_si128(va16);       // elems 0..7
    __m128i a_hi16 = _mm256_extracti128_si256(va16, 1);  // elems 8..15
    __m128i b_lo16 = _mm256_castsi256_si128(vb16);
    __m128i b_hi16 = _mm256_extracti128_si256(vb16, 1);

    // Expand to FP32
    __m256i a_lo32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a_lo16), 16);
    __m256i a_hi32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a_hi16), 16);
    __m256i b_lo32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b_lo16), 16);
    __m256i b_hi32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b_hi16), 16);

    // Add in FP32
    __m256 s_lo = _mm256_add_ps(_mm256_castsi256_ps(a_lo32), _mm256_castsi256_ps(b_lo32));
    __m256 s_hi = _mm256_add_ps(_mm256_castsi256_ps(a_hi32), _mm256_castsi256_ps(b_hi32));

    // Round back to BF16 (nearest; plain truncation is biased toward zero)
    const __m256i rnd = _mm256_set1_epi32(0x8000);
    __m256i u_lo = _mm256_srli_epi32(_mm256_add_epi32(_mm256_castps_si256(s_lo), rnd), 16);
    __m256i u_hi = _mm256_srli_epi32(_mm256_add_epi32(_mm256_castps_si256(s_hi), rnd), 16);

    // Pack per-lane, then fix the lane ordering:
    __m256i packed = _mm256_packus_epi32(u_lo, u_hi);
    __m128i p_lo = _mm256_castsi256_si128(packed);
    __m128i p_hi = _mm256_extracti128_si256(packed, 1);
    __m128i q_lo = _mm_unpacklo_epi64(p_lo, p_hi); // [0..3, 4..7]
    __m128i q_hi = _mm_unpackhi_epi64(p_lo, p_hi); // [8..11, 12..15]
    __m256i out = _mm256_castsi128_si256(q_lo);
    out = _mm256_inserti128_si256(out, q_hi, 1);

    // Store
    _mm256_storeu_si256((__m256i*)ap, out);
}

// A += B (BF16), in-place, round-toward-zero. Assumes count % 32 == 0.
AVX2_TARGET
inline void bf16_add_inplace_avx2
(
    uint16_t* __restrict a,
    const uint16_t* __restrict b,
    size_t count
)
{
    for (size_t i = 0; i < count; i += 32)
    {
        do16(a + i, b + i);
        do16(a + i + 16, b + i + 16);
    }
}

// FP16 wire: widen with F16C, add in FP32, narrow with hardware round-to-nearest-even.
// ~2x the per-add cost of the bf16 integer path on Zen 4 (convert-port bound), still ~1 us
// per 16KB pair-add; used only for kHalf payloads where the wire is then exact
AVX2_F16C_TARGET
inline void do16_fp16(uint16_t* __restrict ap, const uint16_t* __restrict bp)
{
    __m256i va16 = _mm256_loadu_si256((const __m256i*)ap);
    __m256i vb16 = _mm256_loadu_si256((const __m256i*)bp);
    __m256 a_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(va16));
    __m256 a_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(va16, 1));
    __m256 b_lo = _mm256_cvtph_ps(_mm256_castsi256_si128(vb16));
    __m256 b_hi = _mm256_cvtph_ps(_mm256_extracti128_si256(vb16, 1));
    __m256 s_lo = _mm256_add_ps(a_lo, b_lo);
    __m256 s_hi = _mm256_add_ps(a_hi, b_hi);
    __m128i o_lo = _mm256_cvtps_ph(s_lo, _MM_FROUND_TO_NEAREST_INT);
    __m128i o_hi = _mm256_cvtps_ph(s_hi, _MM_FROUND_TO_NEAREST_INT);
    __m256i out = _mm256_castsi128_si256(o_lo);
    out = _mm256_inserti128_si256(out, o_hi, 1);
    _mm256_storeu_si256((__m256i*)ap, out);
}

// A += B (FP16), in-place. Assumes count % 32 == 0.
AVX2_F16C_TARGET
inline void fp16_add_inplace_avx2
(
    uint16_t* __restrict a,
    const uint16_t* __restrict b,
    size_t count
)
{
    for (size_t i = 0; i < count; i += 32)
    {
        do16_fp16(a + i, b + i);
        do16_fp16(a + i + 16, b + i + 16);
    }
}

void enable_fast_fp()
{
    if (is_avx512_supported())
        enable_fast_fp_avx512();
    else
        enable_fast_fp_avx2();
}

AVX2_TARGET
void enable_fast_fp_avx2()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

// Dispatch for the right target - prioritizes AVX-512 if available
void perform_cpu_reduce
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
    uint32_t wire_dtype,
    uint8_t* shbuf_ptr,
    size_t shbuf_size
)
{
    if (is_avx512_supported())
    {
        perform_cpu_reduce_avx512(ctx, data_size, device_mask, wire_dtype, shbuf_ptr, shbuf_size);
    }
    else
    {
        perform_cpu_reduce_avx2(ctx, data_size, device_mask, wire_dtype, shbuf_ptr, shbuf_size);
    }
}

// Perform reduction on current job using AVX2
AVX2_F16C_TARGET
void perform_cpu_reduce_avx2
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
    uint32_t wire_dtype,
    uint8_t* shbuf_ptr,
    size_t shbuf_size
)
{
    // Indexing - original MAX_DEVICES-based layout
    const uint32_t buf_slot_size = (shbuf_size / (MAX_DEVICES + 1) / 1024) * 1024;
    const uint32_t max_buf_stages = buf_slot_size / CPUREDUCE_CHUNK_SIZE;
    TORCH_CHECK(max_buf_stages >= 2, "Shared buffer too small for chunk size (need at least 2 stages)");
    auto host_ptr = [&] (int device, uint32_t stage_idx)
    {
        return shbuf_ptr + buf_slot_size * device + (stage_idx % max_buf_stages) * CPUREDUCE_CHUNK_SIZE;
    };

    int num_chunks = (int) CEIL_DIVIDE(data_size, CPUREDUCE_CHUNK_SIZE);
    size_t rem_data_size = data_size;
     // Single-chunk jobs come from the split kernels (per-device flags); multi-chunk jobs from
    // the striped bandwidth kernel (per-(device, block) flags). Must match the launch choice
    const bool multi = num_chunks > 1;

    // Sync
    atomic_ref<uint32_t> stage_(&ctx->cpusum_stage_cpu);
    uint32_t stage = stage_.load_acquire();
    uint32_t next_stage = (stage + 1u) & 0x7fffffffu;

    // Timeout
    const auto start = std::chrono::high_resolution_clock::now();

    int chunk_idx = 0;
    while (num_chunks)
    {
        // Stage 1: Participating devices are writing one chunk to their respective buffers and will
        // store/release their respective flags. Devices without a contribution to the sum will set
        // flags right away, with MSB set. First device to signal ready with a contribution is copied
        // to the output buffer. Subsequent contributions are added in the order they arrive. Proceed
        // to next stage only when all devices signal ready, include non-contributors since they will
        // still need to sync before receiving the sum.

        size_t stage_size = MIN(rem_data_size, CPUREDUCE_CHUNK_SIZE);
        rem_data_size = MAX(rem_data_size - stage_size, 0);

        // Accumulator ring throttle: the elastic kernel has no lockstep bounding the CPU's lead
        // over the recv blocks, so wait for the acc slot's previous tenant to be fully consumed
        if (multi && chunk_idx >= (int)max_buf_stages)
        {
            uint32_t recv_target = ((stage - max_buf_stages) & 0x7fffffffu) + 1u;
            int throttle_spin = 0;
            while (!cpusum_recv_ready(ctx, device_mask, recv_target))
            {
                _mm_pause();
                if (++throttle_spin > 10000)
                {
                    throttle_spin = 0;
                    const auto now = std::chrono::high_resolution_clock::now();
                    const std::chrono::duration<double, std::milli> elapsed = now - start;
                    if (elapsed > std::chrono::duration<double, std::milli>(45000.0))
                    {
                        printf(" ## CPU reduce process timeout (recv throttle)\n");
                        TORCH_CHECK(false, "CPU reduce process timeout");
                    }
                }
            }
        }

        uint32_t rem_devices = device_mask;
        bool first_contribution = true;
        int timeout_spin = 0;
        while (true)
        {
            for (int device = 0; device < MAX_DEVICES; ++device)
            {
                if (!(rem_devices & (1 << device))) continue;

                bool no_contrib;

                // Device is ready
                if (cpusum_device_arrived(ctx, device, stage, multi, &no_contrib))
                {
                    rem_devices &= ~(1 << device);

                    if (!no_contrib)
                    {
                        uint8_t* src = host_ptr(device, stage);
                        uint8_t* dst = host_ptr(MAX_DEVICES, stage);

                        // First contribution to this chunk: copy
                        if (first_contribution)
                        {
                            // Warm destination before first contribution
                            // #if defined(__GNUC__) || defined(__clang__)
                            //   __builtin_prefetch(dst, 1, 3);
                            // #endif

                            memcpy(dst, src, stage_size);
                            first_contribution = false;
                        }

                        // Subsequent contributions: accumulate
                        else
                        {
                            size_t elem_count = CEIL_DIVIDE(stage_size, 64) * 32;
                            if (wire_dtype == REDUCE_WIRE_FP16)
                                fp16_add_inplace_avx2((uint16_t*) dst, (uint16_t*) src, elem_count);
                            else
                                bf16_add_inplace_avx2((uint16_t*) dst, (uint16_t*) src, elem_count);
                        }
                    }
                }
            }
            if (!rem_devices)
                break;

            // Pause every iter
            _mm_pause();

            // Check timeout every 10k iter
            timeout_spin++;
            if (timeout_spin > 10000)
            {
                timeout_spin = 0;
                const auto now = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double, std::milli> elapsed = now - start;
                if (elapsed > std::chrono::duration<double, std::milli>(45000.0))
                {
                    printf(" ## CPU reduce process timeout\n");
                    TORCH_CHECK(false, "CPU reduce process timeout");
                }
            }
        }

        // Stage 2: Reduced sum is ready, publish and release. All devices are now waiting for the flag and will
        // begin reading back the sum for this stage and simultaneously transmit the next stage if any.
        stage = next_stage;
        next_stage = (stage + 1u) & 0x7fffffffu;
        stage_.store_release(stage);

        // We can exit here if we're done and start waiting for the next job. Devices will still be reading the last
        // chunk, but the host buffer is reserved for this operation and we have at most two active stages in the
        // buffer (and room for many more)
        num_chunks--;
        chunk_idx++;
    }
}
