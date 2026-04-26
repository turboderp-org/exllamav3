#include <immintrin.h>
#include "all_reduce_cpu_avx512.h"
#include "all_reduce_cpu_avx2.h"
#include "../avx512_target.h"
#include "../util.h"

#ifndef __linux__
#include <intrin.h>
#endif

// Add 32 BF16 values from A and B, store to A.
// Uses vpmovdw (AVX-512BW) to avoid packus + lane-fix permute.
AVX512_TARGET
static inline void do32_avx512(uint16_t* __restrict ap, const uint16_t* __restrict bp)
{
    // Load 32 BF16 values from A and B
    __m512i va16 = _mm512_loadu_si512((const __m512i*)ap);
    __m512i vb16 = _mm512_loadu_si512((const __m512i*)bp);

    // Split into low and high 256-bit halves for FP32 expansion
    __m256i a_lo16 = _mm512_castsi512_si256(va16);
    __m256i a_hi16 = _mm512_extracti64x4_epi64(va16, 1);
    __m256i b_lo16 = _mm512_castsi512_si256(vb16);
    __m256i b_hi16 = _mm512_extracti64x4_epi64(vb16, 1);

    // Expand BF16 to FP32 by shifting left 16 bits
    __m512i a_lo32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_lo16), 16);
    __m512i a_hi32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_hi16), 16);
    __m512i b_lo32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_lo16), 16);
    __m512i b_hi32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_hi16), 16);

    // Add in FP32
    __m512 s_lo = _mm512_add_ps(_mm512_castsi512_ps(a_lo32), _mm512_castsi512_ps(b_lo32));
    __m512 s_hi = _mm512_add_ps(_mm512_castsi512_ps(a_hi32), _mm512_castsi512_ps(b_hi32));

    // Truncate FP32 -> BF16 using vpmovdw (AVX-512BW)
    // srli extracts the upper 16 bits (BF16), cvtepi32_epi16 packs 16x32 -> 16x16
    __m256i out_lo = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(s_lo), 16));
    __m256i out_hi = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(s_hi), 16));

    // Combine into single 512-bit register and store
    __m512i result = _mm512_inserti64x4(_mm512_castsi256_si512(out_lo), out_hi, 1);
    _mm512_storeu_si512((__m512i*)ap, result);
}

// Fused copy+add: dst = src_a + src_b, reading from two source buffers.
// Avoids the separate memcpy + accumulate pattern for the first two contributions.
AVX512_TARGET
static inline void do32_avx512_fused(uint16_t* __restrict dst,
                                      const uint16_t* __restrict src_a,
                                      const uint16_t* __restrict src_b)
{
    __m512i va16 = _mm512_loadu_si512((const __m512i*)src_a);
    __m512i vb16 = _mm512_loadu_si512((const __m512i*)src_b);

    __m256i a_lo16 = _mm512_castsi512_si256(va16);
    __m256i a_hi16 = _mm512_extracti64x4_epi64(va16, 1);
    __m256i b_lo16 = _mm512_castsi512_si256(vb16);
    __m256i b_hi16 = _mm512_extracti64x4_epi64(vb16, 1);

    __m512i a_lo32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_lo16), 16);
    __m512i a_hi32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_hi16), 16);
    __m512i b_lo32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_lo16), 16);
    __m512i b_hi32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_hi16), 16);

    __m512 s_lo = _mm512_add_ps(_mm512_castsi512_ps(a_lo32), _mm512_castsi512_ps(b_lo32));
    __m512 s_hi = _mm512_add_ps(_mm512_castsi512_ps(a_hi32), _mm512_castsi512_ps(b_hi32));

    __m256i out_lo = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(s_lo), 16));
    __m256i out_hi = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(s_hi), 16));

    __m512i result = _mm512_inserti64x4(_mm512_castsi256_si512(out_lo), out_hi, 1);
    _mm512_storeu_si512((__m512i*)dst, result);
}

// A += B (BF16), in-place, round-toward-zero.
// Unrolled 4x (128 BF16 values = 256 bytes per iteration) with deep prefetch.
// Uses regular stores since dst may be re-read by subsequent device accumulations.
// Assumes count % 64 == 0.
AVX512_TARGET
void bf16_add_inplace_avx512
(
    uint16_t* __restrict a,
    const uint16_t* __restrict b,
    size_t count
)
{
    size_t i = 0;

    // Main loop: 128 BF16 values per iteration (4 x do32)
    for (; i + 128 <= count; i += 128)
    {
        // Prefetch 4 cache lines ahead (256 bytes = 128 BF16 values)
        _mm_prefetch((const char*)(a + i + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(a + i + 160), _MM_HINT_T0);
        _mm_prefetch((const char*)(b + i + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(b + i + 160), _MM_HINT_T0);

        do32_avx512(a + i,       b + i);
        do32_avx512(a + i + 32,  b + i + 32);
        do32_avx512(a + i + 64,  b + i + 64);
        do32_avx512(a + i + 96,  b + i + 96);
    }

    // Remainder: 64 BF16 values
    for (; i < count; i += 64)
    {
        do32_avx512(a + i,      b + i);
        do32_avx512(a + i + 32, b + i + 32);
    }
}

// dst = src_a + src_b (BF16), fused two-source add.
// Replaces the memcpy + accumulate pattern when exactly two contributions
// arrive before a third. Assumes count % 64 == 0.
AVX512_TARGET
static void bf16_add_twosrc_avx512
(
    uint16_t* __restrict dst,
    const uint16_t* __restrict src_a,
    const uint16_t* __restrict src_b,
    size_t count
)
{
    size_t i = 0;

    for (; i + 128 <= count; i += 128)
    {
        _mm_prefetch((const char*)(src_a + i + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(src_a + i + 160), _MM_HINT_T0);
        _mm_prefetch((const char*)(src_b + i + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(src_b + i + 160), _MM_HINT_T0);

        do32_avx512_fused(dst + i,       src_a + i,       src_b + i);
        do32_avx512_fused(dst + i + 32,  src_a + i + 32,  src_b + i + 32);
        do32_avx512_fused(dst + i + 64,  src_a + i + 64,  src_b + i + 64);
        do32_avx512_fused(dst + i + 96,  src_a + i + 96,  src_b + i + 96);
    }

    for (; i < count; i += 64)
    {
        do32_avx512_fused(dst + i,      src_a + i,      src_b + i);
        do32_avx512_fused(dst + i + 32, src_a + i + 32, src_b + i + 32);
    }
}

// Fast path enable for AVX-512
AVX512_TARGET
void enable_fast_fp_avx512()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

// Perform reduction using AVX-512 - original MAX_DEVICES-based indexing
AVX512_TARGET
void perform_cpu_reduce_avx512
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
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

    // Sync
    atomic_ref<uint32_t> stage_(&ctx->cpusum_stage_cpu);
    uint32_t stage = stage_.load_acquire();
    uint32_t next_stage = (stage + 1u) & 0x7fffffffu;

    // Timeout
    const auto start = std::chrono::high_resolution_clock::now();

    while (num_chunks)
    {
        size_t stage_size = MIN(rem_data_size, CPUREDUCE_CHUNK_SIZE);
        rem_data_size = MAX(rem_data_size - stage_size, 0);

        uint32_t rem_devices = device_mask;
        uint8_t* first_src = nullptr;  // Track first contribution for fused add
        int timeout_spin = 0;

        while (true)
        {
            for (int device = 0; device < MAX_DEVICES; ++device)
            {
                if (!(rem_devices & (1 << device))) continue;

                atomic_ref<uint32_t> device_stage_(&ctx->cpusum_stage_device[device * REDUCE_STAGE_STRIDE]);
                uint32_t device_stage = device_stage_.load_acquire();
                uint32_t no_contrib = device_stage & 0x80000000u;
                device_stage &= 0x7fffffffu;

                if (device_stage != stage)
                {
                    rem_devices &= ~(1 << device);

                    if (no_contrib == 0)
                    {
                        uint8_t* src = host_ptr(device, stage);
                        uint8_t* dst = host_ptr(MAX_DEVICES, stage);
                        size_t elem_count = CEIL_DIVIDE(stage_size, 128) * 64;

                        if (first_src == nullptr)
                        {
                            // First contribution: just remember where it is
                            first_src = src;
                        }
                        else if (first_src != dst)
                        {
                            // Second contribution: fused add of first + second -> dst
                            bf16_add_twosrc_avx512(
                                (uint16_t*) dst,
                                (uint16_t*) first_src,
                                (uint16_t*) src,
                                elem_count
                            );
                            first_src = dst;  // dst now holds accumulated data
                        }
                        else
                        {
                            // Third+ contribution: accumulate into dst
                            bf16_add_inplace_avx512(
                                (uint16_t*) dst,
                                (uint16_t*) src,
                                elem_count
                            );
                        }
                    }
                }
            }
            if (!rem_devices)
                break;

            _mm_pause();

            timeout_spin++;
            if (timeout_spin > 10000)
            {
                timeout_spin = 0;
                const auto now = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double, std::milli> elapsed = now - start;
                if (elapsed > std::chrono::duration<double, std::milli>(6000.0))
                {
                    printf(" ## CPU reduce process timeout\n");
                    TORCH_CHECK(false, "CPU reduce process timeout");
                }
            }
        }

        // Handle case where only one device contributed (no add needed, just copy)
        if (first_src != nullptr && first_src != host_ptr(MAX_DEVICES, stage))
        {
            memcpy(host_ptr(MAX_DEVICES, stage), first_src, stage_size);
        }

        stage = next_stage;
        next_stage = (stage + 1u) & 0x7fffffffu;
        stage_.store_release(stage);

        num_chunks--;
    }
}
