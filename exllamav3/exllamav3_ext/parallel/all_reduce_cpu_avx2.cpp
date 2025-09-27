#include <immintrin.h>
#include "all_reduce_cpu_avx2.h"
#include "../avx2_target.h"
#include "../util.h"

#ifndef __linux__
#include <intrin.h>
#endif
#include <immintrin.h>

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

    // Truncate back to BF16
    __m256i u_lo = _mm256_srli_epi32(_mm256_castps_si256(s_lo), 16);
    __m256i u_hi = _mm256_srli_epi32(_mm256_castps_si256(s_hi), 16);

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

AVX2_TARGET
void enable_fast_fp()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

// A += B (BF16), in-place, round-toward-zero. Assumes count % 32 == 0. Not optimized, little-endian only
static inline void bf16_add_inplace_ref
(
    uint16_t* __restrict a,
    const uint16_t* __restrict b,
    size_t count
)
{
    for (int i = 0; i < count; ++i)
    {
        uint16_t s[2] = {0, 0};
        uint16_t d[2] = {0, 0};
        s[1] = *b++;
        d[1] = *a;
        float* fs = (float*) s;
        float* fd = (float*) d;
        *fd += *fs;
        *a++ = d[1];
    }
}

// Perform reduction on current job
AVX2_TARGET
void perform_cpu_reduce
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
    uint8_t* shbuf_ptr,
    size_t shbuf_size
)
{
    // Indexing
    const uint32_t buf_slot_size = (shbuf_size / (MAX_DEVICES + 1)) / 1024 * 1024;
    const uint32_t max_buf_stages = buf_slot_size / CPUREDUCE_CHUNK_SIZE;
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
        // Stage 1: Participating devices are writing one chunk to their respective buffers and will
        // store/release their respective flags. Devices without a contribution to the sum will set
        // flags right away, with MSB set. First device to signal ready with a contribution is copied
        // to the output buffer. Subsequent contributions are added in the order they arrive. Proceed
        // to next stage only when all devices signal ready, include non-contributors since they will
        // still need to sync before receiving the sum.

        size_t stage_size = MIN(rem_data_size, CPUREDUCE_CHUNK_SIZE);
        rem_data_size = MAX(rem_data_size - stage_size, 0);

        uint32_t rem_devices = device_mask;
        bool first_contribution = true;
        int timeout_spin = 0;
        while (true)
        {
            for (int device = 0; device < MAX_DEVICES; ++device)
            {
                if (!(rem_devices & (1 << device))) continue;

                // Prefetch the stage and buffer if possible
                // auto* device_stage_cookie_ptr = &ctx->cpusum_stage_device[device * REDUCE_STAGE_STRIDE];
                // _mm_prefetch((const char*) device_stage_cookie_ptr, _MM_HINT_T0);
                // #if defined(__GNUC__) || defined(__clang__)
                //     uint8_t* likely_src = host_ptr(device, stage);
                //    __builtin_prefetch(likely_src, 0, 1);
                // #endif
                // uint32_t device_stage = atomic_ref<uint32_t>(device_stage_cookie_ptr).load_acquire();

                atomic_ref<uint32_t> device_stage_(&ctx->cpusum_stage_device[device * REDUCE_STAGE_STRIDE]);
                uint32_t device_stage = device_stage_.load_acquire();
                uint32_t no_contrib = device_stage & 0x80000000u;
                device_stage &= 0x7fffffffu;

                // Device is ready
                if (device_stage != stage)
                {
                    rem_devices &= ~(1 << device);

                    if (no_contrib == 0)
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
                            bf16_add_inplace_avx2((uint16_t*) dst, (uint16_t*) src, CEIL_DIVIDE(stage_size, 64) * 32);
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
                if (elapsed > std::chrono::duration<double, std::milli>(6000.0))
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
    }
}