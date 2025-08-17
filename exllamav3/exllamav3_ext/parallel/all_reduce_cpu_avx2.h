#pragma once

#include <immintrin.h>

#if defined(_MSC_VER) && !defined(__clang__)
    #include <intrin.h>
#endif

// A += B (BF16), in-place, round-toward-zero. Assumes count % 32 == 0.
static inline void bf16_add_inplace_avx2
(
    uint16_t* __restrict a,
    const uint16_t* __restrict b,
    size_t count
)
{
    for (size_t i = 0; i < count; i += 32)
    {
        auto do16 = [&](uint16_t* __restrict ap, const uint16_t* __restrict bp)
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
        };

        do16(a + i, b + i);
        do16(a + i + 16, b + i + 16);
    }
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

inline void enable_fast_fp()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}