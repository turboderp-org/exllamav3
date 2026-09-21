#pragma once
#include <c10/util/Exception.h>

// Bitrate K of an EXL3 tensor: an integer 1..8, or a half-integer 1.5 / 2.5 / 3.5 (mul1 codebook only). Python
// passes LinearEXL3.K (int or float); the C++ boundary takes it as float and decomposes it once here into the
// integer part and the half flag the kernel tables are indexed by
struct BitsK
{
    int bits;
    bool half;
};

inline BitsK bits_from_K(float K)
{
    const int bits = (int) K;
    const float f = K - (float) bits;
    TORCH_CHECK(bits >= 1 && bits <= 8 && (f == 0.0f || (f == 0.5f && bits <= 3)),
                "Unsupported EXL3 bitrate ", K, " (integer 1..8, or 1.5 / 2.5 / 3.5)");
    return { bits, f == 0.5f };
}

// Half-bit units (2 * bits + half) for runtime switches over mixed bitrates
inline int k2_from_K(float K)
{
    const BitsK b = bits_from_K(K);
    return 2 * b.bits + (b.half ? 1 : 0);
}
