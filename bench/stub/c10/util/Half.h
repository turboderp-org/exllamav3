// Minimal c10::Half stub for building moe_mul1.cpp without torch (bench only)
#pragma once
#include <cstdint>

namespace c10 {

struct Half {
    uint16_t x;

    struct from_bits_t {};
    static constexpr from_bits_t from_bits() { return from_bits_t(); }
    constexpr Half() : x(0) {}
    constexpr Half(uint16_t bits, from_bits_t) : x(bits) {}

    explicit operator float() const {
        // fp16 -> fp32
        const uint32_t sign = uint32_t(x & 0x8000u) << 16;
        const uint32_t exp  = (x >> 10) & 0x1fu;
        const uint32_t man  = x & 0x3ffu;
        uint32_t bits;
        if (exp == 0) {
            if (man == 0) bits = sign;
            else {
                uint32_t e = 127 - 15 + 1, m = man;
                while (!(m & 0x400)) { m <<= 1; --e; }
                bits = sign | (e << 23) | ((m & 0x3ff) << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7f800000u | (man << 13);
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
        float f;
        __builtin_memcpy(&f, &bits, 4);
        return f;
    }
};

} // namespace c10