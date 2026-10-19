#include "avx2_target.h"

bool is_avx2_supported()
{
    static bool avx2_check = false;
    static bool avx2_supported = false;
    if (avx2_check) return avx2_supported;
    #if !(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
        // no AVX on this architecture
    #elif defined(__linux__)
        avx2_supported = __builtin_cpu_supports("avx2");
    #else
        int cpuInfo[4];
        __cpuidex(cpuInfo, 7, 0);
        avx2_supported = (cpuInfo[1] & (1 << 5)) != 0;
    #endif
    avx2_check = true;
    // if (avx2_supported) printf("AVX2 supported\n");
    // else printf("AVX2 not supported\n");
    return avx2_supported;
}
bool is_avx_supported()
{
    static bool avx_check = false;
    static bool avx_supported = false;
    if (avx_check) return avx_supported;
    #if !(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
        // no AVX on this architecture
    #elif defined(__linux__)
        avx_supported = __builtin_cpu_supports("avx");
    #else
        // CPUID.1:ECX bit 28 (AVX) and bit 27 (OSXSAVE); the OS must also have enabled the YMM state
        int cpuInfo[4];
        __cpuidex(cpuInfo, 1, 0);
        avx_supported = (cpuInfo[2] & (1 << 28)) != 0 && (cpuInfo[2] & (1 << 27)) != 0
            && (_xgetbv(0) & 0x6) == 0x6;
    #endif
    avx_check = true;
    return avx_supported;
}

bool is_f16c_supported()
{
    static bool f16c_check = false;
    static bool f16c_supported = false;
    if (f16c_check) return f16c_supported;
    #if !(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
        // no AVX on this architecture
    #elif defined(__linux__)
        f16c_supported = __builtin_cpu_supports("f16c");
    #else
        int cpuInfo[4];
        __cpuidex(cpuInfo, 1, 0);
        f16c_supported = (cpuInfo[2] & (1 << 29)) != 0;
    #endif
    f16c_check = true;
    return f16c_supported;
}
