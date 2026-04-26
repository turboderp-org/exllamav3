#include "avx512_target.h"

bool is_avx512_supported()
{
    static bool avx512_check = false;
    static bool avx512_supported = false;
    if (avx512_check) return avx512_supported;

#ifdef __linux__
    // Check for AVX-512F and AVX-512BW support
    avx512_supported = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    // Windows: use __cpuidex to check for AVX-512 support
    int cpuInfo[4];
    
    // Check if leaf 7 is supported
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] < 7)
    {
        avx512_check = true;
        return false;
    }
    
    // Get extended features (leaf 7, subleaf 0)
    __cpuidex(cpuInfo, 7, 0);
    
    // AVX-512F is bit 16 of EBX, AVX-512BW is bit 30 of EBX
    bool avx512f = (cpuInfo[1] & (1 << 16)) != 0;
    bool avx512bw = (cpuInfo[1] & (1 << 30)) != 0;
    avx512_supported = avx512f && avx512bw;
#endif

    avx512_check = true;
    // if (avx512_supported) printf("AVX-512 supported\n");
    // else printf("AVX-512 not supported\n");
    return avx512_supported;
}
