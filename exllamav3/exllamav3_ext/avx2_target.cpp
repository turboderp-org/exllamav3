#include "avx2_target.h"

bool is_avx2_supported()
{
    static bool avx2_check = false;
    static bool avx2_supported = false;
    if (avx2_check) return avx2_supported;
    #ifdef __linux__
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