#pragma once

#ifndef __linux__
    #include <intrin.h>
#endif

bool is_avx2_supported();

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef __linux__
        #define AVX2_TARGET __attribute__((target("avx2")))
        #define AVX2_TARGET_OPTIONAL __attribute__((target_clones("avx2","default")))
    #else
        #define AVX2_TARGET
        #define AVX2_TARGET_OPTIONAL
    #endif
#else
    #define AVX2_TARGET
    #define AVX2_TARGET_OPTIONAL
#endif
