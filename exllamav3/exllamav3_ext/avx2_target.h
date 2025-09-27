#pragma once

#ifndef __linux__
    #include <intrin.h>
#endif

bool is_avx2_supported();

#ifdef __linux__
    #define AVX2_TARGET __attribute__((target("avx2")))
    #define AVX2_TARGET_OPTIONAL __attribute__((target_clones("avx2","default")))
#else
    #define AVX2_TARGET
    #define AVX2_TARGET_OPTIONAL
#endif