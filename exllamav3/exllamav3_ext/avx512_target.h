#pragma once

#ifndef __linux__
    #include <intrin.h>
#endif

bool is_avx512_supported();

#ifdef __linux__
    #define AVX512_TARGET __attribute__((target("avx512f,avx512bw")))
    #define AVX512_TARGET_OPTIONAL __attribute__((target_clones("avx512f,avx512bw","default")))
#else
    #define AVX512_TARGET
    #define AVX512_TARGET_OPTIONAL
#endif
