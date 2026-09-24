#pragma once

#ifndef __linux__
    #include <intrin.h>
#endif

bool is_avx512_supported();

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))  // EXL3_AARCH64_STUB
    #define AVX512_TARGET __attribute__((target("avx512f,avx512bw")))
    #define AVX512_TARGET_OPTIONAL __attribute__((target_clones("avx512f,avx512bw","default")))
#else
    #define AVX512_TARGET
    #define AVX512_TARGET_OPTIONAL
#endif
