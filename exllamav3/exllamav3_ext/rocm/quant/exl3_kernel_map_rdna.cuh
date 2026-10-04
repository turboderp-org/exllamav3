#pragma once

// =============================================================================
// exl3_kernel_map.cuh for RDNA -- same structure, RDNA 3.5 tile shapes
// =============================================================================
//
// Generated from quant/exl3_kernel_map.cuh. Everything matches it except the
// shape table and SMEM_MAX below; the extern/instance macros keep its
// per-(K, cb) translation-unit layout so comp_units_rdna/ defines exactly the
// symbols quant/comp_units/exl3_comp_unit_N.cuh already declares.
//
// Why the shapes differ from the CUDA map
// ---------------------------------------
// RDNA 3.5 has 64 KB of LDS per workgroup; the CUDA map sizes tiles against a
// 90 KB budget (compute capability 8.6+). gfx1151 reports sharedMemPerBlock =
// 65536. exl3_gemm_inner.cuh static_asserts that the stages fit in SMEM_MAX, so
// the CUDA shapes are a compile-time failure here, not a silent slowdown.
//
// The retune:
//   - 256 threads everywhere, never 512   -- saves ~16 KB in sh_c
//   - TILESIZE_K pinned at 16, never 32   -- avoids doubling sh_a
//   - SH_STAGES cut to 4/3/3/3            -- fewer resident stages
//   - N capped at 384, not 512
// Narrow tiles win on RDNA for the same reason: there is no large cache to hide
// a bad tile choice behind.
//
// LDS per shape:
//   Shape 1 (16,16,128): 4-bit 25 KB, 6-bit 29 KB, 8-bit 33 KB
//   Shape 2 (16,16,192): 4-bit 30 KB, 6-bit 35 KB, 8-bit 40 KB
//   Shape 3 (16,16,256): 4-bit 35 KB, 6-bit 41 KB, 8-bit 47 KB
//   Shape 4 (16,16,384): 4-bit 45 KB, 6-bit 53 KB, 8-bit 61 KB
//
// SMEM_MAX is defined unconditionally at exl3_gemm_inner.cuh:7, so the RDNA
// value has to be established by the sibling chain that replaces it, not by a
// -D on the command line.

// -----------------------------------------------------------------------------
// EXL3_RDNA_SMEM_MAX -- compile-time LDS budget, and why it is not per-arch
// -----------------------------------------------------------------------------
// Strix Halo (gfx1151) reports sharedMemPerBlock = 65536. Other RDNA parts are
// reported to tolerate the CUDA 90 KB figure, so this cannot just be
// hardcoded to one number -- but it also cannot be selected with an arch macro,
// which is the obvious-looking fix and is wrong:
//
//   __gfx1151__ and friends are defined ONLY during the device pass. SMEM_MAX is
//   used in host code too (is_compat_shape, exl3_gemm_smem_bytes, the launch
//   sizing). An arch-conditional #if would silently give the host 90 KB and the
//   device 64 KB in the same build, so the host would admit shapes whose device
//   static_assert never ran at that size. That is worse than a wrong constant.
//
// So it is a build-time knob instead, set once for both passes:
//
//   hipcc -DEXL3_RDNA_SMEM_MAX=92160 ...      # 90 KB parts
//
// setup.py should derive it from the target arch list. The default is the
// conservative 64 KB, because a build that targets several archs must satisfy
// the smallest of them: the shape table and the kernel instantiations are shared
// across every arch in a fat binary.
//
// This is a ceiling, not a promise. The real device limit is queried at runtime
// by exl3_rdna_smem_budget() below and shape selection uses the smaller of the
// two, so an over-large compile-time value degrades to "some shapes unavailable"
// rather than to a failed launch.
#ifndef EXL3_RDNA_SMEM_MAX
#define EXL3_RDNA_SMEM_MAX (64 * 1024)   // RDNA 3.5 LDS per workgroup
#endif

#ifndef SMEM_MAX
#define SMEM_MAX EXL3_RDNA_SMEM_MAX
#endif

// Actual sharedMemPerBlock for `device`, queried once and cached. Pass -1 for
// the current device.
size_t exl3_rdna_device_smem_max(int device);

// MIN(SMEM_MAX, device limit) -- the budget shape selection must respect. Every
// admission test uses this rather than SMEM_MAX so that a binary built for a
// 90 KB part still behaves correctly on a 64 KB one.
size_t exl3_rdna_smem_budget(int device);

// -----------------------------------------------------------------------------
// EXL3_GEMM_SH_B_DQ_STRIDE -- single definition, four consumers
// -----------------------------------------------------------------------------
// Row stride, in halves, of the per-warp B staging tile that the inner kernel
// transposes dequantized fragments through on the way to WMMA layout. 18 rather
// than 17: at 17 the adjacent active-lane groups (0-3 vs 16-19, 8-11 vs 24-27)
// land on the same LDS banks. 18 halves = 9 dwords, and 9 is coprime with 32,
// so every bank is covered.
//
// This lives here because several places need to agree on it: the inner
// kernel's own indexing, exl3_gemm_smem_bytes() below, and the launch in
// quant/exl3_gemm.cu.
//
// A retyped literal (`16 * 17` left in exl3_gemm_smem_bytes() after the
// 17 -> 18 change) once made every cooperative launch under-allocate dynamic
// LDS by num_warps * 16 * 2 bytes and run off the end of the last warp's
// staging tile. Derive it, never retype it.
#define EXL3_GEMM_SH_B_DQ_STRIDE 18

// LDS bytes a (bitwidth, shape) pair actually needs. Defined in
// exl3_kernel_map_rdna.cu and mirrored from exl3_gemm_inner_rdna.cuh.
// quant/exl3_gemm.cu sizes its cooperative launches from this rather than from
// SMEM_MAX, which matters on a 64 KB part.
// Returns (size_t) -1 for a disabled shape.
size_t exl3_gemm_smem_bytes(int bits, int shape_idx, bool half_k = false);

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int bits, bool multi);
int exl3_gemm_num_kernel_shapes();
bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int bits, bool half_k = false);
// As in quant/exl3_kernel_map.cuh: dynamic LDS a (shape, bitrate) instantiation requests at launch, and a
// hard gate for forced shapes. On RDNA both are answered by exl3_gemm_smem_bytes(bits, shape_idx, half_k) below
// against exl3_rdna_smem_budget(), not by the CUDA 90 KB shape table
int exl3_gemm_shape_smem(int shape_idx, int bits, bool half_k);
void exl3_gemm_check_smem(int shape_idx, int bits, bool half_k, const char* who);

// bits: integer part of the bitrate; half_k: bitrate is bits + 0.5 (mul1 codebook only, 16 * bits + 8 uint16
// per tile). Same position as in the CUDA map so the comp_units' instance lists match
#define EXL3_GEMM_T_ARGS \
    const int bits, \
    const bool half_k, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

#define EXL3_GEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh

#define EXL3_MGEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t** __restrict__ B_list, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half** __restrict__ suh_list, \
    half* __restrict__ A_had, \
    const half** __restrict__ svh_list, \
    int64_t* B_indices, \
    half* B_weights, \
    const int bszm_in, \
    const int bszm_out, \
    const int min_index, \
    const int max_index, \
    const int num_tokens, \
    const int* __restrict__ size_n_list, \
    void** __restrict__ C_list, \
    const int* __restrict__ n_stride_list, \
    const int* __restrict__ had_src_list, \
    const int num_had_src

typedef void (*fp_exl3_gemm_kernel) (EXL3_GEMM_ARGS);
typedef void (*fp_exl3_mgemm_kernel) (EXL3_MGEMM_ARGS);

// Shape 2 is not 16,16,192. That is not a legal tile and would produce wrong
// results if it were ever selected:
//   - the CUDA kernel asserts TILESIZE_N % 128 == 0, and 192 % 128 = 64;
//   - FRAGS_N_PER_WARP = TILEBLOCKS_N / NUM_WARPS is integer division, and
//     192/16 = 12 blocks over 8 warps = 1, not 1.5. Four of the twelve N-blocks
//     are then never computed -- a third of the output tile silently missing.
// exl3_gemm_inner_rdna.cuh asserts both.
//
// Shape 4 is 512 rather than 384: at TILESIZE_K=16 with SH_STAGES=3 it fits
// 64 KB LDS (~62 KB at 8-bit, the worst case), which the CUDA 512 shape does
// not because it pairs 512 with TILESIZE_K=32. Set EXL3_RDNA_SHAPE4_N to 384 if
// the wide tile is not paying off, or to 0 to drop the instantiation entirely
// -- see the note on that macro below.
// -----------------------------------------------------------------------------
// Shape 4 width -- the one knob for compile cost
// -----------------------------------------------------------------------------
// The widest tile is by far the most expensive to instantiate: it is compiled
// once per (bitwidth, codebook, fp32/fp16, gemm/mgemm), and is the one most
// likely to exhaust build memory. Rather than deleting code, set this:
//
//   512  (default) widest tile, ~62 KB LDS at 8-bit -- test before trusting
//   384  narrower, ~48 KB LDS, cheaper to build
//   0    shape 4 is not instantiated at all; the slot becomes nullptr and the
//        selector skips it (occ_blocks_per_cu returns 0 for a null kernel)
//
// Any value must satisfy N % 128 == 0 and (N/16) % 8 == 0; gemm_inner asserts
// both, so a bad value is a build error rather than a wrong result.
#ifndef EXL3_RDNA_SHAPE4_N
#define EXL3_RDNA_SHAPE4_N 512
#endif

#define EXL3_GEMM_SHAPE_1     16,     16,    128,     4,     3
#define EXL3_GEMM_SHAPE_2     16,     16,    256,     3,     3
#define EXL3_GEMM_SHAPE_3     16,     16,    384,     3,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    EXL3_RDNA_SHAPE4_N,     3,     3

// DEVIATION FROM THE CUDA MAP: it has no SH_STAGES array, because nothing on
// the CUDA side needs to compute LDS usage at runtime. The RDNA shape selector
// does -- it rejects any shape whose LDS exceeds the 64 KB budget -- so the
// stage counts have to be readable from a table. Hardcoding them in the size
// calculation silently desyncs from EXL3_GEMM_SHAPE_n if the shapes are ever retuned.
// Keep this in step with the SH_STAGES field of each EXL3_GEMM_SHAPE_n above.
#define EXL3_GEMM_SH_STAGES   0, 4, 3, 3, 3

#define EXL3_GEMM_TILESIZE_K  0, 16, 16, 16, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 256, 384, EXL3_RDNA_SHAPE4_N
#define EXL3_GEMM_BLOCKDIM  0, 256, 256, 256, 256

#define EXL3_GEMM_NUM_SHAPES 4

// Shape 1 not currently used anywhere
// Shape 4 collapses to a null slot when EXL3_RDNA_SHAPE4_N is 0, so disabling
// it costs no instantiation at all rather than just narrowing the tile.
#if EXL3_RDNA_SHAPE4_N
    #define EXL3_GEMM_INST_S4(_bits, _c_fp32, cb)  exl3_gemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_4>
    #define EXL3_MGEMM_INST_S4(_bits, _c_fp32, cb) exl3_mgemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_4>
#else
    #define EXL3_GEMM_INST_S4(_bits, _c_fp32, cb)  nullptr
    #define EXL3_MGEMM_INST_S4(_bits, _c_fp32, cb) nullptr
#endif

#define EXL3_GEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, \
    exl3_gemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_gemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_gemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    EXL3_GEMM_INST_S4(_bits, _c_fp32, cb)

#define EXL3_MGEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, \
    exl3_mgemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_mgemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_mgemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    EXL3_MGEMM_INST_S4(_bits, _c_fp32, cb)

// Half-integer bitrates (bits + 0.5), mul1 codebook. Same symbols as quant/exl3_kernel_map.cuh's
// EXL3_KERNEL_INSTANCES_H / EXL3_KERNEL_EXTERNS_H, instantiated by comp_units_rdna/exl3_comp_unit_h{1,2,3}.cu
#if EXL3_RDNA_SHAPE4_N
    #define EXL3_GEMM_INST_S4_H(_bits, _c_fp32)  exl3_gemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_4>
    #define EXL3_MGEMM_INST_S4_H(_bits, _c_fp32) exl3_mgemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_4>
#else
    #define EXL3_GEMM_INST_S4_H(_bits, _c_fp32)  nullptr
    #define EXL3_MGEMM_INST_S4_H(_bits, _c_fp32) nullptr
#endif

#define EXL3_GEMM_KERNEL_INSTANCES_H(_bits, _c_fp32) \
    nullptr, \
    exl3_gemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_1>, \
    exl3_gemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_2>, \
    exl3_gemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_3>, \
    EXL3_GEMM_INST_S4_H(_bits, _c_fp32)

#define EXL3_MGEMM_KERNEL_INSTANCES_H(_bits, _c_fp32) \
    nullptr, \
    exl3_mgemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_1>, \
    exl3_mgemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_2>, \
    exl3_mgemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_3>, \
    EXL3_MGEMM_INST_S4_H(_bits, _c_fp32)

#define EXL3_KERNEL_INSTANCES_H(K) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_h##K[] = { EXL3_GEMM_KERNEL_INSTANCES_H(K, true) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_h##K[] = { EXL3_GEMM_KERNEL_INSTANCES_H(K, false) }; \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_h##K[] = { EXL3_MGEMM_KERNEL_INSTANCES_H(K, true) }; \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_h##K[] = { EXL3_MGEMM_KERNEL_INSTANCES_H(K, false) };

#define EXL3_KERNEL_EXTERNS_H(K) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_h##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_h##K[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_h##K[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_h##K[];

#define EXL3_GEMM_BASE_THREADS 256

// Instance arrays are indexed by shape and defined per (K, cb) so each codebook compiles as a separate
// translation unit (see comp_units/exl3_comp_unit_K_cbX.cu)

#define EXL3_KERNEL_EXTERNS_CB(K, cb) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb##cb[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb##cb[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb##cb[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb##cb[]; \

#define ALL_EXL3_KERNEL_EXTERNS(K) \
    EXL3_KERNEL_EXTERNS_CB(K, 0) \
    EXL3_KERNEL_EXTERNS_CB(K, 1) \
    EXL3_KERNEL_EXTERNS_CB(K, 2) \

#define EXL3_KERNEL_INSTANCES_CB(K, cb) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb##cb[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, true, cb) \
    }; \
    \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb##cb[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, cb) \
    }; \
    \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb##cb[] = { \
        EXL3_MGEMM_KERNEL_INSTANCES(K, true, cb) \
    }; \
    \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb##cb[] = { \
        EXL3_MGEMM_KERNEL_INSTANCES(K, false, cb) \
    };

fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    const int cc,
    const int size_m,
    const int size_k,
    const int size_n,
    const int bits,
    const bool c_fp32,
    const int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    const int cb,
    const bool half_k = false
);

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel
(
    const int cc,
    const int size_m,
    const int size_k,
    const int size_n,
    const int K,
    const bool c_fp32,
    const int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    const int cb,
    const int bszm_in,
    const int bszm_out,
    const bool half_k = false
);

fp_exl3_gemm_kernel get_gemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb, bool half_k = false);
fp_exl3_mgemm_kernel get_mgemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb, bool half_k = false);
