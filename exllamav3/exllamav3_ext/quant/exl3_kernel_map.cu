#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include <tuple>
#include <mutex>
#include <map>
#include <climits>
#include <algorithm>
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "comp_units/exl3_comp_unit_1.cuh"
#include "comp_units/exl3_comp_unit_2.cuh"
#include "comp_units/exl3_comp_unit_3.cuh"
#include "comp_units/exl3_comp_unit_4.cuh"
#include "comp_units/exl3_comp_unit_5.cuh"
#include "comp_units/exl3_comp_unit_6.cuh"
#include "comp_units/exl3_comp_unit_7.cuh"
#include "comp_units/exl3_comp_unit_8.cuh"

int exl3_gemm_tilesize_m[] = {EXL3_GEMM_TILESIZE_M};
int exl3_gemm_tilesize_k[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim[] = {EXL3_GEMM_BLOCKDIM};

#if defined(USE_ROCM)

// Blocks of the (bits, shape) instance the runtime keeps resident per WGP, from the driver's
// occupancy query (not derived: it is not cheap and shape selection sits on the launch path).
// cb = 0 stands for every codebook: they differ in decode arithmetic, not in tile shape or LDS
static int occ_blocks_per_cu(int bits, int shape_idx, bool c_fp32)
{
    static int cache[9][EXL3_GEMM_NUM_SHAPES + 1][2];
    static std::once_flag once;
    std::call_once(once, []
    {
        for (int b = 0; b < 9; b++)
            for (int s = 0; s <= EXL3_GEMM_NUM_SHAPES; s++)
                cache[b][s][0] = cache[b][s][1] = -1;
    });
    int& cached = cache[bits][shape_idx][c_fp32 ? 1 : 0];
    if (cached >= 0) return cached;

    fp_exl3_gemm_kernel k = get_gemm_kernel_ptr(bits, shape_idx, c_fp32, 0);
    int smem = exl3_gemm_shape_smem(shape_idx, bits, false);
    int blocks = 0;
    if (k && smem <= SMEM_MAX)
    {
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (const void*) k, exl3_gemm_blockdim[shape_idx], smem) != cudaSuccess)
        {
            (void) cudaGetLastError();
            blocks = 0;
        }
    }
    cached = blocks;
    return cached;
}

// RDNA: cc carries no shape information (every RDNA part is one WMMA generation to the inner, and the
// CUDA thresholds refer to the CUDA tile table), so the shapes that fit are scored instead: wider N
// tiles amortise the weight read, occupancy dominates where it differs, and the device should be filled
// without leaving CUs idle. Divisibility is tested per matrix rather than on the bszm-scaled dims: the
// inner floors size_n / TILESIZE_N per matrix with no remainder pass, so a tile that divides only the
// scaled width would silently drop the last columns of every matrix
int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out)
{
    (void) cc;
    long long eff_k = (long long) size_k * bszm_in;
    long long eff_n = (long long) size_n * bszm_out;

    int device = 0;
    cudaGetDevice(&device);
    int cu_count = MAX(DevCtx::instance().get_num_sms(device), 1);

    int best_shape = 1;
    long long best_score = LLONG_MIN;
    for (int shape = EXL3_GEMM_NUM_SHAPES; shape >= 1; --shape)
    {
        if (!exl3_gemm_shape_compat(shape, size_m, size_k, size_n, K)) continue;
        int blocks_per_cu = occ_blocks_per_cu(K, shape, false);
        if (blocks_per_cu <= 0) continue;

        int tn = exl3_gemm_tilesize_n[shape];
        long long slices = MAX((eff_k / 16) * (eff_n / tn), 1LL);
        long long max_resident = (long long) cu_count * blocks_per_cu;
        long long underfill = MAX(max_resident - slices, 0LL);

        long long score = 0;
        score += (long long) tn * 200;
        score += (long long) blocks_per_cu * 10000;
        score += std::min(slices, max_resident) * 10;
        score -= underfill * (multi ? 30 : 10);
        // Low bitrates read less weight per tile and can afford the wider tiles; high bitrates cannot
        if (K <= 4)      { if (shape == 4) score += 50000; }
        else if (K <= 6) { if (shape == 3) score += 30000; }
        else             { if (shape <= 2) score += 20000; }

        if (score > best_score) { best_score = score; best_shape = shape; }
    }
    return best_shape;
}

#else

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out)
{
    bool mod_256 = (size_n % 256 == 0);
    bool mod_512 = (size_n % 512 == 0);

    size_k *= bszm_in;
    size_n *= bszm_out;

    switch(cc)
    {
        case CC_OLD:
        case CC_AMPERE:
            if (mod_256 && K <= 4)
            {
                if (size_n <= 2048 || size_k <= 2048) return 2;
                return 3;
            }
            if (mod_256 && size_n < 4096) return size_k > 8192 ? 3 : 2;
            if (mod_512 && (size_n * size_k) > (4096 * 4096) && K <= 6) return 4;
            if (mod_256) return 3;
            return 2;

        case CC_ADA:
            if (mod_256 && K <= 3)
            {
                if (size_k <= 2048 && !multi) return 2;
                if (size_n < 4096 && size_k <= 12288) return 2;
                return 3;
            }
            if (size_n <= 16384) return 2;
            if (mod_512 && size_n >= 32768) return 4;
            if (mod_256) return 3;
            return 2;

        case CC_HOPPER:
        case CC_BLACKWELL:
            if ((K == 4 || K == 2) && !multi)
            {
                if (size_k <= 2048) return 1;
            }
            if (K >= 7)
            {
                if (mod_256 && size_n <= 8192) return size_k > 32768 ? 3 : 2;
                if (mod_512 && size_n > 32768) return 4;
                return 2;
            }
            if (mod_256 && size_n <= 4096) return size_k > 8192 && K >= 3 ? 3 : 2;
            if (mod_512 && size_n > 16384) return 4;
            if (mod_256) return 3;
            return 2;
    }
    return 0;
}

#endif

int exl3_gemm_num_kernel_shapes()
{
    return EXL3_GEMM_NUM_SHAPES;
}

// Shared memory a shape/bitrate instantiation needs. Derived from the EXL3_GEMM_SHAPE_n
// macros via exl3_gemm_smem_bytes(), which the kernel itself static_asserts against, so this
// cannot drift from the actual layout. Used to reject shapes exceeding what a device will
// give a block - 64 KB on Turing rather than 90.
//
// shmem_out_had = true: the GEMM path stages a full output tile for the fused output Hadamard,
// which is the larger of the two sh_c variants, so this is the conservative bound for both it
// and the MoE kernel (which passes false).
int exl3_gemm_shape_smem(int shape_idx, int K, bool half_k)
{
    return exl3_gemm_smem_bytes_for_shape(shape_idx, K, half_k, true);
}

bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int K, bool half_k)
{
    int tilesize_m = exl3_gemm_tilesize_m[shape_idx];
    int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
    if (size_k % tilesize_k || size_n % tilesize_n) return false;
    // A wider row tile only pays once it replaces at least two 16-row passes
    if (tilesize_m > 16 && size_m <= tilesize_m / 2) return false;

    // Device-dependent: callers with tensors on a non-current device must set a device guard
    // first (every in-tree caller runs under OptionalCUDAGuard). Only matters on a mixed-arch
    // host, where a 90 KB-capable device would otherwise vouch for a 64 KB one.
    int device;
    cudaGetDevice(&device);
    return exl3_gemm_shape_smem(shape_idx, K, half_k) <= DevCtx::instance().get_smem_request(device);
}

// Hard gate for explicitly forced shapes, which skip the autotuner's shape_compat filter.
// Launching a shape whose static layout exceeds what the launch can request would read past
// the end of the extern __shared__ block - silent corruption rather than a failed launch.
void exl3_gemm_check_smem(int shape_idx, int K, bool half_k, const char* who)
{
    int device;
    cudaGetDevice(&device);
    int need = exl3_gemm_shape_smem(shape_idx, K, half_k);
    int have = DevCtx::instance().get_smem_request(device);
    TORCH_CHECK(need <= have, who, ": shape ", shape_idx, " at ", K, (half_k ? ".5" : ""),
                " bpw needs ", need, " B of shared memory, device provides ", have);
}

// Forced shapes skip the selector's divisibility test. The inner floors size / TILESIZE with no
// remainder pass, so a tile that does not divide the problem would silently skip the tail rather
// than fail
static void exl3_gemm_check_divides(int shape_idx, int size_k, int size_n, const char* who)
{
    int tk = exl3_gemm_tilesize_k[shape_idx], tn = exl3_gemm_tilesize_n[shape_idx];
    TORCH_CHECK(size_k % tk == 0 && size_n % tn == 0, who, ": tile shape ", shape_idx, " (", tk, "x", tn,
                ") does not divide ", size_k, "x", size_n, " -- the kernel would silently drop the remainder");
}

// Instance tables, [K][cb] -> array indexed by shape_idx. Row 0 unused (no K = 0 instances)

#define EXL3_KERNEL_TABLE_ROW(fp, K) \
    { tfp_exl3_gemm_kernel_##fp##_b##K##_cb0, tfp_exl3_gemm_kernel_##fp##_b##K##_cb1, tfp_exl3_gemm_kernel_##fp##_b##K##_cb2 }
#define EXL3_MKERNEL_TABLE_ROW(fp, K) \
    { tfp_exl3_mgemm_kernel_##fp##_b##K##_cb0, tfp_exl3_mgemm_kernel_##fp##_b##K##_cb1, tfp_exl3_mgemm_kernel_##fp##_b##K##_cb2 }

static fp_exl3_gemm_kernel* const tab_gemm_fp32[9][3] =
{
    { nullptr, nullptr, nullptr },
    EXL3_KERNEL_TABLE_ROW(fp32, 1), EXL3_KERNEL_TABLE_ROW(fp32, 2), EXL3_KERNEL_TABLE_ROW(fp32, 3),
    EXL3_KERNEL_TABLE_ROW(fp32, 4), EXL3_KERNEL_TABLE_ROW(fp32, 5), EXL3_KERNEL_TABLE_ROW(fp32, 6),
    EXL3_KERNEL_TABLE_ROW(fp32, 7), EXL3_KERNEL_TABLE_ROW(fp32, 8)
};

static fp_exl3_gemm_kernel* const tab_gemm_fp16[9][3] =
{
    { nullptr, nullptr, nullptr },
    EXL3_KERNEL_TABLE_ROW(fp16, 1), EXL3_KERNEL_TABLE_ROW(fp16, 2), EXL3_KERNEL_TABLE_ROW(fp16, 3),
    EXL3_KERNEL_TABLE_ROW(fp16, 4), EXL3_KERNEL_TABLE_ROW(fp16, 5), EXL3_KERNEL_TABLE_ROW(fp16, 6),
    EXL3_KERNEL_TABLE_ROW(fp16, 7), EXL3_KERNEL_TABLE_ROW(fp16, 8)
};

static fp_exl3_mgemm_kernel* const tab_mgemm_fp32[9][3] =
{
    { nullptr, nullptr, nullptr },
    EXL3_MKERNEL_TABLE_ROW(fp32, 1), EXL3_MKERNEL_TABLE_ROW(fp32, 2), EXL3_MKERNEL_TABLE_ROW(fp32, 3),
    EXL3_MKERNEL_TABLE_ROW(fp32, 4), EXL3_MKERNEL_TABLE_ROW(fp32, 5), EXL3_MKERNEL_TABLE_ROW(fp32, 6),
    EXL3_MKERNEL_TABLE_ROW(fp32, 7), EXL3_MKERNEL_TABLE_ROW(fp32, 8)
};

EXL3_KERNEL_EXTERNS_H(1)
EXL3_KERNEL_EXTERNS_H(2)
EXL3_KERNEL_EXTERNS_H(3)

// Half-integer bitrates: row K = integer part (K + 0.5 bpw), mul1 only
static fp_exl3_gemm_kernel* const tab_gemm_fp32_h[4] =
    { nullptr, tfp_exl3_gemm_kernel_fp32_h1, tfp_exl3_gemm_kernel_fp32_h2, tfp_exl3_gemm_kernel_fp32_h3 };
static fp_exl3_gemm_kernel* const tab_gemm_fp16_h[4] =
    { nullptr, tfp_exl3_gemm_kernel_fp16_h1, tfp_exl3_gemm_kernel_fp16_h2, tfp_exl3_gemm_kernel_fp16_h3 };
static fp_exl3_mgemm_kernel* const tab_mgemm_fp32_h[4] =
    { nullptr, tfp_exl3_mgemm_kernel_fp32_h1, tfp_exl3_mgemm_kernel_fp32_h2, tfp_exl3_mgemm_kernel_fp32_h3 };
static fp_exl3_mgemm_kernel* const tab_mgemm_fp16_h[4] =
    { nullptr, tfp_exl3_mgemm_kernel_fp16_h1, tfp_exl3_mgemm_kernel_fp16_h2, tfp_exl3_mgemm_kernel_fp16_h3 };

static fp_exl3_mgemm_kernel* const tab_mgemm_fp16[9][3] =
{
    { nullptr, nullptr, nullptr },
    EXL3_MKERNEL_TABLE_ROW(fp16, 1), EXL3_MKERNEL_TABLE_ROW(fp16, 2), EXL3_MKERNEL_TABLE_ROW(fp16, 3),
    EXL3_MKERNEL_TABLE_ROW(fp16, 4), EXL3_MKERNEL_TABLE_ROW(fp16, 5), EXL3_MKERNEL_TABLE_ROW(fp16, 6),
    EXL3_MKERNEL_TABLE_ROW(fp16, 7), EXL3_MKERNEL_TABLE_ROW(fp16, 8)
};

fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* num_sms,
    int cb,
    bool half_k
)
{
    int shape_idx = force_shape_idx <= 0 ? select_gemm_shape(cc, size_m, size_k, size_n, K + (half_k ? 1 : 0), false, 1, 1) : force_shape_idx;

    TORCH_CHECK(shape_idx > 0 && shape_idx <= EXL3_GEMM_NUM_SHAPES, "exl3_gemm: no compatible kernel (or invalid forced shape index)");
    exl3_gemm_check_divides(shape_idx, size_k, size_n, "exl3_gemm");
    exl3_gemm_check_smem(shape_idx, K, half_k, "exl3_gemm");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    // Avoid empty blocks
    if (num_sms)
    {
        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n;
        *num_sms = MAX(MIN(max_slices, *num_sms), 1);
    }

    return get_gemm_kernel_ptr(K, shape_idx, c_fp32, cb, half_k);
}

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* num_sms,
    int cb,
    int bszm_in,
    int bszm_out,
    bool half_k
)
{
    int shape_idx = force_shape_idx <= 0 ? select_gemm_shape(cc, size_m, size_k, size_n, K + (half_k ? 1 : 0), true, bszm_in, bszm_out) : force_shape_idx;
    TORCH_CHECK(shape_idx > 0 && shape_idx <= EXL3_GEMM_NUM_SHAPES, "exl3_mgemm: no compatible kernel (or invalid forced shape index)");
    // size_n is the max per-matrix width; mixed-width bundles are covered to the extent it is representative
    exl3_gemm_check_divides(shape_idx, size_k, size_n, "exl3_mgemm");
    exl3_gemm_check_smem(shape_idx, K, half_k, "exl3_mgemm");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    // Avoid empty blocks
    if (num_sms)
    {
        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n / (*num_sms > 128 ? 20 : 24);
        *num_sms = MIN(max_slices, *num_sms);
    }

    return get_mgemm_kernel_ptr(K, shape_idx, c_fp32, cb, half_k);
}


fp_exl3_gemm_kernel get_gemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb, bool half_k)
{
    if (half_k)
    {
        TORCH_CHECK(K >= 1 && K <= 3 && cb == 2, "No kernel for half-integer GEMM bitrate (1.5, 2.5, 3.5 bpw with mul1 only)");
        return (c_fp32 ? tab_gemm_fp32_h : tab_gemm_fp16_h)[K][shape_idx];
    }
    TORCH_CHECK(K >= 1 && K <= 8 && cb >= 0 && cb <= 2, "No kernel for GEMM shape");
    return (c_fp32 ? tab_gemm_fp32 : tab_gemm_fp16)[K][cb][shape_idx];
}


fp_exl3_mgemm_kernel get_mgemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb, bool half_k)
{
    if (half_k)
    {
        TORCH_CHECK(K >= 1 && K <= 3 && cb == 2, "No kernel for half-integer MGEMM bitrate (1.5, 2.5, 3.5 bpw with mul1 only)");
        return (c_fp32 ? tab_mgemm_fp32_h : tab_mgemm_fp16_h)[K][shape_idx];
    }
    TORCH_CHECK(K >= 1 && K <= 8 && cb >= 0 && cb <= 2, "No kernel for GEMM shape");
    return (c_fp32 ? tab_mgemm_fp32 : tab_mgemm_fp16)[K][cb][shape_idx];
}
