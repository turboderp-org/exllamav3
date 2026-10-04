#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../rdna_wmma.cuh"
#include <tuple>
#include <mutex>
#include <map>
#include <climits>
#include <algorithm>
#include <mutex>
#include "exl3_kernel_map_rdna.cuh"
#include "../../quant/exl3_devctx.cuh"
#include "../../quant/comp_units/exl3_comp_unit_1.cuh"
#include "../../quant/comp_units/exl3_comp_unit_2.cuh"
#include "../../quant/comp_units/exl3_comp_unit_3.cuh"
#include "../../quant/comp_units/exl3_comp_unit_4.cuh"
#include "../../quant/comp_units/exl3_comp_unit_5.cuh"
#include "../../quant/comp_units/exl3_comp_unit_6.cuh"
#include "../../quant/comp_units/exl3_comp_unit_7.cuh"
#include "../../quant/comp_units/exl3_comp_unit_8.cuh"

int exl3_gemm_num_kernel_shapes()
{
    return EXL3_GEMM_NUM_SHAPES;
}

int exl3_gemm_tilesize_k[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim[] = {EXL3_GEMM_BLOCKDIM};

// =============================================================================
// DEVIATION FROM THE CUDA PATH: shape selection is replaced wholesale
// =============================================================================
//
// The CUDA select_gemm_shape is a `switch (cc)` over CC_OLD / CC_AMPERE /
// CC_ADA / CC_HOPPER / CC_BLACKWELL. It is not usable here for three reasons,
// each independently sufficient:
//
// 1. gfx1151 is misclassified. exl3_devctx.cu:39 assigns CC_BLACKWELL when
//    prop.major >= 10; on this part hipDeviceProp_t reports major = 11,
//    minor = 5. RDNA therefore lands in the Hopper/Blackwell branch, whose
//    thresholds were tuned for parts with 90 KB+ of shared memory and far
//    more cache. There is no RDNA case.
//
// 2. The branches encode thresholds against the CUDA tile table, which this
//    build replaces (see exl3_kernel_map_rdna.cuh). Constants like "return 4
//    when size_n > 16384" refer to a shape 4 that is 16x16x512 there and
//    16x16x384 here. Reusing the thresholds against different tiles is
//    meaningless.
//
// 3. The CUDA path never has to check whether a shape fits in shared memory,
//    because its shapes always do on its targets. On 64 KB of LDS that is a
//    real constraint that varies with bitwidth, so it must be computed.
//
// The policy below ignores `cc` entirely -- correct here, because this
// translation unit is only ever compiled into the ROCm build -- and instead
// scores each shape that actually fits: larger N tiles preferred, weighted by
// achievable occupancy and by how well the work fills the available CUs.

static int exl3_gemm_sh_stages_tab[] = {EXL3_GEMM_SH_STAGES};

// LDS footprint of one (bitwidth, shape) pair, mirroring the allocations in
// exl3_gemm_inner. Must track that kernel's shared arrays; if it drifts, the
// selector will admit a shape the kernel cannot launch.
size_t exl3_gemm_smem_bytes(int bits, int shape_idx, bool half_k)
{
    const int TILE_M = 16;

    const int threads   = exl3_gemm_blockdim[shape_idx];
    const int tile_n    = exl3_gemm_tilesize_n[shape_idx];
    const int tile_k    = exl3_gemm_tilesize_k[shape_idx];
    const int sh_stages = exl3_gemm_sh_stages_tab[shape_idx];
    if (tile_n <= 0 || tile_k <= 0) return (size_t) -1;   // disabled shape

    const int tileblocks_k = tile_k / 16;
    const int tileblocks_n = tile_n / 16;
    const int num_warps    = threads / 32;

    // Must mirror exl3_gemm_inner_rdna.cuh exactly. If these drift, the
    // selector either rejects a shape that would run, or -- worse -- admits one
    // and the launch fails or silently under-allocates.
    //
    // sh_a uses the padded row stride (TILESIZE_K + 8) that replaces the CUDA
    // kernel's XOR swizzle, and sh_b_dq is the per-warp staging tile for the B
    // transpose into WMMA layout. Neither exists in the CUDA kernel.
    size_t a_bytes  = sizeof(half) * (size_t)(sh_stages * TILE_M * (tile_k + 8));
    // A half-integer bitrate (bits + 0.5, mul1) carries 16 * bits + 8 uint16 per 16x16 tile
    const int tile_u16 = 16 * bits + (half_k ? 8 : 0);
    size_t b_bytes  = sizeof(uint16_t) * (size_t)(sh_stages * tileblocks_k * tileblocks_n * tile_u16);
    size_t dq_bytes = sizeof(half) * (size_t)(num_warps * 16 * EXL3_GEMM_SH_B_DQ_STRIDE);

    // exl3_gemm_kernel instantiates the inner with shmem_out_had = true, so the
    // output tile is always staged in shared memory on this path. The cross-
    // sub_k reduction buffer is zero for every current shape (TILESIZE_K = 16
    // means TILEBLOCKS_K = 1), but is kept in the max for a future deeper-K shape.
    size_t c_reduce = (tileblocks_k > 1)
        ? (size_t) 8 * threads * (tileblocks_n / num_warps) : 0;
    size_t c_had    = (size_t) tile_n * TILE_M;
    size_t c_bytes  = sizeof(float) * (c_reduce > c_had ? c_reduce : c_had);

    return a_bytes + b_bytes + dq_bytes + c_bytes;
}

// -----------------------------------------------------------------------------
// Runtime LDS budget
// -----------------------------------------------------------------------------
// The compile-time SMEM_MAX is a build-time assumption (see the header). This is
// what the device actually has. Shape admission uses the smaller of the two, so
// a binary built assuming 90 KB still runs correctly on Strix Halo's 64 KB
// instead of selecting a shape that cannot launch.
//
// Queried once per device. hipDeviceGetAttribute is not free and shape selection
// sits on the launch path.
static size_t g_smem_max_cache[MAX_DEVICES] = {};

size_t exl3_rdna_device_smem_max(int device)
{
    if (device < 0) cudaGetDevice(&device);
    if (device < 0 || device >= MAX_DEVICES) return (size_t) SMEM_MAX;
    if (!g_smem_max_cache[device])
    {
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess || v <= 0)
        {
            (void) cudaGetLastError();
            v = SMEM_MAX;   // query failed: fall back to the build-time value
        }
        g_smem_max_cache[device] = (size_t) v;
    }
    return g_smem_max_cache[device];
}

size_t exl3_rdna_smem_budget(int device)
{
    size_t dev = exl3_rdna_device_smem_max(device);
    return dev < (size_t) SMEM_MAX ? dev : (size_t) SMEM_MAX;
}

static inline bool is_compat_shape(int shape_idx, int size_k_eff, int size_n_eff, int bits)
{
    int tk = exl3_gemm_tilesize_k[shape_idx];
    int tn = exl3_gemm_tilesize_n[shape_idx];
    // A disabled shape (EXL3_RDNA_SHAPE4_N = 0) has a zero tile size. Reject it
    // before the modulo, which would otherwise divide by zero.
    if (tk <= 0 || tn <= 0) return false;
    if ((size_k_eff % tk) != 0) return false;
    if ((size_n_eff % tn) != 0) return false;
    return exl3_gemm_smem_bytes(bits, shape_idx) <= exl3_rdna_smem_budget(-1);
}

// Occupancy is queried from the driver rather than derived, and cached: the
// query is not cheap and shape selection sits on the launch path.
static int g_occ_cache[9][EXL3_GEMM_NUM_SHAPES + 1][2];
static std::once_flag g_occ_cache_once;

static inline void ensure_occ_cache_init()
{
    std::call_once(g_occ_cache_once, []
    {
        for (int b = 0; b < 9; b++)
            for (int s = 0; s <= EXL3_GEMM_NUM_SHAPES; s++)
                g_occ_cache[b][s][0] = g_occ_cache[b][s][1] = -1;
    });
}

static inline int occ_blocks_per_cu(int bits, int shape_idx, bool c_fp32)
{
    ensure_occ_cache_init();
    int fp = c_fp32 ? 1 : 0;
    int& cached = g_occ_cache[bits][shape_idx][fp];
    if (cached >= 0) return cached;

    // cb = 0 is representative; the codebook variants differ in decode
    // arithmetic, not in tile shape or shared-memory footprint.
    fp_exl3_gemm_kernel k = get_gemm_kernel_ptr(bits, shape_idx, c_fp32, 0);
    if (!k) { cached = 0; return cached; }

    size_t smem = exl3_gemm_smem_bytes(bits, shape_idx);
    if (smem > (size_t) SMEM_MAX) { cached = 0; return cached; }

    int blocks = 0;
    cudaError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, (const void*) k, exl3_gemm_blockdim[shape_idx], smem);
    if (err != hipSuccess) blocks = 0;
    cached = blocks;
    return cached;
}

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out)
{
    (void) cc;        // deliberately unused -- see the note above
    (void) size_m;

    // The CUDA selector applies the multi-GEMM block scaling before choosing a shape and
    // computes divisibility on the effective size. That is wrong for the tile
    // walk in exl3_gemm_inner, which floors size_n / TILESIZE_N *per matrix*
    // with no remainder pass: a tile that divides only the scaled width (e.g.
    // N = 1024 with bszm_out = 3 admits the 384 tile because 3072 % 384 == 0)
    // silently drops the last size_n % TILESIZE_N columns of every matrix.
    // Compatibility below is therefore per-matrix; only the fill scoring uses
    // the effective sizes, which is all the scaling is good for.
    long long eff_k = (long long) size_k * bszm_in;
    long long eff_n = (long long) size_n * bszm_out;

    int device = 0;
    cudaGetDevice(&device);
    int cu_count = DevCtx::instance().get_num_sms(device);
    if (cu_count <= 0) cu_count = 1;

    int best_shape = 1;
    long long best_score = LLONG_MIN;

    for (int shape = EXL3_GEMM_NUM_SHAPES; shape >= 1; --shape)
    {
        if (!is_compat_shape(shape, size_k, size_n, K)) continue;

        int blocks_per_cu = occ_blocks_per_cu(K, shape, false);
        if (blocks_per_cu <= 0) continue;

        int tn = exl3_gemm_tilesize_n[shape];
        long long slices = (eff_k / 16) * (eff_n / tn);
        if (slices < 1) slices = 1;

        long long max_resident = (long long) cu_count * (long long) blocks_per_cu;
        long long underfill = (max_resident > slices) ? (max_resident - slices) : 0;

        long long score = 0;
        score += (long long) tn * 200;                              // wide N tiles amortise the weight read
        score += (long long) blocks_per_cu * 10000;                 // occupancy dominates when it differs
        score += (long long) std::min(slices, max_resident) * 10;   // reward filling the device
        score -= underfill * (multi ? 30 : 10);                     // penalise leaving CUs idle

        // Bitwidth bias: low bitwidths read less weight per tile, so they can
        // afford the wider tiles; high bitwidths cannot.
        if (K <= 4)      { if (shape == 4) score += 50000; }
        else if (K <= 6) { if (shape == 3) score += 30000; }
        else             { if (shape <= 2) score += 20000; }

        if (score > best_score) { best_score = score; best_shape = shape; }
    }

    return best_shape;
}


bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int K, bool half_k)
{
    int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n[shape_idx];

    // The CUDA version checks divisibility only. That is safe on a part whose LDS
    // matches the build-time SMEM_MAX and unsafe otherwise: this is the filter
    // the autotuner uses to build its candidate list, so a shape admitted here
    // gets launched. Without the budget test a 90 KB-assuming build would offer
    // the autotuner shapes that a 64 KB part cannot run, and the failure would
    // surface as a launch error from inside the autotuner rather than as an
    // unavailable shape. is_compat_shape (the non-autotune path) applies the
    // same test.
    if (tilesize_k <= 0 || tilesize_n <= 0) return false;
    if ((size_k % tilesize_k) != 0 || (size_n % tilesize_n) != 0) return false;
    return exl3_gemm_smem_bytes(K, shape_idx, half_k) <= exl3_rdna_smem_budget(-1);
}

// As in quant/exl3_kernel_map.cu: LDS a (shape, bitrate) instantiation requests, and the hard
// gate for forced shapes (which skip shape_compat). The CUDA map derives both from its constexpr
// exl3_gemm_smem_bytes() over the CUDA shape table; here the RDNA accounting above is the single
// source, against the RDNA budget (MIN(SMEM_MAX, device LDS)).
int exl3_gemm_shape_smem(int shape_idx, int K, bool half_k)
{
    if (shape_idx < 1 || shape_idx > EXL3_GEMM_NUM_SHAPES) return 0;
    size_t b = exl3_gemm_smem_bytes(K, shape_idx, half_k);
    return b == (size_t) -1 ? INT_MAX : (int) b;
}

void exl3_gemm_check_smem(int shape_idx, int K, bool half_k, const char* who)
{
    size_t need = exl3_gemm_smem_bytes(K, shape_idx, half_k);
    size_t have = exl3_rdna_smem_budget(-1);
    TORCH_CHECK(need <= have, who, ": shape ", shape_idx, " at ", K, (half_k ? ".5" : ""),
                " bpw needs ", need, " B of shared memory, device provides ", have);
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
    // A half-integer rate is shape-selected as the integer rate above it (the CUDA selector does the same)
    int shape_idx = force_shape_idx <= 0 ? select_gemm_shape(cc, size_m, size_k, size_n, K + (half_k ? 1 : 0), false, 1, 1) : force_shape_idx;

    TORCH_CHECK(shape_idx > 0 && shape_idx <= EXL3_GEMM_NUM_SHAPES, "exl3_gemm: no compatible kernel (or invalid forced shape index)");
    // The inner kernel floors size_n / TILESIZE_N with no remainder pass, so a
    // shape whose tile does not divide the problem silently skips the tail
    // columns. The selector never picks such a shape; a forced one must fail
    // loudly rather than return a fast wrong answer.
    TORCH_CHECK(size_k % exl3_gemm_tilesize_k[shape_idx] == 0 &&
                size_n % exl3_gemm_tilesize_n[shape_idx] == 0,
                "exl3_gemm: tile shape ", shape_idx, " (", exl3_gemm_tilesize_k[shape_idx],
                "x", exl3_gemm_tilesize_n[shape_idx], ") does not divide ", size_k, "x", size_n,
                " -- the kernel would silently drop the remainder");
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

    exl3_gemm_check_smem(shape_idx, K, half_k, "exl3_gemm");
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
    TORCH_CHECK(shape_idx > 0, "exl3_mgemm: no compatible kernel");
    // Same truncation guard as the gemm selector. size_n here is the max
    // per-matrix width (C.size(2)); callers passing a size_n_list with mixed
    // widths are only covered to the extent the max width is representative,
    // which holds for every current caller (uniform experts).
    TORCH_CHECK(size_k % exl3_gemm_tilesize_k[shape_idx] == 0 &&
                size_n % exl3_gemm_tilesize_n[shape_idx] == 0,
                "exl3_mgemm: tile shape ", shape_idx, " (", exl3_gemm_tilesize_k[shape_idx],
                "x", exl3_gemm_tilesize_n[shape_idx], ") does not divide ", size_k, "x", size_n,
                " -- the kernel would silently drop the remainder");
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

    exl3_gemm_check_smem(shape_idx, K, half_k, "exl3_mgemm");
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
