// =============================================================================
// wmma_gemm.cu -- routing and config selection for the WMMA GEMM backend
// =============================================================================
//
// See wmma_gemm.cuh for the backend. Called at the top of hgemm_gemmex_impl
// (hgemm.cu) after the narrow-N GEMV; returns false to leave the call to hipBLAS.
//
// Routed only when all of these hold (anything else falls back, never errors):
//   - EXL3_ROCM_WMMA_GEMM != 0 (default 1); for fp16 output also EXL3_ROCM_WMMA_GEMM_F16 != 0
//   - the device's gcnArchName has a table (gfx1151; gfx1100/gfx1101 via gfx1100's)
//   - the selected table entry's route flag for the output dtype is set (only shapes where
//     WMMA beats hipBLAS), and m >= the table's min_m (512 for an untuned
//     table). EXL3_ROCM_WMMA_GEMM=2 skips these two policy checks (every eligible call runs
//     here: correctness tests); EXL3_ROCM_WMMA_GEMM_CFG=<i> also pins config i (tuning).
//   - m >= EXL3_ROCM_WMMA_GEMM_MIN_M (default 9, i.e. above the m <= 8 narrow/GEMV regime;
//     lower it only to exercise the kernel's edge handling in tests)
//   - C rows packed (c_stride_m == n); A and B are packed by hgemm's own contract
//   - A, B, C base pointers 16-byte aligned, n % 8 == 0 and k % block_k == 0 for the chosen
//     config: every vector load is then 16-byte aligned and every K tile is whole. The
//     kernel's loads are bounded only by the matrix allocation, so a partial K tile would
//     read the next row of A (multiplied by zeros from B's out-of-range rows: finite A gives
//     the right answer, but an inf/NaN there would leak into a neighbouring row). Rows past
//     M and columns past N are handled by the kernel: out-of-allocation loads return zero
//     and the unaligned-epilogue variant bounds-checks every store.
//   - every matrix is < 2 GB (the kernel forms 32-bit byte offsets)
//
// The env switches are read on every call (getenv, no caching) so one process can switch
// them; the cost is negligible next to a GEMM launch.
// =============================================================================

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "wmma_gemm.cuh"
#include "wmma_gemm_table.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define EXL3_WMMA_MAX_DEVICES 64

static int wmma_env_int(const char* name, int dflt)
{
    const char* e = getenv(name);
    return (e && *e) ? atoi(e) : dflt;
}

// Arch bit of a device (0 = no table), from gcnArchName with any ":feature" suffix removed.
// Cached per device; torch caches the properties themselves.
static int wmma_arch_bit(int device)
{
    static int cache[EXL3_WMMA_MAX_DEVICES];
    static bool known[EXL3_WMMA_MAX_DEVICES];
    if (device < 0 || device >= EXL3_WMMA_MAX_DEVICES) return 0;
    if (known[device]) return cache[device];
    const auto* prop = at::cuda::getDeviceProperties(device);
    char arch[64];
    strncpy(arch, prop->gcnArchName, sizeof(arch) - 1);
    arch[sizeof(arch) - 1] = 0;
    char* colon = strchr(arch, ':');
    if (colon) *colon = 0;
    int bit = 0;
    if (!strcmp(arch, "gfx1151")) bit = EXL3_WMMA_ARCH_GFX1151;
    else if (!strcmp(arch, "gfx1100") || !strcmp(arch, "gfx1101")) bit = EXL3_WMMA_ARCH_GFX1100;
    cache[device] = bit;
    known[device] = true;
    return bit;
}

// rocm_wmma_gemm's find_best_config for one layout: exact (M, N, K) hit, else among the
// entries with the closest K, the one with the smallest (dM^2 + dN^2), ties to the larger M.
// Returns an entry index.
static int wmma_select(const Exl3WmmaEntry* t, int len, int m, int n, int k)
{
    for (int i = 0; i < len; ++i)
        if (t[i].m == m && t[i].n == n && t[i].k == k) return i;
    int64_t best_kd = std::numeric_limits<int64_t>::max();
    int closest_k = 0;
    for (int i = 0; i < len; ++i)
    {
        int64_t kd = t[i].k > k ? (int64_t) t[i].k - k : (int64_t) k - t[i].k;
        if (kd < best_kd) { best_kd = kd; closest_k = t[i].k; }
    }
    double best_d = std::numeric_limits<double>::max();
    int best = -1;
    for (int i = 0; i < len; ++i)
    {
        if (t[i].k != closest_k) continue;
        double dm = (double) m - t[i].m;
        double dn = (double) n - t[i].n;
        double d = dm * dm + dn * dn;
        // tie (m halfway between two tuned M): take the larger M, whose tile fits the grid;
        // the library takes the first (smaller) entry, which is slower
        if (d < best_d || (d == best_d && best >= 0 && t[i].m > t[best].m)) { best_d = d; best = i; }
    }
    return best;
}

static bool wmma_gemm_try_impl
(
    const half* a_ptr,
    const half* b_ptr,
    void* c_ptr,
    bool output_fp32,
    int size_m,
    int size_k,
    int size_n,
    int64_t c_stride_m,
    int device,
    hipStream_t stream,
    int* cfg_out
)
{
    // 0: off, 1: on (table route flags decide), 2: on for every eligible call (tests, tuning)
    const int mode = wmma_env_int("EXL3_ROCM_WMMA_GEMM", 1);
    if (mode == 0) return false;
    if (!output_fp32 && wmma_env_int("EXL3_ROCM_WMMA_GEMM_F16", 1) == 0) return false;
    if (size_m < wmma_env_int("EXL3_ROCM_WMMA_GEMM_MIN_M", 9) || size_m < 1) return false;
    if (size_n < 1 || size_k < 1) return false;
    if (c_stride_m != size_n) return false;
    if (size_n % 8) return false;
    if (((uintptr_t) a_ptr | (uintptr_t) b_ptr | (uintptr_t) c_ptr) & 15) return false;
    const int64_t lim = std::numeric_limits<int>::max();
    const int64_t c_elem = output_fp32 ? 4 : 2;
    if ((int64_t) size_m * size_k * 2 >= lim ||
        (int64_t) size_k * size_n * 2 >= lim ||
        (int64_t) size_m * size_n * c_elem >= lim) return false;

    const int bit = wmma_arch_bit(device);
    if (!bit) return false;
    const Exl3WmmaTable* table = nullptr;
    for (const Exl3WmmaTable& t : exl3_wmma_tables)
        if (t.arch_bit == bit) table = &t;
    if (!table) return false;

    const int ei = wmma_select(table->entries, table->count, size_m, size_n, size_k);
    if (ei < 0) return false;
    const Exl3WmmaEntry& entry = table->entries[ei];
    int ci = entry.cfg;
    const int force_cfg = wmma_env_int("EXL3_ROCM_WMMA_GEMM_CFG", -1);  // tuning aid: pin a config index
    const int n_cfgs = (int) (sizeof(exl3_wmma_cfgs) / sizeof(exl3_wmma_cfgs[0]));
    if (force_cfg >= 0 && force_cfg < n_cfgs) ci = force_cfg;
    else if (mode != 2)
    {
        if (size_m < table->min_m) return false;
        if (!(entry.route & (output_fp32 ? 1 : 2))) return false;
    }
    const Exl3WmmaCfg& cfg = exl3_wmma_cfgs[ci];
    if (!(cfg.arch_mask & bit)) return false;  // never launch a stub compiled without a body

    const int block_k = cfg.k_slices * 16;
    if (size_k % block_k) return false;
    const int block_m = cfg.warps_m * cfg.warp_tile_m * 16;
    const int block_n = cfg.warps_n * cfg.warp_tile_n * 16;
    const bool aligned = (size_m % block_m == 0) && (size_n % block_n == 0);

    cfg.launch(c_ptr, a_ptr, b_ptr, size_m, size_n, size_k, output_fp32, aligned, stream);
    *cfg_out = ci;
    return true;
}

bool wmma_gemm_try
(
    const half* a_ptr,
    const half* b_ptr,
    void* c_ptr,
    bool output_fp32,
    int size_m,
    int size_k,
    int size_n,
    int64_t c_stride_m,
    int device,
    hipStream_t stream
)
{
    int ci = -1;
    bool r = wmma_gemm_try_impl(a_ptr, b_ptr, c_ptr, output_fp32, size_m, size_k, size_n, c_stride_m,
                                device, stream, &ci);
    // EXL3_ROCM_WMMA_GEMM_TRACE=1: one stderr line per hgemm call with m > 8 (debug aid)
    if (size_m > 8 && wmma_env_int("EXL3_ROCM_WMMA_GEMM_TRACE", 0))
    {
        if (r) fprintf(stderr, "[wmma_gemm] m=%d n=%d k=%d out=%s -> wmma cfg %d\n",
                       size_m, size_n, size_k, output_fp32 ? "fp32" : "fp16", ci);
        else   fprintf(stderr, "[wmma_gemm] m=%d n=%d k=%d out=%s -> hipblas\n",
                       size_m, size_n, size_k, output_fp32 ? "fp32" : "fp16");
    }
    return r;
}
