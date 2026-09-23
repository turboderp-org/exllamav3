#include <cuda_fp16.h>
#include "trellis_embed.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <map>
#include <mutex>
#include "util.h"
#include "util.cuh"

namespace py = pybind11;

/*

Fused trellis token-embedding dequant for the exl3_trellis_embed format (util/convert_embedding.py):
ONE kernel launch gathers n rows of a host-resident, page-locked and device-mapped packed table
(trellis_embed_register) and writes the dequantized fp32 rows on the GPU. The table itself never
touches VRAM: the kernel reads scattered rows zero-copy over PCIe (GPU MMU page-walks), so the
only device-resident state is the 128 KB mul1 codebook LUT and the D-entry fp32 column scales.

  warp per (row, 256-column-group); grid = ceil(n * G * 32 / 256), block = 256 (8 warps).
  Ring decode is NOT serial: the 16-bit tail-biting recurrence state_i = ((state_{i-1} << K) |
  k_i) & 0xFFFF is a finite window for K >= 6 (closed forms per K, bit-exact with
  embed_trellis.unpack_rows):
    K=8: state_i = (k_{i-1} << 8) | k_i
    K=7: ((k_{i-2} & 0x3) << 14) | (k_{i-1} << 7) | k_i
    K=6: ((k_{i-2} & 0xF) << 12) | (k_{i-1} << 6) | k_i
  K=8 reads its 8 codes + boundary byte as one 8-byte load (codes are bytes; needs every
  row's code base 8-aligned, checked host-side); K=6/7 read a 16-byte window at 4-byte
  alignment and extract the 10-code span from a 64-bit word pair (no __int128: MSVC).
  The ring-wrap head lanes whose 16-byte window would cross the row end fall back to per-byte loads (a few lanes
  per row, never outside the row's code region).
  The QTIP inverse (butterfly + LCG signs keyed by the TABLE ROW ID + column scales) runs in
  registers with the exact fp32 operation order of embed_trellis.dequant_rows_transformed and
  of the CPU AVX2 inv_transform reference, so all three paths are bit-identical. The per-lane
  LCG fast-forward to its 8-element span is NOT serial either: T^t(x) = A_t * x + C_t with
  (A_t, C_t) built from the binary powers of the step, uint64 wraparound (bit-exact with
  embed_trellis.lcg_signs).

  --use_fast_math (unconditional build flag, setup.py) is bit-safe for this kernel: the PTX
  of all six instantiations contains no FMA contraction with or without it (verified, CUDA
  13.3 - there is no mul feeding an add: the codebook->scale muls terminate in the butterfly
  via array stores nvcc does not contract) and no intermediate can be an f32 denormal: every
  nonzero value is a dyadic product of fp16-sourced operands (exponent >= -48) and the
  Hadamard +- stages never rescale, so |values| >= 2^-48 >> the f32 denormal cutoff 2^-126;
  the .ftz suffixes fast_math adds are unreachable.

*/

// ^ read the 10-code window ending two codes before pos; K < 8 or unaligned K = 8 fallback
template<int K>
static __device__ inline void load_codes_bits(const uint8_t* kb, uintptr_t kbaddr, int C, int pos, uint32_t* k)
{
    const int w0 = (pos - 2) * K;                       // first needed bit (code pos-2)
    const uint32_t mask = (1u << K) - 1u;
    uintptr_t byteaddr = kbaddr + (w0 >= 0 ? (w0 >> 3) : 0);
    uintptr_t a0 = byteaddr & ~(uintptr_t) 3;
    uintptr_t tend = kbaddr + (uintptr_t) C;
    int s = (int) (8 * (byteaddr - a0) + (w0 & 7));     // first needed bit inside the window

    if (w0 >= 0 && a0 + 16 <= tend)
    {
        // aligned 16-byte window, no wrap: extract 10*K <= 80 bits at bit offset s (s <= 31)
        // from a 64-bit word pair (no __int128: MSVC)
        const uint32_t* u = (const uint32_t*) a0;
        const uint64_t v0 = (uint64_t) u[0] | ((uint64_t) u[1] << 32);
        const uint64_t v1 = (uint64_t) u[2] | ((uint64_t) u[3] << 32);
        const uint64_t lo = (v0 >> s) | (s ? (v1 << (64 - s)) : 0);
        const uint64_t hi = s ? (v1 >> s) : v1;
        #pragma unroll
        for (int c = 0; c < 10; ++c)
        {
            const int b = c * K;
            uint64_t w;
            if (b + K <= 64) w = lo >> b;
            else if (b >= 64) w = hi >> (b - 64);
            else w = (lo >> b) | (hi << (64 - b));
            k[c] = (uint32_t) w & mask;
        }
    }
    else
    {
        // ring-wrap head (w0 < 0) or row-end window: per-byte loads, clamped into the row
        int b0 = w0 >> 3;                               // floor for w0 < 0 handled below
        if (w0 < 0) b0 = -(((-w0) + 7) >> 3);
        s = w0 - 8 * b0;                                // 0..7
        const int nbytes = (7 + (w0 & 7) + 10 * K) >> 3;  // <= 11
        uint64_t v = 0;
        uint32_t extra = 0;                             // bytes 8..10 (bits 64..87)
        #pragma unroll 4
        for (int q = 0; q < nbytes; ++q)
        {
            int idx = b0 + q;
            if (idx < 0) idx += C;                      // head wrap (ring bitstream)
            const uint32_t b = (uint32_t) kb[idx];
            if (q < 8) v |= (uint64_t) b << (8 * q);
            else extra |= b << (8 * (q - 8));
        }
        const uint64_t lo = (v >> s) | (s ? ((uint64_t) extra << (64 - s)) : 0);
        const uint64_t hi = s ? (uint64_t) (extra >> s) : (uint64_t) extra;
        #pragma unroll
        for (int c = 0; c < 10; ++c)
        {
            const int b = c * K;
            uint64_t w;
            if (b + K <= 64) w = lo >> b;
            else if (b >= 64) w = hi >> (b - 64);
            else w = (lo >> b) | (hi << (64 - b));
            k[c] = (uint32_t) w & mask;
        }
    }
}

template<int K, bool A8>
__global__ void trellis_embed_kernel
(
    const uint16_t* __restrict__ table,   // (V, G + D*K/16) host-mapped packed rows
    const uint16_t* __restrict__ cb,      // (65536,) fp16 mul1 codebook LUT
    const float*    __restrict__ cs,      // (D,) fp32 column scales
    float*          __restrict__ out,     // (n, D) fp32
    const int64_t*  __restrict__ ids,     // (n,) table row ids
    const int N, const int D, const int G, const int64_t n_rows, const uint64_t seed
)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int n = warp / G;
    if (n >= N) return;
    const int g = warp % G;
    const int64_t rid0 = ids[n];
    // an out-of-range id would be an out-of-bounds read of the device-mapped host table
    // (not a trapped fault: silent garbage at best, a sticky context error at worst):
    // clamp to the last row instead - an all-zero pad row in the loader's tables, so a
    // bad id decodes to 0.0 rather than corrupting the process. Loud rejection is the
    // Python caller's job (Embedding._gather raises IndexError for host-resident ids
    // before the H2D copy); the sign stream follows the clamped row, so the decode
    // stays context-free
    const int64_t rid = (rid0 < 0 || rid0 >= n_rows) ? (n_rows - 1) : rid0;
    const int W = G + D * K / 16;
    const uint16_t* row = table + (size_t) rid * W;
    const uint8_t* kb = (const uint8_t*) (row + G);
    const int C = D * K / 8;
    const float s = __half2float(__ushort_as_half(row[g]));    // this group's fp16 scale

    const int pos = g * 256 + lane * 8;                        // this lane's first position
    float v[8];

    if (K == 8 && A8)
    {
        // codes are bytes: 8 consecutive codes = one 8-byte load (pos is 8-aligned, kb 8-aligned)
        const uint2 kk = *(const uint2*) (kb + pos);
        uint32_t k[10];
        k[1] = kk.x & 0xFFu;
        k[2] = (kk.x >> 8) & 0xFFu;
        k[3] = (kk.x >> 16) & 0xFFu;
        k[4] = (kk.x >> 24) & 0xFFu;
        k[5] = kk.y & 0xFFu;
        k[6] = (kk.y >> 8) & 0xFFu;
        k[7] = (kk.y >> 16) & 0xFFu;
        k[8] = (kk.y >> 24) & 0xFFu;
        k[0] = (pos == 0) ? (uint32_t) kb[C - 1] : (uint32_t) kb[pos - 1];   // ring wrap
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = __half2float(__ushort_as_half(cb[(k[j] << 8) | k[j + 1]])) * s;
    }
    else
    {
        uint32_t k[10];
        load_codes_bits<K>(kb, (uintptr_t) kb, C, pos, k);
        constexpr uint32_t m2 = (1u << (16 - 2 * K)) - 1u;     // kept width of k_{i-2}
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            const uint32_t state = ((k[j] & m2) << (2 * K)) | (k[j + 1] << K) | k[j + 2];
            v[j] = __half2float(__ushort_as_half(cb[state])) * s;
        }
    }

    // ---- inverse transform (exact op order of the CPU AVX2 inv_transform reference) ----
    // stage 0: pairs (0,1),(2,3),(4,5),(6,7)
    #pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        const float a = v[2 * q], b = v[2 * q + 1];
        v[2 * q] = a + b;
        v[2 * q + 1] = a - b;
    }
    // stage 1: pairs (0,2),(1,3),(4,6),(5,7)
    {
        float a, b;
        a = v[0]; b = v[2]; v[0] = a + b; v[2] = a - b;
        a = v[1]; b = v[3]; v[1] = a + b; v[3] = a - b;
        a = v[4]; b = v[6]; v[4] = a + b; v[6] = a - b;
        a = v[5]; b = v[7]; v[5] = a + b; v[7] = a - b;
    }
    // stage 2: low 4 vs high 4
    {
        const float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        const float b0l = v[4], b1l = v[5], b2l = v[6], b3l = v[7];
        v[0] = a0 + b0l; v[1] = a1 + b1l; v[2] = a2 + b2l; v[3] = a3 + b3l;
        v[4] = a0 - b0l; v[5] = a1 - b1l; v[6] = a2 - b2l; v[7] = a3 - b3l;
    }
    // stages 3-7: inter-lane, lane distance 1,2,4,8,16
    #pragma unroll
    for (int st = 0; st < 5; ++st)
    {
        const int l = 1 << st;
        #pragma unroll
        for (int q = 0; q < 8; ++q)
        {
            const float other = __shfl_xor_sync(0xFFFFFFFFu, v[q], l);
            v[q] = ((lane & l) == 0) ? (v[q] + other) : (other - v[q]);
        }
    }
    // signs: one continuous LCG per row keyed by the TABLE ROW ID (format-pinned: the row's
    // decode is context-free, ids[n] is both the read index and the sign key). Each lane
    // fast-forwards to its 8-element span (t = g*256 + lane*8 steps) with the affine skip
    // T^t(x) = A_t * x + C_t: binary powers of the step (a_{j+1} = a_j^2, c_{j+1} = c_j * (1 + a_j),
    // all mod 2^64) composed per set bit of t, in ascending bit order (outer = higher bit)
    // = the exact uint64 wraparound arithmetic of embed_trellis.lcg_signs.
    uint64_t sLCG = seed * 0x9E3779B97F4A7C15ULL + (uint64_t) rid * 0xC2B2AE3D27D4EB4FULL + 1;
    const int t = g * 256 + lane * 8;
    uint64_t skipA = 1, skipC = 0;
    uint64_t aj = 6364136223846793005ULL, cj = 1442695040888963407ULL;
    for (int j = 0; t >> j; ++j)
    {
        if ((t >> j) & 1)
        {
            skipA = aj * skipA;
            skipC = aj * skipC + cj;
        }
        cj = cj * (1 + aj);
        aj = aj * aj;
    }
    sLCG = skipA * sLCG + skipC;
    float* o = out + (size_t) n * D + (size_t) g * 256 + lane * 8;
    const float* c = cs + (size_t) g * 256 + lane * 8;
    #pragma unroll
    for (int q = 0; q < 8; ++q)
    {
        sLCG = sLCG * 6364136223846793005ULL + 1442695040888963407ULL;
        const float sg = (sLCG >> 63) ? -1.0f : 1.0f;
        o[q] = ((v[q] * (1.0f / 16.0f)) * sg) * c[q];
    }
}


// ------------------------------------------------------------------ host: registration

struct RegEntry
{
    void* dev_ptr;
    int device;   // device the alias was created for (unregister guards + syncs this one)
    bool fresh;   // WE called cudaHostRegister (vs. region already registered by its owner)
    int refs;
    size_t nbytes;  // registered byte length (gather bounds-checks n_rows against it)
};

static std::map<void*, RegEntry>& reg_cache()
{
    static std::map<void*, RegEntry> cache;
    return cache;
}

// Reverse lookup for the gather's table_ptr validation: the launcher takes a bare int64
// alias, so it must be able to reject pointers no live registration produced and to
// bound n_rows against the registered region (a stale/garbage alias or an oversized
// n_rows would otherwise be an out-of-bounds read of the device-mapped host region)
static std::map<uintptr_t, void*>& dev_to_host()
{
    static std::map<uintptr_t, void*> m;
    return m;
}

static std::mutex& reg_mutex()
{
    static std::mutex mtx;
    return mtx;
}

int64_t trellis_embed_register(const at::Tensor& table, int64_t device_index)
{
    TORCH_CHECK(table.device().is_cpu(), "trellis_embed_register: table must be a CPU tensor");
    TORCH_CHECK_DTYPE(table, kShort);
    TORCH_CHECK(table.is_pinned(), "trellis_embed_register: table storage must be pinned (page-locked)");
    TORCH_CHECK(table.is_contiguous(), "trellis_embed_register: table must be contiguous");
    TORCH_CHECK(((uintptr_t) table.data_ptr() & 3) == 0, "trellis_embed_register: 4-byte aligned storage required");
    TORCH_CHECK(device_index >= -1, "trellis_embed_register: device_index must be -1 (current device) or a device ordinal");

    void* host = table.data_ptr();
    const size_t nbytes = (size_t) table.numel() * table.element_size();

    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(reg_mutex());
    const c10::cuda::CUDAGuard device_guard(device_index >= 0 ? (c10::DeviceIndex) device_index
                                                              : c10::cuda::current_device());
    const int dev = (int) c10::cuda::current_device();

    auto it = reg_cache().find(host);
    if (it != reg_cache().end())
    {
        // The alias is per (host pointer, device): a re-registration for a different device
        // must not silently hand out an alias mapped on another one (kernel-usable host
        // pointers need a UVA device mapping, verified at register time - unavailable
        // under WDDM, platform-dependent elsewhere) - reject instead. A different nbytes
        // at the same pointer means the original tensor was freed without unregister and
        // the address was reused: the stale alias would not cover a larger re-allocation
        TORCH_CHECK(it->second.device == dev,
                    "trellis_embed_register: this pinned region is already registered on cuda:",
                    it->second.device, ", not cuda:", dev, " (unregister it first)");
        TORCH_CHECK(it->second.nbytes == nbytes,
                    "trellis_embed_register: this host pointer was re-registered with a "
                    "different size (", nbytes, " vs ", (int64_t) it->second.nbytes,
                    " bytes): the original tensor was freed without unregister; "
                    "unregister it first");
        ++it->second.refs;
        return (int64_t) (uintptr_t) it->second.dev_ptr;
    }

    // Page-lock (idempotent: torch's pinned allocator may already have registered the region,
    // which surfaces as AlreadyRegistered or InvalidValue) and map it into the device address
    // space; the runtime API does not need a driver-API link
    cudaError_t cr = cudaHostRegister(host, nbytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
    const bool fresh = (cr == cudaSuccess);
    if (cr != cudaSuccess)
    {
        cudaGetLastError();   // clear the sticky error before the next call
        TORCH_CHECK(cr == cudaErrorHostMemoryAlreadyRegistered || cr == cudaErrorInvalidValue,
                    "trellis_embed_register: cudaHostRegister failed: ", cudaGetErrorString(cr));
    }
    void* dev_ptr = nullptr;
    cr = cudaHostGetDevicePointer(&dev_ptr, host, 0);
    if (cr != cudaSuccess)
    {
        if (fresh)
            cudaHostUnregister(host);   // region we just registered: undo it, don't leak until exit
        cudaGetLastError();             // clear the sticky error before the exception
        TORCH_CHECK(false,
                    "trellis_embed_register: cudaHostGetDevicePointer failed: ", cudaGetErrorString(cr),
                    " (the pinned region is not device-mapped)");
    }
    reg_cache()[host] = RegEntry{dev_ptr, dev, fresh, 1, nbytes};
    dev_to_host()[(uintptr_t) dev_ptr] = host;
    return (int64_t) (uintptr_t) dev_ptr;
}

void trellis_embed_unregister(const at::Tensor& table)
{
    void* host = table.data_ptr();
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(reg_mutex());
    auto it = reg_cache().find(host);
    if (it == reg_cache().end()) return;
    if (--it->second.refs > 0) return;
    const bool fresh = it->second.fresh;
    const int device = it->second.device;
    reg_cache().erase(it);
    dev_to_host().erase((uintptr_t) it->second.dev_ptr);
    // Only undo what we created: if the region was page-locked by its owner (torch's caching
    // pinned allocator, cuda_host.cpp, model_tp_shared, ...), tearing the registration down
    // here would leave THEIR aliases - and any block the allocator later re-hands-out to an
    // unrelated pin_memory() - unmapped.
    if (!fresh) return;
    const c10::cuda::CUDAGuard device_guard((c10::DeviceIndex) device);
    // Gather kernels may still be queued or running against the mapped region: unmapping
    // under a live kernel is an unmapped-address fault, i.e. a sticky context error that
    // poisons every later CUDA call in the process. Sync before the unmap.
    (void) cudaDeviceSynchronize();
    cudaError_t cr = cudaHostUnregister(host);
    if (cr != cudaSuccess)
        cudaGetLastError();   // teardown is racy by nature: a not-registered region is benign
}


// ------------------------------------------------------------------ host: gather launcher

void trellis_embed_gather
(
    int64_t table_ptr,
    const at::Tensor& cb,
    const at::Tensor& col_scales,
    const at::Tensor& ids,
    int64_t K,
    int64_t seed,
    int64_t n_rows,
    at::Tensor out
)
{
    TORCH_CHECK(K == 6 || K == 7 || K == 8, "trellis_embed_gather: K must be 6, 7 or 8");
    TORCH_CHECK_DTYPE(cb, kHalf);
    TORCH_CHECK_DTYPE(col_scales, kFloat);
    TORCH_CHECK_DTYPE(ids, kLong);
    TORCH_CHECK_DTYPE(out, kFloat);
    TORCH_CHECK(cb.numel() == 65536 && cb.is_contiguous(), "trellis_embed_gather: cb must be contiguous (65536,) half");
    TORCH_CHECK(col_scales.is_contiguous() && col_scales.dim() == 1 && col_scales.size(0) % 256 == 0,
                "trellis_embed_gather: col_scales must be contiguous (D,) float with D % 256 == 0");
    TORCH_CHECK(ids.is_contiguous() && ids.dim() == 1, "trellis_embed_gather: ids must be contiguous 1-D int64");
    TORCH_CHECK(n_rows >= 0, "trellis_embed_gather: n_rows must be non-negative");
    const int64_t D = col_scales.size(0);
    const int64_t n = ids.size(0);
    TORCH_CHECK(n <= INT32_MAX && D <= INT32_MAX, "trellis_embed_gather: n and D must fit in int32");
    TORCH_CHECK(out.is_contiguous() && out.dim() == 2 && out.size(0) >= n && out.size(1) == D,
                "trellis_embed_gather: out must be contiguous (>=n, D) float");
    TORCH_CHECK((D * K) % 16 == 0, "trellis_embed_gather: D*K must be a multiple of 16");
    TORCH_CHECK((table_ptr & 3) == 0, "trellis_embed_gather: table device pointer must be 4-byte aligned");
    // all tensor operands must be CUDA-resident on ONE device: the kernel runs on
    // cb.device() and reads/writes the others through device pointers (a CPU operand
    // would be a host pointer in a kernel -> illegal-memory fault, i.e. a sticky context
    // error; a second-GPU operand a peer-unmapped pointer). The device-mapped table alias
    // itself is per-device by construction (trellis_embed_register refuses cross-device
    // hand-out) but arrives as a bare int64, so it is only ever as valid as this device
    TORCH_CHECK(cb.is_cuda(), "trellis_embed_gather: cb must be a CUDA tensor");
    TORCH_CHECK(col_scales.device() == cb.device(), "trellis_embed_gather: col_scales must be on the same device as cb");
    TORCH_CHECK(ids.is_cuda(), "trellis_embed_gather: ids must be a CUDA tensor (the .cuh contract: int64 CUDA; a host pointer is kernel-usable only where UVA maps it, verified at register time - unavailable under WDDM)");
    TORCH_CHECK(ids.device() == cb.device(), "trellis_embed_gather: ids must be on the same device as cb");
    TORCH_CHECK(out.device() == cb.device(), "trellis_embed_gather: out must be on the same device as cb");

    const at::cuda::OptionalCUDAGuard device_guard(cb.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (!n) return;

    const int G = (int) (D / 256);
    const int N = (int) n;
    // K = 8 loads each row's 8 codes as one 8-byte vector: every row's code base
    // (base + rid * 2W + 2G bytes) must be 8-aligned, not just the table base
    const int W = G + (int) (D * K / 16);

    // table_ptr must be a live registration on THIS device and n_rows must fit the
    // registered region: an unregistered/stale alias or an oversized n_rows would turn
    // the gather into an out-of-bounds read of the device-mapped host region - silent
    // garbage at best, a sticky context error at worst. Host-side lookup only (no
    // device work, no sync): the ids themselves are clamped in-kernel, and loud
    // rejection of host-resident ids is the Python caller's job (Embedding._gather)
    {
        std::lock_guard<std::mutex> lock(reg_mutex());
        auto dit = dev_to_host().find((uintptr_t) table_ptr);
        TORCH_CHECK(dit != dev_to_host().end(),
                    "trellis_embed_gather: table_ptr was not returned by trellis_embed_register");
        auto it = reg_cache().find(dit->second);
        TORCH_CHECK(it != reg_cache().end() && it->second.device == (int) cb.device().index(),
                    "trellis_embed_gather: table_ptr is not a live registration on this device");
        TORCH_CHECK((size_t) n_rows * (size_t) 2 * (size_t) W <= it->second.nbytes,
                    "trellis_embed_gather: n_rows * row bytes exceeds the registered table size");
    }

    const bool a8 = (table_ptr & 7) == 0 && ((2 * W) % 8 == 0) && ((2 * G) % 8 == 0);
    const int block = 256;
    const int grid = (N * G * 32 + block - 1) / block;

    // A8 (the aligned 8-byte K = 8 load) only exists for K = 8: for K < 8 the flag is
    // dead, so only kernel<8, true> and kernel<K, false> are instantiated (6 of 10)
    #define LAUNCH(KK) do { \
        if (KK == 8 && a8) trellis_embed_kernel<8, true ><<<grid, block, 0, stream>>>( \
            (const uint16_t*) table_ptr, (const uint16_t*) cb.data_ptr(), \
            (const float*) col_scales.data_ptr(), (float*) out.data_ptr(), \
            (const int64_t*) ids.data_ptr(), N, (int) D, G, n_rows, (uint64_t) seed); \
        else trellis_embed_kernel<KK, false><<<grid, block, 0, stream>>>( \
            (const uint16_t*) table_ptr, (const uint16_t*) cb.data_ptr(), \
            (const float*) col_scales.data_ptr(), (float*) out.data_ptr(), \
            (const int64_t*) ids.data_ptr(), N, (int) D, G, n_rows, (uint64_t) seed); \
        } while (0)

    switch (K)
    {
        case 6: LAUNCH(6); break;
        case 7: LAUNCH(7); break;
        default: LAUNCH(8); break;
    }
    #undef LAUNCH
    cuda_check(cudaPeekAtLastError());
}