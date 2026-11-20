// Fused shared-expert gate tail for the RDNA block-sparse MLP decode route.
//
// Replaces the eager to(Float)/matmul/sigmoid/copy_/addcmul_ chain with a single launch:
// the per-row dot of y and the gate weight, its sigmoid, and the gated add of the shared
// expert output into the accumulator. The chain's six dispatched ops cost mostly launch
// and ATen dispatch time at batch 1, so fusing them measurably speeds up decode even
// though the arithmetic is unchanged. The dot accumulates in fp32 with a fixed thread
// deposition order: deterministic per build, numerically equivalent to (though not
// bit-identical with) a gemv reduction.
//
// Two variants, chosen by the host on the accumulator dtype: the decode route may hand
// us either half or float routed/shared/destination buffers. Optional stages are skipped
// by host-chosen parameters, so the tail stays a single launch in every shape: with
// sh == null there is nothing to add and the kernel is a plain copy (dst = routed), and
// with W == 0 there is no gate dot and the shared expert is added unscaled
// (dst = routed + sh).
//
// The grid is one block per output row, and the block computes the gate dot once before
// sweeping H in a strided loop: a grid that split H across blocks would recompute the
// same W-wide dot (and re-read the y row) once per block, pure redundant traffic for a
// result every block needs identically.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// launch-error check at the entry points, and the cuda_* spellings the rest of the rocm
// kernels use through the compat layer
#include "../../util.cuh"

#include <cstdlib>

// The block reduction below is written for a 32-lane wavefront: warp shuffles with a
// full mask and one shared slot per warp. RDNA (gfx10xx/gfx11xx/gfx12xx) is wavefront32
// only; wavefront64 exists solely on pre-RDNA GCN parts, which this route does not run
// on. Fail the build rather than silently mis-reduce if that ever changes.
//
// The __AMDGCN_WAVEFRONT_SIZE predefine is the only compile-time source there is, and only
// where the toolchain emits it: a second source, __builtin_amdgcn_wavefrontsize(), does not
// exist as a constant expression on the ROCm clang in use -- measured on the target toolchain
// (AMD clang 22.0.0git, ROCm 7.2) as "static assertion expression is not an integral
// constant expression" for gfx1200, gfx908 and gfx1030 alike; its 'const' flag in
// BuiltinsAMDGPU.def means side-effect-free, not constexpr. Nor may a missing macro be
// treated as an error: it is not defined on that toolchain at all, device pass included,
// so demanding it would break every build of the very configuration this kernel runs on.
// GCN targets with wave64 capability hit the macro branch (older toolchains emit the
// predefine there), and RDNA (gfx101x/gfx12xx) cannot select wave64 in the first place --
// what is left uncovered is a toolchain without the predefine targeting wave64 GCN, which
// this route does not serve. A preprocessor conditional on a possibly-undefined macro is
// well-formed (it evaluates as 0), so this stays silent when the macro is absent.
//
// The hole left by a toolchain that emits no predefine (the measured target toolchain)
// closes at first launch instead: gate_fused_check_wavefront() below asks the runtime
// for the warp size once per process and raises if the device does not run 32 lanes,
// so an unsupported configuration fails loudly at the first tail rather than silently
// mis-reducing. A compile-time #error on a missing macro is NOT that check: the target
// toolchain has no macro to see, and demanding one breaks every build (581bfece did).
#if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE != 32)
#error "exl3_gate_fused_rdna: block reduction assumes a 32-lane wavefront"
#endif

namespace
{

// Wavefront guard for toolchains that emit no __AMDGCN_WAVEFRONT_SIZE predefine (see the
// compile-time check above): one hipDeviceGetAttribute per process, at the first fused
// tail launch on the device the kernel will run on (the current device of the launch
// stream). A wave64 device raises here instead of mis-reducing the gate dot; on every
// RDNA part the query returns 32 and costs nothing after the first call.
inline void gate_fused_check_wavefront()
{
    static const bool checked = [] {
        int dev = 0;
        // Both queries raise on failure rather than skipping the check: the route only
        // ever runs on an established device context (guard, tensor storage and stream
        // are already in use), so a failing hipGetDevice is a broken runtime, not a
        // benign absence of a device -- and quietly marking the check done would let
        // the flag lie about a check that never happened.
        TORCH_CHECK(hipGetDevice(&dev) == hipSuccess,
                "exl3_gate_fused_rdna: cannot get the current device for the wavefront check");
        int ws = 0;
        TORCH_CHECK(hipDeviceGetAttribute(&ws, hipDeviceAttributeWarpSize, dev) == hipSuccess,
                "exl3_gate_fused_rdna: cannot query the device wavefront size");
        TORCH_CHECK(ws == 32, "exl3_gate_fused_rdna: block reduction assumes a 32-lane wavefront, device reports ", ws);
        return true;
    }();
    (void) checked;
}


// EXL3_RDNA_GATE_FUSED=0 bypasses this kernel and runs the eager chain instead (the
// original to(Float)/matmul/sigmoid/copy_/addcmul_ ops, kept in the host route as the
// fallback). Default on.
inline bool gate_fused_enabled()
{
    static const bool v = [] {
        const char* e = getenv("EXL3_RDNA_GATE_FUSED");
        return !(e && *e == '0');
    }();
    return v;
}

// Half/float boundary through intrinsics only: the extension builds with
// __HIP_NO_HALF_CONVERSIONS__/__HIP_NO_HALF_OPERATORS__, which makes C-style casts
// to/from __half ill-formed. Overloads instead of template specializations; the float
// pair is the identity, so the fp32 variant of the kernel compiles them away.
__device__ __forceinline__ float gate_load_as_float(const float* p)  { return *p; }
__device__ __forceinline__ float gate_load_as_float(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void gate_store_rounded(float* p, float v) { *p = v; }
__device__ __forceinline__ void gate_store_rounded(__half* p, float v) { *p = __float2half(v); }

template<typename T>
__device__ __forceinline__ void moe_gate_fused_body
(
    const __half* __restrict__ y,
    const __half* __restrict__ w,
    const T* __restrict__ routed,
    const T* __restrict__ sh,
    T* __restrict__ dst,
    const int H,
    const int W
)
{
    __shared__ float sred[32]; // one partial per warp, then warp 0 folds them; wavefront32 (see top of file)
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int tpr = blockDim.x;
    float gate = 1.0f;
    if (W > 0)
    {
        float acc = 0.0f;
        for (int k = tid; k < W; k += tpr)
            acc += __half2float(y[(long long) row * W + k]) * __half2float(w[k]);
        for (int o = 16; o > 0; o >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, o);
        if ((tid & 31) == 0)
            sred[tid >> 5] = acc;
        __syncthreads();
        if (tid < 32)
        {
            float v = (tid < (tpr >> 5)) ? sred[tid] : 0.0f;
            for (int o = 16; o > 0; o >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, o);
            if (tid == 0)
                sred[0] = v;
        }
        __syncthreads();
        gate = 1.0f / (1.0f + expf(-sred[0]));
    }
    // One block owns a whole row: sweep H in blockDim strides (grid is one block per row,
    // so the gate above is computed exactly once per row)
    for (int i = tid; i < H; i += tpr)
    {
        // fp32 math, one round to T on store (RNE) -- identical treatment for both variants
        // (__float2half is RNE; the float variant stores unrounded), so the bypass chain
        // and this kernel differ only by fp32 rounding of the gate scalar
        const float r = gate_load_as_float(&routed[(long long) row * H + i]);
        if (sh)
        {
            const float s = gate_load_as_float(&sh[(long long) row * H + i]);
            gate_store_rounded(&dst[(long long) row * H + i], r + s * gate);
        }
        else
            gate_store_rounded(&dst[(long long) row * H + i], r);
    }
}

__global__ void moe_gate_fused_h_kernel
(
    const __half* __restrict__ y, const __half* __restrict__ w,
    const __half* __restrict__ routed, const __half* __restrict__ sh,
    __half* __restrict__ dst, const int H, const int W
)
{
    moe_gate_fused_body<__half>(y, w, routed, sh, dst, H, W);
}

__global__ void moe_gate_fused_f32_kernel
(
    const __half* __restrict__ y, const __half* __restrict__ w,
    const float* __restrict__ routed, const float* __restrict__ sh,
    float* __restrict__ dst, const int H, const int W
)
{
    moe_gate_fused_body<float>(y, w, routed, sh, dst, H, W);
}

}

extern "C" int exl3_moe_gate_fused_rdna_enabled(void)
{
    return gate_fused_enabled() ? 1 : 0;
}

extern "C" void exl3_moe_gate_fused_rdna
(
    const void* y, const void* w, const void* routed, const void* sh, void* dst,
    int H, int W, int bsz, void* stream
)
{
    if (!gate_fused_enabled()) return;
    gate_fused_check_wavefront(); // one-time wavefront guard (see definition)
    // One block per output row; an empty shape is a no-op rather than a zero-dim launch
    if (H <= 0 || bsz <= 0) return;
    dim3 grid(1, bsz);
    hipLaunchKernelGGL(moe_gate_fused_h_kernel, grid, dim3(256), 0, (cudaStream_t) stream,
        (const __half*) y, (const __half*) w, (const __half*) routed, (const __half*) sh,
        (__half*) dst, H, W);
    cuda_check(cudaPeekAtLastError());
}

extern "C" void exl3_moe_gate_fused_rdna_f32
(
    const void* y, const void* w, const void* routed, const void* sh, void* dst,
    int H, int W, int bsz, void* stream
)
{
    if (!gate_fused_enabled()) return;
    gate_fused_check_wavefront(); // one-time wavefront guard (see definition)
    if (H <= 0 || bsz <= 0) return;
    dim3 grid(1, bsz);
    hipLaunchKernelGGL(moe_gate_fused_f32_kernel, grid, dim3(256), 0, (cudaStream_t) stream,
        (const __half*) y, (const __half*) w, (const float*) routed, (const float*) sh,
        (float*) dst, H, W);
    cuda_check(cudaPeekAtLastError());
}
