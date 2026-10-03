#pragma once

// RDNA WMMA fragment types and operations (native 16x16x16 FP16 -> FP32 on gfx11, the fp32-accumulating
// variant on gfx12), used by the RDNA GEMM (rocm/quant/exl3_gemm_*_rdna.cuh).

#include "../ptx.cuh"

// A traditional include guard rather than `#pragma once`, since this header is
// reached through several relative paths from rocm/quant/ and rocm/cpu/.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>   // __hip_bfloat16, for the bf16 WMMA wrappers
#include <stdint.h>

// =============================================================================
// WMMA fragment types (native 16x16x16)
// =============================================================================

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t __attribute__((ext_vector_type(8)));

// int8 WMMA operand/accumulator vectors. A and B pack 16 int8 into 4 dwords;
// the accumulator is 8 int32. The names do not collide with the HIP headers'.
typedef int      int32x4_t __attribute__((ext_vector_type(4)));
typedef int      int32x8_t __attribute__((ext_vector_type(8)));

// bf16 WMMA operands. __bf16 is the compiler's own type; __hip_bfloat16 (what
// __nv_bfloat16 aliases to under HIP) is a distinct spelling but identical at
// 2 bytes / align 2, so the loaders
// below can take the HIP type and reinterpret without a conversion.
typedef __bf16   bf16x16_t __attribute__((ext_vector_type(16)));

struct WmmaFragA
{
    half16_t data;
    __device__ __forceinline__ void clear()
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
            ((_Float16*)&data)[i] = (_Float16)0.0f;
    }
};

struct WmmaFragB
{
    half16_t data;
    __device__ __forceinline__ void clear()
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
            ((_Float16*)&data)[i] = (_Float16)0.0f;
    }
};

struct WmmaFragC
{
    float8_t data;

    __device__ __forceinline__ void clear()
    {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            ((float*)&data)[i] = 0.0f;
    }

    __device__ __forceinline__ float& operator[](int i)
    {
        return ((float*)&data)[i];
    }
    __device__ __forceinline__ const float& operator[](int i) const
    {
        return ((const float*)&data)[i];
    }
};

// fp16-accumulate C fragment.
//
// Named WmmaFragC_f16 rather than WmmaFragC_h to keep it clearly distinct from
// FragC_h above, which is the ptx.cuh-compatible Vec<half2,2> and unrelated.
//
// This is 16 halves wide but a single mma writes only 8 of them: the slot is
// `i*2 + opsel`, and the other half of each pair is left untouched: with
// opsel=0 only slots 0,2,..,14 are written, with opsel=1 only 1,3,..,15, and the unwritten slots keep their
// prior contents.
//
// Two consequences worth knowing before choosing this over the f32 form:
//   - Accuracy is lower: the accumulation itself happens in fp16. The f32 form
//     costs the same 8 VGPRs, so for a single accumulator f32 is strictly
//     better and is what the GEMM inner loop uses.
//   - The preserved half is a real feature, not padding: two independent
//     accumulators can share one fragment (one at opsel=0, one at opsel=1),
//     which halves accumulator register pressure when a kernel is carrying many
//     tiles. That is the reason to reach for this variant.
struct WmmaFragC_f16
{
    half16_t data;

    __device__ __forceinline__ void clear()
    {
        #pragma unroll
        for (int i = 0; i < 16; i++) ((_Float16*)&data)[i] = (_Float16)0.0f;
    }
    // Clear only one of the two interleaved accumulators.
    template <bool opsel>
    __device__ __forceinline__ void clear_half()
    {
        #pragma unroll
        for (int i = 0; i < 8; i++) ((_Float16*)&data)[i * 2 + (opsel ? 1 : 0)] = (_Float16)0.0f;
    }
    // Element i (0..7) of the accumulator selected by opsel.
    template <bool opsel>
    __device__ __forceinline__ half get(int i) const
    {
        return __ushort_as_half(((const unsigned short*)&data)[i * 2 + (opsel ? 1 : 0)]);
    }
};

// bf16 A/B fragments. There is deliberately no bf16 C fragment: this variant
// accumulates into fp32 with exactly the same C layout as the f16 form, so it
// reuses WmmaFragC. store_matrix_c, store_matrix_c_half, load_accumulate_c and
// the _checked variants therefore all work on a bf16 matmul unchanged.
struct WmmaFragA_bf16
{
    bf16x16_t data;
    __device__ __forceinline__ void clear()
    {
        #pragma unroll
        for (int i = 0; i < 16; i++) ((__bf16*)&data)[i] = (__bf16)0.0f;
    }
};

struct WmmaFragB_bf16
{
    bf16x16_t data;
    __device__ __forceinline__ void clear()
    {
        #pragma unroll
        for (int i = 0; i < 16; i++) ((__bf16*)&data)[i] = (__bf16)0.0f;
    }
};

// int8 fragments. Kept at global scope alongside WmmaFragA/B/C so all fragment
// types live in one place and only the operations are namespaced.
struct WmmaFragA_i8 { int32x4_t data; };
struct WmmaFragB_i8 { int32x4_t data; };

struct WmmaFragC_i32
{
    int32x8_t data;

    __device__ __forceinline__ void clear()
    {
        #pragma unroll
        for (int i = 0; i < 8; i++) ((int*)&data)[i] = 0;
    }
    __device__ __forceinline__ int& operator[](int i) { return ((int*)&data)[i]; }
    __device__ __forceinline__ const int& operator[](int i) const { return ((const int*)&data)[i]; }
};

// =============================================================================
// rdna_wmma namespace -- WMMA operations for the GEMM/GEMV kernels
// =============================================================================
//
// Native RDNA 3.5 WMMA: 16x16x16 FP16 -> FP32, computes C += A x B.
//
// Builtin signature -- note B precedes A:
//   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(B, A, C) -> C
//
// Wave32 lane mapping:
//   A fragment: lane L holds row (L % 16), all 16 columns
//   B fragment: lane L holds column (L % 16), all 16 rows
//   C fragment: row = lane % 16; col_base = (lane >= 16) ? 1 : 0;
//               each lane stores 8 floats at columns [i*2 + col_base], i in 0..7
//
// This layout is NOT interchangeable with PTX mma.m16n8k16, which distributes
// 8 halves/lane for A and 4 floats/lane for C. That mismatch is why the GEMM
// inner loop is rewritten around these calls rather than shimmed per-primitive.
// =============================================================================

namespace rdna_wmma
{

// Load A fragment from a row-major matrix.
// Lane L loads row (L % 16), WMMA_K=16 columns from the pointer. The caller
// bakes the sub-K column offset into A, so this is always 16 consecutive halves
// -- one GLOBAL_LOAD_DWORDX8 rather than 16 scalar loads.
// Alignment: this dereferences a 32-byte vector type, so it is worth being
// precise about why that is safe rather than assuming it.
//   - The dynamic LDS base (`extern __shared__`) is 32-byte aligned, and
//     row*stride offsets are multiples of WMMA_K (16 halves = 32 bytes), so
//     aligned accesses are the normal case.
//   - Misaligning A by 16, 8, 4 or 2 bytes does not fault: the backend splits
//     the access rather than requiring natural alignment.
// So the failure mode if a future caller breaks alignment is a slower split
// access, not a fault.
__device__ __forceinline__ void load_matrix_a
(
    WmmaFragA& frag,
    const half* A,
    int stride  // stride = K (columns per row in the tile)
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    // half == _Float16 on AMD; single vector load replaces 16 scalar loads
    frag.data = *((const half16_t*)(A + row * stride));
}

// Load B fragment from a row-major matrix.
// B is [K, N] and lane L takes column (L % 16) -- strided, so no vector load.
__device__ __forceinline__ void load_matrix_b
(
    WmmaFragB& frag,
    const half* B,
    int stride  // stride = N (number of columns in B)
)
{
    int lane = threadIdx.x & 31;
    int col = lane % 16;
    // Direct cast: half == _Float16 on AMD, avoids an f16->f32->f16 round trip.
    // Keep the source const-qualified -- the previous form cast it away, which
    // a C-style cast permits silently and which would let a future edit write
    // through a pointer the caller handed over as read-only.
    const _Float16* src = reinterpret_cast<const _Float16*>(B);
    #pragma unroll
    for (int k = 0; k < 16; k++)
    {
        ((_Float16*)&frag.data)[k] = src[k * stride + col];
    }
}

// Matrix multiply-accumulate. Operand order is (B, A, C) -- see the note above.
__device__ __forceinline__ void mma_sync
(
    WmmaFragC& c,
    const WmmaFragA& a,
    const WmmaFragB& b
)
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    // RDNA4: the gfx11 encoding does not exist, so use the gfx12 form. This
    // wrapper is reached by the dense quantized GEMM (exl3_gemm_inner_rdna.cuh,
    // every EXL3 matmul past the GEMV row limit) as well as the fused-MoE comp
    // units.
    //
    // gfx12 v_wmma_f32_16x16x16_f16 takes 8 halves per lane: lane L holds its
    // row/column (L % 16) for K = 8*(L/16) .. +7. The gfx11 fragments carry all
    // 16 K values in every lane, so each operand is a per-lane half select. The
    // accumulator differs only in which columns a lane pair (L, L^16) holds:
    // gfx11 col = 2*i + h, gfx12 col = i + 8*h (h = L / 16). Four xor-16 swaps
    // convert in and four convert back, so WmmaFragC's layout and every
    // load/store helper stay unchanged. Not optimal: a native gfx12 fragment
    // layout would drop the swaps.
    //
    // Layout validated on gfx1201 against an fp64 reference.
    //
    // Other wrappers in this header keep the bare gfx11 builtin on purpose:
    // if a future instantiation drags them into a gfx12 build, a LOUD compile
    // failure is the correct behavior.
    typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
    const bool h = (threadIdx.x & 16) != 0;
    const half8_t a8 = h ? a.data.hi : a.data.lo;
    const half8_t b8 = h ? b.data.hi : b.data.lo;

    float8_t g;
    #pragma unroll
    for (int t = 0; t < 4; t++)
    {
        float r = __shfl_xor(h ? c.data[t] : c.data[4 + t], 16);
        g[2 * t]     = h ? r : c.data[t];
        g[2 * t + 1] = h ? c.data[4 + t] : r;
    }

    g = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b8, a8, g);

    #pragma unroll
    for (int t = 0; t < 4; t++)
    {
        float r = __shfl_xor(h ? g[2 * t] : g[2 * t + 1], 16);
        c.data[t]     = h ? r : g[2 * t];
        c.data[4 + t] = h ? g[2 * t + 1] : r;
    }
#else
    c.data = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b.data, a.data, c.data);
#endif
}

// Load and accumulate C (FP32) -- no bounds checking
__device__ __forceinline__ void load_accumulate_c
(
    WmmaFragC& frag,
    const float* C,
    int stride
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    const float* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int col = i * 2 + col_base;
        frag[i] += row_ptr[col];
    }
}

// Load and accumulate C (FP16 -> FP32) -- no bounds checking
__device__ __forceinline__ void load_accumulate_c_half
(
    WmmaFragC& frag,
    const half* C,
    int stride
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    const half* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int col = i * 2 + col_base;
        frag[i] += __half2float(row_ptr[col]);
    }
}

// Load and accumulate C (FP32) -- with bounds checking
__device__ __forceinline__ void load_accumulate_c_checked
(
    WmmaFragC& frag,
    const float* C,
    int stride,
    int valid_rows
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows)
    {
        const float* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            int col = i * 2 + col_base;
            frag[i] += row_ptr[col];
        }
    }
}

// Load and accumulate C (FP16 -> FP32) -- with bounds checking
__device__ __forceinline__ void load_accumulate_c_half_checked
(
    WmmaFragC& frag,
    const half* C,
    int stride,
    int valid_rows
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows)
    {
        const half* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            int col = i * 2 + col_base;
            frag[i] += __half2float(row_ptr[col]);
        }
    }
}

// Store C fragment with bounds checking (FP32)
__device__ __forceinline__ void store_matrix_c_checked
(
    float* C,
    const WmmaFragC& frag,
    int stride,
    int valid_rows,
    int valid_cols
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows)
    {
        float* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            int col = i * 2 + col_base;
            if (col < valid_cols) row_ptr[col] = frag[i];
        }
    }
}

// Store C fragment with bounds checking (FP32 -> FP16)
__device__ __forceinline__ void store_matrix_c_half_checked
(
    half* C,
    const WmmaFragC& frag,
    int stride,
    int valid_rows,
    int valid_cols
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows)
    {
        half* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            int col = i * 2 + col_base;
            if (col < valid_cols) row_ptr[col] = __float2half(frag[i]);
        }
    }
}

// Store C fragment (FP32, no bounds checking)
__device__ __forceinline__ void store_matrix_c
(
    float* C,
    const WmmaFragC& frag,
    int stride
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    float* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int col = i * 2 + col_base;
        row_ptr[col] = frag[i];
    }
}

// Store C fragment (FP32 -> FP16, no bounds checking)
__device__ __forceinline__ void store_matrix_c_half
(
    half* C,
    const WmmaFragC& frag,
    int stride
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    half* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int col = i * 2 + col_base;
        row_ptr[col] = __float2half(frag[i]);
    }
}

// =============================================================================
// bf16 WMMA -- v_wmma_f32_16x16x16_bf16
// =============================================================================
//
// Operand order is (B, A, C), same as every other variant -- verified against
// a CPU reference: the swapped order gives transpose-matches, which is the
// signature of an operand-order bug rather than a layout one.
//
// A, B and C layouts are identical to the f16 form, and C is fp32, so this
// reuses WmmaFragC and every existing store/accumulate helper.
//
// Parameters take __hip_bfloat16 (== __nv_bfloat16 under HIP) so call sites
// need no conversion.

__device__ __forceinline__ void load_matrix_a_bf16
(
    WmmaFragA_bf16& frag,
    const __hip_bfloat16* A,
    int stride
)
{
    int lane = threadIdx.x & 31;
    frag.data = *((const bf16x16_t*)(A + (lane % 16) * stride));
}

__device__ __forceinline__ void load_matrix_b_bf16
(
    WmmaFragB_bf16& frag,
    const __hip_bfloat16* B,
    int stride
)
{
    int lane = threadIdx.x & 31;
    int col = lane % 16;
    const unsigned short* src = reinterpret_cast<const unsigned short*>(B);
    #pragma unroll
    for (int k = 0; k < 16; k++)
        ((unsigned short*)&frag.data)[k] = src[k * stride + col];
}

// C (fp32) += A x B, both bf16.
__device__ __forceinline__ void mma_sync_bf16
(
    WmmaFragC& c,
    const WmmaFragA_bf16& a,
    const WmmaFragB_bf16& b
)
{
    c.data = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b.data, a.data, c.data);
}

// =============================================================================
// fp16-accumulate WMMA -- v_wmma_f16_16x16x16_f16
// =============================================================================
//
// Same A and B fragments as the f32 form, and the same (row, col) mapping for
// the result: row = lane % 16, col_base = (lane >= 16) ? 1 : 0, element i at
// column i*2 + col_base. The only difference is that the accumulator is packed
// into every other half-slot, chosen by opsel. Derived empirically against a
// CPU reference, not inferred.
//
// opsel is a template parameter because the instruction encodes it as an
// immediate -- it cannot be a runtime value.

template <bool opsel = false>
__device__ __forceinline__ void mma_sync_f16
(
    WmmaFragC_f16& c,
    const WmmaFragA& a,
    const WmmaFragB& b
)
{
    c.data = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(b.data, a.data, c.data, opsel);
}

template <bool opsel = false>
__device__ __forceinline__ void store_matrix_c_f16
(
    half* C,
    const WmmaFragC_f16& frag,
    int stride
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    half* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++) row_ptr[i * 2 + col_base] = frag.get<opsel>(i);
}

template <bool opsel = false>
__device__ __forceinline__ void store_matrix_c_f16_checked
(
    half* C,
    const WmmaFragC_f16& frag,
    int stride,
    int valid_rows,
    int valid_cols
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows)
    {
        half* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            int col = i * 2 + col_base;
            if (col < valid_cols) row_ptr[col] = frag.get<opsel>(i);
        }
    }
}

// =============================================================================
// int8 WMMA -- v_wmma_i32_16x16x16_iu8
// =============================================================================
//
// gfx1151 has a native int8 tensor-core path. It is the natural target for the
// EXL3 int8 GEMV, which otherwise builds its products out of dp4a/sudot4 chains.
//
// Layout matches the f16 path exactly, verified the same way (CPU reference,
// non-symmetric operands):
//   A fragment: lane L holds row (L % 16), 16 consecutive k as int8  -> 4 VGPRs
//   B fragment: lane L holds column (L % 16), 16 k strided           -> 4 VGPRs
//   C fragment: row = L % 16, col_base = (L >= 16) ? 1 : 0,
//               8 int32 at columns [i*2 + col_base]                  -> 8 VGPRs
//
// TRAP, and the reason these wrappers exist rather than calling the builtin
// directly: the builtin is
//
//   __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(s0, v0, s1, v1, C, clamp)
//
// where each sign flag pairs with the vector argument that FOLLOWS it -- and
// because the operand order is (B, A, C), the *first* flag describes B, not A.
// Writing (signed_a, ..., signed_b, ...) reads naturally and is wrong:
// flags(1,0) computes signed-B times unsigned-A. mma_sync_i8 below
// takes (a, b) in the natural order and does the swap internally.
//
// The flags select sign interpretation of the *stored bytes*; the loaders copy
// bytes verbatim, so signedness is a property of the multiply, not the load.

// Lane L loads row (L % 16): 16 consecutive bytes, one 16-byte load.
__device__ __forceinline__ void load_matrix_a_i8
(
    WmmaFragA_i8& frag,
    const int8_t* A,
    int stride   // stride = K, in bytes
)
{
    int lane = threadIdx.x & 31;
    frag.data = *((const int32x4_t*)(A + (lane % 16) * stride));
}

// B is [K, N]; lane L takes column (L % 16), so the reads are strided.
__device__ __forceinline__ void load_matrix_b_i8
(
    WmmaFragB_i8& frag,
    const int8_t* B,
    int stride   // stride = N, in bytes
)
{
    int lane = threadIdx.x & 31;
    int col = lane % 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) ((int8_t*)&frag.data)[k] = B[k * stride + col];
}

// C += A x B. signed_a / signed_b describe A and B respectively; the swap onto
// the builtin's flag order happens here so call sites cannot get it wrong.
template <bool signed_a = true, bool signed_b = true, bool clamp = false>
__device__ __forceinline__ void mma_sync_i8
(
    WmmaFragC_i32& c,
    const WmmaFragA_i8& a,
    const WmmaFragB_i8& b
)
{
    c.data = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32
    (
        signed_b, b.data,     // first flag/vector pair is B
        signed_a, a.data,
        c.data, clamp
    );
}

__device__ __forceinline__ void store_matrix_c_i32
(
    int* C,
    const WmmaFragC_i32& frag,
    int stride
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    int* row_ptr = C + row * stride;
    #pragma unroll
    for (int i = 0; i < 8; i++) row_ptr[i * 2 + col_base] = frag[i];
}

__device__ __forceinline__ void store_matrix_c_i32_checked
(
    int* C,
    const WmmaFragC_i32& frag,
    int stride,
    int valid_rows,
    int valid_cols
)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_base = (lane >= 16) ? 1 : 0;

    if (row < valid_rows)
    {
        int* row_ptr = C + row * stride;
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            int col = i * 2 + col_base;
            if (col < valid_cols) row_ptr[col] = frag[i];
        }
    }
}

} // namespace rdna_wmma

// =============================================================================
// Memory fence
// =============================================================================
//
// Wave-local s_waitcnt(0) -- forces this wave's outstanding VMEM/LDS operations
// to retire. Always pair with __syncthreads() when cross-thread visibility is
// required: the block barrier provides the actual synchronization, and this
// only ensures the wave's loads/stores have committed before it is reached.
//
// Cross-CU atomic synchronization uses __threadfence() (device scope), not this.

__device__ __forceinline__ void mem_fence()
{
    __builtin_amdgcn_s_waitcnt(0);
}

