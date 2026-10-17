// =============================================================================
// Multi-matrix (expert-batched) GEMV for RDNA at m == 1
// =============================================================================
//
// exl3_mgemm's cooperative kernel is structurally capped in occupancy at
// m == 1: grid.sync() requires every block co-resident, which limits the grid
// to the co-resident workgroup count regardless of how much parallel work
// exists. The plain-launch GEMV reaches the memory roofline on the same
// hardware, but it is single-matrix -- and a single 3072->1024 expert yields
// only 8 blocks, which loses to the tile GEMM.
//
// The multi-matrix form dissolves that trade-off: bszm experts x (size_n / 16)
// tiles gives hundreds to thousands of warps for typical MoE shapes,
// oversubscribing every SIMD without any cooperative launch. Three
// plain kernels replace the one cooperative one:
//
//   1. exl3_mgemv_dot_kernel      the GEMV proper, one warp per (expert, tile);
//      (and _splitk)              per-warp core shared verbatim with the
//                                 single-matrix kernel (exl3_gemv_dot_tile).
//                                 Its prologue rotates the expert's input
//                                 (SUH + 128-point Hadamard) into LDS, block-
//                                 cooperatively, and hosts the graph-patch
//                                 sites -- see below
//                                 Its epilogue, run by the last-arriving warp
//                                 of each 128-wide output segment, is the
//                                 output rotation and the weighted-sum
//                                 reduction -- see "Launch-count fusion"
//   2. exl3_mgemv_had_out_kernel  per-expert output rotation (SVH), folding in
//                                 the routing weight exactly as the cooperative
//                                 kernel does (scale * weights[j]) -- fallback
//                                 form, EXL3_GEMV_FUSE_OUT=0
//   3. exl3_mgemv_reduce_kernel   the weighted-sum epilogue, mirroring the
//                                 cooperative kernel's num_tokens grouping --
//                                 fallback form, as above
//
// Launch-count fusion. The input rotation used to be its own kernel
// (exl3_mgemv_had_in_kernel, writing A_had in global memory for the dot
// kernel to read). The command processor's gap between dependent kernels is
// paid per launch on every runtime (graph replay does not remove it), so the
// rotation moved into the dot kernel's prologue. Each block
// rotates the full K of its expert into LDS (redundantly across the N-tiles of
// that expert; A is m x K halfs, L2-resident, and the 128-point Hadamard is a
// few shuffle stages, so the redundancy is cheap) and the dot
// core reads A from LDS instead of global. The arithmetic is the same
// function, in the same order, so the output is bit-identical to the
// three-kernel form. A_had is no longer written by this path.
//
// The output side is fused the same way. Each 128-wide rotation segment of an
// expert's output is produced by 8 N-tiles -- 8 warps of one block in the
// single-warp form, 8 blocks in the split-K form. After storing its 16 outputs
// a warp fences and increments the segment's arrival counter (per segment,
// not per block, so no tile-to-block alignment is assumed); the warp that
// observes 7 earlier arrivals is the last, and it acquires, resets the counter
// and applies svh + Hadamard + scale to the segment in place, on the same
// stored (fp16-rounded, for half C) values and with the same arithmetic as
// exl3_mgemv_had_out_kernel. When the call carries routing weights, a second
// counter per segment counts rotated slots, and the last of the packed slots
// runs the grouped reduction for its 128 columns in exl3_mgemv_reduce_kernel's
// exact order. Counters live in the per-device parameter block: zeroed at
// allocation, reset by their own last arriver, so no per-launch memset and no
// graph patch site. This is the classic last-block pattern (release fence
// before the increment, acquire fence after observing the final count); it is
// wait-free, so it cannot hang. EXL3_GEMV_FUSE_OUT=0 restores the separate
// rotation and reduction kernels, as does a shape whose counters would not
// fit (none in practice). A MoE expert call is now one launch instead of
// three (four with weights).
//
// Graph capture. exl3_mgemm records four patchable parameters per launch
// (A, C, indices, weights -- add_graph_args in quant/exl3_gemm.cu), and
// Graph::launch() patches ONE site per caller-supplied parameter, walking sites
// and params in lockstep. A multi-kernel path therefore cannot record the same
// parameter on several kernels and expect all copies patched. Instead all four
// sites are recorded on the dot kernel, which takes all four as its first four
// arguments (weights exists there only to host the patch site) and republishes
// the ones the downstream kernels need through a per-device Exl3MgemvParams
// block. Downstream kernels read C/indices/weights from that block, so a
// patched prologue re-propagates automatically on every replay. Site order
// (A, C, indices, weights, GP_end) matches the cooperative path, which is what
// the callers' params subsequences are written against. The dot
// kernel is a template, so the sites are recorded against the instantiation
// that was launched (exl3_mgemv_record_sites at the launch macro).
//
// Index packing (min_index >= 0). The cooperative kernel packs indices within
// [min_index, max_index) into a device global and processes only the packed
// slots; C rows, A_had slabs and the reduce stride all follow PACKED slot
// numbering. BC_BlockSparseMLP passes a real expert range at decode, so this
// path must reproduce that or it would never run. There is no grid.sync() here
// to broadcast a packed list, so each warp/block derives its own view with an
// O(bszm) scan of the indices tensor (bszm <= 128, cached, negligible next to
// the 128-wide hadamard or a full-K dot). Semantics are identical: slot j means
// "the j-th index inside the range", weights follow their original position,
// and the reduce stride is the packed count.
//
// Per-matrix width/output lists (size_n_list / c_list). DS4's bc_dsa fan and
// fan2 sites project ONE input through several matrices of differing widths in
// a single call: size_n is then only the max width, matrix mat_index is
// size_n_list[mat_index] wide, and its output goes to its own tensor at
// c_list[mat_index] (row 0 -- this whole path is m == 1) instead of C's j-th
// row. The entry point enforces no-weights/no-packing/single-token for list
// calls, so the reduce epilogue never runs with lists. Both list pointers are
// device arrays whose contents the kernels read per launch -- the same
// indirection B_list already uses -- so list calls are graph-stable without
// new patch sites. The grid covers the widest matrix; warps (or split-K
// blocks, which are tile-uniform) beyond a narrower matrix's width exit early,
// and the dot core sees the matrix's own n_tiles because B is packed at that
// width.
// =============================================================================

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../../graph.cuh"
#include "exl3_mgemv_rdna.cuh"
#include "exl3_gemv_kernel_rdna.cuh"
#include "exl3_gemv_multirow_rdna.cuh"

#include <cstdlib>
#include <mutex>

// -----------------------------------------------------------------------------
// Kill switch: EXL3_MGEMV = 0 sends every call back to the cooperative
// exl3_mgemm. Re-read on every call so a single process can switch between
// the two paths.
// -----------------------------------------------------------------------------
static bool exl3_mgemv_enabled()
{
    const char* env = std::getenv("EXL3_MGEMV");
    if (!env) return true;
    return atoi(env) != 0;
}

// -----------------------------------------------------------------------------
// Per-device parameter block
// -----------------------------------------------------------------------------
// cudaMalloc'd lazily. The first mgemm call of a BC module runs eagerly (the
// graph records only on the second pass), so allocation normally happens
// outside capture; cudaMalloc is stream-less and remains legal if a first call
// ever does arrive mid-capture.
// allow_alloc must be false while a graph is capturing: hipMalloc synchronizes
// the device, which invalidates an active stream capture. The eager first pass
// every BC module runs before capturing (graph == nullptr there) is where
// allocation happens; a capture arriving on a cold device declines to the
// cooperative kernel rather than allocating.
static Exl3MgemvParams* exl3_mgemv_param_block(int device, bool allow_alloc)
{
    static std::mutex mtx;
    static Exl3MgemvParams* blocks[64] = {};
    if (device < 0 || device >= 64) return nullptr;
    std::lock_guard<std::mutex> lock(mtx);
    if (!blocks[device] && allow_alloc)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(Exl3MgemvParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return nullptr;
        }
        // The arrival counters must start at zero; they self-reset after that
        if (cudaMemset(p, 0, sizeof(Exl3MgemvParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            cudaFree(p);
            return nullptr;
        }
        blocks[device] = (Exl3MgemvParams*) p;
    }
    return blocks[device];
}

// Slot resolution (exl3_mgemv_mat_index / exl3_mgemv_packed_count) lives in
// exl3_gemv_kernel_rdna.cuh since the multi-row path shares it.

// =============================================================================
// Kernel-side glue for the fused epilogue (helpers: exl3_gemv_kernel_rdna.cuh)
// =============================================================================
// Resolves the per-matrix inputs from the launch arguments. Shared by both dot
// kernels; entered by the tile's warp.
template <bool c_fp32>
__device__ __forceinline__ void exl3_mgemv_store_or_fuse
(
    const float accum,
    const bool fuse_out,
    void* __restrict__ C,
    void* const* __restrict__ c_list,
    const uintptr_t* __restrict__ svh_list,
    const half* __restrict__ weights,
    const int64_t* __restrict__ indices,
    Exl3MgemvParams* __restrict__ pb,
    const int size_n,
    const int bszm,
    const int min_index,
    const int max_index,
    const int num_tokens,
    const int j,
    const int mat_index,
    const int orig_pos,
    const int tile_n,
    const int lane
)
{
    void* Cb = c_list ? c_list[mat_index] : C;
    const int64_t row_off = c_list ? 0 : (int64_t) j * size_n;

    if (!fuse_out)
    {
        if (lane < 16)
        {
            const int64_t out_idx = row_off + tile_n * 16 + lane;
            if constexpr (c_fp32)
                ((float*) Cb)[out_idx] = accum;
            else
                ((half*) Cb)[out_idx] = __float2half(accum);
        }
        return;
    }

    float scale = 0.088388347648f;  // 1/sqrt(128)
    if (weights) scale *= __half2float(weights[orig_pos]);
    const int n_segs = size_n / 128;
    int* seg_counter = pb->seg_counters + j * n_segs + tile_n / 8;
    int* red_counter = weights ? pb->red_counters + tile_n / 8 : nullptr;
    const int red_target = weights
        ? exl3_mgemv_packed_count(indices, bszm, min_index, max_index) : 0;

    exl3_gemv_fused_epilogue<c_fp32>
    (
        accum, Cb, row_off, size_n, tile_n, lane,
        (const half*) svh_list[mat_index], scale,
        seg_counter, red_counter, red_target, num_tokens
    );
}

// =============================================================================
// Kernel 1: the GEMV proper, with the input rotation as its prologue
// =============================================================================
// Grid (ceil(n_tiles / WARPS_PER_BLOCK), bszm); one warp per (expert, N-tile).
// The per-warp body is exl3_gemv_dot_tile, shared with the single-matrix GEMV.
//
// Argument order 0-3 is load-bearing: these are the graph patch sites, recorded
// as (A, C, indices, weights) to match the cooperative path's site order.
//
// Every early exit that precedes the __syncthreads after the rotation is
// block-uniform (j is the block's y); the per-warp tile exits come after it.

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
__global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_mgemv_dot_kernel
(
    const half* __restrict__ A,           // 0: GP_mgemm_A
    void* __restrict__ C,                 // 1: GP_mgemm_C
    const int64_t* __restrict__ indices,  // 2: GP_mgemm_indices
    const half* __restrict__ weights,     // 3: GP_mgemm_weights (patch host only)
    Exl3MgemvParams* __restrict__ pb,
    const uintptr_t* __restrict__ B_list,
    const uintptr_t* __restrict__ suh_list,
    const int size_k,
    const int size_n,
    const int bszm_in,
    const int bszm,
    const int min_index,
    const int max_index,
    const int* __restrict__ size_n_list,  // per-matrix widths; size_n is the max
    void* const* __restrict__ c_list,     // per-matrix output bases (row 0; m == 1)
    const uintptr_t* __restrict__ svh_list,
    const int num_tokens,
    const int core,
    const bool fuse_out
)
{
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
        pb->C = C;
        pb->indices = indices;
        pb->weights = weights;
    }

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.y;
    const int n_tiles = size_n / 16;

    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;   // block-uniform

    // Dynamic shared memory -- the single-matrix carve, then the rotated input
    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    half* sh_a = (half*) (sh_b_quant + WARPS_PER_BLOCK * EXL3_GEMV_SH_QUANT_U16);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    // Prologue: rotate this expert's input into LDS
    {
        const half* suh = (const half*) suh_list[mat_index];
        const half* A_src = bszm_in == 1 ? A : A + (int64_t) j * size_k;
        exl3_gemv_rotate_in<WARPS_PER_BLOCK>(A_src, suh, sh_a, size_k, warp_id, lane);
    }
    __syncthreads();

    const int tile_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (tile_n >= n_tiles) return;

    // Per-matrix width (the cooperative kernel's n_j): the grid covers the
    // widest matrix, warps beyond a narrower one exit here. B is packed at the
    // matrix's own width, so the dot core must see n_tiles_j, not the max.
    const int n_tiles_j = size_n_list ? size_n_list[mat_index] / 16 : n_tiles;
    if (tile_n >= n_tiles_j) return;

    const uint16_t* B = (const uint16_t*) B_list[mat_index];

    float accum = exl3_gemv_dot_tile_sel<bits, cb, true, true>
    (
        core, sh_a, B, size_k, n_tiles_j, tile_n, lane, my_sh_b, my_sh_b_quant, 0, size_k / 16
    );

    // Store, or the fused rotation / reduction epilogue (each matrix owns its
    // own output tensor under c_list, with no j stride -- m == 1)
    exl3_mgemv_store_or_fuse<c_fp32>
    (
        accum, fuse_out, C, c_list, svh_list, weights, indices, pb,
        size_n, bszm, min_index, max_index, num_tokens,
        j, mat_index, orig_pos, tile_n, lane
    );
}

// Split-K form of kernel 1: one block per (expert, N-tile), warps share K.
// The expert shapes this path exists for are exactly the narrow-output ones
// the single-warp form starves on (64 tiles at 1024 wide), so in practice
// this is the form MoE decode takes; the threshold keeps parity with the
// single-matrix sites. Same argument contract as the single-warp form.
template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
__global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_mgemv_dot_kernel_splitk
(
    const half* __restrict__ A,           // 0: GP_mgemm_A
    void* __restrict__ C,                 // 1: GP_mgemm_C
    const int64_t* __restrict__ indices,  // 2: GP_mgemm_indices
    const half* __restrict__ weights,     // 3: GP_mgemm_weights (patch host only)
    Exl3MgemvParams* __restrict__ pb,
    const uintptr_t* __restrict__ B_list,
    const uintptr_t* __restrict__ suh_list,
    const int size_k,
    const int size_n,
    const int bszm_in,
    const int bszm,
    const int min_index,
    const int max_index,
    const int* __restrict__ size_n_list,  // per-matrix widths; size_n is the max
    void* const* __restrict__ c_list,     // per-matrix output bases (row 0; m == 1)
    const uintptr_t* __restrict__ svh_list,
    const int num_tokens,
    const int core,
    const bool fuse_out,
    const int tpb                         // N-tiles per block (exl3_gemv_tiles_tpb)
)
{
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    {
        pb->C = C;
        pb->indices = indices;
        pb->weights = weights;
    }

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.y;
    const int tile_n = blockIdx.x * tpb;
    const int n_tiles = size_n / 16;

    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;   // whole block exits together; no sync hazard

    // Per-matrix width: tile_n and mat_index are block-uniform, so the whole
    // block exits together here too and the __syncthreads below are safe
    const int n_tiles_j = size_n_list ? size_n_list[mat_index] / 16 : n_tiles;
    if (tile_n >= n_tiles_j) return;

    const uint16_t* B = (const uint16_t*) B_list[mat_index];

    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    float* sh_red = (float*) (sh_b_quant + WARPS_PER_BLOCK * EXL3_GEMV_SH_QUANT_U16);
    half* sh_a = (half*) (sh_red + WARPS_PER_BLOCK * 16 * EXL3_GEMV_TILES_TMAX);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    // Prologue: rotate this expert's input into LDS
    {
        const half* suh = (const half*) suh_list[mat_index];
        const half* A_src = bszm_in == 1 ? A : A + (int64_t) j * size_k;
        exl3_gemv_rotate_in<WARPS_PER_BLOCK>(A_src, suh, sh_a, size_k, warp_id, lane);
    }
    __syncthreads();

    // tpb adjacent N-tiles per block (tiles core; 1 otherwise), each with the
    // per-tile chain and reduction order of the one-tile form
    float accum[EXL3_GEMV_TILES_TMAX];
    exl3_gemv_dot_tile_splitk_t<bits, cb, WARPS_PER_BLOCK, 1, true, EXL3_GEMV_TILES_TMAX>
    (
        core, tpb, sh_a, 0, B, size_k, n_tiles_j, tile_n, warp_id, lane, sh_red, accum,
        [&](int tn, int k0, int k1, float* o)
        {
            o[0] = core == EXL3_GEMV_CORE_LDS
                ? exl3_gemv_dot_tile<bits, cb>(sh_a, B, size_k, n_tiles_j, tn, lane, my_sh_b, my_sh_b_quant, k0, k1)
                : exl3_gemv_dot_tile_direct<bits, cb>(sh_a, B, n_tiles_j, tn, lane, k0, k1);
        }
    );

    if (warp_id == 0)
        for (int t = 0; t < tpb; ++t)
            exl3_mgemv_store_or_fuse<c_fp32>
            (
                accum[t], fuse_out, C, c_list, svh_list, weights, indices, pb,
                size_n, bszm, min_index, max_index, num_tokens,
                j, mat_index, orig_pos, tile_n + t, lane
            );
}

// =============================================================================
// Kernel 2: per-expert output rotation, weight folded into the scale
// =============================================================================
// Fallback form (EXL3_GEMV_FUSE_OUT=0, or counters that would not fit); the
// dot kernels' fused epilogue reproduces this bit for bit.

template <bool c_fp32>
__global__
__launch_bounds__(256)
void exl3_mgemv_had_out_kernel
(
    const Exl3MgemvParams* __restrict__ pb,
    const uintptr_t* __restrict__ svh_list,
    const int size_n,
    const int bszm,
    const int min_index,
    const int max_index,
    const int* __restrict__ size_n_list,  // per-matrix widths; size_n is the max
    void* const* __restrict__ c_list      // per-matrix output bases (row 0; m == 1)
)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int n_warps = size_n / 128;
    if (warp_id >= bszm * n_warps) return;

    int j   = warp_id / n_warps;
    int seg = warp_id % n_warps;

    const int64_t* indices = pb->indices;
    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;

    // Per-matrix width: warps beyond a narrower matrix's rotation blocks exit.
    // EXL3 trellis padding guarantees n_j % 128 == 0 (the cooperative kernel's
    // output rotation depends on the same).
    const int n_j = size_n_list ? size_n_list[mat_index] : size_n;
    if (seg * 128 >= n_j) return;

    const half* svh = (const half*) svh_list[mat_index];

    float scale = 0.088388347648f;  // 1/sqrt(128)
    if (pb->weights) scale *= __half2float(pb->weights[orig_pos]);

    void* Cb = c_list ? c_list[mat_index] : pb->C;
    int64_t offset = (c_list ? 0 : (int64_t) j * size_n) + seg * 128;

    if constexpr (c_fp32)
        had_ff_r_128_inner<false, true>
        (
            ((const float*) Cb) + offset,
            ((float*) Cb) + offset,
            svh + seg * 128,
            scale
        );
    else
        had_hf_r_128_inner<false, true>
        (
            ((const half*) Cb) + offset,
            ((half*) Cb) + offset,
            svh + seg * 128,
            scale
        );
}

// =============================================================================
// Kernel 3: weighted-sum epilogue
// =============================================================================
// Fallback form, as kernel 2.
// Mirrors the cooperative kernel's reduction (m == 1 form): each of the
// num_tokens groups of (packed / num_tokens) contiguous slots sums into its own
// output row. Accumulation dtype matches the cooperative kernel (half chain for
// fp16 C) so the two paths stay numerically comparable. One thread owns a
// column across every t, preserving the increasing-t ordering the in-place
// overwrite depends on.

template <bool c_fp32>
__global__
__launch_bounds__(256)
void exl3_mgemv_reduce_kernel
(
    const Exl3MgemvParams* __restrict__ pb,
    const int size_n,
    const int num_tokens,
    const int bszm,
    const int min_index,
    const int max_index
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= size_n) return;

    int packed = exl3_mgemv_packed_count(pb->indices, bszm, min_index, max_index);
    int stride = packed / num_tokens;

    for (int t = 0; t < num_tokens; ++t)
    {
        if constexpr (c_fp32)
        {
            float* C_ = ((float*) pb->C) + (int64_t) t * stride * size_n + col;
            float sum = 0.0f;
            for (int jj = 0; jj < stride; ++jj)
            {
                sum += *C_;
                C_ += size_n;
            }
            ((float*) pb->C)[(int64_t) t * size_n + col] = sum;
        }
        else
        {
            half* C_ = ((half*) pb->C) + (int64_t) t * stride * size_n + col;
            half sum = {};
            for (int jj = 0; jj < stride; ++jj)
            {
                sum = __hadd(sum, *C_);
                C_ += size_n;
            }
            ((half*) pb->C)[(int64_t) t * size_n + col] = sum;
        }
    }
}

// =============================================================================
// Host dispatch
// =============================================================================

// Graph patch sites, recorded against the dot-kernel instantiation that was
// launched. Site order matches the cooperative path: A, C, indices, weights.
static void exl3_mgemv_record_sites(Graph* graph, void* k)
{
    graph->record_param(k, GP_mgemm_A, 0);
    graph->record_param(k, GP_mgemm_C, 1);
    graph->record_param(k, GP_mgemm_indices, 2);
    graph->record_param(k, GP_mgemm_weights, 3);
    graph->record_param(k, GP_end, 0);
}

// One warp per (expert, 16-wide tile); 8 warps per block. The per-expert tile
// counts this path exists for (64 at 1024-wide, 192 at 3072-wide) are multiples
// of 8, so no partial blocks in practice; partial blocks are handled anyway.
#define EXL3_MGEMV_WARPS 8

bool exl3_mgemv_try_launch
(
    const half* A_ptr,
    const uintptr_t* B_ptr_ptr,
    void* C_ptr,
    const uintptr_t* suh_ptr_ptr,
    half* A_had_ptr,
    const uintptr_t* svh_ptr_ptr,
    const int64_t* indices_ptr,
    const half* weights_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int bszm_in,
    int bszm_out,
    int min_index,
    int max_index,
    int num_tokens,
    const int* size_n_list,
    void** c_list,
    int device,
    cudaStream_t stream,
    Graph* graph
)
{
    // Hard constraints first, then the kill switch. m == 1 and the 128-multiple
    // dims are kernel requirements (hadamard blocks). Per-matrix width/output
    // lists (DS4's bc_dsa fan sites) are supported under the same constraints
    // the entry point enforces for them -- no weights, no packing, single
    // token; size_n is then the max width, per-matrix widths come from the
    // device-side list and are %16/%128 by trellis padding. Both lists arrive
    // together or not at all; a half-supplied pair falls through.
    if (size_m != 1) return false;
    if (K < 1 || K > 8) return false;
    if (size_k % 128 || size_n % 128) return false;
    if ((size_n_list != nullptr) != (c_list != nullptr)) return false;
    if (size_n_list && (weights_ptr || num_tokens != 1 || min_index >= 0)) return false;
    // num_tokens > 1 is legal WITH range filtering: the cooperative kernel uses
    // position-preserving masking (out-of-range slots -> -1 in place). This path still packs (compacts), which collapses the per-token slot runs
    // the grouped reduce below divides by (stride = packed / num_tokens) -- wrong
    // grouping, and packed need not even divide. Decline and let the cooperative
    // kernel handle it; only TP-sharded / CPU-split expert maps hit this combination.
    if (num_tokens > 1 && min_index >= 0) return false;
    if (!exl3_mgemv_enabled()) return false;

    // 128 = MAX_INDICES, the cooperative kernel's packing-array bound; kept for
    // parity even though this path scans rather than packs
    int bszm = MAX(bszm_in, bszm_out);
    if (bszm < 1 || bszm > 128) return false;
    if (num_tokens < 1) return false;

    Exl3MgemvParams* pb = exl3_mgemv_param_block(device, graph == nullptr);
    if (!pb) return false;
    if (!graph) exl3_gemv_multirow_prewarm(device);   // the m = 2..8 path's block

    // The rotated input lives in the dot kernel's LDS; the caller's A_had
    // scratch is neither read nor written by this path.
    (void) A_had_ptr;

    // Fused output epilogue unless switched off or the arrival counters would
    // not fit this shape (bszm x segments, and segments; see the header)
    const int n_segs = size_n / 128;
    const bool fuse_out = exl3_gemv_fuse_out_enabled()
        && bszm * n_segs <= EXL3_MGEMV_SEG_COUNTERS
        && n_segs <= EXL3_MGEMV_RED_COUNTERS;

    // -------------------------------------------------------------------------
    // 1. GEMV, with the input rotation as its prologue -- split-K when the
    //    per-expert output is narrow (same rule as the single-matrix sites),
    //    which for MoE decode it always is
    // -------------------------------------------------------------------------
    {
        int n_tiles = size_n / 16;
        bool splitk = exl3_gemv_splitk_enabled() && n_tiles <= EXL3_GEMV_SPLITK_MAX_TILES;
        const int lds_core = exl3_gemv_core_mode();

        // The split-K wave count is shape-aware (see exl3_gemv_splitk_warps);
        // the grid here is n_tiles x bszm blocks, so bszm enters the rule.
        // It is computed on n_tiles, not the tpb-reduced grid, so the wave
        // count (and the per-tile reduction order) matches the one-tile form.
        // The single-warp form keeps the fixed EXL3_MGEMV_WARPS.
        int warps = splitk ? exl3_gemv_splitk_warps(size_k / 16, n_tiles, bszm)
                           : EXL3_MGEMV_WARPS;
        const int tpb = splitk ? exl3_gemv_tiles_tpb(lds_core, device, n_tiles, bszm, warps) : 1;

        dim3 grid(splitk ? n_tiles / tpb : CEIL_DIVIDE(n_tiles, warps), bszm);
        dim3 block(warps * 32);
        size_t smem = exl3_gemv_smem_bytes_fused(warps, splitk, size_k);

        #define LAUNCH_MGEMV_FP(bits_val, codebook, warps_val, kernel_name, EXTRA_ARGS) \
            if (c_fp32) { \
                hipLaunchKernelGGL( \
                    (kernel_name<bits_val, true, codebook, warps_val>), \
                    grid, block, smem, stream, \
                    A_ptr, C_ptr, indices_ptr, weights_ptr, pb, B_ptr_ptr, suh_ptr_ptr, \
                    size_k, size_n, bszm_in, bszm, min_index, max_index, \
                    size_n_list, c_list, svh_ptr_ptr, num_tokens, lds_core, fuse_out EXTRA_ARGS); \
                if (graph) exl3_mgemv_record_sites(graph, \
                    (void*) &kernel_name<bits_val, true, codebook, warps_val>); \
            } else { \
                hipLaunchKernelGGL( \
                    (kernel_name<bits_val, false, codebook, warps_val>), \
                    grid, block, smem, stream, \
                    A_ptr, C_ptr, indices_ptr, weights_ptr, pb, B_ptr_ptr, suh_ptr_ptr, \
                    size_k, size_n, bszm_in, bszm, min_index, max_index, \
                    size_n_list, c_list, svh_ptr_ptr, num_tokens, lds_core, fuse_out EXTRA_ARGS); \
                if (graph) exl3_mgemv_record_sites(graph, \
                    (void*) &kernel_name<bits_val, false, codebook, warps_val>); \
            }

        #define MGEMV_TPB , tpb
        #define LAUNCH_MGEMV_CB(bits_val, codebook) \
            if (splitk) { \
                switch (warps) { \
                    case 4:  LAUNCH_MGEMV_FP(bits_val, codebook, 4,  exl3_mgemv_dot_kernel_splitk, MGEMV_TPB) break; \
                    case 8:  LAUNCH_MGEMV_FP(bits_val, codebook, 8,  exl3_mgemv_dot_kernel_splitk, MGEMV_TPB) break; \
                    case 16: LAUNCH_MGEMV_FP(bits_val, codebook, 16, exl3_mgemv_dot_kernel_splitk, MGEMV_TPB) break; \
                } \
            } else { \
                LAUNCH_MGEMV_FP(bits_val, codebook, EXL3_MGEMV_WARPS, exl3_mgemv_dot_kernel, ) \
            }

        #define LAUNCH_MGEMV(bits_val) \
            switch (cb) { \
                case 0: LAUNCH_MGEMV_CB(bits_val, 0); break; \
                case 1: LAUNCH_MGEMV_CB(bits_val, 1); break; \
                case 2: LAUNCH_MGEMV_CB(bits_val, 2); break; \
            }

        switch (K) {
            case 1: LAUNCH_MGEMV(1); break;
            case 2: LAUNCH_MGEMV(2); break;
            case 3: LAUNCH_MGEMV(3); break;
            case 4: LAUNCH_MGEMV(4); break;
            case 5: LAUNCH_MGEMV(5); break;
            case 6: LAUNCH_MGEMV(6); break;
            case 7: LAUNCH_MGEMV(7); break;
            case 8: LAUNCH_MGEMV(8); break;
        }

        #undef LAUNCH_MGEMV
        #undef LAUNCH_MGEMV_CB
        #undef LAUNCH_MGEMV_FP
        #undef MGEMV_TPB
    }

    if (fuse_out) return true;

    // -------------------------------------------------------------------------
    // 2. Output rotation (+ routing weight) -- fallback form
    // -------------------------------------------------------------------------
    {
        int n_warps = size_n / 128;
        int blocks = CEIL_DIVIDE(bszm * n_warps * 32, 256);
        if (c_fp32)
            hipLaunchKernelGGL
            (
                exl3_mgemv_had_out_kernel<true>,
                dim3(blocks), dim3(256), 0, stream,
                pb, svh_ptr_ptr, size_n, bszm, min_index, max_index,
                size_n_list, c_list
            );
        else
            hipLaunchKernelGGL
            (
                exl3_mgemv_had_out_kernel<false>,
                dim3(blocks), dim3(256), 0, stream,
                pb, svh_ptr_ptr, size_n, bszm, min_index, max_index,
                size_n_list, c_list
            );
    }

    // -------------------------------------------------------------------------
    // 3. Weighted-sum epilogue -- fallback form (only when the call has routing weights, exactly
    //    like the cooperative kernel's `if (B_weights)` epilogue; whether the
    //    argument is present is fixed per call site, so this is graph-stable)
    // -------------------------------------------------------------------------
    if (weights_ptr)
    {
        int blocks = CEIL_DIVIDE(size_n, 256);
        if (c_fp32)
            hipLaunchKernelGGL
            (
                exl3_mgemv_reduce_kernel<true>,
                dim3(blocks), dim3(256), 0, stream,
                pb, size_n, num_tokens, bszm, min_index, max_index
            );
        else
            hipLaunchKernelGGL
            (
                exl3_mgemv_reduce_kernel<false>,
                dim3(blocks), dim3(256), 0, stream,
                pb, size_n, num_tokens, bszm, min_index, max_index
            );
    }

    return true;
}
