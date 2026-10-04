#pragma once

// =============================================================================
// Multi-matrix (expert-batched) GEMV for RDNA at m == 1 -- host interface
// =============================================================================
//
// See exl3_mgemv_rdna.cu for the design. This header exists so
// quant/exl3_gemm.cu (the exl3_mgemm entry point) can route to the path
// without pulling in the kernel definitions.

#include <cuda_fp16.h>
#include <cstdint>

class Graph;

// Device-side parameter block, one per device. The dot kernel (whose prologue
// is the input rotation) receives the graph-patchable pointers (C, indices,
// weights) as ordinary kernel arguments -- making it the single node whose
// parameters Graph::launch() patches -- and republishes them here for the
// downstream kernels, whose own argument copies would otherwise go stale after
// a patch. In-stream ordering makes the handoff safe: the dot kernel finishes
// before any consumer launches, and sequential
// calls sharing the block cannot interleave on one stream. Two streams mutating
// one device concurrently would race on this block, but they already race on
// DevCtx's shared `locks` buffer in the cooperative path, so this adds no new
// constraint.
//
// The arrival counters of the fused output epilogue live here too (see the
// "Launch-count fusion" section of exl3_mgemv_rdna.cu): seg_counters is
// indexed [slot][segment] with segment = 128-wide output block, red_counters
// by segment. Zeroed when the block is allocated; each counter is reset by
// the warp that observes its final arrival, so a completed launch leaves them
// all zero again. A shape needing more than this many falls back to the
// separate rotation / reduction kernels.
#define EXL3_MGEMV_SEG_COUNTERS (128 * 256)
#define EXL3_MGEMV_RED_COUNTERS 1024
struct Exl3MgemvParams
{
    void* C;
    const int64_t* indices;
    const half* weights;
    int seg_counters[EXL3_MGEMV_SEG_COUNTERS];
    int red_counters[EXL3_MGEMV_RED_COUNTERS];
};

// Single-matrix analogue for the graph-captured exl3_gemm path, implemented in
// exl3_gemv_rdna.cu beside the non-graph dispatch. Same prologue-republish
// trick with the six GP_gemm_* sites; see the comment there.
// The fused output epilogue's per-segment arrival counters live here as in
// Exl3MgemvParams (one matrix, so segments only); same zero-at-allocation,
// self-resetting contract.
#define EXL3_GEMV_SEG_COUNTERS 1024
struct Exl3GemvGraphParams
{
    const uint16_t* B;
    void* C;
    half* A_had;
    const half* svh;
    int seg_counters[EXL3_GEMV_SEG_COUNTERS];
};

bool exl3_gemv_graph_try_launch
(
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    const half* suh_ptr,
    half* A_had_ptr,
    const half* svh_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream,
    Graph* graph
);

// Defined in exl3_gemv_rdna.cu; the EXL3_GEMV_SPLITK kill switch, re-read per
// call, shared by every split-K launch site.
bool exl3_gemv_splitk_enabled();

// Defined in exl3_gemv_rdna.cu; the EXL3_GEMV_FUSE_OUT kill switch for the
// fused output epilogue of both the single- and multi-matrix paths, re-read
// per call.
bool exl3_gemv_fuse_out_enabled();

// Defined in exl3_gemv_rdna.cu; shape-aware split-K wave count (4, 8 or 16),
// shared by the plain, graph and mgemv split-K sites. bszm is the grid's
// matrix-batch factor (1 for the single-matrix sites); EXL3_GEMV_SPLITK_WARPS
// forces one count everywhere.
int exl3_gemv_splitk_warps(int k_tiles, int n_tiles, int bszm);

// Shape-aware GEMV profitability rule, shared by the graph and non-graph
// routing sites. With the in-block split-K form the GEMV beats the cooperative
// GEMM at m == 1 on every shape, so the rule is currently "always" and EXL3_GEMV
// modes 1 and 2 are equivalent; the hook stays for the day a losing shape
// appears. EXL3_GEMV=0 disables the path entirely.
static inline bool exl3_gemv_rdna_pays(int size_k, int size_n)
{
    (void) size_k;
    (void) size_n;
    return true;
}

// Launches the multi-matrix GEMV pipeline (rotate+dot, rotate, reduce) if
// the call is eligible, returning true. Returns false -- having launched
// nothing -- when the call must fall through to the cooperative exl3_mgemm.
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
    const int* size_n_list,   // per-matrix output widths (device), or nullptr
    void** c_list,            // per-matrix output pointers (device), or nullptr
    int device,
    cudaStream_t stream,
    Graph* graph
);
