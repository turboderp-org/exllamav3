#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

// Fused attention for CPU tensors (vision-tower CPU offload path).
//
// Same math as the CUDA/Triton attention kernels (triton_paged prefill, flash_attn_varlen):
// online-softmax-free two-pass dense attention computed in float32 with a single round
// to the output dtype: scores = (Q @ K^T) * scale, softcap = softcap * tanh(scores /
// softcap) when softcap > 0, softmax with optional per-head sinks (one extra
// exp(sink - max) term in the denominator, zero value contribution, gpt-oss style),
// causal + sliding-window masking, then O = P @ V.
//
// Layouts: q/o [B, Lq, Hq, D] and k/v [B, Lk, Hkv, D], all contiguous CPU in
// {float32, half, bfloat16} in any combination; o may differ in dtype from q/k/v.
// Hq % Hkv == 0 (GQA groups). D in [1, 1024]. numel == 0 is a no-op. Anything outside
// this contract throws; the Python caller falls back to the torch reference, so a
// forward never fails here.
//
// Varlen: cu_seqlens is an optional 1-D contiguous CPU int32/int64 table [S+1] with
// cu[0] == 0 and cu[S] == Lq == Lk (shared across batches for B > 1, matching the
// torch fallback's per-batch slicing). Each segment attends independently with
// positions reset to 0, exactly like flash_attn_varlen_func with cu_q == cu_k.
// Null means one dense segment [0, L) per batch.
//
// Dense position mapping (spans prefix support): query row lq in batch b sits at
// absolute position q_start + lq, keys at absolute k_idx in [0, Lk). Valid keys satisfy
// (!causal || k <= q) && (!left || k >= q - left) && (!right || k <= q + right),
// with left/right < 0 disabling that side. Varlen ignores q_start (positions are
// segment-relative).
//
// sinks is an optional 1-D contiguous CPU float32 [Hq] (one logit per q head).
// cu_seqlens int32/int64 CPU when present.
//
// Threading runs on the persistent process-wide pool shared with the MoE CPU path
// (moe_mul1.cpp), one tile per 4-query block (all heads). ISA dispatch (AVX-512 /
// AVX2 / scalar) reuses the MoE ISA detection, overridable per process with
// EXL3_ATTN_CPU_ISA=avx512|avx2|scalar (testing only).
//
// Performance note: this kernel is SLOWER than torch CPU SDPA on simple
// dense/varlen (measured Ryzen 9700X, 8 threads, fp16 vision shapes, median ms:
// 1k dense torch 6.37 vs AVX 19.79; 4k dense torch 93.64 vs AVX 680.22;
// 32x64 varlen torch-batched 0.78 vs AVX 1.98). Torch lowers to oneDNN/MKL
// blocked GEMM near peak (~500-700 GFLOPS here); this kernel streams K per query
// (each K row reloaded L times from L3, single-accumulator FMA + temp-array hsum
// per dot, no cross-query blocking), reaching only ~100-160 GFLOPS. AVX still
// beats scalar 2-3x (1k 19.79 vs 53.59; 4k 680.22 vs 1393.09). This kernel wins
// where torch has no fast path: SWA/windowed restricts work to O(N*window)
// (L=4096,w=128 native 8.7ms vs dense-full torch 25.2ms) and softcap/sinks/spans
// avoid N^2 temporaries that make torch manual OOM at 4k. The Python caller
// (attention_fn/torch.py) therefore routes simple shapes to torch and reserves
// this kernel for the ceiling.
void attn_cpu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& o,
    double sm_scale,
    bool causal,
    int64_t window_left,
    int64_t window_right,
    double softcap,
    const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& cu_seqlens,
    int64_t q_start);
