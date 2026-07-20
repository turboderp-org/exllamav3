#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <vector>

// CPU-side MoE expert GEMM for mul1 (cb2) EXL3 tensors, standalone from the module code so it
// can be benchmarked and driven directly. A layer is registered once (raw pointers into CPU
// trellis/suh/svh tensors, which the caller must keep alive) and then invoked per forward with
// the routing results. Kernels dispatch at runtime on scalar / AVX2 / AVX-512+VNNI; the mul1
// codebook is affine in a byte-sum, so dequantization and the activation product fuse into
// integer dot products (see exl3_moe_cpu_forward for the math).
//
// Current limits: mul1 codebook only, K in [1, 8]. Gated experts with silu/gelu/swiglu_oai
// (act_limit) or gateless with relu2; optional per-expert biases (uniform per projection).

struct MoeCpuMatrix
{
    const uint16_t* trellis;
    const at::Half* suh;
    const at::Half* svh;
    const at::Half* bias;   // nullable; added after the output transform
    int k;
    int n;
    int bits;
};

struct MoeCpuLayer
{
    std::vector<MoeCpuMatrix> gates;
    std::vector<MoeCpuMatrix> ups;
    std::vector<MoeCpuMatrix> downs;
    // Tensor references keeping the CPU weight storage alive
    std::vector<at::Tensor> refs;
    int num_experts;
    int hidden_size;      // k of gate/up, n of down (unpadded handling is the caller's problem)
    int interm_size;      // n of gate/up, k of down
    int activation;       // 0 = silu, 1 = gelu, 2 = relu2 (gateless), 3 = swiglu_oai
    float act_limit;      // swiglu_oai clamp
};

// Register a layer: per-expert tensor lists (CPU, contiguous). Returns a handle.
int64_t exl3_moe_cpu_make_layer
(
    const std::vector<at::Tensor>& gate_trellis,
    const std::vector<at::Tensor>& gate_suh,
    const std::vector<at::Tensor>& gate_svh,
    const std::vector<at::Tensor>& up_trellis,
    const std::vector<at::Tensor>& up_suh,
    const std::vector<at::Tensor>& up_svh,
    const std::vector<at::Tensor>& down_trellis,
    const std::vector<at::Tensor>& down_suh,
    const std::vector<at::Tensor>& down_svh,
    const std::vector<at::Tensor>& gate_bias,
    const std::vector<at::Tensor>& up_bias,
    const std::vector<at::Tensor>& down_bias,
    int64_t activation,
    double act_limit
);

void exl3_moe_cpu_free_layer(int64_t handle);

// Run the routed experts for one forward:
//   x:        [m, hidden] fp16, CPU
//   selected: [m, top_k] int64, CPU (global expert ids)
//   weights:  [m, top_k] fp16, CPU
//   out:      [m, hidden] fp32, CPU (overwritten)
// Tokens are grouped by expert; each expert runs gate/up GEMVs (int8-VNNI fused mul1 decode),
// the activation, and the down GEMV, accumulating routing-weighted rows into out. Threaded over
// a persistent spin-parked pool; the caller should release the GIL around this.
void exl3_moe_cpu_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
);

// Raw-pointer variant used by the persistent worker (moe_handoff.cu): same computation as
// exl3_moe_cpu_forward, expert selection as int32, buffers caller-owned
void exl3_moe_cpu_forward_raw
(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* w,
    float* out,
    int rows,
    int topk,
    int threads
);

// Copy `count` experts' packed trellis tensors (gate, up, down order; gate absent when
// gateless) of a registered layer into a staging buffer, expert-major, parallelized over the
// worker pool. Offsets are deterministic from the layer's matrix dims so the parent can compute
// the same layout for the VRAM-side views.
void exl3_moe_cpu_stage_experts
(
    int64_t handle,
    const uint32_t* expert_ids,
    int count,
    uint8_t* dst,
    int threads
);

// Per-phase profiling of the compute pool, reported to stdout every 512 jobs. Set once at
// worker startup from MoeCpuTuning.cpu_prof (EXL3_MOE_CPU_PROF env).
void exl3_moe_cpu_set_prof(bool enabled);

// Kernel availability (dispatch happens internally; these are informational)
bool exl3_moe_cpu_has_avx2();
bool exl3_moe_cpu_has_avx512_vnni();
