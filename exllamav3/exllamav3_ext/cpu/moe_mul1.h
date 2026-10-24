#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <utility>
#include <vector>

// CPU-side MoE expert GEMM for mul1 (cb2) EXL3 tensors, standalone from the module code so it
// can be benchmarked and driven directly. A layer is registered once (raw pointers into CPU
// trellis/suh/svh tensors, which the caller must keep alive) and then invoked per forward with
// the routing results. Kernels dispatch at runtime on scalar / AVX2 / AVX-512BW / AVX-512+VNNI;
// the mul1 codebook is affine in a byte-sum, so dequantization and the activation product
// fuse into integer dot products (see exl3_moe_cpu_forward for the math).
//
// Current limits: mul1 codebook only, K in [1, 8] or a half-integer rate 1.5 / 2.5 / 3.5. Gated experts with silu/gelu/swiglu_oai
// (act_limit) or gateless with relu2; optional per-expert biases (uniform per projection).

struct MoeCpuMatrix
{
    const uint16_t* trellis;
    const at::Half* suh;
    const at::Half* svh;
    const at::Half* bias;   // nullable; added after the output transform
    int k;
    int n;
    int bits;               // bits per weight, integer part
    int hb = 0;             // half-integer rate: bits + 0.5
    // Packed trellis layout (integer rates on the tiers that band them; the loader repacks
    // with the group exl3_moe_cpu_swizzle_group returns, so swz is re-derived from the rate,
    // never trusted from the caller). Tile (kt, nt) stored at (nt/g) * tiles_k * g + kt * g +
    // nt % g, so each banded kernel's tile group reads as one sequential k-stream; on the
    // AVX2 tier (g = 2) the tile dwords are additionally planar-repacked (dword w at
    // 8 * (w % bits) + w / bits), which is part of the g = 2 contract, not a separate flag.
    int swz = 0;
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
    // NUMA column-split views (exl3_moe_cpu_forward_raw_numa): a view layer's gate/up matrices
    // span this node's half of the intermediate columns and its down matrices this node's half
    // of the hidden rows, all as pointer ranges inside the layer's full packed tensors. The
    // offsets are in 128-column blocks: where this view's gate/up GEMV outputs (and the down
    // input row they feed) begin in the full intermediate, and where its down outputs begin in
    // the full hidden. Zero on normally registered (unsplit) layers
    int gu_blk_off = 0;
    int d_blk_off = 0;
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
    double act_limit,
    int64_t swizzled        // caller repacked each trellis tensor with exl3_moe_cpu_swizzle_group
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

// NUMA column-split variant: the layer's expert weights are split by 128-column blocks across
// two nodes (node 0 taking the low gate/up columns and low down rows), each node's pool reading
// only its node-local half, so both sockets' memory bandwidth stack on the same job. Bitwise
// identical to exl3_moe_cpu_forward_raw on the same inputs. threads is PER NODE; the layer must
// have been registered with the swizzled packed layout and a rate whose swizzle group is
// nonzero (K8 and EXL3_MOE_CPU_SWIZZLE=0 layers are rejected), and both halves of `out` are
// written. Requires x86 and a non-scalar ISA tier.
void exl3_moe_cpu_forward_raw_numa
(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* w,
    float* out,
    int rows,
    int topk,
    int threads_per_node
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
// Wake helpers before the GPU payload arrives; no work or completion barrier.
void exl3_moe_cpu_pool_prime(int threads);
int64_t exl3_moe_cpu_pool_stress(int threads, int iters, int small, int spin);   // test hook
// Pool topology for host placement: physical-core-first encoded LP order and the physical core
// count; empty when EXL3_MOE_CPU_PIN=0.
std::pair<std::vector<int64_t>, int64_t> exl3_moe_cpu_core_order();

// Per-NUMA-node physical-core-first LP orders (Linux; one entry per node as (order, n_phys),
// derived from /sys/devices/system/node/*/cpulist). Empty when the topology is unreadable or
// the host is single-node.
std::vector<std::pair<std::vector<int64_t>, int64_t>> exl3_moe_cpu_numa_core_orders();

// at::Tensor variant of exl3_moe_cpu_forward_raw_numa for direct testing: same contract, with
// the usual tensor checks. num_threads is PER NODE.
void exl3_moe_cpu_forward_numa
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
);

// Kernel availability (dispatch happens internally; these are informational, post-env-cap).
bool exl3_moe_cpu_has_avx2();
bool exl3_moe_cpu_has_avx512_bw();
bool exl3_moe_cpu_has_avx512_vnni();
bool exl3_moe_cpu_has_avx512_vbmi();

// Packed-layout rules for one trellis rate K (possibly half-integer) under the runtime ISA
// tier. swizzle_group: 0 = native tile order, 8 = band-8 (AVX-512 banded kernels, K8 excepted),
// 2 = band-2 + planar dwords (AVX2, integer rates only). planar_layout: 1 = the intra-tile
// dword repack the AVX2 band-2 kernel requires (always paired with group 2).
//
// Single source of truth: the child loader repacks each tensor with these values, the kernels
// dispatch on them, and the GPU staging path un-does (HIP) or reads (CUDA) the same bytes.
// Query these instead of duplicating the gate. Rollback knob: EXL3_MOE_CPU_SWIZZLE=0 makes
// the loader keep native bytes everywhere and zeroes these for the GPU path.
int exl3_moe_cpu_swizzle_group(double K);
int exl3_moe_cpu_planar_layout(double K);
