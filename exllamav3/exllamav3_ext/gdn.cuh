#pragma once

#include <cstdint>
#include <vector>

class Graph;

void gated_delta_net_fused_op
(
    const at::Tensor& mixed_qkvz,
    const at::Tensor& mixed_ba,
    const at::Tensor& dt_bias,
    const at::Tensor& a_log,
    at::Tensor& mixed_qkv,
    at::Tensor& z,
    at::Tensor& beta,
    at::Tensor& g,
    size_t num_k_heads,
    size_t num_v_heads,
    size_t k_head_dim,
    size_t v_head_dim,
    const float beta_scale
);

void gated_delta_net_fused_op_2
(
    const at::Tensor& b,
    const at::Tensor& a,
    const at::Tensor& dt_bias,
    const at::Tensor& a_log,
    at::Tensor& beta,
    at::Tensor& g,
    const float beta_scale
);

void cuda_recurrent_gated_delta_rule
(
    const at::Tensor& mixed_qkv,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history,
    const c10::optional<at::Tensor>& slots_in = c10::nullopt
);

void cuda_recurrent_gated_delta_rule_gr
(
    const at::Tensor& mixed_qkv,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history,
    const c10::optional<at::Tensor>& slots_in,
    Graph* graph
);

// Mamba2 discretization: dt = clamp(softplus(dt_raw + dt_bias), dt_min, dt_max), g = -exp(a_log) * dt
void mamba2_dt_op
(
    const at::Tensor& dt_raw,       // [B,S,H] float
    const at::Tensor& dt_bias,      // [H] float
    const at::Tensor& a_log,        // [H] float
    at::Tensor& dt,                 // out [B,S,H] bfloat16
    at::Tensor& g,                  // out [B,S,H] float
    float dt_min,
    float dt_max
);

// Mamba2 BC helper: bf16 conv input + discretized dt/g + a contiguous gate copy, from the
// in_proj output [z, xBC, dt]
void mamba2_fused_op_gr
(
    const at::Tensor& proj,         // [B,S,>= v_dim + F + dt_first + H] float
    at::Tensor& xbc,                // out [B, F, S] bfloat16
    at::Tensor& dt,                 // out [B, S, H] bfloat16
    at::Tensor& g,                  // out [B, S, H] float
    at::Tensor& z_gate,             // out [B, S, v_dim] float, contiguous
    const at::Tensor& dt_bias,      // [H] float
    const at::Tensor& a_log,        // [H] float
    int v_dim,
    int dt_first,
    float dt_min,
    float dt_max,
    class Graph* graph
);

// Mamba2 (SSD) recurrence: gated delta rule without the correction term, over conv channel
// order [x, B, C], with dt as input scale and per-head skip y += D * x
void cuda_recurrent_mamba2
(
    const at::Tensor& mixed_xbc,
    const at::Tensor& g,
    const at::Tensor& dt,
    const at::Tensor& D,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history,
    const c10::optional<at::Tensor>& slots_in = c10::nullopt
);

void cuda_recurrent_mamba2_gr
(
    const at::Tensor& mixed_xbc,
    const at::Tensor& g,
    const at::Tensor& dt,
    const at::Tensor& D,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history,
    const c10::optional<at::Tensor>& slots_in,
    Graph* graph
);

void cuda_causal_conv1d_update
(
    const at::Tensor& x,
    at::Tensor& conv_state,
    const c10::optional<at::Tensor>& slots,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& out,
    bool activation,
    bool history
);

void cuda_causal_conv1d_update_gr
(
    const at::Tensor& x,
    at::Tensor& conv_state,
    const c10::optional<at::Tensor>& slots,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& out,
    bool activation,
    bool history,
    Graph* graph
);

// Split-projection (Qwen3.5) helper: cast/transpose qkv to bf16 mixed_qkv and compute beta/g from
// the packed ba projection
void gated_delta_net_fused_op_3_gr
(
    const at::Tensor& qkv,          // [B,S,F] float
    const at::Tensor& ba,           // [B,S,2H] float
    const at::Tensor& dt_bias,      // [H] bfloat16
    const at::Tensor& a_log,        // [H] float or bfloat16
    at::Tensor& mixed_qkv,          // out [B,F,S] bfloat16
    at::Tensor& beta,               // out [B,S,H] bfloat16
    at::Tensor& g,                  // out [B,S,H] float
    const float beta_scale,
    Graph* graph
);

// Small fp16 GEMV with fp32 accumulation/output for the merged b/a projections. Avoids cublas so
// the x pointer can be patched in captured graphs
void gdn_ba_gemv
(
    const at::Tensor& x,            // [.., k] half
    const at::Tensor& w_t,          // [n, k] half
    const c10::optional<at::Tensor>& bias,  // [n] half
    at::Tensor& y                   // [.., n] float
);

void gdn_ba_gemv_gr
(
    const at::Tensor& x,            // [.., k] half
    const at::Tensor& w_t,          // [n, k] half
    const c10::optional<at::Tensor>& bias,  // [n] half
    at::Tensor& y,                  // [.., n] float
    Graph* graph
);

// Float-input fp16-weight GEMV for the KDA low-rank second stages (graph statics, no patching)
void gdn_lowrank_gemv_f_gr
(
    const at::Tensor& x,            // [.., k] float
    const at::Tensor& w_t,          // [n, k] half
    at::Tensor& y,                  // [.., n] float
    Graph* graph
);

// KDA (GLM5.3) helper: cast/transpose qkv to bf16 mixed_qkv, beta = sigmoid(b), per-k-channel
// log decay from the low-rank forget path (safe-gate when lower_bound != 0, else softplus)
void kda_gate_op_gr
(
    const at::Tensor& qkv,          // [B,S,F] float
    const at::Tensor& b,            // [B,S,H] float
    const at::Tensor& f,            // [B,S,H*Dk] float
    const at::Tensor& dt_bias,      // [H*Dk] bfloat16
    const at::Tensor& a_log,        // [H] float or bfloat16
    at::Tensor& mixed_qkv,          // out [B,F,S] bfloat16
    at::Tensor& beta,               // out [B,S,H] bfloat16
    at::Tensor& g,                  // out [B,S,H,Dk] float
    const float lower_bound,
    const float beta_scale,
    Graph* graph
);

// Batched recurrent-state rewind (speculative decoding draft rejection/commit)

// conv_state shift: conv_state[slot, :, :cdim] <- conv_state[slot, :, p-cdim:p]. `dim` independent
// per-channel copies of `cdim` elements, `stride` elements apart in both src and dst (same
// tensor, same per-channel stride). src/dst can overlap when num_tokens < conv_kernel_size.
struct ConvRewindJob
{
    uintptr_t src;
    uintptr_t dst;
    int dim;
    int cdim;
    int stride;

    ConvRewindJob() = default;
    ConvRewindJob(uintptr_t _src, uintptr_t _dst, int _dim, int _cdim, int _stride) :
        src(_src), dst(_dst), dim(_dim), cdim(_cdim), stride(_stride) {}
};

// Recurrent-state replay (the state half of a rewind): rerun the sequential scan over the first
// `prefix` staged tokens of one batch row of the last speculative pass, from the base row into
// the scratch row of the layer's state pool. The scan kernel is the one the pass ran, over the
// same staged inputs and the same initial state, so the result is the state that pass had after
// `prefix` tokens. One job per recurrent layer; a launch carries the jobs of every layer that
// shares the kernel geometry (all layers of a model), so a rewind is one launch per device.
struct ScanReplayJob
{
    uintptr_t qkv;          // staged post-conv qkv rows of this batch row, [prefix, qkv_dim] bf16
    uintptr_t g;            // staged log decay, [prefix, heads] or [prefix, heads, dk] (KDA) fp32
    uintptr_t beta;         // staged beta / dt, [prefix, heads] bf16
    uintptr_t base;         // state row read for the first token, fp32 [heads, dk, dv]
    uintptr_t scratch;      // state row written (first token from base, then in place)
    uintptr_t D;            // Mamba2 skip scales [heads] fp32, else 0
    int prefix;

    ScanReplayJob() = default;
    ScanReplayJob(uintptr_t _qkv, uintptr_t _g, uintptr_t _beta, uintptr_t _base, uintptr_t _scratch,
                  uintptr_t _D, int _prefix) :
        qkv(_qkv), g(_g), beta(_beta), base(_base), scratch(_scratch), D(_D), prefix(_prefix) {}
};

void batched_conv_rewind(std::vector<ConvRewindJob> const& jobs, int device_index);

// kind: 0 = gated delta rule, 1 = Mamba2, 2 = KDA (channelwise decay; requires 128x128 heads)
void batched_scan_replay
(
    std::vector<ScanReplayJob> const& jobs,
    int device_index,
    int kind,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim
);
