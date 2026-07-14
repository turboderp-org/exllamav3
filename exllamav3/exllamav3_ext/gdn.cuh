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
    bool history
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

// Mamba2 bsz-1 BC helper: bf16 conv input + discretized dt/g from the in_proj output [z, xBC, dt]
void mamba2_fused_op_gr
(
    const at::Tensor& proj,         // [1,1,>= v_dim + F + dt_first + H] float
    at::Tensor& xbc,                // out [1, F, 1] bfloat16
    at::Tensor& dt,                 // out [1, 1, H] bfloat16
    at::Tensor& g,                  // out [1, 1, H] float
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
    bool history
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
