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
    size_t v_head_dim
);
