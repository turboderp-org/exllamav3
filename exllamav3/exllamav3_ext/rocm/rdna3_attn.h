#pragma once

#include <memory>
struct TritonKernel;

// HIP flash-decoding split kernel (drop-in for the Triton decode split kernel on RDNA3), or
// nullptr when the configuration is not covered
std::shared_ptr<TritonKernel> rdna3_attn_split_kernel(int qc, int head_dim, int q_len, int n_kv_heads, int group);
