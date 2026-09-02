#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include "ple.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"
#include "norm.cuh"

namespace py = pybind11;

#define NUM_THREADS 256

// PLE stream gate (Qwen3.8-Flash-Next): out = sigmoid(signed_sqrt(gate * gate_scale)) * value,
// the shared value row broadcast over the hyper-connection streams:
//
//     out[b, s, h, :] = sigmoid(ss(gate[b, s, h] * gate_scale)) * value[b, s, :]
//     ss(g) = sign(g) * sqrt(max(|g|, 1e-6)), with sign(0) = 0 (matching torch.sign)
//
// gate (B, S, H) fp32 raw dot products, value (B, S, D) fp16, out (B, S, H, D) fp32.

__global__ __launch_bounds__(NUM_THREADS)
void ple_gate_kernel
(
    const float* __restrict__ gate,
    const half* __restrict__ value,
    float* __restrict__ out,
    const size_t numel,          // B * S * H * D
    const int h_streams,
    const int dim,
    const float gate_scale
)
{
    size_t idx = ((size_t) blockIdx.x * NUM_THREADS + threadIdx.x);
    if (idx >= numel / 4) return;
    size_t e = idx * 4;
    size_t rh = e / dim;
    size_t d = e - rh * dim;
    size_t r = rh / h_streams;

    float g = gate[rh] * gate_scale;
    float a = sqrtf(fmaxf(fabsf(g), 1e-6f));
    float ss = g > 0.0f ? a : (g < 0.0f ? -a : 0.0f);
    float s = __fdividef(1.0f, 1.0f + __expf(-ss));

    half4 v4 = *(const half4*) (value + r * dim + d);
    float4 o4;
    o4.x = s * LOW_TO_FLOAT(v4.x);
    o4.y = s * HIGH_TO_FLOAT(v4.x);
    o4.z = s * LOW_TO_FLOAT(v4.y);
    o4.w = s * HIGH_TO_FLOAT(v4.y);
    *(float4*) (out + e) = o4;
}

void ple_gate_gr
(
    const at::Tensor& gate,
    const at::Tensor& value,
    at::Tensor& out,
    float gate_scale,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(out.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(gate, kFloat);
    TORCH_CHECK_DTYPE(value, kHalf);
    TORCH_CHECK_DTYPE(out, kFloat);
    TORCH_CHECK(out.dim() == 4, "out must be [B, S, H, D]");
    TORCH_CHECK(gate.dim() == 3, "gate must be [B, S, H]");
    TORCH_CHECK(value.dim() == 3, "value must be [B, S, D]");
    TORCH_CHECK(out.size(0) == gate.size(0) && out.size(1) == gate.size(1) &&
                out.size(2) == gate.size(2), "out and gate have incompatible shapes");
    TORCH_CHECK(out.size(0) == value.size(0) && out.size(1) == value.size(1) &&
                out.size(3) == value.size(2), "out and value have incompatible shapes");
    TORCH_CHECK(gate.is_contiguous() && value.is_contiguous() && out.is_contiguous(),
                "ple_gate: tensors must be contiguous");
    TORCH_CHECK(out.size(3) % 4 == 0, "out.size(3) must be a multiple of 4");

    size_t numel = out.numel();
    size_t blocks = CEIL_DIVIDE(numel, 4 * NUM_THREADS);
    ple_gate_kernel<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (const float*) gate.data_ptr(),
        (const half*) value.data_ptr(),
        (float*) out.data_ptr(),
        numel,
        (int) out.size(2),
        (int) out.size(3),
        gate_scale
    );

    cuda_check(cudaPeekAtLastError());
}

void ple_gate
(
    const at::Tensor& gate,
    const at::Tensor& value,
    at::Tensor& out,
    float gate_scale
)
{
    ple_gate_gr(gate, value, out, gate_scale, nullptr);
}

// Whole PLE forward_streams as one C++ call: the layer sits at the very front of the forward
// pass and its many tiny ops made it host-bound (~250us/token mostly GPU idle) when issued
// from python. No graph capture -- just the same op sequence (ATen + the ext kernels) without
// the python/C++ transitions. Mirrors PLELayer.forward_streams (the retained python
// reference) exactly:
//
//   key    = norm_key(emb @ key_w.T)          grouped RMS (streams of D), fp32 out
//   query  = norm_query(streams)              fp32 out
//   gate   = per-stream key . query dots      batched (1, D) x (D, 1) bmm
//   gated  = sigmoid(ss(gate * scale)) * (emb @ value_w.T)     (ple_gate kernel)
//   normed = norm_conv(gated)                 half out
//   conv_stream = [conv_state | normed^T]; out = silu(depthwise dilated conv1d)
//   delta  = gated + out^T

void ple_forward_streams
(
    const at::Tensor& streams,       // (bsz, seq, H, D) fp32 contiguous
    const at::Tensor& emb,           // (bsz, seq, ple_dim) half
    const at::Tensor& key_w,         // (ple_dim, H * D) half (the fp16 Linear's matmul layout)
    const at::Tensor& value_w,       // (ple_dim, D) half
    const at::Tensor& norm_key_w,    // (H * D) half/bf16 raw weight (+1 constant bias)
    const at::Tensor& norm_query_w,
    const at::Tensor& norm_conv_w,
    const at::Tensor& conv_w,        // (H * D, 1, ksize) half
    const c10::optional<at::Tensor>& conv_state,   // (bsz, H * D, state_len) half, or none
    double rms_eps,
    double gate_scale,
    int64_t conv_dilation,
    at::Tensor delta,                // (bsz, seq, H, D) fp32 out
    at::Tensor conv_stream           // (bsz, H * D, state_len + seq) half out
)
{
    py::gil_scoped_release release;
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());

    int64_t bsz = streams.size(0);
    int64_t seq = streams.size(1);
    int64_t H = streams.size(2);
    int64_t D = streams.size(3);
    int64_t R = bsz * seq;
    int64_t hc = H * D;
    int64_t state_len = conv_stream.size(2) - seq;
    float eps = (float) rms_eps;
    TORCH_CHECK(streams.is_contiguous() && emb.is_contiguous(), "ple_forward_streams: contiguous inputs");
    TORCH_CHECK(delta.is_contiguous() && conv_stream.is_contiguous(), "ple_forward_streams: contiguous outputs");

    auto opts_f = streams.options();
    auto opts_h = emb.options();

    at::Tensor emb2 = emb.view({R, emb.size(2)});
    at::Tensor key = at::matmul(emb2, key_w);                                // (R, hc) half
    at::Tensor key_n = at::empty({R * H, D}, opts_f);
    rms_norm(key.view({R * H, D}), norm_key_w, key_n, eps, 1.0f, 1.0f, false, false, (int) H);

    at::Tensor query_n = at::empty({R * H, D}, opts_f);
    at::Tensor s2 = streams.view({R * H, D});
    rms_norm(s2, norm_query_w, query_n, eps, 1.0f, 1.0f, false, false, (int) H);

    at::Tensor gate = at::bmm(query_n.view({R * H, 1, D}), key_n.view({R * H, D, 1}))
        .view({bsz, seq, H});
    at::Tensor value = at::matmul(emb2, value_w).view({bsz, seq, D}).contiguous();

    at::Tensor gated = at::empty({bsz, seq, H, D}, opts_f);
    ple_gate(gate.contiguous(), value, gated, (float) gate_scale);

    at::Tensor normed = at::empty({R * H, D}, opts_h);
    rms_norm(gated.view({R * H, D}), norm_conv_w, normed, eps, 1.0f, 1.0f, false, false, (int) H);

    // conv column stream: [state | new columns], then the dilated depthwise conv
    if (state_len > 0)
    {
        if (conv_state)
            conv_stream.narrow(2, 0, state_len).copy_(conv_state.value());
        else
            conv_stream.narrow(2, 0, state_len).zero_();
    }
    conv_stream.narrow(2, state_len, seq)
        .copy_(normed.view({bsz, seq, hc}).transpose(1, 2));
    at::Tensor y = at::conv1d(conv_stream, conv_w, c10::nullopt,
                              at::IntArrayRef {1}, at::IntArrayRef {0},
                              at::IntArrayRef {conv_dilation}, hc);
    at::Tensor conv_out = at::silu(y).transpose(1, 2);                        // (bsz, seq, hc)

    at::Tensor d2 = delta.view({bsz, seq, hc});
    at::add_out(d2, gated.view({bsz, seq, hc}), conv_out);
}
