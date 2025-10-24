#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include "activation.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "compat.cuh"
#include <cmath>
#include "reduction.cuh"

#define NUM_THREADS 256
#define NUM_THREADS_P 1024
#define ACT_SILU 0
#define ACT_GELU 1
#define ACT_RELU2 2

#include "activation_kernels.cuh"

// silu(x) * y -> z, in-place if z == x or z == y

void silu_mul_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    bool float_input = x.dtype() == at::kFloat;
    if (float_input)
    {
        TORCH_CHECK_DTYPE(y, kFloat);
    }
    else
    {
        TORCH_CHECK_DTYPE(x, kHalf);
        TORCH_CHECK_DTYPE(y, kHalf);
    }

    TORCH_CHECK_DTYPE(z, kHalf);

    size_t numel = x.numel();
    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    if (float_input)
    {
        act_mul_kernel_f<ACT_SILU><<<blocks, NUM_THREADS, 0, stream>>>
        (
            (const float*) x.data_ptr(),
            (const float*) y.data_ptr(),
            (half*) z.data_ptr(),
            numel
        );

        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_SILU>, GP_silu_mul_x, 0);
        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_SILU>, GP_silu_mul_y, 1);
        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_SILU>, GP_silu_mul_z, 2);
        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_SILU>, GP_end, 0);

        cuda_check(cudaPeekAtLastError());
    }
    else
    {
        act_mul_kernel_h<ACT_SILU><<<blocks, NUM_THREADS, 0, stream>>>
        (
            (const half*) x.data_ptr(),
            (const half*) y.data_ptr(),
            (half*) z.data_ptr(),
            numel
        );

        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_SILU>, GP_silu_mul_x, 0);
        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_SILU>, GP_silu_mul_y, 1);
        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_SILU>, GP_silu_mul_z, 2);
        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_SILU>, GP_end, 0);

        cuda_check(cudaPeekAtLastError());
    }
}

void silu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
)
{
    silu_mul_gr(x, y, z, nullptr);
}

// silu(x) * y -> z, in-place if z == x or z == y

void gelu_mul_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    bool float_input = x.dtype() == at::kFloat;
    if (float_input)
    {
        TORCH_CHECK_DTYPE(y, kFloat);
    }
    else
    {
        TORCH_CHECK_DTYPE(x, kHalf);
        TORCH_CHECK_DTYPE(y, kHalf);
    }

    TORCH_CHECK_DTYPE(z, kHalf);

    size_t numel = x.numel();
    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    if (float_input)
    {
        act_mul_kernel_f<ACT_GELU><<<blocks, NUM_THREADS, 0, stream>>>
        (
            (const float*) x.data_ptr(),
            (const float*) y.data_ptr(),
            (half*) z.data_ptr(),
            numel
        );

        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_GELU>, GP_gelu_mul_x, 0);
        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_GELU>, GP_gelu_mul_y, 1);
        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_GELU>, GP_gelu_mul_z, 2);
        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_GELU>, GP_end, 0);

        cuda_check(cudaPeekAtLastError());
    }
    else
    {
        act_mul_kernel_h<ACT_GELU><<<blocks, NUM_THREADS, 0, stream>>>
        (
            (const half*) x.data_ptr(),
            (const half*) y.data_ptr(),
            (half*) z.data_ptr(),
            numel
        );

        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_GELU>, GP_gelu_mul_x, 0);
        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_GELU>, GP_gelu_mul_y, 1);
        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_GELU>, GP_gelu_mul_z, 2);
        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_GELU>, GP_end, 0);

        cuda_check(cudaPeekAtLastError());
    }
}

void gelu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
)
{
    gelu_mul_gr(x, y, z, nullptr);
}

// relu^2(x) * y -> z

void relu2_mul_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    bool float_input = x.dtype() == at::kFloat;
    if (float_input)
    {
        TORCH_CHECK_DTYPE(y, kFloat);
    }
    else
    {
        TORCH_CHECK_DTYPE(x, kHalf);
        TORCH_CHECK_DTYPE(y, kHalf);
    }

    TORCH_CHECK_DTYPE(z, kHalf);

    size_t numel = x.numel();
    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    if (float_input)
    {
        act_mul_kernel_f<ACT_RELU2><<<blocks, NUM_THREADS, 0, stream>>>
        (
            (const float*) x.data_ptr(),
            (const float*) y.data_ptr(),
            (half*) z.data_ptr(),
            numel
        );

        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_RELU2>, GP_relu2_mul_x, 0);
        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_RELU2>, GP_relu2_mul_y, 1);
        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_RELU2>, GP_relu2_mul_z, 2);
        if (graph) graph->record_param((void*) &act_mul_kernel_f<ACT_RELU2>, GP_end, 0);

        cuda_check(cudaPeekAtLastError());
    }
    else
    {
        act_mul_kernel_h<ACT_RELU2><<<blocks, NUM_THREADS, 0, stream>>>
        (
            (const half*) x.data_ptr(),
            (const half*) y.data_ptr(),
            (half*) z.data_ptr(),
            numel
        );

        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_RELU2>, GP_relu2_mul_x, 0);
        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_RELU2>, GP_relu2_mul_y, 1);
        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_RELU2>, GP_relu2_mul_z, 2);
        if (graph) graph->record_param((void*) &act_mul_kernel_h<ACT_RELU2>, GP_end, 0);

        cuda_check(cudaPeekAtLastError());
    }
}

void relu2_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
)
{
    relu2_mul_gr(x, y, z, nullptr);
}

// xielu(x, alpha_p, alpha_n) -> z
// alpha_p and alpha_n must be CPU tensors

void xielu_gr
(
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& alpha_p,
    const at::Tensor& alpha_n,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    bool float_input = x.dtype() == at::kFloat;
    if (!float_input) TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(y, kHalf);

    auto get_alpha = [&] (const at::Tensor& t)
    {
        TORCH_CHECK(t.device().is_cpu(), "alpha_p and alpha_n must be CPU tensors");
        float x = 0.0f;
        if (t.dtype() == at::kFloat) x = *((const float*) t.data_ptr());
        else if (t.dtype() == at::kHalf) x = static_cast<float>(*((const at::Half*) t.data_ptr()));
        else if (t.dtype() == at::kBFloat16) x = static_cast<float>(*((const at::BFloat16*) t.data_ptr()));
        else TORCH_CHECK(false, "Unsupported dtype for alpha_p or alpha_n");
        return x > 20.0f ? x : log1pf(expf(x));
    };

    float p = get_alpha(alpha_p);
    float n = get_alpha(alpha_n) + 0.5f;

    size_t numel = x.numel();
    size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);
    if (float_input)
    {
        xielu_kernel_f<<<blocks, NUM_THREADS, 0, stream>>>
        (
            (const float*) x.data_ptr(),
            (half*) y.data_ptr(),
            numel,
            p,
            n
        );

        if (graph) graph->record_param((void*) &xielu_kernel_f, GP_xielu_x, 0);
        if (graph) graph->record_param((void*) &xielu_kernel_f, GP_xielu_y, 1);
        if (graph) graph->record_param((void*) &xielu_kernel_f, GP_end, 0);

        cuda_check(cudaPeekAtLastError());
    }
    else TORCH_CHECK(false, "xielu not implemented for float16 input dtype");
}

void xielu
(
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& alpha_p,
    const at::Tensor& alpha_n
)
{
    xielu_gr(x, y, alpha_p, alpha_n, nullptr);
}

// x * sigmoid(y) + z -> z

void add_sigmoid_gate_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK_DTYPE(y, kFloat);
    TORCH_CHECK_DTYPE(z, kFloat);

    int dim = x.size(-1);
    int gdim = y.size(-1);
    TORCH_CHECK(gdim == 1, "gate must have size(-1) == 1")

    size_t numel = x.numel();
    size_t blocks = CEIL_DIVIDE(numel, NUM_THREADS);
    add_sigmoid_kernel_f<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (const float*) x.data_ptr(),
        (const float*) y.data_ptr(),
        (float*) z.data_ptr(),
        numel,
        dim
    );

    if (graph) graph->record_param((void*) &add_sigmoid_kernel_f, GP_add_sigmoid_gate_x, 0);
    if (graph) graph->record_param((void*) &add_sigmoid_kernel_f, GP_add_sigmoid_gate_y, 1);
    if (graph) graph->record_param((void*) &add_sigmoid_kernel_f, GP_add_sigmoid_gate_z, 2);
    if (graph) graph->record_param((void*) &add_sigmoid_kernel_f, GP_end, 0);

    cuda_check(cudaPeekAtLastError());
}

void add_sigmoid_gate
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
)
{
    add_sigmoid_gate_gr(x, y, z, nullptr);
}

// x * sigmoid(y @ w) + z -> z

void add_sigmoid_gate_proj_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const at::Tensor& w,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_DTYPE(z, kFloat);
    TORCH_CHECK_DTYPE(w, kHalf);

    int dim = x.size(-1);
    int gdim = w.size(-1);
    TORCH_CHECK(gdim == 1, "gate must have size(-1) == 1")
    TORCH_CHECK_SHAPES(x, -1, w, -2, 1);
    TORCH_CHECK_SHAPES(x, -1, y, -1, 1);

    size_t bsz = x.numel() / dim;
    add_sigmoid_proj_kernel_f<<<bsz, NUM_THREADS_P, 0, stream>>>
    (
        (const float*) x.data_ptr(),
        (const half*) y.data_ptr(),
        (float*) z.data_ptr(),
        (const half*) w.data_ptr(),
        bsz,
        dim
    );

    if (graph) graph->record_param((void*) &add_sigmoid_proj_kernel_f, GP_add_sigmoid_gate_proj_x, 0);
    if (graph) graph->record_param((void*) &add_sigmoid_proj_kernel_f, GP_add_sigmoid_gate_proj_y, 1);
    if (graph) graph->record_param((void*) &add_sigmoid_proj_kernel_f, GP_add_sigmoid_gate_proj_z, 2);
    if (graph) graph->record_param((void*) &add_sigmoid_proj_kernel_f, GP_end, 0);

    cuda_check(cudaPeekAtLastError());
}

void add_sigmoid_gate_proj
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const at::Tensor& w
)
{
    add_sigmoid_gate_proj_gr(x, y, z, w, nullptr);
}