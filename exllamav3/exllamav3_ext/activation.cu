#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include "activation.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "compat.cuh"
#include <cmath>

#define NUM_THREADS 256
#define ACT_SILU 0
#define ACT_GELU 1
#define ACT_RELU2 2

#include "activation_kernels.cuh"

// silu(x) * y -> z, in-place if z == x or z == y

void silu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

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
    }
}

// silu(x) * y -> z, in-place if z == x or z == y

void gelu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

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
        cuda_check(cudaPeekAtLastError());
    }
}

// relu^2(x) * y -> z

void relu2_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

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
    }
}

// xielu(x, alpha_p, alpha_n) -> z
// alpha_p and alpha_n must be CPU tensors

void xielu
(
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& alpha_p,
    const at::Tensor& alpha_n
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

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
    }
    else
    {
        TORCH_CHECK(false, "xielu not implemented for float16 input dtype");
        // xielu_kernel_h<<<blocks, NUM_THREADS, 0, stream>>>
        // (
        //     (const half*) x.data_ptr(),
        //     (half*) y.data_ptr(),
        //     numel,
        //     alpha_p_v,
        //     alpha_n_v
        // );
    }
}

// x * sigmoid(y) + z -> z

void add_sigmoid_gate
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

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
}
