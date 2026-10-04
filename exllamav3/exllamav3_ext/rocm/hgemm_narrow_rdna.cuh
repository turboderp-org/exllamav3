#pragma once

// Narrow-output hgemm for ROCm: m <= 8 rows against N <= 256 columns, any K, fp16 or fp32 C. rocBLAS answers these
// shapes (e.g. BC_Attention's headwise gate projection, one (1 x hidden) @ (hidden x heads) product per layer per
// token) with a single wide-tile workgroup walking K serially, far off the bandwidth bound, while hipBLASLt, which
// handles them well, loses on wide and tall shapes and has no per-call switch. A split-K GEMV covers the narrow
// regime and leaves everything else on the library: one block per K slice writes fp32 partials into the DevCtx
// workspace (the BLAS workspace, used on the same stream), a second launch sums them into C in its dtype and row
// stride. No allocations, so it is also safe under graph capture.

#define HGEMM_NARROW_MAX_M 8
#define HGEMM_NARROW_MAX_N 256
#define HGEMM_NARROW_THREADS 256
#define HGEMM_NARROW_MAX_SPLITS 64
#define HGEMM_NARROW_MIN_K_PER_SPLIT 64

// One block per K slice: thread t owns column t % n and row lane t / n; the lanes stride over the slice's
// rows of B and are reduced through LDS. A (m x K) is tiny and every thread of a lane group reads the same
// element, a broadcast load
__global__ void hgemm_narrow_partial_kernel
(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ ws,
    int size_m,
    int size_k,
    int size_n,
    int k_per_split
)
{
    __shared__ float red[HGEMM_NARROW_MAX_M][HGEMM_NARROW_THREADS];
    const int split = blockIdx.x;
    const int k0 = split * k_per_split;
    const int k1 = min(size_k, k0 + k_per_split);
    const int lanes = HGEMM_NARROW_THREADS / size_n;
    const int t = threadIdx.x;
    const int col = t % size_n;
    const int lane = t / size_n;

    float acc[HGEMM_NARROW_MAX_M];
    #pragma unroll
    for (int r = 0; r < HGEMM_NARROW_MAX_M; ++r) acc[r] = 0.0f;

    if (lane < lanes)
    {
        for (int k = k0 + lane; k < k1; k += lanes)
        {
            float bv = __half2float(b[(int64_t) k * size_n + col]);
            #pragma unroll
            for (int r = 0; r < HGEMM_NARROW_MAX_M; ++r)
                if (r < size_m) acc[r] += __half2float(a[(int64_t) r * size_k + k]) * bv;
        }
    }
    #pragma unroll
    for (int r = 0; r < HGEMM_NARROW_MAX_M; ++r) red[r][t] = acc[r];
    __syncthreads();

    if (t < size_n)
    {
        for (int r = 0; r < size_m; ++r)
        {
            float s = 0.0f;
            for (int l = 0; l < lanes; ++l) s += red[r][l * size_n + t];
            ws[((int64_t) split * size_m + r) * size_n + t] = s;
        }
    }
}

__device__ __forceinline__ void hgemm_narrow_store(half* c, float v) { *c = __float2half(v); }
__device__ __forceinline__ void hgemm_narrow_store(float* c, float v) { *c = v; }

// Sum the slice partials in slice order and write C with its row stride, in its dtype
template <typename T>
__global__ void hgemm_narrow_reduce_kernel
(
    const float* __restrict__ ws,
    T* __restrict__ c,
    int size_m,
    int size_n,
    int splits,
    int64_t c_stride_m
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size_m * size_n) return;
    const int r = i / size_n;
    const int col = i % size_n;
    float s = 0.0f;
    for (int sp = 0; sp < splits; ++sp) s += ws[((int64_t) sp * size_m + r) * size_n + col];
    hgemm_narrow_store(c + (int64_t) r * c_stride_m + col, s);
}

// A and B row-major and contiguous, as hgemm_gemmex_impl already requires
static bool hgemm_narrow_try
(
    const half* a_ptr,
    const half* b_ptr,
    void* c_ptr,
    bool output_fp32,
    int size_m,
    int size_k,
    int size_n,
    int64_t c_stride_m,
    int device,
    cudaStream_t stream
)
{
    if (size_m < 1 || size_m > HGEMM_NARROW_MAX_M || size_n < 1 || size_n > HGEMM_NARROW_MAX_N || size_k < 1)
        return false;

    const int k_per_split = MAX(HGEMM_NARROW_MIN_K_PER_SPLIT, CEIL_DIVIDE(size_k, HGEMM_NARROW_MAX_SPLITS));
    const int splits = CEIL_DIVIDE(size_k, k_per_split);
    float* ws = (float*) DevCtx::instance().get_ws(device);
    if (!ws || (size_t) splits * size_m * size_n * sizeof(float) > (size_t) WORKSPACE_SIZE) return false;

    hgemm_narrow_partial_kernel<<<splits, HGEMM_NARROW_THREADS, 0, stream>>>
        (a_ptr, b_ptr, ws, size_m, size_k, size_n, k_per_split);

    const int blocks = CEIL_DIVIDE(size_m * size_n, HGEMM_NARROW_THREADS);
    if (output_fp32)
        hgemm_narrow_reduce_kernel<float><<<blocks, HGEMM_NARROW_THREADS, 0, stream>>>
            (ws, (float*) c_ptr, size_m, size_n, splits, c_stride_m);
    else
        hgemm_narrow_reduce_kernel<half><<<blocks, HGEMM_NARROW_THREADS, 0, stream>>>
            (ws, (half*) c_ptr, size_m, size_n, splits, c_stride_m);
    cuda_check(cudaPeekAtLastError());
    return true;
}
