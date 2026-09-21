#include <cuda.h>
#include <cuda_fp16.h>
#include <mutex>
#include <unordered_map>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "attention_sm120.cuh"
#include "attention_sm120_kernel.cuh"
#include "cuda_drv.h"
#include "generator/cache.cuh"
#include "util.h"
#include "util.cuh"

namespace {

constexpr int kSm120TileTokens = 64;
constexpr int kTailClearThreads = 256;

__global__ void clear_v_cache_tail_kernel(
    half* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cache_seqlens,
    int q_len,
    int n_kv_heads,
    int dim,
    int pages_per_seq
) {
    constexpr int vector_width = sizeof(uint4) / sizeof(half);
    const int sequence = blockIdx.y;
    const int total_k = cache_seqlens[sequence] + q_len;
    const int tail_tokens = (-total_k) & (kSm120TileTokens - 1);
    const int vectors_per_token = n_kv_heads * (dim / vector_width);
    const int total_vectors = tail_tokens * vectors_per_token;

    for (int vector = blockIdx.x * blockDim.x + threadIdx.x;
         vector < total_vectors;
         vector += blockDim.x * gridDim.x) {
        const int tail_token = vector / vectors_per_token;
        const int token_vector = vector - tail_token * vectors_per_token;
        const int logical_token = total_k + tail_token;
        const int physical_page = block_table[sequence * pages_per_seq + logical_token / 256];
        const int page_offset = logical_token & 255;
        const int64_t dst_vector =
            (int64_t(physical_page) * 256 * vectors_per_token) +
            (int64_t(page_offset) * vectors_per_token) + token_vector;
        reinterpret_cast<uint4*>(v_cache)[dst_vector] = make_uint4(0, 0, 0, 0);
    }
}

static void clear_v_cache_tail(
    at::Tensor& v_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    int q_len,
    cudaStream_t stream
) {
    const int bsz = static_cast<int>(cache_seqlens.size(0));
    const int n_kv_heads = static_cast<int>(v_cache.size(2));
    const int dim = static_cast<int>(v_cache.size(3));
    const int max_vectors = (kSm120TileTokens - 1) * n_kv_heads * (dim / 8);
    const dim3 blocks(CEIL_DIVIDE(max_vectors, kTailClearThreads), bsz, 1);
    clear_v_cache_tail_kernel<<<blocks, kTailClearThreads, 0, stream>>>(
        reinterpret_cast<half*>(v_cache.data_ptr()),
        block_table.data_ptr<int32_t>(),
        cache_seqlens.data_ptr<int32_t>(),
        q_len,
        n_kv_heads,
        dim,
        static_cast<int>(block_table.size(1)));
    cuda_check(cudaPeekAtLastError());
}

struct TensorMapKey {
    uintptr_t ptr;
    uint64_t dim;
    uint64_t heads;
    uint64_t tokens;
    int device;

    bool operator==(const TensorMapKey& other) const {
        return ptr == other.ptr && dim == other.dim && heads == other.heads &&
            tokens == other.tokens && device == other.device;
    }
};

struct TensorMapKeyHash {
    size_t operator()(const TensorMapKey& key) const {
        size_t h = std::hash<uintptr_t>{}(key.ptr);
        auto mix = [&h](uint64_t value) {
            h ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix(key.dim);
        mix(key.heads);
        mix(key.tokens);
        mix(static_cast<uint64_t>(key.device));
        return h;
    }
};

static CUtensorMap cached_cache_map(const at::Tensor& cache) {
    const TensorMapKey key {
        reinterpret_cast<uintptr_t>(cache.data_ptr()),
        static_cast<uint64_t>(cache.size(3)),
        static_cast<uint64_t>(cache.size(2)),
        static_cast<uint64_t>(cache.size(0) * cache.size(1)),
        cache.get_device(),
    };

    static std::mutex mutex;
    static std::unordered_map<TensorMapKey, CUtensorMap, TensorMapKeyHash> maps;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = maps.find(key);
    if (it != maps.end()) return it->second;

    const uint64_t dims[3] = {key.dim, key.heads, key.tokens};
    const uint64_t strides[2] = {
        key.dim * sizeof(half),
        key.heads * key.dim * sizeof(half),
    };
    const uint32_t box[3] = {exl3_sm120::chunk_ne, 1, exl3_sm120::config<128>::nbatch_fa};
    const uint32_t element_strides[3] = {1, 1, 1};
    CUtensorMap map{};
    cuda_check_drv(CudaDrv::instance().tensor_map_encode_tiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        3,
        cache.data_ptr(),
        dims,
        strides,
        box,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    if (maps.size() >= 4096) maps.clear();
    maps.emplace(key, map);
    return map;
}

template <typename Kernel>
static void set_dynamic_smem_once(Kernel kernel, int bytes, int device) {
    static std::mutex mutex;
    static std::unordered_map<const void*, uint64_t> configured;
    const void* key = reinterpret_cast<const void*>(kernel);
    TORCH_CHECK(device >= 0 && device < 64, "sm120_tma_attn: unsupported CUDA device index");
    const uint64_t bit = 1ull << device;
    std::lock_guard<std::mutex> lock(mutex);
    uint64_t& mask = configured[key];
    if (mask & bit) return;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    mask |= bit;
}

template <int D, int NCOLS1, int NCOLS2, bool SOFTCAP, int SPLIT>
static void launch_shape_impl(
    const CUtensorMap& map_k,
    const CUtensorMap& map_v,
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    at::Tensor& out,
    at::Tensor& parts,
    at::Tensor& meta,
    int kv_append_len,
    bool causal,
    float scale,
    float softcap,
    cudaStream_t stream
) {
    const int bsz = static_cast<int>(q.size(0));
    const int q_len = static_cast<int>(q.size(1));
    const int n_q_heads = static_cast<int>(q.size(2));
    const int n_kv_heads = static_cast<int>(k_cache.size(2));
    const int gqa_ratio = n_q_heads / n_kv_heads;
    const int ntiles_x = CEIL_DIVIDE(q_len, NCOLS1);
    const int ntiles_z_gqa = CEIL_DIVIDE(gqa_ratio, NCOLS2);
    const dim3 blocks(ntiles_x, SPLIT * ntiles_z_gqa * n_kv_heads, bsz);
    const dim3 threads(exl3_sm120::WARP_SIZE, exl3_sm120::nwarps + 1, 1);
    constexpr int shared_bytes = exl3_sm120::shared_layout<D, NCOLS1>::shared_bytes;
    auto fn = exl3_sm120::kernel<D, NCOLS1, NCOLS2, SOFTCAP, SPLIT>;
    set_dynamic_smem_once(fn, shared_bytes, q.get_device());
    fn<<<blocks, threads, shared_bytes, stream>>>(
        map_k,
        map_v,
        reinterpret_cast<const half*>(q.data_ptr()),
        block_table.data_ptr<int32_t>(),
        cache_seqlens.data_ptr<int32_t>(),
        reinterpret_cast<half*>(out.data_ptr()),
        SPLIT == 1 ? nullptr : reinterpret_cast<float*>(parts.data_ptr()),
        SPLIT == 1 ? nullptr : reinterpret_cast<float2*>(meta.data_ptr()),
        scale,
        softcap,
        q_len,
        n_q_heads,
        n_kv_heads,
        static_cast<int32_t>(block_table.size(1)),
        kv_append_len,
        causal);
    cuda_check(cudaPeekAtLastError());

    if constexpr (SPLIT == 3) {
        const dim3 combine_blocks(q_len, n_q_heads, bsz);
        const dim3 combine_threads(D, 1, 1);
        exl3_sm120::combine_results<D><<<combine_blocks, combine_threads, 3*sizeof(float2), stream>>>(
            reinterpret_cast<const float*>(parts.data_ptr()),
            reinterpret_cast<const float2*>(meta.data_ptr()),
            reinterpret_cast<half*>(out.data_ptr()));
        cuda_check(cudaPeekAtLastError());
    }
}

template <int D, int NCOLS1, int NCOLS2>
static void dispatch_softcap_split(
    const CUtensorMap& map_k,
    const CUtensorMap& map_v,
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    at::Tensor& out,
    at::Tensor& parts,
    at::Tensor& meta,
    int kv_append_len,
    bool causal,
    float scale,
    float softcap,
    int split_k,
    cudaStream_t stream
) {
    if (split_k == 3) {
        if (softcap == 0.0f) {
            launch_shape_impl<D, NCOLS1, NCOLS2, false, 3>(map_k, map_v, q, k_cache, block_table,
                cache_seqlens, out, parts, meta, kv_append_len, causal, scale, softcap, stream);
        } else {
            launch_shape_impl<D, NCOLS1, NCOLS2, true, 3>(map_k, map_v, q, k_cache, block_table,
                cache_seqlens, out, parts, meta, kv_append_len, causal, scale, softcap, stream);
        }
    } else if (softcap == 0.0f) {
        launch_shape_impl<D, NCOLS1, NCOLS2, false, 1>(map_k, map_v, q, k_cache, block_table,
            cache_seqlens, out, parts, meta, kv_append_len, causal, scale, softcap, stream);
    } else {
        launch_shape_impl<D, NCOLS1, NCOLS2, true, 1>(map_k, map_v, q, k_cache, block_table,
            cache_seqlens, out, parts, meta, kv_append_len, causal, scale, softcap, stream);
    }
}

} // namespace

bool sm120_tma_attn_supported(int device) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
    static std::mutex mutex;
    static std::unordered_map<int, bool> supported;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = supported.find(device);
    if (it != supported.end()) return it->second;

    const auto* prop = at::cuda::getDeviceProperties(device);
    if (prop->major != 12 || prop->minor != 0) {
        supported.emplace(device, false);
        return false;
    }
    const at::cuda::OptionalCUDAGuard device_guard(at::Device(at::kCUDA, device));
    cudaFuncAttributes attr{};
    const cudaError_t err = cudaFuncGetAttributes(
        &attr, exl3_sm120::kernel<128, 8, 8, false, 1>);
    if (err == cudaErrorNoKernelImageForDevice || err == cudaErrorInvalidDeviceFunction) {
        (void) cudaGetLastError();
        supported.emplace(device, false);
        return false;
    }
    cuda_check(err);
    // All attention instantiations share this TU's architectures. A lower-arch PTX
    // image can JIT to sm_120, but its __CUDA_ARCH__-guarded body was compiled out.
    const bool available = attr.ptxVersion == 120 && attr.binaryVersion == 120;
    supported.emplace(device, available);
    return available;
#else
    return false;
#endif
}

at::Tensor sm120_tma_attn_paged(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor block_table,
    at::Tensor cache_seqlens,
    bool causal,
    float sm_scale,
    float softcap,
    int split_k,
    int q_group_mode
) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(q.is_cuda(), "sm120_tma_attn: q must be CUDA");
    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k, kHalf);
    TORCH_CHECK_DTYPE(v, kHalf);
    TORCH_CHECK_DTYPE(k_cache, kHalf);
    TORCH_CHECK_DTYPE(v_cache, kHalf);
    TORCH_CHECK_DTYPE(block_table, kInt);
    TORCH_CHECK_DTYPE(cache_seqlens, kInt);
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
        "sm120_tma_attn: q/k/v must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(),
        "sm120_tma_attn: cache tensors must be contiguous");
    TORCH_CHECK(block_table.is_contiguous() && cache_seqlens.is_contiguous(),
        "sm120_tma_attn: page metadata must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
        "sm120_tma_attn: q/k/v must be rank 4");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
        "sm120_tma_attn: cache tensors must be rank 4");
    TORCH_CHECK(block_table.dim() == 2 && cache_seqlens.dim() == 1,
        "sm120_tma_attn: invalid page metadata rank");

    TORCH_CHECK(sm120_tma_attn_supported(q.get_device()),
        "sm120_tma_attn requires a compiled SM120 kernel and compute capability 12.0");

    const int64_t bsz = q.size(0);
    const int64_t q_len = q.size(1);
    const int64_t n_q_heads = q.size(2);
    const int64_t dim = q.size(3);
    const int64_t n_kv_heads = k.size(2);
    TORCH_CHECK(q_len > 0, "sm120_tma_attn: q_len must be positive");
    TORCH_CHECK(k.size(0) == bsz && v.size(0) == bsz && k.size(1) == q_len && v.size(1) == q_len,
        "sm120_tma_attn: this path requires k/v append length to equal q_len");
    TORCH_CHECK(k.size(2) == n_kv_heads && v.size(2) == n_kv_heads &&
        k.size(3) == dim && v.size(3) == dim, "sm120_tma_attn: incompatible k/v shape");
    TORCH_CHECK(n_q_heads % n_kv_heads == 0, "sm120_tma_attn: invalid GQA ratio");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes() && k_cache.size(1) == 256 &&
        k_cache.size(2) == n_kv_heads && k_cache.size(3) == dim,
        "sm120_tma_attn: incompatible paged cache shape");
    TORCH_CHECK(block_table.size(0) == bsz && cache_seqlens.size(0) == bsz,
        "sm120_tma_attn: page metadata batch mismatch");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
        q.device() == k_cache.device() && q.device() == v_cache.device() &&
        q.device() == block_table.device() && q.device() == cache_seqlens.device(),
        "sm120_tma_attn: all tensors must be on the same device");
    TORCH_CHECK(dim == 128 || dim == 256 || dim == 512,
        "sm120_tma_attn: head_dim must be 128, 256 or 512");
    TORCH_CHECK(split_k == 1 || split_k == 3, "sm120_tma_attn: split_k must be 1 or 3");
    TORCH_CHECK(dim != 128 || split_k == 1, "sm120_tma_attn: D128 only supports split_k=1");
    TORCH_CHECK(q_group_mode == 0 || q_group_mode == 2 || q_group_mode == 8,
        "sm120_tma_attn: q_group_mode must be 0, 2 or 8");

    paged_kv_cache_update(k, v, k_cache, v_cache, block_table, cache_seqlens);
    clear_v_cache_tail(v_cache, block_table, cache_seqlens, static_cast<int>(q_len), stream);

    const CUtensorMap map_k = cached_cache_map(k_cache);
    const CUtensorMap map_v = cached_cache_map(v_cache);
    at::Tensor out = at::empty_like(q);
    const int64_t rows = bsz*q_len*n_q_heads;
    at::Tensor parts = split_k == 3 ? at::empty({rows, 3, dim}, q.options().dtype(at::kFloat)) : q;
    at::Tensor meta = split_k == 3 ? at::empty({rows, 3, 2}, q.options().dtype(at::kFloat)) : q;

    float scale = sm_scale;
    if (softcap != 0.0f) scale /= softcap;
    const int ratio = static_cast<int>(n_q_heads/n_kv_heads);
    int mode = q_group_mode;
    if (mode == 0) mode = ratio % 8 == 0 ? 8 : 2;

    if (dim == 128) {
        dispatch_softcap_split<128, 8, 8>(map_k, map_v, q, k_cache, block_table, cache_seqlens,
            out, parts, meta, static_cast<int>(q_len), causal, scale, softcap, split_k, stream);
    } else if (dim == 256 && mode == 8) {
        dispatch_softcap_split<256, 8, 8>(map_k, map_v, q, k_cache, block_table, cache_seqlens,
            out, parts, meta, static_cast<int>(q_len), causal, scale, softcap, split_k, stream);
    } else if (dim == 256) {
        dispatch_softcap_split<256, 32, 2>(map_k, map_v, q, k_cache, block_table, cache_seqlens,
            out, parts, meta, static_cast<int>(q_len), causal, scale, softcap, split_k, stream);
    } else {
        TORCH_CHECK(mode == 8, "sm120_tma_attn: D512 requires q_group_mode=8");
        dispatch_softcap_split<512, 8, 8>(map_k, map_v, q, k_cache, block_table, cache_seqlens,
            out, parts, meta, static_cast<int>(q_len), causal, scale, softcap, split_k, stream);
    }
    return out;
#else
    TORCH_CHECK(false, "sm120_tma_attn requires CUDA toolkit 12.8 or newer");
#endif
}
