#include "reconstruct.h"
#include "exl3_common.h"
#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// Dense EXL3 CPU reconstruction, fused with both-side Hadamard transforms: emits
// W = diag(suh) . H128 . W_hat . H128 . diag(svh) (per 128x128 tile, 1/sqrt(128) per side),
// i.e. the ORIGINAL-basis weights, matching quant/reconstruct.cu's reconstruct_had_slice.
//
// Tiles are independent, so workers parallelize over 128x128 output tiles with one reusable
// stack scratch each; no per-tile heap allocation and no cross-tile state. The FWHT and the
// scale/store stages use AVX2+F16C when the host supports it; set EXL3_CPU_RECON_SCALAR=1 to
// force the scalar reference path (the numerical contract is unchanged either way).
//
// The reference test bounds deviation from the CUDA kernel at relative max error < 2e-3.

namespace {

constexpr int TILE = 128;

using exl3_common::decode_3inst;
using exl3_common::decode_state_scalar;
using exl3_common::hadamard_128_avx2;
using exl3_common::hadamard_128_scalar;
using exl3_common::HAD_SCALE;

bool force_scalar()
{
    static const bool v = std::getenv("EXL3_CPU_RECON_SCALAR") != nullptr &&
                          std::getenv("EXL3_CPU_RECON_SCALAR")[0] != '0';
    return v;
}

inline void hadamard_128(float* v, bool avx2)
{
    if (avx2) hadamard_128_avx2(v);
    else      hadamard_128_scalar(v);
}

inline void scale_store_row_scalar(at::Half* out, const float* row, float su, const at::Half* svh)
{
    for (int c = 0; c < TILE; ++c)
        out[c] = at::Half(row[c] * HAD_SCALE * su * static_cast<float>(svh[c]));
}

EXL3_CPU_TARGET_AVX2
inline void scale_store_row_avx2(at::Half* out, const float* row, float su, const at::Half* svh)
{
    const __m256 rs = _mm256_set1_ps(HAD_SCALE);
    const __m256 vsu = _mm256_set1_ps(su);
    for (int c = 0; c < TILE; c += 8)
    {
        __m256 v = _mm256_loadu_ps(row + c);
        const __m256 sv = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(svh + c)));
        v = _mm256_mul_ps(_mm256_mul_ps(v, rs), vsu);
        v = _mm256_mul_ps(v, sv);
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(out + c),
            _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
    }
}

inline void scale_store_row_scalar(float* out, const float* row, float su, const at::Half* svh)
{
    for (int c = 0; c < TILE; ++c)
        out[c] = row[c] * HAD_SCALE * su * static_cast<float>(svh[c]);
}

EXL3_CPU_TARGET_AVX2
inline void scale_store_row_avx2(float* out, const float* row, float su, const at::Half* svh)
{
    const __m256 rs = _mm256_set1_ps(HAD_SCALE);
    const __m256 vsu = _mm256_set1_ps(su);
    for (int c = 0; c < TILE; c += 8)
    {
        __m256 v = _mm256_loadu_ps(row + c);
        const __m256 sv = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(svh + c)));
        v = _mm256_mul_ps(_mm256_mul_ps(v, rs), vsu);
        v = _mm256_mul_ps(v, sv);
        _mm256_storeu_ps(out + c, v);
    }
}

template <int bits, int cb>
inline void decode_tile(const uint16_t* packed_tile, float* sub, const std::array<uint16_t, 256>& perm)
{
    for (int t = 0; t < 256; ++t)
    {
        const uint16_t s = decode_state_scalar<bits>(packed_tile, t);
        sub[perm[t]] = decode_3inst<cb>(s);
    }
}

template <int bits>
inline void decode_tile_cb(const uint16_t* packed_tile, float* sub, int cb_index,
                           const std::array<uint16_t, 256>& perm)
{
    switch (cb_index)
    {
        case 1: decode_tile<bits, 1>(packed_tile, sub, perm); break;
        case 2: decode_tile<bits, 2>(packed_tile, sub, perm); break;
        default: decode_tile<bits, 0>(packed_tile, sub, perm); break;
    }
}

struct ReconParams
{
    const uint16_t* p_packed;
    const at::Half* p_suh;
    const at::Half* p_svh;
    int k_dim;
    int n_dim;
    int packed_blocks_n;
    int n_block_offset;
    int K;
    int cb_index;
    bool avx2;
    size_t tile_words;
};

// Tile decode + both-side Hadamard + scaled store, parameterized on the output element type:
// OutT = at::Half preserves the historical fp16 rounding, OutT = float stores the unrounded
// fp32 accumulator (strictly more accurate; same math chain up to the final store).
template <typename OutT>
void reconstruct_run(const ReconParams& rp, OutT* p_out)
{
    constexpr auto perm = exl3_common::make_tc_perm();
    const int blocks_k = rp.k_dim / TILE;
    const int blocks_n = rp.n_dim / TILE;
    const int64_t total_tiles = (int64_t) blocks_k * blocks_n;

    auto process_range = [&](int64_t begin, int64_t end)
    {
        // Per-worker scratch: one dense tile plus one decoded subtile, no heap traffic
        alignas(64) float tile[TILE * TILE];
        float sub[256];

        for (int64_t ti = begin; ti < end; ++ti)
        {
            const int kt = static_cast<int>(ti / blocks_n);
            const int nt = static_cast<int>(ti % blocks_n);

            // Decode the tile's 64 packed 16x16 subtiles into row-major float
            for (int st_k = 0; st_k < 8; ++st_k)
            {
                for (int st_n = 0; st_n < 8; ++st_n)
                {
                    const int row_group = kt * 8 + st_k;
                    const int col_group = nt * 8 + st_n + rp.n_block_offset;
                    const uint16_t* tile_ptr = rp.p_packed +
                        (static_cast<size_t>(row_group) * rp.packed_blocks_n + col_group) * rp.tile_words;

                    switch (rp.K)
                    {
                        case 1: decode_tile_cb<1>(tile_ptr, sub, rp.cb_index, perm); break;
                        case 2: decode_tile_cb<2>(tile_ptr, sub, rp.cb_index, perm); break;
                        case 3: decode_tile_cb<3>(tile_ptr, sub, rp.cb_index, perm); break;
                        case 4: decode_tile_cb<4>(tile_ptr, sub, rp.cb_index, perm); break;
                        case 5: decode_tile_cb<5>(tile_ptr, sub, rp.cb_index, perm); break;
                        case 6: decode_tile_cb<6>(tile_ptr, sub, rp.cb_index, perm); break;
                        case 7: decode_tile_cb<7>(tile_ptr, sub, rp.cb_index, perm); break;
                        case 8: decode_tile_cb<8>(tile_ptr, sub, rp.cb_index, perm); break;
                    }

                    for (int r = 0; r < 16; ++r)
                        std::memcpy(
                            tile + (st_k * 16 + r) * TILE + st_n * 16,
                            sub + r * 16,
                            16 * sizeof(float));
                }
            }

            // Row-side transform (over the k axis): every column of the tile
            for (int c = 0; c < TILE; ++c)
            {
                alignas(64) float v[TILE];
                for (int r = 0; r < TILE; ++r) v[r] = tile[r * TILE + c];
                hadamard_128(v, rp.avx2);
                for (int r = 0; r < TILE; ++r) tile[r * TILE + c] = v[r] * HAD_SCALE;
            }

            // Column-side transform fused with sign scales and the final output-dtype rounding
            OutT* out_tile = p_out + (static_cast<size_t>(kt) * TILE) * rp.n_dim + nt * TILE;
            for (int r = 0; r < TILE; ++r)
            {
                float* row = tile + r * TILE;
                hadamard_128(row, rp.avx2);
                const float su = static_cast<float>(rp.p_suh[kt * TILE + r]);
                const at::Half* svh_row = rp.p_svh + nt * TILE;
                OutT* out_row = out_tile + static_cast<size_t>(r) * rp.n_dim;
                if (rp.avx2) scale_store_row_avx2(out_row, row, su, svh_row);
                else         scale_store_row_scalar(out_row, row, su, svh_row);
            }
        }
    };

    // Explicit fan-out: at::parallel_for runs serially in some torch builds, so split the
    // tile range across a short-lived std::thread pool sized by the intra-op thread count
    int n_workers = (int) std::min<int64_t>(std::max(at::get_num_threads(), 1), total_tiles);
    if (n_workers <= 1)
    {
        process_range(0, total_tiles);
    }
    else
    {
        const int64_t chunk = (total_tiles + n_workers - 1) / n_workers;
        std::vector<std::thread> workers;
        for (int w = 0; w < n_workers; ++w)
        {
            const int64_t b = (int64_t) w * chunk;
            const int64_t e = std::min(total_tiles, b + chunk);
            if (b >= e) break;
            workers.emplace_back(process_range, b, e);
        }
        for (auto& th : workers) th.join();
    }
}

} // namespace

void reconstruct_had_slice_cpu
(
    at::Tensor unpacked,
    at::Tensor packed,
    at::Tensor suh,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1,
    int64_t n_offset
)
{
    TORCH_CHECK(unpacked.is_cpu() && packed.is_cpu() && suh.is_cpu() && svh.is_cpu(),
                "reconstruct_had_slice_cpu: all tensors must be on CPU");
    TORCH_CHECK(unpacked.scalar_type() == at::kHalf || unpacked.scalar_type() == at::kFloat,
                "unpacked must be fp16 or fp32");
    TORCH_CHECK(packed.scalar_type() == at::kShort, "packed must be int16");
    TORCH_CHECK(suh.scalar_type() == at::kHalf, "suh must be fp16");
    TORCH_CHECK(svh.scalar_type() == at::kHalf, "svh must be fp16");
    TORCH_CHECK(packed.dim() == 3 && unpacked.dim() == 2, "unexpected tensor ranks");
    TORCH_CHECK(unpacked.size(0) == packed.size(0) * 16, "shape mismatch: unpacked dim0 vs packed dim0");
    TORCH_CHECK(packed.size(2) == 256 * K / 16, "packed dim2 must be K*16");
    TORCH_CHECK(K >= 1 && K <= 8, "K must be in 1..8");

    if (unpacked.numel() == 0)
        return;

    TORCH_CHECK(unpacked.size(0) % TILE == 0, "reconstruct_had: K dimension must be divisible by 128");
    TORCH_CHECK(unpacked.size(1) % TILE == 0, "reconstruct_had: N dimension must be divisible by 128");
    TORCH_CHECK(n_offset % TILE == 0, "n_offset must be divisible by 128");
    TORCH_CHECK(n_offset >= 0, "n_offset must be non-negative");
    TORCH_CHECK(n_offset + unpacked.size(1) <= packed.size(1) * 16, "reconstruct slice exceeds packed tensor bounds");
    TORCH_CHECK(suh.numel() >= unpacked.size(0), "reconstruct_had: suh size");
    TORCH_CHECK(svh.numel() >= unpacked.size(1), "reconstruct_had: svh size");

    ReconParams rp;
    rp.p_packed = reinterpret_cast<const uint16_t*>(packed.const_data_ptr<int16_t>());
    rp.p_suh = reinterpret_cast<const at::Half*>(suh.const_data_ptr<at::Half>());
    rp.p_svh = reinterpret_cast<const at::Half*>(svh.const_data_ptr<at::Half>());
    rp.k_dim = static_cast<int>(unpacked.size(0));
    rp.n_dim = static_cast<int>(unpacked.size(1));
    rp.packed_blocks_n = static_cast<int>(packed.size(1));
    rp.n_block_offset = static_cast<int>(n_offset / 16);
    rp.K = static_cast<int>(K);
    rp.cb_index = mul1 ? 2 : (mcg ? 1 : 0);
    rp.avx2 = !force_scalar();
    rp.tile_words = static_cast<size_t>(K) * 16;

    if (unpacked.scalar_type() == at::kFloat)
        reconstruct_run(rp, unpacked.data_ptr<float>());
    else
        reconstruct_run(rp, unpacked.data_ptr<at::Half>());
}
