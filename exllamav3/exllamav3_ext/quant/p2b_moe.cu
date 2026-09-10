// Port of vcruz305/vllm-exl3 csrc/p2b_moe.cu (MIT, Copyright Mia AI Lab / vcruz305, see
// THIRD_PARTY notices in that repository) to the exllamav3 tree, extended from one input row
// per launch to a full row-expert slot table so a whole verify window (bsz * top_k slots)
// runs as ONE cooperative launch instead of one per row.
//
// Four grid-synchronized phases over all active expert slots: input Hadamard, batched
// gate/up GEMV (exl3_gemv_kernel.cuh MMA path), SwiGLU + down-input Hadamard, batched down
// GEMV, then output Hadamard and weighted atomic accumulation. Grid size is occupancy-derived;
// every phase grid-strides over its work items.
//
// Dimensions are runtime parameters (the reference build hardcoded hidden 4096 / inter 2048);
// hidden and inter must be multiples of 128 (Hadamard warp tiling). MCG codebook only, K in
// {2, 3, 4} and uniform across gate/up/down, matching MultiLinear pointer tables.

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>

#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "hadamard_inner.cuh"

#include "p2b_moe.cuh"

namespace cg = cooperative_groups;

template <int bits, int cb, int CFG>
__device__ __forceinline__ void run_gemv_tile(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    float (*sh_red)[1][32])
{
    // Verbatim extraction of exllamav3's exl3_gemv_kernel group-loop body (MMODE = 0,
    // SMEM_STAGE = false), the production-validated m=1 decode GEMV. The reference port's
    // tile was textually close but numerically wrong against this tree's kernel.
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const bool r0_ok = lane < 4;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    } else if constexpr (bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_col] : hzero;
            a23[0] = r0_ok ? A2[a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                } else if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Cross-warp reduction over the k splits: lane l holds row l/4, cols
    // tile*16 + frag*8 + 2*(l%4) (+1) -- exactly the production kernel's mapping
    {
        const int c0 = 2 * (lane & 3);
        if (lane < 4)
        {
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                {
                    const int col = t * 16 + f * 8 + c0;
                    sh_red[warp][0][col + 0] = acc0[t][f].x;
                    sh_red[warp][0][col + 1] = acc0[t][f].y;
                }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS; idx += THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][0][idx];
        const int col = group * COLS + idx;
        C[col] = __float2half_rn(sum);
    }
    __syncthreads();
}

template <int BITS, int CB>
__global__ __launch_bounds__(512)
void p2b_moe_batched_kernel(
    const half* __restrict__ x,          // (m, hidden)
    const int64_t* __restrict__ gt_ptrs, // per-expert pointer tables
    const int64_t* __restrict__ gu_ptrs,
    const int64_t* __restrict__ gv_ptrs,
    const int64_t* __restrict__ ut_ptrs,
    const int64_t* __restrict__ uu_ptrs,
    const int64_t* __restrict__ uv_ptrs,
    const int64_t* __restrict__ dt_ptrs,
    const int64_t __restrict__ (*du_ptrs),
    const int64_t* __restrict__ dv_ptrs,
    const int32_t* __restrict__ ids,     // (slots,) local expert id per slot
    const int32_t* __restrict__ rows,    // (slots,) row index per slot
    const half* __restrict__ rw,         // (slots,) routing weight per slot
    half* __restrict__ gate,             // (slots, inter)
    half* __restrict__ up,               // (slots, inter)
    half* __restrict__ down,             // (slots, hidden)
    half* __restrict__ out,              // (m, hidden)
    half* __restrict__ had_gate,         // (slots, hidden)
    half* __restrict__ had_up,           // (slots, hidden)
    half* __restrict__ had_down,         // (slots, inter)
    float* __restrict__ accum,           // (m, hidden)
    int slots,
    int m,
    int hidden,
    int inter,
    int stop_after)
{
    auto grid = cg::this_grid();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int ntiles_gate = inter / 16;
    const int kslices_gate = hidden / 16;
    const int num_groups_gate = inter / 32;

    const int ntiles_down = hidden / 16;
    const int kslices_down = inter / 16;
    const int num_groups_down = hidden / 32;

    __shared__ float sh_red[16][1][32];

    // Zero accum
    for (int j = tid; j < m * hidden; j += total_threads)
        accum[j] = 0.0f;

    // Phase 1: Input Hadamard for Gate and Up across all slots
    {
        int warps_per_exp = hidden / 128;
        int total_warps = slots * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int s = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[s];
            int row = rows[s];
            const half* gu_e = reinterpret_cast<const half*>(gu_ptrs[src]);
            const half* uu_e = reinterpret_cast<const half*>(uu_ptrs[src]);
            half* hg_e = had_gate + (size_t) s * hidden;
            half* hu_e = had_up + (size_t) s * hidden;

            had_hf_r_128_inner<true, false>(x + (size_t) row * hidden + w * 128, hg_e + w * 128, gu_e + (w * 128) % hidden, 0.088388347648f);
            had_hf_r_128_inner<true, false>(x + (size_t) row * hidden + w * 128, hu_e + w * 128, uu_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 2: Batched Gate & Up GEMV across all slots
    if (stop_after >= 2)
    {
        int total_work = 2 * slots * num_groups_gate;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int is_up = item & 1;
            int rem = item >> 1;
            int s = rem / num_groups_gate;
            int group = rem % num_groups_gate;
            int src = ids[s];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>((is_up ? had_up : had_gate) + (size_t) s * hidden);
            half* C = (is_up ? up : gate) + (size_t) s * inter;

            run_gemv_tile<BITS, CB, 0>(B32, A2, C, kslices_gate, hidden, group, ntiles_gate, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Epilogue Hadamard on Gate and Up
    if (stop_after >= 2)
    {
        int warps_per_exp = inter / 128;
        int total_warps = slots * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int s = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[s];
            const half* gv_e = reinterpret_cast<const half*>(gv_ptrs[src]);
            const half* uv_e = reinterpret_cast<const half*>(uv_ptrs[src]);
            half* gp_e = gate + (size_t) s * inter;
            half* up_e = up + (size_t) s * inter;

            had_hf_r_128_inner<false, true>(gp_e + w * 128, gp_e + w * 128, gv_e + (w * 128) % inter, 0.088388347648f);
            had_hf_r_128_inner<false, true>(up_e + w * 128, up_e + w * 128, uv_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 3: SwiGLU activation + Down input Hadamard across all slots
    {
        int total_elements = slots * inter;
        for (int j = tid; j < total_elements; j += total_threads) {
            float g = __half2float(gate[j]);
            float u = __half2float(up[j]);
            float s = g / (1.0f + expf(-g));
            had_down[j] = __float2half(s * u);
        }
        grid.sync();

        int warps_per_exp = inter / 128;
        int total_warps = slots * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int s = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[s];
            const half* du_e = reinterpret_cast<const half*>(du_ptrs[src]);
            half* hd_e = had_down + (size_t) s * inter;

            had_hf_r_128_inner<true, false>(hd_e + w * 128, hd_e + w * 128, du_e + (w * 128) % inter, 0.088388347648f);
        }
        grid.sync();
    }

    // Phase 4: Batched Down GEMV across all slots
    if (stop_after >= 4)
    {
        int total_work = slots * num_groups_down;
        for (int item = blockIdx.x; item < total_work; item += gridDim.x) {
            int s = item / num_groups_down;
            int group = item % num_groups_down;
            int src = ids[s];

            const uint32_t* B32 = reinterpret_cast<const uint32_t*>(dt_ptrs[src]);
            const half2* A2 = reinterpret_cast<const half2*>(had_down + (size_t) s * inter);
            half* C = down + (size_t) s * hidden;

            run_gemv_tile<BITS, CB, 0>(B32, A2, C, kslices_down, inter, group, ntiles_down, warp, lane, sh_red);
        }
        grid.sync();
    }

    // Down output Hadamard and atomic accumulation into accum
    if (stop_after >= 4)
    {
        int warps_per_exp = hidden / 128;
        int total_warps = slots * warps_per_exp;
        int this_warp = warp + (blockDim.x / 32) * blockIdx.x;
        int grid_warps = gridDim.x * (blockDim.x / 32);

        for (; this_warp < total_warps; this_warp += grid_warps) {
            int s = this_warp / warps_per_exp;
            int w = this_warp % warps_per_exp;
            int src = ids[s];
            const half* dv_e = reinterpret_cast<const half*>(dv_ptrs[src]);
            half* dp_e = down + (size_t) s * hidden;

            had_hf_r_128_inner<false, true>(dp_e + w * 128, dp_e + w * 128, dv_e + (w * 128) % hidden, 0.088388347648f);
        }
        grid.sync();

        // Weighted reduction into accum
        int total_elements = slots * hidden;
        for (int j = tid; j < total_elements; j += total_threads) {
            int s = j / hidden;
            int col = j % hidden;
            float w = __half2float(rw[s]);
            atomicAdd(accum + (size_t) rows[s] * hidden + col, w * __half2float(down[j]));
        }
        grid.sync();
    }

    // Write back to out
    if (stop_after >= 4)
    for (int j = tid; j < m * hidden; j += total_threads) {
        out[j] = __float2half(accum[j]);
    }
}

template <int BITS, int CB>
static void launch_moe_batched(
    const at::Tensor& x, const at::Tensor& gt, const at::Tensor& gu,
    const at::Tensor& gv, const at::Tensor& ut, const at::Tensor& uu,
    const at::Tensor& uv, const at::Tensor& dt, const at::Tensor& du,
    const at::Tensor& dv, const at::Tensor& ids, const at::Tensor& rows,
    const at::Tensor& rw, at::Tensor& out, at::Tensor& gate, at::Tensor& up,
    at::Tensor& down, at::Tensor& had_gate, at::Tensor& had_up,
    at::Tensor& had_down, at::Tensor& accum,
    int slots, int m, int hidden, int inter, int stop_after = 4)
{
    int dev = 0, sms = 0, resident = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void* kernel = (void*) p2b_moe_batched_kernel<BITS, CB>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, 0);
    const int grid = std::max(1, resident * sms);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const half* xp = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const int64_t* gtp = gt.data_ptr<int64_t>();
    const int64_t* gup = gu.data_ptr<int64_t>();
    const int64_t* gvp = gv.data_ptr<int64_t>();
    const int64_t* utp = ut.data_ptr<int64_t>();
    const int64_t* uup = uu.data_ptr<int64_t>();
    const int64_t* uvp = uv.data_ptr<int64_t>();
    const int64_t* dtp = dt.data_ptr<int64_t>();
    const int64_t* dup = du.data_ptr<int64_t>();
    const int64_t* dvp = dv.data_ptr<int64_t>();
    const int32_t* idp = ids.data_ptr<int32_t>();
    const int32_t* rowp = rows.data_ptr<int32_t>();
    const half* rwp = reinterpret_cast<const half*>(rw.data_ptr<c10::Half>());

    half* gp = reinterpret_cast<half*>(gate.data_ptr<c10::Half>());
    half* up_p = reinterpret_cast<half*>(up.data_ptr<c10::Half>());
    half* dp = reinterpret_cast<half*>(down.data_ptr<c10::Half>());
    half* op = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* hg_p = reinterpret_cast<half*>(had_gate.data_ptr<c10::Half>());
    half* hu_p = reinterpret_cast<half*>(had_up.data_ptr<c10::Half>());
    half* hd_p = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    float* accp = accum.data_ptr<float>();

    int e = slots;
    void* args[] = {
        (void*)&xp, (void*)&gtp, (void*)&gup, (void*)&gvp,
        (void*)&utp, (void*)&uup, (void*)&uvp,
        (void*)&dtp, (void*)&dup, (void*)&dvp,
        (void*)&idp, (void*)&rowp, (void*)&rwp,
        (void*)&gp, (void*)&up_p, (void*)&dp, (void*)&op,
        (void*)&hg_p, (void*)&hu_p, (void*)&hd_p, (void*)&accp,
        (void*)&e, (void*)&m, (void*)&hidden, (void*)&inter, (void*)&stop_after
    };

    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(512), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rows, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg, bool mul1,
    int64_t hidden, int64_t inter)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "p2b fused MoE requires CUDA fp16 input");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "p2b fused MoE output must be CUDA fp16");
    TORCH_CHECK(kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4), "unsupported fused MoE K");
    TORCH_CHECK(!(mcg && mul1), "specified both mcg and mul1");
    // Codebook select, mirroring exl3_gemv.cu: 0 = default (neither flag), 1 = MCG, 2 = mul1
    const int cb = mcg ? 1 : (mul1 ? 2 : 0);
    TORCH_CHECK(hidden % 128 == 0 && inter % 128 == 0, "p2b fused MoE requires hidden/inter divisible by 128");
    const int slots = static_cast<int>(ids.numel());
    const int m = static_cast<int>(x.numel() / x.size(-1));

    auto gate = at::empty({slots, inter}, x.options());
    auto up = at::empty({slots, inter}, x.options());
    auto down = at::empty({slots, hidden}, x.options());
    auto had_gate = at::empty({slots, hidden}, x.options());
    auto had_up = at::empty({slots, hidden}, x.options());
    auto had_down = at::empty({slots, inter}, x.options());
    auto accum = at::zeros({m, hidden}, x.options().dtype(at::kFloat));

    #define P2B_DISPATCH(BITS, CB) launch_moe_batched<BITS, CB>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rows, rw, out, gate, up, down, had_gate, had_up, had_down, accum, slots, m, hidden, inter)
    #define P2B_SWITCH_CB(BITS) \
        switch (cb) { case 0: P2B_DISPATCH(BITS, 0); break; \
                      case 1: P2B_DISPATCH(BITS, 1); break; \
                      case 2: P2B_DISPATCH(BITS, 2); break; }
    if (kg == 4) P2B_SWITCH_CB(4)
    else if (kg == 3) P2B_SWITCH_CB(3)
    else if (kg == 2) P2B_SWITCH_CB(2)
    #undef P2B_SWITCH_CB
    #undef P2B_DISPATCH

    return out;
}

std::vector<at::Tensor> p2b_stage_debug_cuda(const at::Tensor& x,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rows, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg, bool mul1,
    int64_t hidden, int64_t inter, int64_t stop_after)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "p2b debug requires CUDA fp16 input");
    TORCH_CHECK(kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4), "unsupported fused MoE K");
    TORCH_CHECK(hidden % 128 == 0 && inter % 128 == 0, "p2b requires hidden/inter divisible by 128");
    const int slots = static_cast<int>(ids.numel());
    const int m = static_cast<int>(x.numel() / x.size(-1));

    auto gate = at::empty({slots, inter}, x.options());
    auto up = at::empty({slots, inter}, x.options());
    auto down = at::empty({slots, hidden}, x.options());
    auto had_gate = at::empty({slots, hidden}, x.options());
    auto had_up = at::empty({slots, hidden}, x.options());
    auto had_down = at::empty({slots, inter}, x.options());
    auto accum = at::zeros({m, hidden}, x.options().dtype(at::kFloat));
    auto out = at::empty({m, hidden}, x.options());

    TORCH_CHECK(!(mcg && mul1), "specified both mcg and mul1");
    const int cb = mcg ? 1 : (mul1 ? 2 : 0);
    #define P2B_DBG(BITS, CB) launch_moe_batched<BITS, CB>(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rows, rw, out, gate, up, down, had_gate, had_up, had_down, accum, slots, m, hidden, inter, (int)stop_after)
    #define P2B_DBG_CB(BITS) \
        switch (cb) { case 0: P2B_DBG(BITS, 0); break; \
                      case 1: P2B_DBG(BITS, 1); break; \
                      case 2: P2B_DBG(BITS, 2); break; }
    if (kg == 4) P2B_DBG_CB(4)
    else if (kg == 3) P2B_DBG_CB(3)
    else P2B_DBG_CB(2)
    #undef P2B_DBG_CB
    #undef P2B_DBG
    return {had_gate, had_up, gate, up, had_down, down, out};
}
