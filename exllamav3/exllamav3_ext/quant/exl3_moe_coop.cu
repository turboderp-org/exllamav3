// Fused decode-shaped MoE path (bsz 1..MAX_BSZN): launcher and tensor front end. Kernels in
// exl3_moe_coop_kernel.cuh, instantiated per (K, codebook) in comp_units/exl3_moe_coop_inst_k*.cu

#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <cmath>

#include "../util.h"
#include "../util.cuh"
#include "exl3_devctx.cuh"
#include "exl3_moe_coop.cuh"
#include "comp_units/exl3_moe_coop_instances.cuh"
#include "bits_k.cuh"
#define EXL3_MOE_COOP_DEFINE_ROT
#include "exl3_moe_coop_kernel.cuh"

static MoeCoopKernel moe_coop_kernel_a(float K_, int cb, int Hi, bool wide)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_a_h1(Hi, wide);
            case 2: return exl3_moe_coop_kernel_a_h2(Hi, wide);
            default: return exl3_moe_coop_kernel_a_h3(Hi, wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_a_k1(cb, Hi, wide);
        case 2: return exl3_moe_coop_kernel_a_k2(cb, Hi, wide);
        case 3: return exl3_moe_coop_kernel_a_k3(cb, Hi, wide);
        case 4: return exl3_moe_coop_kernel_a_k4(cb, Hi, wide);
        case 5: return exl3_moe_coop_kernel_a_k5(cb, Hi, wide);
        case 6: return exl3_moe_coop_kernel_a_k6(cb, Hi, wide);
        case 7: return exl3_moe_coop_kernel_a_k7(cb, Hi, wide);
        default: return exl3_moe_coop_kernel_a_k8(cb, Hi, wide);
    }
}

static MoeCoopKernel moe_coop_kernel_b(float K_, int cb, bool wide)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_b_h1(wide);
            case 2: return exl3_moe_coop_kernel_b_h2(wide);
            default: return exl3_moe_coop_kernel_b_h3(wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_b_k1(cb, wide);
        case 2: return exl3_moe_coop_kernel_b_k2(cb, wide);
        case 3: return exl3_moe_coop_kernel_b_k3(cb, wide);
        case 4: return exl3_moe_coop_kernel_b_k4(cb, wide);
        case 5: return exl3_moe_coop_kernel_b_k5(cb, wide);
        case 6: return exl3_moe_coop_kernel_b_k6(cb, wide);
        case 7: return exl3_moe_coop_kernel_b_k7(cb, wide);
        default: return exl3_moe_coop_kernel_b_k8(cb, wide);
    }
}

// Split-k factor: blocks per column chunk, partial outputs summed by the last-arriving block.
// Measured ineffective.
static int moe_coop_pick_ksplit(int max_split)
{
    static int forced = -2;
    if (forced == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_KSPLIT");
        forced = env ? atoi(env) : 0;
    }
    return forced > 0 ? std::min(forced, max_split) : 1;
}

// Tile geometry per stage: the wide (128-column, 4-way k-split) tile for long k, the narrow one
// otherwise. EXL3_MOE_COOP_WIDE = 0 / 1 forces one geometry for every stage (testing)
static int moe_coop_wide_mode()
{
    static int mode = -2;
    if (mode == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_WIDE");
        mode = env ? atoi(env) : -1;
    }
    return mode;
}

static bool moe_coop_pick_wide(int kslices, int slots, int device)
{
    const int mode = moe_coop_wide_mode();
    if (mode >= 0) return mode != 0;
    if (DevCtx::instance().get_cc(device) < CC_BLACKWELL) return true;
    return kslices >= 256 || (kslices >= 128 && slots >= 32);
}

// Dynamic shared memory above the 48 KB default needs an opt-in per kernel (once per process)
static void moe_coop_smem_optin(void* kernel, int smem)
{
    if (smem <= 48 * 1024) return;
    static std::map<void*, int> done;
    auto it = done.find(kernel);
    if (it != done.end() && it->second >= smem) return;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    done[kernel] = smem;
}

void exl3_moe_coop_launch(const MoeCoopParams& p_in, float K_gu, float K_d, int cb, int device, cudaStream_t stream)
{
    MoeCoopParams p = p_in;
    { static int dbg = std::getenv("EXL3_MOE_COOP_DBG") ? atoi(std::getenv("EXL3_MOE_COOP_DBG")) : 0; p.dbg = dbg; }
    const int slots = p.bsz * p.topk;
    const int nproj = p.gated ? 2 : 1;
    p.a_global = p.bsz > 1;

    const bool wide_a = moe_coop_pick_wide(p.Hi / 16, slots, device);
    const bool wide_b = moe_coop_pick_wide(p.I / 16, slots, device);
    MoeCoopKernel ka = moe_coop_kernel_a(K_gu, cb, p.Hi, wide_a);
    MoeCoopKernel kb = moe_coop_kernel_b(K_d, cb, wide_b);
    moe_coop_smem_optin(ka.kernel, ka.smem);
    moe_coop_smem_optin(kb.kernel, kb.smem);

    const int base_a = slots * nproj * (p.I / (wide_a ? 128 : MOE_COOP_COLS));
    const int base_b = slots * (p.Ho / (wide_b ? 128 : MOE_COOP_COLS));
    const int max_split = std::max(1, p.slots_max / slots);
    p.ksplit_a = moe_coop_pick_ksplit(max_split);
    p.ksplit_b = moe_coop_pick_ksplit(max_split);
    const int grid_a = base_a * p.ksplit_a;
    const int grid_b = base_b * p.ksplit_b;

    void* args[] = { (void*) &p };
    if (p.a_global)
    {
        const int rot_items = slots * (p.Hi / 128) * nproj;
        const int rot_grid = CEIL_DIVIDE(rot_items, MOE_COOP_THREADS / 32);
        exl3_moe_coop_ns::exl3_moe_coop_rot_kernel<<<rot_grid, MOE_COOP_THREADS, 0, stream>>>(p);
    }
    cuda_check(cudaLaunchKernel(ka.kernel, dim3(grid_a), dim3(MOE_COOP_THREADS), args, ka.smem, stream));
    cuda_check(cudaLaunchKernel(kb.kernel, dim3(grid_b), dim3(MOE_COOP_THREADS), args, kb.smem, stream));
    cuda_check(cudaPeekAtLastError());
}

MoeCoopParams exl3_moe_coop_prepare
(
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_gate_w,
    float& K_gu, float& K_d, int& cb
)
{
    TORCH_CHECK_DTYPE(ctr, kInt);
    TORCH_CHECK(ctr.is_contiguous(), "exl3_moe_coop: ctr must be contiguous");
    for (const at::Tensor* t : { &g_trellis, &g_suh, &g_svh, &u_trellis, &u_suh, &u_svh, &d_trellis, &d_suh, &d_svh })
        TORCH_CHECK(t->scalar_type() == at::kLong && t->is_contiguous(), "exl3_moe_coop: pointer tables must be contiguous int64");
    TORCH_CHECK(!(mcg && mul1), "exl3_moe_coop: specified both mcg and mul1");
    TORCH_CHECK(Kg >= 1 && Kg <= 8 && Ku >= 1 && Ku <= 8 && Kd >= 1 && Kd <= 8, "exl3_moe_coop: K out of range");
    TORCH_CHECK(!gated || Kg == Ku, "exl3_moe_coop: gate and up must share a bit width");
    TORCH_CHECK(act >= 0 && act <= 3, "exl3_moe_coop: unknown activation");

    MoeCoopParams p = {};
    p.Hi = Hi;
    p.I = (int) gu_u.size(-1);
    p.Ho = (int) d_out.size(-1);
    p.H_out = (int) out.size(-1);
    TORCH_CHECK(p.Hi % 128 == 0 && p.I % 128 == 0 && p.Ho % 128 == 0 && p.H_out <= p.Ho, "exl3_moe_coop: Hi/I/Ho shape");

    TORCH_CHECK_DTYPE(had_g, kHalf);
    TORCH_CHECK_DTYPE(had_u, kHalf);
    TORCH_CHECK_DTYPE(act_out, kHalf);
    TORCH_CHECK_DTYPE(d_out, kFloat);
    TORCH_CHECK_DTYPE(out, kFloat);
    p.gu_f32 = gu_u.scalar_type() == at::kFloat;
    if (!p.gu_f32) TORCH_CHECK_DTYPE(gu_u, kHalf);
    TORCH_CHECK(gu_g.scalar_type() == gu_u.scalar_type(), "exl3_moe_coop: gate/up scratch dtype mismatch");
    // Scratch capacity in slots: the smallest of the per-slot buffers
    const int slots_max = (int) std::min({ had_g.numel() / p.Hi, had_u.numel() / p.Hi, gu_g.numel() / p.I,
                                           gu_u.numel() / p.I, act_out.numel() / p.I, d_out.numel() / p.Ho });
    auto check_scratch = [&] (const at::Tensor& t, int width)
    {
        TORCH_CHECK(t.is_contiguous() && t.size(-1) == width, "exl3_moe_coop: scratch shape");
    };
    check_scratch(had_g, p.Hi);
    check_scratch(had_u, p.Hi);
    check_scratch(gu_g, p.I);
    check_scratch(gu_u, p.I);
    check_scratch(act_out, p.I);
    check_scratch(d_out, p.Ho);
    TORCH_CHECK(out.dim() >= 2 && out.stride(-1) == 1, "exl3_moe_coop: out shape");
    TORCH_CHECK(slots_max >= 1 && slots_max <= 256, "exl3_moe_coop: scratch must hold 1..256 slots");

    // Counters: the gate/up stage's block covers every slot the scratch can hold, the down
    // stage's every token row the output can hold
    const int rows_max = (int) out.size(-2);
    p.slots_max = slots_max;
    p.ctr_a_len = slots_max * (p.I / 128);
    p.ctr_b_len = rows_max * (p.Ho / 128);
    TORCH_CHECK(ctr.numel() >= exl3_moe_coop_ctr_len(slots_max, rows_max, p.I, p.Ho), "exl3_moe_coop: counter scratch too small");
    p.ctr_a = (int*) ctr.data_ptr();
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.rows_max = rows_max;

    p.g_trellis = (const int64_t*) g_trellis.data_ptr();
    p.g_suh = (const int64_t*) g_suh.data_ptr();
    p.g_svh = (const int64_t*) g_svh.data_ptr();
    p.u_trellis = (const int64_t*) u_trellis.data_ptr();
    p.u_suh = (const int64_t*) u_suh.data_ptr();
    p.u_svh = (const int64_t*) u_svh.data_ptr();
    p.d_trellis = (const int64_t*) d_trellis.data_ptr();
    p.d_suh = (const int64_t*) d_suh.data_ptr();
    p.d_svh = (const int64_t*) d_svh.data_ptr();
    p.g_bias = g_bias ? (const int64_t*) g_bias->data_ptr() : nullptr;
    p.u_bias = u_bias ? (const int64_t*) u_bias->data_ptr() : nullptr;
    p.d_bias = d_bias ? (const int64_t*) d_bias->data_ptr() : nullptr;
    p.n_local = (int) u_trellis.size(0);
    p.act = act;
    p.act_limit = act_limit;
    p.gated = gated;

    p.had_g = (half*) had_g.data_ptr();
    p.had_u = (half*) had_u.data_ptr();
    p.gu_g = gu_g.data_ptr();
    p.gu_u = gu_u.data_ptr();
    p.act_out = (half*) act_out.data_ptr();
    p.d_out = (float*) d_out.data_ptr();
    p.out = (float*) out.data_ptr();
    p.out_stride = (int) out.stride(-2);

    if (sh_gate_w)
    {
        TORCH_CHECK_DTYPE(sh_gate_w.value(), kHalf);
        TORCH_CHECK(sh_gate_w->is_contiguous(), "exl3_moe_coop: shared gate weight must be contiguous");
        p.sh_gate_w = (const half*) sh_gate_w->data_ptr();
        p.sh_gate_n = (int) sh_gate_w->numel();
    }

    K_gu = gated ? Kg : Ku;
    K_d = Kd;
    cb = mcg ? 1 : (mul1 ? 2 : 0);
    return p;
}

void exl3_moe_coop_run
(
    MoeCoopParams p, float K_gu, float K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();

    // Per-call checks kept to what can actually vary between calls
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "exl3_moe_coop: x must be (bsz, H) with unit column stride");
    p.x = (const half*) x.data_ptr();
    p.x_stride = (int) x.stride(0);
    p.bsz = (int) x.size(0);
    p.H = (int) x.size(1);
    p.topk = (int) sel.size(-1);
    p.sel = (const int64_t*) sel.data_ptr();
    p.rw = (const half*) rw.data_ptr();
    const int slots = p.bsz * p.topk;
    TORCH_CHECK(p.H % 4 == 0 && p.H <= p.Hi, "exl3_moe_coop: H/Hi shape");
    TORCH_CHECK(slots <= p.slots_max && p.bsz <= p.rows_max, "exl3_moe_coop: batch exceeds the scratch");
    TORCH_CHECK(p.min_expert < 0 || p.max_expert - p.min_expert <= p.n_local, "exl3_moe_coop: expert range exceeds tables");
    if (sh_out)
    {
        TORCH_CHECK(p.H_out == p.H, "exl3_moe_coop: shared expert output width must match the routed output");
        p.sh_out = (const float*) sh_out->data_ptr();
        if (p.sh_gate_w) TORCH_CHECK(p.sh_gate_n == p.H && p.H % 2 == 0, "exl3_moe_coop: shared gate width");
    }
    else
        p.sh_gate_w = nullptr;
    exl3_moe_coop_launch(p, K_gu, K_d, cb, device, stream);
}

void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(rw, kHalf);
    TORCH_CHECK(sel.is_contiguous() && rw.is_contiguous(), "exl3_moe_coop: sel/rw must be contiguous");
    TORCH_CHECK(sel.sizes() == rw.sizes() && sel.dim() == 2 && sel.size(0) == x.size(0), "exl3_moe_coop: sel/rw must be (bsz, topk)");
    if (sh_out)
    {
        TORCH_CHECK_DTYPE(sh_out.value(), kFloat);
        TORCH_CHECK(sh_out->is_contiguous() && sh_out->size(-1) == x.size(1) && sh_out->numel() >= x.size(0) * x.size(1),
                    "exl3_moe_coop: shared expert output shape");
    }
    float K_gu, K_d; int cb;
    MoeCoopParams p = exl3_moe_coop_prepare(Hi, g_trellis, g_suh, g_svh, u_trellis, u_suh, u_svh, d_trellis, d_suh, d_svh,
                                            g_bias, u_bias, d_bias, Kg, Ku, Kd, mcg, mul1, act, act_limit, gated,
                                            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out, sh_gate_w, K_gu, K_d, cb);
    p.min_expert = min_expert;
    p.max_expert = max_expert;
    exl3_moe_coop_run(p, K_gu, K_d, cb, x, sel, rw, sh_out);
}
