#include <Python.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "attention.h"
#include "../util.h"
#include "../util.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../hgemm.cuh"
#include "../rope.cuh"
#include "../cache/q_cache.cuh"
#include "../add.cuh"
#include "../activation.cuh"
#include "../norm.cuh"
#include "../dsa_topk.cuh"

BC_Attention::BC_Attention
(
    int _num_q_heads,
    int _num_kv_heads,
    int _head_dim,
    int _hidden_size,
    int _hidden_size_padded,
    int _page_size,
    std::shared_ptr<BC_LinearEXL3> _q_proj,
    std::shared_ptr<BC_LinearEXL3> _k_proj,
    std::shared_ptr<BC_LinearEXL3> _v_proj,
    c10::optional<at::Tensor> _kv_ptrs_trellis,
    c10::optional<at::Tensor> _kv_ptrs_suh,
    c10::optional<at::Tensor> _kv_ptrs_svh,
    int _kv_K,
    bool _kv_mcg,
    bool _kv_mul1,
    std::shared_ptr<BC_LinearEXL3> _o_proj,
    bool _use_k_as_v,
    int _gate_mode,
    bool _gate_softplus,
    std::shared_ptr<BC_LinearEXL3> _g_proj,
    c10::optional<at::Tensor> _g_weight,
    c10::optional<at::Tensor> _qg_ptrs_trellis,
    c10::optional<at::Tensor> _qg_ptrs_suh,
    c10::optional<at::Tensor> _qg_ptrs_svh,
    int _qg_K,
    bool _qg_mcg,
    bool _qg_mul1,
    c10::optional<at::Tensor> _q_norm,
    c10::optional<at::Tensor> _k_norm,
    float _norm_eps,
    float _norm_constant_bias,
    bool _v_norm,
    c10::optional<at::Tensor> _v_norm_w,
    float _v_norm_eps,
    float _v_norm_constant_bias,
    float _v_norm_constant_scale,
    c10::optional<at::Tensor> _inv_freq,
    int _rope_style,
    float _attn_factor,
    float _l4_scaling_beta,
    int _l4_scaling_original,
    int _rotate_dims,
    bool _quant_cache,
    at::Tensor _cache_k,
    at::Tensor _cache_v,
    c10::optional<at::Tensor> _cache_k_scales,
    c10::optional<at::Tensor> _cache_v_scales,
    at::Tensor _xh,
    at::Tensor _h32,
    c10::optional<at::Tensor> _sinks
) :
    num_q_heads         (_num_q_heads),
    num_kv_heads        (_num_kv_heads),
    head_dim            (_head_dim),
    hidden_size         (_hidden_size),
    hidden_size_padded  (_hidden_size_padded),
    page_size           (_page_size),
    sinks               (std::move(_sinks)),
    q_proj              (_q_proj),
    k_proj              (_k_proj),
    v_proj              (_v_proj),
    kv_ptrs_trellis     (std::move(_kv_ptrs_trellis)),
    kv_ptrs_suh         (std::move(_kv_ptrs_suh)),
    kv_ptrs_svh         (std::move(_kv_ptrs_svh)),
    kv_K                (_kv_K),
    kv_mcg              (_kv_mcg),
    kv_mul1             (_kv_mul1),
    o_proj              (_o_proj),
    use_k_as_v          (_use_k_as_v),
    gate_mode           (_gate_mode),
    gate_softplus       (_gate_softplus),
    g_proj              (_g_proj),
    g_weight            (std::move(_g_weight)),
    qg_ptrs_trellis     (std::move(_qg_ptrs_trellis)),
    qg_ptrs_suh         (std::move(_qg_ptrs_suh)),
    qg_ptrs_svh         (std::move(_qg_ptrs_svh)),
    qg_K                (_qg_K),
    qg_mcg              (_qg_mcg),
    qg_mul1             (_qg_mul1),
    q_norm              (std::move(_q_norm)),
    k_norm              (std::move(_k_norm)),
    norm_eps            (_norm_eps),
    norm_constant_bias  (_norm_constant_bias),
    v_norm              (_v_norm),
    v_norm_w            (std::move(_v_norm_w)),
    v_norm_eps          (_v_norm_eps),
    v_norm_constant_bias (_v_norm_constant_bias),
    v_norm_constant_scale (_v_norm_constant_scale),
    inv_freq            (std::move(_inv_freq)),
    rope_style          (_rope_style),
    attn_factor         (_attn_factor),
    l4_scaling_beta     (_l4_scaling_beta),
    l4_scaling_original (_l4_scaling_original),
    rotate_dims         (_rotate_dims),
    quant_cache         (_quant_cache),
    cache_k             (std::move(_cache_k)),
    cache_v             (std::move(_cache_v)),
    cache_k_scales      (std::move(_cache_k_scales)),
    cache_v_scales      (std::move(_cache_v_scales)),
    xh                  (std::move(_xh)),
    h32                 (std::move(_h32))
{
    TORCH_CHECK(gate_mode >= 0 && gate_mode <= 3, "BC_Attention: unsupported gate mode");
    TORCH_CHECK(gate_mode != 1 || (g_weight && !qg_ptrs_trellis && !g_proj), "BC_Attention: headwise gate requires an fp16 gate weight");
    TORCH_CHECK(!g_weight || gate_mode == 1 || (gate_mode == 2 && !qg_ptrs_trellis && !g_proj), "BC_Attention: fp16 gate weight requires full gate mode without a quantized g projection");
    slots.resize(2 * MAX_BSZ * MAX_QLEN);
}

void BC_Attention::set_qsa
(
    std::shared_ptr<BC_LinearEXL3> _qk_proj,
    at::Tensor _q_norm_w,
    at::Tensor _k_norm_w,
    float _norm_eps,
    int _n_heads,
    int _head_dim,
    int _topk,
    int _compress_ratio,
    at::Tensor _raw_plane,
    at::Tensor _pool_plane
)
{
    TORCH_CHECK(_qk_proj, "BC_Attention: QSA requires a quantized index_qk projection");
    TORCH_CHECK(!quant_cache, "BC_Attention: QSA requires the fp16 cache");
    qsa = true;
    qsa_qk_proj = std::move(_qk_proj);
    qsa_q_norm_w = std::move(_q_norm_w);
    qsa_k_norm_w = std::move(_k_norm_w);
    qsa_norm_eps = _norm_eps;
    qsa_n_heads = _n_heads;
    qsa_head_dim = _head_dim;
    qsa_topk = _topk;
    qsa_cr = _compress_ratio;
    qsa_raw_plane = std::move(_raw_plane);
    qsa_pool_plane = std::move(_pool_plane);
}

bool BC_Attention::needs_configure(int bsz, int q_len, int regime)
{
    TORCH_CHECK(1 <= bsz && bsz <= MAX_BSZ && 1 <= q_len && q_len <= MAX_QLEN, "BC_Attention: shape out of range");
    TORCH_CHECK(regime == 0 || regime == 1, "BC_Attention: bad regime");
    return !slot(bsz, q_len, regime).configured;
}

void BC_Attention::configure_slot
(
    int bsz,
    int q_len,
    int regime,
    at::Tensor q,
    at::Tensor kv,
    at::Tensor o,
    at::Tensor partial_o,
    at::Tensor partial_ml,
    c10::optional<at::Tensor> gate_a,
    c10::optional<at::Tensor> gate_b,
    std::shared_ptr<TritonKernel> k_split,
    std::shared_ptr<TritonKernel> k_combine,
    std::shared_ptr<TritonKernel> k_update,
    int block_n,
    int splits_cap,
    c10::optional<at::Tensor> xp,
    c10::optional<at::Tensor> yp
)
{
    TORCH_CHECK(regime == 0 || qsa, "BC_Attention: sparse regime without QSA indexer");
    Slot& s = slot(bsz, q_len, regime);
    int R = bsz * q_len;

    s.q = std::move(q);
    s.kv = std::move(kv);
    s.o = std::move(o);
    s.partial_o = std::move(partial_o);
    s.partial_ml = std::move(partial_ml);
    if (gate_a) s.gate_a = gate_a.value();
    if (gate_b) s.gate_b = gate_b.value();
    s.k_split = k_split;
    s.k_combine = k_combine;
    s.k_update = k_update;
    s.block_n = block_n;
    s.splits_cap = splits_cap;

    if (hidden_size_padded != hidden_size || g_weight)
    {
        // Zero-padded staging for the projection input (also required for an fp16 gate: the
        // captured cublas node reads a static input). The pad columns of xp are zeroed
        // python-side at configure and never written after
        TORCH_CHECK(xp, "BC_Attention: padded hidden size or fp16 gate requires the xp static");
        s.xp = xp.value();
        TORCH_CHECK(s.xp.is_contiguous(), "BC_Attention: statics must be contiguous");
        TORCH_CHECK(s.xp.size(0) >= R && s.xp.size(1) == hidden_size_padded, "BC_Attention: bad xp shape");
    }
    if (hidden_size_padded != hidden_size)
    {
        // Padded o_proj output, trimmed to the exact width at the end of the graph
        TORCH_CHECK(yp, "BC_Attention: padded hidden size requires the yp static");
        s.yp = yp.value();
        TORCH_CHECK(s.yp.is_contiguous(), "BC_Attention: statics must be contiguous");
        TORCH_CHECK(s.yp.size(0) >= R && s.yp.size(1) == hidden_size_padded, "BC_Attention: bad yp shape");
    }

    TORCH_CHECK(s.q.is_contiguous() && s.kv.is_contiguous() && s.o.is_contiguous(), "BC_Attention: statics must be contiguous");
    TORCH_CHECK(quant_cache == (k_update == nullptr), "BC_Attention: k_update iff fp16 cache");
    TORCH_CHECK(k_combine, "BC_Attention: combine kernel required");

    s.q2 = s.q.view({R, num_q_heads * head_dim});
    s.q4 = s.q.view({bsz, q_len, num_q_heads, head_dim});
    s.k4 = s.kv.select(0, 0).view({bsz, q_len, num_kv_heads, head_dim});
    s.v4 = s.kv.select(0, 1).view({bsz, q_len, num_kv_heads, head_dim});
    s.o2 = s.o.view({R, num_q_heads * head_dim});
    s.o4 = s.o.view({bsz, q_len, num_q_heads, head_dim});

    int n_q = num_q_heads * head_dim;
    if (gate_mode == 1)
    {
        // Headwise gate: one fp16 value per head, from the captured cublas gemm over the staged
        // input
        TORCH_CHECK(gate_a, "BC_Attention: headwise gate requires the g static");
        s.g2 = s.gate_a.view({R, num_q_heads});
    }
    else if (gate_mode == 2)
    {
        // Full gate: q aliases qg[0] (python passes q as that slice); mgemm writes qg whole
        TORCH_CHECK(gate_a, "BC_Attention: full gate requires the qg static");
        s.qg2 = s.gate_a.view({2, R, n_q});
        s.g2 = s.qg2.select(0, 1);
        TORCH_CHECK(s.q.data_ptr() == s.qg2.select(0, 0).data_ptr(), "BC_Attention: q must alias qg[0]");
    }
    else if (gate_mode == 3)
    {
        // Interleaved: q_proj emits (R, 2 * n_q), deinterleaved into q and g
        TORCH_CHECK(gate_a && gate_b, "BC_Attention: interleaved gate requires qg and g statics");
        s.qg2 = s.gate_a.view({R, 2 * n_q});
        s.g2 = s.gate_b.view({R, n_q});
    }

    int group_size = num_q_heads / num_kv_heads;
    int block_m = 1; while (block_m < q_len) block_m <<= 1;
    int block_h = MAX(16 / block_m, 1);
    int h_blocks = CEIL_DIVIDE(group_size, block_h);
    s.programs = bsz * num_kv_heads * h_blocks;
    s.upd_grid = dim3(bsz * q_len, num_kv_heads, 1);

    s.graph = std::make_unique<Graph>();
    s.runs = 0;
    s.configured = true;
}

void BC_Attention::configure_slot_qsa
(
    int bsz,
    int q_len,
    int regime,
    at::Tensor qk,
    at::Tensor q,
    at::Tensor kraw,
    std::shared_ptr<TritonKernel> k_stage,
    std::shared_ptr<TritonKernel> k_raw_append,
    std::shared_ptr<TritonKernel> k_pool_update,
    int rotate_dims,
    c10::optional<at::Tensor> wts,
    c10::optional<at::Tensor> scores,
    c10::optional<at::Tensor> pool_idx,
    c10::optional<at::Tensor> indices,
    std::shared_ptr<TritonKernel> k_fewq,
    std::shared_ptr<TritonKernel> k_expand,
    std::shared_ptr<TritonKernel> k_split,
    std::shared_ptr<TritonKernel> k_combine,
    int fewq_gy,
    int qsa_splits,
    int qsa_split_len,
    int qsa_programs
)
{
    TORCH_CHECK(qsa, "BC_Attention: configure_slot_qsa without set_qsa");
    Slot& s = slot(bsz, q_len, regime);
    TORCH_CHECK(s.configured, "BC_Attention: configure_slot before configure_slot_qsa");
    TORCH_CHECK(k_stage && k_raw_append && k_pool_update, "BC_Attention: QSA plane kernels missing");

    s.qsa_qk = std::move(qk);
    s.qsa_q = std::move(q);
    s.qsa_kraw = std::move(kraw);
    s.qsa_q4 = s.qsa_q.view({bsz, q_len, qsa_n_heads, qsa_head_dim})
        .narrow(3, 0, rotate_dims);
    s.k_qsa_stage = k_stage;
    s.k_qsa_raw_append = k_raw_append;
    s.k_qsa_pool_update = k_pool_update;

    if (regime == 1)
    {
        // Sparse slots are single-job (bsz 1) but cover q_len 1..MAX_QLEN: every query row has
        // its own index list, scored/expanded with per-row causal bounds (SEQ constexpr), and
        // the gather kernel treats rows as its batch axis with a shared block-table row
        TORCH_CHECK(bsz == 1 && q_len <= MAX_QLEN, "BC_Attention: QSA sparse slots are single-job");
        TORCH_CHECK(wts && scores && pool_idx && indices && k_fewq && k_expand &&
                    k_split && k_combine, "BC_Attention: QSA sparse statics/kernels missing");
        s.qsa_wts = wts.value();
        s.qsa_scores = scores.value();
        s.qsa_pool_idx = pool_idx.value();
        s.qsa_indices = indices.value();
        s.k_qsa_fewq = k_fewq;
        s.k_qsa_expand = k_expand;
        s.k_qsa_split = k_split;
        s.k_qsa_combine = k_combine;
        s.qsa_fewq_gy = fewq_gy;
        s.qsa_splits = qsa_splits;
        s.qsa_split_len = qsa_split_len;
        s.qsa_programs = qsa_programs;
    }
}

// Live split configuration from the current block-table bound (same formula as the python
// dispatch path, so the two produce identical numerics)
static inline void split_config(int bt_width, int page_size, int q_len, int block_n, int splits_cap,
                                int* num_splits, int* split_len)
{
    int bound = bt_width * page_size + q_len;
    *num_splits = MAX(1, MIN(splits_cap, CEIL_DIVIDE(bound, 4 * block_n)));
    *split_len = CEIL_DIVIDE(CEIL_DIVIDE(bound, *num_splits), block_n) * block_n;
}

void BC_Attention::run_gr
(
    int bsz,
    int q_len,
    Slot& s,
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int64_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    const c10::optional<at::Tensor>& inv_freq_override,
    int regime,
    int64_t t_total,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    int R = bsz * q_len;
    bool use_mgemm = kv_ptrs_trellis.has_value() && (R <= 32 || !k_proj);
    TORCH_CHECK(use_mgemm || (k_proj && (v_proj || use_k_as_v)), "BC_Attention: no k/v projection path for this batch shape");

    at::Tensor xh_flat = xh.view({-1});
    at::Tensor x2 = x.view({R, hidden_size});

    // Padded hidden dim: stage the input through the zero-padded static; everything downstream
    // works at the padded width (the projections' actual K). An fp16 gate also stages: the
    // gate gemm is a captured cublas node with no patchable sites, so its input must be static
    const int hs = hidden_size_padded;
    if (hs != hidden_size || g_weight)
    {
        at::Tensor xp2 = s.xp.narrow(0, 0, R);
        copy2d_gr(x2, xp2, graph);
        x2 = xp2;
    }

    // Q (and gate) projections into the static buffers
    at::Tensor xh_q = xh_flat.narrow(0, 0, (int64_t) R * hs).view({R, hs});
    bool use_qg_mgemm = gate_mode == 2 && qg_ptrs_trellis.has_value() && R <= 32;
    if (gate_mode == 3)
    {
        exl3_gemm_gr(x2, q_proj->trellis, s.qg2, q_proj->suh, xh_q, q_proj->svh, -1, q_proj->mcg, q_proj->mul1, 0, graph);
        if (q_proj->bias)
            add_gr(s.qg2, q_proj->bias.value(), s.qg2, graph);
        deinterleave_qg_gr(s.qg2, s.q2, s.g2, head_dim, graph);
    }
    else if (use_qg_mgemm)
    {
        at::Tensor x3q = x2.view({1, R, hs});
        at::Tensor xh_qg = xh_flat.narrow(0, 0, (int64_t) 2 * R * hs).view({2, R, hs});
        exl3_mgemm_gr(x3q, qg_ptrs_trellis.value(), s.qg2, qg_ptrs_suh.value(), xh_qg, qg_ptrs_svh.value(),
                      c10::nullopt, c10::nullopt, qg_K, -1, qg_mcg, qg_mul1, -1, -1, 0, graph);
    }
    else
    {
        exl3_gemm_gr(x2, q_proj->trellis, s.q2, q_proj->suh, xh_q, q_proj->svh, -1, q_proj->mcg, q_proj->mul1, 0, graph);
        if (q_proj->bias)
            add_gr(s.q2, q_proj->bias.value(), s.q2, graph);
        if (gate_mode == 1)
        {
            TORCH_CHECK(g_weight, "BC_Attention: headwise gate requires the fp16 gate weight");
            hgemm_gr(x2, g_weight.value(), s.g2, graph);
        }
        else if (gate_mode == 2)
        {
            if (g_weight)
            {
                // Unquantized gate: cublas gemm over the staged (static) input, output static,
                // weight static -- nothing to patch at replay
                hgemm_gr(x2, g_weight.value(), s.g2, graph);
            }
            else
            {
                TORCH_CHECK(g_proj, "BC_Attention: full gate without fused qg needs a g projection");
                exl3_gemm_gr(x2, g_proj->trellis, s.g2, g_proj->suh, xh_q, g_proj->svh, -1, g_proj->mcg, g_proj->mul1, 0, graph);
                if (g_proj->bias)
                    add_gr(s.g2, g_proj->bias.value(), s.g2, graph);
            }
        }
    }

    at::Tensor kv2 = s.kv.view({2, R, num_kv_heads * head_dim});
    if (use_k_as_v)
    {
        // V shares the K projection output; copy it out before norm + RoPE modify K in place
        at::Tensor k2 = kv2.select(0, 0);
        at::Tensor xh_k = xh_flat.narrow(0, 0, (int64_t) R * hs).view({R, hs});
        exl3_gemm_gr(x2, k_proj->trellis, k2, k_proj->suh, xh_k, k_proj->svh, -1, k_proj->mcg, k_proj->mul1, 0, graph);
        if (k_proj->bias)
            add_gr(k2, k_proj->bias.value(), k2, graph);
        at::Tensor v2 = kv2.select(0, 1);
        if (v_norm)
        {
            // Norm is per head: view (R, kvh * hd) as (R * kvh, hd)
            at::Tensor k2h = k2.view({R * num_kv_heads, head_dim});
            at::Tensor v2h = v2.view({R * num_kv_heads, head_dim});
            rms_norm_gr(k2h, v_norm_w, v2h, v_norm_eps, v_norm_constant_bias, v_norm_constant_scale, graph);
        }
        else
            cuda_check(cudaMemcpyAsync(
                v2.data_ptr(), k2.data_ptr(),
                (size_t) R * num_kv_heads * head_dim * sizeof(half),
                cudaMemcpyDeviceToDevice, stream
            ));
    }
    else if (use_mgemm)
    {
        at::Tensor x3 = x2.view({1, R, hs});
        at::Tensor xh_kv = xh_flat.narrow(0, 0, (int64_t) 2 * R * hs).view({2, R, hs});
        exl3_mgemm_gr(x3, kv_ptrs_trellis.value(), kv2, kv_ptrs_suh.value(), xh_kv, kv_ptrs_svh.value(),
                      c10::nullopt, c10::nullopt, kv_K, -1, kv_mcg, kv_mul1, -1, -1, 0, graph);
    }
    else
    {
        at::Tensor k2 = kv2.select(0, 0);
        at::Tensor v2 = kv2.select(0, 1);
        at::Tensor xh_k = xh_flat.narrow(0, 0, (int64_t) R * hs).view({R, hs});
        exl3_gemm_gr(x2, k_proj->trellis, k2, k_proj->suh, xh_k, k_proj->svh, -1, k_proj->mcg, k_proj->mul1, 0, graph);
        if (k_proj->bias)
            add_gr(k2, k_proj->bias.value(), k2, graph);
        exl3_gemm_gr(x2, v_proj->trellis, v2, v_proj->suh, xh_k, v_proj->svh, -1, v_proj->mcg, v_proj->mul1, 0, graph);
        if (v_proj->bias)
            add_gr(v2, v_proj->bias.value(), v2, graph);
    }

    if (v_norm && !use_k_as_v)
    {
        // Norm is per head: view (R, kvh * hd) as (R * kvh, hd)
        at::Tensor v2h = kv2.select(0, 1).view({R * num_kv_heads, head_dim});
        rms_norm_gr(v2h, v_norm_w, v2h, v_norm_eps, v_norm_constant_bias, v_norm_constant_scale, graph);
    }

    // Fused head norm + RoPE, in place on the statics. All position sources and inv_freq are
    // patched per call; the kernel branches on the pointers at runtime, so one graph covers the
    // scalar/positions/position_ids modes. NoPE modules (no inv_freq, and by eligibility no
    // fused head norms either) skip the stage entirely; whether the stage exists must be
    // constant between capture and replay, so it keys on the member alone
    TORCH_CHECK(inv_freq || !inv_freq_override, "BC_Attention: inv_freq override on a NoPE module");
    if (inv_freq)
    {
        c10::optional<at::Tensor> out_k4 = s.k4;
        const at::Tensor& ivf = inv_freq_override ? inv_freq_override.value() : inv_freq.value();
        rope_gr(s.q4, s.q4, s.k4, out_k4, ivf, (uint32_t) position, positions, position_ids, rope_style,
                attn_factor, q_norm, k_norm, norm_eps, norm_constant_bias, l4_scaling_beta,
                l4_scaling_original, rotate_dims, 0, graph);
    }

    // Cache append (before attention: the split kernel counts the new tokens as part of the
    // sequence and reads them back from the cache)
    if (quant_cache)
    {
        quant_cache_paged_gr(s.k4, cache_k, cache_k_scales.value(), s.v4, cache_v, cache_v_scales.value(),
                             cache_seqlens, block_table, page_size, q_len, 0.0f, true, graph);
    }
    else
    {
        std::vector<void*> args =
        {
            (void*) s.k4.data_ptr(),
            (void*) s.v4.data_ptr(),
            (void*) cache_k.data_ptr(),
            (void*) cache_v.data_ptr(),
            (void*) block_table.data_ptr(),
            (void*) cache_seqlens.data_ptr(),
            (void*) (intptr_t) (int) block_table.size(1),
        };
        s.k_update->launch(s.upd_grid.x, s.upd_grid.y, s.upd_grid.z, args, stream);
        if (graph)
        {
            graph->record_param(s.k_update->handle(), GP_attn_block_table, 4);
            graph->record_param(s.k_update->handle(), GP_attn_seqlens, 5);
            graph->record_param(s.k_update->handle(), GP_attn_num_pages, 6, 4);
            graph->record_param(s.k_update->handle(), GP_end, 0);
        }
    }

    // QSA indexer planes (both regimes, so the selection has complete history once it
    // activates): fused q/raw-key projection, staging split + per-head q norm, raw-key plane
    // append and the pooled keys touched by this append
    if (qsa)
    {
        exl3_gemm_gr(x2, qsa_qk_proj->trellis, s.qsa_qk, qsa_qk_proj->suh,
                     xh_flat.narrow(0, 0, (int64_t) R * hs).view({R, hs}),
                     qsa_qk_proj->svh, -1, qsa_qk_proj->mcg, qsa_qk_proj->mul1, 0, graph);
        {
            std::vector<void*> args =
            {
                (void*) s.qsa_qk.data_ptr(),
                (void*) qsa_q_norm_w.data_ptr(),
                (void*) s.qsa_q.data_ptr(),
                (void*) s.qsa_kraw.data_ptr(),
                (void*) (intptr_t) R,
            };
            s.k_qsa_stage->launch(R, qsa_n_heads + 1, 1, args, stream);
        }
        {
            std::vector<void*> args =
            {
                (void*) s.qsa_kraw.data_ptr(),
                (void*) qsa_raw_plane.data_ptr(),
                (void*) block_table.data_ptr(),
                (void*) cache_seqlens.data_ptr(),
                (void*) (intptr_t) (int) block_table.size(1),
                (void*) (intptr_t) q_len,
            };
            s.k_qsa_raw_append->launch(R, 1, 1, args, stream);
            if (graph)
            {
                graph->record_param(s.k_qsa_raw_append->handle(), GP_attn_block_table, 2);
                graph->record_param(s.k_qsa_raw_append->handle(), GP_attn_seqlens, 3);
                graph->record_param(s.k_qsa_raw_append->handle(), GP_attn_num_pages, 4, 4);
                graph->record_param(s.k_qsa_raw_append->handle(), GP_end, 0);
            }
        }
        {
            std::vector<void*> args =
            {
                (void*) qsa_raw_plane.data_ptr(),
                (void*) qsa_pool_plane.data_ptr(),
                (void*) qsa_k_norm_w.data_ptr(),
                inv_freq ? (void*) inv_freq.value().data_ptr() : (void*) s.qsa_kraw.data_ptr(),
                (void*) block_table.data_ptr(),
                (void*) cache_seqlens.data_ptr(),
                (void*) (intptr_t) (int) block_table.size(1),
                (void*) (intptr_t) q_len,
            };
            // grid height = compiled MAXPOOLS = q_len / P + 1
            s.k_qsa_pool_update->launch(bsz, q_len / qsa_cr + 1, 1, args, stream);
            if (graph)
            {
                graph->record_param(s.k_qsa_pool_update->handle(), GP_attn_block_table, 4);
                graph->record_param(s.k_qsa_pool_update->handle(), GP_attn_seqlens, 5);
                graph->record_param(s.k_qsa_pool_update->handle(), GP_attn_num_pages, 6, 4);
                graph->record_param(s.k_qsa_pool_update->handle(), GP_end, 0);
            }
        }
    }

    if (regime == 1)
    {
        // Sparse regime (bsz == 1, q_len == 1): rope the indexer queries, score the pooled
        // plane (uniform head weights), select top-k blocks, expand to token indices plus the
        // query's tail block, and gather-attend. The index list is a slot static; causality
        // lives entirely in the selection
        int64_t t_scan = t_total / qsa_cr;
        {
            c10::optional<at::Tensor> no_k = {};
            c10::optional<at::Tensor> no_ko = {};
            c10::optional<at::Tensor> no_n = {};
            const at::Tensor& ivf = inv_freq_override ? inv_freq_override.value() : inv_freq.value();
            rope_gr(s.qsa_q4, s.qsa_q4, no_k, no_ko, ivf, (uint32_t) position, positions,
                    position_ids, rope_style, attn_factor, no_n, no_n, norm_eps, 0.0f, 0.0f,
                    0, rotate_dims, 0, graph);
        }
        {
            std::vector<void*> args =
            {
                (void*) s.qsa_q.data_ptr(),
                (void*) s.qsa_wts.data_ptr(),
                (void*) qsa_pool_plane.data_ptr(),
                (void*) s.qsa_scores.data_ptr(),
                (void*) (uintptr_t) (uint32_t) (int) t_scan,
                (void*) (uintptr_t) (uint32_t) R,
                (void*) (uintptr_t) (uint32_t) (int) position,
                (void*) (uintptr_t) (uint32_t) (int) t_scan,
                (void*) block_table.data_ptr(),
                (void*) (uintptr_t) (uint32_t) 0,
            };
            s.k_qsa_fewq->launch(R, s.qsa_fewq_gy, 1, args, stream);
            if (graph)
            {
                graph->record_param(s.k_qsa_fewq->handle(), GP_dsa_T, 4, 4);
                graph->record_param(s.k_qsa_fewq->handle(), GP_dsa_qpos, 6, 4);
                graph->record_param(s.k_qsa_fewq->handle(), GP_dsa_bound_max, 7, 4);
                graph->record_param(s.k_qsa_fewq->handle(), GP_attn_block_table, 8);
                graph->record_param(s.k_qsa_fewq->handle(), GP_end, 0);
            }
        }
        dsa_topk_gr(s.qsa_scores, s.qsa_pool_idx, qsa_topk, graph);
        {
            std::vector<void*> args =
            {
                (void*) s.qsa_pool_idx.data_ptr(),
                (void*) s.qsa_indices.data_ptr(),
                (void*) (uintptr_t) (uint32_t) (int) position,
            };
            int k_pad = (int) s.qsa_indices.size(1);
            s.k_qsa_expand->launch(R, CEIL_DIVIDE(k_pad, 256), 1, args, stream);
            if (graph)
            {
                graph->record_param(s.k_qsa_expand->handle(), GP_dsa_qpos, 2, 4);
                graph->record_param(s.k_qsa_expand->handle(), GP_end, 0);
            }
        }
        {
            // num_pages_per_seq 0: the kernel's per-row "batch" index then always selects
            // block-table row 0, shared by all q_len rows of the single job
            std::vector<void*> args =
            {
                (void*) s.q.data_ptr(),
                (void*) cache_k.data_ptr(),
                (void*) cache_v.data_ptr(),
                (void*) block_table.data_ptr(),
                (void*) s.qsa_indices.data_ptr(),
                (void*) s.partial_o.data_ptr(),
                (void*) s.partial_ml.data_ptr(),
                (void*) (intptr_t) (int) s.qsa_indices.size(1),
                (void*) (intptr_t) 0,
                (void*) (intptr_t) s.qsa_splits,
                (void*) (intptr_t) s.qsa_split_len,
            };
            s.k_qsa_split->launch(s.qsa_programs, s.qsa_splits, 1, args, stream);
            if (graph)
            {
                graph->record_param(s.k_qsa_split->handle(), GP_attn_block_table, 3);
                graph->record_param(s.k_qsa_split->handle(), GP_end, 0);
            }
        }
        {
            std::vector<void*> args =
            {
                (void*) s.partial_o.data_ptr(),
                (void*) s.partial_ml.data_ptr(),
                (void*) s.o.data_ptr(),
                (void*) h32.data_ptr(),
                (void*) (intptr_t) s.qsa_splits,
                (void*) s.q.data_ptr(),  // sinks: dead arg, HAS_SINKS = false
            };
            s.k_qsa_combine->launch(s.qsa_programs, 1, 1, args, stream);
        }
    }
    else
    {

    // Flash-decoding split kernel + combine, with the split configuration derived from the
    // current block-table bound per call
    void* scales_k = quant_cache ? cache_k_scales.value().data_ptr() : s.q.data_ptr();
    void* scales_v = quant_cache ? cache_v_scales.value().data_ptr() : s.q.data_ptr();
    int num_splits, split_len;
    split_config((int) block_table.size(1), page_size, q_len, s.block_n, s.splits_cap, &num_splits, &split_len);
    {
        std::vector<void*> args =
        {
            (void*) s.q.data_ptr(),
            (void*) cache_k.data_ptr(),
            (void*) cache_v.data_ptr(),
            (void*) block_table.data_ptr(),
            (void*) cache_seqlens.data_ptr(),
            (void*) s.o.data_ptr(),
            (void*) s.partial_o.data_ptr(),
            (void*) s.partial_ml.data_ptr(),
            scales_k,
            scales_v,
            (void*) h32.data_ptr(),
            (void*) (intptr_t) split_len,
            (void*) (intptr_t) (int) block_table.size(1),
            (void*) (intptr_t) num_splits,
            (void*) s.q.data_ptr(),  // sinks: dead arg, slots compile with HAS_SINKS = false
        };
        // Launched at the split cap so the captured grid never changes; splits at or above the
        // live count exit without storing
        s.k_split->launch(s.programs, s.splits_cap, 1, args, stream);
        if (graph)
        {
            graph->record_param(s.k_split->handle(), GP_attn_block_table, 3);
            graph->record_param(s.k_split->handle(), GP_attn_seqlens, 4);
            graph->record_param(s.k_split->handle(), GP_attn_split_len, 11, 4);
            graph->record_param(s.k_split->handle(), GP_attn_num_pages, 12, 4);
            graph->record_param(s.k_split->handle(), GP_attn_num_splits, 13, 4);
            graph->record_param(s.k_split->handle(), GP_end, 0);
        }
    }
    {
        std::vector<void*> args =
        {
            (void*) s.partial_o.data_ptr(),
            (void*) s.partial_ml.data_ptr(),
            (void*) s.o.data_ptr(),
            (void*) h32.data_ptr(),
            (void*) (intptr_t) num_splits,
            // Static sink pointer when the module has learned sinks (combine compiled with
            // HAS_SINKS), dead arg otherwise
            sinks ? (void*) sinks.value().data_ptr() : (void*) s.q.data_ptr(),
        };
        s.k_combine->launch(s.programs, 1, 1, args, stream);
        if (graph)
        {
            graph->record_param(s.k_combine->handle(), GP_attn_num_splits, 4, 4);
            graph->record_param(s.k_combine->handle(), GP_end, 0);
        }
    }

    }   // regime == 0

    // Output gate
    if (gate_mode == 1)
    {
        // Headwise: one gate value per head, broadcast across head_dim
        at::Tensor g3 = s.g2.view({bsz, q_len, num_q_heads});
        if (gate_softplus)
            mul_softplus_broadcast__gr(s.o4, g3, graph);
        else
            mul_sigmoid_broadcast__gr(s.o4, g3, graph);
    }
    else if (gate_mode == 2 || gate_mode == 3)
        mul_sigmoid__gr(s.o2, s.g2, graph);

    // Output projection. With a padded hidden dim the GEMM writes the padded static (N of the
    // quantized o_proj), bias included, and the exact-width columns are copied out to y
    at::Tensor y2 = y.view({R, hidden_size});
    at::Tensor c2 = y2;
    if (hs != hidden_size)
        c2 = s.yp.narrow(0, 0, R);
    at::Tensor xh_o = xh_flat.narrow(0, 0, (int64_t) R * num_q_heads * head_dim).view({R, num_q_heads * head_dim});
    exl3_gemm_gr(s.o2, o_proj->trellis, c2, o_proj->suh, xh_o, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);
    if (o_proj->bias)
        add_gr(c2, o_proj->bias.value(), c2, graph);
    if (hs != hidden_size)
        copy2d_gr(c2, y2, graph);
}

void BC_Attention::run
(
    int bsz,
    int q_len,
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int64_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    const c10::optional<at::Tensor>& inv_freq_override,
    int regime,
    int64_t t_total
)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    Slot& s = slot(bsz, q_len, regime);
    TORCH_CHECK(s.configured, "BC_Attention: slot not configured");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "BC_Attention: x and y must be contiguous");
    TORCH_CHECK(regime == 0 || qsa, "BC_Attention: sparse regime without QSA indexer");

    // First run per slot executes eagerly (GEMM autotune, kernel warmup); the second run is
    // captured, then launched below like every later run, with only the I/O pointers patched
    if (s.runs == 0)
    {
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, inv_freq_override, regime, t_total, nullptr);
        s.runs = 1;
        return;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, inv_freq_override, regime, t_total, s.graph.get());
        s.graph->capture_end();
        s.runs = 2;
    }

    int R = bsz * q_len;
    bool use_mgemm = kv_ptrs_trellis.has_value() && (R <= 32 || !k_proj);
    bool use_qg_mgemm = gate_mode == 2 && qg_ptrs_trellis.has_value() && R <= 32;

    std::vector<PPTR> params;
    params.reserve(40);

    // Padded hidden dim or fp16 gate: x feeds the staging copy at the head of the graph and the
    // projections read the (static) buffer; otherwise the projections read x directly
    bool padded = hidden_size_padded != hidden_size;
    bool staged = padded || g_weight.has_value();
    void* xptr = staged ? s.xp.data_ptr() : (void*) x.data_ptr();
    if (staged)
        params.emplace_back(GP_copy2d_src, (void*) x.data_ptr());

    // Q / gate projections (an fp16 gate is a cublas node with no patchable sites)
    if (use_qg_mgemm)
        params.emplace_back(GP_mgemm_A, xptr);
    else
    {
        params.emplace_back(GP_gemm_A, xptr);
        if (gate_mode == 2 && !g_weight)
            params.emplace_back(GP_gemm_A, xptr);
    }

    // K/V projections
    if (use_k_as_v)
    {
        params.emplace_back(GP_gemm_A, xptr);
    }
    else if (use_mgemm)
    {
        params.emplace_back(GP_mgemm_A, xptr);
    }
    else
    {
        params.emplace_back(GP_gemm_A, xptr);
        params.emplace_back(GP_gemm_A, xptr);
    }

    // RoPE: which position source is active is a runtime branch in the kernel, so nulls are
    // patched like any other value. NoPE graphs contain no rope node
    if (inv_freq)
    {
        const at::Tensor& ivf = inv_freq_override ? inv_freq_override.value() : inv_freq.value();
        int pid_stride = (position_ids && position_ids.value().dim() == 3) ? rotate_dims : 1;
        params.emplace_back(GP_rope_inv_freq, (void*) ivf.data_ptr());
        params.emplace_back(GP_rope_position, (void*) (uintptr_t) (uint32_t) position);
        params.emplace_back(GP_rope_positions, positions ? (void*) positions.value().data_ptr() : nullptr);
        params.emplace_back(GP_rope_position_ids, position_ids ? (void*) position_ids.value().data_ptr() : nullptr);
        params.emplace_back(GP_rope_pid_stride, (void*) (uintptr_t) pid_stride);
    }

    // Cache append and attention: the block-table geometry and split configuration are runtime
    // kernel arguments, patched per call like the pointers, so context growth never recaptures
    int bt_width = (int) block_table.size(1);
    if (quant_cache)
    {
        params.emplace_back(GP_qcache_seqlens, (void*) cache_seqlens.data_ptr());
        params.emplace_back(GP_qcache_block_table, (void*) block_table.data_ptr());
        params.emplace_back(GP_qcache_blocks_per_seq, (void*) (uintptr_t) bt_width);
    }
    else
    {
        params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
        params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
        params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
    }

    // QSA plane stages: qk projection, raw append, pool update
    if (qsa)
    {
        params.emplace_back(GP_gemm_A, xptr);
        params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
        params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
        params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
        params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
        params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
        params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
    }

    if (regime == 1)
    {
        // Indexer-query rope, scoring/top-k bounds (in block units), tail expansion and the
        // gathered attention. The index list and split configuration are slot statics
        int t_scan = (int) (t_total / qsa_cr);
        if (inv_freq)
        {
            const at::Tensor& ivf = inv_freq_override ? inv_freq_override.value() : inv_freq.value();
            int pid_stride = (position_ids && position_ids.value().dim() == 3) ? rotate_dims : 1;
            params.emplace_back(GP_rope_inv_freq, (void*) ivf.data_ptr());
            params.emplace_back(GP_rope_position, (void*) (uintptr_t) (uint32_t) position);
            params.emplace_back(GP_rope_positions, positions ? (void*) positions.value().data_ptr() : nullptr);
            params.emplace_back(GP_rope_position_ids, position_ids ? (void*) position_ids.value().data_ptr() : nullptr);
            params.emplace_back(GP_rope_pid_stride, (void*) (uintptr_t) pid_stride);
        }
        params.emplace_back(GP_dsa_T, (void*) (uintptr_t) (uint32_t) t_scan);
        params.emplace_back(GP_dsa_qpos, (void*) (uintptr_t) (uint32_t) position);
        params.emplace_back(GP_dsa_bound_max, (void*) (uintptr_t) (uint32_t) t_scan);
        params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
        params.emplace_back(GP_dsa_T, (void*) (uintptr_t) (uint32_t) t_scan);   // top-k
        params.emplace_back(GP_dsa_qpos, (void*) (uintptr_t) (uint32_t) position);   // expand
        params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());    // gather
    }
    else
    {
        int num_splits, split_len;
        split_config(bt_width, page_size, q_len, s.block_n, s.splits_cap, &num_splits, &split_len);
        params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
        params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
        params.emplace_back(GP_attn_split_len, (void*) (uintptr_t) split_len);
        params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
        params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);
        params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);   // combine kernel
    }
    void* yptr = padded ? s.yp.data_ptr() : (void*) y.data_ptr();
    params.emplace_back(GP_gemm_C, yptr);
    if (o_proj->bias)
    {
        params.emplace_back(GP_add_x, yptr);
        params.emplace_back(GP_add_z, yptr);
    }
    if (padded)
        params.emplace_back(GP_copy2d_dst, (void*) y.data_ptr());
    s.graph->launch(params, stream);
}
