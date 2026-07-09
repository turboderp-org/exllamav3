#include <Python.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "attention.h"
#include "../util.h"
#include "../util.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../rope.cuh"
#include "../cache/q_cache.cuh"
#include "../add.cuh"
#include "../activation.cuh"
#include "../norm.cuh"

BC_Attention::BC_Attention
(
    int _num_q_heads,
    int _num_kv_heads,
    int _head_dim,
    int _hidden_size,
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
    std::shared_ptr<BC_LinearEXL3> _g_proj,
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
    at::Tensor _inv_freq,
    int _rope_style,
    float _attn_factor,
    float _l4_scaling_beta,
    int _l4_scaling_original,
    bool _post_rope_norm,
    int _rotate_dims,
    bool _quant_cache,
    at::Tensor _cache_k,
    at::Tensor _cache_v,
    c10::optional<at::Tensor> _cache_k_scales,
    c10::optional<at::Tensor> _cache_v_scales,
    at::Tensor _xh,
    at::Tensor _h32
) :
    num_q_heads         (_num_q_heads),
    num_kv_heads        (_num_kv_heads),
    head_dim            (_head_dim),
    hidden_size         (_hidden_size),
    page_size           (_page_size),
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
    g_proj              (_g_proj),
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
    post_rope_norm      (_post_rope_norm),
    rotate_dims         (_rotate_dims),
    quant_cache         (_quant_cache),
    cache_k             (std::move(_cache_k)),
    cache_v             (std::move(_cache_v)),
    cache_k_scales      (std::move(_cache_k_scales)),
    cache_v_scales      (std::move(_cache_v_scales)),
    xh                  (std::move(_xh)),
    h32                 (std::move(_h32))
{
    TORCH_CHECK(gate_mode == 0 || gate_mode == 2 || gate_mode == 3, "BC_Attention: unsupported gate mode");
    slots.resize(MAX_BSZ * MAX_QLEN);
}

bool BC_Attention::needs_configure(int bsz, int q_len)
{
    TORCH_CHECK(1 <= bsz && bsz <= MAX_BSZ && 1 <= q_len && q_len <= MAX_QLEN, "BC_Attention: shape out of range");
    return !slot(bsz, q_len).configured;
}

void BC_Attention::configure_slot
(
    int bsz,
    int q_len,
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
    int splits_cap
)
{
    Slot& s = slot(bsz, q_len);
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
    if (gate_mode == 2)
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
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    int R = bsz * q_len;
    bool use_mgemm = kv_ptrs_trellis.has_value() && (R <= 32 || !k_proj);
    TORCH_CHECK(use_mgemm || (k_proj && (v_proj || use_k_as_v)), "BC_Attention: no k/v projection path for this batch shape");

    at::Tensor xh_flat = xh.view({-1});
    at::Tensor x2 = x.view({R, hidden_size});

    // Q (and gate) projections into the static buffers
    at::Tensor xh_q = xh_flat.narrow(0, 0, (int64_t) R * hidden_size).view({R, hidden_size});
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
        at::Tensor x3q = x2.view({1, R, hidden_size});
        at::Tensor xh_qg = xh_flat.narrow(0, 0, (int64_t) 2 * R * hidden_size).view({2, R, hidden_size});
        exl3_mgemm_gr(x3q, qg_ptrs_trellis.value(), s.qg2, qg_ptrs_suh.value(), xh_qg, qg_ptrs_svh.value(),
                      c10::nullopt, c10::nullopt, qg_K, -1, qg_mcg, qg_mul1, -1, -1, 0, graph);
    }
    else
    {
        exl3_gemm_gr(x2, q_proj->trellis, s.q2, q_proj->suh, xh_q, q_proj->svh, -1, q_proj->mcg, q_proj->mul1, 0, graph);
        if (q_proj->bias)
            add_gr(s.q2, q_proj->bias.value(), s.q2, graph);
        if (gate_mode == 2)
        {
            TORCH_CHECK(g_proj, "BC_Attention: full gate without fused qg needs a g projection");
            exl3_gemm_gr(x2, g_proj->trellis, s.g2, g_proj->suh, xh_q, g_proj->svh, -1, g_proj->mcg, g_proj->mul1, 0, graph);
            if (g_proj->bias)
                add_gr(s.g2, g_proj->bias.value(), s.g2, graph);
        }
    }

    at::Tensor kv2 = s.kv.view({2, R, num_kv_heads * head_dim});
    if (use_k_as_v)
    {
        // V shares the K projection output; copy it out before norm + RoPE modify K in place
        at::Tensor k2 = kv2.select(0, 0);
        at::Tensor xh_k = xh_flat.narrow(0, 0, (int64_t) R * hidden_size).view({R, hidden_size});
        exl3_gemm_gr(x2, k_proj->trellis, k2, k_proj->suh, xh_k, k_proj->svh, -1, k_proj->mcg, k_proj->mul1, 0, graph);
        if (k_proj->bias)
            add_gr(k2, k_proj->bias.value(), k2, graph);
        at::Tensor v2 = kv2.select(0, 1);
        if (v_norm)
            rms_norm_gr(k2, v_norm_w, v2, v_norm_eps, v_norm_constant_bias, v_norm_constant_scale, graph);
        else
            cuda_check(cudaMemcpyAsync(
                v2.data_ptr(), k2.data_ptr(),
                (size_t) R * num_kv_heads * head_dim * sizeof(half),
                cudaMemcpyDeviceToDevice, stream
            ));
    }
    else if (use_mgemm)
    {
        at::Tensor x3 = x2.view({1, R, hidden_size});
        at::Tensor xh_kv = xh_flat.narrow(0, 0, (int64_t) 2 * R * hidden_size).view({2, R, hidden_size});
        exl3_mgemm_gr(x3, kv_ptrs_trellis.value(), kv2, kv_ptrs_suh.value(), xh_kv, kv_ptrs_svh.value(),
                      c10::nullopt, c10::nullopt, kv_K, -1, kv_mcg, kv_mul1, -1, -1, 0, graph);
    }
    else
    {
        at::Tensor k2 = kv2.select(0, 0);
        at::Tensor v2 = kv2.select(0, 1);
        at::Tensor xh_k = xh_flat.narrow(0, 0, (int64_t) R * hidden_size).view({R, hidden_size});
        exl3_gemm_gr(x2, k_proj->trellis, k2, k_proj->suh, xh_k, k_proj->svh, -1, k_proj->mcg, k_proj->mul1, 0, graph);
        if (k_proj->bias)
            add_gr(k2, k_proj->bias.value(), k2, graph);
        exl3_gemm_gr(x2, v_proj->trellis, v2, v_proj->suh, xh_k, v_proj->svh, -1, v_proj->mcg, v_proj->mul1, 0, graph);
        if (v_proj->bias)
            add_gr(v2, v_proj->bias.value(), v2, graph);
    }

    if (v_norm && !use_k_as_v)
    {
        at::Tensor v2 = kv2.select(0, 1);
        rms_norm_gr(v2, v_norm_w, v2, v_norm_eps, v_norm_constant_bias, v_norm_constant_scale, graph);
    }

    // Fused head norm + RoPE, in place on the statics. All position sources and inv_freq are
    // patched per call; the kernel branches on the pointers at runtime, so one graph covers the
    // scalar/positions/position_ids modes
    c10::optional<at::Tensor> out_k4 = s.k4;
    const at::Tensor& ivf = inv_freq_override ? inv_freq_override.value() : inv_freq;
    rope_gr(s.q4, s.q4, s.k4, out_k4, ivf, (uint32_t) position, positions, position_ids, rope_style,
            attn_factor, q_norm, k_norm, norm_eps, norm_constant_bias, l4_scaling_beta,
            l4_scaling_original, post_rope_norm, rotate_dims, graph);

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
        };
        s.k_combine->launch(s.programs, 1, 1, args, stream);
        if (graph)
        {
            graph->record_param(s.k_combine->handle(), GP_attn_num_splits, 4, 4);
            graph->record_param(s.k_combine->handle(), GP_end, 0);
        }
    }

    // Output gate
    if (gate_mode == 2 || gate_mode == 3)
        mul_sigmoid__gr(s.o2, s.g2, graph);

    // Output projection
    at::Tensor y2 = y.view({R, hidden_size});
    at::Tensor xh_o = xh_flat.narrow(0, 0, (int64_t) R * num_q_heads * head_dim).view({R, num_q_heads * head_dim});
    exl3_gemm_gr(s.o2, o_proj->trellis, y2, o_proj->suh, xh_o, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);
    if (o_proj->bias)
        add_gr(y2, o_proj->bias.value(), y2, graph);
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
    const c10::optional<at::Tensor>& inv_freq_override
)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    Slot& s = slot(bsz, q_len);
    TORCH_CHECK(s.configured, "BC_Attention: slot not configured");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "BC_Attention: x and y must be contiguous");

    // First run per slot executes eagerly (GEMM autotune, kernel warmup); the second run is
    // captured, then launched below like every later run, with only the I/O pointers patched
    if (s.runs == 0)
    {
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, inv_freq_override, nullptr);
        s.runs = 1;
        return;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, inv_freq_override, s.graph.get());
        s.graph->capture_end();
        s.runs = 2;
    }

    int R = bsz * q_len;
    bool use_mgemm = kv_ptrs_trellis.has_value() && (R <= 32 || !k_proj);
    bool use_qg_mgemm = gate_mode == 2 && qg_ptrs_trellis.has_value() && R <= 32;

    std::vector<PPTR> params;
    params.reserve(18);

    // Q / gate projections
    if (use_qg_mgemm)
        params.emplace_back(GP_mgemm_A, (void*) x.data_ptr());
    else
    {
        params.emplace_back(GP_gemm_A, (void*) x.data_ptr());
        if (gate_mode == 2)
            params.emplace_back(GP_gemm_A, (void*) x.data_ptr());
    }

    // K/V projections
    if (use_k_as_v)
    {
        params.emplace_back(GP_gemm_A, (void*) x.data_ptr());
    }
    else if (use_mgemm)
    {
        params.emplace_back(GP_mgemm_A, (void*) x.data_ptr());
    }
    else
    {
        params.emplace_back(GP_gemm_A, (void*) x.data_ptr());
        params.emplace_back(GP_gemm_A, (void*) x.data_ptr());
    }

    // RoPE: which position source is active is a runtime branch in the kernel, so nulls are
    // patched like any other value
    const at::Tensor& ivf = inv_freq_override ? inv_freq_override.value() : inv_freq;
    int pid_stride = (position_ids && position_ids.value().dim() == 3) ? rotate_dims : 1;
    params.emplace_back(GP_rope_inv_freq, (void*) ivf.data_ptr());
    params.emplace_back(GP_rope_position, (void*) (uintptr_t) (uint32_t) position);
    params.emplace_back(GP_rope_positions, positions ? (void*) positions.value().data_ptr() : nullptr);
    params.emplace_back(GP_rope_position_ids, position_ids ? (void*) position_ids.value().data_ptr() : nullptr);
    params.emplace_back(GP_rope_pid_stride, (void*) (uintptr_t) pid_stride);

    // Cache append and attention: the block-table geometry and split configuration are runtime
    // kernel arguments, patched per call like the pointers, so context growth never recaptures
    int bt_width = (int) block_table.size(1);
    int num_splits, split_len;
    split_config(bt_width, page_size, q_len, s.block_n, s.splits_cap, &num_splits, &split_len);
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
    params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
    params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
    params.emplace_back(GP_attn_split_len, (void*) (uintptr_t) split_len);
    params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
    params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);
    params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);   // combine kernel
    params.emplace_back(GP_gemm_C, (void*) y.data_ptr());
    if (o_proj->bias)
    {
        params.emplace_back(GP_add_x, (void*) y.data_ptr());
        params.emplace_back(GP_add_z, (void*) y.data_ptr());
    }
    s.graph->launch(params, stream);
}
