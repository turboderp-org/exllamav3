#include <Python.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "shared_kv_attention.h"
#include "../util.h"
#include "../util.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../rope.cuh"
#include "../add.cuh"

// Key blocks per split, as in the standard attention path
#define SKV_SPLIT_MIN_BLOCKS 1

BC_SharedKVAttention::BC_SharedKVAttention
(
    int _num_q_heads,
    int _num_kv_heads,
    int _head_dim,
    int _hidden_size,
    int _hidden_size_padded,
    int _page_size,
    std::shared_ptr<BC_LinearEXL3> _q_proj,
    std::shared_ptr<BC_LinearEXL3> _o_proj,
    c10::optional<at::Tensor> _q_norm,
    float _norm_eps,
    float _norm_constant_bias,
    at::Tensor _inv_freq,
    int _rope_style,
    float _attn_factor,
    float _l4_scaling_beta,
    int _l4_scaling_original,
    int _rotate_dims,
    at::Tensor _cache_k,
    at::Tensor _cache_v,
    at::Tensor _xh,
    at::Tensor _h32
) :
    num_q_heads(_num_q_heads),
    num_kv_heads(_num_kv_heads),
    head_dim(_head_dim),
    hidden_size(_hidden_size),
    hidden_size_padded(_hidden_size_padded),
    page_size(_page_size),
    q_proj(std::move(_q_proj)),
    o_proj(std::move(_o_proj)),
    q_norm(std::move(_q_norm)),
    norm_eps(_norm_eps),
    norm_constant_bias(_norm_constant_bias),
    inv_freq(std::move(_inv_freq)),
    rope_style(_rope_style),
    attn_factor(_attn_factor),
    l4_scaling_beta(_l4_scaling_beta),
    l4_scaling_original(_l4_scaling_original),
    rotate_dims(_rotate_dims),
    cache_k(std::move(_cache_k)),
    cache_v(std::move(_cache_v)),
    xh(std::move(_xh)),
    h32(std::move(_h32))
{
    TORCH_CHECK(q_proj && o_proj, "BC_SharedKVAttention: q and o projections required");
    TORCH_CHECK(cache_k.is_contiguous() && cache_v.is_contiguous(), "BC_SharedKVAttention: source pages must be contiguous");
    TORCH_CHECK(cache_k.dim() == 4 && cache_k.size(1) == page_size && cache_k.size(2) == num_kv_heads && cache_k.size(3) == head_dim,
        "BC_SharedKVAttention: source pages must be (pages, page_size, kv_heads, head_dim)");
    TORCH_CHECK(cache_v.sizes() == cache_k.sizes(), "BC_SharedKVAttention: K and V pages differ");
    slots.resize(MAX_BSZ * MAX_QLEN);
}

bool BC_SharedKVAttention::needs_configure(int bsz, int q_len)
{
    return !slot(bsz, q_len).configured;
}

void BC_SharedKVAttention::configure_slot
(
    int bsz,
    int q_len,
    at::Tensor q,
    at::Tensor o,
    at::Tensor partial_o,
    at::Tensor partial_ml,
    std::shared_ptr<TritonKernel> k_split,
    std::shared_ptr<TritonKernel> k_combine,
    int block_n,
    int splits_cap,
    int programs,
    c10::optional<at::Tensor> xp,
    c10::optional<at::Tensor> yp
)
{
    TORCH_CHECK(bsz >= 1 && bsz <= MAX_BSZ && q_len >= 1 && q_len <= MAX_QLEN, "BC_SharedKVAttention: slot out of range");
    Slot& s = slot(bsz, q_len);
    int R = bsz * q_len;
    s.q = std::move(q);
    s.o = std::move(o);
    s.partial_o = std::move(partial_o);
    s.partial_ml = std::move(partial_ml);
    s.k_split = k_split;
    s.k_combine = k_combine;
    s.block_n = block_n;
    s.splits_cap = splits_cap;
    s.programs = programs;
    TORCH_CHECK(k_split && k_combine, "BC_SharedKVAttention: split and combine kernels required");
    TORCH_CHECK(s.q.is_contiguous() && s.o.is_contiguous(), "BC_SharedKVAttention: statics must be contiguous");
    if (hidden_size_padded != hidden_size)
    {
        TORCH_CHECK(xp && yp, "BC_SharedKVAttention: padded hidden size requires the xp/yp statics");
        s.xp = xp.value();
        s.yp = yp.value();
        TORCH_CHECK(s.xp.is_contiguous() && s.yp.is_contiguous(), "BC_SharedKVAttention: statics must be contiguous");
        TORCH_CHECK(s.xp.size(0) >= R && s.xp.size(1) == hidden_size_padded, "BC_SharedKVAttention: bad xp shape");
        TORCH_CHECK(s.yp.size(0) >= R && s.yp.size(1) == hidden_size_padded, "BC_SharedKVAttention: bad yp shape");
    }
    s.q2 = s.q.view({R, num_q_heads * head_dim});
    s.q4 = s.q.view({bsz, q_len, num_q_heads, head_dim});
    s.o2 = s.o.view({R, num_q_heads * head_dim});
    s.graph = std::make_unique<Graph>();
    s.runs = 0;
    s.configured = true;
}

static inline void skv_split_config(int bt_width, int page_size, int block_n, int splits_cap, int* num_splits, int* split_len)
{
    // No appended tokens: the bound is the block table's reach
    int bound = bt_width * page_size;
    *num_splits = MAX(1, MIN(splits_cap, CEIL_DIVIDE(bound, SKV_SPLIT_MIN_BLOCKS * block_n)));
    *split_len = CEIL_DIVIDE(CEIL_DIVIDE(bound, *num_splits), block_n) * block_n;
}

void BC_SharedKVAttention::run_gr
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
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    int R = bsz * q_len;
    const int hs = hidden_size_padded;
    at::Tensor xh_flat = xh.view({-1});
    at::Tensor x2 = x.view({R, hidden_size});

    // Zero-extended staging when the quantized projections are wider than the hidden size
    if (hs != hidden_size)
    {
        at::Tensor xp2 = s.xp.narrow(0, 0, R);
        copy2d_gr(x2, xp2, graph);
        x2 = xp2;
    }

    // Query projection
    at::Tensor xh_q = xh_flat.narrow(0, 0, (int64_t) R * hs).view({R, hs});
    exl3_gemm_gr(x2, q_proj->trellis, s.q2, q_proj->suh, xh_q, q_proj->svh, -1, q_proj->mcg, q_proj->mul1, 0, graph);
    if (q_proj->bias)
        add_gr(s.q2, q_proj->bias.value(), s.q2, graph);

    // Fused q-norm and RoPE, in place, at the caller's positions
    {
        c10::optional<at::Tensor> no_k = {};
        c10::optional<at::Tensor> no_ko = {};
        c10::optional<at::Tensor> no_norm = {};
        rope_gr(s.q4, s.q4, no_k, no_ko, inv_freq, (uint32_t) position, positions, position_ids, rope_style,
                attn_factor, q_norm, no_norm, norm_eps, norm_constant_bias, l4_scaling_beta,
                l4_scaling_original, rotate_dims, 0, graph);
    }

    // Flash-decoding over the source pages: split, then combine
    int num_splits, split_len;
    skv_split_config((int) block_table.size(1), page_size, s.block_n, s.splits_cap, &num_splits, &split_len);
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
            (void*) s.q.data_ptr(),      // k_scales: dead arg, fp16 pages
            (void*) s.q.data_ptr(),      // v_scales: dead arg
            (void*) h32.data_ptr(),
            (void*) (intptr_t) split_len,
            (void*) (intptr_t) (int) block_table.size(1),
            (void*) (intptr_t) num_splits,
            (void*) s.q.data_ptr(),      // sinks: dead arg, HAS_SINKS = false
        };
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
            (void*) s.q.data_ptr(),      // sinks: dead arg
        };
        s.k_combine->launch(s.programs, s.k_combine->grid_y, 1, args, stream);
        if (graph)
        {
            graph->record_param(s.k_combine->handle(), GP_attn_num_splits, 4, 4);
            graph->record_param(s.k_combine->handle(), GP_end, 0);
        }
    }

    // Output projection, through the padded static when the hidden size is padded
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

void BC_SharedKVAttention::run
(
    int bsz,
    int q_len,
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int64_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids
)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    Slot& s = slot(bsz, q_len);
    TORCH_CHECK(s.configured, "BC_SharedKVAttention: slot not configured");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "BC_SharedKVAttention: x and y must be contiguous");

    // First call runs eagerly (warmup), the second captures, later calls replay with patched arguments.
    // Where graphs are disabled every call takes the eager sequence
    if (s.runs == 0 || s.graph->disabled)
    {
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, nullptr);
        s.runs = 1;
        return;
    }
    if (!s.graph->ready)
    {
        GraphCapture capture(*s.graph);
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, s.graph.get());
        capture.finish();
        s.runs = 2;
    }

    int R = bsz * q_len;
    bool padded = hidden_size_padded != hidden_size;
    void* xptr = padded ? s.xp.data_ptr() : (void*) x.data_ptr();
    std::vector<PPTR> params;
    params.reserve(20);
    if (padded)
        params.emplace_back(GP_copy2d_src, (void*) x.data_ptr());
    params.emplace_back(GP_gemm_A, xptr);
    if (q_proj->bias)
    {
        params.emplace_back(GP_add_x, (void*) s.q2.data_ptr());
        params.emplace_back(GP_add_z, (void*) s.q2.data_ptr());
    }
    int pid_stride = (position_ids && position_ids.value().dim() == 3) ? rotate_dims : 1;
    params.emplace_back(GP_rope_inv_freq, (void*) inv_freq.data_ptr());
    params.emplace_back(GP_rope_position, (void*) (uintptr_t) (uint32_t) position);
    params.emplace_back(GP_rope_positions, positions ? (void*) positions.value().data_ptr() : nullptr);
    params.emplace_back(GP_rope_position_ids, position_ids ? (void*) position_ids.value().data_ptr() : nullptr);
    params.emplace_back(GP_rope_pid_stride, (void*) (uintptr_t) pid_stride);
    int bt_width = (int) block_table.size(1);
    int num_splits, split_len;
    skv_split_config(bt_width, page_size, s.block_n, s.splits_cap, &num_splits, &split_len);
    params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
    params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
    params.emplace_back(GP_attn_split_len, (void*) (uintptr_t) split_len);
    params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
    params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);
    params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);   // combine kernel
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
