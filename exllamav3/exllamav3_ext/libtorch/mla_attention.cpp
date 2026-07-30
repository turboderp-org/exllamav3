#include <Python.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "mla_attention.h"
#include "../util.h"
#include "../util.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../rope.cuh"
#include "../cache/q_cache.cuh"
#include "../norm.cuh"

BC_MLAttention::BC_MLAttention
(
    int _num_q_heads,
    int _hidden_size,
    int _page_size,
    int _kv_lora_rank,
    int _qk_rope_head_dim,
    int _qk_nope_head_dim,
    int _v_head_dim,
    int _q_lora_rank,
    std::shared_ptr<BC_LinearEXL3> _q_proj,
    std::shared_ptr<BC_LinearEXL3> _q_a_proj,
    c10::optional<at::Tensor> _q_a_norm_w,
    std::shared_ptr<BC_LinearEXL3> _kv_a_proj,
    std::shared_ptr<BC_LinearEXL3> _o_proj,
    at::Tensor _kv_norm_w,
    float _norm_eps,
    at::Tensor _inv_freq,
    int _rope_style,
    float _attn_factor,
    int _rotate_dims,
    at::Tensor _w_uk_flat,
    at::Tensor _w_uv_flat,
    bool _quant_cache,
    at::Tensor _cache_ckv,
    at::Tensor _cache_kpe,
    c10::optional<at::Tensor> _cache_scales,
    at::Tensor _xh,
    at::Tensor _h32
) :
    num_q_heads         (_num_q_heads),
    hidden_size         (_hidden_size),
    page_size           (_page_size),
    kv_lora_rank        (_kv_lora_rank),
    qk_rope_head_dim    (_qk_rope_head_dim),
    qk_nope_head_dim    (_qk_nope_head_dim),
    qk_head_dim         (_qk_nope_head_dim + _qk_rope_head_dim),
    v_head_dim          (_v_head_dim),
    q_lora_rank         (_q_lora_rank),
    q_proj              (_q_proj),
    q_a_proj            (_q_a_proj),
    q_a_norm_w          (std::move(_q_a_norm_w)),
    kv_a_proj           (_kv_a_proj),
    o_proj              (_o_proj),
    kv_norm_w           (std::move(_kv_norm_w)),
    norm_eps            (_norm_eps),
    inv_freq            (std::move(_inv_freq)),
    rope_style          (_rope_style),
    attn_factor         (_attn_factor),
    rotate_dims         (_rotate_dims),
    w_uk_flat           (std::move(_w_uk_flat)),
    w_uv_flat           (std::move(_w_uv_flat)),
    quant_cache         (_quant_cache),
    cache_ckv           (std::move(_cache_ckv)),
    cache_kpe           (std::move(_cache_kpe)),
    cache_scales        (std::move(_cache_scales)),
    xh                  (std::move(_xh)),
    h32                 (std::move(_h32))
{
    TORCH_CHECK((q_lora_rank > 0) == (q_a_proj != nullptr), "BC_MLAttention: q_a_proj iff q_lora");
    TORCH_CHECK(!quant_cache || cache_scales, "BC_MLAttention: quantized cache requires scales");
    slots.resize(MAX_BSZ * MAX_QLEN);
}

bool BC_MLAttention::needs_configure(int bsz, int q_len)
{
    TORCH_CHECK(1 <= bsz && bsz <= MAX_BSZ && 1 <= q_len && q_len <= MAX_QLEN, "BC_MLAttention: shape out of range");
    return !slot(bsz, q_len).configured;
}

void BC_MLAttention::configure_slot
(
    int bsz,
    int q_len,
    at::Tensor q_full,
    c10::optional<at::Tensor> q_a,
    at::Tensor ckv_kpe,
    at::Tensor ckv,
    at::Tensor kpe,
    at::Tensor q_pe,
    at::Tensor q_lat,
    at::Tensor o_lat,
    at::Tensor o,
    at::Tensor partial_o,
    at::Tensor partial_ml,
    c10::optional<at::Tensor> qtmp,
    c10::optional<at::Tensor> stmp,
    std::shared_ptr<TritonKernel> k_stage,
    std::shared_ptr<TritonKernel> k_absorb,
    std::shared_ptr<TritonKernel> k_append,
    std::shared_ptr<TritonKernel> k_split,
    std::shared_ptr<TritonKernel> k_combine,
    std::shared_ptr<TritonKernel> k_unfold,
    int block_n,
    int splits_cap,
    int programs,
    int absorb_gx,
    int absorb_gy,
    int unfold_gx
)
{
    Slot& s = slot(bsz, q_len);
    int R = bsz * q_len;

    s.q_full = std::move(q_full);
    if (q_a) s.q_a = q_a.value();
    s.ckv_kpe = std::move(ckv_kpe);
    s.ckv = std::move(ckv);
    s.kpe = std::move(kpe);
    s.q_pe = std::move(q_pe);
    s.q_lat = std::move(q_lat);
    s.o_lat = std::move(o_lat);
    s.o = std::move(o);
    s.partial_o = std::move(partial_o);
    s.partial_ml = std::move(partial_ml);
    if (qtmp) s.qtmp = qtmp.value();
    if (stmp) s.stmp = stmp.value();
    s.k_stage = k_stage;
    s.k_absorb = k_absorb;
    s.k_append = k_append;
    s.k_split = k_split;
    s.k_combine = k_combine;
    s.k_unfold = k_unfold;
    s.block_n = block_n;
    s.splits_cap = splits_cap;
    s.programs = programs;
    s.absorb_gx = absorb_gx;
    s.absorb_gy = absorb_gy;
    s.unfold_gx = unfold_gx;

    TORCH_CHECK(s.q_full.is_contiguous() && s.ckv_kpe.is_contiguous() && s.ckv.is_contiguous() &&
                s.kpe.is_contiguous() && s.q_pe.is_contiguous() && s.q_lat.is_contiguous() &&
                s.o_lat.is_contiguous() && s.o.is_contiguous(), "BC_MLAttention: statics must be contiguous");
    TORCH_CHECK(quant_cache == (qtmp && stmp), "BC_MLAttention: quant temporaries iff quantized cache");
    TORCH_CHECK(!(q_lora_rank > 0) || q_a, "BC_MLAttention: q_a static required for the LoRA q path");
    TORCH_CHECK(s.q_pe.numel() == (int64_t) R * num_q_heads * qk_rope_head_dim, "BC_MLAttention: bad q_pe shape");

    s.q_pe4 = s.q_pe.view({bsz, q_len, num_q_heads, qk_rope_head_dim});
    s.kpe4 = s.kpe.view({bsz, q_len, 1, qk_rope_head_dim});

    s.graph = std::make_unique<Graph>();
    s.runs = 0;
    s.configured = true;
}

// Live split configuration from the current block-table bound. Mirrors the dispatch wrapper
// (mla_attn_triton_decode) exactly, so the two paths produce identical numerics; note the MLA
// bound is num_pages * page_size, without the q_len term the MHA path adds
static inline void mla_split_config(int bt_width, int page_size, int block_n, int splits_cap,
                                    int* num_splits, int* split_len)
{
    int max_k_len = bt_width * page_size;
    *num_splits = MAX(1, MIN(splits_cap, CEIL_DIVIDE(max_k_len, 4 * block_n)));
    *split_len = CEIL_DIVIDE(CEIL_DIVIDE(max_k_len, *num_splits), block_n) * block_n;
}

void BC_MLAttention::run_gr
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

    at::Tensor xh_flat = xh.view({-1});
    at::Tensor x2 = x.view({R, hidden_size});

    // Q projections into the static buffers (direct, or LoRA down/norm/up)
    if (q_a_proj)
    {
        at::Tensor xh_x = xh_flat.narrow(0, 0, (int64_t) R * hidden_size).view({R, hidden_size});
        exl3_gemm_gr(x2, q_a_proj->trellis, s.q_a, q_a_proj->suh, xh_x, q_a_proj->svh, -1, q_a_proj->mcg, q_a_proj->mul1, 0, graph);
        rms_norm_gr(s.q_a, q_a_norm_w, s.q_a, norm_eps, 0.0f, 1.0f, graph);
        int rank = (int) s.q_a.size(1);
        at::Tensor xh_a = xh_flat.narrow(0, 0, (int64_t) R * rank).view({R, rank});
        exl3_gemm_gr(s.q_a, q_proj->trellis, s.q_full, q_proj->suh, xh_a, q_proj->svh, -1, q_proj->mcg, q_proj->mul1, 0, graph);
    }
    else
    {
        at::Tensor xh_x = xh_flat.narrow(0, 0, (int64_t) R * hidden_size).view({R, hidden_size});
        exl3_gemm_gr(x2, q_proj->trellis, s.q_full, q_proj->suh, xh_x, q_proj->svh, -1, q_proj->mcg, q_proj->mul1, 0, graph);
    }

    // Latent projection
    {
        at::Tensor xh_x = xh_flat.narrow(0, 0, (int64_t) R * hidden_size).view({R, hidden_size});
        exl3_gemm_gr(x2, kv_a_proj->trellis, s.ckv_kpe, kv_a_proj->suh, xh_x, kv_a_proj->svh, -1, kv_a_proj->mcg, kv_a_proj->mul1, 0, graph);
    }

    // Staging: kv_a norm into ckv, rope key split, per-head rope-query gather. All operands are
    // statics, so the captured node needs no patching
    {
        std::vector<void*> args =
        {
            (void*) s.q_full.data_ptr(),
            (void*) s.ckv_kpe.data_ptr(),
            (void*) kv_norm_w.data_ptr(),
            (void*) s.q_pe.data_ptr(),
            (void*) s.ckv.data_ptr(),
            (void*) s.kpe.data_ptr(),
        };
        s.k_stage->launch(R, 1, 1, args, stream);
    }

    // Partial RoPE on the rope halves, in place on the statics; the position sources are patched
    // per call and the kernel branches on the pointers at runtime
    {
        c10::optional<at::Tensor> out_k = s.kpe4;
        rope_gr(s.q_pe4, s.q_pe4, s.kpe4, out_k, inv_freq, (uint32_t) position, positions, position_ids,
                rope_style, attn_factor, c10::nullopt, c10::nullopt, norm_eps, 0.0f, 0.0f, 0, false,
                rotate_dims, graph);
    }

    // Absorb W_UK into the queries, per head, straight from the flat layout
    {
        std::vector<void*> args =
        {
            (void*) s.q_full.data_ptr(),
            (void*) w_uk_flat.data_ptr(),
            (void*) s.q_lat.data_ptr(),
            (void*) (intptr_t) R,
        };
        s.k_absorb->launch(s.absorb_gx, s.absorb_gy, num_q_heads, args, stream);
    }

    // Cache append (before attention: the split kernel counts the new rows as part of the
    // sequence and reads them back from the cache)
    if (quant_cache)
    {
        quant_cache_cont_gr(s.ckv, s.qtmp, s.stmp, 0.0f, graph);
        std::vector<void*> args =
        {
            (void*) s.qtmp.data_ptr(),
            (void*) s.stmp.data_ptr(),
            (void*) s.kpe.data_ptr(),
            (void*) cache_ckv.data_ptr(),
            (void*) cache_scales.value().data_ptr(),
            (void*) cache_kpe.data_ptr(),
            (void*) block_table.data_ptr(),
            (void*) cache_seqlens.data_ptr(),
            (void*) (intptr_t) (int) block_table.size(1),
            (void*) (intptr_t) q_len,
        };
        s.k_append->launch(R, 1, 1, args, stream);
        if (graph)
        {
            graph->record_param(s.k_append->handle(), GP_attn_block_table, 6);
            graph->record_param(s.k_append->handle(), GP_attn_seqlens, 7);
            graph->record_param(s.k_append->handle(), GP_attn_num_pages, 8, 4);
            graph->record_param(s.k_append->handle(), GP_end, 0);
        }
    }
    else
    {
        std::vector<void*> args =
        {
            (void*) s.ckv.data_ptr(),
            (void*) s.kpe.data_ptr(),
            (void*) cache_ckv.data_ptr(),
            (void*) cache_kpe.data_ptr(),
            (void*) block_table.data_ptr(),
            (void*) cache_seqlens.data_ptr(),
            (void*) (intptr_t) (int) block_table.size(1),
            (void*) (intptr_t) q_len,
        };
        s.k_append->launch(R, 1, 1, args, stream);
        if (graph)
        {
            graph->record_param(s.k_append->handle(), GP_attn_block_table, 4);
            graph->record_param(s.k_append->handle(), GP_attn_seqlens, 5);
            graph->record_param(s.k_append->handle(), GP_attn_num_pages, 6, 4);
            graph->record_param(s.k_append->handle(), GP_end, 0);
        }
    }

    // Flash-decoding split + combine, split configuration derived from the block-table bound
    // per call. The slots compile with FINAL = false, so the combine pass always runs (identical
    // math at one split, and the split count can then vary freely at replay)
    int num_splits, split_len;
    mla_split_config((int) block_table.size(1), page_size, s.block_n, s.splits_cap, &num_splits, &split_len);
    void* scales_ptr = quant_cache ? cache_scales.value().data_ptr() : s.q_lat.data_ptr();
    {
        std::vector<void*> args =
        {
            (void*) s.q_lat.data_ptr(),
            (void*) s.q_pe.data_ptr(),
            (void*) cache_ckv.data_ptr(),
            (void*) cache_kpe.data_ptr(),
            scales_ptr,
            (void*) h32.data_ptr(),
            (void*) block_table.data_ptr(),
            (void*) cache_seqlens.data_ptr(),
            (void*) s.o_lat.data_ptr(),
            (void*) s.partial_o.data_ptr(),
            (void*) s.partial_ml.data_ptr(),
            (void*) (intptr_t) split_len,
            (void*) (intptr_t) (int) block_table.size(1),
            (void*) (intptr_t) num_splits,
        };
        // Launched at the split cap so the captured grid never changes; splits at or above the
        // live count exit without storing
        s.k_split->launch(s.programs, s.splits_cap, 1, args, stream);
        if (graph)
        {
            graph->record_param(s.k_split->handle(), GP_attn_block_table, 6);
            graph->record_param(s.k_split->handle(), GP_attn_seqlens, 7);
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
            (void*) s.o_lat.data_ptr(),
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

    // Unfold W_UV per head from the flat layout, emitting token-major o_proj input
    {
        std::vector<void*> args =
        {
            (void*) s.o_lat.data_ptr(),
            (void*) w_uv_flat.data_ptr(),
            (void*) s.o.data_ptr(),
            (void*) (intptr_t) R,
        };
        s.k_unfold->launch(s.unfold_gx, num_q_heads, 1, args, stream);
    }

    // Output projection
    at::Tensor y2 = y.view({R, hidden_size});
    at::Tensor xh_o = xh_flat.narrow(0, 0, (int64_t) R * num_q_heads * v_head_dim).view({R, num_q_heads * v_head_dim});
    exl3_gemm_gr(s.o, o_proj->trellis, y2, o_proj->suh, xh_o, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);
}

void BC_MLAttention::run
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
    TORCH_CHECK(s.configured, "BC_MLAttention: slot not configured");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "BC_MLAttention: x and y must be contiguous");

    // First run per slot executes eagerly (GEMM autotune, kernel warmup); the second run is
    // captured, then launched below like every later run, with only the I/O pointers patched
    if (s.runs == 0)
    {
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, nullptr);
        s.runs = 1;
        return;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, s.graph.get());
        s.graph->capture_end();
        s.runs = 2;
    }

    std::vector<PPTR> params;
    params.reserve(20);

    // Q / latent projections. The q_b GEMM of the LoRA path reads the q_a static -- its entry is
    // a no-op that keeps the sequential patcher aligned (two same-id sites in a row)
    params.emplace_back(GP_gemm_A, (void*) x.data_ptr());
    if (q_a_proj)
        params.emplace_back(GP_gemm_A, (void*) s.q_a.data_ptr());
    params.emplace_back(GP_gemm_A, (void*) x.data_ptr());

    // RoPE position sources; which one is active is a runtime branch in the kernel
    {
        int pid_stride = (position_ids && position_ids.value().dim() == 3) ? rotate_dims : 1;
        params.emplace_back(GP_rope_inv_freq, (void*) inv_freq.data_ptr());
        params.emplace_back(GP_rope_position, (void*) (uintptr_t) (uint32_t) position);
        params.emplace_back(GP_rope_positions, positions ? (void*) positions.value().data_ptr() : nullptr);
        params.emplace_back(GP_rope_position_ids, position_ids ? (void*) position_ids.value().data_ptr() : nullptr);
        params.emplace_back(GP_rope_pid_stride, (void*) (uintptr_t) pid_stride);
    }

    // Cache append and attention: block-table geometry and split configuration are runtime
    // kernel arguments, patched per call, so context growth never recaptures
    int bt_width = (int) block_table.size(1);
    int num_splits, split_len;
    mla_split_config(bt_width, page_size, s.block_n, s.splits_cap, &num_splits, &split_len);
    params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
    params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
    params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
    params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
    params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
    params.emplace_back(GP_attn_split_len, (void*) (uintptr_t) split_len);
    params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
    params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);
    params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);   // combine kernel

    // Output projection
    params.emplace_back(GP_gemm_C, (void*) y.data_ptr());

    s.graph->launch(params, stream);
}
