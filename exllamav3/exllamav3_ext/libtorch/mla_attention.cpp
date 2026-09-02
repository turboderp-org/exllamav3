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
#include "../hgemm.cuh"
#include "../add.cuh"
#include "../dsa_topk.cuh"

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
    float _l4_scaling_beta,
    int _l4_scaling_original,
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
    l4_scaling_beta     (_l4_scaling_beta),
    l4_scaling_original (_l4_scaling_original),
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
    slots.resize(2 * MAX_BSZ * MAX_QLEN);
}

void BC_MLAttention::set_indexer
(
    int mode,
    std::shared_ptr<BC_LinearEXL3> wq_b,
    c10::optional<at::Tensor> wk_w,
    c10::optional<at::Tensor> k_norm_w,
    c10::optional<at::Tensor> k_norm_b,
    c10::optional<at::Tensor> weights_w,
    c10::optional<at::Tensor> kidx,
    int n_heads,
    int head_dim,
    int topk,
    int kpool,
    bool kpool_tail,
    c10::optional<at::Tensor> gate_w,
    c10::optional<at::Tensor> kpool_ape,
    c10::optional<at::Tensor> kpool_plane
)
{
    TORCH_CHECK(mode == 1 || mode == 2, "BC_MLAttention: indexer mode must be 1 (full) or 2 (shared)");
    TORCH_CHECK(mode != 1 || (wq_b && wk_w && k_norm_w && k_norm_b && weights_w && kidx),
                "BC_MLAttention: full indexer requires all indexer tensors");
    TORCH_CHECK(!kpool || mode != 1 || (gate_w && kpool_ape && kpool_plane),
                "BC_MLAttention: kpool full indexer requires gate weight, APE and pooled plane");
    idx_mode = mode;
    idx_wq_b = std::move(wq_b);
    idx_wk_w = std::move(wk_w);
    idx_k_norm_w = std::move(k_norm_w);
    idx_k_norm_b = std::move(k_norm_b);
    idx_weights_w = std::move(weights_w);
    cache_kidx = std::move(kidx);
    index_n_heads = n_heads;
    index_head_dim = head_dim;
    index_topk = topk;
    index_kpool = kpool;
    index_kpool_tail = kpool_tail;
    idx_gate_w = std::move(gate_w);
    idx_kpool_ape = std::move(kpool_ape);
    cache_kpool = std::move(kpool_plane);
}

bool BC_MLAttention::needs_configure(int bsz, int q_len, int regime)
{
    TORCH_CHECK(1 <= bsz && bsz <= MAX_BSZ && 1 <= q_len && q_len <= MAX_QLEN, "BC_MLAttention: shape out of range");
    TORCH_CHECK(regime == 0 || regime == 1, "BC_MLAttention: bad regime");
    return !slot(bsz, q_len, regime).configured;
}

void BC_MLAttention::configure_slot
(
    int bsz,
    int q_len,
    int regime,
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
    Slot& s = slot(bsz, q_len, regime);
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

void BC_MLAttention::configure_slot_dsa
(
    int bsz,
    int q_len,
    int regime,
    c10::optional<at::Tensor> x_st,
    c10::optional<at::Tensor> kidx,
    c10::optional<at::Tensor> kidx_n,
    c10::optional<at::Tensor> qidx,
    c10::optional<at::Tensor> wts,
    c10::optional<at::Tensor> scores,
    c10::optional<at::Tensor> indices,
    c10::optional<at::Tensor> dsa_arr,
    c10::optional<at::Tensor> dsa_ws_ml,
    c10::optional<at::Tensor> dsa_ws_acc,
    std::shared_ptr<TritonKernel> k_idx_norm,
    std::shared_ptr<TritonKernel> k_plane_append,
    std::shared_ptr<TritonKernel> k_fewq,
    std::shared_ptr<TritonKernel> k_dsa_split,
    std::shared_ptr<TritonKernel> k_dsa_combine,
    int dsa_hb,
    int dsa_splits,
    int fewq_gy,
    c10::optional<at::Tensor> gidx,
    c10::optional<at::Tensor> pool_idx,
    std::shared_ptr<TritonKernel> k_gate_append,
    std::shared_ptr<TritonKernel> k_pool_update,
    std::shared_ptr<TritonKernel> k_pool_expand
)
{
    TORCH_CHECK(idx_mode != 0, "BC_MLAttention: configure_slot_dsa without set_indexer");
    Slot& s = slot(bsz, q_len, regime);
    TORCH_CHECK(s.configured, "BC_MLAttention: configure_slot before configure_slot_dsa");

    if (idx_mode == 1)
    {
        TORCH_CHECK(x_st && kidx && kidx_n, "BC_MLAttention: full indexer statics missing");
        s.x_st = x_st.value();
        s.kidx = kidx.value();
        s.kidx_n = kidx_n.value();
        s.kidx4 = s.kidx_n.view({bsz, q_len, 1, index_head_dim})
            .narrow(3, 0, qk_rope_head_dim);
        s.k_idx_norm = k_idx_norm;
        s.k_plane_append = k_plane_append;
        if (index_kpool)
        {
            TORCH_CHECK(gidx && k_gate_append && k_pool_update,
                        "BC_MLAttention: kpool statics/kernels missing");
            s.gidx = gidx.value();
            s.k_gate_append = k_gate_append;
            s.k_pool_update = k_pool_update;
        }
    }
    if (regime == 1)
    {
        TORCH_CHECK(indices && dsa_ws_ml && dsa_ws_acc,
                    "BC_MLAttention: sparse statics missing");
        TORCH_CHECK(bsz == 1 || dsa_arr, "BC_MLAttention: batched sparse slot without state array");
        s.indices = indices.value();
        if (dsa_arr) s.dsa_arr = dsa_arr.value();
        s.dsa_ws_ml = dsa_ws_ml.value();
        s.dsa_ws_acc = dsa_ws_acc.value();
        s.k_dsa_split = k_dsa_split;
        s.k_dsa_combine = k_dsa_combine;
        s.dsa_hb = dsa_hb;
        s.dsa_splits = dsa_splits;
        if (idx_mode == 1)
        {
            TORCH_CHECK(qidx && wts && scores, "BC_MLAttention: scoring statics missing");
            s.qidx = qidx.value();
            s.qidx4 = s.qidx.view({bsz, q_len, index_n_heads, index_head_dim})
                .narrow(3, 0, qk_rope_head_dim);
            s.wts = wts.value();
            s.scores = scores.value();
            if (index_kpool)
            {
                TORCH_CHECK(pool_idx && k_pool_expand,
                            "BC_MLAttention: kpool sparse statics/kernels missing");
                s.pool_idx = pool_idx.value();
                s.k_pool_expand = k_pool_expand;
            }
            s.k_fewq = k_fewq;
            s.fewq_gy = fewq_gy;
        }
    }
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
    int regime,
    int64_t t_total,
    const c10::optional<at::Tensor>& ext_indices,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    int R = bsz * q_len;

    // EXL3_BCM_DEBUG=1: synchronize and check after every stage of the eager warmup run, to
    // localize faults that otherwise surface at the next launch site
    static const bool bcm_debug = [](){ const char* e = getenv("EXL3_BCM_DEBUG"); return e && *e == '1'; }();
    auto dbg = [&](const char* tag)
    {
        if (!bcm_debug || graph) return;
        cudaStreamSynchronize(stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("BC_MLAttention debug: fault after stage %s: %s\n", tag, cudaGetErrorString(err));
            fflush(stdout);
            TORCH_CHECK(false, "BC_MLAttention stage fault, see stdout");
        }
    };

    at::Tensor xh_flat = xh.view({-1});
    at::Tensor x2 = x.view({R, hidden_size});

    dbg("entry");

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

    // Llama-4 position scale on the full query (Mistral-Small-4): one per-token scalar over
    // all heads, nope and rope halves alike, before the pe-gather/rope/absorb stages consume
    // q_full
    if (l4_scaling_beta > 0.0f)
    {
        int pid_stride = (position_ids && position_ids.value().dim() == 3) ? rotate_dims : 1;
        l4_scale_q_gr(s.q_full, bsz, q_len, num_q_heads * qk_head_dim, (int) s.q_full.size(1),
                      (uint32_t) position, positions, position_ids, l4_scaling_beta,
                      l4_scaling_original, pid_stride, graph);
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
    // per call and the kernel branches on the pointers at runtime. NoPE models (GLM5.3,
    // qk_rope_head_dim 0) skip the stage entirely
    if (qk_rope_head_dim > 0)
    {
        c10::optional<at::Tensor> out_k = s.kpe4;
        rope_gr(s.q_pe4, s.q_pe4, s.kpe4, out_k, inv_freq, (uint32_t) position, positions, position_ids,
                rope_style, attn_factor, c10::nullopt, c10::nullopt, norm_eps, 0.0f, 0.0f, 0,
                rotate_dims, 0, graph);
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

    // DSA indexer keys (full-indexer layers, both regimes): stage x for the unquantized
    // GEMMs, project + biased-LayerNorm + partial rope the chunk's keys, and append them to
    // the paged side plane so the selection has complete history once it activates
    if (idx_mode == 1)
    {
        at::Tensor x_rows = s.x_st.narrow(0, 0, R);
        copy2d_gr(x2, x_rows, graph);
        at::Tensor kidx_rows = s.kidx.narrow(0, 0, R);
        hgemm_gr(x_rows, idx_wk_w.value(), kidx_rows, graph);
        {
            std::vector<void*> args =
            {
                (void*) s.kidx.data_ptr(),
                (void*) idx_k_norm_w.value().data_ptr(),
                (void*) idx_k_norm_b.value().data_ptr(),
                (void*) s.kidx_n.data_ptr(),
                (void*) (intptr_t) R,
            };
            s.k_idx_norm->launch(R, 1, 1, args, stream);
        }
        dbg("idx_norm");
        if (qk_rope_head_dim > 0)
        {
            c10::optional<at::Tensor> no_k = {};
            c10::optional<at::Tensor> no_ko = {};
            c10::optional<at::Tensor> no_n = {};
            rope_gr(s.kidx4, s.kidx4, no_k, no_ko, inv_freq, (uint32_t) position, positions,
                    position_ids, rope_style, attn_factor, no_n, no_n, norm_eps, 0.0f, 0.0f, 0,
                    rotate_dims, 0, graph);
        }
        {
            std::vector<void*> args =
            {
                (void*) s.kidx_n.data_ptr(),
                (void*) cache_kidx.value().data_ptr(),
                (void*) block_table.data_ptr(),
                (void*) cache_seqlens.data_ptr(),
                (void*) (intptr_t) (int) block_table.size(1),
                (void*) (intptr_t) q_len,
            };
            s.k_plane_append->launch(R, 1, 1, args, stream);
            dbg("plane_append");
            if (graph)
            {
                graph->record_param(s.k_plane_append->handle(), GP_attn_block_table, 2);
                graph->record_param(s.k_plane_append->handle(), GP_attn_seqlens, 3);
                graph->record_param(s.k_plane_append->handle(), GP_attn_num_pages, 4, 4);
                graph->record_param(s.k_plane_append->handle(), GP_end, 0);
            }
        }
        if (index_kpool)
        {
            // Gate scores land in the packed plane next to the keys, then the pools touched by
            // this append are (re)built in the pooled plane (softmax(gate + ape)-weighted key
            // means; partial pools are written but never selected)
            at::Tensor gidx_rows = s.gidx.narrow(0, 0, R);
            hgemm_gr(x_rows, idx_gate_w.value(), gidx_rows, graph);
            dbg("gate_hgemm");
            {
                std::vector<void*> args =
                {
                    (void*) s.gidx.data_ptr(),
                    (void*) cache_kidx.value().data_ptr(),
                    (void*) block_table.data_ptr(),
                    (void*) cache_seqlens.data_ptr(),
                    (void*) (intptr_t) (int) block_table.size(1),
                    (void*) (intptr_t) q_len,
                };
                s.k_gate_append->launch(R, 1, 1, args, stream);
                dbg("gate_append");
                if (graph)
                {
                    graph->record_param(s.k_gate_append->handle(), GP_attn_block_table, 2);
                    graph->record_param(s.k_gate_append->handle(), GP_attn_seqlens, 3);
                    graph->record_param(s.k_gate_append->handle(), GP_attn_num_pages, 4, 4);
                    graph->record_param(s.k_gate_append->handle(), GP_end, 0);
                }
            }
            {
                std::vector<void*> args =
                {
                    (void*) cache_kidx.value().data_ptr(),
                    (void*) cache_kpool.value().data_ptr(),
                    (void*) idx_kpool_ape.value().data_ptr(),
                    (void*) block_table.data_ptr(),
                    (void*) cache_seqlens.data_ptr(),
                    (void*) (intptr_t) (int) block_table.size(1),
                    (void*) (intptr_t) q_len,
                };
                // grid height = compiled MAXPOOLS = q_len / P + 1
                s.k_pool_update->launch(bsz, q_len / index_kpool + 1, 1, args, stream);
                dbg("pool_update");
                if (graph)
                {
                    graph->record_param(s.k_pool_update->handle(), GP_attn_block_table, 3);
                    graph->record_param(s.k_pool_update->handle(), GP_attn_seqlens, 4);
                    graph->record_param(s.k_pool_update->handle(), GP_attn_num_pages, 5, 4);
                    graph->record_param(s.k_pool_update->handle(), GP_end, 0);
                }
            }
        }
    }

    if (regime == 1)
    {
        // Sparse regime: score + select (full) or take the external selection (shared), pack
        // the absorbed queries, and gather-attend over the selected latent rows. Batched
        // slots (MULTIROW kernels) read their per-job scan widths and causal clamps from the
        // device state array, filled here from cache_seqlens; the single-job slots patch the
        // host scalars instead
        bool multirow = bsz > 1;
        int* arr_pos = nullptr;
        int* arr_bound = nullptr;
        c10::optional<at::Tensor> arr_bound_t;
        // Only the scoring/top-k stages consume the state array; shared-indexer slots
        // gather through an external selection and need none of it
        if (multirow && idx_mode == 1)
        {
            dsa_seq_state_gr(cache_seqlens, s.dsa_arr, bsz, q_len, graph);
            dbg("seq_state");
            arr_pos = (int*) s.dsa_arr.data_ptr();
            arr_bound = arr_pos + (int) s.dsa_arr.size(1);
            arr_bound_t = s.dsa_arr.select(0, 1).narrow(0, 0, bsz);
        }
        if (idx_mode == 1)
        {
            exl3_gemm_gr(s.q_a, idx_wq_b->trellis, s.qidx, idx_wq_b->suh,
                         xh.view({-1}).narrow(0, 0, (int64_t) R * q_lora_rank).view({R, q_lora_rank}),
                         idx_wq_b->svh, -1, idx_wq_b->mcg, idx_wq_b->mul1, 0, graph);
            if (qk_rope_head_dim > 0)
            {
                c10::optional<at::Tensor> no_k = {};
                c10::optional<at::Tensor> no_ko = {};
                c10::optional<at::Tensor> no_n = {};
                rope_gr(s.qidx4, s.qidx4, no_k, no_ko, inv_freq, (uint32_t) position, positions,
                        position_ids, rope_style, attn_factor, no_n, no_n, norm_eps, 0.0f, 0.0f,
                        0, rotate_dims, 0, graph);
            }
            at::Tensor wts_rows = s.wts;   // full padded height (cuBLASLt M >= 8)
            hgemm_gr(s.x_st, idx_weights_w.value(), wts_rows, graph);
            dbg("wts_hgemm");
            {
                // Scoring covers the full static width every step (bounds written as -inf), so
                // no stale region survives shorter contexts. T / q_pos0 / bound_max patch
                // kpool: score over the pooled plane; scan width and causal bound count in
                // POOL units (the kernel's compress_rate handles the per-row token bound)
                int64_t t_scan = index_kpool ? t_total / index_kpool : t_total;
                std::vector<void*> args =
                {
                    (void*) s.qidx.data_ptr(),
                    (void*) s.wts.data_ptr(),
                    index_kpool ? (void*) cache_kpool.value().data_ptr()
                                : (void*) cache_kidx.value().data_ptr(),
                    (void*) s.scores.data_ptr(),
                    multirow ? (void*) arr_bound : (void*) (uintptr_t) (uint32_t) (int) t_scan,
                    (void*) (uintptr_t) (uint32_t) R,
                    multirow ? (void*) arr_pos : (void*) (uintptr_t) (uint32_t) (int) position,
                    multirow ? (void*) arr_bound : (void*) (uintptr_t) (uint32_t) (int) t_scan,
                    (void*) block_table.data_ptr(),
                    (void*) (uintptr_t) (uint32_t) (multirow ? (int) block_table.size(1) : 0),
                };
                s.k_fewq->launch(R, s.fewq_gy, 1, args, stream);
                dbg("fewq");
                if (graph)
                {
                    if (!multirow)
                    {
                        graph->record_param(s.k_fewq->handle(), GP_dsa_T, 4, 4);
                        graph->record_param(s.k_fewq->handle(), GP_dsa_qpos, 6, 4);
                        graph->record_param(s.k_fewq->handle(), GP_dsa_bound_max, 7, 4);
                    }
                    graph->record_param(s.k_fewq->handle(), GP_attn_block_table, 8);
                    if (multirow)
                        graph->record_param(s.k_fewq->handle(), GP_attn_num_pages, 9, 4);
                    graph->record_param(s.k_fewq->handle(), GP_end, 0);
                }
            }
            // Batched: the per-job bound rides as a device pointer (t_seq = q_len), so the
            // top-k needs no scan-width patch and only reads freshly written score rows
            if (multirow)
                dsa_topk_gr(s.scores, s.indices, index_topk, graph, arr_bound_t, q_len);
            else if (index_kpool)
            {
                // Select pools, then expand to raw token indices (x P) and append the query's
                // incomplete tail pool per row
                dsa_topk_gr(s.scores, s.pool_idx, index_topk / index_kpool, graph);
                dbg("pool_topk");
                std::vector<void*> args =
                {
                    (void*) s.pool_idx.data_ptr(),
                    (void*) s.indices.data_ptr(),
                    (void*) (uintptr_t) (uint32_t) (int) position,
                };
                int k_pad = (int) s.indices.size(1);
                s.k_pool_expand->launch(R, CEIL_DIVIDE(k_pad, 256), 1, args, stream);
                if (graph)
                {
                    graph->record_param(s.k_pool_expand->handle(), GP_dsa_qpos, 2, 4);
                    graph->record_param(s.k_pool_expand->handle(), GP_end, 0);
                }
            }
            else
                dsa_topk_gr(s.scores, s.indices, index_topk, graph);
        dbg("topk");
        }

        {
            // Gathered attention: causality lives in the selection, V is the latent. The
            // kernels read the q_lat/q_pe statics directly (Q_SPLIT: ring carries q_pe,
            // ring_stride carries R) and never build the rope half of the output. The
            // indices pointer is patched every call (own static for full layers, the
            // producing layer's tensor for shared ones)
            at::Tensor idx_t = ext_indices ? ext_indices.value() : s.indices;
            int k_pad = (int) idx_t.size(1);
            std::vector<void*> args =
            {
                (void*) s.q_lat.data_ptr(),
                (void*) s.q_pe.data_ptr(),       // ring slot: token-major rope queries
                (void*) s.q_lat.data_ptr(),      // kv_chunk (unused, HAS_WINDOW = False)
                (void*) cache_ckv.data_ptr(),
                (void*) cache_kpe.data_ptr(),
                (void*) block_table.data_ptr(),
                (void*) idx_t.data_ptr(),
                (void*) s.dsa_ws_ml.data_ptr(),
                (void*) s.dsa_ws_acc.data_ptr(),
                (void*) (uintptr_t) (uint32_t) k_pad,
                (void*) (uintptr_t) (uint32_t) 0,   // win_len
                (void*) (uintptr_t) (uint32_t) 0,   // pool_len (gathered mode)
                // MULTIROW: one block-table row per job; else a single shared row
                (void*) (uintptr_t) (uint32_t) (multirow ? (int) block_table.size(1) : 0),
                (void*) (uintptr_t) (uint32_t) (int) position,
                (void*) (uintptr_t) (uint32_t) 0,   // win_floor
                (void*) (uintptr_t) (uint32_t) 0,   // ring_beg
                (void*) (uintptr_t) (uint32_t) 0,   // slot_ids
                (void*) (uintptr_t) (uint32_t) R,   // ring_stride slot: q_lat row stride
            };
            s.k_dsa_split->launch(R * s.dsa_hb, s.dsa_splits, 1, args, stream);
            dbg("dsa_split");
            if (graph)
            {
                graph->record_param(s.k_dsa_split->handle(), GP_attn_block_table, 5);
                graph->record_param(s.k_dsa_split->handle(), GP_dsa_indices, 6);
                if (multirow)
                    graph->record_param(s.k_dsa_split->handle(), GP_attn_num_pages, 12, 4);
                graph->record_param(s.k_dsa_split->handle(), GP_end, 0);
            }
        }
        {
            // Combine emits the head-major latent (OUT_LATENT) straight into o_lat, the
            // unfold's input
            std::vector<void*> args =
            {
                (void*) s.dsa_ws_ml.data_ptr(),
                (void*) s.dsa_ws_acc.data_ptr(),
                (void*) s.dsa_ws_ml.data_ptr(),  // sinks (unused, HAS_SINKS = False)
                (void*) s.dsa_ws_ml.data_ptr(),  // derot_inv_freq (unused)
                (void*) s.o_lat.data_ptr(),
                (void*) (uintptr_t) (uint32_t) (int) position,
                (void*) (uintptr_t) (uint32_t) R,
                (void*) (uintptr_t) (uint32_t) s.dsa_splits,
            };
            int gy = CEIL_DIVIDE(kv_lora_rank, 128);
            s.k_dsa_combine->launch(R * s.dsa_hb, gy, 1, args, stream);
        }
        dbg("dsa_combine");
    }
    else
    {

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

    }   // regime == 0

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
    dbg("unfold");

    // Output projection
    at::Tensor y2 = y.view({R, hidden_size});
    at::Tensor xh_o = xh_flat.narrow(0, 0, (int64_t) R * num_q_heads * v_head_dim).view({R, num_q_heads * v_head_dim});
    exl3_gemm_gr(s.o, o_proj->trellis, y2, o_proj->suh, xh_o, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);
    dbg("o_proj");
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
    const c10::optional<at::Tensor>& position_ids,
    int regime,
    int64_t t_total,
    const c10::optional<at::Tensor>& ext_indices
)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    Slot& s = slot(bsz, q_len, regime);
    TORCH_CHECK(s.configured, "BC_MLAttention: slot not configured");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "BC_MLAttention: x and y must be contiguous");
    TORCH_CHECK(regime == 0 || idx_mode != 0, "BC_MLAttention: sparse regime without indexer");
    TORCH_CHECK(idx_mode != 2 || regime == 0 || ext_indices,
                "BC_MLAttention: shared-indexer sparse step requires external indices");
    TORCH_CHECK(!index_kpool || bsz == 1, "BC_MLAttention: kpool indexer requires bsz 1");

    // First run per slot executes eagerly (GEMM autotune, kernel warmup); the second run is
    // captured, then launched below like every later run, with only the I/O pointers patched
    if (s.runs == 0)
    {
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, regime, t_total, ext_indices, nullptr);
        s.runs = 1;
        return;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_gr(bsz, q_len, s, x, y, cache_seqlens, block_table, position, positions, position_ids, regime, t_total, ext_indices, s.graph.get());
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

    int pid_stride = (position_ids && position_ids.value().dim() == 3) ? rotate_dims : 1;
    if (l4_scaling_beta > 0.0f)
    {
        params.emplace_back(GP_rope_position, (void*) (uintptr_t) (uint32_t) position);
        params.emplace_back(GP_rope_positions, positions ? (void*) positions.value().data_ptr() : nullptr);
        params.emplace_back(GP_rope_position_ids, position_ids ? (void*) position_ids.value().data_ptr() : nullptr);
        params.emplace_back(GP_rope_pid_stride, (void*) (uintptr_t) pid_stride);
    }

    // Latent projection
    params.emplace_back(GP_gemm_A, (void*) x.data_ptr());

    // RoPE position sources (NoPE models never captured the stage)
    if (qk_rope_head_dim > 0)
    {
        params.emplace_back(GP_rope_inv_freq, (void*) inv_freq.data_ptr());
        params.emplace_back(GP_rope_position, (void*) (uintptr_t) (uint32_t) position);
        params.emplace_back(GP_rope_positions, positions ? (void*) positions.value().data_ptr() : nullptr);
        params.emplace_back(GP_rope_position_ids, position_ids ? (void*) position_ids.value().data_ptr() : nullptr);
        params.emplace_back(GP_rope_pid_stride, (void*) (uintptr_t) pid_stride);
    }

    // Cache append: block-table geometry is a runtime kernel argument, patched per call, so
    // context growth never recaptures
    int bt_width = (int) block_table.size(1);
    params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
    params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
    params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);

    // DSA indexer-key stages (full-indexer layers): x staging, key rope and the plane append
    if (idx_mode == 1)
    {
        params.emplace_back(GP_copy2d_src, (void*) x.data_ptr());
        if (qk_rope_head_dim > 0)
        {
            params.emplace_back(GP_rope_inv_freq, (void*) inv_freq.data_ptr());
            params.emplace_back(GP_rope_position, (void*) (uintptr_t) (uint32_t) position);
            params.emplace_back(GP_rope_positions, positions ? (void*) positions.value().data_ptr() : nullptr);
            params.emplace_back(GP_rope_position_ids, position_ids ? (void*) position_ids.value().data_ptr() : nullptr);
            params.emplace_back(GP_rope_pid_stride, (void*) (uintptr_t) pid_stride);
        }
        params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
        params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
        params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
        if (index_kpool)
        {
            // gate plane append, then pool update
            params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
            params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
            params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
            params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
            params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
            params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
        }
    }

    if (regime == 1)
    {
        bool multirow = bsz > 1;
        if (idx_mode == 1)
        {
            // qidx rope, then per mode: the batched slots derive their scoring bounds on
            // device (dsa_seq_state reads cache_seqlens) and the top-k bound is a device
            // pointer, so only the pointer-typed params patch; the single-job slots patch
            // the scalar scan width and clamps
            if (multirow)   // seq-state derive precedes the qidx stages in the graph
                params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
            if (qk_rope_head_dim > 0)
            {
                params.emplace_back(GP_rope_inv_freq, (void*) inv_freq.data_ptr());
                params.emplace_back(GP_rope_position, (void*) (uintptr_t) (uint32_t) position);
                params.emplace_back(GP_rope_positions, positions ? (void*) positions.value().data_ptr() : nullptr);
                params.emplace_back(GP_rope_position_ids, position_ids ? (void*) position_ids.value().data_ptr() : nullptr);
                params.emplace_back(GP_rope_pid_stride, (void*) (uintptr_t) pid_stride);
            }
            if (multirow)
            {
                params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr()); // fewq
                params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
            }
            else
            {
                // kpool: fewq/topk scan widths in POOL units; the expand kernel's row-0
                // position pairs as the second GP_dsa_qpos
                int t_scan = index_kpool ? (int) (t_total / index_kpool) : (int) t_total;
                params.emplace_back(GP_dsa_T, (void*) (uintptr_t) (uint32_t) t_scan);
                params.emplace_back(GP_dsa_qpos, (void*) (uintptr_t) (uint32_t) (int) position);
                params.emplace_back(GP_dsa_bound_max, (void*) (uintptr_t) (uint32_t) t_scan);
                params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
                params.emplace_back(GP_dsa_T, (void*) (uintptr_t) (uint32_t) t_scan);   // topk
                if (index_kpool)
                    params.emplace_back(GP_dsa_qpos, (void*) (uintptr_t) (uint32_t) (int) position);
            }
        }
        at::Tensor idx_t = ext_indices ? ext_indices.value() : s.indices;
        params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
        params.emplace_back(GP_dsa_indices, (void*) idx_t.data_ptr());
        if (multirow)
            params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);   // gather
    }
    else
    {
        int num_splits, split_len;
        mla_split_config(bt_width, page_size, s.block_n, s.splits_cap, &num_splits, &split_len);
        params.emplace_back(GP_attn_block_table, (void*) block_table.data_ptr());
        params.emplace_back(GP_attn_seqlens, (void*) cache_seqlens.data_ptr());
        params.emplace_back(GP_attn_split_len, (void*) (uintptr_t) split_len);
        params.emplace_back(GP_attn_num_pages, (void*) (uintptr_t) bt_width);
        params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);
        params.emplace_back(GP_attn_num_splits, (void*) (uintptr_t) num_splits);   // combine kernel
    }

    // Output projection
    params.emplace_back(GP_gemm_C, (void*) y.data_ptr());

    s.graph->launch(params, stream);
}
