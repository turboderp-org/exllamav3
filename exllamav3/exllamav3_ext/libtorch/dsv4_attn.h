#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include "linear.h"
#include "dsv4_compressor.h"
#include "../triton_kernel.h"

// Whole-attention-step graph for DeepSeek-V4 DSA decode (BC_Attention v2 pattern): x staged
// via one patched copy2d, then q_a -> q_norm -> q_b / wkv -> fused-norm rope -> compressor
// BC(s) -> [indexer scoring + capture-safe topk, long-ctx regime] -> AOT split/combine
// attention -> wo_a mgemm -> wo_b, plus the SWA ring append, captured per slot and replayed
// with a handful of value patches. Instances are built per (module, cache layer, job slot):
// the per-slot state tensors (ring, compressor rings, snapshot rings, pools) are baked into
// the graphs. Position flows through the pos/ring_beg device scalars (rope positions
// tensor, compressor pos_ptr, ring append); the split/combine/indexer bounds are patched
// scalar args. bsz 1 per call, seq 1..MAX_QLEN; regime = dense (ec <= topk) or gathered.

struct BC_DSV4Attention
{
    static constexpr int MAX_QLEN = 16;

    // Projections (EXL3 only; eligibility enforced python-side)
    std::shared_ptr<BC_LinearEXL3> q_a;
    std::shared_ptr<BC_LinearEXL3> q_b;
    std::shared_ptr<BC_LinearEXL3> wkv;
    std::shared_ptr<BC_LinearEXL3> wo_b;
    std::shared_ptr<BC_LinearEXL3> idx_wq_b;        // csa only
    c10::optional<at::Tensor> idx_weights_w;        // (hidden, H_i) fp16 (cublas, static in)

    std::shared_ptr<BC_DSV4Compressor> comp_bc;     // null for sliding layers
    std::shared_ptr<BC_DSV4Compressor> idx_bc;      // csa only

    // wo_a group slices as a multilinear (uniform K/mcg/mul1)
    at::Tensor woa_trellis, woa_suh, woa_svh, woa_indices;
    int woa_k;
    bool woa_mcg, woa_mul1;

    // x-side projection fan: q_a / wkv / comp wkv+wgate / idx wkv+wgate as ONE per-matrix-N
    // mgemm (requires uniform bits/format across the group; python declines otherwise)
    c10::optional<at::Tensor> fan_trellis, fan_suh, fan_svh, fan_n, fan_indices;

    // Tables
    at::Tensor q_norm_w;         // (q_lora)
    at::Tensor q_ones;           // (hd) unweighted q head norm
    at::Tensor kv_norm_w;        // (hd)
    at::Tensor inv_freq;         // (rd / 2) fp32, layer's rope table
    at::Tensor inv_freq_neg;     // negated (derot)
    at::Tensor sinks;            // (H) fp32

    // Per-slot state (baked): SWA ring + compressor state + pools + device position scalars
    at::Tensor ring;             // (ring_rows, D)
    c10::optional<at::Tensor> comp_buf_kv, comp_buf_gate, comp_ovl, pool_c, pool_r;
    c10::optional<at::Tensor> idx_buf_kv, idx_buf_gate, idx_ovl, pool_idx;
    at::Tensor pool_bt;          // (1, cap / PAGE_SIZE) int32 identity
    at::Tensor pos_dev;          // (1,) int32 -- shared per (device, job slot)
    at::Tensor ring_beg_dev;     // (1,) int32

    // Dims / config
    int hidden_size, num_q_heads, head_dim, rope_dim, q_lora, o_groups, o_lora;
    int index_n_heads, index_head_dim, index_topk;
    int window, m;
    float sm_scale, rms_norm_eps;
    int n_splits, block_h;

    struct Slot
    {
        bool configured = false;

        at::Tensor x_st;                                // (seq, hidden) half, staged input
        at::Tensor qa_st;                               // (seq, q_lora) half
        at::Tensor qres_st;                             // (seq, q_lora) half (post q_norm)
        at::Tensor q_st;                                // (seq, H * hd) half
        at::Tensor kv_st;                               // (seq, hd) half
        at::Tensor xh_a, xh_b;                          // gemm hadamard scratch: (seq, hidden), (seq, q_lora)
        c10::optional<at::Tensor> mgc_comp, mgc_idx;    // (2, seq, W) mgemm outs
        c10::optional<at::Tensor> qidx_st;              // (seq, H_i * D_i) half
        c10::optional<at::Tensor> wts_st;               // (seq, H_i) half
        c10::optional<at::Tensor> scores_st;            // (seq, S_max) half, -inf init
        c10::optional<at::Tensor> indices_st;           // (seq, K_pad) int32
        at::Tensor ws_ml, ws_acc;                       // split workspaces fp32
        at::Tensor attn_out_st;                         // (G, seq, hpg * hd) half
        at::Tensor woa_c_st;                            // (G, seq, o_lora) half
        at::Tensor woa_t_st;                            // (seq, G * o_lora) half (seq > 1 transpose; else view)
        at::Tensor woa_xh;                              // (G * seq, hpg * hd) half mgemm scratch
        at::Tensor y_st;                                // (seq, hidden) out dtype fp32
        c10::optional<at::Tensor> fan_c_ptrs;           // (F,) int64 -> slot statics
        c10::optional<at::Tensor> fan_ahad;             // (F, seq, hidden) half

        std::shared_ptr<TritonKernel> k_split, k_combine, k_fewq;
        std::unique_ptr<Graph> graph;
    };
    std::vector<Slot> slots;                            // [seq - 1][regime]

    BC_DSV4Attention
    (
        std::shared_ptr<BC_LinearEXL3> _q_a,
        std::shared_ptr<BC_LinearEXL3> _q_b,
        std::shared_ptr<BC_LinearEXL3> _wkv,
        std::shared_ptr<BC_LinearEXL3> _wo_b,
        std::shared_ptr<BC_LinearEXL3> _idx_wq_b,
        c10::optional<at::Tensor> _idx_weights_w,
        std::shared_ptr<BC_DSV4Compressor> _comp_bc,
        std::shared_ptr<BC_DSV4Compressor> _idx_bc,
        at::Tensor _woa_trellis, at::Tensor _woa_suh, at::Tensor _woa_svh,
        at::Tensor _woa_indices, int _woa_k, bool _woa_mcg, bool _woa_mul1,
        at::Tensor _q_norm_w, at::Tensor _q_ones, at::Tensor _kv_norm_w,
        at::Tensor _inv_freq, at::Tensor _inv_freq_neg, at::Tensor _sinks,
        at::Tensor _ring,
        c10::optional<at::Tensor> _comp_buf_kv, c10::optional<at::Tensor> _comp_buf_gate,
        c10::optional<at::Tensor> _comp_ovl,
        c10::optional<at::Tensor> _pool_c, c10::optional<at::Tensor> _pool_r,
        c10::optional<at::Tensor> _idx_buf_kv, c10::optional<at::Tensor> _idx_buf_gate,
        c10::optional<at::Tensor> _idx_ovl, c10::optional<at::Tensor> _pool_idx,
        at::Tensor _pool_bt, at::Tensor _pos_dev, at::Tensor _ring_beg_dev,
        int _hidden_size, int _num_q_heads, int _head_dim, int _rope_dim, int _q_lora,
        int _o_groups, int _o_lora, int _index_n_heads, int _index_head_dim, int _index_topk,
        int _window, int _m, float _sm_scale, float _rms_norm_eps,
        int _n_splits, int _block_h,
        c10::optional<at::Tensor> _fan_trellis = {}, c10::optional<at::Tensor> _fan_suh = {},
        c10::optional<at::Tensor> _fan_svh = {}, c10::optional<at::Tensor> _fan_n = {},
        c10::optional<at::Tensor> _fan_indices = {}
    ) :
        q_a(_q_a), q_b(_q_b), wkv(_wkv), wo_b(_wo_b), idx_wq_b(_idx_wq_b),
        idx_weights_w(std::move(_idx_weights_w)),
        comp_bc(_comp_bc), idx_bc(_idx_bc),
        woa_trellis(std::move(_woa_trellis)), woa_suh(std::move(_woa_suh)),
        woa_svh(std::move(_woa_svh)), woa_indices(std::move(_woa_indices)),
        woa_k(_woa_k), woa_mcg(_woa_mcg), woa_mul1(_woa_mul1),
        q_norm_w(std::move(_q_norm_w)), q_ones(std::move(_q_ones)),
        kv_norm_w(std::move(_kv_norm_w)),
        inv_freq(std::move(_inv_freq)), inv_freq_neg(std::move(_inv_freq_neg)),
        sinks(std::move(_sinks)),
        ring(std::move(_ring)),
        comp_buf_kv(std::move(_comp_buf_kv)), comp_buf_gate(std::move(_comp_buf_gate)),
        comp_ovl(std::move(_comp_ovl)),
        pool_c(std::move(_pool_c)), pool_r(std::move(_pool_r)),
        idx_buf_kv(std::move(_idx_buf_kv)), idx_buf_gate(std::move(_idx_buf_gate)),
        idx_ovl(std::move(_idx_ovl)), pool_idx(std::move(_pool_idx)),
        pool_bt(std::move(_pool_bt)), pos_dev(std::move(_pos_dev)),
        ring_beg_dev(std::move(_ring_beg_dev)),
        hidden_size(_hidden_size), num_q_heads(_num_q_heads), head_dim(_head_dim),
        rope_dim(_rope_dim), q_lora(_q_lora), o_groups(_o_groups), o_lora(_o_lora),
        index_n_heads(_index_n_heads), index_head_dim(_index_head_dim),
        index_topk(_index_topk), window(_window), m(_m),
        sm_scale(_sm_scale), rms_norm_eps(_rms_norm_eps),
        n_splits(_n_splits), block_h(_block_h),
        fan_trellis(std::move(_fan_trellis)), fan_suh(std::move(_fan_suh)),
        fan_svh(std::move(_fan_svh)), fan_n(std::move(_fan_n)),
        fan_indices(std::move(_fan_indices))
    {
        slots.resize(MAX_QLEN * 2);
    }

    Slot& slot(int seq, int regime) { return slots[(seq - 1) * 2 + regime]; }

    bool needs_configure(int seq, int regime);

    void configure_slot
    (
        int seq, int regime,
        at::Tensor x_st, at::Tensor qa_st, at::Tensor qres_st, at::Tensor q_st,
        at::Tensor kv_st, at::Tensor xh_a, at::Tensor xh_b,
        c10::optional<at::Tensor> mgc_comp, c10::optional<at::Tensor> mgc_idx,
        c10::optional<at::Tensor> qidx_st, c10::optional<at::Tensor> wts_st,
        c10::optional<at::Tensor> scores_st, c10::optional<at::Tensor> indices_st,
        at::Tensor ws_ml, at::Tensor ws_acc, at::Tensor attn_out_st,
        at::Tensor woa_c_st, at::Tensor woa_t_st, at::Tensor woa_xh, at::Tensor y_st,
        std::shared_ptr<TritonKernel> k_split,
        std::shared_ptr<TritonKernel> k_combine,
        std::shared_ptr<TritonKernel> k_fewq,
        c10::optional<at::Tensor> fan_c_ptrs = {},
        c10::optional<at::Tensor> fan_ahad = {}
    );

    void run_gr(const at::Tensor& x, int seq, int regime, int pos, int win_beg,
                Slot& s, class Graph* graph);

    // Returns the y static (seq, hidden) fp32; consumed by the caller before the next replay
    at::Tensor run(const at::Tensor& x, int pos, int win_beg, int regime);
};
