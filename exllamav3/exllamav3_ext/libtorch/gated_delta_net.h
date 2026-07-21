#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include "linear.h"
#include "gated_rmsnorm.h"

struct BC_GatedDeltaNet
{
    at::Tensor mixed_qkv;
    at::Tensor z;
    at::Tensor beta;
    at::Tensor g;
    at::Tensor qkvz;
    at::Tensor ba;
    at::Tensor conv_temp_a;
    at::Tensor conv_temp_b;
    at::Tensor core_attn_out;
    at::Tensor core_attn_out_f;
    std::shared_ptr<BC_LinearEXL3> qkvz_proj;
    std::shared_ptr<BC_LinearFP16> ba_proj;
    at::Tensor dt_bias;
    at::Tensor a_log;
    int num_k_heads;
    int num_v_heads;
    int k_head_dim;
    int v_head_dim;
    at::Tensor conv1d_weight;
    c10::optional<at::Tensor> conv1d_bias;
    std::shared_ptr<BC_GatedRMSNorm> norm;
    std::shared_ptr<BC_LinearEXL3> o_proj;
    const float beta_scale;

    BC_GatedDeltaNet
    (
        at::Tensor _mixed_qkv,
        at::Tensor _z,
        at::Tensor _beta,
        at::Tensor _g,
        at::Tensor _qkvz,
        at::Tensor _ba,
        at::Tensor _conv_temp_a,
        at::Tensor _conv_temp_b,
        at::Tensor _core_attn_out,
        at::Tensor _core_attn_out_f,
        std::shared_ptr<BC_LinearEXL3> _qkvz_proj,
        std::shared_ptr<BC_LinearFP16> _ba_proj,
        at::Tensor _dt_bias,
        at::Tensor _a_log,
        int _num_k_heads,
        int _num_v_heads,
        int _k_head_dim,
        int _v_head_dim,
        at::Tensor _conv1d_weight,
        c10::optional<at::Tensor> _conv1d_bias,
        std::shared_ptr<BC_GatedRMSNorm> _norm,
        std::shared_ptr<BC_LinearEXL3> _o_proj,
        const float _beta_scale
    ) :
        mixed_qkv       (std::move(_mixed_qkv)),
        z               (std::move(_z)),
        beta            (std::move(_beta)),
        g               (std::move(_g)),
        qkvz            (std::move(_qkvz)),
        ba              (std::move(_ba)),
        conv_temp_a     (std::move(_conv_temp_a)),
        conv_temp_b     (std::move(_conv_temp_b)),
        core_attn_out   (std::move(_core_attn_out)),
        core_attn_out_f (std::move(_core_attn_out_f)),
        qkvz_proj       (_qkvz_proj),
        ba_proj         (_ba_proj),
        dt_bias         (std::move(_dt_bias)),
        a_log           (std::move(_a_log)),
        num_k_heads     (_num_k_heads),
        num_v_heads     (_num_v_heads),
        k_head_dim      (_k_head_dim),
        v_head_dim      (_v_head_dim),
        conv1d_weight   (std::move(_conv1d_weight)),
        conv1d_bias     (std::move(_conv1d_bias)),
        norm            (_norm),
        o_proj          (_o_proj),
        beta_scale      (_beta_scale)
    {}

    at::Tensor run_bsz1_a
    (
        const at::Tensor& x
    );

    void run_bsz1_b
    (
        at::Tensor& mixed_qkv,
        at::Tensor& y,
        at::Tensor& recurrent_state
    );
};


// Split-projection variant (Qwen3.5: in_proj_qkv/in_proj_z/in_proj_b/in_proj_a). b/a projections
// are merged into one fp16 GEMV at construction.
//
// Split-projection GDN, generalized over (bsz, seqlen) up to MAX_BSZ x MAX_QLEN and over
// save_history (captured graph is tied to a specific history-branch kernel). Each (bsz, seqlen)
// shape gets its own lazily-configured Slot with exact-size scratch statics
struct BC_GatedDeltaNetSplit
{
    static constexpr int MAX_BSZ = 8;
    static constexpr int MAX_QLEN = 16;

    std::shared_ptr<BC_LinearEXL3> qkv_proj;
    std::shared_ptr<BC_LinearEXL3> z_proj;
    std::shared_ptr<BC_LinearEXL3> o_proj;
    at::Tensor ba_weight_t;       // (2*Nv, hidden) half
    c10::optional<at::Tensor> ba_bias;
    at::Tensor dt_bias;
    at::Tensor a_log;
    int num_k_heads;
    int num_v_heads;
    int k_head_dim;
    int v_head_dim;
    at::Tensor conv1d_weight;     // (F, K) bfloat16
    c10::optional<at::Tensor> conv1d_bias;
    std::shared_ptr<BC_GatedRMSNorm> norm;
    const float beta_scale;

    struct Slot
    {
        bool configured = false;

        // Per-shape statics (python tensor cache), exact-sized for this slot's (bsz, seqlen)
        at::Tensor qkv;               // (bsz, seqlen, F) float
        at::Tensor z;                 // (bsz, seqlen, Nv, Hv) float
        at::Tensor z_flat;            // (bsz, seqlen, Nv*Hv) view of z
        at::Tensor ba;                // (bsz, seqlen, 2*Nv) float
        at::Tensor beta;              // (bsz, seqlen, Nv) bfloat16
        at::Tensor g;                 // (bsz, seqlen, Nv) float
        at::Tensor mixed_qkv;         // (bsz, F, seqlen) bfloat16
        at::Tensor conv_out;          // (bsz, seqlen, F) bfloat16
        at::Tensor core_attn_out;     // (bsz, seqlen, Nv, Hv) bfloat16
        at::Tensor core_attn_out_f;   // (bsz, seqlen, Nv*Hv) half

        // Hadamard scratch for the bypassed exl3_gemm_gr calls (qkv_proj/z_proj/o_proj), shaped
        // like each projection's own input
        at::Tensor qkv_xh, z_xh, o_xh;

        std::unique_ptr<Graph> graph;
    };
    std::vector<Slot> slots;        // history == false (only seqlen == 1 ever populated)
    std::vector<Slot> slots_hist;   // history == true

    int graph_state_size;
    int graph_hist_stride;

    BC_GatedDeltaNetSplit
    (
        std::shared_ptr<BC_LinearEXL3> _qkv_proj,
        std::shared_ptr<BC_LinearEXL3> _z_proj,
        std::shared_ptr<BC_LinearEXL3> _o_proj,
        at::Tensor _ba_weight_t,
        c10::optional<at::Tensor> _ba_bias,
        at::Tensor _dt_bias,
        at::Tensor _a_log,
        int _num_k_heads,
        int _num_v_heads,
        int _k_head_dim,
        int _v_head_dim,
        at::Tensor _conv1d_weight,
        c10::optional<at::Tensor> _conv1d_bias,
        std::shared_ptr<BC_GatedRMSNorm> _norm,
        const float _beta_scale
    ) :
        qkv_proj        (_qkv_proj),
        z_proj          (_z_proj),
        o_proj          (_o_proj),
        ba_weight_t     (std::move(_ba_weight_t)),
        ba_bias         (std::move(_ba_bias)),
        dt_bias         (std::move(_dt_bias)),
        a_log           (std::move(_a_log)),
        num_k_heads     (_num_k_heads),
        num_v_heads     (_num_v_heads),
        k_head_dim      (_k_head_dim),
        v_head_dim      (_v_head_dim),
        conv1d_weight   (std::move(_conv1d_weight)),
        conv1d_bias     (std::move(_conv1d_bias)),
        norm            (_norm),
        beta_scale      (_beta_scale),
        graph_state_size(-1),
        graph_hist_stride(-1)
    {
        slots.resize(MAX_BSZ * MAX_QLEN);
        slots_hist.resize(MAX_BSZ * MAX_QLEN);
    }

    Slot& slot(int bsz, int seqlen, bool history)
    {
        std::vector<Slot>& v = history ? slots_hist : slots;
        return v[(bsz - 1) * MAX_QLEN + (seqlen - 1)];
    }

    bool needs_configure(int bsz, int seqlen, bool history);

    void configure_slot
    (
        int bsz,
        int seqlen,
        bool history,
        at::Tensor qkv,
        at::Tensor z,
        at::Tensor ba,
        at::Tensor beta,
        at::Tensor g,
        at::Tensor mixed_qkv,
        at::Tensor conv_out,
        at::Tensor core_attn_out,
        at::Tensor core_attn_out_f,
        at::Tensor qkv_xh,
        at::Tensor z_xh,
        at::Tensor o_xh
    );

    void run_bszN_gr
    (
        const at::Tensor& x,
        at::Tensor& y,
        at::Tensor& conv_state,
        at::Tensor& recurrent_state,
        const at::Tensor& slots,
        bool history,
        Slot& s,
        Graph* graph
    );

    void run_bszN
    (
        const at::Tensor& x,
        at::Tensor& y,
        at::Tensor& conv_state,
        at::Tensor& recurrent_state,
        const at::Tensor& slots,
        bool history
    );
};


// Mamba2 (NemotronH): in_proj -> [z, xBC, dt] split -> conv1d -> SSD recurrence -> grouped gated
// norm -> o_proj.
//
// Mamba2 (NemotronH), generalized over (bsz, seqlen), see BC_GatedDeltaNetSplit above. Padded
// projection dims (hidden % 128 != 0) stage through per-slot zero-padded statics like BC_Attention:
// x copies into xp at the graph head, the padded o_proj output trims into y at the tail
struct BC_Mamba2
{
    static constexpr int MAX_BSZ = 8;
    static constexpr int MAX_QLEN = 16;

    std::shared_ptr<BC_LinearEXL3> in_proj;
    std::shared_ptr<BC_LinearEXL3> o_proj;
    at::Tensor dt_bias;             // [H] float
    at::Tensor a_log;               // [H] float
    at::Tensor d_skip;              // [H] float
    float dt_min;
    float dt_max;
    int dt_first;                   // TP: rank's head offset into the replicated dt section
    int num_k_heads;
    int num_v_heads;
    int k_head_dim;
    int v_head_dim;
    int hidden_size;                // exact width of x and y
    at::Tensor conv1d_weight;       // (F, K) bfloat16
    c10::optional<at::Tensor> conv1d_bias;
    std::shared_ptr<BC_GatedRMSNorm> norm;
    int v_dim;
    bool padded_in;                 // in_proj K padded (hidden_size != in_proj->K)
    bool padded_out;                // o_proj N padded (hidden_size != o_proj->N)

    struct Slot
    {
        bool configured = false;

        c10::optional<at::Tensor> xp;   // (R, K_padded) half, zero-padded input staging
        at::Tensor proj;                // (bsz, seqlen, N_padded) float, in_proj output
        at::Tensor mixed_xbc;           // (bsz, F, seqlen) bfloat16, conv input
        at::Tensor dt;                  // (bsz, seqlen, H) bfloat16
        at::Tensor g;                   // (bsz, seqlen, H) float
        at::Tensor conv_out;            // (bsz, seqlen, F) bfloat16
        at::Tensor core_attn_out;       // (bsz, seqlen, H, Hv) bfloat16
        at::Tensor core_attn_out_f;     // (bsz, seqlen, H*Hv) half
        c10::optional<at::Tensor> yp;   // (R, No_padded) o_dtype, padded o_proj output

        // z_gate is its own contiguous buffer, NOT a narrow()+view() of proj: proj's row stride
        // is N_padded (> v_dim), so such a view has gaps between rows once bsz*seqlen > 1 that
        // gated_rms_norm_kernel's raw rows*dim indexing silently misreads. mamba2_fused_op_gr
        // writes a real contiguous copy of proj[.., :v_dim] into it instead
        at::Tensor z_gate;              // (bsz, seqlen, groups, gs) float, contiguous
        at::Tensor core_g;              // (bsz, seqlen, groups, gs) view of core_attn_out (contiguous, safe)
        at::Tensor core_f_g;            // (bsz, seqlen, groups, gs) view of core_attn_out_f (contiguous, safe)

        // Hadamard scratch for the bypassed exl3_gemm_gr calls (in_proj/o_proj)
        at::Tensor in_xh, o_xh;

        std::unique_ptr<Graph> graph;
    };
    std::vector<Slot> slots;        // history == false (only seqlen == 1 ever populated)
    std::vector<Slot> slots_hist;   // history == true

    int graph_state_size;
    int graph_hist_stride;

    BC_Mamba2
    (
        std::shared_ptr<BC_LinearEXL3> _in_proj,
        std::shared_ptr<BC_LinearEXL3> _o_proj,
        at::Tensor _dt_bias,
        at::Tensor _a_log,
        at::Tensor _d_skip,
        float _dt_min,
        float _dt_max,
        int _num_k_heads,
        int _num_v_heads,
        int _k_head_dim,
        int _v_head_dim,
        int _hidden_size,
        at::Tensor _conv1d_weight,
        c10::optional<at::Tensor> _conv1d_bias,
        std::shared_ptr<BC_GatedRMSNorm> _norm,
        bool _padded_in,
        bool _padded_out,
        int _dt_first = 0
    ) :
        in_proj         (_in_proj),
        o_proj          (_o_proj),
        dt_bias         (std::move(_dt_bias)),
        a_log           (std::move(_a_log)),
        d_skip          (std::move(_d_skip)),
        dt_min          (_dt_min),
        dt_max          (_dt_max),
        num_k_heads     (_num_k_heads),
        num_v_heads     (_num_v_heads),
        k_head_dim      (_k_head_dim),
        v_head_dim      (_v_head_dim),
        hidden_size     (_hidden_size),
        conv1d_weight   (std::move(_conv1d_weight)),
        conv1d_bias     (std::move(_conv1d_bias)),
        norm            (_norm),
        padded_in       (_padded_in),
        padded_out      (_padded_out),
        dt_first        (_dt_first),
        graph_state_size(-1),
        graph_hist_stride(-1)
    {
        v_dim = num_v_heads * v_head_dim;
        slots.resize(MAX_BSZ * MAX_QLEN);
        slots_hist.resize(MAX_BSZ * MAX_QLEN);
    }

    Slot& slot(int bsz, int seqlen, bool history)
    {
        std::vector<Slot>& v = history ? slots_hist : slots;
        return v[(bsz - 1) * MAX_QLEN + (seqlen - 1)];
    }

    bool needs_configure(int bsz, int seqlen, bool history);

    void configure_slot
    (
        int bsz,
        int seqlen,
        bool history,
        c10::optional<at::Tensor> xp,
        at::Tensor proj,
        at::Tensor mixed_xbc,
        at::Tensor dt,
        at::Tensor g,
        at::Tensor z_gate,
        at::Tensor conv_out,
        at::Tensor core_attn_out,
        at::Tensor core_attn_out_f,
        c10::optional<at::Tensor> yp,
        at::Tensor in_xh,
        at::Tensor o_xh
    );

    void run_bszN_gr
    (
        const at::Tensor& x,
        at::Tensor& y,
        at::Tensor& conv_state,
        at::Tensor& recurrent_state,
        const at::Tensor& slots,
        bool history,
        Slot& s,
        Graph* graph
    );

    void run_bszN
    (
        const at::Tensor& x,
        at::Tensor& y,
        at::Tensor& conv_state,
        at::Tensor& recurrent_state,
        const at::Tensor& slots,
        bool history
    );
};