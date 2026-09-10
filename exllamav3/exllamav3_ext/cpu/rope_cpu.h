#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

// Fused rotary position embedding for CPU tensors (vision-tower CPU offload path).
//
// Same math as the CUDA rope kernel (rope.cu rope_kernel), computed in float32 with a
// single round to the output dtype. Covers the three vision call shapes, selected by
// the inv_freq geometry and the position source (mirroring rope_gr, rotate_offset 0):
//
// - frequency vector (inv_freq 1-D, P = numel * 2): angles are pos * freq, with pos
//   resolving position_ids (2-D shared [bsz, seq], or 3-D per-section
//   [bsz, seq, rotate_dims] for Gemma4 partial rotary) > positions[batch] + token >
//   position + token. rotate_dims sections at offset P * rdim; rotate_dims == 1
//   rotates the leading P columns, otherwise head_dim must equal P * rotate_dims.
// - angle table (inv_freq.dim() > 1, rows are precomputed angles [.., P/2]):
//   3-D tables carry batch first (Step3.7 patches: row = table[batch, pos]);
//   2-D tables are indexed by position only (Mistral gathered table).
//   sin/cos are sin/cos(row) * attn_factor.
//
// rope_style is ROPESTYLE_GPTJ (adjacent-pair rotation, Step3.7) or ROPESTYLE_NEOX
// (split-half rotation); other values throw. attn_factor scales sin/cos.
//
// Fused RMS norm (G4, like rope.cu apply_norm): q_norm/k_norm are optional 1-D [hd]
// f32/f16/bf16 weight tables with norm_eps/norm_constant_bias. Each present table
// norms its tensor over the full head_dim (variance over hd, scale x (w + bias))
// before rotation; tails (rotate_dims == 1, P < hd) are normalized but not rotated.
// q/k gate independently, so the dflash k-only call (weight on the q arg, k absent)
// just works. Weights fold once to float32 per call; the hot loop stays pure float32
// vector math (FMA accumulation, one streaming variance pass + one streaming
// scale+rotate pass, no temp buffer).
//
// Layouts: q/out_q [bsz, seq, nheads_q, hd] with stride(3) == 1 and identical
// strides; k/out_k optional with the same bsz/seq/hd. q/k/out in
// {float32, half, bfloat16} in any combination; inv_freq float32 CPU;
// positions/position_ids int32/int64 CPU when present (at most one of them).
// out may alias its input (the in_place vision path): every row is fully read
// into registers before it is written, so aliasing is exact.
// numel == 0 is a no-op. Anything outside this contract throws; the Python caller
// falls back to the torch reference, so a forward never fails here.
//
// Threading runs on the persistent process-wide pool shared with the MoE CPU path
// (moe_mul1.cpp), one tile per row-block. ISA dispatch (AVX-512 / AVX2 / scalar)
// reuses the MoE ISA detection, overridable per process with
// EXL3_ROPE_CPU_ISA=avx512|avx2|scalar (testing only).
void rope_cpu(
    const at::Tensor& q,
    at::Tensor& out_q,
    const c10::optional<at::Tensor>& k,
    c10::optional<at::Tensor>& out_k,
    const at::Tensor& inv_freq,
    int64_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    int rope_style,
    float attn_factor,
    int rotate_dims,
    const c10::optional<at::Tensor>& q_norm,
    const c10::optional<at::Tensor>& k_norm,
    float norm_eps,
    float norm_constant_bias);
