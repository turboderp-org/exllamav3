#pragma once

#include <ATen/Tensor.h>

// CPU gather-dequant for int8 (q8_0-style, 32-column block scales) embed_tokens
// storage, replacing the multi-pass torch index_select + cast chain in
// exllamav3/modules/embedding.py. One pass per row: read 32 int8 + 1 fp16 scale,
// write 32 fp32 or fp16 outputs; threaded over contiguous row chunks.
//
// fp16 output is bit-exact with the torch path (ATen CPU fp16 elementwise ops use
// opmath_t = float; vcvtps2ph rounds to nearest even like c10::Half). fp32 output
// is exact vs the reference dequant (no intermediate rounding).
//
// The output is a fresh allocation per call; its pages are unmapped by a
// background janitor thread instead of on the caller's path (a fresh >32 MB
// output costs ~4 ms in page faults on alloc and ~4 ms on free - glibc
// mmaps/munmaps above its dynamic threshold, Windows demand-zero pages pay
// the same). Measured N=4096, hidden=5120, 80 MiB fp32 out: the alloc-to-
// free cycle drops from ~9.6 ms to ~6.3 ms; steady-state retained RAM is
// zero. Inconsequential at decode sizes (N=1, ~20 KB).
//
// The caller must release the GIL around this call.

namespace EXL3
{

// q_table: [V, hidden] int8 CPU; scale_table: [V, hidden/32] float16 CPU;
// ids: [n] int64 CPU. Returns [n, hidden] float32 (f32_out) or float16.
at::Tensor emb8_dequant
(
    const at::Tensor& q_table,
    const at::Tensor& scale_table,
    const at::Tensor& ids,
    bool f32_out
);

} // namespace EXL3