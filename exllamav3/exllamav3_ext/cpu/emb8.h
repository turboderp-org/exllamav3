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
// The output is a fresh allocation per call. For large prefill chunks that
// pays a page-fault cost on alloc + free (glibc mmaps/munmaps above its
// dynamic threshold; Windows demand-zero pages cost the same): measured
// N=4096, hidden=5120, 80 MiB fp32 out, the 2.9 ms kernel pass grows to a
// ~9.6 ms alloc-to-free cycle. The cost is bounded by the output size and is
// not visible at decode sizes (N=1, ~20 KB).
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