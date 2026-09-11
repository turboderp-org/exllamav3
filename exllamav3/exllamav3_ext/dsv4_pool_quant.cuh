#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

class Graph;

// Quantize freshly compressed DSA pool entries (fp16 rows staged by dsv4_compress in
// stage_rel mode) into the packed paged pool: the nope part in the cache-quant format
// (32-value groups, H32 rotation, absmax midpoint grid, bit-plane packing; same as
// CacheLayer_quant / CacheLayer_MLA_quant, so the shared Triton loaders read it), the rope
// part copied as fp16. Entry rows are addressed through the job's block-table row at
// position // m + w, exactly like the direct paged store. Reads the position from
// position_tensor when given (graph-safe: the grid is padded and idle blocks retire).
void dsv4_pool_quant_scatter_gr
(
    const at::Tensor& stage,                         // (nw_max, hd) or (B, nw_max, hd) half
    at::Tensor& pool_q,                              // (rows, G * bits) int32, G = D_c / 32
    at::Tensor& pool_s,                              // (rows, G) half
    at::Tensor& pool_r,                              // (rows, hd - D_c) half
    const at::Tensor& pool_bt,                       // (1 or B, num_pages) int32
    int position,
    const c10::optional<at::Tensor>& position_tensor,
    int m,
    int seq,                                         // tokens per job this step
    int pool_epp,
    Graph* graph
);

void dsv4_pool_quant_scatter
(
    const at::Tensor& stage,
    at::Tensor pool_q,
    at::Tensor pool_s,
    at::Tensor pool_r,
    const at::Tensor& pool_bt,
    int position,
    const c10::optional<at::Tensor>& position_tensor,
    int m,
    int seq,
    int pool_epp
);
