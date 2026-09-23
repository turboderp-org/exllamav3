#pragma once

#include <ATen/Tensor.h>

// Host-resident trellis embedding table -> device-mapped alias (int64 pointer usable in
// kernels). The tensor must be a contiguous CPU tensor with pinned (page-locked) storage;
// the registration is cached per data pointer, refcounted and idempotent. device_index
// selects the device the alias is mapped for (-1 = current); re-registering the same
// pointer for a different device raises (kernel-usable host pointers need a UVA device
// mapping, verified at register time - unavailable under WDDM, platform-dependent
// elsewhere), and re-registering the same pointer with a different size raises (the
// original tensor was freed without unregister and the address was reused). The
// returned pointer does NOT own the memory: the registered tensor must outlive the
// registration, otherwise the alias dangles (device reads of a freed region). Call
// trellis_embed_unregister before releasing; it syncs the registered device and only
// unmaps registrations this module itself created.
int64_t trellis_embed_register
(
    const at::Tensor& table,
    int64_t device_index
);

void trellis_embed_unregister
(
    const at::Tensor& table
);

// Gather + fused decode: rows ids (n,) int64 CUDA of a host-mapped packed table
// (trellis_embed_register pointer) -> out (n, D) fp32 CUDA, bit-exact with
// embed_trellis.dequant_rows_transformed. cb = mul1 codebook fp16 LUT (65536,),
// col_scales = (D,) fp32, both CUDA-resident; cb/col_scales/ids/out must all live on the
// SAME CUDA device (the kernel launches on cb.device() and dereferences the other
// operands' device pointers; a host pointer is kernel-usable only where UVA maps it,
// verified at register time - unavailable under WDDM). K in {6, 7, 8}.
// n_rows = packed table row count: table_ptr must be a live registration on this
// device and n_rows must fit the registered region (TORCH_CHECK, no sync). ids
// outside [0, n_rows) are clamped to the last row by the kernel (an OOB id would be
// an out-of-bounds read of the device-mapped host memory, not a trapped fault); loud
// rejection is the caller's job (Embedding._gather raises IndexError for host-resident
// ids before the H2D copy).
void trellis_embed_gather
(
    int64_t table_ptr,
    const at::Tensor& cb,
    const at::Tensor& col_scales,
    const at::Tensor& ids,
    int64_t K,
    int64_t seed,
    int64_t n_rows,
    at::Tensor out
);