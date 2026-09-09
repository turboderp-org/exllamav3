# BF16 decode extension API

These entry points accept BF16 activations and write BF16 output around the existing EXL3 FP16 Hadamard/GEMM calculation. They use the existing packed K5/K6 weights. The FP16 APIs and checkpoint format are unchanged.

The current implementation supports MCG decode with 1 <= M <= 16 and positive K/N dimensions divisible by 128. It requires compute capability 8.0 or newer, cooperative launches and 90 KiB of dynamic shared memory. The wrapper checks cooperative residency before launch. Runtime validation covers RTX 5090 (SM120); these capability checks do not establish runtime coverage on other GPUs.

## Buffers

All activation, weight and workspace tensors must be contiguous and on the input device, except `had_group_ids`, which is CPU metadata. Outputs and workspaces are caller-owned. Keep the tensors referenced by pointer tables alive until the launch completes, and for the lifetime of any captured graph. The wrapper validates pointer-table shapes and dtypes, but cannot validate the allocation or contents behind each raw pointer.

| Entry point | Input and weights | Workspaces | Output |
| --- | --- | --- | --- |
| `exl3_gemm_bf16_io` | BF16 A[M,K]; int16 B[K/16,N/16,bits*16]; FP16 suh[K], svh[N] | FP16 A_had[M,K], C_scratch[M,N] | BF16 C[M,N] |
| `exl3_mgemm_bf16_io` | BF16 A[M,K]; CUDA int64 pointer tables B, suh, svh of length count | FP16 A_had[count,M,K], C_scratch[count,M,N] | BF16 C[M,output_stride] |
| `exl3_mgemm_bf16_io_grouped_had` | As MGEMM, but unique_suh has one pointer per distinct input transform; CPU int32 had_group_ids[count] selects the transform | FP16 A_had[group_count,M,K], C_scratch[count,M,N] | BF16 C[M,output_stride] |

MGEMM bundles use a common K, N and bit width. Matrix j writes columns `[j*N:(j+1)*N]`; `output_stride` must equal the allocated output width and be at least count*N. Trailing columns are untouched. Referenced weights have the single-GEMM packed shape, and each suh/svh points to its corresponding FP16 vector. The caller remains responsible for those pointed-to layouts.

Group IDs are immutable launch metadata. The wrapper validates their range on the CPU and passes the values into the kernel arguments. Changing the CPU tensor after graph capture does not update the captured arguments; recapture when the mapping changes. Grouped-Hadamard shares a transform only when the matrices use the same input vector, not merely vectors of the same length.

## Launch options

Single GEMM takes `force_shape_idx=2` and `mcg=True`. MGEMM takes `bits=5` or `bits=6` and `mcg=True`. `force_num_sms=0` selects a grid from device properties and kernel occupancy. Positive overrides must fit the device and cooperative residency limits.

MGEMM's `direct_output` chooses between the scratch-output path and a fused BF16 output epilogue. `final_group_barrier` and single GEMM's `final_grid_sync` control terminal synchronization. Skipping a group barrier requires the matrices to be resident; grouped-Hadamard requires residency for the complete bundle. Internal synchronization required by the computation remains in place.

The extension uses device-level lock storage inherited from EXL3. This contribution has not established safety for overlapping independent calls on different streams. Callers should serialize operations using that shared storage.

## Numerical behavior and tests

The BF16 boundary changes where conversion and rounding occur. Tests compare it with the existing FP16 boundary using a relative-RMS threshold of 0.003; they do not require bitwise equality between those two paths. Direct/scratch and ordinary/grouped modes, and graph replay versus eager execution of the same input, are checked exactly in the covered cases.

Run from a checkout with the BF16 extension built and installed:

```bash
python3 -m pytest -q tests/test_exl3_bf16_io.py
```

The fixtures construct packed weights and transform vectors without model files. The suite covers K5/K6, M1/M4/M16, multiple Hadamard groups, output padding, repeated changing-input graph replay and invalid launch metadata. The larger synthetic bundle uses K5120/N512 with four matrices; it checks the API behavior without claiming model-quality coverage.

## Interface review

The three entry points keep the current prototype separate from the existing per-N MGEMM API. For an upstream interface, the open decision is whether BF16 output, shared input transforms and output placement should become options of that existing API. This patch contains extension code and tests; it does not install a vLLM caller or change ExLlamaV3's default execution path. M24/M32 kernels and downstream EXL3/MXFP6 routing are separate work.
