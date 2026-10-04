# ROCm support (RDNA3 / gfx11)

ExLlamaV3 runs on AMD RDNA GPUs through PyTorch's ROCm builds. Scope is **wave32
parts only** (RDNA, gfx10/gfx11); CDNA (wave64) devices are rejected on first device use.
Developed and measured on gfx1100 (RX 7900 XTX).

## Building

With a ROCm-enabled PyTorch installed (`torch.version.hip` set), the extension
builds through the same paths as CUDA: `pip install .` for a precompiled
install, or the JIT build on first import. `PYTORCH_ROCM_ARCH` is derived from
the visible GPUs' `gcnArchName` if unset.

`Dockerfile.rocm` builds a complete image (rocm-dev base, torch rocm wheel,
extension compiled at image build time):

```
docker build -f Dockerfile.rocm -t exllamav3:rocm .
docker run --rm -it --device=/dev/kfd --device=/dev/dri exllamav3:rocm
```

## What the port changes

- `exllamav3_ext/hip_compat.cuh` — shims for CUDA constructs hipify does not
  translate: widened warp-sync masks, `__dp4a` via `__builtin_amdgcn_sudot4`,
  `__hip_atomic_*` scoped atomics for the TP collectives, nontemporal
  load/stores, `__nanosleep`, `__hmax2`/`__hmin2`, bf16 conversions.
- `ptx.cuh` — m16n8k16 MMA emulated with `__shfl` math; `cp.async`, `ldmatrix`,
  `lop3` and grid barriers replaced with portable equivalents.
- `cuda_drv.*`, `triton_kernel.*`, `graph.*` — HIP module/graph APIs stand in
  for the CUDA driver API (Triton kernels load hsaco; warp size is read from
  the device instead of assumed 32).
- `exl3_devctx` — `CC_RDNA3` device class from `gcnArchName`.
- Cooperative launches: RDNA over-reports co-residency, so occupancy launches
  one fewer block/SM and coop autotune concurrency is clamped to 1 to avoid
  `grid.sync()` deadlocks.
- `cpu/moe_handoff.cu` — `hipStreamWriteValue32`/`WaitValue32` memops for the
  CPU-MoE offload handshake.

## CUDA-only kernels

`dflash2.cu`, `hc_mix_tiled.cu` and `routing_gemm.cu` contain inline PTX
(tensor-core MMA, cp.async, ldmatrix) that does not hipify. They are excluded
from HIP builds; their bindings are `#if !defined(USE_ROCM)`. The affected
architectures (`DFlash2`, `KimiLinear`, `MiMoV2`) fail at config load with an
explicit "not supported on ROCm" message; `gr_mix_tiled` and the deterministic
int8 router projection fall back to the cuBLAS/GEMV paths, which remain
correct (those paths trade determinism across ranks, not accuracy).

## Performance notes (gfx1100)

- Decode: the int8 GEMV `msq` dispatch (one regular launch per MGEMM call)
  replaces the cooperative kernel that underfills RDNA; int8 GEMV max K is 6
  on RDNA3. `EXL3_CUMODE=1` builds with `-mcumode` (each workgroup pinned to
  one CU rather than a WGP pair), measured +7.6% decode.
- `EXL3_RECONSTRUCT_THRESHOLD`: the GEMV/reconstruct row crossover is ~16 on
  RDNA3 vs 144 on NVIDIA; `Dockerfile.rocm` sets 16.
- `EXL3_HGEMM_F16OUT` defaults on under HIP: the fp32-output reconstruct GEMM
  is ~5x slower than fp16-out + widen on RDNA3.
- `EXL3_BC_ATTN` defaults off under HIP: the captured decode-attention block
  replays slower than eager dispatch on gfx1100.
- `EXL3_GDN_CHUNK_MIN`/`EXL3_GDN_BC_MAX_QLEN` gate the GDN chunk kernel and
  HIP-graph captures on tiny prefills.

## Numerics
Decode output is not bitwise identical across batch sizes on RDNA3, same as on
CUDA: small batches take the per-row-quantized int8 GEMV path (m <= 4 on ROCm,
m <= 2 upstream on CUDA; ~0.9% output RMS deviation) while larger batches take
the fp16 reconstruct + GEMM path. A near-tie logit can flip one greedy token at
the dispatch boundary. Same-m reruns are bitwise deterministic.
`EXL3_HGEMM_F16OUT=1` additionally rounds each fp32-output reconstruct GEMM to
fp16 once (the residual stream's own precision; measured worst model-level KLD
1.9e-3).


## Known issues

- CDNA (MI-series, wave64) is not supported and fails fast at import.
- No WMMA path: mma.m16n8k16 is emulated via shuffles. A hardware WMMA
  prototype measured slower and had correctness issues on M>=3 shapes; it is
  not included.
- Transient wrong triton prefill output has been observed once during
  full-suite test runs with a co-resident inference server on the same GPU;
  never reproduced in isolation. If a run returns all-zero prefill rows,
  re-run the command.
