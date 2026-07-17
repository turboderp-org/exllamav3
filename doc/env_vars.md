# Environment variables

Runtime and build-time toggles recognized by ExLlamaV3. All of these have sensible defaults;
they exist mainly for A/B testing, debugging and working around platform quirks.

Boolean-ish variables treat `0` as off and any other value as on unless noted. C++-side
variables are read once (on first use) and cached; Python-side variables are read at import
time. Either way, set them before loading a model.

## Attention

### `EXL3_BC_ATTN` (default: `1`)

Graph-captured C++ decode attention. For decode steps (bsz ≤ 4, q_len ≤ 16) the whole attention
block — q/k/v projections, fused head norm + RoPE, cache append, flash-decoding attention and
o_proj — runs as a single C++ call, captured as one CUDA graph per (bsz, q_len) shape and
replayed with only the input/output/position/block-table pointers patched. Removes effectively
all Python host time from the attention block; the largest gains are on host-bound setups
(small or hybrid models, fast GPUs, contended CPUs).

Module or cache configurations the path does not support (TP, headwise gates, LayerNorm or
span-heads head norms, non-EXL3 projections, compander-enabled quant cache, ...) fall back to
the regular dispatch path by design. Unexpected errors while building the path are raised, not
swallowed. Set to `0` to disable the path entirely.

### `EXL3_BC_ATTN_TRACE` (default: `0`)

Print one line per attention module/cache-layer pair when the graph-captured decode path is
built or declined (module key, device). Activation check for A/B tests: a benchmark comparing
`EXL3_BC_ATTN` settings is only meaningful if the enabled run actually built the path.

### `EXL3_QC_ATTN` (default: `1`)

Quantized-cache direct attention: packed K/V cache tensors feed the attention kernels directly,
with new K/V quantized into the cache before attention and dequantization fused into the kernel
loads. Set to `0` to fall back to the legacy path (dequantize the cache into full-size fp16
temporaries, then attend), which costs both time and the peak VRAM for the temporaries. Only
affects quantized caches.

### `EXL3_PREFER_FA2` (default: `0`)

Put the flash-attn-2 backends ahead of the built-in Triton attention kernels in the dispatch
order. flash-attn is an optional dependency; when it is not installed, this switch is ignored
(with a warning) and the built-in kernels serve everything. The Triton kernels match or beat
FA2 across supported hardware and cover more cases (quantized caches, head dims > 256,
attention sinks); this switch exists for A/B comparison.

## EXL3 GEMM / GEMV

### `EXL3_GEMV` (default: `1`)

QTIP-style small-m fp16 GEMV path, dispatched from the main GEMM entry point when the shape
heuristic applies. `0` disables, `1` uses the measured heuristic envelope (default), `2` forces
the path wherever its hard constraints allow (testing).

### `EXL3_GEMV_SMEM` (default: `-1`)

Weight-extraction strategy inside the fp16 GEMV kernel: `-1` picks per bitrate (default), `0`
forces shuffle extraction, `1` forces shared-memory staging. Testing only.

### `EXL3_INT8_GEMV` (default: `2`)

Fused int8-activation GEMV for tensors quantized with the mul1 codebook: one cooperative launch
covering the input Hadamard, activation quantization, dp4a GEMV and output Hadamard. `2`
(default) is the plain int8 mode, `1` the error-feedback residual mode (~15–16 bit effective
activation precision, slightly slower), `0` disables the path.

Tensors quantized with other codebooks are unaffected and keep their regular kernels. When the
mode is enabled, gate/up (and other same-input) tensor pairs that the int8 path can take are
also *unfused* from the batched MGEMM when each matrix is wide enough to fill the GPU on its
own — see the two thresholds below. The graphed decode paths (BC modules) handle both the fused
and unfused configurations.

### `EXL3_MGEMM_K_THRESHOLD` (default: `6`), `EXL3_MGEMM_N_THRESHOLD` (default: `8192`)

Unfusing heuristics applied when the int8 GEMV mode is enabled, to mul1 tensor pairs only: keep
the fused MGEMM when the bitrate K is at or above the K threshold (the int8 path declines those
anyway), or when the matrices are narrower than the N threshold (too narrow for separate GEMV
calls to fill the GPU; batching is what restores utilization there).

### `EXLLAMAV3_TUNE_CACHE` (default: platform cache dir)

Override the path of the on-disk autotune cache for the cooperative GEMM kernels (kernel shape
selection results, persisted across runs).

## Sampling

### `EXL3_FUSED_SAMPLER` (default: `1`)

Collapse eligible sampler stacks into fused kernels at sampler construction. Stacks ending in
greedy or temperature/min-P/top-K/top-P/Gumbel steps (in the orders emitted by the preset
samplers, optionally preceded by repetition/presence/frequency penalties) run as a few custom
kernels working directly in logit space, instead of the step-by-step softmax/sort pipeline.
Collapsed temperature/min-P stacks sample the same token as the uncollapsed reference for the
same seed, up to float rounding at exact ties; top-K/top-P stacks keep the same token set as
the sort-based reference (ties at the exact cutoff are all kept) but draw their Gumbel noise by
token id rather than sorted position, so individual seeds map to different samples from the
same distribution. Stacks the collapse does not recognize fall back to the step-by-step path by
design. Set to `0` to disable collapsing entirely, e.g. for A/B validation against the
reference implementation.

## Multi-GPU

### `EXLLAMA_NO_P2P_COPY` (default: unset)

When set, device-to-device tensor moves in the layer split bounce through host memory instead
of using peer-to-peer copies. Workaround for platforms with broken or misreported P2P support.

### `EXLLAMA_MASTER_ADDR` (default: `127.0.0.1`), `EXLLAMA_MASTER_PORT` (default: auto)

Rendezvous address and port for the tensor-parallel backend. The port defaults to a free port
picked at startup.

### `EXL3_TP_NO_FWD_BARRIER` (default: `1`)

Skip the pass-start barrier in tensor-parallel forward passes. The native collectives are each
ordered by their own stage counters, so the barrier is not required for correctness; skipping it
saves one spin-kernel launch per rank per pass. Set to `0` to restore the barrier (one aligned
sync point per pass at the cost of a small amount of GPU spin time).

### `EXL3_TP_NO_FP16_WIRE` (default: `0`)

The native backend's CPU-assisted all-reduce moves fp16 payloads over an fp16 wire when the CPU
supports F16C (universal on AVX2-era hardware, probed at runtime): exactly-rounded results for
two ranks, fp16-level rounding beyond, at the same PCIe traffic as the bf16 wire. fp32 payloads
always use the bf16 wire (fp16 lacks the range for residual-stream outliers). Set to `1` to
force the bf16 wire for fp16 payloads too, e.g. for A/B comparison.

### `EXL3_TP_TRACE_WIRE` (default: `0`)

Print a line (once per process) when the fp16 all-reduce wire first activates. Activation check
for numerics A/B tests: whether the wire engages depends on the model's residual dtype, so a
comparison is only meaningful if the fp16-wire run actually used it.

### `EXL3_TP_REDUCE_THREADS` (default: number of participating ranks)

Number of threads slicing each large-payload accumulate in the native backend's CPU-reduce
helper (persistent workers, spin-parked between jobs; AVX-512 path only). The default of one
thread per participating rank covers the cases where a single thread's ~31 GB/s wire rate falls
behind: three or more ranks (multiple adds per chunk) and PCIe 5.0 links. Set to `1` to force
the single-threaded accumulate. Decode-size reduces are always single-threaded.

### `EXL3_TP_SPIN_RECV` (default: `0`)

Milliseconds each tensor-parallel child worker hot-polls its command pipe after finishing a
command before falling back to a blocking receive. A blocking receive pays scheduler wake
latency (tens to hundreds of microseconds, worse with deep C-states) at the start of every
forward pass; during decode the next command arrives within a few milliseconds, so a short spin
window (e.g. `4`) catches it with no wake cost, at the price of one busy core per rank for the
window. `0` disables the spin. Mostly useful on hosts where TP profiling shows a large stagger
between the main process and child workers reaching their first kernel launch.

## Debug

### `EXLLAMA_DEBUGLOG_<CATEGORY>` (default: unset)

Enables timestamped debug logging for the given category when the corresponding variable is
present in the environment. Categories are defined at the call sites (see
`exllamav3/util/debug.py`); mostly hooks for development.

## Build (JIT extension)

These only matter when the C++/CUDA extension is compiled at import time rather than installed
prebuilt.

### `CUDAHOSTCXX` (default: unset)

Host compiler passed to nvcc (`-ccbin`), for systems whose default compiler is too new for the
installed CUDA toolkit.

### `TORCH_CUDA_ARCH_LIST` (default: auto)

Standard PyTorch variable; overrides the compute architectures the extension is built for. When
unset, ExLlamaV3 derives the list from the GPUs present in the system.
