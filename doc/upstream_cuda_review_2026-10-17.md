# Upstream CUDA performance review — 2026-10-17

The 61 incoming upstream commits contain relevant optimizations for recurrent
models, CPU-offloaded MoE, tensor-parallel DFlash2, and repeated-prompt caching.
**I found no direct speedup to this fork's dense Gemma 4 E4B CUDA prefill or
plain-decode kernels.** No new end-to-end CUDA throughput was measured in this
review. The one numerical result below is an upstream-reported CPU-offload
benchmark, not a measurement of this fork or of Orin.

## Rebase and review scope

| Item | Revision / result |
| --- | --- |
| Public fork branch | `main` |
| Previous fork tip | `7e2b44024d492719f22b5a7285cca7fe94064de7` |
| Previous upstream base | `3a513ac9` |
| Fetched upstream target | `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a` (`upstream/master`, includes v1.6.3) |
| Rebased code tip, before this memo | `291b487b3afc7ec03e85a24d0e2a5958f4b4156a` |
| Incoming range | `3a513ac9..a3bfbfd0`: 61 commits, including merges; net changes in 99 files |
| Fork commits replayed | 8; no conflicts; all eight patches identical under `git range-diff` |
| Recovery branch | `backup/main-before-upstream-rebase-20261017` |

The rebase is local, in the existing `main` worktree at
`/home/zero/Workspaces/.release-worktrees/exllamav3-orin`. Remote `origin/main`
has not been rewritten. The primary checkout remains on `research`; research,
integration, and experiment branches retain their original histories.

This review compares upstream against the old upstream base, so it does not
misattribute the fork's existing GELU FP32 fix or FP16 prefill-weight cache to
the incoming changes. The README's October 11 timing table predates this rebase.

## Changes relevant to performance

### Recurrent CUDA scan and state storage

[0f685aa9](https://github.com/turboderp-org/exllamav3/commit/0f685aa9)
keeps each thread's recurrent-state slice in registers across a scan, reading
it once and writing it once per pass. It changes both generic and 128-wide
scan bodies in `exllamav3/exllamav3_ext/gdn.cu`, and fixes an invalid four-way
split for Mamba replay. This reduces repeated global state traffic, especially
for multi-token scan/replay; it is a credible decode/speculative-verification
optimization, with no measured throughput claim here. The long-prefill chunk
algorithm is not newly optimized by this patch.

[02446529](https://github.com/turboderp-org/exllamav3/commit/02446529)
makes GDN/KDA recurrent-state **storage** FP16 by default, while scans still
compute in FP32. This halves the state tensor's bytes, not all cache or model
memory. Smaller state reads/writes and checkpoints can benefit bandwidth-bound
decode and cache capacity. Chunked prefill converts stored state to FP32;
Mamba2 retains FP32 storage. `Cache(recurrent_dtype=torch.float)` or
`--recurrent_state_fp32` restores FP32. Review long-generation quality and
speculative rewind before treating the precision tradeoff as free. This does
not apply to dense Gemma's attention layers. The earlier opt-in proposal
`b87823e6` is superseded by the final default implementation.

### CPU-offloaded MoE with a CUDA GPU

Several changes target heterogeneous inference, not fully GPU-resident dense
GEMM/GEMV:

- [5aaf6579](https://github.com/turboderp-org/exllamav3/commit/5aaf6579)
  combines split issue/collect bookkeeping, stream waits, and launches into
  native handoff functions, reducing Python/native call overhead per layer.
- [f50a86df](https://github.com/turboderp-org/exllamav3/commit/f50a86df)
  overlaps expert work, reduces worker-pool dispatches to two per token, and
  replaces phase barriers with atomic counters.
  [21cfd60f](https://github.com/turboderp-org/exllamav3/commit/21cfd60f)
  adds split-K CPU GEMV and work stealing. These can reduce GPU waiting when
  CPU experts are the limiting stage.
- [82cebdf1](https://github.com/turboderp-org/exllamav3/commit/82cebdf1)
  promotes recently missed experts into spare GPU slots asynchronously with
  the pinned arena. The final default is four spare slots per layer, and at
  least one GPU-resident expert is required. Persistent hot/cold rankings
  (`e4a928f9`, corrected in `345eca1c` and `f35b51cc`) improve initial placement
  across reloads. Gains depend on routing locality, transfer cost, and memory
  headroom. `EXL3_MOE_CPU_PROMOTE=0` selects the periodic-sweep alternative.
- [c8680a7b](https://github.com/turboderp-org/exllamav3/commit/c8680a7b)
  adds explicit `cudaMemcpyAsync` for registered buffers PyTorch does not
  recognize as pinned; the final promotion/demotion code uses it. It is not a
  general faster-copy switch for dense prefill.

**Upstream-reported measurement:** on a Threadripper 7960X with LFM2.5-8B
split decode (`-mcs 16 -mcpa`), the opt-in `EXL3_MOE_CPU_SPREAD=1 -mct 12`
configuration achieved about **240 tok/s**, versus **232 tok/s** with the
same 12 workers in default core order: about **3.4%**. The comparison against
222–225 tok/s with 23 default workers also changes thread count and does not
isolate affinity. Full offload was approximately unchanged. The source does
not identify the GPU or provide a complete reproducibility record alongside
these numbers. This is evidence for a particular CPU-offload configuration,
not an Orin or general CUDA speedup.
[Source: pinned upstream environment documentation](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0/doc/env_vars.md#exl3_moe_cpu_spread-default-0).

### Tensor-parallel speculative decoding and grouped routing

[9cc3bd15](https://github.com/turboderp-org/exllamav3/commit/9cc3bd15)
enables DFlash2 with tensor-parallel targets.
[e98785b9](https://github.com/turboderp-org/exllamav3/commit/e98785b9)
then selects top-K candidates on each LM-head shard before gathering scores
and vocabulary IDs, replacing full-logit gathers in that path. This reduces
communication volume and can improve speculative decode on multiple GPUs;
acceptance rate and local top-K cost still determine the net gain. It does
not accelerate single-GPU plain greedy decoding.

[ec72130d](https://github.com/turboderp-org/exllamav3/commit/ec72130d)
adds a grouped DS3 router used by Ling 3.x: group selection, expert selection,
and weights run in one CUDA kernel, with a specialized FP32 GEMV for one row.
This is a plausible routing-overhead reduction in prefill and decode versus
the retained torch-composed reference. Ling is newly supported in this range,
so this is not evidence of a measured speedup for an already supported model.
The same patch deliberately bypasses the specialized BC-MLA path when a head
gate is needed for correctness; architecture support is not uniformly faster.

### Persistent prefix caching and sparse-MLA memory

[bea81f48](https://github.com/turboderp-org/exllamav3/commit/bea81f48)
and [2caf2775](https://github.com/turboderp-org/exllamav3/commit/2caf2775)
add and refine a persistent disk tier behind the CPU page cache, including
recurrent checkpoints and canonical tensor-parallel layouts. It is disabled
unless `disk_cache_dir` is set, and requires `cpu_cache_size > 0`. Reusing
prefixes across process restarts can avoid prefill work and lower time to first
token, including for ordinary attention models. It does **not** increase the
kernel rate of uncached prefill or steady-state decode. Measure SSD/CPU/GPU
transfer cost and hit rate separately from newly computed prompt tokens.

[716533ae](https://github.com/turboderp-org/exllamav3/commit/716533ae)
releases otherwise retained query storage before sparse MLA attention when
the RoPE width is zero. This reduces peak live memory and may permit larger
prefill chunks or avoid allocation pressure; a direct tok/s gain is not
established. It does not change dense Gemma attention.

## Changes that should not be counted as new CUDA speedups

- `6b3d0fe8` fixes a ROCm/gfx1100 MLA compilation issue. CUDA's absorb signature
  is unchanged; the nearby comment about vectorized loads is not a new CUDA
  benchmark.
- `ef55f38b` fixes sparse top-K workspace allocation before graph capture.
  It repairs a regression rather than introducing a faster attention algorithm.
- `7b3e6898` changes top-P sampling semantics, including the threshold-crossing
  token and handling ties. Output changes in sampled generation cannot be
  treated as speed evidence. It can affect dense workloads using that sampler.
- FP8 scale-loading fixes, higher-quality conversion modes, ROCm quantizer
  shared-memory changes, Windows/build fixes, and TP startup/ISA compatibility
  work do not demonstrate higher steady-state CUDA inference throughput.
- Existing quantized GEMM/GEMV inference implementations and the fork's optional
  FP16 prefill cache are not retuned in the incoming range. The new changes
  under the upstream `quant/` CUDA directory concern conversion.

## Validation and recommended follow-up

Completed: clean starting worktree; backup ref; conflict-free rebase;
`git range-diff` reports `=` for all eight fork patches; `git diff --check`
passes; all **528 tracked Python files compile syntactically**. The code tip
contains upstream entirely (`main...upstream/master`: eight ahead, zero behind,
before adding this memo).

Runtime validation is **blocked by the local environment**, not reported as
passing. PyTorch is `2.13.0+cu130`, `torch.cuda.is_available()` is false,
`nvidia-smi` cannot communicate with a driver, and no precompiled
`exllamav3_ext` is installed. The targeted pytest invocation below stopped
during collection with 19 errors and one skip: the first import could not
infer a CUDA build architecture, and later imports lacked the extension.
No CUDA build, kernel correctness, model parity, or throughput was validated.

```sh
python -m pytest tests/generator tests/model_setup tests/cache \
  tests/test_gemma4_e.py tests/test_gelu_precision.py -q
git range-diff 3a513ac9..backup/main-before-upstream-rebase-20261017 \
  a3bfbfd0..291b487b
git log --reverse --oneline 3a513ac9..a3bfbfd0
```

For further review, first rebuild the extension on the target CUDA machine and
run Gemma parity, GELU, quantized-linear/prefill-cache, and generator tests.
Compare the old fork tip against `291b487b` with the same checkpoint, input
IDs, CUDA/PyTorch versions, clocks, cache budget, and graph settings. Repeat
the documented 400/930-token prompts and 128 greedy decode steps, discard
warmup, and report prefill, first-token latency, decode, and peak memory
separately. Keep prefix reuse and speculation disabled for that baseline.

Prioritize additional benchmarks only for workloads actually used: FP16/FP32
GDN state with long-generation parity; CPU-MoE placement/handoff/thread count;
multi-GPU DFlash2 with acceptance statistics; or restarted-process prefix-cache
hits. **For dense Gemma E4B on Orin, this review supplies no evidence that the
61 upstream commits improve raw CUDA prefill or plain-decode rate.**
