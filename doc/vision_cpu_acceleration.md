# Qwen 3.5 Vision Encoder: CPU Execution Path and Acceleration

Engineering record for the opt-in native CPU execution path for the Qwen 3.5 vision
encoder in exllamav3, its verification, and its subsequent acceleration on x86.
Companion to `qwen35-vision-cpu-offload-plan.md`, `VisionArchitecture.md`,
`OffloadMechanisms.md`, `VerificationPatterns.md`.

- Date: 2026-08-24
- Host: AMD Ryzen 7 9700X (Zen 5, 8C/16T), dual-channel DDR5-7200 (~115 GB/s theoretical)
- ISA available: AVX-512 F/VL/BF16/VNNI (verified via `/proc/cpuinfo`); **no AMX** (Intel-only)
- Software: torch 2.10.0+cu130, Python 3.10, workspace venv (`~/Documents/workspace/ML/.venv`)
- Checkpoint: `q3.5-0.8-exl3-q4` — 0.8B, EXL3 4bpw text, **BF16 vision** (depth 12,
  hidden 768, 12 heads × 64, MLP intermediate 3072, out_hidden 1024, no deepstack)
- Workload shape: single image → ~2000 patch tokens (50×40 grid), one forward per image

---

## 1. Starting point: the basic CPU path

Before any acceleration work, the CPU path existed and was correct but slow:

- `Model.from_config(config, component="vision")` + `vision_model.load(device="cpu")`;
  documented in `Model.load_gen`; `examples/multimodal.py` gained `vision_device = None | "cpu"`.
- Primitives: `cpu_matmul` (+ `cpu_matmul_chunk`) in `exllamav3/util/tensor.py`; CPU branches in
  `modules/conv.py` (flat 3D conv-as-GEMM), `modules/quant/fp16.py` (`LinearFP16`), RoPE with
  external `inv_freq`, attention dispatch falling back to torch SDPA on CPU.
- EXL3 on CPU: `reconstruct_had_slice_cpu` in `exllamav3_ext/cpu/reconstruct.{h,cpp}`
  (parallel + SIMD reconstruction into a dense half buffer per call),
  wired through `LinearEXL3.reconstruct_cpu_matmul`.
- Config fix: `image_token_id` read in both VL base configs (required by
  `Tokenizer.hf_chat_template(embeddings=...)`).
- Tests: `tests/test_cpu_vision.py` (22 synthetic tests incl. CPU-vs-CUDA EXL3 reconstruct),
  `tests/models/smoke_qwen3_5_arch_.py --vision_cpu --full_load`.

**Baseline cost: ~210 seconds per image** (210,229 ms wall, 8–16 threads). Functional, unusable.

## 2. Verification before speed

A behavioral check ("model says kitten") cannot distinguish precision noise from a wrong
kernel, so we built `tests/compare_vision_gpu_cpu.py`:

- Preprocess **once**; feed bit-identical pixel tensor (`1×2000×1536` fp16), `grid_thw`,
  `inv_freq` to two instances of the same component (one CUDA, one CPU).
- Snapshot every module output by wrapping `forward` (exllamav3 modules are not
  `torch.nn.Module` — hook APIs do not exist).
- Per-tap metrics: max/mean abs diff, relative L2, max relative error over significant
  elements, cosine. PASS/FAIL thresholds on the final merger output
  (cosine > 0.999, rel_l2 < 0.02).

### The capture-aliasing trap (cost us one false alarm)

`TransformerBlock.forward` updates the residual **in place** (`x += y`) and returns the *same*
tensor object. On CPU, `t.detach().float().cpu()` is a **no-op alias** when the tensor is
already float32-on-CPU, so every naive "snapshot" recorded a live view of the one shared
residual buffer. First run showed blocks 0–10 mutually orthogonal while the final merger
matched — impossible, and a useful signature: all taps identical to each other ⇒ capture
artifact, not model divergence. Fix: force a real copy with
`t.detach().to(device="cpu", dtype=torch.float32, copy=True)`.

### Baseline agreement (fp16 era)

| tap | rel_l2 | cosine |
|---|---|---|
| patch_embed.proj | 9.4e-7 | 1.0000000 |
| blocks 0→11 | 5.0e-4 → 4.9e-3 (monotone) | ≥ 1.0000008 |
| final merger | 5.07e-3 | 1.0000008 |

Monotone growth = accumulation-order rounding between cuBLAS/cuDNN kernels and CPU math.
No jumps, no structural divergence. Large `max_abs` outliers at deep blocks sit on
large-magnitude elements (fp16 ULP at ~16k is 8.0); `mean_abs` stays ~e-2.

## 3. Profiling: where the 210 s went

Method: module-level timers installed by wrapping each module's `forward`
(perf_counter accumulation), plus one `torch.profiler` pass ranked by self CPU time.
Script: `~/Documents/workspace/ML/stuff/profile_vision_cpu.py`.

Result (before any fix):

| op | share | notes |
|---|---|---|
| `aten::mm` | **99.5%** (208.9 s, 73 calls, avg 2.86 s) | all linears + conv-as-GEMM |
| `_scaled_dot_product_flash_attention_for_cpu` | 0.2% (447 ms, 12 calls) | already fast |
| everything else (norms, adds, casts, rope) | < 0.4% | |

Root cause: **torch CPU has no vectorized fp16 GEMM kernel.** Our linears ran fp16×fp16
matmuls on a scalar fallback path (~3 GFLOP/s). The same chip runs fp32 SGEMM at
~0.9 TFLOP/s and bf16 GEMM at ~3.3 TFLOP/s (oneDNN → AVX-512 BF16 `dpbf16`).

### Dtype microbenchmark at real shapes

`~/Documents/workspace/ML/stuff/bench_mm_dtypes.py`; M×K×N, time per matmul:

| shape | fp16 | bf16 | fp32 |
|---|---|---|---|
| 2000×768×768 (qkv/proj) | 664 ms · 3.6 GF/s | 1.15 ms · 2049 GF/s | 4.56 ms · 517 GF/s |
| 2000×768×3072 (mlp_up) | 4847 ms · 1.9 GF/s | 2.85 ms · 3311 GF/s | 15.2 ms · 619 GF/s |
| 2000×3072×768 (mlp_dn) | 3768 ms · 2.5 GF/s | 3.05 ms · 3091 GF/s | 13.8 ms · 686 GF/s |
| 2000×1536×768 (conv) | 1359 ms · 3.5 GF/s | 1.51 ms · 3129 GF/s | 9.91 ms · 476 GF/s |
| 500×4096×1024 (merger) | 5570 ms · 0.8 GF/s | 5.96 ms · 704 GF/s | 8.79 ms · 477 GF/s |

At 8 threads (vs 16) bf16/fp32 got **faster** (up to 3.6 TF/s bf16, ~0.9 TF/s fp32);
fp16 stayed catastrophic. Conclusion: SMT helps nothing here — torch's default of
8 intra-op threads (= physical cores) is already right; do not raise it.

Attention dtype probe (SDPA, 1×12 heads×2000×64, bidirectional):
fp16 35.6 ms, **bf16 17.2 ms, fp32 17.6 ms** — fp16 again the pathological choice;
bf16 and fp32 identical.

Bandwidth sanity check: dual-channel DDR5-7200 ≈ 85 GB/s practical. Even streaming
fp32-resident weights every pass would be tens of ms against a 733 ms pass; the
workload is compute-bound, which is why GEMM kernel quality dominates everything.

## 4. What was changed

Four files, all changes keyed on runtime state (device/dtype of actual tensors), never on
construction-time properties:

1. **`exllamav3/util/tensor.py` — `cpu_matmul` fast paths.**
   If the resident weight is fp32 or bf16, cast activations to the weight dtype, run plain
   `torch.matmul` (oneDNN dispatches SGEMM / AVX512-BF16 GEMM with fp32 accumulation), then
   cast to the requested output dtype. fp16 matmul is now unreachable from this primitive.
   Mixed cases not covered fall through to the pre-existing chunked fp32-accumulate path.

2. **`exllamav3/modules/quant/fp16.py` — `LinearFP16.forward`.**
   On first CPU execution, the resident weight is converted once fp16 → **fp32**
   (subsequent passes skip the branch). Comment documents why conversion happens lazily
   (see §5.3). `get_tensors()` exports fp32 residents back as fp16 so checkpoints written
   from CPU-loaded instances keep the on-disk format (lossless for fp16-origin values).

3. **`exllamav3/modules/conv.py` — flat 3D branch.**
   Same lazy fp16 → fp32 conversion for the conv weight on first CPU forward; dtype assert
   widened to accept half/bf16/fp32 residents.

4. **`exllamav3/modules/attention_fn/torch.py` — SDPA fallback.**
   On CPU, fp16 q/k/v are upcast to **fp32** for `scaled_dot_product_attention` and the
   output cast back to the caller's dtype. Contained inside `fn_torch_sdpa_fallback_nocache`;
   no ripple into rope/norm logic.

5. **`exllamav3/modules/quant/exl3.py` — `reconstruct_cpu_matmul`.**
   The per-call reconstructed dense weight feeds `cpu_matmul`'s fp32 GEMM fast path. Since the
   native fp32 writer landed in the CPU reconstructor (§7.2), the buffer is produced directly
   in fp32; the interim per-call full-weight cast pass is gone.

### Why fp32 residency and not bf16

bf16 residency was implemented first and is measurably faster (424 ms vs 733 ms per pass),
but it failed the fidelity gates:

- GPU-vs-CPU final merger drifted to rel_l2 6.09e-2 / cosine 0.99816 (thresholds 0.02 / 0.999).
  Deviation still grew smoothly block-by-block — rounding-envelope failure, not structural —
  and end-to-end generation still produced " kitten". But bf16 weight rounding (8-bit
  mantissa) accumulates visibly across 12 blocks and some cancellation-sensitive elements
  diverged up to 6× relative.
- Synthetic suite tripped two cross-precision thresholds (LinearFP16 CPU-vs-CUDA 0.0034 vs
  3e-3 limit; Conv flat 0.0036 vs 3e-3).

fp32 is a strict superset of fp16, so CPU outputs return to the fp16-era envelope *without
touching any acceptance threshold*. Cost: 1.7× slower than bf16, 2× weight RAM
(~+1 GB for this tower). Correctness-first picks fp32; the bf16 numbers are kept here as the
documented fallback if RAM-constrained users prefer speed with a known envelope.

## 5. Things tried that did not help (or actively hurt)

1. **fp16 anything on CPU.** Not slow-ish — three orders of magnitude off. Both matmul
   (~3 GF/s) and SDPA (2× slower than fp32). If a CPU path touches fp16 compute in torch,
   that is the bug.
2. **16 SMT threads for GEMM.** Slower than 8 physical cores across every shape/dtype
   (AVX-512 units contended). Leave `torch.set_num_threads` at default; never set 16 here.
3. **Construction-time weight conversion (reverted).** First implementation converted
   weights in `__init__` when `weight.device == "cpu"` and in `Conv.load(device)` when
   `device.type == "cpu"`. This corrupted **CUDA** instances: deferred/lazy tensor loads
   mean construction-time device is not reliable evidence of execution device. Result was
   spectacular — GPU `patch_embed` diverged at rel_l2 0.365 (hgemm reading bf16 bits as
   fp16), NaN cascade through all blocks, smoke test emitting `'\n\n\n...'`. Lesson:
   convert only on **first CPU forward**, where `x.is_cpu` is ground truth.

## 6. Net effect

| metric | baseline | after |
|---|---|---|
| vision forward, CPU, per image (wall) | 210,229 ms | **733 ms (~287×)** |
| dominant op | `aten::mm` fp16 scalar | SGEMM ~25% of a 750 ms pass, healthy profile |
| GPU-vs-CPU final merger | rel_l2 5.07e-3, cos 1.0000008 | **rel_l2 4.76e-3, cos 1.0000023 (PASS)** |
| `tests/test_cpu_vision.py` | 22/22 | **22/22, original thresholds** |
| smoke `--vision_cpu --full_load` | " kitten", VRAM clean | **" kitten", VRAM clean** |
| weight RAM (CPU residency) | 2 B/param | 4 B/param (≈ +1 GB for this tower) |

Reproduce:

```sh
cd ~/Documents/workspace/ML/exllama/exllamav3
PATH="$HOME/Documents/workspace/ML/.venv/bin:$PATH" \
  python tests/compare_vision_gpu_cpu.py --model_dir ../q3.5-0.8-exl3-q4
python -m pytest tests/test_cpu_vision.py -q
python tests/models/smoke_qwen3_5_arch_.py --model_dir ../q3.5-0.8-exl3-q4 \
  --vision_cpu --full_load --multimodal_image examples/media/cat.png
```

Profiling helpers: `~/Documents/workspace/ML/stuff/{profile_vision_cpu.py,bench_mm_dtypes.py}`.

## 7. Future work, ordered by expected value

1. **Vision-quantized checkpoint (optional now, no longer blocking).** The `LinearEXL3` CPU
   kernel is validated on real tensors: every EXL3 linear in `q3.5-0.8-exl3-q4`'s text model
   (all 24 layers, ~100 projections) reconstructs on CPU within worst-case rel_l2 9.8e-4 of
   the CUDA kernel — real codebooks, real Hadamard scalings, real K=4 distribution.
   `tests/bench_exl3_cpu_reconstruct.py` now takes `--component` (default `vision`) and
   reports real-tensor reconstruct throughput (~0.6–0.7 packed GB/s at 8 threads). What a
   `--vision_bits 1..8` conversion would still add is coverage of EXL3 *inside the vision
   forward* (2000-token activation shapes, full-tower GPU-vs-CPU comparison through
   quantized vision weights) — only relevant if quantized vision towers become something
   we ship.
2. **Native fp32 (or direct-bf16) writer in `reconstruct_had_slice_cpu`:** RESOLVED. The CPU
   reconstructor accepts fp32 output tensors and stores tiles directly in fp32 (fp16 outputs
   still supported); `reconstruct_cpu_matmul` allocates the dense weight in fp32 and passes it
   straight to `cpu_matmul`, removing the extra full-weight cast pass per linear per call.
   Direct-bf16 output remains unbuilt: bf16 residency failed the §4 fidelity gates, so revisit
   it under item 7 if ever needed.
3. **W8A8 int8 GEMM with AVX512-VNNI** (the core of how ggml gets its integer throughput):
   quantize activations once per layer to int8 with per-block scales, reuse across all
   output rows, accumulate i32, rescale in fp32 (`vpdpbusd`). Estimated to land well under
   300 ms/pass and cut weight RAM 2–4×. Requires a custom extension and a quality evaluation
   (per-block-scale int8 for attention inputs needs care).
4. **Register-blocked bf16 tinyBLAS extension** (llamafile `mnpack` style) if bf16-quality
   concerns are ever resolved (e.g. bf16 weights + fp32 activation tiles, or per-layer
   sensitivity selection). ~2× over fp32 SGEMM on paper; only worth it after 3.
5. **Batched/multi-image and long-video paths:** RESOLVED in two rounds. Correctness/wiring
   (packed varlen multi-image, `get_video_embeddings`, CPU varlen SDPA backend) is recorded in
   `doc/vision_known_issues.md`. The throughput round (§7 there) added an op-attribution
   harness (`compare_vision_gpu_cpu.py --profile`), an equal-length batched varlen SDPA fast
   path (kept: wall-neutral here, semantics pinned by tests), and opt-in bounded-RAM chunked
   packing (`EXL3_VISION_PACK_CHUNK`). Measured attribution at 32k–64k tokens: fp32 GEMM
   ~48–54% of CPU wall, attention 22–25%, casts ~10% — further large gains require revisiting
   items 3/7 (int8/bf16 weight residency), not attention tiling; split-KV-style parallelism is
   moot with per-segment length bounded at ~2000.
6. **Load-time hygiene:** mmap-backed safetensors reads (page cache, MAP_POPULATE-style
   prefetch) to cut cold-start latency; one preallocated workspace across ops is already
   effectively provided by torch's allocator here.
7. **Optional knob:** expose weight residency dtype (fp32 default / bf16 fast) per component,
   with the §5 envelope documented, for RAM-constrained machines.

## 8. References (what we took from llama.cpp)

Repo clone surveyed: `~/Documents/workspace/ML/exllama/llama.cpp`. We copied no code; the
transfers were conceptual:

- never run fp16 compute on CPU (ggml converts F16→F32 lanes via F16C everywhere);
- use the widest dot-product instruction the chip offers (AVX512-BF16 / VNNI on Zen5,
  AMX exists but Intel-only);
- activations are transformed once per layer and shared across all weight rows;
- layout work belongs at load time, hot loops stay load+FMA;
- flash attention on CPU is solved with online softmax + small tiles + fp32 accumulators
  (torch's `_scaled_dot_product_flash_attention_for_cpu` already qualifies at our sizes);
- thread pools persist across ops with dynamic chunk stealing (we inherit equivalent
  behavior from oneDNN/torch instead of owning it).
