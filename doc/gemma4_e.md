# Gemma 4 E4B on Jetson Orin

This fork adds E-series text inference and conversion to the ordinary upstream
EXL3 interfaces: `Config`, `Model`, `Cache`, `Tokenizer`, `Generator`, `convert.py`
and the existing examples. It implements per-layer embeddings and residual
inputs, shared-KV tail attention, and checkpointed calibration side inputs.
E4B text inference is validated. E2B configuration is accepted but untested;
E-series image/audio inference and tensor parallelism are not validated here.

The public target head retains the full vocabulary. Classroom transports,
decision heads, custom assistant adapters and reduced-vocabulary draft heads are
outside this public branch. Upstream features for other supported architectures
remain available as before.

## Tested Orin environment

Orin Nano 8 GB Super (`cortex1`), Jetson Linux R39.2.1, Ubuntu 24.04, Python
3.12.3, CUDA toolkit 13.2.86, GCC 13.3, PyTorch 2.13.0+cu130 and Triton 3.7.1.
The older `cortex0` is an Orin NX; its historical results are separate.

Build the extension for **SM 8.7**. Install a suitable ARM64 CUDA-enabled PyTorch
first, then the fork's dependencies in a virtual environment. The tested generic
PyTorch wheel warns that Orin is unsupported; synthetic CUDA tests and actual
E4B cached/full inference were checked explicitly. This is a tested combination,
not a guarantee that any ARM64 PyTorch wheel supports Jetson.

```sh
git clone https://github.com/cortexist/exllamav3.git
cd exllamav3
git checkout main
python3 -m venv .venv
. .venv/bin/activate
# Install a CUDA-enabled ARM64 PyTorch compatible with your Jetson first.
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.7
export MAX_JOBS=1
python -m pip install --no-build-isolation .
```

On memory-constrained Orin, one compilation worker avoids large concurrent C++
compiler processes. The initial extension build can take considerable time.
Use your CUDA toolkit's supported host compiler. Do not install a desktop x86
prebuilt extension on Jetson. The upstream release/PyPI wheels do not contain
this fork's E4B additions; build this source tree.

The tested toolkit installation lacked development headers for cuRAND,
cuSPARSE and cuSOLVER. Supply matching development packages, or make a separate
include directory containing **only those library headers** from the matching
wheel/toolkit and add it to `CPATH`. Do not add a wheel's entire CUDA include
tree ahead of the toolkit: mixing core CUDA headers caused compilation failures.

Before importing EXL3 in the tested PyTorch environment:

```sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export EXL3_EMBED_STREAM=1
export EXL3_INT8_GEMV=1
```

Disabling expandable segments avoids the tested wheel's unsupported NVML fabric
query. Streaming both large embedding tables gathers rows on demand, reducing
resident memory; cold access and storage speed still matter. Mode 1 enables INT8
activation error feedback: default mode 2 showed greater batch/decode numerical
differences for this conversion. Standard upstream block CUDA graphs retain
their defaults. No custom outer graph/session adapter is needed.

## Download or convert

[Published E4B EXL3 weights](https://huggingface.co/cortexist/gemma-4-E4B-it-qat-exl3-4.0bpw)
contain the tokenizer/configuration and a 4-bit body/embedding conversion with a
6-bit **full-vocabulary** output head. The model card records calibration,
backbone identity, numerical checks and quality limitations.

```sh
hf download cortexist/gemma-4-E4B-it-qat-exl3-4.0bpw --local-dir MODEL
python examples/chat.py -m MODEL -cs 4096 -mode gemma4
```

Or convert a floating-point checkpoint with tokenizer and processor configuration:

```sh
python convert.py -i ORIGINAL -o EXL3_MODEL -w WORK \
  -b 4 -hb 6 -eb 4 -vb 16 -cr 32 -cc 512
python convert.py -w WORK -r  # resume an interrupted conversion
```

`-eb` applies to token and per-layer embedding tables. The example retains unused
vision weights without quantizing them; text inference does not load them.
Requantizing a QAT checkpoint changes outputs. The published first conversion
used limited 32×512 calibration and is not certified quality-equivalent to its
original floating-point weights or a QAT GGUF.

## Cache-only prefill

The existing `Model.prefill(input_ids, params)` fills K/V and stops at the final
KV producer. For E4B it skips the 18 shared-KV tail layers that do not contribute
new cache entries. A subsequent normal `Model.forward` processes the last prompt
token through the full model and full-vocabulary output head. `Generator` uses
the standard upstream prefill path; no application-specific prompt handler is
required. Paged caches and the sliding-window masks are retained.

Cache-only tokens/s excludes the final prompt forward and token selection.
Prompt-to-first-token includes that work. Neither is end-to-end speech latency.
See the [fork's published measurements](https://github.com/cortexist/exllamav3/blob/879a7f2b23852a77ccc5483675ee0c9500456ff2/README.md#e4b-on-orin-nano)
for exact boundaries.

## Optional FP16 prefill-weight cache

`LinearEXL3.prepare_reconstruct_cache(original_basis=True)` retains a dense FP16
copy of one projection and returns its byte count. This avoids reconstructing
that weight on each large-row prefill. Small-row/decode dispatch still uses the
packed EXL3 weights. Nothing is cached by default; model placement does not
reserve this additional memory. Allocate after loading and before graph capture.

This reproducible E4B policy selects small projections first within an explicit
budget, considering only modules before the final KV producer. It leaves the
last producer and shared-KV tail uncached. It is a simple policy, not a claim of
optimal allocation for every model or prompt length:

```python
from exllamav3.modules.quant.exl3 import LinearEXL3, MAX_RECONSTRUCT_SLICE_N

# model has already been loaded; run before constructing/using Generator.
budget = 512 * 1024**2  # extra device memory, separate from packed weights and KV
candidates, seen = [], set()
for top in model.modules[:model.last_kv_module_idx]:
    for module in top:
        inner = getattr(module, "inner", None)
        if not isinstance(inner, LinearEXL3) or id(inner) in seen:
            continue
        seen.add(id(inner))
        if (inner.out_features > MAX_RECONSTRUCT_SLICE_N
                or inner.in_features % 128 or inner.out_features % 128):
            continue
        size = 2 * inner.in_features * inner.out_features
        candidates.append((size, inner.key, inner))
used = 0
for size, key, inner in sorted(candidates, key=lambda item: (item[0], item[1])):
    if used + size <= budget:
        used += inner.prepare_reconstruct_cache(original_basis=True)
print(f"Extra FP16 weights: {used / 1024**2:.1f} MiB")
```

Use the same model/cache with the ordinary `Generator` afterwards. Increasing
the budget consumes unified system RAM and may leave less room for other
applications. The [published measurements](https://github.com/cortexist/exllamav3/blob/879a7f2b23852a77ccc5483675ee0c9500456ff2/README.md#e4b-on-orin-nano)
report both uncached and explicitly budgeted results.
Initialization/reconstruction time is outside the warm throughput measurements.

Original-basis reconstruction folds the Hadamard transforms into FP16 weights,
changing rounding relative to the usual rotated-basis path. Check numerical and
task behavior for your workload; greedy sequences need not always be identical.
`original_basis=False` caches rotated weights and retains per-forward Hadamards.
The cache does not change the checkpoint or vocabulary. Sliced wide projections
are rejected. `clear_reconstruct_cache()` releases a projection's extra weights;
unloading also clears them. Do not clear or replace buffers while a captured
CUDA graph that references them can still be replayed.

## Validation

```sh
python -m pytest tests/test_gemma4_e.py tests/test_gelu_precision.py -q
```

The E4B tests require CUDA and Transformers with Gemma 4 support (tested with
5.14.1). They compare synthetic full/chunked and cache-only inference to
Transformers across a sliding-window boundary, verify preservation of conversion
side inputs, and check GELU precision over finite FP16 inputs. Real-model tests
also check plain generation and single-versus-batched `Generator` results.

The GELU change replaces two unsuffixed literals with FP32 literals, avoiding
accidental FP64 instructions on Orin while retaining the tested FP32 accuracy.
It is the only native kernel change in this public patch set.
