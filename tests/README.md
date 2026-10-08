# exllamav3 test suite

Everything here runs under pytest from the repository root. Tests exercise the source tree, not an installed
copy, and need no hard-coded paths or devices: models come from a registry, devices from options.

```sh
pytest tests                                   # everything this machine can run (missing models/GPUs skip)
pytest tests/kernels                           # one subsystem by directory
pytest tests -m attention                      # one topic across kernels/, modules/, graph/, e2e/
pytest tests -m "not model"                    # only self-contained tests (no checkpoints)
pytest tests -n 4 --devices 0,2,3,4            # pytest-xdist, one GPU per worker
pytest tests/e2e --model-root /mnt/models      # end-to-end feature matrix over the canonical models
```

## Layout

The tree follows the structure of the library, from the leaves up. Lower levels test functions in isolation
against references; higher levels compose them.

| Directory     | Contents | Needs |
|---------------|----------|-------|
| `kernels/`    | One file per extension function or Triton kernel family, against a torch/NumPy reference or an exact contract (`norm`, `activation`, `gemm`, `quant`, `routing`, `sampling`, `attention`, `dsa`, `recurrent`, `moe`, `cache`, `hyperconnections`, `strings`, `portability`) | GPU |
| `modules/`    | One directory per module type, built from synthetic weights (`testlib.checkpoint`, `testlib.exl3`), against an explicit reference forward or another path of the same module | GPU |
| `graph/`      | CUDA-graph (BC_*) paths against the eager path they replace | GPU, some need models |
| `cache/`      | Cache layers, recurrent state, CPU page tier | |
| `generator/`  | Job, scheduling, sampler stack, n-gram drafting, streaming text | mostly CPU stubs |
| `tokenizer/`, `loader/`, `model_setup/` | Tokenization, checkpoint loading, config parsing, warmup | |
| `moe_cpu/`    | CPU expert offload: ISA tiers, arena, host affinity, thread pool, streaming | CPU, some GPU |
| `tp/`         | Tensor-parallel export/import, collectives plumbing, worker lifecycle | |
| `conversion/` | Quantizer and conversion pipeline pieces | GPU |
| `e2e/`        | Feature tests over the canonical model set (generation, CPU offload modes, TP, drafting, prompt caching) | models |
| `parity/`     | Per-architecture parity against HF transformers or the reference implementation | models, transformers |
| `examples/`   | Helpers of the example scripts | |
| `tools/`      | Benchmarks, probes and diagnostics (not collected) | |
| `testlib/`    | Shared helpers: environment queries, model registry, synthetic weights, references, tolerances | |

Every directory name is also a pytest marker, so `-m moe` selects `kernels/moe`, `modules/moe` and anything a
file marks with `pytestmark = pytest.mark.moe`.

## Writing a test

- **Device**: take the `device` fixture (or `devices` for multi-GPU tests); never write `"cuda:N"`. The test
  device is made current before each test, so Triton launches land on it.
- **Models**: name a role from `tests/models.yaml` with `@pytest.mark.model("dense")` and take `model_dir`, or
  run over every model with some tags with `@pytest.mark.models("moe")` and take `model_id`. Never hard-code a
  path. Missing models skip.
- **No module-level work**: building models, allocating GPU memory or calling `torch.cuda.set_device` at import
  breaks collection for every other test. Use fixtures (module- or session-scoped for expensive setup).
- **Requirements are markers**, not early `return`s (which report a pass): `nogpu`, `multi_gpu(n)`,
  `cc(major, minor)`, `cuda_only`, `rocm_only`, `hf`, `cpu_flags("avx512bw")`, `platform("linux")`, `slow`.
  Tests that need no GPU must say `nogpu`; everything else is skipped on machines without one.
- **References** belong in `testlib` when more than one file needs them (`testlib.exl3` for synthetic EXL3
  weights, dequantization and MLP activations; `testlib.compare` for tolerances and logit metrics).
- **Synthetic checkpoints** go through the real loader: `testlib.checkpoint.module_config(tensors, tmp_path)`.
- **Subprocesses** inherit the source tree on `PYTHONPATH` (set by conftest).

## Models

`tests/models.yaml` defines the canonical roles (`dense`, `moe`, `recurrent`, `mla`, `swa`, ...), each a small
model covering one feature family, with tags. Paths are relative to `--model-root` / `$EXL3_TEST_MODEL_ROOT`.
Per-machine overrides go in `tests/models.local.yaml` (git-ignored) or `$EXL3_TEST_MODELS`:

```yaml
root: /mnt/models
models:
  moe:
    path: /elsewhere/qwen3-30b-a3b-exl3
```

## End-to-end tests

`e2e/` tests load real models through `model_init` (`testlib.e2e.load_model`), so a configuration is the
command-line arguments a user would pass: `load_model(model_dir, device, "-mcs", "8")` is split expert offload,
`"-cq", "8"` an 8-bit cache, `"-mtp"` / `"-dm", draft_dir` / `"-ngram", "2"` drafting, `"-tp", "-gs", gpu_split(devices)`
tensor parallelism. Each test file is one feature, run over the models whose tags make it applicable, against a
reference configuration of the same model:

| File | Contract | Matrix |
|------|----------|--------|
| `test_decode_consistency.py` | cached prefill + decode (paged cache, recurrent state, graphs) == cache-less forward, within the model's own noise floor | every model × fp16 / 8-bit cache |
| `test_moe_offload.py` | `--moe_cpu_offload` / `--moe_cpu_split` == all-GPU load (prefill, decode, batched decode) | models tagged `moe` + `mul1` × placement |
| `test_speculative.py` | greedy output with MTP / DFlash / n-gram drafting == plain greedy (divergence only at near-ties), drafts accepted | `DRAFT_CASES` |
| `test_prompt_cache.py` | a shared prefix reuses cached pages (and recurrent checkpoints), and restores from the CPU tier after eviction, continuing as a fresh prefill would | every model |
| `test_tensor_parallel.py` | `-tp` over two devices == single-device load (forward and cached decode) | every model with TP support |

Comparisons between two paths of one model go through `testlib.e2e.assert_logits_agree`: median and mean KL
under an absolute bound or a multiple of the model's own noise floor (`noise_floor`: two forward passes that
differ only in length), and top-1 agreement at confident positions (the reference leads by more than a logit).
MoE models flip expert choices on routing near-ties whenever the arithmetic changes, and any text has near-tie
positions that two correct paths may resolve differently; a broken path moves these statistics by orders of
magnitude. Greedy runs are compared with `assert_greedy_equivalent` (identical up to a divergence at a near-tie).

Model matrices (`@pytest.mark.models(...)`) leave out roles tagged `draft` or `hf_reference` unless a tag names
them. To add a case to a feature, add a registry model with the right tags, or a row to the file's case table.
