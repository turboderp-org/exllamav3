# Upstream v1.4.8 sync

Merged `turboderp-org/exllamav3` master at
`6ff3a17` (v1.4.8), 29 commits since the previously merged
`ca13bdd` (v1.4.7) ancestor, into fork master at `cf97b22`.

What upstream brought in:

- Packaging moved to `pyproject.toml` (`MANIFEST.in` removed, `setup.py` reduced to the
  CUDA-extension build; version stays in `exllamav3/version.py`). The fork's `setup.py` and
  `MANIFEST.in` were identical to upstream's, and the fork ships no extra package data, so the
  upstream files were taken as-is. `requirements.txt` is unchanged and the README quick start
  still applies.
- flash-attn-2 backend removed; padded Triton attention for non-power-of-two head dims.
  The training path imports `flash_attn.flash_attn_func` directly (it needs the autograd
  backward), so it is unaffected.
- Quantized-cache support for DSA, DSA-on-MLA and QSA; VRAM accounting utility
  (`exllamav3/util/memory.py`); expandable segments on by default.
- Scratch allocations in `BlockSparseMLP` routing and `GatedResidual` routed through the
  bucketed `g_tensor_cache`.
- Generator fixes: non-causal spans for Gemma4 and multimodal bounds during rewind prefill;
  no requeue after `max_new_tokens` without a window.

Integration decisions:

- `exllamav3/modules/mla_attn.py`: upstream dropped the "sparse DSA over a quantized cache"
  per-step decline from the graph-path comment (now supported). Kept the fork's runtime-LoRA
  guard (`has_runtime_lora(...)` on the projections and DSA indexer weights) on the
  graph-path condition and merged the comment accordingly.
- `exllamav3/modules/block_sparse_mlp.py`: upstream's `_routing_buffers` hoist is orthogonal
  to the fork's runtime-LoRA guards on the fused expert paths; auto-merge kept both. The
  differentiable training routing reimplements the routing math and does not touch these
  shared inference scratch buffers.
- `exllamav3/modules/attn.py`: upstream's QSA quantized-cache changes merged identically;
  no LoRA interaction.
- `.gitignore`: kept the fork's training-output rules and added upstream's `uv.lock`.
- `README.md`: kept the fork's README; bumped both upstream-version references to v1.4.8.

Validation (CPU, Python 3.11, PyTorch 2.14.0):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_qlora_grad.py tests/test_native_llama.py tests/test_lora_init.py \
  tests/test_fused_ce.py tests/test_gdn.py tests/test_shortconv.py \
  tests/test_vision_training.py tests/test_preference.py \
  tests/test_quant_aware.py tests/test_lora_fused_path.py \
  -k 'not has_runtime_lora_semantics and not real_exl3_layer'
python -m compileall -q exllamav3 training examples tests
git diff --cached --check
```

Result: 94 passed, 1 skipped, 2 deselected. The fork's CPU tests stub the package, so
upstream's new CPU test `tests/test_multimodal_rewind_prefill.py` (which imports the real
package and therefore the native extension) could not be collected without a CUDA build;
real-package import and GPU/model tests were excluded or skipped as in the v1.4.7 sync.

CUDA was unavailable. Native extension compilation, upstream CUDA tests
(`test_triton_paged_hdpad.py`, `test_dsa_kernels.py`, `test_mla_dsa.py`), and real-model
training/inference parity were not validated. Rebuild the native extension before GPU use;
a binary from before this sync lacks the new `dsv4_pool_quant` kernels and bindings.
