# Upstream v1.4.9 sync

Merged `turboderp-org/exllamav3` master at
`be57335` (v1.4.9 + 1 commit: torch 2.12/2.13 build actions), 41 commits since the
previously merged `6ff3a17` (v1.4.8) ancestor, into fork master at `72623aa`.

What upstream brought in:

- MGEMM sliced mode: all attention projections (Q/K/V and a full or interleaved gate) run as
  one launch of equal-width column slices (`SlicedMultiLinear`, `project_qkv_sliced`) for
  decode-sized inputs (`bsz * q_len <= 32`); takes precedence over the pairwise Q/G and K/V
  bundles. Same for the GDN qkv/z pair (`project_qkvz_sliced`). Toggle: `EXL3_QKV_SLICE=0`.
- flash-linear-attention functions vendored into `exllamav3/vendor/fla/` (drops the fla and
  indirect transformers dependencies; ~2.6 s faster import); lazy xformers / SDPA imports.
- New architectures: Spark2.5 (`Spark2.5ForCausalLM`), LFM2 (`Lfm2ForCausalLM`),
  GLM-4.7-Flash (`glm4_moe_lite`), DeepseekV4 vision tower; `model_diff` handles 3D
  residuals with hyperconnections.
- DSA-on-MLA staged cache quantization; loader measures per-layer transients explicitly.
- CPU MoE: AVX-512BW kernel tier (no VNNI required), swizzle on by default on all AVX512
  tiers, background hugepage migration, configurable worker start timeout.
- Misc: `/dev/shm` size check before allocating shared buffers, `chunk_size` as a standard
  model_init arg, Spark chat template, pinned-vision MLP handle rebuild, qbench error message.

Integration decisions:

- `exllamav3/modules/attn.py`, `sliding_attn.py`: upstream's new sliced-bundle early return in
  `project_qkv` landed on top of the fork's runtime-LoRA comment. The sliced launch has no
  LoRA epilogue (it reads trellis storage directly, like the pairwise bundles), so the
  dispatch is gated on `not has_runtime_lora(q_proj, k_proj, v_proj, g_proj)`: with an
  adapter loaded, decode takes the fork's pairwise Q/G + K/V mgemm branches, which add the
  low-rank delta onto the mgemm output (same fused speed as before this sync). Without an
  adapter the sliced path runs as upstream intends. Adding a delta-on-top epilogue to the
  sliced path itself is a possible follow-up, not done here.
- `exllamav3/modules/gated_delta_net.py`: auto-merged, but the same guard was needed on the
  new `multi_qkvz` sliced dispatch of the torch (non-graph) path; the split-graph path was
  already guarded.
- `tests/test_lora_fused_path.py`: new source tripwires for the three sliced dispatch guards;
  the sliding-attention padded-mgemm fixture now binds the real `finish_qkv` (upstream
  refactored `project_qkv` to end in it) and carries a `multi_qkv` bundle without a
  `project_qkv_sliced`, so a dropped guard fails the test instead of silently passing.
- `README.md`: kept the fork's README; bumped both upstream-version references to v1.4.9.
- Everything else auto-merged (`mla_attn.py`, `mlp.py`, `block_sparse_mlp.py`, `mamba2.py`,
  `doc/env_vars.md`, ...). The fork's graph-path guards (`bc_attn`/`bc_swa`, BC MLP, GDN
  split graph, MLA, Mamba2, fused MoE experts) are intact; `bc_attn.py` now also reads
  `module.multi_qkv`, which is only reached when the graph path is taken, i.e. never while
  a runtime LoRA is loaded.

Validation (CPU, Python 3.12.3, PyTorch 2.8.0+cu128):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_qlora_grad.py tests/test_native_llama.py tests/test_lora_init.py \
  tests/test_fused_ce.py tests/test_gdn.py tests/test_shortconv.py \
  tests/test_vision_training.py tests/test_preference.py \
  tests/test_quant_aware.py tests/test_lora_fused_path.py \
  -k 'not has_runtime_lora_semantics and not real_exl3_layer'
python -m compileall -q exllamav3 training examples tests
git diff --cached --check   # only upstream's vendored fla files report EOF blank lines
```

Result: 97 passed, 1 skipped, 2 deselected. Negative check: removing the sliding_attn guard
fails the new tripwire and all six padded-mgemm fixture cases.

Not validated on GPU: the native extension was not rebuilt and no real-model decode/training
parity was run. The cached `exllamav3_ext` build predates this sync (new sliced-mgemm entry
points, DSA staged-quant kernels); the next import will JIT-rebuild it. A LoRA decode parity
check on a real EXL3 quant (`tests/test_lora_fused_path.py::test_mgemm_lora_delta_parity_gpu`)
is the remaining box item for the sliced-path guard.
