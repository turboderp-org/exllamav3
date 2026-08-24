# Qwen 3.5 Vision: Multi-image / Video / Varlen — Resolution Record

Scope: the issues enumerated here were found by code inspection while scoping "batched/
multi-image and long-video paths" (future-work item 5 of `vision_cpu_acceleration.md`).
All items are now **implemented and verified** on `torch 2.10.0+cu130` against the
`q3.5-0.8-exl3-q4` checkpoint. Conventions were settled by reading the pinned reference,
transformers **5.15.1** (`vision_utils.py`, `models/qwen3_vl/modeling_qwen3_vl.py`) — the
exact version this repo targets. A golden-value cross-check against HF weights is deferred
(§6); everything short of it is verified.

---

## 1. Multi-image input was effectively broken — FIXED

Old behavior: the list path in `get_image_embeddings` did
`torch.cat(image_tensor, dim = 0)` (requires equal token counts per image) and built
`grid_thw`/`inv_freq`/every `MMEmbedding` grid from the **last** image only. Since
`qwen2_smart_resize` preserves aspect ratio, real photo sets essentially never matched;
multi-image worked only by coincidence of identical resized dimensions.

New behavior (`exllamav3/architecture/qwen3_vl.py`):

- All images are preprocessed separately and packed into ONE `(1, T_total, D)` sequence
  (`_embed_packed`): tokens concatenated along the sequence dim, one attention segment per
  image via `cu_seqlens`/`max_seqlen`, per-grid vision RoPE tables concatenated along dim 0,
  `grid_thw` as an `(N, 3)` tensor. `Qwen3VLPosEmbedding.fast_pos_embed_interpolate` already
  supported multiple grids; the patch merger is grid-independent.
- The tower forwards once over the pack; merger output and every deepstack entry are split
  by per-image merged lengths (`T_i // merge²`) into one `MMEmbedding` per image with its
  own grid, metadata, and token string. Public signature/semantics unchanged; empty list
  raises `ValueError("No images provided")`.

Verified: `tests/test_cpu_vision.py::test_qwen35_vision_multi_image_packing` (packed output
matches standalone per-image runs within atol 1e-3, deepstack slices included);
comparison-tool `--images` mode passes GPU-vs-CPU gates on three different-size images
(rel_l2 8.3e-3, cosine 0.99996); end-to-end generation on real weights describes two
different images correctly (`mm_lengths [500, 1110]`, grids `(1,50,40)` / `(1,74,60)`).

## 2. Varlen attention had no CPU backend — FIXED

`fn_torch_sdpa_fallback_nocache` (`exllamav3/modules/attention_fn/torch.py`) no longer
declines varlen: it iterates `cu_seqlens` intervals, applies SDPA per segment (each segment
starts at position 0, so `is_causal` is correct per segment), and concatenates along the
sequence dim. The fp32 upcast for CPU fp16 happens once before the loop. Requests with
`cu_seqlens` but batch > 1 are still declined (consistent with all other varlen backends).
This covers any future CPU varlen caller and CUDA hosts without flash-attn/triton.

The qwen3-VL vision `Attention` is now constructed with `use_cu_seqlens = True`
(`architecture/qwen3_vl.py`), matching qwen2.5-VL. Callers that pass no `cu_seqlens`
(single-image path, load-time warmup) keep today's dense path via the `attn.py` guard.

Verified: `test_sdpa_varlen_parity` (segmented execution == per-segment dense SDPA),
full pytest suite green (26 tests).

## 3. Video — RESOLVED (both open questions answered, path wired)

Two questions existed: whether the reference adds a temporal term to vision RoPE, and what
attention spans video uses. Both are settled by transformers 5.15.1 source:

- **Vision RoPE has NO temporal term**: `vision_utils.get_vision_position_ids` defaults
  `include_temporal=False` → h/w-only ids repeated per frame. exllamav3's
  `qwen2_position_embedding_grid_2d` already matches exactly. No change was needed or made.
- **Attention is PER FRAME, not per clip**: `get_vision_cu_seqlens` defaults
  `merge_temporal=False` → `t` segments of `h*w`. The earlier claim in this doc ("full
  attention over the whole clip", O(T²) in total tokens) was wrong; the true cost is
  O(t·(h·w)²). §5's bench below demonstrates the difference empirically (a t=32 clip at
  2000 tokens/frame costs 29 s on CPU segmented vs 221 s if forced through one segment).

Wiring: new public API

```python
Qwen3VLVisionModel.get_video_embeddings(tokenizer, frames: list[Image], text_alias=None) -> MMEmbedding
```

It preprocesses the frame list (existing video branch: truncates to a multiple of
`temporal_patch_size`, tiles single frames, asserts equal frame sizes), builds per-frame
`cu_seqlens`, and returns one `MMEmbedding` whose `grid_thw = (t, h, w)` feeds the existing
text-side mrope — `gen_mrope_pos_ids` already computes the temporal coordinate for
t > 1 grids, so no extension changes were needed.

Verified: comparison tool `--video_frames 8` passes GPU-vs-CPU gates (rel_l2 6.0e-3,
cosine 0.99998); `test_qwen35_vision_video_frames`; end-to-end generation from an
8-frame clip produces a coherent description.

## 4. Test coverage — CLOSED

- `tests/test_cpu_vision.py`: added `test_sdpa_varlen_parity`,
  `test_sdpa_fallback_dense_path_unchanged`, `test_qwen35_vision_multi_image_packing`,
  `test_qwen35_vision_video_frames` (26 tests total, all green).
- `tests/compare_vision_gpu_cpu.py`: new modes `--images f1 f2 ...` (packed varlen),
  `--video_frames K` (per-frame segments), `--bench` (timing sweep). Single-image default
  behavior unchanged (same gates as before the changes: rel_l2 4.76e-3, cosine 0.99999).
- Smoke regression unchanged and passing:
  `tests/models/smoke_qwen3_5_arch_.py --vision_cpu --full_load --multimodal_image` →
  " kitten", no transient VRAM.

## 5. Measured T scaling (CPU, fp32-upcast SDPA; RTX 3090 host, torch 2.10.0)

`python tests/compare_vision_gpu_cpu.py --model_dir <dir> --bench` (synthetic grids
`(1,50,40)` = 2000 tokens each; wall time of one forward after warmup):

| mode   | shape            | tokens | segments | cpu ms   | gpu ms |
|--------|------------------|-------:|---------:|---------:|-------:|
| images | 1 x (1,50,40)    |   2000 |        1 |    725   |   17   |
| images | 4 x (1,50,40)    |   8000 |        4 |   3081   |   50   |
| images | 16 x (1,50,40)   |  32000 |       16 |  14414   |  198   |
| video  | t=8 x (1,50,40)  |  16000 |        8 |   6439   |   96   |
| video  | t=32 x (1,50,40) |  64000 |       32 |  28968   |  367   |

With per-segment length bounded (~2000), CPU cost scales ~linearly in total tokens
(×16 tokens → ×20 time). For contrast, forcing the t=32 clip through a SINGLE segment
(the incorrect "whole clip" model) measured **221 s CPU / 3058 ms GPU** — 7.6×/8.3× worse
than the segmented path at identical token count. Attention remains a modest share; the
CPU budget is dominated by GEMMs, so any future CPU optimization should target matmul, not
attention. Re-measure with `--bench` before acting on this.

## 6. Deferred: HF golden-value cross-check

Per decision on 2026-08-25 the original Qwen3.5-VL weights have NOT been downloaded yet;
verification so far is internal-consistency only (packed vs standalone equality, GPU-vs-CPU
gates, end-to-end generation). Conventions were verified by reading transformers 5.15.1
source (RoPE spatial-only; per-frame segments), which removes most cross-validation risk,
but a numeric check against the reference tower is still worthwhile once weights are
available. Procedure, ready to execute:

1. Download the original Qwen3.5-VL checkpoint (HF format; ~2 GB for the 0.8B) into the
   workspace. The installed `transformers==5.15.1` already matches the version this work
   was verified against — do not upgrade casually.
2. For each case — (a) single image, (b) two+ different-size images, (c) a short video —
   build the pixel tensor ONCE with exllamav3's `preprocess`, then feed the identical
   tensor + `grid_thw` to both towers.
   - Reference entry point: `AutoModel.from_config(config.vision_config)` /
     `visual(pixel_values, grid_thw=...)` (see `models/qwen3_5/modeling_qwen3_5.py`,
     vision forward around lines 1002–1093). Deepstack outputs arrive as
     `BaseModelOutputWithDeepstackFeatures`.
   - Gates: cosine > 0.999 and rel_l2 < 0.02 on the final merged output (same thresholds
     as the GPU-vs-CPU tool; reuse `tensor_metrics` from `tests/compare_vision_gpu_cpu.py`).
3. A natural home for the script is `tests/compare_vision_hf.py`, mirroring the existing
   comparison tool's structure (preprocess once, two loads, tap/gate table).
