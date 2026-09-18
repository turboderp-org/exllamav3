# DFlash2 speculative drafting

DFlash2 (`DFlash2DraftModel` checkpoints) is a diffusion-style draft model:
a fixed block of mask tokens is denoised in one forward pass and a
candidate selector walks one token path per position, which the target then
verifies. It plugs into the existing DFlash v1 draft path — same cache
layout, same `update_kv_from_target` machinery, same rollback.

## Running

Point `-dm` at the draft checkpoint; it is autodetected by its arch string,
no new flags:

```
python examples/chat.py -m <target> -dm <dflash2-draft> -dds -dc 0.4
```

- `-dm / --draft_model_dir`: draft checkpoint directory.
- `-ndt / --num_draft_tokens`: draft window ceiling (default: the
  checkpoint's `block_size - 1`).
- `-dds / --dynamic_draft`: truncate the drafted block per round using the
  selector's transition scores.
- `-dc / --draft_confidence`: acceptance-probability target, default 0.4.
  Measured at 8k on Qwen3.8-27B: full-block (no `-dds`) beats `-dds -dc 0.4`
  on both tau (2.29 vs 1.73) and tok/s — verify positions are cheap, so crop
  less before reaching for kernels.

Omit `-dm` to run the target alone; in-checkpoint MTP (`--mtp`) is
unaffected and cannot be combined with `-dm`.

## Behavior notes

- Greedy walk by default (token-match verify, as v1). The walk runs sampled
  with q-aware rejection sampling only when every active job uses the same
  plain temperature / top-k / top-p sampler — lossless w.r.t. the quantized
  target model. Default presets with min-p or penalties use the greedy walk;
  for q-aware verify use a plain sampler (`--min_p 0`, no rep penalty).
- Unsupported, fails loudly at setup: tensor-parallel targets, CFG /
  multi-sequence jobs, recurrent targets without enough `max_history`, and
  draft caches smaller than the draft block.
- Checkpoints carrying `output_multiplier` / `final_logit_softcapping` /
  `input_embedding_scale` get them applied (logits before the selector walk;
  embedding scale on mask columns only, anchor unscaled).
- Long-context prefill at 128k runs the GPU at low SM utilization; chunk
  size changes the utilization picture but paired TTFT shows no wall-time
  effect at 64k, and a pinned-staging A/B at 128k moved TTFT 62.6s → 62.2s
  (no effect). No prefill finding is claimed here.

## Validating

`eval/eval_dflash2_acceptance.py` runs seeded prompts and reports
accepted/rejected, tau (accepted per verify-forward), and tok/s; exit 2 on
the all-zero tripwires. See its header for the cell matrix and pass bars.

## Converting

Backbone Linears quantize normally. Conv base kernels, selector codebooks,
and conv/selector projections stay raw fp16 (no quantization role; the
selector's `retain_raw_fp16` caps flag carries them through convert).
Uncalibrated convert is wired; ship published draft weights until a
calibrated convert is measured.
