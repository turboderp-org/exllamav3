# QLoRA on EXL3 quants vs bitsandbytes NF4: Qwen3.5-4B, same recipe, 3/4/6 bpw

Date: 2026-10-11. Box: 2x RTX 3090. Repo: exl3-qlora (fork of exllamav3 with a
native training path on the EXL3 trellis format).

## What was compared

Five runs of the same fine-tune, changing only the frozen base format:

- exl3-qlora on `Qwen3.5-4B-exl3` at 3.00, 4.00 and 6.00 bpw
- axolotl 0.20.0 (transformers 5.17, peft 0.21, bitsandbytes 0.50.2) with
  `load_in_4bit` NF4 on the BF16 `Qwen/Qwen3.5-4B`

Everything else is matched (exl3 runs on branch `fix/lora-a-init`, which is
master plus the two fixes described below):

| | |
|---|---|
| data | 98 "love" SFT rows, 24 held-out rows (same files, same order of magnitude: 100 to 360 tokens) |
| template | the model's own `chat_template.jinja`, `enable_thinking: false` (empty `<think></think>` pre-closed), assistant turns supervised, prompt masked |
| LoRA | r = alpha = 32, rsLoRA, dropout 0, fp32 adapters, bf16 compute |
| targets | q/k/v/o, gate/up/down, plus the Gated DeltaNet `in_proj_qkv` / `in_proj_z` / `out_proj` on the 24 linear-attention layers. 60,948,480 trainable params on both sides, verified from the saved adapters |
| optimizer | AdamW (torch), lr 1e-5, cosine, no warmup, weight decay 0, clip 1.0 |
| batch | 8 sequences per step, no grad accumulation, no packing, seq 512 |
| length | 1 epoch = 13 optimizer steps, eval every 2 steps |
| eval | mean of per-example mean CE on the 24 held-out rows (axolotl: `eval_batch_size: 1`, which makes HF's `eval_loss` the same reduction the exl3 trainer prints) |

## Result (the table to quote)

Fixed LoRA-A init, lr 1e-4, the 17 held-out rows both tools score, branch
`fix/lora-a-init`:

| step | 0 | 2 | 4 | 6 | 8 | 10 | 12 | drop |
|---|---|---|---|---|---|---|---|---|
| exl3 6.00bpw | 2.873 | 2.370 | 2.207 | 2.128 | 2.118 | 2.128 | 2.128 | 0.746 |
| exl3 4.00bpw | 2.849 | 2.415 | 2.227 | 2.157 | 2.150 | 2.154 | 2.152 | 0.696 |
| exl3 3.00bpw | 2.805 | 2.381 | 2.214 | 2.160 | 2.159 | 2.166 | 2.165 | 0.641 |
| bnb NF4 (axolotl) | 2.880 | 2.310 | 2.215 | 2.207 | 2.205 | 2.187 | 2.179 | 0.701 |

Finish order 6bpw < 4bpw < 3bpw < NF4; NF4 drops fastest in the first two
steps and plateaus highest. 13 steps at this lr: 44 to 46 s on exl3 at every
bitrate, 81 s on axolotl including 7 checkpoint saves.

Everything below is the path to that table: the earlier passes, the init bug
they exposed, and what was ruled out.

## Held-out loss

First pass, before the init fix described below (exl3 effectively at ~9x the
learning rate):

| step | 0 | 2 | 4 | 6 | 8 | 10 | 12 | drop |
|---|---|---|---|---|---|---|---|---|
| exl3 3.00bpw | 2.787 | 2.419 | 2.289 | 2.244 | 2.239 | 2.240 | 2.239 | 0.548 |
| exl3 4.00bpw | 2.812 | 2.436 | 2.279 | 2.223 | 2.211 | 2.207 | 2.204 | 0.608 |
| exl3 6.00bpw | 2.829 | 2.397 | 2.263 | 2.201 | 2.185 | 2.184 | 2.183 | 0.646 |
| bnb NF4 (axolotl) | 2.880 | 2.671 | 2.541 | 2.475 | 2.439 | 2.419 | 2.415 | 0.465 |

That gap was too good. Pre-clip grad norms were 40 to 110 on exl3 and 5 to 10
on axolotl at the same loss, and the cause turned out to be the LoRA-A init:
the exl3 trainer stores A as `[in_features, r]` and handed it to torch's
kaiming-uniform init, which reads fan_in from dim 1 and so used r = 32 instead
of in = 2560. A came out sqrt(in/r) = 8.9x larger than PEFT's. With Adam, B
moves about lr per step regardless of gradient scale, so the effective update
B@A per step was ~9x PEFT's: exl3 at lr 1e-5 was behaving like PEFT at ~9e-5.
Fixed the same day (`init_lora_a_`, bound 1/sqrt(in_features), all four init
sites). Grad norms after the fix: 5 to 11, same as axolotl.

Second pass, fixed init, otherwise identical:

| step | 0 | 2 | 4 | 6 | 8 | 10 | 12 | drop |
|---|---|---|---|---|---|---|---|---|
| exl3 3.00bpw | 2.787 | 2.670 | 2.574 | 2.522 | 2.488 | 2.471 | 2.465 | 0.322 |
| exl3 4.00bpw | 2.812 | 2.655 | 2.546 | 2.489 | 2.454 | 2.435 | 2.430 | 0.382 |
| exl3 6.00bpw | 2.829 | 2.656 | 2.545 | 2.488 | 2.453 | 2.435 | 2.429 | 0.400 |
| bnb NF4 (axolotl) | 2.880 | 2.671 | 2.541 | 2.475 | 2.439 | 2.419 | 2.415 | 0.465 |

Read plainly: at matched init and lr 1e-5, one epoch of 13 steps, all four
arms land within 0.05 nats of each other, and bnb NF4 ends 0.015 below exl3
4bpw despite starting 0.07 above it. The quantization-quality difference is
visible at step 0 (exl3 3bpw 2.787 vs NF4 2.880) and is mostly gone after
13 steps at this learning rate; the 4bpw and 6bpw exl3 curves are
indistinguishable (0.001 apart). Why NF4 recovers slightly more is not
established; candidates are the different base rounding (a worse base has more
easy loss to recover) and axolotl's Triton LoRA kernels. Either way the
difference is of the order of the run-to-run noise on 24 rows (see caveats).

Third pass, fixed init, lr 9e-5 on both (the lr the first pass was secretly
running at), rsLoRA on both:

| step | 0 | 2 | 4 | 6 | 8 | 10 | 12 | drop |
|---|---|---|---|---|---|---|---|---|
| exl3 4.00bpw | 2.812 | 2.427 | 2.278 | 2.221 | 2.206 | 2.205 | 2.203 | 0.609 |
| bnb NF4 (axolotl) | 2.880 | 2.315 | 2.223 | 2.207 | 2.208 | 2.194 | 2.187 | 0.693 |

The exl3 curve reproduces the first-pass 1e-5 curve to within 0.01 at every
point (2.203 vs 2.204 final), which closes the init question: old init at lr
x = new init at lr ~9x. At this lr the NF4 arm again ends slightly lower
(0.016) and gets there faster (0.565 nats in two steps vs 0.385). Same
adapted modules, same init scale, same optimizer settings, same grad clip,
same schedule. Not explained yet; the first diagnostics would be a single
shared batch through both trainers comparing the step-1 adapter delta, and the
dtype the LoRA delta is accumulated in on each side. Grad norms: exl3 5 to 11,
axolotl 4 to 9.


### Matched eval set (the comparison to read)

axolotl drops any sequence longer than `sequence_len` instead of truncating it,
on the eval set too. Seven of the 24 held-out rows are 602 to 956 tokens, so
every axolotl `eval_loss` above is over the 17 short rows, while the exl3
trainer kept all 24 and truncated the long ones to 512. Same rows on both
sides (the 17 rows <= 512 tokens, `love_test_le512.jsonl`), fixed init:

| step | 0 | 2 | 4 | 6 | 8 | 10 | 12 | drop |
|---|---|---|---|---|---|---|---|---|
| exl3 3.00bpw, lr 1e-5 | 2.805 | 2.672 | 2.560 | 2.500 | 2.462 | 2.442 | 2.435 | 0.370 |
| exl3 4.00bpw, lr 1e-5 | 2.849 | 2.664 | 2.538 | 2.473 | 2.433 | 2.411 | 2.405 | 0.444 |
| exl3 6.00bpw, lr 1e-5 | 2.873 | 2.665 | 2.539 | 2.473 | 2.432 | 2.410 | 2.404 | 0.469 |
| bnb NF4, lr 1e-5 | 2.880 | 2.671 | 2.541 | 2.475 | 2.439 | 2.419 | 2.415 | 0.465 |
| exl3 3.00bpw, lr 9e-5 | 2.805 | 2.387 | 2.224 | 2.161 | 2.162 | 2.173 | 2.172 | 0.633 |
| exl3 4.00bpw, lr 9e-5 | 2.849 | 2.415 | 2.229 | 2.156 | 2.151 | 2.153 | 2.151 | 0.698 |
| exl3 6.00bpw, lr 9e-5 | 2.873 | 2.373 | 2.212 | 2.137 | 2.124 | 2.129 | 2.128 | 0.745 |
| bnb NF4, lr 9e-5 | 2.880 | 2.315 | 2.223 | 2.207 | 2.208 | 2.194 | 2.187 | 0.693 |

So: at lr 9e-5 the final held-out loss orders by bitrate, 6bpw 2.128 <
4bpw 2.151 < 3bpw 2.172 < NF4 2.187, with 6bpw also pulling ahead at every
step from step 2 on. At 1e-5 (too cold for 13 steps) 4bpw, 6bpw and NF4 are
on top of each other and 3bpw trails by 0.03. The step-0 baselines on these
17 rows are NOT in bitrate order (3bpw lowest, 6bpw highest, 0.07 apart);
after training they are. I read the baseline order as 17-row noise and the
post-training order as real, but the two are the same size, so hold it
loosely. The NF4 arm drops fastest in the first two steps at 9e-5 and then
plateaus highest.

Ruled out for the early-step
difference: token masking (identical, verified with `--inspect` and the
prepared dataset: assistant text + `<|im_end|>`, think block and trailing
newline masked), adapted module set and parameter count, A-init scale,
optimizer and schedule order (both step the optimizer then the scheduler),
grad clipping, and axolotl's auto-enabled fused LoRA Triton kernels (a run
with them off reproduces the kernels-on curve to 0.001). What is left is the
base itself: NF4 rounding leaves more structured error that an adapter can
undo in a step or two, which fits both the higher start and the faster first
drop. Hypothesis, not established.

The practical lesson for exl3-qlora users: every learning rate that "worked"
on the fork before 2026-10-11 was ~sqrt(in_features/r) hotter than the same
number in PEFT. With the fix in, expect to raise lr by roughly that factor
(about 9x at r=32 on a 2.5k-wide model, 5.7x at r=128) to recover old
behaviour.

## Speed and memory

Warm kernel cache on both (first-ever step costs 42 s on exl3 and 80 s on
axolotl for Triton compiles, then never again).

| | exl3 (any bpw) | axolotl NF4 |
|---|---|---|
| 13 steps + 7 evals | 55 s | 52 s (+ 7 checkpoint saves) |
| steady optimizer step | 2.15 s | about 1 s |
| peak VRAM | 12.8 GB (torch) | 12.8 GiB allocated, 15.2 GB on nvidia-smi |

axolotl is faster per step at this size, roughly 2x: its auto-enabled Triton
LoRA kernels help, and the exl3 step is 67% backward (dequant-on-the-fly in
both passes). Bitrate does not change exl3 training speed (3/4/6 bpw all 55 s).

## Caveats, stated plainly

- 17 to 24 held-out rows. A repeat of the exl3 4bpw run landed 0.005 nats from the
  first (2.1994 vs 2.2039), so differences under about 0.01 are nothing. The
  baselines are not monotonic in bitrate (3bpw starts lowest); I read that as
  eval-set noise, not a property of the quants.
- One epoch, one dataset, one model size. This is a sanity comparison, not a
  benchmark.
- The first-pass table is kept for the record; the second-pass table is the comparison.
- An earlier axolotl run with the stock `q_proj k_proj v_proj o_proj gate_proj
  up_proj down_proj` target list trained only 42.5M params, because those names
  only exist on the 8 full-attention layers in the HF checkpoint (the GDN
  layers use `in_proj_*` / `out_proj`). It ended at 2.463. If you fine-tune
  Qwen3.5 with PEFT, name the GDN projections explicitly.

## Reproduce

exl3: `training/experiments/love_qwen35_4b_exl3_{3,4,6}bpw_lr1e4_ev17.yaml` (the headline table), `..._fixinit.yaml` (second pass), `love_qwen35_4b_exl3_4bpw_fixinit_lr9e5.yaml` (third pass), `love_qwen35_4b_exl3_{3,4,6}bpw_fixinit_lr{1e5,9e5}_ev17.yaml` (matched 17-row eval, the final table; the
files without `_fixinit` are the first pass, same settings) via
`python training/qlora_train.py --config <yaml>`.
axolotl: `training/experiments/axolotl/love_qwen35_4b_bnb4_v2.yaml` (1e-5) `_v3_lr9e5.yaml` (9e-5) `_v4_lr9e5_nokernels.yaml` and `_v5_lr1e4.yaml` (headline) via
`axolotl train <yaml>` (needs torchvision for the Qwen3.5 processor).
Adapters, run reports and trainer states: `/mnt/two/qlora_out/love_qwen35_4b_*`.
