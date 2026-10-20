# 2026-10-19/20: upstream v1.6.3 sync validation, ORPO + GRPO box runs, community PRs

Box: 2x RTX 3090, torch 2.14.1+cu130, venv `~/exl3/qlora-venv`. Model for every GPU check:
`/mnt/two/weights/Qwen3.5-9B-exl3-6.00bpw` (GDN hybrid: 8 attention + 24 linear-attention
layers, exercises the merged recurrent-state changes). The 4B love quant is gone from disk.

## 1. Upstream v1.6.3 sync (`origin/claude/serene-heisenberg-ggrn8z`, local `sync/v1.6.3`) -- PASS

The merge was done in a cloud session without a GPU; this was its first build and run.

- Extension JIT-builds (2m42s). Only `README.md` conflicted in the merge; the training
  modules carry clean upstream changes plus two post-merge fixes (KDA gate LoRAs guarded on
  the fused GDN path; realtime cache invalidation across every tier).
- CPU training suite (the v1.5.0 recipe + the five repaired tests): **141 passed, 3 skipped**.
- `qlora_validate_native.py` on the 9B, bf16, `--check-backward`: 100% argmax agreement,
  head-slice PASS, dequant parity PASS (default + pissa), backward smoke PASS.
- Real SFT run, 49 steps (love_accepted, r32 lr 9e-5 seq 512 b2), **same yaml on master as A/B**:

  | branch | best held-out | final ema | peak VRAM | step time |
  |---|---|---|---|---|
  | sync/v1.6.3 | 1.9051 @30 | 1.9525 | 21.63 GB | 1.00 s |
  | master | 1.9173 @30 | 1.9531 | 21.63 GB | 1.00 s |

  Identical resource profile; loss differences are bf16 noise. The adapter loads on the native
  inference path and steers generation (`training/qlora_infer_native.py`).
- One sync-introduced test bug, fixed (in stash, see §5): upstream's new session fixture
  `_isolated_expert_profiles` imports the real package when a GPU is visible, but
  `test_vision_training` installs a stand-in `exllamav3` at collection time, so every test
  errored with `'exllamav3' is not a package` on a GPU box. Guard: skip the import when the
  module in `sys.modules` has no `__path__`.

## 2. ORPO (PR #172, `feat/orpo`) -- PASS, first GPU run

`training/experiments/orpo_qwen35_9b_love_v1.yaml`: love_dpo.jsonl (185 pairs), beta 0.1,
`orpo_nll: full`, r32, lr 2e-5 cosine, seq 1024, 93 steps, 4m44s, peak 10.5 GB.

- Pair accuracy 0.0 -> 1.0 by step ~40, margin negative -> positive, log-odds -1.9 -> +1.0..1.7.
- SFT held-out (love_test_le512): 2.7557 -> 2.0922 @23 (best), 2.1182 final.
- Base vs adapter at inference: the adapter answers in the love register ("I missed you. The
  room felt a little quieter until you returned..."); base stays assistant-flavoured.
- 44 CPU tests (`test_preference`, `test_realtime`) pass. Loss math checked against TRL's
  `odds_ratio_loss`: identical.

## 3. GRPO (PR #173, `feat/grpo`) -- mechanics PASS, one reward bug fixed

- `--self-test` on raw GSM8K: rollouts, rewards, policy pass, gradient probe all ran, but every
  reward was 0 for two reasons: (a) 84% of the 9B's completions hit the 256-token cap before an
  answer, and (b) **bug**: `rewards.exact_answer` compared the extracted completion answer
  against the raw `answer` column, which in GSM8K is the whole worked solution ending in
  `#### N`, so it could never match. Fixed in `exllamav3/training/rewards.py` (a reference
  containing `####` or `\boxed{}` is reduced with `extract_answer` too; bare references stay
  verbatim) + `test_exact_answer_reduces_worked_reference`. 14 GRPO CPU tests pass.
- Real run, `training/experiments/grpo_qwen35_9b_gsm8k_v1.yaml`: GSM8K pre-rendered with a
  brevity system prompt and bare numeric answers (`out/grpo_qwen35_9b_gsm8k_v1/data/`),
  G=8, 4 prompts/step, max_new 384, lr 1e-5, beta 0, dapo, 100 steps, 55 min, peak 13.6 GB,
  ~32 s/step (53% of it generation).
  - Train reward (temp 1.0 samples) EMA 0.63 -> 0.94; truncation 0.16 -> 0.00; completion
    length bounded (100-210 tok); `clip_frac` 0 throughout at mu=1 as expected; kl 0 (beta 0).
  - Held-out reward (greedy, 32 prompts) was 1.0 at baseline and stayed 1.0: the 9B already
    solves GSM8K, so the task is ceiling-limited. The learning signal shows in the sampled
    train reward and in `frac_zero_std` (groups all-correct) rising toward 1.0 at the end.
- Still owed from the handoff box list: `--rollout-sampler exact` comparison, `--beta 0.04`
  kl check, `--num-iterations 2` clip_frac check, and a harder task for a real curve.

## 4. Community PRs (adamo1139 / melatonina)

- **#174 attn_varlen -> flash_attn_varlen_func directly.** Real fix: on current master packed
  bf16 training raises `No matching attention function` (reproduced here) because upstream's
  dispatch rework (017b4408, in master since the v1.5.x sync) dropped the FA2 varlen backend.
  Box-tested the PR's `attn_varlen` against per-document eager attention (GQA 8/2, head_dim
  128, three documents, windows 0/64, softcap 0/30): forward max abs diff <= 0.010 (bf16),
  grad cosines 0.999-1.002 for q/k/v. **Merge.**
- **#175 `--head-vocab-chunk` in materialized logits.** Tiled head reconstruction; new test
  passes with `test_native_llama` / `test_fused_ce` / `test_preference` (39 passed). **Merge.**
- **#176 validate: `--split-even`, `--verbose-load`, `--head-vocab-chunk`.** Flag pass-throughs
  via a `make_net` helper; compiles, help renders. **Merge.**

## 5. Box finding: flash_attn had been broken since 2026-10-03

torch was bumped to 2.14.1+cu130 on 10-03; the installed flash_attn 2.8.3.post1 wheel was a
torch 2.8 build and stopped linking (`undefined symbol: ..c10_cuda_check_implementation..`).
`attn_impl: auto` silently fell back to eager (materialized [t,t] bias) in every run since,
including the user's rp_orion and ornith runs (`attn: ... flash_attn NOT importable` in their
logs). Upstream inference no longer uses flash-attn (Triton kernels, FA2 only behind
`EXL3_PREFER_FA2=1`); the dependency is the trainer's, for an autograd attention backward.

Fixed 2026-10-20: `flash_attn-2.8.3+cu130torch2.14-cp312` from
mjun0812/flash-attention-prebuild-wheels (no official wheel exists for torch 2.14). Verified:
flash vs eager forward/backward parity on synthetic GQA tensors; the validate gate passes
with `--attn-impl flash` (8x flash); the PR #174 varlen test above. Decision: keep FA2 for
the training backward; investigate torch SDPA / a Triton backward later only if speed warrants.

## 6. Uncommitted state at end of session

- `feat/grpo` checkout: `exllamav3/training/rewards.py` + `tests/test_grpo.py` (reward fix),
  `training/qlora_infer_native.py` (encode special tokens; `--template-vars`; was a
  pre-existing bug: the demo fed `<|im_start|>` as plain text and opened a think block).
- `stash@{0}` (on sync/v1.6.3): `tests/conftest.py` stand-in guard + the same infer fix.
- `stash@{1}` (on master): the pre-session WIP (`model_init --lora`, `last_turn_only`, mmlu).
- New yamls: `sync_v163_*`, `orpo_qwen35_9b_love_v1`, `grpo_qwen35_9b_gsm8k_v1`.
- Run outputs under `out/sync_v163/`, `out/orpo_qwen35_9b_love_v1/`, `out/grpo_qwen35_9b_gsm8k_v1/`.

## 7. Integration branch `integrate/v1.6.3-orpo-grpo-pr174-176` (2026-10-20)

master + sync/v1.6.3 + feat/orpo + feat/grpo (+ reward and infer fixes) + PRs #174, #175, #176,
merged in that order with no conflicts, plus the conftest stand-in guard (keyed on a missing
`__spec__`; test_grpo_trainer's stub carries a `__path__`).

- `compileall` clean; the full CPU suite incl. GRPO: **160 passed, 3 skipped**, both with a GPU
  visible and CPU-only.
- validate gate on Qwen3.5-9B with `--attn-impl flash`: PASS (8x flash).
- **Packed training end to end** (`training/experiments/integrate_packed_muse_glimmer.yaml`:
  Muse-Glimmer-30B 6bpw, semancy v5, v12 recipe, 1 epoch, split across both cards):
  52x flash, 798 docs -> 152 bfd blocks of 3072 (98.2% filled), 76 steps at 25 s/step,
  peak 14.5 / 11.5 GB, held-out 2.4816 -> 1.9308 @38 -> **1.8773 @76**. This is the path
  that raised `No matching attention function` on master.
