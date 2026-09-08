# EXL3 QLoRA training scripts

User-facing entry points for the QLoRA-on-EXL3 training path. The importable
library code behind them lives in the `exllamav3.training` package
(`exllamav3/training/`); this directory holds the runnable scripts, in the same
way `examples/` holds upstream's inference examples. `examples/` itself is kept
byte-identical to upstream — everything fork-specific lives here.

Quick start (full version in the repo [README](../README.md)):

```bash
# 1. prove the differentiable forward is correct for YOUR model (run this first)
python training/qlora_validate_native.py --model /path/to/exl3-model --compute-dtype bfloat16

# 2. edit the config, then train
python training/qlora_train.py --config training/qlora_train_config.yaml

# 3. before/after comparison on the native inference path
python training/qlora_infer_native.py --model /path/to/exl3-model --adapter out/my_adapter
```

## Files

- `qlora_train.py` + `qlora_train_config.yaml` — the YAML launcher (single
  command entry point) and its fully-commented reference config. `method:`
  picks the objective — `sft` (next-token CE, default), `ebft`, or a preference
  objective `dpo` / `kto` / `simpo` — and `parallel:` picks the single-GPU,
  layer-split, or DDP backend (`ebft` and the preference methods are
  single/split only). A paired SFT-vs-EBFT A/B is the same config with only
  `method` (and `out`/`run_name`) changed; see `semancer_llama1b_{sft,ebft}.yaml`
  in the repo root for a worked pair. The preference methods need a
  preference-shaped dataset, so they get their own ready-to-run template,
  `qlora_train_pref_config.yaml` (switch `method` between `dpo`/`kto`/`simpo`).
- `qlora_train_native.py` — the single-GPU / layer-split SFT trainer (plain
  PyTorch, no transformers). Also home to the shared data/tokenization helpers
  the other trainers import.
- `chat_turns.py` — multi-turn chat rendering with exact loss-mask
  segmentation, shared by the SFT and preference trainers. OpenAI-style
  `messages` conversations (SFT `--messages-key`, DPO/KTO conversational
  prompt columns) render turn-by-turn with ONLY assistant turns supervised —
  user/system/tool turns and assistant headers are masked to -100 by default,
  with exact boundaries because each segment is tokenized separately (no
  Unsloth-style template-marker search). Multi-turn data needs an explicit
  `--prompt-format` (`auto`'s per-arch default prompt is single-turn only —
  such rows fail fast instead of being silently truncated, which is what the
  old `extract_single_turn` path did). CPU-tested in
  `tests/test_chat_turns.py`.
- `vision_data.py` — `--vision`: image+text rows. Flattens content-parts
  messages (Axolotl/OpenAI layout: `{type: image, path|url|base64|image}`,
  or bare image parts drawn from the row's `--images-key` column) into
  sentinel-carrying strings so every renderer above works unchanged, then
  splices the architecture's own image token layout in at the sentinels and
  attaches the per-image bookkeeping (feature keys, 3-D mRoPE positions) the
  differentiable forward needs. The frozen vision tower and the splice math
  live in `exllamav3/training/vision.py`. CPU-tested in
  `tests/test_vision_training.py`; the box gate is
  `qlora_validate_native.py --image`.
- `chat_jinja.py` — `--prompt-format jinja`: the same segment contract, but
  rendered through the model directory's own Jinja chat template
  (`chat_template.jinja` / `chat_template.json` / `tokenizer_config.json`;
  `--chat-template-file` overrides). Segments come from incremental prefix
  rendering (`messages[:i]` diffs, always `add_generation_prompt=False` for
  the conversation itself), with the assistant header/prefill split off via
  the template's own `add_generation_prompt=True` rendering — so everything
  the model would generate (reasoning span, content, tool-call block, turn
  close) is supervised and everything it would be prompted with is masked.
  This unlocks what the hardcoded formats reject: `tool` roles,
  `tool_calls` + `reasoning_content` message keys, a per-row `tools` column,
  and template variables — `--template-vars '{"enable_thinking": false}'`
  globally, plus per-row `template_vars` / `chat_template_kwargs` columns
  (synonyms; the tabby and llama-server names for the same bag). So a
  dataset row can be `{messages, tools, template_vars}`. An assistant turn
  with `tool_calls` is supervised through its turn-close token, which is
  exactly what triggers the tool-call finish reason at inference — the
  template takes care of it. Caveats: the template must be prefix-monotonic
  (templates that strip `reasoning_content` from history render fine as long
  as reasoning appears only on the final assistant turn — the shape
  OAI-style APIs return; rows that violate this are skipped and counted).
  Multimodal content-parts lists pass through the template (placeholders and
  all) but training is text-only — no pixel features are attached; such rows
  are counted and reported. CPU-tested in `tests/test_chat_jinja.py`.
- `qlora_train_native_ddp.py` — the multi-GPU DDP variant (run under
  `torchrun`).
- `qlora_train_pref.py` — DPO / KTO / SimPO preference training on the native
  path. DPO/KTO use the adapter-disabled base as the frozen reference (no
  second model copy); SimPO (`--method simpo`) is reference-free —
  length-normalized rewards with a target margin `--gamma`, no reference
  forward at all (roughly half the compute of a DPO step), optional
  `--sft-weight` NLL mix (CPO-SimPO).
- `qlora_train_ebft.py` — Energy-Based Fine-Tuning (EBFT, arXiv:2603.12248):
  on-policy feature-matching policy gradient. The frozen feature network is
  the adapter-disabled base (the DPO/KTO reference trick); rollouts use the
  exact sampler over the differentiable forward; rewards/RLOO live in
  `exllamav3/training/ebft.py` (reference-faithful to `sjelassi/ebft_openrlhf`,
  CPU-tested in `tests/test_ebft.py`). Run `--self-test` first on a new
  model. First known EBFT + LoRA/quantized implementation — treat results
  as research, compare against an SFT baseline on the same data.
- `run_report.py` — the default local logging path (replaces wandb for
  shareable dashboards). Every run with an `--out` streams config + per-step
  metrics + evals to `<out>/run_report/` and renders a self-contained
  `report.html` (inline vanilla-JS SVG charts, no CDN, no account) on finish;
  a crash still renders whatever it logged. Used by both the SFT
  (`qlora_train_native.py`) and EBFT trainers with a shared metric schema, so
  SFT-vs-EBFT runs render comparable dashboards. `--no-report` opts out;
  `--wandb-project` is still available but off by default. CPU-tested in
  `tests/test_run_report.py`.
  `--live-report` (SFT, EBFT, and DDP trainers) additionally serves a LIVE
  monitor from a localhost http thread and opens it in the browser at run
  start: the same report page redraws its charts as metrics stream in, plus a
  step viewer that decodes the exact examples any optimizer step trains on --
  browsable backward AND forward, since the data order is deterministic and
  batches are recomputed on demand from the trainer's memory (under DDP, every
  rank's shard, labeled by rank). Nothing about the dataset is written to disk
  or into `report.html`: the shareable artifact stays dataset-free, and the
  live view dies with the trainer process.
- `realtime_chat.py` — interactive demo of REAL-TIME (inference-time)
  training: one loaded model chats through the normal generator AND trains
  its LoRA adapter in place (`exllamav3.training.RealtimeQLoRA`). `/learn
  <corrected reply>` retrains the last exchange, `/ingest file.jsonl` feeds a
  batch of samples, `/lr` sets the constant learning rate live, `/unload`
  compares against the base model. The coordinator alternates serving and
  training under a readers/writer lock, pushes updated adapter weights into
  the runtime LoRA slots in memory after every ingest (no save/load
  round-trip), nukes the generator's page table on each update (cached KV is
  stale after a weight change; target only `q/o/gate/up/down_proj` for an
  adapter-free KV cache instead), and writes timestamped rollback
  checkpoints (ordinary PEFT dirs — they load everywhere the offline
  trainers' adapters do). Between ingests the persistent training state
  (fp32 LoRA masters, Adam moments, PiSSA offsets) is parked in system RAM
  and the CUDA cache flushed, so an idle-but-trainable server holds only its
  serving footprint in VRAM; the next ingest moves it back, value-exact both
  ways (`offload_when_idle`, on by default; `--no-idle-offload` to keep it
  resident). The mirror image covers serving-only component models: a vision
  tower or MTP/draft head loaded next to the text trunk is never trained (the
  LoRA targets live in the trunk), so `rt.attach_aux_models(vision_model,
  draft_model)` parks them OUT of VRAM for the duration of every ingest and
  restores them — same devices, value-exact, guaranteed even if the ingest
  fails — before serving resumes (`offload_aux_when_training`, on by default;
  `exllamav3.training.ModelParker` is the reusable mechanism). The demo's
  `--mtp` flag loads the model's MTP head for speculative decoding and wires
  exactly this. The offline trainer loads a component only when asked to
  (`--vision` for the tower, `--mtp-targets` to *train* the head — see the
  MTP section below); otherwise only `Model.from_config(config)`'s text trunk
  — so this applies to the serve-and-train path only. The script doubles as the reference wiring
  for server backends (e.g. tabbyAPI's `backends/exllamav3/model.py`); the
  `[integration]`-marked lines are the complete glue. CPU-tested in
  `tests/test_realtime.py`.
- `qlora_validate_native.py` — the correctness gates: compares the
  differentiable training forward against exllamav3's own inference forward.
  Run this FIRST on any new model/architecture.
- `qlora_infer_native.py` — before/after generation with an adapter on the
  native inference path.
- `expert_demo.py` — BASE → ADAPTED → UNLOADED generation check for MoE
  routed-expert adapters, using the trainer's chat formats (gemma4-nothink,
  qwen3.5-nothink, …) and layer-split loading; greedy so unload can be
  compared byte-for-byte (see the MoE routing-tie caveat in its docstring).
- `merge_lora_bf16.py` — fold a trained adapter into the unquantized bf16 HF
  weights (`W += (alpha/r)·B@A`, matching the inference loader), preserving the
  shard layout so `convert.py` can requantize the result. This is the
  merge-and-requantize deploy path (baked-in adapter, no runtime LoRA). Default
  and pissa inits; rejects mixed-rank (`rank_pattern`) adapters.
- `qlora_train_bnb.py` — the bitsandbytes-NF4 comparison arm (matched
  benchmark harness; needs its own transformers+peft+bitsandbytes venv).
- `experiments/` — one-off, experiment-specific tooling (dataset generation,
  style metrics, run scripts); kept for reproducibility, not part of the
  reusable path. See its README.

## MoE models (Qwen3-MoE, Qwen3.5-MoE, Gemma4 MoE)

Supported with the std softmax top-k router (incl. Qwen3.5-MoE's shared
expert + sigmoid shared gate, and the Gemma4 MoE layout: routing + routed
experts fed from the raw post-attention residual through their own pre-norms,
routed/shared post-norms, per-expert scale). Plain
`gate_proj`/`up_proj`/`down_proj` targets adapt the dense / shared-expert
paths only; opt in to the routed
experts with `--targets ... expert_gate_proj expert_up_proj expert_down_proj`
(consider a small `--expert-r` — it's one adapter pair per expert per layer).
The router is always frozen and no aux load-balancing loss is added. Caveat:
routed-expert adapters DO apply in native generation (fixed in Session 26,
box-verified end-to-end in Session 28 — `expert_demo.py` shows the trained
style at runtime on both Qwen3.5-MoE-family and Gemma4-MoE), but MoE decode is
significantly slower while such an adapter is loaded — for serving speed,
deploy by merge-and-requantize. See the Session 20/21/26/28 notes in
`doc/qlora_handoff.md`.

## MTP (multi-token prediction) head training

Qwen3.5/3.6 checkpoints ship a one-block MTP head -- the self-speculative
draft the generator runs with `draft_model = Model.from_config(config,
component="mtp")`. It is a separate component model that `qlora_train_native.py`
otherwise never loads, and its linears sit outside the trunk's `--targets`
matching entirely, so by default an adapter never touches it. Fine-tuning the
trunk shifts the hidden states the head reads, so a stale head's draft
acceptance rate drifts down; `--mtp-targets` retrains it.

- **What trains.** `--mtp-targets [LEAF ...]` loads the head (on the trunk's
  output device; `--mtp-device` to override) and adapts the named leaves
  INSIDE it: the same vocabulary as `--targets` (`q_proj` ... `down_proj`,
  `expert_*` on the MoE head) plus `fc`, the head's `[2d -> d]` input
  projection. A bare `--mtp-targets` = the default attn+mlp list + `fc`.
  `--targets` and `--mtp-targets` are disjoint: neither reaches into the
  other's model, so `--targets --mtp-targets` (an empty trunk list) trains
  **only the MTP tensors**.
- **The loss.** The head's own next-token CE through the shared LM head: at
  position `i` it sees the trunk's final-norm state from `i-1` plus token
  `i`'s embedding and predicts token `i+1` -- exactly the generator's draft
  wiring (`cat(zero_carry, target_hidden[:, :-1])`; a zero carry at position
  0 and at every packed document start). Trunk training jointly: `loss =
  trunk + --mtp-loss-weight * mtp` (the log line shows both). Head only: the
  head's loss *is* the loss, and the trunk runs under `no_grad` (no
  checkpointing, no saved activations -- a cheap run). The eval / best-val
  loss follows the same composition.
- **Two-stage recipe (recommended).** Train the trunk as usual, then fit the
  head to the tuned trunk it will draft for:
  ```
  # stage 1: trunk
  python training/qlora_train_native.py --model M --targets q_proj k_proj v_proj o_proj \
      gate_proj up_proj down_proj ... --out out/trunk
  # stage 2: head, trunk adapter loaded and frozen (same --r/--targets as stage 1)
  python training/qlora_train_native.py --model M --resume out/trunk --freeze-trunk \
      --targets q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --mtp-targets ... --out out/trunk_mtp
  ```
  `--freeze-trunk` keeps the resumed trunk adapters applied (and re-exported)
  but out of the optimizer; the optimizer starts cold. Or train both at once
  in one run (`--targets ... --mtp-targets`), which costs a second head pass
  per step.
- **The adapter.** One dir: the head's tensors sit in the same
  `adapter_model.safetensors` under their own `mtp.layers.N...` / `mtp.fc`
  keys (`adapter_config.json` records `mtp_target_modules`). The runtime
  loader matches keys by module suffix, so `LoRA.from_directory(model, dir)`
  applies the trunk's and skips the head's, and `LoRA.from_directory(
  draft_model, dir)` the reverse -- `qlora_infer_native.py --mtp` does both
  per arm and prints the arm's draft acceptance rate, the number a retrained
  head is meant to move. Resuming a trunk-only checkpoint into an MTP run
  starts the head's adapters fresh (a note is printed).
- **Gate first.** `qlora_validate_native.py --check-mtp` compares the native
  head forward against the inference draft path on the real quant; run it
  before the first `--mtp-targets` run on a new model. Layout supported: the
  Qwen3.5/3.6 head (two pre-fc norms + `fc` over `[embedding | trunk state]`,
  N TransformerBlocks, final norm); the DeepSeek/GLM pre-norm-residual heads
  and the Qwen3.8 hyper-connection head are rejected loudly. Not wired into
  the DDP / preference / EBFT trainers or the realtime coordinator; not
  combinable with `--vision` image batches. **Not yet box-validated** -- the
  CPU test (`tests/test_native_llama.py`) checks the forward against an
  independent reference, the `--check-mtp` gate is the on-box proof.

## MuseGlimmer

Built but NOT box-validated yet — run `qlora_validate_native.py` against the
quant before spending a run on it. The text tower rides the existing feature
path (full-width attention output gate keyed `self_attn.gate_proj` like AFMoE,
unweighted q/k-norm with the `qk_scale_factor` folded into `sm_scale`, Gemma
sandwich norms, per-layer RoPE theta with NoPE on the full-attention layers,
sliding window on the rest). Two things were specific to it and are handled
outside the block: the unweighted **norm on the token embeddings**
(`embed_tokens.embed_norm`, applied by the native forward right after the
embedding lookup) and the head's **logit pre-scale** (`output_multiplier`,
folded into the head input so every head path — materialized logits, the
trainable/LoRA head, both fused-CE heads — gets it before the softcap).

Data side: use `--prompt-format jinja`. Muse is Harmony-like — a turn ends with
`<|eot|>` (not the EOS) and an assistant turn opens with a
`to=user<|message|>` recipient header — so the model's own chat template is the
only renderer that gets both the header masking and the stop token right. (The
`auto` fallback now picks `<|eot|>` too, but it is single-turn and emits no
recipient header.) The vision tower is not built or trained; text only.

## Docs

- `doc/qlora_handoff.md` — the full engineering log (per-session results,
  decision records, backlog).
- `doc/ebft.md` — Energy-Based Fine-Tuning: design decisions, what's verified,
  how to run, and open work. Standalone context-refresh doc for the EBFT path.
- `doc/qlora_feasibility.md`, `doc/qlora_multigpu_plan.md`,
  `doc/qlora_optimization_audit.md` — design rationale and plans.
