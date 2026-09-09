# exl3-qlora

**QLoRA fine-tuning directly on EXL3-quantized models.** This repo began as a fork of [turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3) and still contains the full inference library (synced to upstream v1.4.8 — see the [upstream README](https://github.com/turboderp-org/exllamav3#readme) for inference, conversion, and installation documentation). On top of it there is a self-contained, **transformers-free** training path: a differentiable forward built over the EXL3 trellis (validated against the native inference forward to 100% argmax agreement), a plain-PyTorch trainer, and adapters that save in standard PEFT format.

### Why train on EXL3 instead of bitsandbytes?

- Train on 2-8bpw quants, including non-integer. BNB is 4 or 8 bit.
- EXL3 is higher accuracy than BNB, theoretically this should help? - still needs benchmarking
- EXL3 is more performant in the lower and mid-size batches that are often used when squeezing big models onto consumer cards.

### Quick start

```bash
# install: PyTorch (CUDA 12.4+) first, then this repo (builds the CUDA extension),
# plus the one training dep
pip install -r requirements.txt
pip install .
pip install datasets            # optional extras: flash-attn, liger-kernel, bitsandbytes

# 1. prove the differentiable forward is correct for YOUR model (run this first)
python training/qlora_validate_native.py --model /path/to/exl3-model --compute-dtype bfloat16

# 2. edit the config, then train
python training/qlora_train.py --config training/qlora_train_config.yaml

# 3. before/after comparison on the native inference path
python training/qlora_infer_native.py --model /path/to/exl3-model --adapter out/my_adapter
```

Everything is driven by one YAML file ([`training/qlora_train_config.yaml`](training/qlora_train_config.yaml) is the fully-commented reference — its keys mirror the CLI flags of `training/qlora_train_native.py` one-to-one). A minimal config looks like:

```yaml
model: /models/Llama-3.2-3B-Instruct-exl3-4bpw
out: out/my_adapter
parallel: split           # single | split (layer-split across GPUs) | ddp (torchrun)

r: 64
alpha: 64.0               # use alpha = r with init_lora: pissa
lr: 5e-6
epochs: 1.0
batch: 3
seq_len: 8192
pack: true                # sample packing (best-fit-decreasing, ~98% fill)

dataset: /data/my-set.jsonl
messages_key: messages    # OpenAI-style chats; or instruction/context/response keys
prompt_format: auto

init_lora: pissa          # default | pissa | qerr | eva  (see below)
eval_split: test          # held-out eval from the dataset's own split
eval_every: 50
compute_dtype: bfloat16
use_liger: true
optim: paged_adamw8bit
```

### What's in the box

- **Single-GPU, multi-GPU layer-split (`parallel: split`), and DDP (`parallel: ddp`)** training of LoRA adapters over a frozen EXL3 base — plus optional embedding/LM-head training (full or low-rank).
- **Preference optimization: DPO, KTO, and SimPO** (`training/qlora_train_pref.py --method dpo|kto|simpo`, or via the YAML launcher with `method: dpo|kto|simpo` and the `qlora_train_pref_config.yaml` template). DPO/KTO use the frozen quantized base as the reference model (adapter-disable trick — no second model copy); SimPO is reference-free (length-normalized rewards + target margin γ — no reference forward, so a step costs about half a DPO step). Loss semantics follow [HuggingFace TRL](https://github.com/huggingface/trl)'s stable `DPOTrainer`/`KTOTrainer`/`CPOTrainer` (with credit — see below), so β/loss-variant hyperparameters transfer directly; variants: sigmoid/cDPO, hinge (SLiC), IPO, KTO, APO-zero-unpaired, SimPO (+ optional CPO-style SFT mix).
- **Real-time (inference-time) training**: one loaded model serves generation and trains its adapter in place (`exllamav3.training.RealtimeQLoRA`) — `ingest()` a batch of samples, the updated adapter is live in the very next generation (in-memory sync into the runtime LoRA slots; KV cache invalidated on update, or target only non-KV projections and skip invalidation). Constant externally-settable LR, timestamped rollback checkpoints, a readers/writer lock separating serving from training. Interactive demo + server-integration reference: `training/realtime_chat.py`.
- **Memory levers** for long context on consumer cards: gradient checkpointing, activation offload to CPU RAM, fused/chunked cross-entropy (chunked over the vocab too, for 256k-vocab models), 8-bit and paged optimizers, Liger kernels (RMSNorm/RoPE/SwiGLU).
- **A real eval harness**: held-out loss from your dataset's own split (`eval_split`) or a carved fraction (`val_frac`), an optional second monitor set (`eval2_*`, e.g. wikitext LM loss watched next to your task loss), `save_best` checkpointing, periodic live sample generations, and a per-run CSV log of hyperparameters/losses/VRAM/throughput.
- **Correctness gates, not vibes**: `qlora_validate_native.py` checks the differentiable forward against the native inference forward, the Liger backward against plain torch, packing isolation, and each adapter init's step-0 math — before you spend GPU-days. A CPU test suite covers the gradient path end-to-end.
- **Vision (image+text) SFT** on the VLM bases exllamav3 serves — Qwen2.5/3/3.5-VL, Gemma3/4, Mistral3 (`vision: true`): the model's own vision tower runs frozen, its features are spliced into the text tower at the image positions with the arch's exact image token layout (3-D mRoPE, deepstack, Gemma4's bidirectional image spans all reproduced), and only the language model's adapters train — Axolotl's recipe, with the same content-parts dataset layout. See [Vision training](#vision-image--text-training) below.
- **Standard outputs**: adapters save as PEFT-format safetensors, loadable by exllamav3's native LoRA loader (TabbyAPI), PEFT, or merge scripts. Runtime adapters apply correctly on every inference path; note that decode is slower **while an adapter is loaded** (the graph-fused decode paths can't run under one — `unload()` restores full speed), and deploying via merge-and-requantize has no hit at all.

### Supported architectures (training)

The differentiable forward reads every norm/activation/scale from the loaded modules and is validated against the native inference forward per architecture. Unsupported layouts are **rejected loudly at construction** — nothing silently mistrains.

| Architecture family | Examples | Status |
|---|---|---|
| Llama (plain pre-norm dense) | Llama 3.x, DeciLM-lite | **Box-proven** (SFT, DPO/KTO, packing, DDP, split) |
| Mistral dense | Mistral 7B v0.3, Mistral-Nemo (Rocinante-XL-16B), Mistral Small/Medium 3.x (`mistral3`) | **Box-proven** (16B metharme SFT; Medium-3.5-128B) |
| Qwen2 dense | Qwen2/2.5 | Accepted (same plain path as Llama) |
| Qwen3 dense | Qwen3 4B/8B/14B | **Box-proven** (q/k-norm path) |
| Qwen3-MoE / Qwen3.5-MoE | Qwen3-30B-A3B, Qwen3.6-35B-A3B | **Box-proven** (std softmax router; shared expert + sigmoid shared gate; routed-expert adapters opt-in via `expert_*` targets) |
| Qwen3.5/3.6 hybrids | Qwen3.5 0.8B–4B, Qwen3.6-27B | **Box-proven** (differentiable Gated DeltaNet + gated attention; no sample packing on GDN models) |
| LFM2 / LFM2-MoE hybrids | LFM2.5-8B-A1B | Accepted (differentiable ShortConv gated-causal-conv layers + q/k-normed attention; dots sigmoid MoE router with expert bias; no sample packing on ShortConv models) — **not yet box-tested**; run `qlora_validate_native.py` first |
| Gemma 3/4 | Gemma3, Gemma4-12B, Gemma4 MoE (MeroMero-26B) | **Box-proven** (sandwich norms, GeGLU, sliding/full, softcap, big-head, Gemma4 MoE alt-residual layout) |
| AFMoE | **Trinity-Nano** (Arcee), dots.llm1-style sigmoid routers | **Box-proven** (10-step SFT + fast-vs-legacy A/B + adapter steering generation at inference) — dots sigmoid router (selection bias, normalize-over-selected, route scale), full-width attention output gate, NoPE full-attention layers, muP embedding, ungated shared expert, dense-first-N layers |
| MuseGlimmer | Muse Glimmer (text tower) | Accepted (full-width attention gate, scaleless q/k-norm + q scale factor, sandwich norms, per-layer RoPE theta with NoPE full-attention layers, embedding norm, logit pre-scale on the softcapped head) — **not yet box-tested**; run `qlora_validate_native.py` first, and use `--prompt-format jinja` (its `<\|eot\|>` turn-end and `to=user` recipient header come from the model's own template) |
| Mixtral | Mixtral 8x7B | Accepted (std router, no shared expert) — not yet box-tested |
| Qwen-VL text towers | Qwen2.5/3-VL, Qwen3.5-VL | **Box-proven**, text-only (mRoPE collapses to 1D RoPE). Image+text via `vision: true` (3-D mRoPE + deepstack) — built, **not yet box-tested**; run `qlora_validate_native.py --image` first |
| Gemma3 / Gemma4 / Mistral3 vision | Gemma3-4B/12B-it, Gemma4-12B-it, Mistral Small 3.1 | Text-only box-proven (Gemma). Image+text via `vision: true` (fixed-size soft tokens; Gemma4 bidirectional image spans; Mistral3 `[IMG_BREAK]` rows) — built, **not yet box-tested** |
| Rejected loudly | Qwen3-Next (fused-qkvz GDN), grouped ds3-router MoE (DeepSeek-V3), headwise attention gating, non-NeoX RoPE | — |

MoE note: the plain `gate_proj`/`up_proj`/`down_proj` targets adapt dense MLPs and the always-active shared expert; routed experts are opt-in (`expert_gate_proj` etc., with `--expert-r` for rank). Routers stay frozen. On AFMoE the *attention* gate is keyed `self_attn.gate_proj` in the checkpoint and rides the `gate_proj` target (or `attn_gate_proj` to adapt it alone).

### Prompt formats (`--prompt-format` / `prompt_format:`)

| Format | Template | Use with |
|---|---|---|
| `auto` (default) | The model's own architecture template (`default_chat_prompt`) + arch-correct turn-end token | Any supported base |
| `llama3` | `<\|start_header_id\|>…<\|eot_id\|>` headers | Llama-3 family (explicit / cross-arch) |
| `mistral` | `<s>[SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST]` (V7+/V13, no spaces) | Mistral instruct family |
| `chatml` (= `qwen3.5`) | `<\|im_start\|>role\n…<\|im_end\|>` | Qwen, **Trinity/AFMoE**, any ChatML base |
| `qwen3.5-nothink` | ChatML with the `<think>` block pre-closed empty | Qwen3.5/3.6 reasoning bases, trained to answer directly |
| `gemma4-nothink` | Gemma4 turns with the thought channel pre-closed | Gemma4 |
| `metharme` | `<\|system\|>/<\|user\|>/<\|model\|>` markers | Pygmalion-style tunes on any base |

All formats do exact prompt/response boundary masking (prompt and response are tokenized separately) and single-BOS normalization; verify any new base with `--inspect 3` before training.

### Vision (image + text) training

`vision: true` turns the SFT trainer into a VLM fine-tuner on any base whose exllamav3 architecture has a vision component (Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL incl. MoE, Gemma3, Gemma4, Mistral Small 3.x). The approach mirrors how [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) does multimodal fine-tuning: the vision tower + projector stay frozen, LoRA targets only the language model, and the dataset is OpenAI-style `messages` whose content is a parts list:

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image", "path": "images/0001.png"},
    {"type": "text",  "text": "What breed is this dog?"}]},
  {"role": "assistant", "content": "A border collie."}]}
```

An image part names its pixels by `path` (relative to the dataset file), `url`, `base64`, `image` (a PIL image / HF `datasets.Image` column, e.g. from a parquet set) or OpenAI's `image_url`; a bare `{"type": "image"}` part takes the next entry of the row's `images` column (`images_key`), the layout of most HF VLM sets. Images can sit anywhere in any turn, several per row, on every prompt format (`auto`, the explicit formats, or `jinja`); text-only rows in the same set train as usual.

What happens under the hood, and why it's exact:

- **Features come from exllamav3's own vision component** (`Model.from_config(config, component="vision")` → `get_image_embeddings`), the forward the generator runs. They're encoded once at dataset build (cached in CPU RAM up to `vision_cache_gb`; if everything fits, the tower is unloaded again to free its VRAM) and spliced into the text tower's embedding stream at the image positions — the same `indexed_embeddings` substitution `modules.Embedding` performs at inference. The image token layout is the arch's own (`<|vision_start|>` + N slots + `<|vision_end|>` on Qwen-VL, `<start_of_image>` + 256 + `<end_of_image>` on Gemma3, `[IMG]`/`[IMG_BREAK]` rows on Mistral3, ...), so train and serve see identical token streams. Image tokens are always masked.
- **The text tower's multimodal math is reproduced differentiably**: Qwen-VL's 3-D mRoPE position ids (a mirror of the `gen_mrope_pos_ids` kernel, CPU-tested against a transliteration of it) and the per-band interleaved rotation (a mirror of `RoPE.get_mrope_freqs`); Qwen3-VL / Qwen3.5-VL **deepstack** (vision intermediate features added after the first blocks — the `DeepstackEmbed` modules, which the training layout check now accepts instead of rejecting a VL text tower); Gemma4's **bidirectional attention inside each image span** (the eager attention path takes the span mask on those batches).
- **A correctness gate**: `qlora_validate_native.py --image cat.png` runs one image through the vision tower and compares the differentiable forward (features through the training splice) against the native multimodal forward (the generator's prefill params), position by position — plus the Python mRoPE ids against the extension kernel. Run it on a new base before a `vision: true` run, exactly as for a new architecture.

Knobs: `image_max_pixels` bounds the image token count (the VRAM lever — Qwen-VL at its default bounds can emit thousands of tokens per image), `vision_cache_gb` the feature cache, `vision_device` where the tower loads. `--inspect N` shows each example's image count / token positions. Not combinable with `pack` (image rows aren't packed, as in Axolotl); single/split only. Status: the whole path is built and CPU-tested but **awaits its first box run** (see the handoff log's box checklist).

### Modern PEFT: SVD adapter initializations

Short SFT runs spend a large fraction of their steps just growing the adapter off the ground (the default zero-init of B). This fork implements the current crop of SVD-based initializations, adapted to an immutable quantized base — select with one config key, `init_lora`:

- **`pissa`** ([PiSSA](https://arxiv.org/abs/2404.02948)) — the adapter starts as the top-r principal components of the base weights, trained against a residual base realized as a frozen offset (the trellis itself is never rewritten). Exports as a converted rank-2r standard LoRA so any consumer loads it correctly. **Current default recommendation: it won its first A/B clearly** (use `alpha = r`).
- **`eva`** ([EVA](https://arxiv.org/abs/2410.07170)) — A is initialized to the top-r right-singular vectors of each layer's *input activations*, streamed from your actual training data through the actual quantized forward in a short pre-pass. Function-preserving at step 0. Freshly built; being evaluated against pissa now.
- **`qerr`** (LoftQ-style, single-shot) — the adapter starts as the closest rank-r repair of the *quantization error* vs the original bf16 weights, aimed at the low-bpw regime where that error is large.
- **`use_rslora`** — rank-stabilized scaling (`alpha/sqrt(r)`) for rank sweeps.

Each init has a hard step-0 gate in `qlora_validate_native.py --init-lora <mode>`.

### Credits

The DPO/KTO/SimPO preference-training implementation follows the loss semantics of **[HuggingFace TRL](https://github.com/huggingface/trl)**'s stable `DPOTrainer`, `KTOTrainer` (KTO stabilized in [trl#6175](https://github.com/huggingface/trl/pull/6175)), and `CPOTrainer` (`loss_type="simpo"`). TRL is Apache-2.0 licensed, Copyright The HuggingFace Team; this fork reimplements the formulations independently against the EXL3 native training path rather than reusing TRL code. Underlying methods: DPO ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)), KTO ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)), IPO, SLiC, SimPO ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)), CPO ([Xu et al. 2024](https://arxiv.org/abs/2401.08417)).

### Project status

Research project under active development. It started as an exllamav3 fork, but has diverged enough to stand alone (upstream is still merged in periodically — currently at v1.4.8 parity — so the inference side stays current). The core mechanism is proven end-to-end: validated forward parity on the quantized weights, healthy trainings from 1B to 16B models on 1–2× RTX 3090 (including 8k-context packed runs on a 12B), adapters that load and steer generation on the native inference path. Training-side architecture support currently covers **Llama-family, Gemma 3/4 (incl. Gemma4 MoE), Qwen3-dense, Qwen3-MoE, Qwen3.5/3.6 hybrids (differentiable Gated DeltaNet + gated attention) incl. Qwen3.5-MoE, AFMoE (Trinity-Nano — dots sigmoid router, gated attention, NoPE), Mistral(-Nemo) dense models, MuseGlimmer text towers (built, awaiting its first box validation), and LFM2 / LFM2-MoE hybrids (LFM2.5-8B-A1B — differentiable ShortConv layers; built, awaiting its first box validation)** (see the supported-architectures table above; no sample packing on Gated DeltaNet or ShortConv models); unsupported features are rejected loudly rather than silently mistrained. **Vision (image+text) SFT** on the Qwen-VL / Gemma3 / Gemma4 / Mistral3 bases is built and CPU-tested (frozen vision tower, differentiable mRoPE / deepstack / bidirectional image spans, `qlora_validate_native.py --image` gate) and likewise awaits its first box run — see [Vision training](#vision-image--text-training). Interfaces may still move between sessions — the full engineering log with per-session results and rationale lives in [`doc/qlora_handoff.md`](doc/qlora_handoff.md), and experiment-specific tooling is quarantined in [`training/experiments/`](training/experiments/).
