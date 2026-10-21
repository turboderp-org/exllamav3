# Plan: MemT-1.0-Nano (native persistent memory) on exl3-qlora

Written 2026-10-21 (branch `claude/brave-lovelace-myoqvj`). **Status: Phase 0 built and
CPU-tested; nothing box-verified.** The HuggingFace model repo and arXiv were unreachable
from the session that wrote this (egress proxy), so everything below comes from the
released code at <https://github.com/MemTensor/MemT-1.0> (the SGLang integration under
`inference/memt_sglang/`, vendored fla `gdn2` kernels) and the tech report shipped in that
repo (`tech_report.pdf`, = arXiv 2610.26825). The two things that MUST be checked against
the real checkpoint before trusting any of this are listed in section 6.

## 0. What MemT-1.0 is

MemTensor's "memory foundation model" family (Oct 19 2026): a frozen backbone plus
**native memory blocks** on selected layers that keep a small persistent state per
conversation/trajectory, so history is *stored in the state* rather than re-streamed as
context. Two releases:

| | MemT-1.0-Nano | MemT-1.0-Flash |
|---|---|---|
| Backbone | **Qwen3.6-35B-A3B, as released** (3B active) | DeepSeek-V4-Flash after a post-training pass (17B active) |
| Memory layers | the 10 `full_attention` layers (3, 7, ..., 39 of 40) | 20 compressed-attention layers |
| Memory geometry | 16 heads x 256 x 256 per layer | 64 heads x 512 x 512 |
| Trainable memory params | **450,931,360** | 3.6B |
| Persistent state / trajectory | 10,485,760 fp32 = **40 MiB** | 1280 MiB |
| License | PolyForm Noncommercial (code), CC BY-NC-SA 4.0 (paper) | same |

Paper headline for Nano at 128K history: ~1.86x end-to-end speedup and 66x less persistent
storage than a full-context KV cache; memory benchmarks above the parametric baselines
(Temp-LoRA etc.), "largely preserving" general capability.

**Why it fits this fork.** The backbone is frozen (paper app. B.3: "Backbone parameters,
including the output projections used for fusion, are frozen during memory training") and
for Nano it is the stock Qwen3.6-35B-A3B, which the fork already serves and trains
(Qwen3.5-MoE: GDN + gated attention + MoE, box-proven). So MemT-Nano = a Qwen3.6-35B-A3B
EXL3 quant + ~900 MB of bf16 memory tensors + 40 MiB of state per session. That is exactly
the "frozen quantized base + small trainable surface" shape this project is built around:
memory training on a 3-4 bpw backbone (13-18 GB) plus 451M fp32-master-weight memory
params (~6 GB with Adam, less with the 8-bit optimizers) is a single-3090 workload.

## 1. The architecture, precisely (Nano)

Per memory layer `l` (an `Attention` block with the Qwen3.5 interleaved sigmoid gate), with
`x` = the block's input after `input_layernorm` (`[b, t, 2048]`), `H = 16`, `D = 256`:

```
in_proj  [5*H*D + 2*D, 2048] = [20992, 2048]   rows: q | k | v | erase | write | decay_latent(D) | gate_latent(D)
f_down   [H*D, D]      forget-gate up-projection (rank-256 factorization)
g_down   [H*D, D]      output-gate up-projection
decay_scale_log [H]    s_h            decay_bias [H*D]  b_lambda          o_norm.weight [D]
```
(45,093,136 params/layer; x10 = the paper's 450,931,360 exactly, which is how the layout
was confirmed without the checkpoint.)

Read (every token of a turn, against the SAME snapshot `M_t` from the start of the turn):
```
q_n = l2norm(x W_q)                 R = q_n M_t / sqrt(D)                       [b,t,H,D]
G   = (x W_g/d) W_g/u               out = concat_h[ RMSNorm_D(R_h) * w_o * 2*tanh(silu(G_h)/2) ]
o_proj_input = sigmoid(attn_gate) * attn_ctx  +  out        <- fused HERE, before o_proj
```
Write ("commit", one pass over the completed turn `X_t || Y_t`, published as `M_{t+1}`),
per token i and head h, `u = sqrt(sigmoid(x W_beta)) * l2norm(x W_k)` (a row over dk):
```
S <- diag(exp(lambda_i)) S            lambda = -exp(s_h) * softplus((x W_f) W_lambda + b_lambda)   (per k-channel, <= 0)
S <- S - u^T (u S)                    symmetrized erase  (I - u^T u) S   -- the "stable GDN-2" fix, app. C
S <- S + u^T (sigmoid(x W_omega) * v)
```
Memory starts at zero for a new trajectory. With a zero state the readout is exactly zero
(RMSNorm of 0), so the memory-free forward IS the backbone forward. This is the GDN-2
(NVIDIA Gated DeltaNet-2) recurrence with a per-channel erase vector and a per-channel
write gate, symmetrized; it is a cousin of the KDA path the fork already trains
(per-k-channel decay) but with a different erase/write and no conv / no output
projection of its own (the backbone's frozen `o_proj` is the fusion).

Serving semantics from the release (`runtime/executor.py`, `client/session.py`): a request
either *reads* (ordinary generation: read-only from the session's published state; `set_memory_read(False)` turns even that off) or *commits* (`enable_memory=True` on a turn:
after the reply completes the whole turn is re-run as one un-split pass from the published
state, and the result is published). Snapshots are safetensors files with one fp32
`memory.{layer}` tensor of `[H, D, D]` per memory layer plus `format`/`identity` metadata.

Mixture-of-Storage (E memory experts with dense routing) is in the paper but **E = 1 in
both released checkpoints**; the released kernels implement only that. Not built here.

## 2. What is built (Phase 0, this branch)

| File | What |
|---|---|
| `exllamav3/training/memt.py` | Plain-torch, differentiable reference math: `MemTSpec`/`MemTWeights`, the in_proj row layout, gates, l2norm, read, the sequential stable-GDN-2 scan (`memt_update_reference`), the gated softcap norm, `memt_block_forward(x, w, spec, read_state, commit, write_state)`, and `parse_memt_block` (the config `memt` object validated the way the release validates it: pinned switches, no unknown keys, heads == backbone heads) |
| `tests/test_memt.py` | 8 CPU tests, all passing: Nano geometry == Table 6; zero state is an exact no-op; scan == explicit matrix form of eq. (22) in fp64; whole block == a port of the release's `QwenMemoryBlock` (its own row bookkeeping) with the Triton scan swapped for the matrix oracle; symmetrized erase non-expansive + long commit bounded; within-turn readout independent of earlier tokens; config parser; fp64 gradcheck through all six tensors and the incoming state |
| `exllamav3/modules/arch_specific/memt_memory.py` | `MemTMemory(Module)`: loads the six bf16 tensors (unquantized), carried through conversion like norms/A_log; `fuse(x, o, params)` adds the readout to the gated context before `o_proj`; runs the commit scan when asked. State rides in `params["memt_read"]` / `params["memt_commit"]` (dicts `{layer_idx: fp32 [H,D,D] or [bsz,H,D,D]}`) |
| `exllamav3/architecture/memt_qwen3_5_moe.py` | `MemTQwen3_5MoeConfig` (arch `MemTQwen3_5MoeForCausalLM`; the release's `backbone_configs` wrapper is merged by the base `Config`'s `config_key`, so the stock Qwen3.5-MoE parser reads the backbone unchanged) and `MemTQwen3_5MoeModel` (stock Qwen3.5-MoE model + a memory block attached to every full-attention layer's `Attention`; helpers `memt_zero_state`, `memt_commit(input_ids, states)`, `memt_save_state`, `memt_load_state` in the release's snapshot layout) |
| `exllamav3/modules/attn.py` | `Attention(memt=...)` / `attach_memt()`; `o = self.memt.fuse(x, o, params)` after the sigmoid gate in both decode paths; the graph-captured decode block is bypassed while a state is attached; TP export asserts no memory |
| `exllamav3/architecture/architectures.py` | registry entry |
| `tests/model_setup/test_memt_config.py` | synthetic-config test through the real registry (needs the package importable, i.e. the box): backbone read through the wrapper, memory on layers 3/7 of a 8-layer toy, blocks attached and iterable, pinned switch / head geometry refused |

Not built: the fla `gdn2` chunked kernels for long commits (the scan is a python loop over
tokens: fine for turn-sized commits, O(t) kernel launches), per-job memory in the generator
(all rows of a batched forward share one state unless `[bsz,H,D,D]` is passed), the
`enable_memory` turn-end commit inside the generator, MoS, TP, and everything in section 4.

## 3. Phase 1 -- box validation (next; needs the checkpoint + a GPU)

1. **Get the weights.** `huggingface.co/MemTensor/MemT-1.0-Nano` (bf16 safetensors, ~70 GB).
   Confirm `config.json` matches the layout the parser expects: top-level
   `model_type: memt_qwen3_5_moe`, `architectures: ["MemTQwen3_5MoeForCausalLM"]`, a
   `memt` object with exactly the 16 keys in `MEMT_FIXED_FIELDS` + `MEMT_SIZE_FIELDS`, and
   `backbone_configs: {model_type: qwen3_5_moe, text_config: {...}}`. The parser refuses
   anything else on purpose; if the released file differs, fix the parser, not the file.
2. **Cheap route first: sidecar instead of requantizing.** Since the backbone is Qwen3.6-35B-A3B
   as released, an existing EXL3 quant of it should serve as the backbone. Verify with a
   checksum that `model.language_model.*` tensors of MemT-Nano are byte-identical to
   Qwen3.6-35B-A3B's (a short safetensors script; do NOT assume). If identical: copy the
   EXL3 quant directory, drop in a safetensors file holding only the
   `model.language_model.layers.{3,7,...,39}.self_attn.memt.*` tensors (~900 MB bf16),
   rewrite `config.json` to the MemT layout. The loader finds tensors by key across files,
   so nothing else changes. If NOT identical: `convert.py` the MemT repo directly (the memory
   tensors pass through unquantized via `get_tensors`).
3. **Memory-free parity gate**: load the EXL3 MemT model, run the usual
   `qlora_validate_native.py` / `examples/chat.py` with no state -> must behave exactly as the
   plain Qwen3.6 quant (the readout is zero). Cheap sanity that the wiring is inert.
4. **Readout parity vs the release.** Start the release server
   (`sh start_memt_nano.sh`, two GPUs, bf16) on the same prompt; `MemTSession.chat(...,
   enable_memory=True)` for one turn, `save_state()` the snapshot; then on our side
   `states = model.memt_load_state(snap)`; `model.forward(ids, {"memt_read": states})` and
   compare logits for the follow-up turn against the server's (`NativeClient` exposes
   logprobs). Also compare our `memt_commit(ids, zero)` against their snapshot after the
   first turn: fp32 state, expect ~1e-3 relative (bf16 GEMMs on both sides, EXL3 noise on
   the hidden state). This is the gate; the CPU tests only prove the math against the
   release's *code*, not its numerics.
5. **Generator integration (serving).** A per-job `memt_read` state (job -> state, stacked to
   `[bsz,H,D,D]` in the batch order, through `params`), `set_memory_read` equivalent, and
   turn-end commit = `model.memt_commit` over the finished turn (the release re-runs the
   turn un-split from the published state, so it is one extra prefill-cost pass per
   turn). Cache invalidation is not needed: memory touches the hidden state only through
   `o_proj`'s input of the *next* turn, and the read state is constant within a turn.
   Long commits: port the release's vendored `fla.ops.gdn2` chunked forward (they vendor
   fla 0.5.2's `chunk_intra` / `wy_fast` + their own `_fwd_h` state scan with per-sample
   snapshots) behind `memt_update_reference`.

## 4. Phase 2 -- training the memory on the quantized backbone

What MemT's own training does (sec. 3.3, app. B.3): backbone frozen; trainable = the six
memory tensors per layer; data in a unified context -> query -> answer form; objective =
CE on the answer + top-K forward-KL distillation from the *same backbone given the full
context* (teacher = backbone without memory, so no second model: the fork's
adapter-disable trick generalizes -- run the frozen base with full context in
`no_grad` for the teacher logits, then the student with the context committed to memory
and only the query in context). Temperature/KL/CE weights in Table 7: 1 / 1 / 0 for the
released checkpoints (pure distillation!), Top-64 KL. Memory state and decay/norm math in
fp32, projections bf16; forget biases initialized from log-uniform half-lives in
[2048, 32768] tokens; `s_h` and `b_lambda` excluded from weight decay.

On this fork that becomes:
1. `native_llama.py`: memory readout in `_block_forward` for `attn` blocks carrying a
   `memt` entry (after the sigmoid gate, before `o_proj`; `training.memt.memt_block_forward`
   is already differentiable), with the per-layer read state passed in like the vision
   splice. A `commit` forward mode that returns `{layer: next_state}` with grad.
2. A new trainable surface: `memt_*` targets (full-parameter, fp32 master, the six
   tensors) alongside the LoRA targets -- the adapter save format needs a MemT variant
   (the release has no adapter format; save the six tensors per layer in the HF key
   layout so the result drops into the release server too).
3. Trainer: multi-turn samples; per sample, run the context through `commit` (with grad)
   -> state -> query/answer forward reading it -> loss; optional distillation teacher = the
   same quantized backbone with the full context, `no_grad`. Backprop through the commit
   scan: the sequential reference keeps every per-token state in the graph
   (`O(t * H * D * D)` fp32 = 4 MB/token/layer -- 1k-token contexts are fine, longer needs
   the fla `gdn2` backward or checkpointing the scan in blocks).
4. Gates: `qlora_validate_native.py` parity on the memory readout against the inference
   module (same recipe as every other arch), gradcheck already in `tests/test_memt.py`.

Then the interesting experiments: (a) **memory-LoRA**: LoRA on `in_proj` instead of full
memory training (43M of the 45M params per layer are `in_proj`); (b) continue-training
Nano's memory on domain multi-turn data; (c) **bolt memory onto other backbones the fork
serves** (Qwen3.6-27B dense, Gemma4...) -- the init recipe is in app. B (q from the
backbone's q_proj, k/v copied from the 2 KV heads into 16 memory heads, gates
`N(0, 0.02^2)`, forget expansion zero, output expansion `N(0, 0.001^2)`), i.e. a MemT-style
memory can be grown on any EXL3 quant with the distillation objective. That is the one
nobody else can do cheaply.

## 5. Open questions / risks

- The exact `config.json` of the release is unseen (section 6). `tie_word_embeddings` is
  read at the merged top level; if the release keeps it inside `text_config` the parser
  falls back to False (correct for 35B-A3B, which has a separate `lm_head`).
- Numerics: the release runs bf16 GEMMs for the memory projections and bf16 hidden
  states; here `x` is fp16 (cast to bf16 in `fuse`) and the backbone is EXL3 -> expect
  state drift vs the server over many turns; the parity gate in 3.4 measures it.
- The readout is fused after the backbone sigmoid gate (release: `o_proj` pre-hook, and
  the paper's app. B.1 says the gate applies only to the original branch). Both decode
  paths of `Attention` apply the gate before `project_o`, so the hook sits in the right
  place; the graph-captured path is bypassed while a state is attached (slower decode
  while memory is on, like a runtime LoRA).
- Batched serving shares one state unless the caller stacks per-row states; the generator
  work in 3.5 is what makes sessions real.
- License: PolyForm Noncommercial / CC BY-NC-SA on the release and weights.

## 6. Must-verify list (things assumed without the checkpoint in hand)

1. `config.json` layout and the exact `memt` keys (parser is strict by design).
2. Backbone tensors byte-identical to Qwen3.6-35B-A3B (enables the sidecar route).
3. Memory tensor dtype bf16 and shapes as in section 1 (the loader asserts shapes).
4. Readout + commit parity vs the release server (3.4).
