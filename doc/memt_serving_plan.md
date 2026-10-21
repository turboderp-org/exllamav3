# MemT-1.0-Nano: validation findings and the TabbyAPI serving plan

Written 2026-10-21 on branch `claude/brave-lovelace-myoqvj` (pushed through dde61e31). For review
before any serving code is written. Background and the architecture are in `doc/memt_plan.md`.

## 1. Where things stand

MemT-1.0-Nano is Qwen3.6-35B-A3B, frozen, plus a memory block on each of the 10 full-attention
layers (3, 7, ..., 39). The memory is a per-conversation fp32 state of `[1, 16, 256, 256]` per
layer, 40 MiB in total. Inference *reads* the state (the readout is added to the gated attention
context before `o_proj`); a *commit* runs a turn through the model and folds it into the state.

Everything below ran on this box (2x3090) unless marked otherwise.

### Weights

- `/mnt/two/weights/MemT-1.0-Nano` -- the release, bf16, 66 GB, 26 shards plus a separate
  `memt_block.safetensors` (902 MB) holding only the memory tensors.
- `/mnt/two/weights/yeasah_Qwen3.6-35B-A3B-exl3-6.00bpw` -- a stock EXL3 quant of the backbone.
- `/mnt/two/weights/MemT-1.0-Nano-exl3-6.00bpw` -- the serving model: the yeasah quant
  hardlinked, plus `memt_block.safetensors`, plus MemT's `config.json` (with the quant's
  `quantization_config` added), `chat_template.jinja`, `tokenizer_config.json` and
  `generation_config.json`. No requantization.

The sidecar route rests on the backbone being stock. Checked by comparing tensors: all 693 shared
tensors have the same dtype and shape, and 138 sampled tensors (every tensor in layers 0-3, 19,
20, 38, 39, plus embed, head and final norm) are byte-identical. MemT drops the vision tower and
the MTP head; no keys are renamed.

MemT's chat template differs from stock Qwen in one branch: it passes through an assistant
message that opens `<think>` but never closes it (used when replaying turns into memory).

### Validation results

| check | result |
|---|---|
| Memory-free parity (`util/memt_parity_memfree.py`) | MemT with no state and with a zero state is **bitwise equal** to the plain quant, when both land on the same layer split. An autosplit that moves layer 34 to the other card changes its rounding (max dlogit 0.3, KL 3e-4); not a MemT effect. |
| Recall | One committed turn ("my cat is named Quillon and my locker code is 4817"), asked in a fresh context: with memory "Your cat's name is Quillon, and your locker code is 4817.", identical to the answer with the fact in context. Without memory the model says it has no access to that information. |
| End to end vs the release (`util/memt_release_parity.py`) | Release = the authors' `modeling_memt.py`, bf16, all on CPU. Committed state: cosine >= 0.9994 on every layer, relative error 0.8% (layer 3) rising to 3.8% (layer 39). Follow-up KL vs the release: 1.05e-2 nats reading our state, 1.13e-2 reading the release's state, against a no-memory floor of 7.8e-4. The memory effect itself is KL 4.1. |
| Math isolation (`util/memt_release_math_check.py`) | Our memory math on the release's exact captured activations: agrees with the release to 0.17-0.35% (read and commit, every layer), and is closer to an fp64 evaluation than the release on all 20 rows (0.15-0.23% vs 0.23-0.41%). |

Conclusion: the port is correct. The end-to-end gap over the floor comes from the quantized
backbone handing the memory slightly different activations, not from the memory code.

### Bugs and gotchas found on the way

- `MemTMemory` could not be constructed (it never implemented `Module`'s abstract `forward`); the
  CPU tests never built the module. Fixed in d4059abf.
- The release code needs transformers 5.12.1 and does not run on our 5.18. The reference runs use
  an isolated install at `/mnt/two/weights/.tf5121` (transformers 5.12.1, tokenizers 0.22.2,
  kernels 0.12.x) via `PYTHONPATH`; the training venv is untouched.
- accelerate's GPU+CPU offload of the release model produced NaN states and garbage logits on the
  offloaded layers (27 onward). The all-CPU run is clean and takes seconds for short prompts.

## 2. TabbyAPI findings

`~/exl3/tabbyAPI` is the live checkout; its venv imports this fork as an editable install. It
was updated today from e632af4 (2026-08-13) to upstream 9979d01 (2026-10-21), 127 commits. The one
local change (`draft_cache_mode` type) was already upstream and is kept in `git stash` as
`stash@{0}`. Base dependencies did not change, so nothing was reinstalled. The first import
rebuilt the extension against Tabby's torch 2.8 / CUDA 12.8 (separate build dir from the training
venv's torch 2.14.1 / CUDA 13.0), and `MemTQwen3_5MoeForCausalLM` is in the registry.

Model loading needs no Tabby change: Tabby has no architecture list of its own and defers to
exllamav3's registry. Tabby cannot read the context length from MemT's nested config and would
fall back to 8192, so `max_seq_len` has to be set in `config.yml`. Tensor parallel is not
supported for MemT.

How a chat request flows (file:line in `~/exl3/tabbyAPI` before the update; spot-check after):
- `endpoints/OAI/router.py` renders the chat template to text, `backends/exllamav3/model.py`
  `generate_gen` tokenizes it and creates one `AsyncJob`. That is the only job creation site.
- Tabby keeps no conversation state. Every request resends the full message list. There is no
  session id; the OpenAI `user` field exists but is not parsed, and unknown body fields are
  dropped.
- The generator's prefix caching (paged KV by token hash, the CPU/disk KV tier, GDN recurrent
  checkpoints) is keyed on token ids only.

The new built-in web UI (`endpoints/ui/`, served at `/ui`, `web_ui: true` by default) is a static
page that talks to the regular `/v1/chat/completions` endpoint:
- chats ("sessions") live in browser localStorage, each with an id that is never sent to the server;
- every request carries the full history as plain OpenAI messages, with reasoning text left out;
- messages can be edited and regenerated, so history is rewritten under the server.

## 3. The serving plan

### The constraint

A commit is one-way: once a turn is folded into the state it cannot be taken out. The UI sends no
session id and rewrites history freely (edit, regenerate, separate chats). So memory cannot be
tied to a session; it has to follow the conversation content.

### Design: content-addressed snapshots, all inside the exllamav3 generator

No Tabby fork. Everything lives in this fork's generator and only activates when the loaded model
is MemT (`model.caps["memt"]`); every other model behaves exactly as now.

1. **Split the prompt at turn boundaries.** Tabby hands the generator the full rendered prompt
   ids. MemT is Qwen-format only, so the generator splits at `<|im_start|>` tokens into: the
   system turn, completed user/assistant turns, and the final user turn plus generation prompt.
2. **Look up the longest committed prefix.** Snapshots are keyed by a hash of the exact token ids
   of the completed-turn prefix they cover. Find the longest prefix of this request that has one.
3. **Commit lazily, on the next request.** Commit the remaining completed turns from that
   snapshot, one turn at a time, storing a snapshot at each boundary. Because commits happen when
   the next request arrives, the keys are always the client's own rendering of the history, so
   template differences (dropped reasoning, re-tokenized replies) cannot break matching.
4. **Generate with a short prompt.** Run the job on system turn + final user turn (plus
   `memt_context_turns` raw recent turns, see below), reading the matched state. No turn is
   counted twice.
5. **Edits, regenerates, separate chats** fall out of the prefix lookup: a changed history
   matches a shorter prefix and rolls memory back to that point; different chats never share a
   prefix past the system prompt.

Config (generator option, surfaced in Tabby later only if useful): `memt_context_turns`, default
0, keeps the last N completed turns as raw text in the prompt next to the memory, in case recall
alone is too lossy.

Snapshot store: in RAM, LRU, capped by count or bytes (40 MiB each). Disk persistence later if
wanted.

### Generator work in the fork (the bulk of it)

- A per-job memory state (`Job` next to `embeddings`), carried through the requeue re-init (which
  rebuilds jobs from an explicit argument list and would otherwise drop it).
- `params["memt_read"]` in prefill (one job) and in batched decode (stack per-row states into
  `[bsz, H, D, D]` in batch order; rows without memory get zeros). Also the draft/MTP param
  builders if drafting is used with MemT.
- Prefix-cache safety: KV pages and GDN checkpoints computed while reading one state are wrong for
  another. Simplest: no page sharing or recurrent-checkpoint reuse for jobs that read memory.
  Better later: salt the cache keys with the snapshot id.
- The commit forward (`model.memt_commit`) runs between generator iterations, so it never
  collides with ongoing decoding. It does not use the paged cache.

### Tradeoffs and open questions for review

- **Commit-every-turn is our policy, not the release's.** The release keeps turns as raw text
  until a turn is sent with `enable_memory=True`, then replays all uncommitted turns. Committing
  every turn means the model leans on memory from the first exchange; real chats will show how
  well that holds up. `memt_context_turns` is the escape hatch.
- **What gets committed.** We replay turns as the template re-renders them (likely without
  reasoning); the release replays the exact generated ids, reasoning included. Matching the
  client's resend is what makes the stateless UI work, but it differs from what the memory was
  trained on.
- **Latency.** Each request first commits the previous turn: one extra forward over that turn
  before generation starts. Small for chat-length turns, larger after long replies. The commit
  scan is a per-token Python loop (fine for turns, slow for documents; the release's chunked
  `gdn2` kernel is the fix if needed).
- **Turn splitting is Qwen-specific** (`<|im_start|>` boundaries). Fine for MemT, which is Qwen-only.
- **Throughput.** The graph-captured decode path is bypassed while a state is attached, like a
  runtime LoRA, so decode with memory is slower than without.
- **Hash collisions / identity.** Keys are exact token-id hashes, so any change in system prompt,
  template or tokenizer starts a fresh memory. That is the safe direction.
- **VRAM.** The 6.00bpw sidecar is ~27 GB and needs both cards; snapshots in RAM cost 40 MiB each.

### Order of work

1. Generator: per-job state through prefill, decode and requeue; prefix-cache bypass for memory
   jobs. Test: two concurrent jobs with different states produce the same tokens as each run alone.
2. Generator: turn splitter, snapshot store, lazy commit. Test: a scripted multi-turn chat
   through the generator with full history resent each time recalls facts from early turns, and an
   edited history rolls back.
3. Tabby: set `max_seq_len`, load the sidecar, chat in `/ui`. No Tabby code changes expected.

Not in scope now: training the memory (Phase 2 in `doc/memt_plan.md`), disk persistence,
document import, TP.

## 4. Status (2026-10-21, later the same day)

All three steps are built and box-tested; the user's review of section 3 is still wanted, the
design was not changed on the way.

1. **Per-job state** (commit 77c68c3a): `Job(memt_state=..., memt_key=...)`; the key seeds the
   sequence's page-hash chain, so K/V pages and GDN checkpoints are shared only between jobs reading
   the same snapshot (chosen over "no sharing for memory jobs": one line per hash-chain seed, and a
   regenerate still hits the cache). `util/memt_gen_test.py`: committed / zero / no-memory jobs on the
   same prompt give identical tokens alone, batched (states stacked per row) and from cache; zero
   state == plain backbone; a job requeued every 256 tokens matches the unbounded one over 700 tokens.
2. **Conversation memory** (commit 32d56456): `generator/memt_memory.py`, wired into
   `Generator.enqueue`. `util/memt_chat_test.py`: a 4-turn chat recalls facts from every turn with a
   43-token live prompt against a 129-token history; regenerate commits nothing and reproduces the
   reply; editing turn 2 rolls back and recalls the edited facts only; a second chat behind the same
   system prompt shares nothing. Passes with `context_turns` 0 and 1.
3. **Tabby**: no Tabby changes. Memory defaults ON for a MemT model (`memt_memory=None` in
   `Generator`; env `EXL3_MEMT_CONTEXT_TURNS`, `EXL3_MEMT_CACHE_MB`, `EXL3_MEMT_VERBOSE`). Launched
   from `~/exl3/tabbyAPI` with flag overrides on a spare port, the user's `config.yml` untouched:

       EXL3_MEMT_VERBOSE=1 ./venv/bin/python main.py --model-name /mnt/two/weights/MemT-1.0-Nano-exl3-6.00bpw \
         --draft-mode disabled --max-seq-len 32768 --cache-size 32768 --gpu-split 20 23 --port 5055 \
         --tool-format auto --muse-glimmer false --reasoning-start-token "<think>" --reasoning-end-token "</think>"

   Loads in 10-40 s (19.4 + 7.7 GiB). A stateless OpenAI client resending full history
   (`/v1/chat/completions`, streaming and not) recalls four facts from three earlier turns, rolls back
   on an edited turn, and the log shows one commit per new turn (0.12-0.22 s; the very first forward
   through the memory path in a fresh process costs ~8 s of warmup, once).

Gotchas seen: the `muse_glimmer` response parser from the user's config returns null content for
Qwen output (overridden with `--muse-glimmer false --tool-format auto`); thinking is on by default
under the current config, so a request needs `"template_vars": {"enable_thinking": false}` or a
large budget, otherwise the whole budget is reasoning and `content` is null.

Open from section 3, unchanged: commit-every-turn vs the release's raw-until-asked policy, replies
committed as the client re-renders them (no reasoning), the per-token commit scan for long documents,
decode off the graph path while a state is attached, disk persistence of snapshots.
