# KV streaming: fast-path, span copies and auto placement — work report

Branch: `kvStreaming` (MVP committed as `911b6b44`, everything below is on top of it, uncommitted at
the time of writing). Design reference: `doc/kv_streaming.md` (§11 documents the mechanisms);
environment variables: `doc/env_vars.md`.

The work targets the case KV streaming exists for: a model whose KV cache does **not** fit in VRAM.
The local Qwen3.5-0.8B / 24 GB box is only a test harness — it can hold every byte of a 64k-token KV
(789 MiB), so it is exactly the configuration where streaming must be free, and it is the worst case
for a slow streaming implementation because there is nothing to hide behind.

Three improvements were selected from the "worth stealing" list, plus a regression that turned up
while measuring them:

1. a stale `_wrote_upto` made every step refill the whole resident window (found and fixed)
2. all-resident decode runs the stock `EXL3_BC_ATTN` graph over the resident window
3. H2D/D2H copies coalesce into contiguous runs instead of one copy per page
4. the resident window is sized automatically to fill the free device memory at first use

Item 4 of the original list (speculative decoding sharing the ring, a weight/workspace/KV arena) was
left out of scope, as agreed.

---

## 1. Regression: `_wrote_upto` kept a stale maximum

**Symptom.** In an all-resident configuration (`EXL3_KV_STREAM_RESIDENT=128p`, 8k context) decode
ran at 142 tok/s against 407 tok/s for the stock cache — a 65% loss, worse than streaming a window
that must actually cross PCIe. Every token was slower than the configuration that moves 300+ MiB per
step.

**Root cause.** `_write_append()` advanced the publication ledger with

```python
self._wrote_upto = max(self._wrote_upto, b)     # wrong
```

`_wrote_upto` is not a high-water mark: the next call's `a` must equal it to continue the same
sequence. Any sequence that starts *below* the previous sequence's end — a new job, a shorter prompt,
prompt-cache reuse, a rollback — leaves the stale maximum in place, so `a != self._wrote_upto` stays
true for the whole sequence. The driver treats a position mismatch as a new sequence: it bumps the
epoch and invalidates every slot owner, and the following `_ensure_state()` then refills the entire
resident window from the host mirror — on **every layer, every step**.

**Instrumentation.** `/tmp/opencode/kv_diag.py` counted per call: 318/318 calls took the epoch-bump
path, ~31 pages were refilled per call, 6 layers × ~31 pages × 512 KiB ≈ 93 MiB of mirror→slot
copies per step, 7.0 ms/step.

**Fix.** Assign the plain end (`self._wrote_upto = b`) and let `_account_rollback(a)` clamp the
publication ledger for rejected rows. The epoch now changes once per real sequence change, and the
steady state needs no refill at all (`_valid_from` short-circuits the per-page scan; stale pages are
pruned as they are verified). 7.0 ms/step → 2.5 ms/step on the same configuration.

## 2. All-resident decode on the stock graph path

While the whole live history fits the resident window (`L <= R * PAGE_SIZE`), the window *is* a plain
page-major KV pool: token page `i` lives in slot `i`. That is the stock cache layout, so the stock
captured attention block applies unchanged.

**How.**

- `StreamLayer.bc_attn_step()` (new) mirrors the eager driver's prologue — rollback accounting, epoch
  check, `_wait_publish()`, `_ensure_state()` — then calls the stock `BCAttn.step()` with an
  **identity block table** (`[0 … R-1]`, int32, width `R`, built once per layer) so the captured
  update kernel and split kernels address the resident slots. The append length reaches the graph
  through the same 1-element device tensor the driver uses (`_stage` filled with the pre-append
  length `a`), keeping the "driver state is written device-side, stream-ordered" rule.
- The **graph appends the rows itself** (in-graph cache update, projections and rope included), so
  after the step the layer only has to do what streaming adds: `_publish_append(a, L)` sets the slot
  owners and publishes rows `[a, L)` to the host mirror on the copy stream, fenced by `_pub_event`.
  The mirror stays authoritative and the next step's `_ensure_state()` sees a consistent state.
- `Attention.bc_attn_step()` delegates when the resolved layer is a `StreamLayer` (in `modules/attn.py`),
  so the decision is made after layer resolution and nothing else in the decode path changes.
- `build_bc_stream(module, layer)` declines, falling back to the chunked driver for that step (the
  same layer can alternate between the two paths across steps):
  - modules the stock builder declines (sliding window, logit softcap, `cu_seqlens` batching,
    asymmetric V head dims, TP, …)
  - QSA indexer modules (their side planes do not exist on a streaming layer)
  - quantized layers with `compand_a != 0` (the in-graph update kernel hardcodes zero companding)
  - steps that are not `fp16`/contiguous/`bsz == 1`, non-causal callers, or `L > R * PAGE_SIZE`
    (window slid)
  - callers without a pinned host `cache_seqlens`/`block_table` (a device→host sync inside the step
    would stall the pipeline; the generator always supplies pinned copies)
- `layer._bc_stream` caches the built graph (or `False` for declined) and is invalidated in
  `alloc()`/`free()`.

## 3. Span copies instead of per-page copies

`_runs(p0, p1)` splits a page range into runs whose mirror pages *and* destination slots are both
contiguous; `_copy_pages_to_ring()` (H2D, prefetch and inline fills) and `_publish_rows()` (D2H) issue
one copy per component per run. A sequential pool with a sequential block table collapses a whole
chunk into a single copy; a publish is a partial head page, one full-page run, and a partial tail
page. The publish fence also reuses one marker event per layer (`_ev_mark`) instead of allocating a
fresh `torch.cuda.Event()` per publish (a wait snapshots the event's current record, so re-recording
is safe).

## 4. Auto placement: fill free VRAM, stream only the excess

`EXL3_KV_STREAM_RESIDENT=auto` (the default) now defers the window size to **first use**, because at
`alloc()` time the weights are not on the device yet and the free-memory reading is meaningless.

`_ensure_resident()` (called from the first forward, when every cache layer of the model exists)
computes per layer:

```
budget = free_device - ring (S*C pages) - min(2 * n_layers * 64 MiB, free/4) - max(512 MiB, free/16)
pages  = budget // (per_page_bytes * n_layers)      # clamped to [1, capacity_pages]
```

- the ring term is `S * C` pages per layer (shared ring, per layer admitted);
- the partials term reserves the decode and prefill walk scratch, each bounded by a 64 MiB budget per
  layer, capped at a quarter of the free memory so a huge per-page model cannot starve the window;
- headroom (`512 MiB` or a sixteenth of free) covers workspace and allocator fragmentation;
- `n_layers` is counted by `_stream_config()` as the layers of a cache build their configs, so the
  memoized config cannot undercount.

Explicit `EXL3_KV_STREAM_RESIDENT` values (pages or bytes) keep their old meaning and are resolved at
load time as before.

---

## Verification

| suite | result | coverage |
|---|---|---|
| `tests/cache/test_kv_stream.py` | 30 passed | driver vs stock wrappers per placement (fp16/quant, window, ring, prefetch, serial/overlap), permuted block tables, sequence reuse and rollback, `copy_page`/`rotate_pages`, placement policy incl. the new auto budget arithmetic |
| `tests/e2e/test_kv_stream.py --slow` | 7 passed, 21 skipped | whole-model logits vs stock within the model's own noise floor, fp16 and 8-bit cache; all-resident case (asserts the auto window and that the graph path was taken); resident→streamed transition case (window covers the prefill, slides mid-decode, asserts both paths ran and logits still match) |
| `tests/e2e/test_decode_consistency.py`, `test_prompt_cache.py --slow` | 5 passed, 80 skipped | stock regressions (skips are models not present locally) |

Correctness is asserted per placement at the driver level and end-to-end at the model level; the
all-resident graph path is checked against the stock run of the same model, not only against the
chunked driver.

## Measurements

RTX 3090 (24 GB, PCIe 4.0 ×16 ≈ 26.8 GB/s pinned, measured effective streaming bandwidth ≈ 19–21 GB/s).
Qwen3.5-0.8B exl3 4 bpw, 6 full-attention layers; fp16 KV: 512 KiB/page/layer (12 KiB/token), 8-bit
KV: 272 KiB/page/layer. Cache geometry: 263 pages (67k tokens). Decode = greedy, median over the last
48 steps of a warm run; prefill = second run of a same-shape job. Source: `/tmp/opencode/kv_bench.py`,
results in `/tmp/opencode/kv_bench_v4.json`.

Measured with the machine idle. An earlier run of the same sweep was discarded: a game started on
the box during it (GPU at 100% utilization, power-capped, 85 °C), which halved the streaming
variants' effective bandwidth (20.3 → 11.5 GB/s) and even slowed the stock variants (`stock_q8` at
32k: 2.9 → 6.6 ms/step) — contamination, not a regression. That run is kept at
`/tmp/opencode/kv_bench_v3_CONTAMINATED_shared_gpu.json`.

### Decode throughput (tokens/s)

| variant | 8k | 32k | 64k | notes |
|---|---|---|---|---|
| `stock_fp16` | 377.5 | 357.9 | 293.4 | full pool in VRAM (789 MiB), reference |
| `stream_auto` | 374.2 | 351.0 | 293.4 | window = capacity (263 pages), 6/6 layers on the graph path |
| `stock_q8` | 375.6 | 304.0 | 274.3 | 8-bit cache, reference |
| `stream_auto_q8` | 331.7 | 299.8 | 263.4 | same window; repeat runs at 8k: 400–404 vs 398–400 tok/s (parity) |
| `stream_full128` | 389.3 | 292.6 | 48.4 | 128-page window; slides at 64k → 384 MiB/step |
| `stream_r16` | 199.0 | 55.7 | 27.3 | 16-page window; 720 MiB/step at 64k |
| `stream_r2` | 153.6 | 50.8 | 26.1 | 2-page window; 762 MiB/step at 64k |
| `stream_r2_q8` | 207.7 | 72.5 | 36.1 | same window, half the bytes per step |

### Decode latency (ms/step, median)

| variant | 8k | 32k | 64k |
|---|---|---|---|
| `stock_fp16` | 2.65 | 2.79 | 3.41 |
| `stream_auto` | 2.67 | 2.85 | 3.41 |
| `stock_q8` | 2.66 | 3.29 | 3.65 |
| `stream_auto_q8` | 3.01 | 3.34 | 3.80 |
| `stream_full128` | 2.57 | 3.42 | 20.67 |
| `stream_r16` | 5.03 | 17.96 | 36.69 |
| `stream_r2` | 6.51 | 19.68 | 38.36 |
| `stream_r2_q8` | 4.81 | 13.80 | 27.72 |

### Streamed H2D traffic per decode step

| variant | 8k | 32k | 64k |
|---|---|---|---|
| `stream_auto`, `stream_auto_q8` | 0 | 0 | 0 (everything resident) |
| `stream_full128` | 0 | 0 | 384 MiB @ 19.5 GB/s |
| `stream_r16` | 48 MiB @ 10.0 GB/s | 336 MiB @ 19.6 GB/s | 720 MiB @ 20.6 GB/s |
| `stream_r2` | 90 MiB @ 14.5 GB/s | 378 MiB @ 20.1 GB/s | 762 MiB @ 20.8 GB/s |
| `stream_r2_q8` | 48 MiB @ 10.4 GB/s | 201 MiB @ 15.3 GB/s | 405 MiB @ 15.3 GB/s |

### Prefill (tokens/s)

| variant | 8k | 32k | 64k |
|---|---|---|---|
| `stock_fp16` | 27996 | 25589 | 15165 |
| `stream_auto` | 28795 | 22318 | 12760 |
| `stock_q8` | 29369 | 25409 | 15438 |
| `stream_auto_q8` | 21581 | 11711 | 5989 |
| `stream_full128` | 28715 | 21807 | 12625 |
| `stream_r16` | 28768 | 20960 | 12726 |
| `stream_r2` | 25062 | 20277 | 11494 |
| `stream_r2_q8` | 21225 | 12002 | 6192 |

### Memory, after the runs

| variant | device KV | ring | host mirror | peak after load | R | graph layers |
|---|---|---|---|---|---|---|
| `stock_fp16` | 789 MiB | – | – | 1487 MiB | – | – |
| `stream_auto` | 789 MiB | 0 | 789 MiB pinned | 1548 MiB | 263 | 6 |
| `stock_q8` | 419 MiB | – | – | 1239 MiB | – | – |
| `stream_auto_q8` | 419 MiB | 0 | 419 MiB pinned | 1178 MiB | 263 | 6 |
| `stream_full128` | 384 MiB | 32 MiB | 789 MiB | 1677 MiB | 128 | 6 |
| `stream_r16` | 48 MiB | 32 MiB | 789 MiB | 1789 MiB | 16 | 6 |
| `stream_r2` | 6 MiB | 32 MiB | 789 MiB | 1238 MiB | 2 | 0 |
| `stream_r2_q8` | 3 MiB | 17 MiB | 419 MiB | 1220 MiB | 2 | 0 |

The auto window is allocated lazily at first use, so these are post-run readings: 263 pages × 512 KiB
× 6 layers = 789 MiB of device KV for the fp16 cache, 419 MiB for the 8-bit one, against 789/419 MiB
for the stock caches — the same bytes, placed in the resident window and mirrored to pinned host
memory. The ring registers as 0 MiB for the all-resident runs because no chunk ever needs a ring slot.
"Graph layers" counts the layers that built the all-resident graph path at some point in the run (for
the small windows that is the 1024-token warmup, after which the window slides and the walk takes
over).

### Residual per-step overhead of the graph path

Direct A/B profile at 8k (`/tmp/opencode/kv_bcprof.py`, stock first, then stream in the same
conditions): stock 2.50 ms/step vs stream 2.42 ms/step. Host-side attribution per step (6 layers):
`_ensure_state` 0.106 ms, `_publish_append` 0.190 ms, `bc_attn_step` 0.038 ms, remaining wrapper
bookkeeping ≈0.105 ms — ~0.44 ms/step of added host work inside a ~2.4 ms step whose GPU side leaves
the host slack, which is why the decode rate does not move. The sweep agrees: 2.67 vs 2.65 ms/step
(full precision), and six alternating repeat runs at 8k give 400–404 vs 398–400 tok/s for the 8-bit
pair — parity for both cache types. Run-to-run spread on this box is a few percent (fresh process per
variant, thermals, allocator state); a single-sweep cell that looks worse than parity is noise unless
it reproduces, as the `stream_auto_q8` 8k cell shows.

### Naive offload baseline: all KV in pinned host memory

The alternative to streaming is to keep the whole KV history in system RAM and let the attention
kernels read it where it lives — no staging, no prefetch, no device window. That scheme is easy to
build on top of the streaming layer: point its "resident" window at pinned host tensors and size it to
the full capacity, so nothing is ever copied to VRAM (`/tmp/opencode/kv_naive_offload.py`, raw output
in `/tmp/opencode/kv_naive_offload.log`). Linux/UVA makes pinned host memory directly addressable by
the kernels, so this is the *best case* for the naive scheme; pageable memory — what a regular
offload/mmap path would use, and what the llama.cpp fork measured at 2.5 vs 26 tok/s at 128k — is much
slower.

Decode, ms/step (tok/s):

| variant | 8k | 16k | 32k | 64k |
|---|---|---|---|---|
| naive offload (KV stays in host RAM, read in place) | 7.4 (134.9) | 11.4–12.2 (82–88) | 21.2 (47.2) | 39.8 (25.1) |
| naive offload, BC graph reading host memory | 6.7 (149.2) | 11.7 (85.4) | – | – |
| streaming, 2-page window | 6.6 (150.6) | 11.9 (84.3) | 18.9 (53.0) | 37.8 (26.4) |
| streaming, auto window | 2.7 (366.6) | 2.8 (352.0) | 3.1 (320.1) | 4.0 (251.9) |

Prefill, time to first token (ms):

| variant | 8k | 16k | 32k | 64k |
|---|---|---|---|---|
| naive offload | 1400 | 5077–5335 | 19250 | 69936 |
| streaming, 2-page window | 352 | 811 | 2026 | 5723 |
| streaming, auto window | 305 | 693–725 | 1817 | 5533 |

Where it lands:

- **Decode at the same near-zero-VRAM budget is a wash.** 7.4 vs 6.6 ms/step at 8k, 39.8 vs 37.8 at
  64k. Once the bytes have to cross PCIe every step, the link is the wall for both: the naive in-place
  reads reach 12.9–19.3 GB/s against the streaming walk's 20.0–20.8 GB/s. The streaming machinery does
  not multiply link bandwidth by itself.
- **Prefill is 4× to 12× faster** (1.4 s vs 0.35 s at 8k, 70 s vs 5.7 s at 64k), and the gap grows
  with context: the naive scheme re-reads the history from host memory once per query block, so its
  host traffic scales with context length *and* prompt length, while streaming copies each page once
  into the ring and every attention re-read lands in VRAM.
- **The resident window is the real difference.** The naive scheme cannot keep any KV in VRAM by
  construction; streaming keeps whatever fits (here: everything), so nothing crosses PCIe at all:
  2.7–4.0 ms/step against 7.4–39.8, i.e. 2.7× faster than the naive baseline at 8k and ~10× at 64k,
  with prefill 6–13× faster.
- On a model whose KV exceeds VRAM, the same streaming layer keeps the fraction that fits in VRAM for
  free and moves only the overflow at ~20 GB/s. The naive scheme has no such dial.

## What was achieved

- **All-resident decode is free.** With the window covering the history, decode runs the stock
  captured block: measured parity with the stock cache — 2.67 vs 2.65 ms/step at 8k and 3.41 vs
  3.41 ms/step at 64k on the fp16 cache, 2.5 vs 2.5 ms/step in alternating repeat runs at 8k on the
  8-bit cache. Only the mirror publish is left as streaming-specific work. Before the `_wrote_upto`
  fix the same configuration ran 7.0 ms/step; with the fix alone, 2.5 ms/step; the graph path keeps it
  there without the eager walk's launch overhead.
- **"Fill VRAM, stream the excess" works.** `auto` resolves to the largest window the free device
  memory can hold (263 pages = 67k tokens here), so nothing crosses PCIe for any context this box can
  produce, and the same policy on a model whose KV exceeds VRAM keeps everything that fits resident
  and streams only the overflow. The policy is lazy, so it sees the post-load free memory rather than
  the pre-load reading.
- **It beats naive offload in a head-to-head on this box.** With all KV in pinned host memory and the
  kernels reading it in place (a best case for that scheme: no page faults, large sequential reads),
  streaming is at parity to ~10% faster on decode at a 2-page window, 4–12× faster on prefill, and
  2.7–10× faster on decode with the auto window — the table above. The naive scheme's host traffic
  grows with prompt length as well as context length, so its prefill disadvantage widens as context
  grows (70 s vs 5.7 s at 64k).
- **Streaming cost is now predictable and bandwidth-bound.** When the window slides, the step cost is
  `streamed_bytes / ~20 GB/s` plus the walk: 384 MiB/step → 20.7 ms (128-page window at 64k),
  720 MiB → 36.7 ms (16 pages), 762 MiB → 38.4 ms (2 pages) — i.e. 19.5–20.8 GB/s of a ≈26.8 GB/s
  link, with the runs contiguous in both mirror and slot, so the copies are as few operations as the
  layout allows. An 8-bit cache halves the bytes: the same 2-page window streams 405 MiB/step and
  takes 27.7 ms (15.3 GB/s including the scale planes), 28% faster than the fp16 cache at the same
  window.
- **The window size is a clean capacity/speed dial**, and the default dial is now "as large as the
  hardware allows".

## Known limits and leftovers

- Prefill still runs the eager walk. On the fp16 cache it is at parity at 8k (28795 vs 27996 tok/s)
  and 16% slower at 64k (12760 vs 15165). On the 8-bit cache it is 21581 vs 29369 tok/s at 8k and
  5989 vs 15438 at 64k: the quantized walk pays for staging and dequantizing the packed pages, which
  the two-pass dequant staging follow-up below would address. Decode is unaffected.
- CUDA-graph capture of the *chunk walk* (window sliding) is not implemented; only the all-resident
  step runs the graph path. Sliding-window layers, `bsz > 1`, TP, MLA/QSA/DSA geometries and the
  CPU/disk tiers are still declined loudly (`doc/kv_streaming.md` §5).
- Deferred/batched mirror publish (a row ledger flushed at `_quiesce`/eviction) and two-pass dequant
  staging for the streamed quantized prefill (the q8 prefill cost above) remain ideas, not
  implemented.
- Speculative decoding sharing the ring (item 4 of the original list) is out of scope.

## Files touched

- `exllamav3/cache/stream.py` — fast path (`bc_attn_step`, `_identity_table`), publish split
  (`_publish_append`/`_publish_rows`), run coalescing (`_runs`, `_copy_pages_to_ring`), auto placement
  (`_ensure_resident`, `_auto_resident_pages`, `n_layers`), `_wrote_upto` fix, `_ev_mark` reuse,
  `alloc`/`free` hooks.
- `exllamav3/modules/attention_fn/bc_attn.py` — `build_bc_stream()`.
- `exllamav3/modules/attn.py` — `Attention.bc_attn_step()` delegation for `StreamLayer`.
- `tests/cache/test_kv_stream.py`, `tests/e2e/test_kv_stream.py` — policy tests, all-resident test,
  resident→streamed transition test.
- `doc/kv_streaming.md` (§5, §4.9, §10, §11), `doc/env_vars.md` — design and policy updates.

Run the checks with:

```bash
python -m pytest tests/cache/test_kv_stream.py -q
python -m pytest tests/e2e/test_kv_stream.py --slow -q
```
