# KV cache streaming (design)

**Status:** implemented behind `EXL3_KV_STREAM=1` (`exllamav3/cache/stream.py` plus the streaming launch paths in
`modules/attention_fn/triton_paged.py`); M0-M2 complete, M3 largely (prefill streaming and prompt-cache page
surgery included, see §11 for the actual scope and the deviations from this document). Target: Qwen3.5-class
hybrid models (full attention every N layers, linear attention elsewhere) on CUDA, bsz 1, decode and prefill
over contexts whose KV does not fit in VRAM.

This document adapts the *design* of the adaptive KV streaming in
[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)
(MIT) to ExLlamaV3's architecture. No code is taken from that project; it is a C++/ggml codebase and none of its
infrastructure maps onto this one. What transfers is the memory model and the scheduling discipline. Everything
below is expressed in terms of machinery this fork already has.

## 1. Problem

A quantized model whose weights fit a 16 GiB card leaves little room for KV as context grows. When the KV cache
exceeds VRAM, the options today are:

- shrink it with a lower-bitrate quantized cache (`CacheLayer_quant`), which runs out eventually, or
- let it spill through unified/managed memory, which migrates pages under demand and degrades decode by an order of
  magnitude (the llama.cpp fork measures ~2.5 tok/s at 128 Ki that way vs ~26 tok/s with explicit streaming).

Streaming keeps the authoritative KV history in pinned host RAM, keeps a chosen fraction resident on the GPU, and
moves the remainder through a small GPU ring buffer per layer as attention needs it. Attention still runs on the
GPU; the cost is PCIe traffic, not compute.

## 2. Reference design (what we adapt)

From the llama.cpp fork, reduced to the transferable pieces:

1. **Host-authoritative pages.** The complete encoded K/V history lives in pinned host RAM, split into logical
   pages of 256 token positions. Page bytes follow the K/V encoding (quantized formats included).
2. **Resident set + shared ring.** GPU memory holds a configurable subset of pages plus one ring of reusable
   slots. The ring is shared across layers, not carved up per layer.
3. **Bounded attention spans.** Attention consumes the history in chunks, split and resumed without changing the
   reduction order, so streaming is numerically a no-op against the resident path.
4. **Prefetch with ready/consumed ordering.** H2D copies fill ring slots ahead of use (ready fence); slots are
   reused only after their last reader completed (consumed fence). Copied volume is the non-resident fraction;
   newly written rows are published back D2H, and unchanged pages are never copied twice.
5. **Placement policy.** As context grows, pages are moved between resident and streamed placement, and prefetch
   lookahead follows the next actual consumer (including draft phases) rather than a fixed distance.

The fork's shared "phase arena" (prefill/decode/vision/draft scratch reusing one device parent) is *not* part of
this design: it exists because ggml allocates per-phase workspaces. This fork's vision tower already offloads to
CPU here, and the text phases' workspaces are managed per module. Nothing in KV streaming requires that arena.

## 3. Existing machinery this design builds on

| Piece | Where | What it gives us |
|---|---|---|
| Paged cache, 256-token pages | `exllamav3/constants.py` (`PAGE_SIZE`), `exllamav3/cache/cache.py` (`CacheLayer`, `Cache`), `cache/fp16.py`, `cache/quant.py` | Logical pages, page tables, and storage that is already a small pool of page tensors |
| Block-table addressing in every kernel | `modules/attention_fn/triton_paged.py`, `modules/attn.py` | Kernels read physical pages through the block table; where a page lives is the table's problem, not the kernel's |
| Packed quantized KV, byte-addressable | `cache/quant.py` (`qk/sk/qv/sv`), Triton `_qc_load_kt`/`_qc_load_v` | A host mirror can store the same packing; H2D/D2H are raw byte copies at quantized bitrate |
| Flash-decoding split + combine | `triton_paged.py` `_paged_attn_decode_split_kernel` (~L890) and `_paged_attn_decode_combine_kernel` (~L1047) | Per-KV-range partial accumulators (`acc`, `m`, `l`, fp32) and an ordered, split-count-agnostic reduction — the merge primitive for chunked streaming |
| Runtime launch parameters | same kernels: `split_len`, `num_splits`, `num_pages_per_seq` are runtime ints; the graph path already launches wider than needed with idle splits | Chunk geometry can change without recompiles/graph recapture |
| Bounded-window staging | `triton_paged.py` (~L1910) quantized prefill staging: dequantize the *referenced window* into a fixed scratch, remap the block table to `arange` | A near-prototype of "materialize a bounded span, point the kernel at it"; the swap is source (host copy) instead of dequantize-in-place |
| Bounded KV range per program | `_paged_attn_prefill_inner` takes `n_start`/`n_end` per program | The prefill kernel already consumes a KV sub-range; streaming prefill is the same chunk walk |
| Quant-direct decode | `cache/quant.py` `update_kv_direct`, `get_qkv`; `attention_fn/dispatch.py` (~L126) | New rows are quantized straight into the packed cache; no fp16 layer tensors to mirror |
| Pinned host memory | `util/pinned_arena.py`, `util/device_copy.py`, pinned rings in `exllamav3_ext/stloader.h` | Host mirror allocation and async copy conventions already exist |
| Python layer loop | `model/model.py` `forward` (~L445) | Where cross-layer prefetch scheduling plugs in |
| Hybrid layer maps | `architecture/qwen3_5.py` (`read_qwen3_5_layer_types`, `full_attention_interval`) | Only `full_attention` layers carry a growing KV; linear/recurrent layers keep bounded state (`cache/recurrent.py`) and are untouched |

## 4. Design

### 4.1 Memory model

Per model, with `L` = number of full-attention layers, `P` = `PAGE_SIZE` = 256:

- **Host mirror.** One pinned buffer per layer-shaped KV component (`k`/`v` fp16 layers; `qk/sk/qv/sv` packed
  layers), same page geometry as today's `CacheLayer` tensors, spanning the full logical capacity
  (`max_num_tokens`). Authoritative for every page that is not currently mutable on the GPU.
- **Resident set.** A GPU page pool of `R` pages per model, addressed exactly like today's cache tensors. Pages in
  the set are read directly by the kernels; no copies happen for them.
- **Ring.** A GPU buffer of `S` slots, one *chunk* each, shared across layers. A chunk is a whole number of pages
  (`C = S_chunk` pages, ring total `S * C` pages). The ring is only ever read by one layer's attention at a time
  (layers run sequentially), so it needs no per-layer partitioning.
- **Tail pin.** The page(s) currently being appended to are always resident and are never scheduled for H2D. They
  are mutable on the GPU; the host mirror for those pages is only authoritative up to the last published length.
- **Formats.** The ring and resident set use the layer's native storage: fp16 pages for `CacheLayer_fp16`, packed
  `qk/sk/qv/sv` for `CacheLayer_quant`. The host mirror uses the same layout. Nothing is dequantized for
  transfer; dequantization stays where it is today (in-kernel or via staging).

Byte copies therefore move exactly the bytes the quantized cache stores — Q8/Q4-style bitrates halve PCIe traffic
against fp16 mirrors.

### 4.2 Addressing: logical vs. physical page tables

The generator already owns a logical block table (`seq.block_index_tensor`, filled by page allocation in
`generator/generator.py` / `job.py`). Under streaming:

- The generator's table keeps its meaning: **logical** page ids for positions, used for allocation, prompt-cache
  hashing, `copy_page`, etc.
- Each streaming layer derives a **kernel table** — its own device copy, width = pages of the live context — whose
  entries are *physical* placements: resident-set index, ring slot (per chunk), or tail slot.
- `update_kv_direct` and the attention calls use the kernel table. The stream layer owns it; callers never see
  physical ids.

For MVP this is one table per streaming layer per sequence, rebuilt only when placements change (context growth,
chunk switches, or page set changes), and patched per chunk in place.

### 4.3 Decode attention: chunked split/combine

This is the core of the design, and it reuses the existing flash-decoding split path almost as-is.

For one streaming layer at decode (`q_len` ≤ 16, `bsz` = 1):

```
resident_prefix_pages  = pages present in the resident set
stream_chunks          = the remaining pages, in order, grouped into chunks of C pages
tail_pages             = pages holding the rows appended this step (resident)

attention(q, layer_kv):
    published = min(published_len, resident/streamed span)   # host mirror is valid up to here
    place chunk 0 for H2D; issue async copies (copy stream)
    for i, chunk in enumerate(stream_chunks):
        wait ready(chunk i)                                  # event recorded after its H2D copy
        patch kernel table entries for chunk i's pages -> ring slot(i mod S)
        launch _paged_attn_decode_split_kernel over KV range of chunk i
            (split_base = chunk i's first KV position, split_off = first split slot to store to)
        record consumed(slot(i mod S))                       # after the kernel launch
        issue H2D for chunk i+1 into slot((i+1) mod S)
            after waiting consumed(slot((i+1) mod S))
    launch _paged_attn_decode_combine_kernel over all splits
    restore kernel table entries if needed
```

Why the split path is the right lever, in exllamav3 terms (`triton_paged.py`):

- The split kernel already writes unnormalized accumulators per (program, split): `acc` fp32, plus `m` and `l`,
  into `partial_o` / `partial_ml`. The combine kernel reduces any split count in fixed order
  (`sidx = 0..num_splits`), masking idle splits. Splitting the streamed range across launches is the same
  arithmetic the kernel already does across in-tree splits: results keep the stock reduction order, so streamed
  and resident runs agree numerically up to the usual kernel determinism.
- `split_len` and `num_splits` are runtime ints; the kernel comments at ~L904 note the graph path already
  launches more splits than needed with extra splits idling. Adding two runtime ints — a KV-position base and a
  partial-store slot base — is a minimal, non-intrusive patch. With both at 0 the kernel behaves exactly as
  today, so the existing paths are untouched.
- `partial_o` / `partial_ml` are currently allocated per call. For streaming they become persistent per
  streaming layer (sized for the worst case split count), which also improves the graph story later.

Ring void cases fall out naturally: all-resident layers (short context) take the existing unchanged path;
all-streamed context runs one chunk walk with no resident entries; the tail stays resident and causal masking
handles the appended rows exactly as today.

Trade-off to tune: ring size vs split parallelism. Larger chunks = fewer transfers and fewer combine inputs, but
fewer concurrent split programs per launch. The existing `attn_decode_config` ladder picks tile shapes; the ring
sizing pass should pick `C` so that the split grid retains ~2×SM programs for the model's head geometry.

### 4.4 Prefill

Prefill cannot dodge streaming for very long prompts: for each query chunk and layer, attention reads all KV so
far. Two mechanisms already point the way:

- `_paged_attn_prefill_inner` runs a KV range `[n_start, n_end)` per program, and the prefill split/combine pass
  (`IS_SPLIT` around ~L1574) reduces q-tile partials. The same chunk walk as decode applies, with the prefill
  kernel and its combine.
- The quantized-cache staging path (~L1910) dequantizes a bounded window into a fixed scratch and hands the
  kernel an identity block table. Under streaming the scratch becomes the ring, the source becomes the host
  mirror, and fp16 layers skip dequantization entirely: copy pages into the ring, patch the table, launch.

Prefill has better arithmetic intensity than decode — each streamed byte feeds many queries — so the PCIe cost is
amortized and the results should scale closer to today's prefill numbers than to decode's transfer bound.

Ordering note: prefill chunks append KV as they go. Newly appended pages of the *current* q chunk must be
published to the host mirror before they can be evicted from the resident set, and pages referenced by later q
chunks stream as usual. The page lifecycle states in 4.6 cover both.

### 4.5 Publish and eviction

- After each layer writes its step's rows (`update_kv_direct` → `ext.quant_cache_paged` /
  `ext.paged_kv_cache_update`), issue an async D2H of the affected tail page rows to the host mirror on the copy
  stream and advance `published_len` when it lands (event).
- A page may leave the resident set / tail **only after** its latest contents are published. Mutable pages
  (current tail) are never eviction candidates.
- Rollback (speculative rejection truncates `cache_seqlens`): rows beyond the new length may contain draft junk in
  GPU memory and previously published bytes in the mirror. Validity is governed by `published_len`, which moves
  back with truncation; subsequent H2D of the page serves rows below `published_len` only, and causal masking
  bounds reads at `cache_seqlens`, so junk never reaches attention. Implementation must keep both invariants
  explicit.

### 4.6 Page states

Each logical page of each streaming layer is in exactly one state:

```
HOST_ONLY   authoritative in the mirror; no GPU copy outside the ring
RESIDENT    live GPU page in the resident set, published and clean
TAIL        live GPU page being appended to (may be dirty); published up to published_len
IN_FLIGHT   H2D copy into a ring slot scheduled or running (not yet readable)
RING        H2D complete, readable by the current chunk's kernel
CONSUMING   kernel launched over the ring slot; slot not reusable until its consumed event fires
```

Transitions are event-ordered copies: HOST_ONLY → IN_FLIGHT → RING → CONSUMING → HOST_ONLY (or → RESIDENT when the
placement policy promotes it). TAIL → RESIDENT when the page closes and R→S placement changes happen between
steps, never inside a chunk walk.

### 4.7 Synchronization

- One **compute stream** (the default stream the layer loop runs on) and one **copy stream** for H2D/D2H. RING
  slots are guarded by two events each: *ready* (recorded on the copy stream after the H2D) and *consumed*
  (recorded on the compute stream after the last kernel that reads the slot).
- Cross-layer lookahead: while layer L's attention runs, layer L+1's first chunk H2D may already be in flight,
  provided the copy volume fits the time budget. This is the "next consumer" discipline from the reference
  design; MVP may start with next-chunk lookahead only and add next-layer lookahead in M2.
- No host syncs in the token loop. Every decision uses device-side counters or Python-side known lengths. The
  existing decode path's `max_kv_len` hint (avoids reading `cache_seqlens` on the host) is the pattern to keep.

### 4.8 Placement policy

- Initial placement: first `R` pages resident (prefix), rest streamed. Kept deliberately simple; a policy like
  "most recently published resident" does not help full-attention layers, which read the whole history every
  step.
- Budget arithmetic: mirror bytes ≈ logical capacity × per-page bytes; GPU bytes ≈ `(R + ring) × per-page
  bytes`; PCIe traffic per token ≈ `L × (streamed fraction) × per-layer KV bytes`. Given a VRAM target and a
  capacity target, the ring size and resident fraction fall out. A tuning helper can report the modeled tok/s
  ceiling from measured copy bandwidth.
- Growth: when context grows past the resident budget, demote pages (publish, then release) rather than growing
  the resident set. When the model's allocator reports headroom (e.g. after vision offload returns memory), the
  reverse promotion can happen between steps.

### 4.9 Integration points

- **Cache layer.** New `CacheLayer_stream` (subclass of the fp16 or quant layer, or a wrapper around it)
  implementing the existing `CacheLayer` interface plus the streaming surface:
  `stream_attention(q, k, v, params) -> o`, `publish()`, `evict/promote()`, `kernel_table`. Its `alloc()` only
  allocates resident + ring + mirror; `storage_size()` reports host and GPU parts separately so the loader's
  accounting stays honest. `copy_page` (prompt caching) operates on logical pages and must go through the mirror
  when either side is host-only.
- **Dispatch.** `attn_dispatch` / `Attention.decode_flash_attn` gain a branch: if the resolved cache layer is a
  streaming layer, call the chunked driver instead of the plain dispatch. Everything upstream (projections, RoPE,
  norms) and downstream (output projection, gates) is unchanged. The driver delegates to
  `paged_attn_triton_decode`-compatible launches (patched kernels) so tie-in points stay in one place.
- **BC graph path.** `EXL3_BC_ATTN` captures fixed-shape graphs with cache pointers baked in. Streaming layers
  decline it initially (`build_bc_attn` returns None for them). A later milestone can capture the chunk loop with
  static ring addresses and the existing idle-split trick.
- **Recurrent/linear layers.** Untouched: `cache/recurrent.py` state is bounded, no pages, no streaming.
- **Qwen3.5.** Only `full_attention` layers stream. With the default `full_attention_interval = 4` that is a
  quarter of the layers, the same topology class the reference implementation qualifies.

## 5. Non-goals (MVP)

- MLA / QSA / DSA cache geometries (different packing and access patterns; separate design).
- Tensor parallelism and multi-GPU.
- `bsz > 1` and continuous batching (ring shared across sequences; per-sequence chunk walks).
- The CUDA graph decode path.
- ROCm-specific tuning (Triton and the copy paths should work; bandwidth profiles differ).
- The vision tower: already CPU-offloaded in this fork and unaffected.

## 6. Numerics and testing

- **Parity tests.** Attention outputs and logits, resident vs streamed, at page boundaries and non-boundaries,
  several context lengths around ring/`PAGE_SIZE` edges, fp16 and at least one quantized cache format. The
  combine already walks splits in order, so expected differences are kernel-determinism only.
- **Rotation tests.** Force eviction/repromotion mid-sequence, including a page that was published, evicted,
  streamed back, and appended to.
- **Rollback tests.** Speculative accept/reject at chunk boundaries; assert `published_len` behavior and that
  post-rollback H2D serves the accepted prefix.
- **Debug mode.** A serialized mode (single stream, `S = 1`, synchronous copies) for bisecting sync bugs, plus
  an env toggle to log chunk/slot/event transitions.
- **Budget tests.** Allocation accounting vs configured VRAM targets; no growth beyond the ring across a long
  decode.

## 7. Performance model

Per decode token, PCIe traffic ≈ `L × streamed_bytes_per_layer` (K+V, packed). With a Q8/Q4 cache the reference
implementation lands at ~9.6–26 tok/s at 128–256 Ki on a 16 GiB card; expect the same order here, bounded by
device copy bandwidth over the streamed fraction. Short contexts (fully resident) pay nothing. The dial is the
resident fraction: more resident = faster, less capacity. If the whole KV fits, streaming should be a no-op that
never activates.

The new cost center is host polling of events and the per-chunk launch overhead; chunk sizes and prefetch depth
exist to keep the copy engine saturated while the attention kernels consume. That tuning is milestone M2's whole
purpose.

## 8. Staged roadmap

- **M0 — Skeleton and parity.** `CacheLayer_stream` with host mirror and a fixed resident set, placements only
  static, no ring streaming. All layers run the existing attention paths through the kernel table. Env vars and
  budget plumbing. Parity tests pass. (This alone is a usable "resident + host parking" mode for batch inactive
  sequences.)
- **M1 — Decode streaming, no overlap.** Chunk walk over the split kernel with the two new runtime ints; serial
  copies; persistent partial buffers. Correctness first: parity, rotation, rollback tests.
- **M2 — Overlap and prefetch.** Copy stream, ready/consumed events, double-buffered ring, next-layer lookahead.
  Benchmark harness: sweep resident fraction, ring size, prefetch depth.
- **M3 — Prefill streaming.** Bounded-range prefill splits plus the staging generalization; publish path for
  prefill-appended pages; prompt-cache and `copy_page` correctness.
- **M4 — Extensions.** `q_len > 1` speculative decoding hardening, sliding-window layers, `bsz > 1`, quantized
  format matrix, CUDA-graph variant, TP.

## 9. Risks and open questions

- **Sync bugs are the whole risk.** Event ordering mistakes surface as rare wrong-token corruption. Mitigations:
  the serial debug mode, invariant asserts behind an env toggle, stress tests that mix streaming with rollback
  and eviction.
- **PCIe contention with weight streaming.** The `stloader` pinned rings also move weights; a queueing policy
  between the two is needed so KV prefetch does not starve weight loads (and vice versa during model load).
- **Pinned host budget.** The mirror is large (full logical capacity). On systems where the vision CPU offload
  already uses pinned memory, total pinned usage must stay visible and bounded (`util/pinned_arena.py`
  accounting).
- **Loader accounting.** The loader's autosplit and fit logic measures cache allocations via `storage_size()`;
  host-side bytes need representation there or the fit heuristics will misjudge.
- **Prompt cache semantics.** Page hashing and reuse assume the cache tensors are complete; with streaming they
  are not. `job.py`'s page-hash path must hash mirror content (it is authoritative), which needs the publish
  ordering to be airtight.
- **Determinism across sessions.** If the resident fraction or ring size changes between runs, split counts
  change; results stay valid but bit-identical comparison across configs is not a promise (same caveat the
  reference implementation documents).

## 10. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `EXL3_KV_STREAM` | `0` | Enable streaming cache layers |
| `EXL3_KV_STREAM_MIRROR` | auto | Host mirror budget (e.g. `8G`). Reserved: the mirror is allocated for the full logical capacity and a smaller budget only warns |
| `EXL3_KV_STREAM_RESIDENT` | auto | Resident window size, in pages (`12` / `12p`) or bytes (`1G`); per full-attention layer, default `min(capacity, 16)` pages |
| `EXL3_KV_STREAM_RING` | `64` | Ring size in pages (shared across layers) |
| `EXL3_KV_STREAM_PREFETCH` | `2` | Ring slots in flight (double buffering and beyond) |
| `EXL3_KV_STREAM_SERIAL` | `0` | Serialized debug mode (no copy stream, no overlap) |
| `EXL3_KV_STREAM_TRACE` | `0` | Log chunk/slot/event transitions per step |

Documented in `doc/env_vars.md` ("KV cache streaming").

## 11. Implementation notes (as built)

The code follows the design above, with these deviations and clarifications. `exllamav3/cache/stream.py` is the
whole streaming layer, the driver being `StreamLayer.stream_attn()`; the rest of the tree only knows about it
through `Attention.cache_layer_type()` (mapping a requested fp16/quant layer to `CacheLayer_stream` /
`CacheLayer_stream_quant`), the `decode_flash_attn` branch that calls `stream_attn()`, an error in
`attn_dispatch` for any other caller, and guards against combining streaming layers with the CPU page tier or
with tensor parallelism.

**Placement is a suffix window, not a prefix.** The resident set holds the newest `R` pages
(`token page i -> slot i % R`), not the first `R` pages. This subsumes the tail pin (the pages being appended to
are resident by construction) and removes the need for promotion: pages enter the window as they are written and
leave it (remain published in the mirror) as the context grows past `R` pages. It also means a step never mixes
"new pages" into a streamed chunk: `stream_attn()` sub-chunks an append so that the pages it touches stay within
the window.

**The kernel table is a pure function of the live page count.** Placement does not depend on which physical pool
page a logical page maps to: streamed page `i` lives in ring slot `(i // C) % S * C + i % C`, resident page `i`
in slot `i % R`. The table therefore only changes when the live length crosses a page or window boundary, and
the block table is consulted only for the mirror's own indexing (below).

**The mirror is indexed by pool page.** `mirror[pool_page]`, exactly like the stock paged tensors, so
`copy_page`/`rotate_pages` (prompt caching, which operate on pool pages through `cache.py`/`pagetable.py`) stay
byte-copy moves between mirrors, and the layer keeps a `_tok2pool` list (rebuilt per step from the host copy of
the block table) to translate live token pages to mirror pages. `_pub_rows[pool_page]` tracks how many rows of
each pool page are valid in the mirror; `copy_page` sets it directly, `rotate_pages` permutes it with the pages.

**Refill validity is explicit.** A resident slot holds `(epoch, token page, pool page)` in `_slot_owner`;
anything else (a new sequence, a rollback, a rotation, a page that changed its pool mapping) invalidates the slot
and refills it from the mirror before the step's write. Rollbacks move `_wrote_upto` back and bump the epoch, so
post-rollback reads serve the accepted prefix.

**Driver state is written device-side and stream-ordered.** The append length reaches the kernels through a
1-element device tensor (`fill_()`), and the kernel table is built on the device from `torch.arange`/`torch.where`
(`_kt_host` mirrors it for Python-side lookups only). Pinned staging plus async H2D was the first
implementation and it raced: the host runs ahead of the stream and can rewrite the staging buffer before the DMA
reads it. Everything a kernel reads is now either a device tensor or written through the compute stream.

**Ring creation is ordered against the H2D copies.** The ring's one-time zero fill runs on the compute stream of
whichever layer creates the registry entry, while the copies that fill the ring run on the layers' copy streams:
nothing ordered the two, and a fill that lands late zeroes the slots the first copies just wrote. That reads as
"the first walk that touches a fresh ring computes from zeroed K/V": deterministic wrong outputs, only on the
first use per ring, and only under host-side contention. `_ring_entry()` records the fill; every layer's copy
stream waits that event before its first copy. Together with the staging fix this closes the "first walk after
load" failures seen in the contention repro.

**Quantized caches: the scales ride the placement.** `CacheLayer_stream_quant` mirrors and rings the packed
`qk/qv` pages *and* their `sk/sv` scales, and the patched launch wrappers take optional scale tensors alongside
k/v, so the walk reads scales from whichever placement it reads the page from (the first version read them from
the resident tensors and broke on streamed chunks).

**The ring is shared per (device, component shapes, chunk count)** through a module-level registry, with per-slot
`used` / `ready` / `consumed` events. A layer prefetches its first `min(S, n_chunks)` chunks on entry and then
issues the next H2D right after recording that a slot's consumer has been launched (next-chunk lookahead only;
the design's cross-layer lookahead is not implemented). Partial accumulators are persistent per layer and sized
for the worst case, and `_plan_chunks()` caps the split count per launch so a step's partial buffers stay within a
64 MiB budget.

**Prefill is the same walk** with the prefill wrapper: the patched prefill kernel takes `n_start`/`n_end` and a
partial-slot range, so a query chunk walks the streamed history and the resident window exactly like decode,
then one combine reduces the partials in KV order. Publish (D2H of newly written rows) happens on the same copy
stream as the H2Ds, and `_pub_event` orders the compute stream after the last publish (the mirror refills read
publishes from earlier steps).

**Not implemented, and declined loudly rather than silently:** MLA/QSA/DSA geometries, tensor parallelism,
`bsz > 1`, the CUDA graph decode path, sliding-window layers, logit softcapping, `cu_seqlens` batching,
asymmetric V head dims, non-causal spans, and the CPU/disk page tiers (a `StreamLayer` in a cache that is also
tiered raises). ROCm is untested; the copy paths and the Triton kernels are the same code, but the bandwidth
profiles the design's tuning assumes are not.

Status of the roadmap in §8: M0, M1 and M2 are done (the resident set is dynamic by construction; copy stream,
ready/consumed events and prefetch depth are in; serial mode remains as the debug switch). M3 is done: prefill
walks the history with the bounded-range prefill launches, pages appended during prefill are published like
decode rows, and prompt-cache reuse works through `copy_page`/`rotate_pages`, which move mirror pages (the page
hashes that drive reuse are over the prompt tokens, not over the KV image, so hashing is unaffected by where the
KV lives; the design's worry under "Prompt cache semantics" does not apply). M4 has not been started, and the
M4 caveat that results need not be bit-identical across different stream configurations applies as documented:
streaming is a placement change rather than a numerics change, but split counts follow the chunking.
