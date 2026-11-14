# FlashStates-style hybrid state management

This is an implementation of the two core ideas from
[FlashStates](https://github.com/FutureMLS-Lab/FlashStates) (FutureMLS-Lab, 2026) for exllamav3:

1. **Arbitrary state checkpointing (ASC).** A recurrent/SSM checkpoint is stored at *any* token position, not
   only at KV-page boundaries: at the end of a prompt, at the end of a request, and at any prefill chunk
   boundary. The next request that extends exactly that prefix resumes there.
2. **Cost-aware hybrid state management.** A checkpoint and the K/V pages it needs are admitted and evicted
   together, and the checkpoint cache can evict by GreedyDual-Size over the prefill time a state saves per
   byte it occupies.

exllamav3 already had page-aligned recurrent checkpoints. The change is that a checkpoint may now end in the
middle of a page.

## Why it matters

A coding agent resends a growing context on every turn. On a hybrid model (attention + linear-attention /
SSM layers) a cached prefix is only reusable together with the recurrent state at its exact end. With only
page-aligned checkpoints, every turn replays up to the previous 256-token boundary plus (because generation
checkpoints are spaced by `recurrent_checkpoint_interval`) up to that interval. With ASC the next turn
resumes at the exact token where the previous one stopped.

## Design

### Checkpoint key

A checkpoint is keyed by the chained content hash of the K/V prefix it resumes from.

* At a page boundary (the historical case) the key is the hash of the last complete page, which is exactly
  what `Sequence.prepare()` computes for a later request.
* Inside a page, the key is `blake2b(prev_page_hash || tokens[start:position])`, i.e. the content hash of the
  partial page chained onto the last complete page. The partial page's K/V is stored in the page tiers under
  the same key, with `prev_hash` pointing at the complete page, so the existing chain walk
  (`PageTable.is_resumable`) covers state and K/V together.

A later request rebuilds the key from its own prompt tokens and compares it, so two prompts that share the
complete pages but diverge inside the partial one cannot produce a false hit.

### Where the partial page lives

`CPUPageCache.store(page, ..., key=...)` copies the partially filled page into the same pinned slab format a
complete page uses, under the partial key. It is written through to the disk tier as any other page, so the
checkpoint survives a restart. No schema change was needed: the position comes from the checkpoint
(`offset` in the stash), not from the page entry.

### Storing

`Job.frontier_checkpoint()` runs at three moments:

* when prefill reaches the end of the prompt (`maybe_stash_recurrent`, `new_tokens == 0`);
* when a request finishes;
* when a request requeues (the requeued job then resumes at the exact requeue point).

At a page boundary it is an ordinary `RecurrentCache.put`. Inside a page it stores the partial page in the
tier and calls `RecurrentCache.put_partial`, recording `anchor_hash` and `offset` in the stash so the index
can be rebuilt from disk after a restart.

### Restoring

`Sequence.find_resume()` enumerates the page-boundary checkpoints of the shared complete pages and the
partial checkpoints indexed under those pages, validates each partial candidate against the request's own
tokens, and picks the deepest one whose chain is still restorable. `PageTable.allocate_pages(...,
partial_resume=...)` then allocates the complete pages by hash as usual and restores the partial page into
the first new page with `kv_position = offset`. The sequence's `kv_position` and the recurrent state's
position become the exact checkpoint position, `Job.partial_prefix` permits prefill to start inside a page,
and the forward pass receives `cache_seqlens = position`.

### Eviction

* `RecurrentCache` evicts stranded checkpoints first (their K/V chain is gone), then either LRU (default) or
  GreedyDual-Size (`recurrent_eviction="gds"`). GDS priority is
  `(D * saved_tokens + E) / checkpoint_bytes`, with the classic inflation term.
* The page tiers already prune whole chains tail-first and the disk tier drops a page together with the
  checkpoint stored under the same key, so a partial page and its state leave together.
* `PageTable.evict` and the recurrent cache's stranded check prune the two-way dependency, as before.

## Configuration

exllamav3 `Generator`:

| Argument | Default | Meaning |
|---|---|---|
| `recurrent_frontier_checkpoints` | `True` | Store exact-end checkpoints (needs `cpu_cache_size > 0` for the partial page) |
| `recurrent_eviction` | `"lru"` | `"lru"` or `"gds"` |
| `prefill_cost_coefficients` | `(1.0, 0.0)` | `(D, E)` for `"gds"`; only the ratio matters |

TabbyAPI per-model config (`tabby_config.yml` `model:` section, or a load request). This patch only adds the
options to `Generator`; exposing them through TabbyAPI's model loader is a companion change
([theroyallab/tabbyAPI#519](https://github.com/theroyallab/tabbyAPI/pull/519)):

```yaml
model:
  recurrent_frontier_checkpoints: true
  recurrent_eviction: gds
  prefill_cost_coefficients: [2.230e-5, 4.34e-6]   # Qwen3.8-Flash-Next (FlashStates benchmark config)
```

For reference, FlashStates' GLM-5.3-Flash config uses `1.862e-5 3.33e-6`; only the ratio of the two values
matters.

The partial page needs the system-memory tier (`memory.sysmem_kv_cache`); the state cache is
`memory.sysmem_recurrent_cache`. With `sysmem_kv_cache: 0`, only page-aligned checkpoints can be stored and a
warning is logged.

## Metrics

* `PageTable.metrics["partial_stores"]` — partial pages stored by frontier checkpoints.
* `PageTable.metrics["alloc_partial_pages"]` — allocations that resumed inside a page.
* `RecurrentCache.metrics["stash_partial"]` — arbitrary-position states stored.
* `Generator.get_cache_stats()["alloc_partial_pages"]` — the allocation metric, alongside the other hit
  statistics.

## Scope

Implemented: arbitrary-position checkpoints for the recurrent state *and* the partially filled K/V page,
exact-position resume, disk persistence, cost-aware state eviction, tests. TabbyAPI loader plumbing is a
companion change ([theroyallab/tabbyAPI#519](https://github.com/theroyallab/tabbyAPI/pull/519)), not part of
this patch.

Not implemented, and why:

* **Mooncake store tier.** exllamav3 has no Mooncake connector; the system-memory (`CPUPageCache`) and disk
  tiers already provide the same "write through to a host store, evict whole states" role for one engine.
* **A dedicated mid-prefill export kernel.** exllamav3 prefill is already chunked and carries the state
  across chunk boundaries, so a checkpoint can be taken at any chunk end. Chunks are page-aligned, so those
  checkpoints remain page-aligned; exact-end checkpoints come from the frontier path.
* **Leaf-last eviction** (`--prefix-cache-leaf-last`) is approximated by the existing tail-first chain
  pruning and the stranded-first rule; the disk tier already keeps each chain's deepest checkpoint.
