# Recurrent checkpoint staging

`EXL3_COALESCED_CHECKPOINTS=1` opts into bounded pinned-host staging for
single-device GDN/PLE checkpoint copies. It is disabled by default.

The fast path issues the tensor copies on the current CUDA stream and waits
once before publishing a checkpoint. A reusable 128 MiB pinned arena stages
the transfers; retained checkpoints use ordinary pageable CPU storage.
Restore preserves the same synchronous ownership contract. Neither KV/recurrent
precision, checkpoint intervals nor the recurrent-cache LRU capacity changes.

Only explicitly supported recurrent-layer layouts use the fast path. TP,
multi-device, CPU-only, oversized, unaligned or unsupported layouts retain
their normal per-layer copies. Failure to allocate the staging arena also
falls back to the normal stash path. Checkpoint restore rejects incompatible
stored layouts rather than silently changing precision or truncating data.

CPU-resident token histories are copied into independently owned snapshots in
both paths. In particular, PLE token IDs must not be retained with `.cpu()`
alone, since that can alias the live CPU history.

Run `python tests/test_recurrent_transfer.py -v` for ownership/fallback tests
and CUDA-conditional byte-equality, arena-reuse and constructor-restore tests.
No model weights are needed. CPU-only hosts skip the CUDA-specific cases.
