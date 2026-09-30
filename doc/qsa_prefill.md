# Bounded QSA prefill

`EXL3_QSA_PREFILL=1` opts into the Q8 QSA prefill path; disabled by default.
It is independent of `EXL3_COALESCED_CHECKPOINTS=1`. The current supported
layout is Flash-Next on SM121: FP16 queries, 24 query heads, two KV heads,
256-dimensional heads, integer 8-bit K/V, and 2080-column padded selections.
Other devices/layouts, non-cached calls, batched sequences and decode retain
native attention. Independently scheduled single-sequence prefills can use it
while several requests are active.

The path stages the referenced logical Q8 window once in its rotated FP16
domain, clips padded selection work without dropping the incomplete tail,
writes single-split results directly, and uses native dense prefill for the
first 2048 still-dense tokens. Dense-prefix calls disable the generic Q8 staging
only for that call, rather than mutating a process-global setting. Sparse
selection, cache storage precision and the model's context limit do not change.

## Loader stability

Staging is bounded to **262144 tokens per request** (at most 512 MiB combined
K/V scratch for this layout), not the aggregate shared cache-pool size.
EXL3 autosplit can synthesize a prefill at the end of a much larger shared pool.
The initial local adapter tried to stage that probe and raised an exception
before serving. This port includes the fix: contexts beyond the staging bound
return to native attention **before allocation or any CUDA-device query**.
It never truncates indices, expands the staging allocation to the whole pool,
or reduces the configured shared cache. Valid windows keep the optimized path.

## Validation and provenance

The Triton kernel bodies derive from the separately validated local EXL3 1.5.2
QSA implementation. The upstream integration in this PR is new: explicit
request context replaces local hooks, a call-local dense-prefill override
replaces a temporary global mutation, and unsupported layouts fall back.
No GDN scan, MoE, GEMM, loader-chunk or deployment-specific optimization from
the larger local bundle is included here.

`python tests/test_qsa_prefill.py -v` checks CPU-only dispatch, oversized-pool
fallback, the supported-window boundary, decode fallback and the call-local
dense-prefix override. These tests do not compile or execute CUDA kernels.
GPU numerical equivalence, full-model validation and matched throughput
against unmodified upstream master remain unrun for this port. No additional
benchmarks were authorized for this update. Historical whole-bundle results
are not evidence of this PR's isolated speedup or correctness.
