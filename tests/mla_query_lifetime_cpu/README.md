# Sparse projected-query storage lifetime

For a NoPE MLA layer, the zero-width rope slice still owns the full projected
query's storage. The sparse path uses the independently absorbed query instead.
This candidate removes an unused NoPE slice and gives the unused zero-width rope
argument empty storage before deleting the projected query. Nonzero rope queries
retain their storage; the dense paths are unchanged.

The CPU regression extracts the actual `_attend` method without importing the
engine or native extension. Projection, indexing and attention helpers are CPU
stand-ins, so the checks establish Python storage ownership and preserved
stand-in results, rather than attention-kernel numerics. The public fixture is
the upstream MIT-licensed method at dev
`6123763150abbadecdc6e9650dd4c9a02c9a5a98`.

```sh
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \
  python -m pytest -q tests/mla_query_lifetime_cpu
```

Sixteen cases cover sparse and dense decode/MHA prefill, zero/nonzero rope widths,
and one/two-item batches. Test collection does not change other tests' GPU
visibility. A direct script run also accepts explicit baseline/candidate paths,
hides CUDA/HIP devices, and verifies CUDA remains uninitialized.

```sh
python tests/mla_query_lifetime_cpu/test_query_lifetime.py \
  --baseline <baseline-mla.py> --candidate <candidate-mla.py>
```

A separate owner-run deployed-source GPU campaign used source
`16a49792a3c93d8432d72e6c4bce800841566577`, a GLM target with DFlash2 K7,
384K cache allocation, chunk size 2048, workspace margin 128 MiB, and configured
BC attention off. Its one cold 16,528-token request per arm produced identical
choices and usage at the 256-token output cap. The first request-time R=2048
observation from each of 11 sparse MLA layers had 67,108,864 fewer allocated
bytes at sparse entry and zero rather than 67,108,864 bytes backing its unused
rope argument. Both arms used the same private instrumentation, which samples
the first occurrence per layer/batch/query shape. It does not observe every call.

Baseline MLA SHA256 was
`90eed44b790f4810ed1eeddb70cb4e4bdd38a82b54d43fcf7208148d4d250d10`;
the candidate was
`4caec9d0b95dfae36a5fb1b869699ee635d734b6f84bd41b4a0ff1f552ecc747`.
The upstream dev source file and patched candidate have those same respective
hashes. That establishes exact file applicability, while the GPU campaign remains
a deployed-engine result, rather than a whole current-dev runtime test.

This is bounded evidence for local query-storage relief. It does not establish
whole-process peak/reservation relief, increased context capacity, an OOM cure,
general model quality, universal output equivalence or a throughput improvement.
