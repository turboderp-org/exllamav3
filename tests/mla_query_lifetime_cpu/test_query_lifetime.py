"""CPU-only storage-lifetime check of the actual MLAttention._attend method.

Use CPU-capable Torch with optional baseline/candidate paths. Relative GPU helper imports
are removed from the extracted method; projections/indexing/attention are CPU
stand-ins. Checks original query-owner lifetime at the actual sparse-call site and
identical stand-in output. This is not a real attention numerical or GPU test.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

# A direct script invocation hides devices. Test collection must not change the
# visibility of other upstream GPU tests in the same pytest process.
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["HIP_VISIBLE_DEVICES"] = ""
import torch
import pytest

pytestmark = pytest.mark.nogpu


def extracted_attend(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLAttention")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_attend")

    class RemoveRelativeImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.level else node

    function = RemoveRelativeImports().visit(function)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "torch": torch,
        "get_for_device": lambda params, key, device, default=None: params.get(key, default),
        "to_device": lambda tensor, device: tensor.to(device),
        "_dbg_sync": lambda *args: None,
        "MAX_DECODE_QLEN": 16,
        "_prefill_mode": "mha",
        "_debug_sync": False,
        "NoFittingConfig": RuntimeError,
    }

    def absorb(query, weight, heads, nope):
        # New owned output, preserving a reproducible function of projected q.
        return query[..., :nope].repeat(1, 1, 2).permute(1, 0, 2).contiguous()

    def unfold(latent, weight, width):
        return latent[:, :, :width].permute(1, 0, 2).contiguous()

    def dense(query_latent, query_rope, *args, **kwargs):
        return query_latent.clone()

    namespace.update(mla_absorb=absorb, mla_unfold=unfold,
                     mla_attn_triton_decode=dense, mla_attn_triton_prefill=dense)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_attend"]


def run_case(function, rope_width, batch, sequence, sparse=True):
    heads, nope, latent, value_width = 4, 8, 16, 4
    query_width = nope + rope_width
    seen = {}
    owner = SimpleNamespace(
        device=torch.device("cpu"),
        num_q_heads=heads, indexer_mode="full" if sparse else None, index_topk=4,
        qk_head_dim=query_width, qk_nope_head_dim=nope,
        qk_rope_head_dim=rope_width, kv_lora_rank=latent,
        index_kpool=4, l4_beta=False, rope=None,
        w_uk_flat=None, w_uv_flat=None, v_head_dim=value_width, sm_scale=1.0,
    )

    def project_query(x, params, return_resid):
        query = torch.empty((batch, sequence, heads * query_width), dtype=torch.half)
        query.copy_(torch.arange(query.numel()).reshape(query.shape) % 17)
        owner.query_ref = weakref.ref(query)
        return query, None

    owner.project_q = project_query
    owner.kv_a_proj_with_mqa = SimpleNamespace(
        forward=lambda x, params: torch.ones((batch, sequence, latent + rope_width), dtype=torch.half))
    owner.kv_a_layernorm = SimpleNamespace(forward=lambda x, params, out_dtype: x.clone())
    owner._indexer_keys = lambda *args: torch.zeros((batch, sequence, 8), dtype=torch.half)
    owner._indexer_topk_kpool = lambda *args, **kwargs: torch.zeros((batch * sequence, 4), dtype=torch.int32)
    owner.o_proj = SimpleNamespace(forward=lambda x, params: x.clone())
    owner._mha_form = lambda query, *args: query[..., :value_width].reshape(batch, sequence, heads * value_width).clone()

    def sparse(query_latent, query_rope, *args, **kwargs):
        gc.collect()
        seen["query_owner_alive_at_sparse_call"] = owner.query_ref() is not None
        seen["rope_numel"] = query_rope.numel()
        seen["rope_storage_bytes"] = query_rope.untyped_storage().nbytes()
        result = query_latent.clone()
        if rope_width:
            result += query_rope[..., 0].reshape(batch * sequence, heads).T.unsqueeze(-1)
        return result

    owner._attend_sparse = sparse
    result = function(owner, torch.ones((batch, sequence, 4), dtype=torch.half), batch, sequence,
                      {}, None, None, torch.zeros((batch, 1), dtype=torch.int32), None,
                      lambda *args: None, qc=None, host_seqlens=[8] * batch, idx_layer=None)
    assert result.shape == (batch, sequence, heads * value_width)
    return seen, result


@pytest.fixture(scope="module")
def methods():
    here = Path(__file__).resolve().parent
    return (extracted_attend(here / "fixtures/upstream_dev_attend.py.txt"),
            extracted_attend(here.parents[1] / "exllamav3/modules/mla_attn.py"))


@pytest.mark.parametrize("rope", (0, 4))
@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("sequence", (32, 64))
def test_sparse_query_owner_and_empty_rope_storage(methods, rope, batch, sequence):
    old, new = methods
    original, a = run_case(old, rope, batch, sequence)
    candidate, b = run_case(new, rope, batch, sequence)
    assert original["query_owner_alive_at_sparse_call"]
    assert candidate["query_owner_alive_at_sparse_call"] == bool(rope)
    assert torch.equal(a, b)
    if not rope:
        assert original["rope_numel"] == candidate["rope_numel"] == 0
        assert original["rope_storage_bytes"] > 0 and candidate["rope_storage_bytes"] == 0


@pytest.mark.parametrize("rope", (0, 4))
@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("sequence", (8, 32))
def test_dense_decode_and_mha_prefill_outputs_preserved(methods, rope, batch, sequence):
    old, new = methods
    _, a = run_case(old, rope, batch, sequence, sparse=False)
    _, b = run_case(new, rope, batch, sequence, sparse=False)
    assert torch.equal(a, b)


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--baseline", type=Path, default=here / "fixtures/upstream_dev_attend.py.txt")
    parser.add_argument("--candidate", type=Path, default=here.parents[1] / "exllamav3/modules/mla_attn.py")
    options = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU test unexpectedly has an initialized CUDA context")
    old, new = extracted_attend(options.baseline), extracted_attend(options.candidate)
    rows = []
    for rope in (0, 4):
        for batch in (1, 2):
            for sequence in (32, 64):
                original, a = run_case(old, rope, batch, sequence)
                candidate, b = run_case(new, rope, batch, sequence)
                assert original["query_owner_alive_at_sparse_call"]
                assert candidate["query_owner_alive_at_sparse_call"] == bool(rope)
                assert torch.equal(a, b)
                if not rope:
                    assert original["rope_numel"] == candidate["rope_numel"] == 0
                    assert original["rope_storage_bytes"] > 0 and candidate["rope_storage_bytes"] == 0
                rows.append({"rope_width": rope, "batch": batch, "sequence": sequence,
                             "path": "sparse", "baseline": original, "candidate": candidate,
                             "stand_in_output_equal": True})
    for rope in (0, 4):
        for batch in (1, 2):
            for sequence in (8, 32):
                original, a = run_case(old, rope, batch, sequence, sparse=False)
                candidate, b = run_case(new, rope, batch, sequence, sparse=False)
                assert torch.equal(a, b)
                rows.append({"rope_width": rope, "batch": batch, "sequence": sequence,
                             "path": "dense_decode" if sequence == 8 else "dense_mha_prefill",
                             "stand_in_output_equal": True})
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during CPU test")
    print(json.dumps({"torch": torch.__version__, "cuda_initialized": False,
                      "baseline_sha256": hashlib.sha256(options.baseline.read_bytes()).hexdigest(),
                      "candidate_sha256": hashlib.sha256(options.candidate.read_bytes()).hexdigest(),
                      "passed_cases": len(rows), "tests": rows,
                      "scope": "Actual Python method plus CPU helper stand-ins; storage lifetime only, no GPU/attention proof."},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
