"""
Bounded KV-sliced staging of the quantized-cache prefill (triton_paged._qc_staging_pages): a window wider
than the fixed fp16 scratch is dequantized and attended one block-table slice at a time into the same buffer,
with the running softmax state merged across slices, so the result must match the single-launch path over the
same inputs. The small-bound and large-bound runs use the same fp16 kernel (EXL3_SDPA_PREFILL is forced off,
which pins the two-pass query-length threshold at 256 on every device, so both sides stage deterministically),
isolating the slicing arithmetic from the sm_75 SDPA path. The second test turns that path back on: at head_dim
256 it is fa75, whose unnormalized (o, m, l) partials let the same slices merge through the combine, so both
sides must still agree.
"""
from collections import namedtuple

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn import triton_paged
from exllamav3.modules.attention_fn.triton_paged import (
    _paged_kv_update, paged_attn_triton_prefill,
)
from testlib.attention import gather_pages, quant_roundtrip, rand_paged_cache, ref_attn
from testlib.compare import assert_rel_close


Case = namedtuple("Case", "q qk sk qv sv bt sl kdeq vdeq T kvh bits")


def build_case(bsz, past, q_len, hd, device):
    """Inputs of one case: the fp16 paged cache with this chunk's rows already written, the q.cache round
    trip over it, and the dequantized window the fp32 reference attends over"""
    torch.manual_seed(bsz * 1000 + past + q_len + hd)
    kvh, n_heads, bits = 2, 8, 4
    T = past + q_len

    kc, vc, bt, sl = rand_paged_cache(bsz, T + 3, kvh, hd, past, device)
    k = torch.randn((bsz, q_len, kvh, hd), dtype = torch.half, device = device)
    v = torch.randn_like(k)
    q = torch.randn((bsz, q_len, n_heads, hd), dtype = torch.half, device = device)
    _paged_kv_update(k, v, kc, vc, bt, sl, bt.shape[1], bsz, q_len, kvh, PAGE_SIZE, hd)

    pages = kc.shape[0]
    qk, sk, kdeq = quant_roundtrip(kc.view(pages, PAGE_SIZE, kvh * hd), bits)
    qv, sv, vdeq = quant_roundtrip(vc.view(pages, PAGE_SIZE, kvh * hd), bits)
    return Case(q, qk, sk, qv, sv, bt, sl, kdeq.view(pages, PAGE_SIZE, kvh, hd),
                vdeq.view(pages, PAGE_SIZE, kvh, hd), T, kvh, bits)


def run_staged(c, q_len, bound, monkeypatch):
    """One two-pass prefill with the staging window bound to `bound` cache pages"""
    monkeypatch.setattr(triton_paged, "_qc_staging_pages", bound)
    return paged_attn_triton_prefill(
        c.q, None, None, c.qk, c.qv, c.bt, c.sl, causal = True,
        qc = (c.sk, c.sv, c.bits, c.bits), pre_appended_len = q_len, n_kv_heads_override = c.kvh,
    )


def check_case(c, q_len, monkeypatch):
    """Sliced (2 pages per slice) against the wide-bound single-launch run and against the fp32 reference"""
    out_small = run_staged(c, q_len, 2, monkeypatch)
    out_large = run_staged(c, q_len, 1 << 20, monkeypatch)
    assert_rel_close(out_small, out_large, 2e-3, "sliced vs single-slice")
    ref = ref_attn(c.q, gather_pages(c.kdeq, c.bt, c.T), gather_pages(c.vdeq, c.bt, c.T), causal = True)
    assert_rel_close(out_small, ref, 1.2e-2, "sliced vs fp32 reference")


# (bsz, past, q_len, head_dim); the 2324-context rows are the case where query rows span a slice boundary
CASES = [
    (1, 1800, 300, 128),
    (3, 1800, 300, 128),
    (1, 1800, 300, 256),
    (3, 1800, 300, 256),
    (1, 1800, 2048, 128),
    (3, 1800, 2048, 128),
    (1, 1800, 2048, 256),
    (3, 1800, 2048, 256),
    (1, 2324, 300, 128),
    (3, 2324, 300, 128),
    (1, 2324, 300, 256),
]


@pytest.mark.parametrize("bsz,past,q_len,hd",
                         [pytest.param(*c, id = "B{}-T{}-Q{}-D{}".format(*c)) for c in CASES])
def test_qc_staging_slices(device, monkeypatch, bsz, past, q_len, hd):
    monkeypatch.setenv("EXL3_SDPA_PREFILL", "0")   # same Triton fp16 kernel on both sides
    monkeypatch.setattr(triton_paged, "_qc_staging", 1)
    check_case(build_case(bsz, past, q_len, hd, device), q_len, monkeypatch)


# head_dim 256 only: the slice path is fa75, which has no other head dimension. The hd-128 cases would
# run the Triton slice loop against the SDPA wide side -- a re-run of the first test
@pytest.mark.cuda_only
@pytest.mark.cc(7, 5)
@pytest.mark.parametrize("bsz,past,q_len,hd",
                         [pytest.param(*c, id = "B{}-T{}-Q{}-D{}".format(*c)) for c in CASES if c[3] == 256])
def test_qc_staging_slices_fa75(device, monkeypatch, bsz, past, q_len, hd):
    # both sides on the fa75 path: the wide-bound run attends the whole window in one fa75 call, the
    # bounded one merges fa75's per-slice partials in the combine
    monkeypatch.setenv("EXL3_SDPA_PREFILL", "1")
    monkeypatch.setenv("EXL3_FA75", "1")
    monkeypatch.setattr(triton_paged, "_qc_staging", 1)
    check_case(build_case(bsz, past, q_len, hd, device), q_len, monkeypatch)
