"""
The RDNA3 HIP flash-decoding split kernels (rocm/rdna3_attn.cu), which bc_attn takes on gfx11 in place of the Triton
split kernel for causal full attention, against that Triton kernel (EXL3_RDNA3_ATTN=0) in the generator's cached
paths: teacher-forced single-token decode (q_len 1, the v_dot2 kernel) on an fp16 and an 8-bit cache, and n-gram
speculative decoding, whose verify rows (q_len 2..) on the 8-bit cache take the WMMA kernel. The switch is read at
import, so every setting runs in its own process. Models: the test roles with full-attention layers (head_dim 128
and 256 among them).

Tolerance: testlib.e2e.assert_logits_agree for the decode logits (the kernels differ only in accumulation order);
greedy equivalence for the speculative output.
"""

import pytest
import torch

from testlib.e2e import REFERENCE_TEXT, assert_greedy_equivalent, assert_logits_agree, greedy, load_model, \
    prefill_decode_logits
from testlib.isolated import run_isolated

pytestmark = [pytest.mark.rocm_only, pytest.mark.models("attention"), pytest.mark.slow]

DECODE_STEPS = 48
NEW_TOKENS = 96


def _gfx11(device) -> bool:
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx11")


def _hip_q_lens() -> list:
    """Wrap ext.rdna3_attn_split_kernel to record the q_len of every kernel it hands out"""
    from exllamav3.ext import exllamav3_ext as ext
    seen = []
    build = ext.rdna3_attn_split_kernel

    def wrapped(kb, hd, q_len, kvh, group_size):
        k = build(kb, hd, q_len, kvh, group_size)
        if k is not None:
            seen.append(q_len)
        return k

    ext.rdna3_attn_split_kernel = wrapped
    return seen


def _decode_worker(model_dir: str, device: str, cache_bits: int):
    seen = _hip_q_lens()
    args = ("-cq", str(cache_bits)) if cache_bits else ()
    with load_model(model_dir, torch.device(device), *args, cache_tokens = 4096) as lm:
        ids = lm.tokenizer.encode(REFERENCE_TEXT * 4, add_bos = True)[:, :1500]
        logits = prefill_decode_logits(lm, ids, DECODE_STEPS)
    return logits, sorted(set(seen))


def _speculative_worker(model_dir: str, device: str):
    seen = _hip_q_lens()
    with load_model(model_dir, torch.device(device), "-cq", "8", "-ngram", "2", cache_tokens = 4096) as lm:
        # A repetitive prompt so the n-gram drafter proposes (and the model verifies) several tokens per step
        text = " ".join(f"row {i % 9}: alpha beta gamma delta {i % 9}" for i in range(120))
        ids = lm.tokenizer.encode(text, add_bos = True)
        tokens, logits, _ = greedy(lm.generator(), ids, NEW_TOKENS, return_logits = True)
    return tokens, logits, sorted(set(seen))


@pytest.mark.parametrize("cache_bits", [0, 8])
def test_decode_matches_triton(model_id, model_dir, device, cache_bits):
    if not _gfx11(device):
        pytest.skip("RDNA3 (gfx11) only")
    ref, triton_q = run_isolated(_decode_worker, model_dir, str(device), cache_bits, env = {"EXL3_RDNA3_ATTN": "0"})
    got, hip_q = run_isolated(_decode_worker, model_dir, str(device), cache_bits)
    assert not triton_q
    assert 1 in hip_q, f"the HIP kernel was not used for single-token decode (q_len seen: {hip_q})"
    assert_logits_agree(ref, got, tag = f"{cache_bits or 16}-bit cache")


def test_speculative_verify_matches_triton(model_id, model_dir, device):
    if not _gfx11(device):
        pytest.skip("RDNA3 (gfx11) only")
    ref_tokens, ref_logits, _ = run_isolated(_speculative_worker, model_dir, str(device), env = {"EXL3_RDNA3_ATTN": "0"})
    tokens, _, hip_q = run_isolated(_speculative_worker, model_dir, str(device))
    assert any(q > 1 for q in hip_q), f"no multi-row verify went through the HIP kernel (q_len seen: {hip_q})"
    assert_greedy_equivalent(ref_tokens, ref_logits, tokens, tag = "n-gram speculative")
