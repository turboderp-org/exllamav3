"""
KV cache streaming, end to end (cache/stream.py, EXL3_KV_STREAM=1).

A streaming cache layer keeps the authoritative K/V in a host mirror and only a bounded working set on the
device: the most recent pages in the resident window, everything older read back through a shared ring. This is
exercised on a model whose context is longer than the window, and the streamed decode must reproduce the stock
decode of the same model and the same cache geometry; as in test_decode_consistency.py the comparison is
calibrated per model against its own noise floor (two cache-less forward passes of different lengths), because
the two paths differ in their accumulation order (the chunk walk re-splits a document history).

The streamed run must also actually stream: its logits could match for the wrong reason if the whole history
happened to fit the resident window, so every streamed layer is checked for a live page count above its window
and for a ring that was created (only the streamed-chunk path creates one).

Variants: fp16 cache, 8-bit cache (a second storage format, and its own quantization path in the walk), the
serial debug mode (no copy stream at all) and a tiny ring (more, smaller chunks than the default would give).
"""

import pytest
import torch

from exllamav3.cache.stream import StreamLayer
from testlib.e2e import REFERENCE_TEXT, assert_logits_agree, load_model, noise_floor, prefill_decode_logits

pytestmark = [pytest.mark.models("recurrent"), pytest.mark.cuda_only, pytest.mark.slow]

PROMPT_TOKENS = 1024
DECODE_STEPS = 32
CACHE_TOKENS = 2048

# A two-page resident window leaves everything but the newest 512 tokens (plus the decode tail) in the ring, so
# every variant walks streamed chunks; the ring/prefetch overrides give the tiny_ring variant several small
# chunks instead of the few large ones the default (64 ring pages) produces with this model
STREAM_ENV = {
    "EXL3_KV_STREAM": "1",
    "EXL3_KV_STREAM_RESIDENT": "2p",
    "EXL3_KV_STREAM_RING": "64",
    "EXL3_KV_STREAM_PREFETCH": "2",
    "EXL3_KV_STREAM_SERIAL": "0",
    "EXL3_KV_STREAM_TRACE": "0",
}

# (env overrides, absolute median-KL bound for that variant's cache geometry)
VARIANTS = {
    "fp16": ({}, 1e-3),
    "fp16_serial": ({"EXL3_KV_STREAM_SERIAL": "1"}, 1e-3),
    "fp16_tiny_ring": ({"EXL3_KV_STREAM_RING": "4"}, 1e-3),
    "q8": ({}, 5e-3),
}


def _long_ids(lm, n_tokens: int) -> torch.Tensor:
    """Teacher-forcing token sequence (1, n_tokens), built from the reference text (BOS once, repeated body)"""
    head = lm.encode(max_tokens = None)[0].tolist()
    body = lm.tokenizer.encode(REFERENCE_TEXT, add_bos = False)[0].tolist()
    ids = list(head)
    while len(ids) < n_tokens:
        ids += body
    return torch.tensor([ids[:n_tokens]], dtype = torch.long)


def _reset_stream_config():
    # The placement policy is memoized per cache id, and a cache object of an earlier load in the same process
    # can hand its address (and therefore its id) to a later one
    from exllamav3.cache import stream
    stream._STREAM_CONF.clear()


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_stream_matches_stock(model_id, model_dir, device, variant, monkeypatch):
    overrides, abs_median_kl = VARIANTS[variant]
    cache_args = ["-cq", "8"] if variant == "q8" else []

    # Stock reference: same model, same cache geometry, no streaming
    monkeypatch.setenv("EXL3_KV_STREAM", "0")
    _reset_stream_config()
    with load_model(model_dir, device, *cache_args, cache_tokens = CACHE_TOKENS) as lm:
        ids = _long_ids(lm, PROMPT_TOKENS)
        ref = prefill_decode_logits(lm, ids, DECODE_STEPS)
        floor = noise_floor(lm, ids, DECODE_STEPS)
        assert not any(isinstance(l, StreamLayer) for l in lm.cache.layers.values()), \
            "streaming layers active with EXL3_KV_STREAM=0"

    # Streamed run
    env = dict(STREAM_ENV)
    env.update(overrides)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _reset_stream_config()
    with load_model(model_dir, device, *cache_args, cache_tokens = CACHE_TOKENS) as lm:
        streamed = [l for l in lm.cache.layers.values() if isinstance(l, StreamLayer)]
        if not streamed:
            pytest.skip(f"{model_id}: no cache layer supports streaming")
        got = prefill_decode_logits(lm, ids, DECODE_STEPS)

        # The run must have gone through the ring: a live page count above the resident window (so some pages
        # were streamed), and a ring, which only the streamed-chunk path creates
        for l in streamed:
            R = l.cfg["R"]
            assert len(l._tok2pool) > R, \
                f"{model_id}: streamed layer saw {len(l._tok2pool)} pages, its window is {R}: nothing streamed"
            assert l._ring is not None, f"{model_id}: no ring entry was created"
        if variant == "fp16_tiny_ring":
            assert all(l.cfg["C"] == 2 and l.cfg["S"] == 2 for l in streamed), \
                f"{model_id}: ring override did not take effect: " \
                + str([(l.cfg["C"], l.cfg["S"]) for l in streamed][:1])
        if variant == "fp16_serial":
            assert all(l._copy_stream is None for l in streamed), \
                f"{model_id}: serial mode did not take effect"
        if variant == "q8":
            from exllamav3.cache.stream import CacheLayer_stream_quant
            assert all(isinstance(l, CacheLayer_stream_quant) for l in streamed), \
                f"{model_id}: 8-bit cache did not map to the quantized streaming layer"

    assert_logits_agree(ref, got, floor, f"{model_id} [{variant}]", abs_median_kl = abs_median_kl)
