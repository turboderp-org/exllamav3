"""
Offline SAM corpus builder (util/build_sam.py): bounded sampling with document boundaries and deduplication,
tool-call message normalization, recipe validation, atomic writes, and the frozen bank format. The bank is
checked against the live BC_SAM it was exported from (byte planes decode to the exported arrays) and against
brute-force substring search over the corpus.
"""

import bisect
import json
import random
from array import array
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from testlib.repo_scripts import load_repo_script

zstandard = pytest.importorskip("zstandard")

pytestmark = pytest.mark.nogpu


@pytest.fixture(scope = "module")
def builder():
    return load_repo_script("util/build_sam.py")


class ByteTokenizer:
    def encode(self, text, add_special_tokens = False):
        return list(text.encode())


def test_bounded_sampling_and_boundaries(builder):
    tokens, ends = array("i"), array("i")
    rows = iter([{"text": "abc"}, {"text": "abc"}, {"text": "def"}, {"text": "never fetched"}])
    source = dict(mode = "text", field = "text", max_rows = 3)
    stats = builder.collect(source, rows, ByteTokenizer(), {}, 6, 100, 100, tokens, ends, set())
    assert tokens.tolist() == [97, 98, 99, -1, 100, 101, 102, -1]
    assert ends.tolist() == [3, 7]
    assert stats["duplicates"] == 1
    assert next(rows)["text"] == "never fetched"
    with pytest.raises(ValueError, match = "Token limit"):
        builder.collect(source, iter([{"text": "hello"}]), ByteTokenizer(), {}, 5, 100, 3, array("i"), array("i"), set())


def test_tool_normalization(builder):
    messages = builder.normalize_messages(json.dumps([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "tool_call_ids": ["x"], "content": [{"type": "text", "text": "file.py"}]},
    ]))
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert messages[1]["name"] == "bash"
    assert messages[1]["tool_call_id"] == "x"
    assert messages[1]["content"] == "file.py"
    with pytest.raises(ValueError):
        builder.normalize_messages([{"role": "user", "content": [{"type": "image"}]}])


def test_recipe_validation(builder, tmp_path):
    builder.load_recipe(Path(builder.__file__).with_name("sam_recipes") / "coding.yaml")
    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\nsources: []\n")
    with pytest.raises(ValueError):
        builder.load_recipe(p)


def test_atomic_failure_preserves_output(builder, tmp_path):
    path = tmp_path / "bank.sam.zst"
    path.write_bytes(b"existing")
    with patch.object(zstandard, "ZstdCompressor", side_effect = RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match = "interrupted"):
            builder.write_bank(path, {"corpus": np.arange(10, dtype = np.int32)}, {}, force = True)
    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]


def test_frozen_roundtrip_and_matching(builder, tmp_path):
    from exllamav3.ext import exllamav3_ext as ext
    rng = random.Random(123)
    # Wide root, high-degree nonroot states, clones, and exceptional IDs.
    corpus = [x for i in range(160) for x in [17, i, -1]]
    corpus += [rng.randrange(100) for _ in range(2000)] + [2**31 - 1, -1]
    sam = ext.BC_SAM()
    sam.accept_tensor(torch.tensor(corpus, dtype = torch.int64))
    arrays = dict(zip(builder.GRAPH_ARRAYS, (t.numpy() for t in sam.export_csr())))
    arrays["corpus"] = np.array(corpus, dtype = np.int32)
    arrays["document_ends"] = np.array([i for i, t in enumerate(corpus) if t == -1], dtype = np.int32)
    # Export must own independent copies and leave the original bank usable.
    sam.accept(1234)
    sam.reset(0)
    path = tmp_path / "bank.sam.zst"
    builder.write_bank(path, arrays, {"token_count": len(corpus)}, level = 1)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        builder.write_bank(path, arrays, {})
    assert path.read_bytes() == original
    magic, version, size, payload_size = builder.PREFIX.unpack_from(original)
    assert (magic, version) == (builder.MAGIC, 1)
    start = builder.PREFIX.size
    meta = json.loads(original[start:start + size])
    payload = zstandard.ZstdDecompressor().decompress(original[start + size:])
    assert len(payload) == payload_size
    decoded = {}
    for section in meta["sections"]:
        at, n = section["offset"], section["count"]
        assert at % 64 == 0
        planes = np.frombuffer(payload[at:at + 4 * n], dtype = np.uint8).reshape(4, n)
        values = planes.T.copy().view("<i4").ravel()
        np.testing.assert_array_equal(values, arrays[section["name"]])
        decoded[section["name"]] = values
    offsets, labels, to = (decoded[k] for k in ("edge_offsets", "edge_token", "edge_to"))
    assert offsets[-1] == len(labels)
    for s in range(len(offsets) - 1):
        a, b = offsets[s:s + 2]
        if b - a > 1:
            assert np.all(labels[a + 1:b] > labels[a:b - 1])
    # Every substring of short queries must lead to an actual occurrence.
    for _ in range(300):
        qstart = rng.randrange(len(corpus) - 8)
        query = corpus[qstart:qstart + rng.randrange(1, 9)]
        state = 0
        for token in query:
            a, b = offsets[state:state + 2]
            e = bisect.bisect_left(labels, token, int(a), int(b))
            assert e < b
            assert labels[e] == token
            state = to[e]
        end = int(decoded["min_end"][state]) + 1
        assert corpus[end - len(query):end] == query
    for token, state in enumerate(decoded["root"]):
        e = bisect.bisect_left(labels, token, 0, int(offsets[1]))
        expected = to[e] if e < offsets[1] and labels[e] == token else -1
        assert state == expected
    broken = bytearray(original[start + size:])
    broken[-1] ^= 1
    with pytest.raises(zstandard.ZstdError):
        zstandard.ZstdDecompressor().decompress(broken)
