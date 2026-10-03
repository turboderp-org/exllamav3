"""Offline builder tests: bounded sampling, tool rendering, frozen graph round-trip."""
import bisect
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import zstandard

spec = importlib.util.spec_from_file_location('build_sam', Path(__file__).resolve().parents[1] / 'util/build_sam.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class BuilderTests(unittest.TestCase):
    def test_bounded_sampling_and_boundaries(self):
        from array import array
        tokens, ends = array('i'), array('i')
        rows = iter([{'text': 'abc'}, {'text': 'abc'}, {'text': 'def'}, {'text': 'never fetched'}])
        source = dict(mode='text', field='text', max_rows=3)
        stats = builder.collect(source, rows, ByteTokenizer(), {}, 6, 100, 100, tokens, ends, set())
        self.assertEqual(tokens.tolist(), [97, 98, 99, -1, 100, 101, 102, -1])
        self.assertEqual(ends.tolist(), [3, 7])
        self.assertEqual(stats['duplicates'], 1)
        self.assertEqual(next(rows)['text'], 'never fetched')
        with self.assertRaisesRegex(ValueError, 'Token limit'):
            builder.collect(source, iter([{'text': 'hello'}]), ByteTokenizer(), {}, 5, 100, 3, array('i'), array('i'), set())

    def test_tool_normalization(self):
        messages = builder.normalize_messages(json.dumps([
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'x', 'function': {'name': 'bash', 'arguments': '{"command":"ls"}'}}]},
            {'role': 'tool', 'tool_call_ids': ['x'], 'content': [{'type': 'text', 'text': 'file.py'}]},
        ]))
        self.assertEqual(messages[0]['tool_calls'][0]['function']['arguments'], {'command': 'ls'})
        self.assertEqual(messages[1]['name'], 'bash')
        self.assertEqual(messages[1]['tool_call_id'], 'x')
        self.assertEqual(messages[1]['content'], 'file.py')
        with self.assertRaises(ValueError):
            builder.normalize_messages([{'role': 'user', 'content': [{'type': 'image'}]}])

    def test_recipe_validation(self):
        builder.load_recipe(Path(builder.__file__).with_name('sam_recipes') / 'coding.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.yaml'
            p.write_text('version: 1\nsources: []\n')
            with self.assertRaises(ValueError):
                builder.load_recipe(p)

    def test_atomic_failure_preserves_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bank.sam.zst'
            path.write_bytes(b'existing')
            with patch.object(zstandard, 'ZstdCompressor', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    builder.write_bank(path, {'corpus': np.arange(10, dtype=np.int32)}, {}, force=True)
            self.assertEqual(path.read_bytes(), b'existing')
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_frozen_roundtrip_and_matching(self):
        from exllamav3.ext import exllamav3_ext as ext
        rng = random.Random(123)
        # Wide root, high-degree nonroot states, clones, and exceptional IDs.
        corpus = [x for i in range(160) for x in [17, i, -1]]
        corpus += [rng.randrange(100) for _ in range(2000)] + [2**31 - 1, -1]
        sam = ext.BC_SAM()
        sam.accept_tensor(torch.tensor(corpus, dtype=torch.int64))
        arrays = dict(zip(builder.GRAPH_ARRAYS, (t.numpy() for t in sam.export_csr())))
        arrays['corpus'] = np.array(corpus, dtype=np.int32)
        arrays['document_ends'] = np.array([i for i, t in enumerate(corpus) if t == -1], dtype=np.int32)
        # Export must own independent copies and leave the original bank usable.
        sam.accept(1234)
        sam.reset(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bank.sam.zst'
            builder.write_bank(path, arrays, {'token_count': len(corpus)}, level=1)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                builder.write_bank(path, arrays, {})
            self.assertEqual(path.read_bytes(), original)
            magic, version, size, payload_size = builder.PREFIX.unpack_from(original)
            self.assertEqual((magic, version), (builder.MAGIC, 1))
            start = builder.PREFIX.size
            meta = json.loads(original[start:start + size])
            payload = zstandard.ZstdDecompressor().decompress(original[start + size:])
            self.assertEqual(len(payload), payload_size)
            decoded = {}
            for section in meta['sections']:
                at, n = section['offset'], section['count']
                self.assertEqual(at % 64, 0)
                planes = np.frombuffer(payload[at:at+4*n], dtype=np.uint8).reshape(4, n)
                values = planes.T.copy().view('<i4').ravel()
                np.testing.assert_array_equal(values, arrays[section['name']])
                decoded[section['name']] = values
            offsets, labels, to = (decoded[k] for k in ('edge_offsets', 'edge_token', 'edge_to'))
            self.assertEqual(offsets[-1], len(labels))
            for s in range(len(offsets)-1):
                a, b = offsets[s:s+2]
                self.assertTrue(np.all(labels[a+1:b] > labels[a:b-1]) if b-a > 1 else True)
            # Every substring of short queries must lead to an actual occurrence.
            for _ in range(300):
                start = rng.randrange(len(corpus) - 8)
                query = corpus[start:start + rng.randrange(1, 9)]
                state = 0
                for token in query:
                    a, b = offsets[state:state+2]
                    e = bisect.bisect_left(labels, token, int(a), int(b))
                    self.assertLess(e, b)
                    self.assertEqual(labels[e], token)
                    state = to[e]
                end = int(decoded['min_end'][state]) + 1
                self.assertEqual(corpus[end-len(query):end], query)
            for token, state in enumerate(decoded['root']):
                e = bisect.bisect_left(labels, token, 0, int(offsets[1]))
                expected = to[e] if e < offsets[1] and labels[e] == token else -1
                self.assertEqual(state, expected)
            broken = bytearray(original[start + size:])
            broken[-1] ^= 1
            with self.assertRaises(zstandard.ZstdError):
                zstandard.ZstdDecompressor().decompress(broken)


if __name__ == '__main__':
    unittest.main()
