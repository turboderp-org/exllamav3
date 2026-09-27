"""Checkpoint argument validation, without loading a model or allocating GPU memory."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.generator.generator import Generator


class CheckpointIntervalsTest(unittest.TestCase):

    def make_generator(self, recurrent = True, default_interval = None, **kwargs):
        caps = {"recurrent_states": recurrent}
        if default_interval is not None:
            caps["default_recurrent_checkpoint_interval"] = default_interval
        model = SimpleNamespace(config = SimpleNamespace(vocab_size = 256), caps = caps)
        cache = SimpleNamespace(num_slots = 4, reset_states = Mock())
        with patch("exllamav3.generator.generator.PageTable", return_value = SimpleNamespace(max_pages = 16)), \
             patch("exllamav3.generator.generator.ThreadPoolExecutor"):
            return Generator(model, cache, tokenizer = None, **kwargs)

    def test_rejects_unaligned_prefill_interval(self):
        for recurrent in (False, True):
            for interval in (1, 255, 257, 1000, 32769):
                with self.subTest(recurrent = recurrent, interval = interval):
                    with self.assertRaisesRegex(AssertionError, "checkpoint interval must be a multiple"):
                        self.make_generator(recurrent = recurrent, recurrent_checkpoint_interval_pp = interval)

    def test_rejects_unaligned_generation_interval(self):
        for interval in (1, 255, 257, 1000, 32769):
            with self.subTest(interval = interval):
                with self.assertRaisesRegex(AssertionError, "checkpoint interval must be a multiple"):
                    self.make_generator(recurrent_checkpoint_interval = interval)

    def test_defaults(self):
        gen = self.make_generator()
        self.assertEqual(gen.recurrent_checkpoint_interval, 2048)
        self.assertEqual(gen.recurrent_checkpoint_interval_pp, 32768)

    def test_architecture_default(self):
        gen = self.make_generator(default_interval = 8192)
        self.assertEqual(gen.recurrent_checkpoint_interval, 8192)
        self.assertEqual(gen.recurrent_checkpoint_interval_pp, 32768)

    def test_explicit_interval_overrides_architecture_default(self):
        gen = self.make_generator(default_interval = 8192, recurrent_checkpoint_interval = 256)
        self.assertEqual(gen.recurrent_checkpoint_interval, 256)

    def test_aligned_prefill_interval_still_rounds_to_chunk_size(self):
        for interval, expected in ((256, 2048), (2048, 2048), (2304, 4096), (32768, 32768)):
            with self.subTest(interval = interval):
                gen = self.make_generator(recurrent_checkpoint_interval_pp = interval, max_chunk_size = 2048)
                self.assertEqual(gen.recurrent_checkpoint_interval_pp, expected)


if __name__ == "__main__":
    unittest.main()
