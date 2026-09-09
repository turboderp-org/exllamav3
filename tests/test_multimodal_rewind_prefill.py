"""CPU regression tests for multimodal rewind prefill.

Use the real Sequence and MRoPE builder, and intercept model calls to check
their inputs before GPU execution. No model weights are needed.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.job import Job
from exllamav3.tokenizer import MMEmbedding
from exllamav3.util.rope import RoPE, RopeSettings


class ModelBoundaryReached(Exception):
    pass


class ModelProbe:
    def __init__(self, rope):
        self.g_rope = rope
        self.caps = {}
        self.calls = []

    def prefill(self, *, input_ids, params):
        self.calls.append((input_ids, params))
        raise ModelBoundaryReached

    forward = prefill


def make_job(prompt_length, sequence_length, replay_start, *, image=True, image_at_end=False, mtp=False, atomic=False):
    rope = RoPE("cpu", RopeSettings(
        head_dim=24,
        partial_rotary_factor=0.5,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 2, 2]},
    ))
    ids = torch.ones((1, sequence_length), dtype=torch.long)
    embeddings = []
    if image:
        grid = (1, 4, 4) if image_at_end else (3, 8, 12)
        count = grid[0] * (grid[1] // 2) * (grid[2] // 2)
        embedding = MMEmbedding(
            embeddings=torch.zeros((count, 1)),
            token_string=torch.full((1, count), -1, dtype=torch.long),
            grid_thw=grid,
            mrope_merge_size=2,
        )
        start = prompt_length - count if image_at_end else 248
        ids[:, start:start + count] = embedding.token_string
        embeddings = [embedding]

    # Match queue preparation: metadata covers the prompt, then only IDs grow.
    prompt = ids[:, :prompt_length].clone()
    job = Job(input_ids=prompt, embeddings=embeddings)
    seq = job.sequences[0]
    seq.sequence_ids.append(ids[:, prompt_length:])
    seq.kv_position = replay_start
    seq.block_index_tensor = torch.tensor([[0]], dtype=torch.int32)
    job.recurrent_state = SimpleNamespace(position=replay_start)
    job.generator = SimpleNamespace(
        model=ModelProbe(rope),
        max_chunk_size=PAGE_SIZE,
        recurrent_cache=object(),
        cache=None,
        draft_model=None,
        mtp_draft=mtp,
    )
    if atomic:
        # Gemma 4: chunks ending inside an image span are extended to cover the whole span
        job.generator.model.caps = {"atomic_mm_prefill": True}
    if image:
        job.alt_rope_freqs, next_position = rope.get_mrope_freqs(prompt, embeddings, prompt_length)
        job.alt_rope_offset = next_position - prompt_length
    else:
        job.alt_rope_freqs = None
        job.alt_rope_offset = 0
    rope.get_mrope_freqs = Mock(wraps=rope.get_mrope_freqs)
    return job


class MultimodalRewindPrefillTest(unittest.TestCase):
    def check_chunk(self, prompt_length, sequence_length, replay_start, replay_end, **kwargs):
        job = make_job(prompt_length, sequence_length, replay_start, **kwargs)
        old_table = job.alt_rope_freqs
        old_offset = job.alt_rope_offset
        with self.assertRaises(ModelBoundaryReached):
            job.prefill([])
        self.assertEqual(len(job.generator.model.calls), 1)
        ids, params = job.generator.model.calls[0]
        self.assertEqual(params["cache_seqlens"].item(), replay_start)
        self.assertEqual(ids.shape[-1], replay_end - replay_start)
        torch.testing.assert_close(ids, job.sequences[0].sequence_ids.torch_slice(replay_start, replay_end))
        self.assertEqual(job.alt_rope_offset, old_offset)
        table = params["inv_freq"]
        if old_table is None:
            self.assertIsNone(table)
            job.generator.model.g_rope.get_mrope_freqs.assert_not_called()
            return job, params

        # The table must cover the whole chunk, even if it starts inside the prompt.
        self.assertGreaterEqual(table.shape[-2], replay_end)
        self.assertIs(table, job.alt_rope_freqs)
        torch.testing.assert_close(table[:, :prompt_length], old_table, rtol=0, atol=0)
        if replay_end <= prompt_length:
            self.assertIs(table, old_table)
            job.generator.model.g_rope.get_mrope_freqs.assert_not_called()
        else:
            job.generator.model.g_rope.get_mrope_freqs.assert_called_once()
            positions = torch.arange(prompt_length, table.shape[-2]) + old_offset
            expected = positions.float()[None, :, None] * job.generator.model.g_rope.inv_freq[None, None, :]
            torch.testing.assert_close(table[:, prompt_length:], expected, rtol=0, atol=0)
        return job, params

    def test_replay_inside_prompt_preserves_multimodal_prefix(self):
        _, params = self.check_chunk(1125, 1175, 256, 512)
        self.assertEqual(params["mm_span_prefix"], 8)

    def test_replay_starts_inside_and_ends_beyond_prompt(self):
        _, params = self.check_chunk(1125, 1175, 1024, 1174)
        self.assertEqual(params["mm_span_prefix"], 0)

    def test_replay_starts_at_prompt_boundary(self):
        _, params = self.check_chunk(1024, 1175, 1024, 1174)
        self.assertEqual(params["mm_span_prefix"], 0)

    def test_prompt_boundary_preserves_trailing_multimodal_prefix(self):
        _, params = self.check_chunk(256, 301, 256, 300, image_at_end=True)
        self.assertEqual(params["mm_span_prefix"], 4)

    def test_replay_starts_beyond_prompt(self):
        for length in (1401, 1665, 1921):
            with self.subTest(sequence_length=length):
                _, params = self.check_chunk(1125, length, 1280, min(length - 1, 1536))
                self.assertEqual(params["mm_span_prefix"], 0)

    def test_chunk_ending_at_prompt_boundary_reuses_table(self):
        self.check_chunk(1024, 1175, 768, 1024)

    def test_text_only_replay_needs_no_multimodal_metadata(self):
        _, params = self.check_chunk(1125, 1401, 1280, 1400, image=False)
        self.assertEqual(params["mm_span_prefix"], 0)

    def test_mtp_target_receives_extended_table(self):
        _, params = self.check_chunk(1125, 1401, 1280, 1400, mtp=True)
        self.assertEqual(params["last_tokens_only"], 1)

    def test_builder_preserves_prefix_and_decode_offset_after_growth(self):
        job = make_job(1125, 1125, 1024)
        rope = job.generator.model.g_rope
        original = job.alt_rope_freqs
        ids = job.sequences[0].sequence_ids.torch()
        for length in (1175, 1401, 1921):
            with self.subTest(sequence_length=length):
                extended = torch.cat((ids, torch.ones((1, length - 1125), dtype=torch.long)), dim=-1)
                table, next_position = rope.get_mrope_freqs(extended, job.embeddings, length)
                self.assertEqual(table.shape[-2], length)
                self.assertEqual(next_position - length, job.alt_rope_offset)
                torch.testing.assert_close(table[:, :1125], original, rtol=0, atol=0)
                positions = torch.arange(1125, length) + job.alt_rope_offset
                expected = positions.float()[None, :, None] * rope.inv_freq[None, None, :]
                torch.testing.assert_close(table[:, 1125:], expected, rtol=0, atol=0)

    def test_atomic_replay_beyond_prompt_does_not_index_mask(self):
        # Gemma-style atomic MM prefill walks the mask forward from the chunk end; a rewind
        # replay chunk past the prompt must not index beyond the prompt-length mask
        for length in (1401, 1665):
            with self.subTest(sequence_length=length):
                _, params = self.check_chunk(1125, length, 1280, min(length - 1, 1536), atomic=True)
                self.assertEqual(params["mm_span_prefix"], 0)

    def test_atomic_chunk_inside_prompt_still_extends_over_image_span(self):
        # The image span [248, 320) straddles the first chunk's end at 256: the atomic path
        # must still extend the chunk to the end of the span
        job = make_job(1125, 1175, 0, atomic=True)
        with self.assertRaises(ModelBoundaryReached):
            job.prefill([])
        ids, params = job.generator.model.calls[0]
        self.assertEqual(ids.shape[-1], 320)
        self.assertEqual(params["cache_seqlens"].item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
