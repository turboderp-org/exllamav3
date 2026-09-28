import os, sys, unittest
from unittest.mock import patch
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.generator.job import Job


class JobParamsTest(unittest.TestCase):

    def test_max_new_tokens_none_and_exact(self):
        ids = torch.tensor([[1, 2, 3]])
        self.assertIsNone(Job(input_ids = ids).max_new_tokens)                 # documented default: resolved at enqueue
        self.assertIsNone(Job(input_ids = ids, max_new_tokens = None).max_new_tokens)
        for k in (1, 2, 3, 17):
            self.assertEqual(Job(input_ids = ids, max_new_tokens = k).max_new_tokens, k)   # no off-by-one, 1 != 2
        with self.assertRaises(AssertionError):
            Job(input_ids = ids, max_new_tokens = 0)

    def test_single_sequence_only(self):
        ids = torch.tensor([[1, 2, 3]])
        self.assertEqual(len(Job(input_ids = [ids]).sequences), 1)
        with self.assertRaises(AssertionError):
            Job(input_ids = [ids, ids.clone()])

    def test_requeue_carries_token_count(self):
        """new_tokens restarts in every requeued segment, so the count handed to the next segment has to
        include what earlier segments handed to this one"""
        job = Job(input_ids = torch.tensor([[1, 2, 3]]), max_new_tokens = 1000, max_rq_tokens = 256)
        job.cached_pages, job.cached_tokens = 0, 0
        total = 0
        with patch.object(Job, "prepare_for_queue"):
            for segment in (250, 256, 100):
                job.new_tokens = segment
                job.sequences[0].sequence_ids.append(torch.zeros((1, segment), dtype = torch.long))
                total += segment
                job = job.prepare_for_requeue()
                self.assertTrue(job.is_requeued)
                self.assertEqual(job.new_tokens, 0)
                self.assertEqual(job.rq_new_tokens, total)
                self.assertEqual(job.max_new_tokens, 1000 - total)
                self.assertEqual(job.rq_prompt_tokens, 3)

if __name__ == "__main__":
    unittest.main()
