import os, sys, unittest
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


if __name__ == "__main__":
    unittest.main()
