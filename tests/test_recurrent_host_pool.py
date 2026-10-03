"""
Recurrent checkpoint host buffers (cache/recurrent.py HostPool): stashes copy into pooled host
buffers that eviction returns, so the allocator sees no per-checkpoint churn (issue #432).
"""
import sys, os, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.cache.recurrent import HostPool, host_copy, host_pool


class TestHostPool(unittest.TestCase):

    def test_take_reuses_exact_shape_and_dtype(self):
        pool = HostPool()
        a = pool.take((4, 8), torch.float32)
        b = pool.take((4, 8), torch.float32)
        self.assertEqual(pool.allocated, 2)
        pool.give(a)
        c = pool.take((4, 8), torch.float32)
        self.assertIs(c, a)
        self.assertEqual((pool.allocated, pool.reused), (2, 1))
        # neither a different shape nor a different dtype may be served from a free buffer
        pool.give(c)
        d = pool.take((8, 4), torch.float32)
        e = pool.take((4, 8), torch.float16)
        self.assertEqual(pool.allocated, 4)
        self.assertEqual(d.shape, (8, 4)); self.assertEqual(e.dtype, torch.float16)
        self.assertIsNot(e, c); self.assertIs(pool.take((4, 8), torch.float32), c)

    def test_give_walks_a_stashed_structure(self):
        pool = HostPool()
        t = [pool.take((3,), torch.float32) for _ in range(4)]
        stashed = {"position": 2048, "checkpoint_size": 12345, "tp_handle": 7,
                   "layer_a": (t[0], t[1]), "layer_b": [t[2], (t[3],)]}
        pool.give(stashed)
        self.assertEqual(sum(len(v) for v in pool.free.values()), 4)
        ids = {id(x) for x in t}
        for _ in range(4):
            self.assertIn(id(pool.take((3,), torch.float32)), ids)
        self.assertEqual(pool.reused, 4)
        # device tensors are never pooled
        if torch.cuda.is_available():
            pool.give(torch.zeros(3, device = "cuda:0"))
            self.assertEqual(sum(len(v) for v in pool.free.values()), 0)

    def test_release_drops_idle_buffers_only(self):
        pool = HostPool()
        held = pool.take((5,), torch.float32)
        idle = pool.take((5,), torch.float32)
        pool.give(idle)
        pool.release()
        self.assertEqual(sum(len(v) for v in pool.free.values()), 0)
        self.assertIsNot(pool.take((5,), torch.float32), idle)
        held.fill_(1.0)                       # the buffer still in use is untouched

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_host_copy_matches_cpu_for_strided_sources(self):
        before = host_pool.allocated
        state = torch.randn(4, 6, 16, 16, device = "cuda:0")
        conv = torch.randn(4, 24, 8, device = "cuda:0")
        for src in (state[2, :1], conv[1, :, :4], state[3]):
            out = host_copy(src)
            self.assertEqual(out.device.type, "cpu")
            self.assertTrue(torch.equal(out, src.cpu()))
            self.assertEqual(out.shape, src.shape)
        self.assertEqual(host_pool.allocated, before + 3)
        # a second stash of the same shapes reuses the buffers once the first is returned
        host_pool.give([host_copy(state[0, :1])])
        host_copy(state[1, :1])
        self.assertGreaterEqual(host_pool.reused, 1)
        host_pool.release()


if __name__ == "__main__":
    unittest.main()
