"""
GatedResidual prefill mix (ext.gr_mix_tiled, hc_mix_tiled.cu): the tiled deterministic kernels
must match the fp32 torch reference to half-precision tolerance on every proj padding class,
row count and both module forms, agree with the cuBLAS GEMM path to the same tolerance, and be
bit-reproducible run to run (the property the TP replicated-routing design relies on).
"""
import sys, os, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.modules.hyperconnections import GatedResidual

DEVICE = torch.device("cuda:0")


def make_site(D, rank, use_combine, seed = 0):
    torch.manual_seed(seed)
    H = 4
    m = GatedResidual(config = None, key = "site", hc_mult = H, hidden_size = D,
                      rms_norm_eps = 1e-6, use_combine = use_combine)
    m.device = DEVICE
    m.norm_w_raw = (torch.randn(H * D, device = DEVICE) * 0.1)
    down = torch.randn(rank, H * D, device = DEVICE) * (1.0 / (H * D) ** 0.5)
    up = torch.randn(H * D, rank, device = DEVICE) * (1.0 / rank ** 0.5)
    inject = torch.randn(H, H * D, device = DEVICE) * (1.0 / (H * D) ** 0.5) if use_combine else None
    m._prepare(down, up, inject, keep_source_weights = True)
    return m


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGrMixTiled(unittest.TestCase):

    CASES = [
        # (D, rank, use_combine): padded proj rows 64 (NW 1), 128, 384 (Qwen3.8 shape), 512
        (256, 64, False),
        (256, 64, True),
        (1024, 320, True),
        (512, 512, False),
        (512, 448, True),
    ]
    ROWS = [33, 64, 100, 257, 1000, 2048]

    def test_tiled_matches_reference_and_gemm(self):
        for D, rank, use_combine in self.CASES:
            m = make_site(D, rank, use_combine)
            self.assertTrue(m.tiled)
            for R in self.ROWS:
                torch.manual_seed(R)
                x = torch.randn(1, R, 4, D, device = DEVICE) * 3.0
                ref_post, ref_mixed = m._mix_ref(x)
                ref_mixed = ref_mixed.view(R, D)
                post, mixed = m._mix(x)
                m.tiled = False
                post_g, mixed_g = m._mix(x)
                m.tiled = True
                self.assertLess(rel(mixed, ref_mixed), 3e-3, (D, rank, use_combine, R))
                self.assertLess(rel(mixed, mixed_g), 3e-3, (D, rank, use_combine, R))
                if use_combine:
                    self.assertLess(rel(post, ref_post.view(R, 4)), 1e-3, (D, rank, use_combine, R))
                    self.assertLess(rel(post, post_g), 1e-3, (D, rank, use_combine, R))
                else:
                    self.assertIsNone(post)

    def test_fused_decode_matches_reference(self):
        # The fused decode pair (R <= FUSED_MAX_R) on the Qwen3.8 shape and a larger one
        for D, rank in ((2560, 320), (4096, 512), (1024, 320)):
            m = make_site(D, rank, True)
            for R in (1, 2, 5, 8):
                torch.manual_seed(R)
                x = torch.randn(1, R, 4, D, device = DEVICE) * 3.0
                ref_post, ref_mixed = m._mix_ref(x)
                post, mixed = m._mix(x, cached = False)
                self.assertLess(rel(mixed, ref_mixed.view(R, D)), 3e-3, (D, rank, R))
                self.assertLess(rel(post, ref_post.view(R, 4)), 1e-3, (D, rank, R))
                post2, mixed2 = m._mix(x, cached = False)
                self.assertTrue(torch.equal(mixed, mixed2) and torch.equal(post, post2))

    def test_tiled_is_deterministic(self):
        m = make_site(1024, 320, True)
        for R in (100, 2048):
            x = torch.randn(1, R, 4, 1024, device = DEVICE) * 3.0
            post_a, mixed_a = m._mix(x)
            post_a, mixed_a = post_a.clone(), mixed_a.clone()
            for _ in range(3):
                # different workspaces and other work in between must not change a bit
                torch.randn(2048, 2048, device = DEVICE) @ torch.randn(2048, 2048, device = DEVICE)
                post_b, mixed_b = m._mix(x)
                self.assertTrue(torch.equal(mixed_a, mixed_b))
                self.assertTrue(torch.equal(post_a, post_b))

    def test_shape_fallback(self):
        # rank not a multiple of 64 -> cuBLAS path, still correct
        m = make_site(256, 96, True)
        self.assertFalse(m.tiled)
        x = torch.randn(1, 200, 4, 256, device = DEVICE)
        ref_post, ref_mixed = m._mix_ref(x)
        post, mixed = m._mix(x)
        self.assertLess(rel(mixed, ref_mixed.view(200, 256)), 3e-3)
        self.assertLess(rel(post, ref_post.view(200, 4)), 1e-3)


    def test_cross_device_identity(self):
        # The int8 tensor-core path must agree bit for bit across GPU architectures
        n = torch.cuda.device_count()
        if n < 2:
            self.skipTest("needs two or more GPUs")
        torch.manual_seed(5)
        D, rank = 1024, 320
        H = 4
        norm = torch.randn(H * D) * 0.1
        down = torch.randn(rank, H * D) / (H * D) ** 0.5
        up = torch.randn(H * D, rank) / rank ** 0.5
        inject = torch.randn(H, H * D) / (H * D) ** 0.5
        for R in (100, 1000):
            x = torch.randn(1, R, H, D) * 3.0
            outs = []
            for d in range(n):
                dev = torch.device(f"cuda:{d}")
                m = GatedResidual(config = None, key = "site", hc_mult = H, hidden_size = D,
                                  rms_norm_eps = 1e-6, use_combine = True)
                m.device = dev
                m.norm_w_raw = norm.to(dev)
                m._prepare(down.to(dev), up.to(dev), inject.to(dev), keep_source_weights = True)
                post, mixed = m._mix(x.to(dev))
                outs.append((post.cpu(), mixed.cpu()))
            for d in range(1, n):
                self.assertTrue(torch.equal(outs[0][0], outs[d][0]), (R, torch.cuda.get_device_name(d)))
                self.assertTrue(torch.equal(outs[0][1], outs[d][1]), (R, torch.cuda.get_device_name(d)))


    def test_source_weights_released(self):
        # Inference loads drop the fp16 sources once the kernel tables exist; conversion keeps them
        m = make_site(256, 64, True)
        self.assertIsNotNone(m.down_h)
        torch.manual_seed(0)
        m2 = GatedResidual(config = None, key = "site", hc_mult = 4, hidden_size = 256, rms_norm_eps = 1e-6, use_combine = True)
        m2.device = DEVICE
        m2.norm_w_raw = torch.randn(4 * 256, device = DEVICE) * 0.1
        m2._prepare(torch.randn(64, 1024, device = DEVICE), torch.randn(1024, 64, device = DEVICE), torch.randn(4, 1024, device = DEVICE))
        self.assertTrue(m2.tiled)
        self.assertIsNone(m2.down_h); self.assertIsNone(m2.proj_h); self.assertIsNone(m2.up_h)
        self.assertIsNotNone(m2.proj_i8); self.assertIsNotNone(m2.fn_h)
        x = torch.randn(1, 100, 4, 256, device = DEVICE)
        m2._mix(x)                                   # tiled path still works
        m2._mix(x[:, :8])                            # fused decode path too
        with self.assertRaises(AssertionError):
            m2.get_tensors()


if __name__ == "__main__":
    unittest.main()
