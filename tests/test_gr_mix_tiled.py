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
from exllamav3.ext import exllamav3_ext as ext

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


@unittest.skipUnless(torch.cuda.is_available() and ext.HAS_GR_MIX_TILED, "CUDA build with the tiled mix kernels required")
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
        # One table set serves both kernel paths: the fp16 projection stays (the decode kernel
        # reads it and the tiled path derives its int8 tables per call), the checkpoint-layout
        # up goes (the repacked copy remains)
        self.assertIsNone(m2.up_h)
        self.assertIsNotNone(m2.proj_h); self.assertIsNotNone(m2.upx_h)
        self.assertFalse(any(k for k in vars(m2) if k in ("fn_h", "proj_i8", "up_i8")))
        x = torch.randn(1, 100, 4, 256, device = DEVICE)
        m2._mix(x)                                   # tiled path still works
        m2._mix(x[:, :8])                            # fused decode path too
        with self.assertRaises(AssertionError):
            m2.get_tensors()
        with self.assertRaises(AssertionError):
            m2._mix_ref(x)

    def test_tiled_tables_match_stored(self):
        # The per-call int8 derivation reproduces what a load-time copy would hold, bit for bit
        m = make_site(1024, 320, True)
        from exllamav3.ext import exllamav3_ext as ext
        def ws(shape, dtype):
            return torch.empty(shape, dtype = dtype, device = DEVICE)
        a = m._tiled_tables(ws)
        b = m._tiled_tables(ws)
        Mpad = m.proj_h.shape[0]
        ref_i8 = torch.empty((2, Mpad, 4 * 1024), dtype = torch.int8, device = DEVICE)
        ref_sb = torch.empty((Mpad,), dtype = torch.float, device = DEVICE)
        ext.det_quant_weight(m.proj_h, ref_i8, ref_sb)
        up_i8 = torch.empty((2, 4 * 1024, 320), dtype = torch.int8, device = DEVICE)
        up_sb = torch.empty((4 * 1024,), dtype = torch.float, device = DEVICE)
        ext.det_quant_weight(m.up_h, up_i8, up_sb)
        for x, y in zip(a, (ref_i8, ref_sb, up_i8, up_sb)):
            self.assertTrue(torch.equal(x, y))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))

    def test_weighted_copy_handoff(self):
        # apply_ of one site emits the next site's weighted stream copy; the next mix must use
        # it and agree with the in-kernel weighting to fp32 rounding, and both with the reference
        for D, rank in ((1024, 320), (2560, 320)):     # generic and shape-templated dots kernels
            a, b = make_site(D, rank, True, seed = 1), make_site(D, rank, True, seed = 2)
            GatedResidual.link_sites([a, b])
            self.assertIs(a.next_site, b); self.assertIsNone(b.next_site)
            for R in (1, 3, 8):
                torch.manual_seed(R)
                x = torch.randn(1, R, 4, D, device = DEVICE) * 3.0
                y = torch.randn(1, R, D, device = DEVICE, dtype = torch.half)
                post = torch.rand(1, R, 4, device = DEVICE) * 2
                params = {}
                x_ref = x + post.unsqueeze(-1) * y.float().unsqueeze(-2)
                a.apply_(x, y, post, None, params)
                self.assertLess(rel(x, x_ref), 1e-6)            # (the kernel's update contracts to an FMA)
                ent = params["gr_weighted"]
                self.assertIs(ent[0], b)
                self.assertTrue(torch.equal(ent[2], x.view(R, 4, D) * b.w_h.float().view(1, 4, D)))
                post_h, _, mixed_h = (t.clone() if t is not None else None for t in b.mix(x, params))   # consumes the copy
                self.assertNotIn("gr_weighted", params)
                post_k, _, mixed_k = (t.clone() if t is not None else None for t in b.mix(x, {}))       # in-kernel weighting
                ref_post, ref_mixed = b._mix_ref(x)
                self.assertLess(rel(mixed_h, ref_mixed.view(R, D)), 3e-3, R)
                self.assertLess(rel(post_h.view(R, 4), ref_post.view(R, 4)), 1e-3, R)
                self.assertLess(rel(mixed_h, mixed_k), 1e-3, R)
                self.assertLess(rel(post_h, post_k), 1e-4, R)
                # and the kernel really reads the copy: a poisoned one changes the gates
                a.apply_(x, torch.zeros_like(y), post, None, params)
                params["gr_weighted"][2].mul_(0.0)
                post_p, _, _ = b.mix(x, params)
                self.assertGreater(rel(post_p, post_k), 1e-2, R)
            # Prefill row counts take the tiled path and emit no copy
            R = 100
            x = torch.randn(1, R, 4, D, device = DEVICE); y = torch.randn(1, R, D, device = DEVICE, dtype = torch.half)
            params = {}
            a.apply_(x, y, torch.rand(1, R, 4, device = DEVICE), None, params)
            self.assertNotIn("gr_weighted", params)

    def test_weighted_copy_rejected_when_stale(self):
        # A copy for another site, another stream tensor or another device is ignored (and
        # consumed), and the mix falls back to the in-kernel weighting
        D, rank = 1024, 320
        a, b = make_site(D, rank, True, seed = 1), make_site(D, rank, True, seed = 2)
        GatedResidual.link_sites([a, b])
        R = 4
        x = torch.randn(1, R, 4, D, device = DEVICE) * 3.0
        y = torch.randn(1, R, D, device = DEVICE, dtype = torch.half)
        params = {}
        a.apply_(x, y, torch.rand(1, R, 4, device = DEVICE), None, params)
        ref_post, ref_mixed = b._mix_ref(x)
        # same site, different stream tensor (a moved copy): must not use the entry
        x2 = x.clone()
        xw = params["gr_weighted"][2]
        xw.fill_(0)                                   # a used copy would give garbage
        post, _, mixed = b.mix(x2, params)
        self.assertNotIn("gr_weighted", params)
        self.assertLess(rel(mixed, ref_mixed.view(R, D)), 3e-3)
        # entry addressed to another site
        a.apply_(x, y, torch.zeros(1, R, 4, device = DEVICE), None, params)
        params["gr_weighted"][2].fill_(0)
        post, _, mixed = a.mix(x, params)
        self.assertNotIn("gr_weighted", params)
        self.assertLess(rel(mixed, a._mix_ref(x)[1].view(R, D)), 3e-3)

    def test_link_sites_chain(self):
        class Block:
            def __init__(self, attn_hc, mlp_hc):
                self.attn_hc, self.mlp_hc = attn_hc, mlp_hc
        class Other:
            pass
        s = [make_site(256, 64, True, seed = i) for i in range(6)]
        mixer = make_site(256, 64, False, seed = 9)
        GatedResidual.link_sites([Other(), Block(s[0], s[1]), Block(s[2], s[3]), Other(), Block(s[4], s[5]), mixer])
        self.assertIs(s[0].next_site, s[1]); self.assertIs(s[1].next_site, s[2]); self.assertIs(s[2].next_site, s[3])
        self.assertIsNone(s[3].next_site)              # the foreign module breaks the chain
        self.assertIs(s[4].next_site, s[5]); self.assertIs(s[5].next_site, mixer); self.assertIsNone(mixer.next_site)


if __name__ == "__main__":
    unittest.main()
