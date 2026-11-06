"""
Dynamic sparse LM head (modules/sparse_head.py) and the column extraction it rests on.

LinearEXL3.reconstruct_columns must hand back the columns of the dequantized weight (within fp16 rounding of
both the python and the fused dequant references) for any set of columns, across block-group boundaries. On top of it, SparseHead must:
fill its capacity from the observed counts, answer the drafter's argmax with the full head's token wherever that
token is in the subset (with the same confidence), keep its index tables consistent through swaps, respect the
swap margin (a challenger inside the margin leaves the incumbent in place, one beyond it does not) and count
target tokens outside the subset as misses.
"""

import pytest
import torch

from exllamav3.modules import Linear

from testlib.checkpoint import module_config
from testlib.exl3 import CODEBOOKS, checkpoint_tensors, dequant, generator

KEY = "lm_head"


def build(directory, device, k, n, K, codebook, bias, seed):
    t = checkpoint_tensors(KEY, k, n, K, generator(seed), codebook = codebook, bias = bias)
    module = Linear(module_config(t, directory), KEY, k, n)
    module.load(device)
    assert module.quant_type == "exl3"
    return module, {name: v.to(device) for name, v in t.items()}


@pytest.mark.parametrize("codebook", CODEBOOKS)
@pytest.mark.parametrize("max_blocks", [3, 64])
def test_reconstruct_columns(tmp_path, device, codebook, max_blocks):
    k, n, K = 512, 8192, 4
    module, t = build(tmp_path, device, k, n, K, codebook, False, 1)
    full_py = module.inner.get_weight_tensor()
    full_fused = dequant(t[f"{KEY}.trellis"], t[f"{KEY}.suh"], t[f"{KEY}.svh"], K, codebook)
    g = torch.Generator(device = "cpu").manual_seed(2)
    for m in (1, 17, 300):
        cols = torch.randperm(n, generator = g)[:m].to(device)
        out = torch.empty((k, m), dtype = torch.half, device = device)
        module.inner.reconstruct_columns(cols, out, max_blocks = max_blocks)
        # The python reference is the same fp32 math in a different accumulation order (fp16 ulps); the fused
        # kernel folds the Hadamards in fp16 and sits a few ulps away, so it is held at the RMS level
        assert (out.float() - full_py[:, cols].float()).abs().max().item() <= 1e-2
        ref = full_fused[:, cols].float()
        assert ((out.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt()).item() < 5e-3


@pytest.mark.parametrize("bias", [False, True])
def test_sparse_argmax_matches_full_head(tmp_path, device, bias):
    k, n, K = 512, 4096, 4
    module, _ = build(tmp_path, device, k, n, K, "mul1", bias, 3)
    cap = 512
    sh = module.make_sparse_head(cap, n, rebuild_interval = 64, min_evidence = 1024.0)
    g = torch.Generator(device = "cpu").manual_seed(4)
    # Zipf-like token stream over a shuffled vocabulary: a clear frequent subset plus a long tail
    ranks = torch.randperm(n, generator = g)
    probs = 1.0 / torch.arange(1, n + 1, dtype = torch.float)
    ids = ranks[torch.multinomial(probs, 4096, replacement = True, generator = g)]
    assert not sh.ready
    sh.observe(ids[:2048], sh.prompt_weight)
    sh.observe(ids[2048:], 1.0)
    sh.maintain()
    assert sh.ready and sh.n_active == cap and sh.evidence > 1024
    # Index tables agree in both directions
    cols = sh.cols[:cap]
    assert torch.equal(sh.member[cols], torch.arange(cap, device = device))
    assert (sh.member >= 0).sum().item() == cap

    x = torch.randn((64, k), generator = g).half().to(device) * 0.1
    full = module.forward(x, {})[..., :n].float()
    top2 = full.topk(2, dim = -1).values
    clear = (top2[:, 0] - top2[:, 1]) > 0.02 * top2[:, 0].abs().clamp(min = 1e-3)
    full_ids = full.argmax(dim = -1)
    sp_ids, sp_conf = sh.argmax(x)
    in_subset = sh.member[full_ids] >= 0
    check = clear & in_subset
    assert check.sum().item() > 0
    assert torch.equal(sp_ids[check], full_ids[check])
    assert torch.allclose(sp_conf[check], full.max(dim = -1).values[check], rtol = 2e-2, atol = 2e-2)
    # Every answer is a member of the subset
    assert (sh.member[sp_ids] >= 0).all()
    # Top-k: equals the full head's top-k restricted to the subset, wherever that set is clear of the next one
    k = 8
    masked = full.clone()
    masked[:, sh.member < 0] = float("-inf")
    ref_topk = masked.topk(k + 1, dim = -1)
    clear_k = (ref_topk.values[:, k - 1] - ref_topk.values[:, k]) > 0.02 * ref_topk.values[:, 0].abs().clamp(min = 1e-3)
    unary, cands = sh.topk(x, k)
    assert unary.shape == (x.shape[0], k) and cands.shape == (x.shape[0], k)
    rows = clear_k.nonzero().squeeze(1)
    assert rows.numel() > 0
    for r in rows.tolist():
        assert set(cands[r].tolist()) == set(ref_topk.indices[r, :k].tolist())
        assert torch.allclose(unary[r].sort().values, ref_topk.values[r, :k].sort().values, rtol = 2e-2, atol = 2e-2)


def test_swap_margin_and_misses(tmp_path, device):
    k, n, K = 256, 2048, 3
    module, _ = build(tmp_path, device, k, n, K, "mul1", False, 5)
    cap = 64
    sh = module.make_sparse_head(cap, n, rebuild_interval = 1, decay_tokens = 10**9, swap_margin = 0.5, min_evidence = 1.0)
    # Incumbents: ids 0..63, ten observations each
    sh.observe(torch.arange(cap).repeat(10), 1.0)
    sh.maintain()
    assert sh.n_active == cap and sh.ready
    installed = sh.swapped
    # A challenger inside the margin (14 < 10 * 1.5) does not displace anyone; its tokens stay misses
    sh.observe(torch.full((14,), 1000), 1.0)
    sh.maintain()
    assert sh.swapped == installed
    assert sh.member[1000].item() < 0
    assert sh.stats()["misses"] == 14
    # Beyond the margin it replaces exactly one incumbent, and the tables stay consistent
    sh.observe(torch.full((3,), 1000), 1.0)
    sh.maintain()
    assert sh.swapped == installed + 1
    slot = sh.member[1000].item()
    assert slot >= 0 and sh.cols[slot].item() == 1000
    assert (sh.member >= 0).sum().item() == cap
    assert torch.equal(sh.member[sh.cols[:cap]], torch.arange(cap, device = device))
    # Its column is the real one
    w = torch.empty((k, 1), dtype = torch.half, device = device)
    module.inner.reconstruct_columns(torch.tensor([1000], device = device), w)
    assert torch.equal(sh.weight[:, slot], w[:, 0])


def test_buffer_growth(tmp_path, device):
    """The column buffer grows in steps as distinct ids arrive; the argmax stays correct across a growth boundary"""
    from exllamav3.modules.sparse_head import _ALLOC_STEP
    k, n, K = 256, 4096, 3
    module, _ = build(tmp_path, device, k, n, K, "mul1", False, 6)
    cap = 2 * _ALLOC_STEP
    sh = module.make_sparse_head(cap, n, rebuild_interval = 1, min_evidence = 1.0)
    first = torch.arange(0, _ALLOC_STEP - 10)
    sh.observe(first.repeat(2), 1.0); sh.maintain()
    assert sh.n_active == first.numel() and sh.n_alloc == _ALLOC_STEP
    second = torch.arange(_ALLOC_STEP - 10, _ALLOC_STEP + 500)
    sh.observe(second, 1.0); sh.maintain()
    assert sh.n_active == _ALLOC_STEP + 500 and sh.n_alloc == cap
    assert torch.equal(sh.member[sh.cols[:sh.n_active]], torch.arange(sh.n_active, device = device))
    x = torch.randn((32, k), generator = torch.Generator().manual_seed(7)).half().to(device) * 0.1
    full = module.forward(x, {})[..., :n].float()
    full_ids = full.argmax(dim = -1)
    sp_ids, _ = sh.argmax(x)
    inside = sh.member[full_ids] >= 0
    assert torch.equal(sp_ids[inside], full_ids[inside])
    assert (sh.member[sp_ids] >= 0).all()


def test_cost_model_hysteresis(tmp_path, device):
    """The sparse mode switches off when the attributable miss rate exceeds the break-even rate by the margin
    and back on when it falls under it; probe rounds keep being issued either way"""
    k, n, K = 256, 2048, 3
    module, _ = build(tmp_path, device, k, n, K, "mul1", False, 8)
    sh = module.make_sparse_head(64, n, rebuild_interval = 64, decay_tokens = 10**9, min_evidence = 1.0, probe_interval = 10)
    sh.observe(torch.arange(64).repeat(4), 1.0); sh.maintain()
    assert sh.ready
    # saving 0.3 ms per draft token, 20 ms rounds: break-even miss rate 3%
    sh.t_full, sh.t_sparse = 0.00035, 0.00005
    sh.note_round(0.020, 4, 4)
    assert abs(sh.breakeven_miss_rate - 0.03) < 1e-9
    # 20% of target tokens outside the subset for a few hundred tokens, each id seen once so none earns a
    # column (a token the rebuild installs is a transient miss and does not count): off
    for i in range(8):
        sh.observe(torch.cat([torch.arange(64)[:52], torch.arange(1000 + 12 * i, 1012 + 12 * i)]), 1.0); sh.maintain()
    assert sh.miss_rate > 0.03 * 1.25 and not sh.mode_sparse and not sh.ready
    flags = [sh.round_flag() for _ in range(20)]
    assert flags.count("probe") == 2 and all(f in (False, "probe") for f in flags)
    # all inside for long enough: back on
    for _ in range(40):
        sh.observe(torch.arange(64), 1.0); sh.maintain()
    assert sh.mode_sparse and sh.ready and sh.mode_switches == 2
    flags = [sh.round_flag() for _ in range(20)]
    assert flags.count("probe") == 2 and flags.count(True) == 18
    # probe bookkeeping
    full_ids = torch.tensor([0, 1, 1500], device = device)
    sh.record_probe(full_ids, torch.tensor([0, 2, 3], device = device))
    assert sh.probes == 3 and sh.probe_disagree_inside == 1 and sh.probe_disagree_outside == 1


def test_profile_roundtrip(tmp_path, device):
    """Counts saved by one head are restored by another, which fills its subset from them at construction"""
    k, n, K = 256, 2048, 3
    module, _ = build(tmp_path, device, k, n, K, "mul1", False, 9)
    path = str(tmp_path / "profile.pt")
    sh = module.make_sparse_head(32, n, rebuild_interval = 1, min_evidence = 1.0, profile_path = path)
    sh.observe(torch.arange(100, 132).repeat(3), 1.0); sh.maintain()
    sh.save_profile()
    assert __import__("os").path.exists(path)
    sh2 = module.make_sparse_head(32, n, rebuild_interval = 1, min_evidence = 1.0, profile_path = path)
    assert sh2.n_active == 32 and sh2.ready
    assert torch.equal(torch.sort(sh2.cols[:32]).values, torch.arange(100, 132, device = device))
    assert torch.equal(sh2.counts, sh.counts)
    # A session shorter than the minimum does not overwrite the file
    sh2.observe(torch.arange(5), 1.0); sh2.maintain()
    mtime = __import__("os").path.getmtime(path)
    sh2.save_profile()
    assert __import__("os").path.getmtime(path) == mtime


class _FakeEmbedding:
    """Stands in for the model's Embedding module: a table on the device, the module's forward signature"""

    def __init__(self, vocab, hidden, device):
        self.hidden_size = hidden
        self.table = torch.randn((vocab, hidden), generator = torch.Generator().manual_seed(11)).half().to(device)

    def forward(self, ids, params, out_dtype = None):
        return self.table[ids.to(self.table.device)].to(out_dtype or torch.half)


@pytest.mark.parametrize("bias", [False, True])
def test_fused_argmax_matches_torch_path(tmp_path, device, bias):
    """The fused kernel (GEMV, bias, mask, argmax, id map, embedding gather) agrees with the torch path for 1..8
    rows, with the buffer only partly filled, and hands back the winners' embedding rows"""
    from exllamav3.ext import exllamav3_ext as ext
    if not hasattr(ext, "sparse_head_argmax"):
        pytest.skip("fused sparse head kernel not built")
    k, n, K = 512, 4096, 4
    module, _ = build(tmp_path, device, k, n, K, "mul1", bias, 12)
    emb = _FakeEmbedding(n, k, device)
    sh = module.make_sparse_head(1500, n, rebuild_interval = 1, min_evidence = 1.0, embedding = emb)
    assert sh.fused and sh.emb_rows is not None
    g = torch.Generator().manual_seed(13)
    ids = torch.randperm(n, generator = g)[:1100]
    sh.observe(ids, 1.0); sh.maintain()
    assert sh.n_active == 1100 and sh.n_alloc == 1536 and sh.n_alloc % 128 == 0
    assert torch.equal(sh.emb_rows[:1100], emb.table[sh.cols[:1100]])
    for m in (1, 2):
        x = torch.randn((m, k), generator = g).half().to(device) * 0.1
        ids_f, conf_f, emb_f = sh.argmax(x, embedding = True)
        sh.fused = False
        ids_t, conf_t, emb_t = sh.argmax(x, embedding = True)
        sh.fused = True
        # Differences can only come from near-ties under a different fp32 summation order
        logits = sh._transform(sh._logits(x))
        top2 = logits.topk(2, dim = -1).values
        clear = (top2[:, 0] - top2[:, 1]) > 1e-3
        assert torch.equal(ids_f[clear], ids_t[clear])
        assert torch.allclose(conf_f, conf_t, rtol = 1e-3, atol = 1e-3)
        assert torch.equal(emb_f[clear], emb_t[clear])
        assert torch.equal(emb_f, emb.table[ids_f])
    # Three-dimensional input keeps its leading shape
    x = torch.randn((2, 3, k), generator = g).half().to(device) * 0.1
    ids_f, conf_f, emb_f = sh.argmax(x, embedding = True)
    assert ids_f.shape == (2, 3) and conf_f.shape == (2, 3) and emb_f.shape == (2, 3, k)
