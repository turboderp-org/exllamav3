"""ext-vs-eager parity for the native DFlash2 kernels (dflash2.cu).

Gate for ANY dispatch wiring: exact token equality on fixed tensors, then on
live greedy rounds. Scores get tolerance; token paths get torch.equal -- the
walk is autoregressive, so one divergent step cascades (the searchsorted lesson).

Gated on the ext symbols: skips cleanly on trees without the sm_120 build.
"""
import pytest
import torch

ext = pytest.importorskip("exllamav3_ext", reason="ext not built")
pytestmark = pytest.mark.skipif(
    not all(hasattr(ext, s) for s in ("dflash2_dynconv", "dflash2_selector_walk", "dflash2_topk")),
    reason="dflash2 native kernels absent",
)

from exllamav3.modules.arch_specific.dflash2 import grouped_dynamic_convolve


def test_dispatch_engages_ext_when_built():
    # the one-dot import bug (..ext vs ...ext) silently disabled dispatch while
    # unit tests hit the .pyd directly. If the symbols exist, dispatch must engage.
    from exllamav3.modules.arch_specific import dflash2 as D
    import importlib.util
    spec = importlib.util.find_spec("exllamav3_ext")
    if spec is None:
        pytest.skip("no .pyd at all")
    import exllamav3_ext as e
    if not all(hasattr(e, s) for s in ("dflash2_dynconv", "dflash2_selector_walk", "dflash2_topk")):
        pytest.skip("symbols absent")
    assert D._get_ext() is not None, "dispatch dead while kernels built"

DEV = "cuda"
H, K, GROUPS = 512, 2, 8
V, TOPK, RANK, T = 1024, 16, 32, 7


def test_topk_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(1, T, V, device=DEV, dtype=torch.float)
    values = torch.empty((1, T, TOPK), dtype=torch.float, device=DEV)
    indices = torch.empty((1, T, TOPK), dtype=torch.long, device=DEV)
    ext.dflash2_topk(logits, V, 1.0, 0.0, values, indices)
    ref_v, ref_i = torch.topk(logits, TOPK, dim=-1, sorted=False)
    # same SET per row (order may differ when sorted=False on both sides)
    assert torch.equal(values.sort(-1).values, ref_v.sort(-1).values)
    assert torch.equal(
        values.gather(-1, values.argsort(-1)).sort(-1).values,
        values.sort(-1).values,
    )
    # indices must point at the values they claim
    assert torch.equal(logits.gather(-1, indices).sort(-1).values, ref_v.sort(-1).values)


def test_dynconv_matches_eager():
    torch.manual_seed(0)
    x = torch.randn(1, T, H, device=DEV, dtype=torch.half)
    dyn = torch.randn(1, T, K, GROUPS, device=DEV, dtype=torch.half)
    base = torch.randn(K, H, device=DEV, dtype=torch.half)
    out = torch.empty_like(x)
    ext.dflash2_dynconv(x, dyn, base, out, H // GROUPS, False)
    ref = grouped_dynamic_convolve(x, dyn, base, H // GROUPS)
    assert (out.float() - ref.float()).abs().max() < 1e-2


def test_walk_matches_torch_path():
    torch.manual_seed(0)
    unary = torch.randn(1, T, TOPK, device=DEV, dtype=torch.float).contiguous()
    cands = torch.randint(0, V, (1, T, TOPK), device=DEV, dtype=torch.long).contiguous()
    gate = torch.randn(1, T, RANK, device=DEV, dtype=torch.float).contiguous()
    pred_cb = torch.randn(V, RANK, device=DEV, dtype=torch.half)
    succ_cb = torch.randn(V, RANK, device=DEV, dtype=torch.half)
    anchor = torch.randint(0, V, (1,), device=DEV, dtype=torch.long)
    out = torch.empty((1, T + 1), dtype=torch.long, device=DEV)
    ext.dflash2_selector_walk(unary, cands, gate, pred_cb, succ_cb, anchor, out, None)
    # greedy reference walk in plain torch (mirrors selector_select greedy leg)
    pred = anchor.clone()
    path = [pred]
    for i in range(T):
        a = pred_cb[pred].float()
        b = succ_cb[cands[:, i]].float()
        scores = unary[:, i] + torch.einsum("br,bkr->bk", a * gate[:, i].float(), b)
        pred = cands[:, i].gather(-1, scores.argmax(-1, keepdim=True))[:, 0]
        path.append(pred)
    ref = torch.stack(path, dim=1)
    assert torch.equal(out, ref)


def test_topk_tied_scores_order_irrelevant_to_walk():
    # ties may order differently between ext and torch topk; the walk consumes
    # candidate ORDER, so verify the walk itself is order-faithful: same
    # (unary, cands) in -> same path out, even with heavy ties upstream.
    torch.manual_seed(1)
    unary = torch.zeros(1, T, TOPK, device=DEV, dtype=torch.float).contiguous()
    cands = torch.randint(0, V, (1, T, TOPK), device=DEV, dtype=torch.long).contiguous()
    gate = torch.zeros(1, T, RANK, device=DEV, dtype=torch.float).contiguous()
    pred_cb = torch.randn(V, RANK, device=DEV, dtype=torch.half)
    succ_cb = torch.randn(V, RANK, device=DEV, dtype=torch.half)
    anchor = torch.zeros((1,), device=DEV, dtype=torch.long)
    out = torch.empty((1, T + 1), dtype=torch.long, device=DEV)
    ext.dflash2_selector_walk(unary, cands, gate, pred_cb, succ_cb, anchor, out, None)
    # all-unary-zero + zero gate -> scores all zero -> argmax picks index 0 every row
    assert torch.equal(out[:, 1:], cands[:, :, 0])
