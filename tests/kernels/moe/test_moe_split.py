"""
The CPU expert-split plumbing kernels (routing.cu) between BlockSparseMLP and the CPU MoE worker
(modules/block_sparse_mlp_cpu.py, model/moe_cpu_host.py):

moe_split_map(sel, map, hist, sel_cpu, first): for i < n = sel.numel(), with r = sel[i] (router id) and
p = map[r] (physical slot): hist[r] += 1, sel[i] = p in place, sel_cpu[i] = p - first if p >= first else -1.
sel_cpu past n, map and the other hist entries are untouched. Rejects wrong dtypes and a short sel_cpu.

moe_split_issue(sel, map, hist, y, w, h_sel, h_x, h_w, dev_count, slot, hi, first): stages one decode job into a
(mapped pinned) slot. n = sel.numel() picks over rows = y.shape[0] rows. With map (dynamic placement) it first does
moe_split_map's hist bump and in-place translate; then h_sel[i] = int32(p - first) or -1, and
dev_count[slot] = 1 if any pick is CPU-resident else 0 (other slots untouched). Only when some pick is CPU-resident:
h_w[:n] = w and h_x (rows, hi) = y zero-padded from h_ = y.shape[1] to hi columns (vectorized when h_ and hi are
multiples of 8, scalar otherwise); otherwise h_x and h_w are not written. Rejects non-contiguous inputs, map
without hist and wrong dtypes.

moe_split_collect_add(final, h_out, dev_count, slot, ho): if dev_count[slot] != 0, final (rows, h_) fp32 +=
h_out (rows, ho)[:, :h_]; otherwise nothing, and h_out is never read. Rejects non-fp32 or non-contiguous final
and a non-int32 dev_count.

References: plain torch indexing / bincount, exact (integer data, copies and single fp32 additions).
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.isolated import run_isolated


def _map_case(E, n, first, seed, device):
    g = torch.Generator().manual_seed(seed)
    sel = torch.randint(0, E, (n,), generator = g)
    perm = torch.randperm(E, generator = g)          # router id -> physical slot
    return sel.to(device), perm.to(device), first


@pytest.mark.parametrize("n", [1, 8, 1023, 1024, 1025, 3000])
@pytest.mark.parametrize("E, first", [(16, 12), (64, 40), (128, 0), (128, 128)])
@torch.inference_mode()
def test_moe_split_map(device, n, E, first):
    sel, pmap, first = _map_case(E, n, first, n * 7 + E, device)
    sel0, map0 = sel.clone(), pmap.clone()
    hist = torch.arange(E, dtype = torch.float, device = device)
    sel_cpu = torch.full((n + 5,), -99, dtype = torch.long, device = device)
    ext.moe_split_map(sel, pmap, hist, sel_cpu, first)
    p = map0[sel0]
    assert torch.equal(sel, p)
    assert torch.equal(sel_cpu[:n], torch.where(p >= first, p - first, -1))
    assert (sel_cpu[n:] == -99).all()
    assert torch.equal(hist, torch.arange(E, device = device).float() + torch.bincount(sel0, minlength = E).float())
    assert torch.equal(pmap, map0)


@torch.inference_mode()
def test_moe_split_map_rejections(device):
    sel = torch.zeros(8, dtype = torch.long, device = device)
    pmap = torch.arange(4, dtype = torch.long, device = device)
    hist = torch.zeros(4, device = device)
    with pytest.raises(RuntimeError, match = "sel_cpu too small"):
        ext.moe_split_map(sel, pmap, hist, torch.zeros(7, dtype = torch.long, device = device), 2)
    with pytest.raises(RuntimeError):
        ext.moe_split_map(sel.int(), pmap, hist, torch.zeros(8, dtype = torch.long, device = device), 2)
    with pytest.raises(RuntimeError):
        ext.moe_split_map(sel, pmap, hist.half(), torch.zeros(8, dtype = torch.long, device = device), 2)


class _Slot:
    """Staging buffers for one job: device memory, or pinned host memory passed by its device alias"""

    def __init__(self, rows, topk, hi, ho, pinned, device):
        n = rows * topk
        kw = dict(pin_memory = True) if pinned else dict(device = device)
        self.sel = torch.full((n + 4,), -77, dtype = torch.int32, **kw)
        self.x = torch.full((rows * hi + 8,), 5.0, dtype = torch.half, **kw)
        self.w = torch.full((n + 4,), 5.0, dtype = torch.half, **kw)
        self.out = torch.full((rows * ho,), 0.0, dtype = torch.float, **kw)
        self.pinned = pinned

    def ptr(self, t):
        return ext.cuda_host_get_device_pointer(t.data_ptr()) if self.pinned else t.data_ptr()


def _issue_case(rows, topk, h_, hi, E, first, any_cpu, seed, device):
    g = torch.Generator().manual_seed(seed)
    if any_cpu:
        sel = torch.randint(0, E, (rows, topk), generator = g)
        sel.view(-1)[0] = E - 1      # at least one CPU-resident pick under the identity / static maps below
    else:
        sel = torch.randint(0, first, (rows, topk), generator = g)
    y = torch.randn((rows, h_), generator = g).half()
    w = torch.rand((rows, topk), generator = g).half()
    return sel.to(device), y.to(device), w.to(device)


ISSUE_CASES = [
    # rows, topk, h_, hi
    (1, 8, 256, 256),
    (1, 8, 2880, 2944),      # zero-padded, vectorized
    (3, 6, 200, 256),        # vectorized pad
    (2, 4, 99, 128),         # scalar path (h_ odd)
    (5, 2, 120, 132),        # scalar path (hi not a multiple of 8)
    (200, 8, 64, 64),        # n = 1600 > one block of threads
]


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("use_map", [False, True])
@pytest.mark.parametrize("any_cpu", [True, False])
@pytest.mark.parametrize("rows, topk, h_, hi", ISSUE_CASES)
@torch.inference_mode()
def test_moe_split_issue(device, rows, topk, h_, hi, any_cpu, use_map, pinned):
    E, first, slot_idx = 32, 24, 2
    sel, y, w = _issue_case(rows, topk, h_, hi, E, first, any_cpu, rows * 31 + h_, device)
    n = rows * topk
    sel0 = sel.clone()
    if use_map:
        # Identity map keeps the CPU / GPU split of the generated picks; the translate is checked separately
        pmap = torch.arange(E, dtype = torch.long, device = device)
        hist = torch.full((E,), 2.0, device = device)
    else:
        pmap = hist = None
    slot = _Slot(rows, topk, hi, hi, pinned, device)
    dev_count = torch.full((4,), 9, dtype = torch.int32, device = device)
    ext.moe_split_issue(sel.view(-1), pmap, hist, y, w, slot.ptr(slot.sel), slot.ptr(slot.x), slot.ptr(slot.w),
                        dev_count, slot_idx, hi, first)
    torch.cuda.synchronize(device)

    p = sel0.view(-1)
    exp_sel = torch.where(p >= first, p - first, -1).int()
    assert torch.equal(slot.sel[:n].to(device), exp_sel)
    assert (slot.sel[n:] == -77).all()
    assert dev_count.tolist() == [9, 9, int(any_cpu), 9]
    assert torch.equal(sel, sel0)
    if use_map:
        assert torch.equal(hist, 2.0 + torch.bincount(p, minlength = E).float())
    xs, ws = slot.x.to(device), slot.w.to(device)
    if any_cpu:
        x_exp = torch.zeros((rows, hi), dtype = torch.half, device = device)
        x_exp[:, :h_] = y
        assert torch.equal(xs[: rows * hi].view(rows, hi), x_exp)
        assert torch.equal(ws[:n], w.view(-1))
    else:
        assert (xs[: rows * hi] == 5.0).all() and (ws[:n] == 5.0).all(), "payload written for a job with no CPU picks"
    assert (xs[rows * hi :] == 5.0).all() and (ws[n:] == 5.0).all()


@torch.inference_mode()
def test_moe_split_issue_translates(device):
    E, first, rows, topk = 64, 40, 7, 8
    sel, pmap, _ = _map_case(E, rows * topk, first, 5, device)
    sel0 = sel.clone()
    hist = torch.zeros(E, device = device)
    y = torch.randn((rows, 128), device = device).half()
    w = torch.rand((rows, topk), device = device).half()
    slot = _Slot(rows, topk, 128, 128, False, device)
    dev_count = torch.zeros((1,), dtype = torch.int32, device = device)
    ext.moe_split_issue(sel, pmap, hist, y, w, slot.ptr(slot.sel), slot.ptr(slot.x), slot.ptr(slot.w),
                        dev_count, 0, 128, first)
    p = pmap[sel0]
    assert torch.equal(sel, p)
    assert torch.equal(hist, torch.bincount(sel0, minlength = E).float())
    assert torch.equal(slot.sel[: rows * topk], torch.where(p >= first, p - first, -1).int())
    assert dev_count.item() == int((p >= first).any().item())


@torch.inference_mode()
def test_moe_split_issue_rejections(device):
    sel = torch.zeros(8, dtype = torch.long, device = device)
    y = torch.zeros((1, 64), dtype = torch.half, device = device)
    w = torch.zeros((1, 8), dtype = torch.half, device = device)
    slot = _Slot(1, 8, 64, 64, False, device)
    cnt = torch.zeros(2, dtype = torch.int32, device = device)
    args = lambda **kw: dict(dict(sel = sel, map = None, hist = None, y = y, w = w, cnt = cnt), **kw)

    def call(a):
        ext.moe_split_issue(a["sel"], a["map"], a["hist"], a["y"], a["w"], slot.ptr(slot.sel), slot.ptr(slot.x),
                            slot.ptr(slot.w), a["cnt"], 0, 64, 4)

    with pytest.raises(RuntimeError, match = "contiguous"):
        call(args(y = torch.zeros((1, 128), dtype = torch.half, device = device)[:, ::2]))
    with pytest.raises(RuntimeError, match = "map requires hist"):
        call(args(map = torch.arange(8, device = device)))
    with pytest.raises(RuntimeError, match = "rows \\* topk"):
        call(args(sel = torch.zeros(9, dtype = torch.long, device = device)))
    with pytest.raises(RuntimeError, match = "rows \\* topk"):
        call(args(w = torch.zeros((1, 7), dtype = torch.half, device = device)))
    with pytest.raises(RuntimeError, match = "rows \\* topk"):
        call(args(sel = torch.zeros(9, dtype = torch.long, device = device),
                  w = torch.zeros((1, 9), dtype = torch.half, device = device),
                  y = torch.zeros((2, 64), dtype = torch.half, device = device)))
    with pytest.raises(RuntimeError, match = "y must have 2 dimensions"):
        call(args(y = y.view(1, 1, 64)))
    with pytest.raises(RuntimeError, match = "map is incorrect datatype"):
        call(args(map = torch.arange(8, dtype = torch.int32, device = device), hist = torch.zeros(8, device = device)))
    with pytest.raises(RuntimeError, match = "hist is incorrect datatype"):
        call(args(map = torch.arange(8, device = device), hist = torch.zeros(8, dtype = torch.half, device = device)))
    with pytest.raises(RuntimeError):
        call(args(y = y.float()))
    with pytest.raises(RuntimeError):
        call(args(cnt = cnt.long()))
    with pytest.raises(RuntimeError):
        call(args(sel = sel.int()))


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("rows, h_, ho", [(1, 256, 256), (1, 2880, 2944), (7, 100, 128), (40, 64, 64)])
@pytest.mark.parametrize("active", [True, False])
@torch.inference_mode()
def test_moe_split_collect_add(device, rows, h_, ho, active, pinned):
    g = torch.Generator().manual_seed(rows + h_)
    final = torch.randn((rows, h_), generator = g).to(device)
    final0 = final.clone()
    slot = _Slot(rows, 1, ho, ho, pinned, device)
    part = torch.randn((rows, ho), generator = g)
    slot.out.copy_(part.view(-1).to(slot.out.device))
    dev_count = torch.tensor([1, int(active), 1], dtype = torch.int32, device = device)
    ext.moe_split_collect_add(final, slot.ptr(slot.out), dev_count, 1, ho)
    expect = final0 + part[:, :h_].to(device) if active else final0
    assert torch.equal(final, expect)


@torch.inference_mode()
def test_moe_split_collect_add_rejections(device):
    final = torch.zeros((2, 64), device = device)
    out = torch.zeros(128, device = device)
    cnt = torch.ones(1, dtype = torch.int32, device = device)
    with pytest.raises(RuntimeError):
        ext.moe_split_collect_add(final.half(), out.data_ptr(), cnt, 0, 64)
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.moe_split_collect_add(torch.zeros((64, 2), device = device).t(), out.data_ptr(), cnt, 0, 64)
    with pytest.raises(RuntimeError):
        ext.moe_split_collect_add(final, out.data_ptr(), cnt.long(), 0, 64)


def _inactive_null_worker():
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from testlib.env import get_test_device
    device = get_test_device()
    torch.cuda.set_device(device)
    final = torch.ones((3, 64), device = device)
    ext.moe_split_collect_add(final, 0, torch.zeros(1, dtype = torch.int32, device = device), 0, 64)
    torch.cuda.synchronize(device)
    return final.cpu()


def test_moe_split_collect_inactive_reads_nothing(device):
    """An empty job never touches the slot: a null output pointer is fine"""
    assert torch.equal(run_isolated(_inactive_null_worker), torch.ones((3, 64)))


# Zero-size inputs: moe_split_map with no picks is a no-op (hist and sel_cpu untouched). moe_split_issue with no
# picks still runs: it must record the slot as empty (dev_count[slot] = 0) so the collect skips the slot's stale
# contents, and stages nothing. moe_split_collect_add with no rows or no columns is a no-op (the slot is not read,
# so a null pointer is fine)

@torch.inference_mode()
def test_moe_split_map_empty(device):
    pmap = torch.arange(16, device = device)
    hist = torch.zeros(16, device = device)
    sel_cpu = torch.full((4,), -9, dtype = torch.long, device = device)
    ext.moe_split_map(torch.empty(0, dtype = torch.long, device = device), pmap, hist, sel_cpu, 8)
    torch.cuda.synchronize(device)
    assert not hist.any() and (sel_cpu == -9).all()
    assert (torch.ones(4, device = device) + 1).sum().item() == 8.0, "a later torch op failed after the empty call"


@pytest.mark.parametrize("use_map", [False, True])
@torch.inference_mode()
def test_moe_split_issue_empty(device, use_map):
    slot = _Slot(1, 4, 64, 64, False, device)
    sel_h, x_h, w_h = slot.sel.clone(), slot.x.clone(), slot.w.clone()
    pmap = torch.arange(16, device = device) if use_map else None
    hist = torch.zeros(16, device = device) if use_map else None
    dev_count = torch.ones(3, dtype = torch.int32, device = device)
    ext.moe_split_issue(torch.empty(0, dtype = torch.long, device = device), pmap, hist,
                        torch.empty((0, 64), dtype = torch.half, device = device),
                        torch.empty(0, dtype = torch.half, device = device),
                        slot.ptr(slot.sel), slot.ptr(slot.x), slot.ptr(slot.w), dev_count, 1, 64, 8)
    torch.cuda.synchronize(device)
    assert dev_count.tolist() == [1, 0, 1]
    assert torch.equal(slot.sel, sel_h) and torch.equal(slot.x, x_h) and torch.equal(slot.w, w_h)
    assert hist is None or not hist.any()


@pytest.mark.parametrize("rows, h_", [(0, 64), (3, 0)])
@torch.inference_mode()
def test_moe_split_collect_add_empty(device, rows, h_):
    final = torch.empty((rows, h_), device = device)
    ext.moe_split_collect_add(final, 0, torch.ones(1, dtype = torch.int32, device = device), 0, 64)
    torch.cuda.synchronize(device)
    assert (torch.ones(4, device = device) + 1).sum().item() == 8.0, "a later torch op failed after the empty call"
