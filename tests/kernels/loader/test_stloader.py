"""
The safetensors byte loader (exllamav3_ext/stloader.cpp, stloader_cu.cu) on temporary files, against the file bytes
read back with NumPy (and torch's bf16 -> fp16 cast for the conversion flag):

- stloader_open_file(path): STLOADER_THREADS (8) handles, one shared stream on Linux; a missing file raises.
  stloader_close_file(handles) closes them and releases the pinned staging pool, which the next CUDA load
  re-allocates.
- stloader_read(handles, offset, size, target): exactly bytes [offset, offset + size) land in the first size bytes of
  a contiguous CPU, pinned or CUDA target; the rest of the target is untouched; sizes around the 512 KiB CPU block,
  the 1 MiB CUDA run and the 4 MiB staging slot, unaligned offsets, size 0. Rejects non-contiguous or too-small
  targets, fewer than 8 handles, and ranges past the end of the file ("unexpected end of file").
- stloader_deferred_cpu / stloader_deferred_cuda(jobs[, max_chunk_size]): every job writes its byte range to its
  destination (adjacent, gapped, interleaved files, one slot exactly), bf16 jobs converted in place to fp16 (round to
  nearest even, as torch); bytes between destinations untouched. Rejects jobs larger than their declared
  destination, odd-length bf16 jobs, too few handles, jobs over the staging slot (CUDA), odd or oversized
  max_chunk_size.
- stloader_deferred_batch(file_handles, loads, max_chunk_size): one int64 row per load (layout of
  StloaderBatchColumn), host and device rows mixed, loads larger than max_chunk_size split into chunks (bf16 converted
  across chunk boundaries), zero-size rows skipped, the fp32 flag carried but not acted on (the Python front end
  converts fp32 staging tensors itself, so the bytes land raw). Rejects malformed load tables, file indices out of
  range, negative fields and loads larger than their destination.
- bf16 conversion into CUDA destinations that are only 2-byte aligned (an even max_chunk_size that is not a multiple
  of 4, or a destination at an odd fp16 element) must still convert correctly. Run in a child process: a misaligned
  device access poisons the CUDA context.
"""
import os

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.isolated import device_env, run_isolated

pytestmark = pytest.mark.platform("linux")

THREADS = 8
BLOCK = 512 * 1024
DIRECT_RUN = 1024 * 1024
SLOT = 4 * 1024 * 1024
RAW_BYTES = 24 * 1024 * 1024 + 4321
BF16_NUMEL = 3 * 1024 * 1024 + 7

F_BF16, F_FP32, F_CUDA = 1, 2, 4


def bf16_values(n, seed):
    g = torch.Generator().manual_seed(seed)
    v = (torch.randn(n, generator = g) * 100).bfloat16()
    # Specials and rounding edge cases: overflow to inf, fp16 subnormals, signed zeros, NaN
    sp = torch.tensor([float("inf"), -float("inf"), 0.0, -0.0, 1e6, -7e4, 6.55e4, 1e-6, -3e-7, 1e-9, float("nan")])
    v[: sp.numel()] = sp.bfloat16()
    return v


@pytest.fixture(scope = "module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("stloader")
    raw_path = str(d / "raw.bin")
    bf_path = str(d / "bf16.bin")
    raw = np.random.default_rng(0).integers(0, 256, RAW_BYTES, dtype = np.uint8)
    raw.tofile(raw_path)
    bf = bf16_values(BF16_NUMEL, 1)
    bf.view(torch.uint8).numpy().tofile(bf_path)
    h_raw = ext.stloader_open_file(raw_path)
    h_bf = ext.stloader_open_file(bf_path)
    yield dict(raw_path = raw_path, bf_path = bf_path, raw = torch.from_numpy(raw), bf = bf, h_raw = h_raw, h_bf = h_bf)
    ext.stloader_close_file(h_raw)
    ext.stloader_close_file(h_bf)


def assert_fp16_equal(got: torch.Tensor, bf: torch.Tensor):
    exp = bf.to(torch.half)
    got = got.cpu()
    nan = torch.isnan(exp)
    assert torch.equal(torch.isnan(got), nan)
    assert torch.equal(got[~nan].view(torch.int16), exp[~nan].view(torch.int16))


# ---------------------------------------------------------------------------------------------------------------
# open / close

def test_open_handles(files, tmp_path):
    h = files["h_raw"]
    assert len(h) == THREADS
    assert len(set(h)) == 1 and h[0] != 0       # Linux: one positional-read stream shared by all workers
    with pytest.raises(RuntimeError, match = "Error opening file"):
        ext.stloader_open_file(str(tmp_path / "missing.safetensors"))


def test_close_reopen(files, device):
    """Closing releases the pinned pool; a CUDA load through freshly opened handles re-allocates it"""
    h = ext.stloader_open_file(files["raw_path"])
    t = torch.zeros(3 * DIRECT_RUN, dtype = torch.uint8, device = device)
    ext.stloader_read(h, 777, t.numel(), t)
    ext.stloader_close_file(h)
    h = ext.stloader_open_file(files["raw_path"])
    t2 = torch.zeros_like(t)
    ext.stloader_read(h, 777, t.numel(), t2)
    ext.stloader_close_file(h)
    assert torch.equal(t.cpu(), files["raw"][777 : 777 + t.numel()])
    assert torch.equal(t2, t)


# ---------------------------------------------------------------------------------------------------------------
# stloader_read

READ_RANGES = [
    (0, 0), (0, 1), (13, 1000), (BLOCK - 3, 7), (5, BLOCK * THREADS + 11), (DIRECT_RUN - 7, 3 * DIRECT_RUN + 5),
    (12345, SLOT + 1), (0, RAW_BYTES), (RAW_BYTES - 10, 10),
]


@pytest.mark.parametrize("target", ["cpu", "pinned", "cuda"])
@pytest.mark.parametrize("offset,size", READ_RANGES, ids = [f"{o}+{s}" for o, s in READ_RANGES])
def test_read(files, device, target, offset, size):
    pad = 64
    if target == "cuda":
        t = torch.full((size + pad,), 0xA5, dtype = torch.uint8, device = device)
    else:
        t = torch.full((size + pad,), 0xA5, dtype = torch.uint8)
        if target == "pinned":
            t = t.pin_memory()
    ext.stloader_read(files["h_raw"], offset, size, t)
    t = t.cpu()
    assert torch.equal(t[:size], files["raw"][offset : offset + size])
    assert (t[size:] == 0xA5).all(), "bytes past the read written"


@pytest.mark.parametrize("on_cuda", [False, True])
def test_read_typed(files, device, on_cuda):
    n = 300001
    t = torch.empty((n,), dtype = torch.half, device = device if on_cuda else "cpu")
    ext.stloader_read(files["h_bf"], 0, 2 * n, t)
    assert torch.equal(t.cpu().view(torch.int16), files["bf"][:n].view(torch.int16))


@pytest.mark.parametrize("on_cuda", [False, True])
def test_read_rejects(files, device, on_cuda):
    dev = device if on_cuda else "cpu"
    h = files["h_raw"]
    with pytest.raises(RuntimeError, match = "target too small"):
        ext.stloader_read(h, 0, 101, torch.zeros(100, dtype = torch.uint8, device = dev))
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.stloader_read(h, 0, 10, torch.zeros((10, 2), dtype = torch.uint8, device = dev)[:, 0])
    with pytest.raises(RuntimeError, match = "file handles"):
        ext.stloader_read(h[:4], 0, 10, torch.zeros(10, dtype = torch.uint8, device = dev))
    with pytest.raises(RuntimeError, match = "unexpected end of file"):
        ext.stloader_read(h, RAW_BYTES - 10, 11, torch.zeros(11, dtype = torch.uint8, device = dev))
    with pytest.raises(RuntimeError, match = "unexpected end of file"):
        ext.stloader_read(h, RAW_BYTES + SLOT, 4096, torch.zeros(4096, dtype = torch.uint8, device = dev))
    # The loader stays usable after a failed read
    t = torch.zeros(1000, dtype = torch.uint8, device = dev)
    ext.stloader_read(h, 0, 1000, t)
    assert torch.equal(t.cpu(), files["raw"][:1000])


# ---------------------------------------------------------------------------------------------------------------
# deferred jobs

def job(handles, off, size, dest: torch.Tensor, dest_off = 0, bf16 = False, cuda = False, dest_size = None):
    ptr = dest.data_ptr() + dest_off
    cap = dest.nbytes - dest_off if dest_size is None else dest_size
    dev = dest.device.index if dest.is_cuda else -1
    return ext.TensorLoadJob(handles, off, size, ptr, cap, bf16, False, cuda, dev)


def raw_job_plan():
    """(file offset, size, dest offset) of raw jobs into one buffer: adjacent, small gap, gap > 512 KiB merge
    limit, one exactly a slot, odd sizes and offsets, sorted by file offset"""
    plan = []
    off, dst = 3, 0
    for size, gap in [(1000, 0), (77, 0), (BLOCK, 100), (3, 0), (SLOT, 0), (12345, 600 * 1024), (65536, 0),
                      (2 * DIRECT_RUN + 1, 4096), (1, 1)] + [(4099 * i + 1, 13) for i in range(1, 20)]:
        plan.append((off, size, dst))
        off += size + gap
        dst += size + 16      # 16-byte guard between destinations
    return plan, dst


@pytest.mark.parametrize("on_cuda", [False, True])
def test_deferred_raw(files, device, on_cuda):
    plan, total = raw_job_plan()
    dev = device if on_cuda else "cpu"
    buf = torch.full((total,), 0x5A, dtype = torch.uint8, device = dev)
    jobs = [job(files["h_raw"], o, s, buf, d, cuda = on_cuda) for o, s, d in plan]
    if on_cuda:
        ext.stloader_deferred_cuda(jobs, SLOT)
    else:
        ext.stloader_deferred_cpu(jobs)
    got = buf.cpu()
    expect = torch.full((total,), 0x5A, dtype = torch.uint8)
    for o, s, d in plan:
        expect[d : d + s] = files["raw"][o : o + s]
    assert torch.equal(got, expect)


@pytest.mark.parametrize("on_cuda", [False, True])
def test_deferred_bf16_and_interleaved_files(files, device, on_cuda):
    """bf16 jobs from one file interleaved with raw jobs from another, each into its own tensor"""
    dev = device if on_cuda else "cpu"
    bf = files["bf"]
    pieces = [(0, 11), (11, 1000), (5000, 1 << 20), (5000 + (1 << 20), 333), (2 << 20, (1 << 20) + 7)]
    outs, jobs, raws = [], [], []
    for i, (e0, n) in enumerate(pieces):
        o = torch.full((n,), -1.0, dtype = torch.half, device = dev)
        outs.append(o)
        jobs.append(job(files["h_bf"], 2 * e0, 2 * n, o, bf16 = True, cuda = on_cuda))
        r = torch.zeros(1000 + i, dtype = torch.uint8, device = dev)
        raws.append((i * 100000, r))
        jobs.append(job(files["h_raw"], i * 100000, r.numel(), r, cuda = on_cuda))
    if on_cuda:
        ext.stloader_deferred_cuda(jobs, SLOT)
    else:
        ext.stloader_deferred_cpu(jobs)
    for (e0, n), o in zip(pieces, outs):
        assert_fp16_equal(o, bf[e0 : e0 + n])
    for off, r in raws:
        assert torch.equal(r.cpu(), files["raw"][off : off + r.numel()])


@pytest.mark.parametrize("on_cuda", [False, True])
def test_deferred_rejects(files, device, on_cuda):
    dev = device if on_cuda else "cpu"
    run = ext.stloader_deferred_cuda if on_cuda else ext.stloader_deferred_cpu
    call = (lambda j: run(j, SLOT)) if on_cuda else run
    buf = torch.zeros(4096, dtype = torch.uint8, device = dev)
    h = files["h_raw"]
    with pytest.raises(RuntimeError, match = "would write"):
        call([job(h, 0, 4097, buf, cuda = on_cuda)])
    with pytest.raises(RuntimeError, match = "would write"):
        call([job(h, 0, 100, buf, cuda = on_cuda, dest_size = 99)])
    with pytest.raises(RuntimeError, match = "odd length"):
        call([job(h, 0, 101, buf, bf16 = True, cuda = on_cuda)])
    with pytest.raises(RuntimeError, match = "file handles"):
        call([job(h[:2], 0, 100, buf, cuda = on_cuda)])
    with pytest.raises(RuntimeError, match = "unexpected end of file"):
        call([job(h, RAW_BYTES - 100, 200, buf, cuda = on_cuda)])
    if on_cuda:
        big = torch.zeros(SLOT + 2, dtype = torch.uint8, device = dev)
        with pytest.raises(RuntimeError, match = "larger than staging slot"):
            run([job(h, 0, SLOT + 2, big, cuda = True)], SLOT)
        with pytest.raises(RuntimeError, match = "exceeds staging slot"):
            run([job(h, 0, 100, buf, cuda = True)], SLOT + 2)
        with pytest.raises(RuntimeError, match = "must be even"):
            run([job(h, 0, 100, buf, cuda = True)], 1001)
    # Usable afterwards
    call([job(h, 10, 4096, buf, cuda = on_cuda)])
    assert torch.equal(buf.cpu(), files["raw"][10 : 10 + 4096])


# ---------------------------------------------------------------------------------------------------------------
# stloader_deferred_batch

def batch_row(file_idx, off, size, dest: torch.Tensor, flags = 0, dest_size = None, dest_off = 0):
    cuda = dest.is_cuda
    return [file_idx, off, size, dest.data_ptr() + dest_off, dest.nbytes - dest_off if dest_size is None else dest_size,
            flags | (F_CUDA if cuda else 0), dest.device.index if cuda else -1]


@pytest.mark.parametrize("chunk", [SLOT, 1 << 20, 4096 + 4])
def test_deferred_batch(files, device, chunk):
    bf = files["bf"]
    raw = files["raw"]
    handles = [files["h_raw"], files["h_bf"]]
    dests, rows, checks = [], [], []

    def add(file_idx, off, size, dest, flags = 0, check = None):
        dests.append(dest)
        rows.append(batch_row(file_idx, off, size, dest, flags))
        checks.append(check)

    # bf16 tensors spanning several chunks (conversion runs per chunk), on both sides
    for dev, e0, n in [(device, 100, 3 * 1024 * 1024 // 2 + 3), ("cpu", 17, 700001), (device, 0, 5), ("cpu", 9, 1)]:
        o = torch.full((n,), 7.0, dtype = torch.half, device = dev)
        add(1, 2 * e0, 2 * n, o, F_BF16, ("bf16", e0, n))
    # raw loads, incl. one larger than a slot and adjacent ones that coalesce
    for dev, off, n in [(device, 0, 1000), (device, 1000, 2000), (device, 3000, SLOT + SLOT // 2 + 1),
                        ("cpu", 99, 5 * 1024 * 1024 + 3), ("cpu", RAW_BYTES - 5, 5)]:
        r = torch.zeros(n, dtype = torch.uint8, device = dev)
        add(0, off, n, r, 0, ("raw", off, n))
    # fp32 flag: the C++ side copies the bytes; conversion is the Python front end's job
    f32 = torch.zeros(1000, dtype = torch.float32, device = device)
    add(0, 4000, 4000, f32, F_FP32, ("raw", 4000, 4000))
    # Zero-size load: skipped, destination untouched
    z = torch.full((4,), 9, dtype = torch.uint8, device = device)
    add(0, 0, 0, z, 0, ("untouched", 9, 4))

    loads = torch.tensor(rows, dtype = torch.int64)
    ext.stloader_deferred_batch(handles, loads, chunk)
    for d, c in zip(dests, checks):
        kind, a, n = c
        if kind == "bf16":
            assert_fp16_equal(d, bf[a : a + n])
        elif kind == "raw":
            assert torch.equal(d.cpu().view(torch.uint8), raw[a : a + n])
        else:
            assert (d.cpu() == a).all()


def test_deferred_batch_unsorted(files, device):
    """Rows out of file order (the caller normally sorts; correctness must not depend on it)"""
    sizes = [(5 * 100000, 70000), (0, 1234), (300000, 99999), (100, 50), (2 * SLOT, SLOT)]
    outs = [torch.zeros(n, dtype = torch.uint8, device = device) for _, n in sizes]
    loads = torch.tensor([batch_row(0, o, n, t) for (o, n), t in zip(sizes, outs)], dtype = torch.int64)
    ext.stloader_deferred_batch([files["h_raw"]], loads, SLOT)
    for (o, n), t in zip(sizes, outs):
        assert torch.equal(t.cpu(), files["raw"][o : o + n])


def test_deferred_batch_rejects(files, device):
    h = [files["h_raw"]]
    buf = torch.zeros(100, dtype = torch.uint8, device = device)
    good = batch_row(0, 0, 100, buf)
    with pytest.raises(RuntimeError, match = "loads must be"):
        ext.stloader_deferred_batch(h, torch.zeros((1, 7), dtype = torch.int32), SLOT)
    with pytest.raises(RuntimeError, match = "loads must be"):
        ext.stloader_deferred_batch(h, torch.tensor([good[:6]], dtype = torch.int64), SLOT)
    with pytest.raises(RuntimeError, match = "loads must be"):
        ext.stloader_deferred_batch(h, torch.tensor([good, good], dtype = torch.int64).t().contiguous().t(), SLOT)
    with pytest.raises(RuntimeError, match = "max_chunk_size must be positive"):
        ext.stloader_deferred_batch(h, torch.tensor([good], dtype = torch.int64), 0)
    with pytest.raises(RuntimeError, match = "out of range"):
        ext.stloader_deferred_batch(h, torch.tensor([[1] + good[1:]], dtype = torch.int64), SLOT)
    with pytest.raises(RuntimeError, match = "negative"):
        ext.stloader_deferred_batch(h, torch.tensor([[0, -1] + good[2:]], dtype = torch.int64), SLOT)
    with pytest.raises(RuntimeError, match = "into a destination of"):
        ext.stloader_deferred_batch(h, torch.tensor([batch_row(0, 0, 101, buf, dest_size = 100)], dtype = torch.int64),
                                    SLOT)
    ext.stloader_deferred_batch(h, torch.tensor([good], dtype = torch.int64), SLOT)
    assert torch.equal(buf.cpu(), files["raw"][:100])


# ---------------------------------------------------------------------------------------------------------------
# bf16 conversion into 2-byte-aligned device destinations (child process)

def _bf16_misaligned_worker(raw_dir: str, mode: str):
    """Returns (error or None, conversion matches torch)"""
    dev = torch.device("cuda:0")
    n = 1000
    bf = bf16_values(n, 2)
    path = os.path.join(raw_dir, "bf16_small.bin")
    bf.view(torch.uint8).numpy().tofile(path)
    h = ext.stloader_open_file(path)
    try:
        if mode == "chunk":
            # Even chunk size that is not a multiple of 4: the second chunk starts 2 bytes past a 4-byte boundary
            dst = torch.zeros(n, dtype = torch.half, device = dev)
            chunk = 1002
        else:
            # Destination at an odd fp16 element
            dst = torch.zeros(n + 1, dtype = torch.half, device = dev)[1:]
            chunk = SLOT
        loads = torch.tensor([[0, 0, 2 * n, dst.data_ptr(), 2 * n, F_BF16 | F_CUDA, 0]], dtype = torch.int64)
        try:
            ext.stloader_deferred_batch([h], loads, chunk)
            torch.cuda.synchronize(dev)
        except RuntimeError as e:
            return str(e).splitlines()[0], False
        exp = bf.to(torch.half)
        got = dst.cpu()
        nan = torch.isnan(exp)
        ok = torch.equal(torch.isnan(got), nan) and torch.equal(got[~nan].view(torch.int16), exp[~nan].view(torch.int16))
        return None, ok
    finally:
        ext.stloader_close_file(h)


@pytest.mark.parametrize("mode", ["chunk", "dest"])
def test_bf16_cuda_2byte_aligned(device, tmp_path, mode):
    err, ok = run_isolated(_bf16_misaligned_worker, str(tmp_path), mode, env = device_env(device), timeout = 300)
    assert err is None, f"bf16 -> fp16 into a 2-byte-aligned device destination failed: {err}"
    assert ok
