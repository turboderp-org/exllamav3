"""
CPU halves of the n-gram embedding fast path (exllamav3_ext/ngram.cu):

ngram_hash_cpu(ids, seq_len, multipliers, offsets, sizes, heads_per_ngram, eos, uids, inverse, heads) -> U
  For the last seq_len positions of each (bsz, ctx + seq_len) int64 history row, the eos-segmented n-gram hash of
  every head (wrapping int64 multiply / xor, Python-sign remainder by the head's vocab size, plus its offset), then
  a global dedup: uids[:U] are the sorted unique row ids, inverse[i] maps output slot i = (b, s, h) to its uid,
  heads[:U] the head whose [offset, offset + size) range holds each uid. References: NGramEmbedding.compute_ngram_ids
  + torch.unique (the torch path the HF parity test pins down) and an independent brute-force Python loop. Rejects
  non-int64 / non-contiguous ids, inconsistent head counts, seq_len > history and undersized output buffers.

ngram_gather_cpu(fd, base_offset, row_bytes, uids, uid_base, out) (Linux, pread pool)
  Row i of out receives the row_bytes bytes at base_offset + (uids[i] - uid_base) * row_bytes of the file, byte-exact,
  for single rows, contiguous runs, scattered rows (past the 64-task grouping) and mixtures; rows of out past U are
  untouched; U = 0 is a no-op; concurrent callers do not interfere. A row past the end of the file raises (short
  read), as do out shapes that do not match row_bytes.
"""
import os
import threading

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.ngram_embedding import NGramEmbedding, _find_nth_prime_after
from exllamav3.modules.quant.exl3_lib.ngram_codec import ROW_DIM

pytestmark = pytest.mark.nogpu

EOS = 248044


def hash_params(ngram_size, hpn, seed, small = False):
    g = torch.Generator().manual_seed(seed)
    H = (ngram_size - 1) * hpn
    mult = torch.randint(-2 ** 62, 2 ** 62, (ngram_size,), generator = g, dtype = torch.int64) * 2 + 1
    base = 50 if small else 100000
    sizes = torch.tensor([_find_nth_prime_after(base + 37 * h, 1) for h in range(H)], dtype = torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype = torch.int64), torch.cumsum(sizes, 0)[:-1]])
    return mult, offsets, sizes


def module_ref(ids, seq_len, ngram_size, hpn, mult, offsets, sizes):
    H = (ngram_size - 1) * hpn
    m = NGramEmbedding(config = None, key = "ngram", ngram_size = ngram_size, heads_per_ngram = hpn,
                       ple_embed_dim = H * ROW_DIM, eos_token_id = EOS)
    m.layer_multipliers, m.head_offsets, m.head_vocab_sizes = mult, offsets, sizes
    flat = m.compute_ngram_ids(ids, seq_len).reshape(-1)
    uids, inverse = torch.unique(flat, return_inverse = True)
    heads = torch.searchsorted(offsets, uids, right = True) - 1
    return flat, uids, inverse, heads


def brute_ref(ids, seq_len, ngram_size, hpn, mult, offsets, sizes):
    """Direct transcription of the definition: per position, per n-gram order, per head"""
    M = 2 ** 64
    def s64(x):
        x %= M
        return x - M if x >= 2 ** 63 else x
    bsz, T = ids.shape
    ctx = T - seq_len
    out = []
    for b in range(bsz):
        row = ids[b].tolist()
        for p in range(ctx, T):
            seg = max([q + 1 for q in range(p) if row[q] == EOS], default = 0)
            src = [row[p - s] if (p - s >= seg and p - s >= 0) else EOS for s in range(ngram_size)]
            for order in range(2, ngram_size + 1):
                mixed = 0
                for s in range(order):
                    mixed ^= s64(src[s] * int(mult[s])) % M
                mixed = s64(mixed)
                for h in range((order - 2) * hpn, (order - 1) * hpn):
                    out.append(mixed % int(sizes[h]) + int(offsets[h]))
    return torch.tensor(out, dtype = torch.int64)


def run_hash(ids, seq_len, mult, offsets, sizes, hpn, extra = 0):
    n = ids.shape[0] * seq_len * offsets.numel()
    uids = torch.full((n + extra,), -3, dtype = torch.int64)
    inverse = torch.full((n + extra,), -3, dtype = torch.int64)
    heads = torch.full((n + extra,), -3, dtype = torch.int32)
    U = ext.ngram_hash_cpu(ids, seq_len, mult, offsets, sizes, hpn, EOS, uids, inverse, heads)
    return U, uids, inverse, heads, n


def make_ids(bsz, T, seed, vocab = 248320, eos_positions = ()):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (bsz, T), generator = g, dtype = torch.int64)
    for b, p in eos_positions:
        ids[b, p] = EOS
    return ids


# (bsz, seq_len, ngram_size, hpn, eos positions)
HASH_CASES = [
    (1, 1, 3, 8, []),                                  # decode: one position, ctx = ngram_size - 1
    (1, 1, 3, 8, [(0, 1)]),                            # eos right before the position
    (1, 1, 3, 8, [(0, 2)]),                            # the position itself is eos
    (2, 64, 3, 8, [(0, 0), (0, 10), (0, 11), (1, 65)]),  # eos at the start, consecutive, last
    (3, 200, 4, 2, [(1, 50), (2, 3)]),
    (1, 37, 2, 1, [(0, 20)]),
    (2, 1024, 3, 8, [(0, 300), (1, 1)]),
]


@pytest.mark.parametrize("bsz,seq_len,ngram_size,hpn,eos", HASH_CASES,
                         ids = [f"b{c[0]}-s{c[1]}-n{c[2]}-h{c[3]}-eos{len(c[4])}" for c in HASH_CASES])
def test_ngram_hash(bsz, seq_len, ngram_size, hpn, eos):
    mult, offsets, sizes = hash_params(ngram_size, hpn, 1)
    ids = make_ids(bsz, ngram_size - 1 + seq_len, 2, eos_positions = eos)
    U, uids, inverse, heads, n = run_hash(ids, seq_len, mult, offsets, sizes, hpn, extra = 5)
    flat, r_uids, r_inv, r_heads = module_ref(ids, seq_len, ngram_size, hpn, mult, offsets, sizes)
    assert U == r_uids.numel()
    assert torch.equal(uids[:U], r_uids)
    assert torch.equal(inverse[:n], r_inv)
    assert torch.equal(heads[:U].long(), r_heads)
    assert torch.equal(uids[:U][inverse[:n]], flat)
    # Slots past n of inverse are not written
    assert (inverse[n:] == -3).all()
    if bsz * seq_len <= 200:
        assert torch.equal(flat, brute_ref(ids, seq_len, ngram_size, hpn, mult, offsets, sizes))


def test_ngram_hash_many_duplicates():
    """Tiny vocab and tiny head sizes: heavy dedup, every head's range hit"""
    mult, offsets, sizes = hash_params(3, 4, 3, small = True)
    ids = make_ids(4, 2 + 500, 4, vocab = 5, eos_positions = [(0, 7), (2, 100)])
    ids[ids == 4] = EOS
    U, uids, inverse, heads, n = run_hash(ids, 500, mult, offsets, sizes, 4)
    flat, r_uids, r_inv, r_heads = module_ref(ids, 500, 3, 4, mult, offsets, sizes)
    assert U == r_uids.numel() < n
    assert torch.equal(uids[:U], r_uids) and torch.equal(inverse[:n], r_inv)
    assert torch.equal(heads[:U].long(), r_heads)
    assert torch.equal(flat[: 500 * 8], brute_ref(ids[:1], 500, 3, 4, mult, offsets, sizes))


def test_ngram_hash_longer_history():
    """History longer than ngram_size - 1 + seq_len: only the last seq_len positions are emitted, but eos before
    them still segments"""
    mult, offsets, sizes = hash_params(3, 8, 5)
    ids = make_ids(2, 40, 6, eos_positions = [(0, 30), (1, 2)])
    U, uids, inverse, heads, n = run_hash(ids, 8, mult, offsets, sizes, 8)
    flat, r_uids, r_inv, _ = module_ref(ids, 8, 3, 8, mult, offsets, sizes)
    assert torch.equal(uids[:U], r_uids) and torch.equal(inverse[:n], r_inv)
    assert torch.equal(flat, brute_ref(ids, 8, 3, 8, mult, offsets, sizes))


def test_ngram_hash_empty():
    mult, offsets, sizes = hash_params(3, 8, 7)
    ids = make_ids(1, 2, 8)
    e64 = torch.empty((0,), dtype = torch.int64)
    assert ext.ngram_hash_cpu(ids, 0, mult, offsets, sizes, 8, EOS, e64, e64, torch.empty((0,), dtype = torch.int32)) == 0


def test_ngram_hash_rejects():
    mult, offsets, sizes = hash_params(3, 8, 9)
    ids = make_ids(1, 10, 10)
    n = 8 * 16
    b64 = torch.zeros((n,), dtype = torch.int64)
    b32 = torch.zeros((n,), dtype = torch.int32)
    with pytest.raises(RuntimeError, match = "contiguous int64"):
        ext.ngram_hash_cpu(ids.int(), 8, mult, offsets, sizes, 8, EOS, b64, b64.clone(), b32)
    with pytest.raises(RuntimeError, match = "contiguous int64"):
        ext.ngram_hash_cpu(make_ids(10, 2, 1).t(), 8, mult, offsets, sizes, 8, EOS, b64, b64.clone(), b32)
    with pytest.raises(RuntimeError, match = "dims"):
        ext.ngram_hash_cpu(ids, 8, mult, offsets, sizes, 4, EOS, b64, b64.clone(), b32)
    with pytest.raises(RuntimeError, match = "dims"):
        ext.ngram_hash_cpu(ids, 11, mult, offsets, sizes, 8, EOS, b64, b64.clone(), b32)
    with pytest.raises(RuntimeError, match = "too small"):
        ext.ngram_hash_cpu(ids, 8, mult, offsets, sizes, 8, EOS, b64[:-1], b64.clone(), b32)
    with pytest.raises(RuntimeError, match = "too small"):
        ext.ngram_hash_cpu(ids, 8, mult, offsets, sizes, 8, EOS, b64, b64.clone(), b32[:-1])


# --------------------------------------------------------------------------------------------------------------
# ngram_gather_cpu

ROWS = 5000


@pytest.fixture(scope = "module")
def table_file(tmp_path_factory):
    """(path, fd, base_offset, row_bytes, data) of a file with a header gap and ROWS random rows"""
    path = tmp_path_factory.mktemp("ngram") / "table.bin"
    row_bytes = 2 * ROW_DIM
    base = 4096 + 24
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, (ROWS, row_bytes), dtype = np.uint8)
    with open(path, "wb") as f:
        f.write(rng.integers(0, 256, base, dtype = np.uint8).tobytes())
        f.write(data.tobytes())
    fd = os.open(path, os.O_RDONLY)
    yield path, fd, base, row_bytes, data
    os.close(fd)


def gather(table_file, uids, uid_base = 0, extra_rows = 0, dtype = torch.int16):
    _, fd, base, row_bytes, _ = table_file
    u = torch.tensor(uids, dtype = torch.int64)
    out = torch.full((len(uids) + extra_rows, row_bytes // torch.empty((), dtype = dtype).element_size()),
                     -1, dtype = dtype)
    ext.ngram_gather_cpu(fd, base, row_bytes, u, uid_base, out)
    return out


def expect_rows(table_file, uids, uid_base = 0):
    data = table_file[4]
    return torch.from_numpy(np.ascontiguousarray(data[np.array(uids, dtype = np.int64) - uid_base]))


def rows_equal(out, expect):
    return torch.equal(out.contiguous().view(torch.uint8).view(expect.shape), expect)


def scattered(n, seed, runs = False):
    g = np.random.default_rng(seed)
    if runs:
        starts = np.sort(g.choice(ROWS - 20, n, replace = False))
        ids = np.unique(np.concatenate([np.arange(s, s + g.integers(1, 8)) for s in starts]))
    else:
        ids = np.sort(g.choice(ROWS, n, replace = False))
    return ids.tolist()


@pytest.mark.platform("linux")
@pytest.mark.parametrize("case", ["one", "first_last", "run", "scattered_16", "scattered_300", "runs_200", "all"])
def test_ngram_gather(table_file, case):
    uids = {
        "one": [1234],
        "first_last": [0, ROWS - 1],
        "run": list(range(100, 400)),
        "scattered_16": scattered(16, 1),
        "scattered_300": scattered(300, 2),      # > 64 runs: run groups per task
        "runs_200": scattered(200, 3, runs = True),
        "all": list(range(ROWS)),
    }[case]
    out = gather(table_file, uids, extra_rows = 3)
    assert rows_equal(out[: len(uids)], expect_rows(table_file, uids))
    assert (out[len(uids):] == -1).all(), "rows past U written"


@pytest.mark.platform("linux")
def test_ngram_gather_uid_base_and_dtype(table_file):
    uids = [b + 1000 for b in scattered(40, 4)]
    out = gather(table_file, uids, uid_base = 1000, dtype = torch.uint8)
    assert rows_equal(out, expect_rows(table_file, uids, uid_base = 1000))


@pytest.mark.platform("linux")
def test_ngram_gather_empty(table_file):
    out = gather(table_file, [], extra_rows = 2)
    assert (out == -1).all()


@pytest.mark.platform("linux")
def test_ngram_gather_concurrent(table_file):
    sets = [scattered(100 + 10 * i, 10 + i, runs = i % 2 == 1) for i in range(8)]
    results = [None] * len(sets)

    def work(i):
        for _ in range(5):
            results[i] = gather(table_file, sets[i])

    threads = [threading.Thread(target = work, args = (i,)) for i in range(len(sets))]
    for t in threads: t.start()
    for t in threads: t.join()
    for s, r in zip(sets, results):
        assert rows_equal(r, expect_rows(table_file, s))


@pytest.mark.platform("linux")
def test_ngram_gather_rejects(table_file):
    _, fd, base, row_bytes, _ = table_file
    u = torch.tensor([ROWS], dtype = torch.int64)                       # one past the last row
    out = torch.zeros((1, row_bytes // 2), dtype = torch.int16)
    with pytest.raises(RuntimeError, match = "short read"):
        ext.ngram_gather_cpu(fd, base, row_bytes, u, 0, out)
    u = torch.tensor([5, ROWS + 10, ROWS + 50], dtype = torch.int64)   # pool path
    out = torch.zeros((3, row_bytes // 2), dtype = torch.int16)
    with pytest.raises(RuntimeError, match = "short read"):
        ext.ngram_gather_cpu(fd, base, row_bytes, u, 0, out)
    u = torch.tensor([1, 2], dtype = torch.int64)
    with pytest.raises(RuntimeError, match = "out shape"):
        ext.ngram_gather_cpu(fd, base, row_bytes, u, 0, torch.zeros((2, row_bytes), dtype = torch.int16))
    with pytest.raises(RuntimeError, match = "out shape"):
        ext.ngram_gather_cpu(fd, base, row_bytes, u, 0, torch.zeros((1, row_bytes // 2), dtype = torch.int16))
    with pytest.raises(RuntimeError, match = "contiguous int64"):
        ext.ngram_gather_cpu(fd, base, row_bytes, u.int(), 0, torch.zeros((2, row_bytes // 2), dtype = torch.int16))
