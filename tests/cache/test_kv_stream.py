"""
KV cache streaming (cache/stream.py + the streaming launch paths in modules/attention_fn/triton_paged.py).

A streaming layer keeps the authoritative K/V in a pinned host mirror and only a bounded GPU working set on the
device: a resident window of R pages (the most recent token pages, slot = page % R) and a shared ring of S
chunks of C pages for the older history. Attention goes through the chunked driver: the history is walked in
placement chunks, each launch takes a KV span and a range of partial slots, and one combine reduces the
partials in KV order.

The contract tested here is that this driver reproduces the stock paged wrappers over the same K/V, for every
placement and bookkeeping path it can take:

    - the single-launch fast path (whole history resident, no combine),
    - chunked prefill (several launches, partials reduced in KV order) and single-token decode walks,
    - the resident window refilled from the mirror and the ring serving streamed chunks (read-ahead included),
    - non-trivial block tables (a permuted page pool) and prompt-cache imports (copy_page), where pages enter
      the mirror without going through this layer's write path,
    - sequence reuse and rollback (a shorter follow-up call rewrites the tail of the previous sequence),
    - page rotations (rotate_pages) that invalidate the resident slots and permute the host mirror,
    - fp16 and 8-bit caches, and the serial debug mode against the overlapped default.

The reference for the fp16 variants is the stock wrapper writing the same rows into a plain paged pool; for the
8-bit variants a real CacheLayer_quant, written through its own update_kv_direct (the same quantization the
streaming layer applies) and read through the stock wrappers' qc path. Accumulation orders differ between the
two paths (chunk splits, tile shapes), so outputs are compared with a small absolute tolerance.
"""

from types import SimpleNamespace

import pytest
import torch
import warnings

from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.stream import CacheLayer_stream, CacheLayer_stream_quant
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode, paged_attn_triton_prefill

N_KVH = 2
N_QH = 8
HD = 128
TOL = 5e-3

_cache_ids = iter(range(1, 1 << 30))


def attn_stub():
    return SimpleNamespace(num_q_heads = N_QH, num_kv_heads = N_KVH, head_dim = HD, key = "kv_stream_test")


def cs(a: int, device) -> torch.Tensor:
    return torch.tensor([a], dtype = torch.int32, device = device)


def synth(total: int, seed: int, device):
    """(q, k, v) over `total` rows, fp16 on the device"""
    g = torch.Generator(device = "cpu").manual_seed(seed)
    k = torch.randn(1, total, N_KVH, HD, generator = g, dtype = torch.float32).half().to(device)
    v = torch.randn(1, total, N_KVH, HD, generator = g, dtype = torch.float32).half().to(device)
    q = torch.randn(1, total, N_QH, HD, generator = g, dtype = torch.float32).half().to(device)
    return q, k, v


def make_stream_layer(kind: str, device, resident: int, ring: int, prefetch: int, capacity_pages: int,
                      serial: bool = False):
    cls = CacheLayer_stream if kind == "fp16" else CacheLayer_stream_quant
    args = (None, attn_stub(), next(_cache_ids), capacity_pages * PAGE_SIZE)
    if kind == "fp16":
        layer = cls(*args)
    else:
        layer = cls(*args, k_bits = 8, v_bits = 8)
    layer.alloc(device)
    return layer


class StockRef:
    """The stock path's store: a plain paged pool (fp16) or a real quantized layer (8-bit)"""

    def __init__(self, kind: str, device, pages: int):
        self.kind = kind
        self.device = device
        self.layer = None
        self.qc = None
        if kind == "fp16":
            self.kc = torch.zeros(pages, PAGE_SIZE, N_KVH, HD, dtype = torch.half, device = device)
            self.vc = torch.zeros_like(self.kc)
        else:
            self.layer = CacheLayer_quant(None, attn_stub(), next(_cache_ids), pages * PAGE_SIZE, 8, 8)
            self.layer.alloc(device)
            self.kc, self.vc = self.layer.qk, self.layer.qv
            self.qc = (self.layer.sk, self.layer.sv, self.layer.k_bits, self.layer.v_bits)

    def attn(self, q, k, v, a: int, length: int, bt, prefill: bool):
        """Rows [a, a + length) written into the pool and attended over the whole history [0, a + length)"""
        kwargs = dict(causal = True)
        if self.layer is not None:
            self.layer.update_kv_direct(cs(a, self.device), bt, k[:, a:a + length].contiguous(),
                                        v[:, a:a + length].contiguous(), length)
            kwargs.update(qc = self.qc, pre_appended_len = length, n_kv_heads_override = N_KVH)
            k2 = v2 = None
        else:
            k2, v2 = k[:, a:a + length], v[:, a:a + length]
        fn = paged_attn_triton_prefill if prefill else paged_attn_triton_decode
        return fn(q[:, a:a + length], k2, v2, self.kc, self.vc, bt, cs(a, self.device), **kwargs)


class Case:
    """One synthetic history plus the streaming layer and the stock reference it is driven through"""

    def __init__(self, device, monkeypatch, total: int, seed: int, resident: int, ring: int, prefetch: int = 2,
                 kind: str = "fp16", serial: bool = False, pages: int | None = None,
                 table: torch.Tensor | None = None):
        monkeypatch.setenv("EXL3_KV_STREAM", "1")
        monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", str(resident))
        monkeypatch.setenv("EXL3_KV_STREAM_RING", str(ring))
        monkeypatch.setenv("EXL3_KV_STREAM_PREFETCH", str(prefetch))
        monkeypatch.setenv("EXL3_KV_STREAM_SERIAL", "1" if serial else "0")

        self.total = total
        self.n_pages = -(-total // PAGE_SIZE)
        self.pages = pages if pages is not None else self.n_pages + 4
        self.q, self.k, self.v = synth(total, seed, device)
        if table is None:
            table = torch.arange(self.pages, dtype = torch.int32)
        self.bt_host = table
        self.bt = table[None].to(device)
        self.layer = make_stream_layer(kind, device, resident, ring, prefetch, self.pages, serial)
        self.ref = StockRef(kind, device, self.pages)

    def drive(self, a: int, length: int, bt = None, bt_host = None):
        device = self.q.device
        return self.layer.stream_attn(
            self.q[:, a:a + length], self.k[:, a:a + length], self.v[:, a:a + length],
            cs(a, device), self.bt if bt is None else bt,
            host_seqlens = torch.tensor([a], dtype = torch.int32),
            host_block_table = self.bt_host if bt_host is None else bt_host)

    def stock(self, a: int, length: int, bt = None, prefill: bool | None = None):
        if prefill is None:
            prefill = length > 16
        return self.ref.attn(self.q, self.k, self.v, a, length, self.bt if bt is None else bt, prefill)


def compare(got, ref, tag: str):
    d = (got.float() - ref.float()).abs().max().item()
    assert d <= TOL, f"{tag}: peak |stream - stock| = {d:.3e} over {got.shape}"


# -------------------------------------------------------------------------------------------------
# Driver parity across placements

CASES = [
    # total, probe, resident, ring, prefetch, seed
    (48, 1, 2, 4, 2, 1),        # single launch, everything resident
    (255, 1, 2, 4, 2, 1),
    (256, 1, 2, 4, 2, 1),       # page boundary
    (257, 1, 2, 4, 2, 1),
    (700, 1, 2, 4, 2, 2),       # one streamed chunk, decode probe
    (1200, 1, 2, 4, 2, 3),
    (1200, 50, 2, 4, 2, 4),     # prefill probe over the full history
    (1200, 300, 3, 4, 2, 5),    # prefill probe with internal sub-chunks
    (2500, 1, 4, 8, 2, 6),      # many streamed chunks
    (3000, 400, 2, 4, 2, 7),    # long prefill walk, resident window tiny
    (520, 100, 1, 2, 1, 8),     # smallest ring: one chunk of one page, prefetch 1
    (900, 60, 16, 64, 2, 9),    # resident window covers prefill results, ring idle
]


@pytest.mark.parametrize("total, probe, resident, ring, prefetch, seed", CASES)
def test_stream_walk_matches_stock(device, monkeypatch, total, probe, resident, ring, prefetch, seed):
    n_hist = total - probe
    case = Case(device, monkeypatch, total, seed, resident, ring, prefetch)
    got_hist = case.drive(0, n_hist)
    got_probe = case.drive(n_hist, probe)
    ref_hist = case.stock(0, n_hist)
    ref_probe = case.stock(n_hist, probe)
    tag = f"total={total} probe={probe} R={resident} ring={ring} prefetch={prefetch}"
    compare(got_hist, ref_hist, f"hist {tag}")
    compare(got_probe, ref_probe, f"probe {tag}")


QUANT_CASES = [
    (300, 1, 2, 4, 2, 11),
    (900, 1, 2, 4, 2, 12),
    (1200, 50, 2, 4, 2, 13),
    (1200, 300, 2, 2, 2, 14),
    (2500, 1, 4, 8, 2, 15),
]


@pytest.mark.parametrize("total, probe, resident, ring, prefetch, seed", QUANT_CASES)
def test_quant_stream_walk_matches_stock(device, monkeypatch, total, probe, resident, ring, prefetch, seed):
    """8-bit caches: the scales must follow the placement (ring vs resident) with the packed data"""
    n_hist = total - probe
    case = Case(device, monkeypatch, total, seed, resident, ring, prefetch, kind = "q8")
    got_hist = case.drive(0, n_hist)
    got_probe = case.drive(n_hist, probe)
    ref_hist = case.stock(0, n_hist)
    ref_probe = case.stock(n_hist, probe)
    tag = f"total={total} probe={probe} R={resident} ring={ring}"
    compare(got_hist, ref_hist, f"quant hist {tag}")
    compare(got_probe, ref_probe, f"quant probe {tag}")


@pytest.mark.parametrize("kind", ["fp16", "q8"])
def test_permuted_block_table(device, monkeypatch, kind):
    """The kernel table is a pure function of the token page index: the pool behind those pages is arbitrary,
    and reads must be served by whichever pool page the block table names"""
    total = 1100
    pages = 7
    table = torch.tensor([5, 0, 6, 2, 4, 1, 3], dtype = torch.int32)
    case = Case(device, monkeypatch, total, 21, 2, 4, 2, kind = kind, pages = pages, table = table)
    got_hist = case.drive(0, 800)
    got_probe = case.drive(800, 300)
    ref_hist = case.stock(0, 800)
    ref_probe = case.stock(800, 300)
    compare(got_hist, ref_hist, f"permuted {kind} hist")
    compare(got_probe, ref_probe, f"permuted {kind} probe")


def test_serial_and_overlap_agree(device, monkeypatch):
    """EXL3_KV_STREAM_SERIAL=1 only removes the copy stream; the results must be identical"""
    total = 1500
    overlap = Case(device, monkeypatch, total, 31, 2, 4, 2, kind = "q8")
    got_a = overlap.drive(0, 1200).clone()
    got_a2 = overlap.drive(1200, 300).clone()
    monkeypatch.setenv("EXL3_KV_STREAM_SERIAL", "1")
    serial = Case(device, monkeypatch, total, 31, 2, 4, 2, kind = "q8")
    got_b = serial.drive(0, 1200)
    got_b2 = serial.drive(1200, 300)
    compare(got_b, got_a, "serial vs overlap (hist)")
    compare(got_b2, got_a2, "serial vs overlap (probe)")


def test_sequence_reuse_and_rollback(device, monkeypatch):
    """A call that does not continue the last published position starts a new sequence over the same pool:
    the publication ledger is clamped and every slot is refilled, so no stale mirror row can be read"""
    total = 900
    case = Case(device, monkeypatch, total, 41, 2, 4, 2)
    case.drive(0, 700)                                  # sequence A
    case.ref.attn(case.q, case.k, case.v, 0, 700, case.bt, True)

    q2, k2, v2 = synth(600, 42, device)                 # sequence B rewrites the first 600 rows
    case.q, case.k, case.v = q2, k2, v2
    got = case.drive(0, 300)                            # shorter than the published 700: rollback
    ref = case.stock(0, 300)
    compare(got, ref, "rollback first span")
    got2 = case.drive(300, 300)                         # continue within the rewritten region
    ref2 = case.stock(300, 300)
    compare(got2, ref2, "rollback continued")


def test_copy_page_import_is_read_from_the_mirror(device, monkeypatch):
    """Prompt caching: pages imported with copy_page never go through this layer's write path, so the resident
    window and the ring must serve them from the mirror"""
    src = Case(device, monkeypatch, 1000, 51, 2, 4, 2)
    src.drive(0, 800)
    dst = Case(device, monkeypatch, 1100, 52, 2, 4, 2)
    for p in range(3):
        dst.layer.copy_page(src.layer, p, p, PAGE_SIZE)

    # The driver appends rows [768, 1100) behind the imported prefix and attends the new span
    got = dst.drive(768, 332)

    # Reference: a plain pool holding the source's rows [0, 768) then the destination's rows [768, 1100)
    ref = StockRef("fp16", device, dst.pages)
    kk = torch.cat([src.k[:, :768], dst.k[:, 768:1100]], dim = 1)
    vv = torch.cat([src.v[:, :768], dst.v[:, 768:1100]], dim = 1)
    paged_attn_triton_prefill(dst.q[:, :768], kk[:, :768], vv[:, :768], ref.kc, ref.vc, dst.bt,
                              cs(0, device), causal = True)
    r = ref.attn(dst.q, kk, vv, 768, 332, dst.bt, prefill = True)
    compare(got, r, "copy_page import")


def test_rotate_pages(device, monkeypatch):
    """rotate_pages permutes the mirror (and the ledger) and invalidates the resident slots; reading the
    permuted pool through the matching block table must give the same outputs as the unrotated pool"""
    total = 1600
    pages = 10
    table = torch.arange(pages, dtype = torch.int32)
    case = Case(device, monkeypatch, total, 61, 2, 4, 2, pages = pages, table = table)
    ref = StockRef("fp16", device, pages)
    case.drive(0, 1200)
    ref.attn(case.q, case.k, case.v, 0, 1200, case.bt, prefill = True)

    # Swap pool pages 0 and 1 with the sequential (dst, src, -1 = temp) plan rotate_pages takes, and point the
    # block table at the moved pages
    case.layer.rotate_pages(torch.tensor([-1, 0, 0, 1, 1, -1], dtype = torch.int32))
    new_of_old = torch.tensor([1, 0] + list(range(2, pages)), dtype = torch.int32)
    case.bt = new_of_old[table][None].to(device)

    # The appended span reads pages 0..4 (rotated, from the ring) and writes pages 5, 6
    got = case.drive(1200, 400)
    r = ref.attn(case.q, case.k, case.v, 1200, 400, table[None].to(device), prefill = True)
    compare(got, r, "rotated probe")


# -------------------------------------------------------------------------------------------------
# Placement policy: env parsing and budget arithmetic (no GPU needed)

def _cfg(capacity_pages: int, per_page: int = 2048, cache_id: int | None = None):
    """Resolved stream config for a layer with this per-page byte size (a fresh cache id per call)"""
    from exllamav3.cache import stream
    layer = SimpleNamespace(stream_per_page_bytes = lambda: per_page)
    return stream._stream_config(cache_id if cache_id is not None else next(_cache_ids), layer,
                                 capacity_pages)


def _clear_stream_env(monkeypatch):
    for k in ("EXL3_KV_STREAM_RESIDENT", "EXL3_KV_STREAM_RING", "EXL3_KV_STREAM_PREFETCH",
              "EXL3_KV_STREAM_MIRROR", "EXL3_KV_STREAM_SERIAL", "EXL3_KV_STREAM_TRACE"):
        monkeypatch.delenv(k, raising = False)


@pytest.mark.nogpu
def test_stream_config_defaults(monkeypatch):
    """Auto window: the resident size is deferred to first use (the budget needs the free device memory,
    which is only meaningful once the weights are loaded), and the ring is 64 pages split by the
    prefetch depth"""
    _clear_stream_env(monkeypatch)
    cfg = _cfg(32)
    assert (cfg["R"], cfg["ring_pages"], cfg["prefetch"], cfg["S"], cfg["C"]) == (None, 64, 2, 2, 32)
    assert cfg["serial"] is False and cfg["trace"] is False
    assert cfg["n_layers"] == 1
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "auto")
    assert _cfg(32)["R"] is None       # spelled out auto is the same policy


@pytest.mark.nogpu
def test_stream_config_counts_layers(monkeypatch):
    """Auto placement divides the free memory between the layers sharing the cache id, so one config
    accumulates one count per layer"""
    _clear_stream_env(monkeypatch)
    cid = next(_cache_ids)
    cfg = _cfg(32, cache_id = cid)
    assert cfg["n_layers"] == 1
    assert _cfg(32, cache_id = cid)["n_layers"] == 2
    assert _cfg(32, cache_id = cid) is cfg
    assert cfg["n_layers"] == 3


@pytest.mark.nogpu
def test_auto_resident_budget(monkeypatch):
    """Auto placement fills the free device memory: the window gets what is left after the ring, a
    capped reserve for the walk partials and workspace headroom"""
    from exllamav3.cache.stream import StreamLayer, _PARTIAL_BUDGET
    free = 20 << 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (free, 24 << 30))
    per_page = 1 << 20
    layer = SimpleNamespace(
        cfg = dict(per_page = per_page, S = 2, C = 32, n_layers = 4),   # ring: 64 MiB
        capacity_pages = 1 << 20,
        device = torch.device("cpu"),
    )
    ring = 2 * 32 * per_page
    partials = min(2 * 4 * _PARTIAL_BUDGET, free // 4)
    slack = max(512 << 20, free // 16)
    assert partials == 2 * 4 * _PARTIAL_BUDGET
    assert StreamLayer._auto_resident_pages(layer) == (free - ring - partials - slack) // (4 * per_page)
    layer.capacity_pages = 8
    assert StreamLayer._auto_resident_pages(layer) == 8      # clamped to the cache capacity
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (1 << 20, 24 << 30))
    assert StreamLayer._auto_resident_pages(layer) == 1      # never 0, however tight the budget


@pytest.mark.nogpu
def test_stream_config_resident_forms(monkeypatch):
    _clear_stream_env(monkeypatch)
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "12")
    assert _cfg(32)["R"] == 12
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "12p")
    assert _cfg(32)["R"] == 12
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "0")
    assert _cfg(32)["R"] == 1          # pages are the floor
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "64")
    assert _cfg(32)["R"] == 32         # clamped to the capacity
    # Byte budgets divide by the layer's per-page bytes (per_page = 1 MiB here)
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "64M")
    assert _cfg(100, per_page = 1 << 20)["R"] == 64
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "8G")
    assert _cfg(100, per_page = 1 << 20)["R"] == 100
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "bogus")
    with pytest.raises(ValueError):
        _cfg(32)


@pytest.mark.nogpu
def test_stream_config_ring_and_prefetch(monkeypatch):
    """C = ring_pages // S, with S capped by the ring size so a prefetch depth above it just uses the whole
    ring"""
    _clear_stream_env(monkeypatch)
    monkeypatch.setenv("EXL3_KV_STREAM_RING", "4")
    monkeypatch.setenv("EXL3_KV_STREAM_PREFETCH", "3")
    cfg = _cfg(32)
    assert (cfg["ring_pages"], cfg["S"], cfg["C"]) == (4, 3, 1)
    monkeypatch.setenv("EXL3_KV_STREAM_PREFETCH", "8")
    cfg = _cfg(32)
    assert (cfg["S"], cfg["C"]) == (4, 1)
    monkeypatch.setenv("EXL3_KV_STREAM_RING", "0")
    with pytest.raises(ValueError):
        _cfg(32)
    monkeypatch.setenv("EXL3_KV_STREAM_RING", "abc")
    with pytest.raises(ValueError):
        _cfg(32)


@pytest.mark.nogpu
def test_stream_config_mirror_budget_and_flags(monkeypatch):
    """The mirror budget is reserved: a budget below the full mirror warns once, a larger one is quiet.
    SERIAL and TRACE are plain flags"""
    from exllamav3.cache import stream
    _clear_stream_env(monkeypatch)
    monkeypatch.setattr(stream, "_warned", set())

    # per_page = 1 MiB, capacity 100 pages -> a full mirror of 100 MiB
    monkeypatch.setenv("EXL3_KV_STREAM_MIRROR", "512M")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = _cfg(100, per_page = 1 << 20)
    assert cfg["ring_pages"] == 64

    monkeypatch.setenv("EXL3_KV_STREAM_MIRROR", "8M")
    with pytest.warns(UserWarning, match = "below the full mirror"):
        _cfg(100, per_page = 1 << 20)
    # Only the first is reported
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cfg(100, per_page = 1 << 20)

    monkeypatch.setenv("EXL3_KV_STREAM_SERIAL", "1")
    monkeypatch.setenv("EXL3_KV_STREAM_TRACE", "1")
    cfg = _cfg(32)
    assert cfg["serial"] is True and cfg["trace"] is True


@pytest.mark.nogpu
def test_stream_config_is_memoized_per_cache(monkeypatch):
    """One placement policy per cache: the first layer of a load decides, later layers of the same cache
    reuse it"""
    _clear_stream_env(monkeypatch)
    cid = next(_cache_ids)
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "3")
    first = _cfg(32, cache_id = cid)
    monkeypatch.setenv("EXL3_KV_STREAM_RESIDENT", "7")
    again = _cfg(32, cache_id = cid)
    assert first is again and again["R"] == 3
