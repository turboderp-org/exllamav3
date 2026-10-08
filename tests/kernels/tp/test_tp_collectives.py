"""
Native tensor-parallel collectives (parallel/*.cu, TPBackendNative) across rank processes that run the real backend
over the real shared-memory regions, plus the CPU reduce helper (testlib.tp_native). Two configurations: two
ranks on two GPUs, and two ranks on one physical GPU (distinct CUDA ordinals for the same device; the kernels index
the shared context by ordinal). Every rank's inputs are regenerated in the test process from the same seeds.

Contracts (all bitwise; every output buffer sits between sentinel guards, which must survive):

- pg_all_reduce_cpu (backend.all_reduce): every rank receives the same sum of the contributing ranks' tensors; a
  non-contributor's contents are ignored. Wire formats, modelled exactly: fp16 goes over an fp16 wire (CPU has
  F16C), so with two ranks the result is the round-to-nearest-even fp16 of the exact sum; fp32 and bf16 go over a
  bf16 wire, rounded as (bits + 0x8000) >> 16 (half away from zero) on the way in, summed in fp32 by the CPU and
  rounded the same way; fp32 receives the bf16 sum widened. With EXL3_TP_NO_FP16_WIRE=1 fp16 takes the bf16 wire
  and receives the bf16 sum rounded to fp16. Sizes on both sides of the single-chunk / multi-chunk boundary
  (65536 16-bit elements) and past the R-buffer ring (16 chunks per slot). numel % 8 != 0 and dtypes other than
  fp16/bf16/fp32 are rejected on every rank without desynchronizing the ranks. A reduce no rank contributes to
  yields zeros.
- run_cpu_reduce_jobs (CPU helper): its 50 s wait timeout bounds the wait for the next job, not the length of a
  round that keeps delivering jobs (slow).
- pg_broadcast / pg_broadcast_ll (backend.broadcast picks LL at <= 2048 bytes): the source's bytes land on every
  other rank, the source is untouched; any even byte count, 2-byte aligned data (uint16 copy path), ragged tails
  after 16-byte bulk copies, payloads past the 16 MB staging ring, multi-iteration LL payloads (direct call);
  odd byte counts and 1-byte alignment are rejected.
- pg_gather (OutputGather): the output device receives the ranks' [batch, ldim_r] slices concatenated along the
  last dim in device order; the output device may hold an empty slice; payloads past the per-rank ring;
  send_ldim * esize % 128 != 0 and an ldims list of the wrong length are rejected.
- pg_gather_small (lm-head argmax gather): the same layout for any byte width (int64 indices, fp16 values,
  0/1-wide slices as the caller uses); payloads beyond the 16 KB small buffer are rejected.
- pg_barrier (fwd_barrier): no rank's stream passes the barrier before every rank's stream has reached it (a
  witness written by a sleeping rank before its barrier is visible to the other rank's stream after its barrier),
  either rank being the coordinator; also when a barrier over a subset of the ranks (the closing barrier of a
  gather over gather_devices) runs in between.
- pg_all_reduce (GPU ring, no Python caller): fp32 sum, bitwise a + b for two ranks, including payloads past the
  per-rank ring buffer.
- All of the above back to back in one stream with no host synchronization (stage counters and sequence numbers
  of one collective must not leak into the next).
- pg_init_context alone initializes every word the protocols read: a run whose context region was filled with
  garbage and re-initialized only by pg_init_context behaves the same.
"""

import random
import zlib

import pytest
import torch

from testlib.tp_native import run_ranks, single_device_ranks

pytestmark = pytest.mark.multi_gpu(2)

GUARD = 256          # sentinel bytes on each side of every output buffer
SENTINEL = 0xA5
SLEEP_CYCLES = 200_000_000   # torch.cuda._sleep: ~0.1 s, long against the barrier's spin granularity

HALF, BF16, FP32 = torch.half, torch.bfloat16, torch.float


# ---------------------------------------------------------------------------------------------------------------
# Inputs (deterministic per case and rank; shared by the rank processes and the test process)

def seed_of(cid: str, rank: int) -> int:
    return zlib.crc32(f"{cid}/{rank}".encode())


SPECIAL_F16 = [
    [3e-6, -3e-6, 6e-8, 1e-5, 65504.0, -65504.0, float("inf"), 0.0,
     -0.0, 1.0, 2048.0, 0.5, float("-inf"), 1e-4, 6.1e-5, -0.0],
    [3e-6, 3e-6, 6e-8, -1e-5, 65504.0, -65504.0, 1.0, -0.0,
     -0.0, -1.0, 1.0, 0.00048828125, -1.0, 1e-4, 6.1e-5, 0.0],
]
SPECIAL_F32 = [
    [1.00390625, -1.00390625, 3e38, float("inf"), 0.0, -0.0, 1e-30, 257.0,
     -3e38, 1.0, 65536.0, 0.3, float("-inf"), 2.5, 1e30, -0.0],
    [0.0, 0.0, 3e38, 1.0, -0.0, -0.0, 1e-30, 0.0,
     -3e38, 2 ** -9, 1.0, 0.1, 1.0, -2.5, 1e30, 0.0],
]


def ar_input(c: dict, rank: int) -> torch.Tensor:
    """all_reduce input: random normal (scaled), a special-value pattern, or garbage for a non-contributor"""
    n, dtype = c["numel"], c["dtype"]
    g = torch.Generator().manual_seed(seed_of(c["id"], rank))
    if not c["contrib"][rank]:
        # Ignored by the contract; NaN/inf garbage makes any use of it visible
        x = torch.randn(n, generator = g) * 1e4
        x[::3] = float("nan")
        x[1::7] = float("inf")
        return x.to(dtype)
    if c.get("special"):
        pattern = (SPECIAL_F16 if dtype == HALF else SPECIAL_F32)[rank]
        return torch.tensor(pattern * (n // len(pattern)), dtype = torch.float).to(dtype)
    return (torch.randn(n, generator = g) * c.get("scale", 1.0)).to(dtype)


def bytes_input(cid: str, nbytes: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed_of(cid, 0))
    return torch.randint(0, 256, (nbytes,), dtype = torch.uint8, generator = g)


def gather_input(c: dict, rank: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed_of(c["id"], rank))
    shape = (c["batch"], c["ldims"][rank])
    if c["dtype"] == torch.long:
        return torch.randint(-2 ** 62, 2 ** 62, shape, dtype = torch.long, generator = g)
    return torch.randn(shape, generator = g).to(c["dtype"])


def ring_input(c: dict, rank: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed_of(c["id"], rank))
    return torch.randn(c["numel"], generator = g)


# ---------------------------------------------------------------------------------------------------------------
# Rank side

def guarded(nbytes: int, device, offset: int = 0, fill: torch.Tensor | None = None):
    """(sentinel-filled uint8 buffer, uint8 view of nbytes at GUARD + offset), view filled from `fill` (bytes)"""
    buf = torch.full((2 * GUARD + offset + nbytes,), SENTINEL, dtype = torch.uint8, device = device)
    v = buf[GUARD + offset : GUARD + offset + nbytes]
    if fill is not None:
        v.copy_(fill.reshape(-1).view(torch.uint8))
    return buf, v


def as_dtype(v: torch.Tensor, dtype, shape = None) -> torch.Tensor:
    t = v.view(dtype) if v.numel() else torch.empty(0, dtype = dtype, device = v.device)
    return t.view(shape) if shape is not None else t


def op_all_reduce(ctx, c):
    x = ar_input(c, ctx.rank)
    buf, v = guarded(x.numel() * x.element_size(), ctx.device, fill = x)
    ctx.backend.all_reduce(as_dtype(v, x.dtype), c["contrib"][ctx.rank])
    return buf


def op_broadcast(ctx, c):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.model.model_tp_backend import SHBUF_SIZE_LL
    b = ctx.backend
    src = c["src"]
    payload = bytes_input(c["id"], c["nbytes"])
    buf, v = guarded(c["nbytes"], ctx.device, c.get("offset", 0), payload if ctx.rank == src else None)
    t = as_dtype(v, c["dtype"])
    src_device = ctx.devices[src]
    if c["path"] == "backend":
        b.broadcast(t, src_device)
    elif c["path"] == "ll":
        ext.pg_broadcast_ll(b.ptr_g, b.dev_g, b.active_devices, b.device, src_device, t, b.dev_ll, SHBUF_SIZE_LL,
                            b.abort_flag)
    else:
        ext.pg_broadcast(b.ptr_g, b.dev_g, b.active_devices, b.device, src_device, t, b.dev_b, b.shbuf_size,
                         b.abort_flag)
    return buf


def gather_plan(c, devices):
    """(gather_devices, ldims) as the callers build them: sorted, empty slices left out except the output's"""
    out_dev = devices[c["out"]]
    gd = [d for d, l in zip(devices, c["ldims"]) if l > 0 or d == out_dev]
    ld = [l for d, l in zip(devices, c["ldims"]) if l > 0 or d == out_dev]
    return gd, ld


def op_gather(ctx, c):
    gd, ld = gather_plan(c, ctx.devices)
    out_dev = ctx.devices[c["out"]]
    if ctx.device not in gd:
        return None
    x = gather_input(c, ctx.rank).to(ctx.device)
    x_before = x.clone()
    buf = out = None
    if ctx.device == out_dev:
        esize = x.element_size()
        buf, v = guarded(c["batch"] * sum(ld) * esize, ctx.device)
        out = as_dtype(v, x.dtype, (c["batch"], sum(ld)))
    fn = ctx.backend.gather_small if c["small"] else ctx.backend.gather
    fn(x, out, gd, out_dev, ld)
    return buf, torch.equal(x, x_before)


def op_barrier(ctx, c):
    # Witness word at the end of the B buffer (no other collective in flight: the suite syncs around this case)
    w = ctx.backend.tensor_b[-64:].view(torch.int32)
    if ctx.rank == c["sleeper"]:
        m = torch.tensor([c["marker"]], dtype = torch.int32, device = ctx.device)
        torch.cuda._sleep(SLEEP_CYCLES)
        w[0:1].copy_(m, non_blocking = True)
        ctx.backend.fwd_barrier()
        return None
    ctx.backend.fwd_barrier()
    got = torch.empty(1, dtype = torch.int32, device = ctx.device)
    got.copy_(w[0:1], non_blocking = True)
    return got


def op_ring(ctx, c):
    from exllamav3.ext import exllamav3_ext as ext
    b = ctx.backend
    x = ring_input(c, ctx.rank)
    buf, v = guarded(x.numel() * 4, ctx.device, fill = x)
    ext.pg_all_reduce(b.ptr_g, b.dev_g, b.active_devices, b.device, b.active_devices[0], as_dtype(v, FP32), b.dev_b,
                      b.shbuf_size, b.abort_flag)
    return buf


def op_reject(ctx, c):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.model.model_tp_backend import SHBUF_SIZE_LL, SHBUF_SIZE_S
    b, dev = ctx.backend, ctx.device
    what = c["what"]
    try:
        if what == "all_reduce_numel":
            b.all_reduce(torch.zeros(12, dtype = HALF, device = dev))
        elif what == "all_reduce_dtype":
            b.all_reduce(torch.zeros(64, dtype = torch.int32, device = dev))
        elif what == "broadcast_odd":
            ext.pg_broadcast(b.ptr_g, b.dev_g, b.active_devices, dev, b.active_devices[0],
                             torch.zeros(4099, dtype = torch.uint8, device = dev), b.dev_b, b.shbuf_size, b.abort_flag)
        elif what == "broadcast_ll_odd":
            ext.pg_broadcast_ll(b.ptr_g, b.dev_g, b.active_devices, dev, b.active_devices[0],
                                torch.zeros(3, dtype = torch.uint8, device = dev), b.dev_ll, SHBUF_SIZE_LL, b.abort_flag)
        elif what == "broadcast_misaligned":
            t = torch.zeros(8192, dtype = torch.uint8, device = dev)[1:4097]
            ext.pg_broadcast(b.ptr_g, b.dev_g, b.active_devices, dev, b.active_devices[0], t, b.dev_b, b.shbuf_size,
                             b.abort_flag)
        elif what == "gather_ldim":
            t = torch.zeros(2, 60, dtype = HALF, device = dev)
            out = torch.zeros(2, 120, dtype = HALF, device = dev) if dev == ctx.output_device else None
            ext.pg_gather(b.ptr_g, b.dev_g, b.active_devices, dev, ctx.output_device, t, out, [60] * ctx.world,
                          b.dev_b, b.shbuf_size, b.abort_flag)
        elif what == "gather_ldims_count":
            t = torch.zeros(2, 64, dtype = HALF, device = dev)
            out = torch.zeros(2, 128, dtype = HALF, device = dev) if dev == ctx.output_device else None
            ext.pg_gather(b.ptr_g, b.dev_g, b.active_devices, dev, ctx.output_device, t, out, [64] * (ctx.world + 1),
                          b.dev_b, b.shbuf_size, b.abort_flag)
        elif what == "gather_small_capacity":
            t = torch.zeros(1100, 1, dtype = torch.long, device = dev)
            out = torch.zeros(1100, ctx.world, dtype = torch.long, device = dev) if dev == ctx.output_device else None
            ext.pg_gather_small(b.ptr_g, b.dev_g, b.active_devices, dev, ctx.output_device, t, out, [1] * ctx.world,
                                b.dev_s, SHBUF_SIZE_S, b.abort_flag)
        else:
            raise AssertionError(what)
    except RuntimeError as e:
        return str(e)
    return None


def stress_cases(seed: int, n: int, world: int) -> list[dict]:
    """A deterministic random mix of collectives, sizes and roots"""
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        cid = f"stress{seed}_{i}"
        kind = rng.choice(["all_reduce", "all_reduce", "broadcast", "broadcast", "gather", "gather_small", "barrier",
                           "ring"])
        if kind == "all_reduce":
            numel = 8 * rng.choice([1, rng.randint(1, 64), rng.randint(64, 8192), rng.randint(8192, 40000)])
            contrib = tuple(rng.random() < 0.8 for _ in range(world))
            if not any(contrib):
                contrib = (True,) + contrib[1:]
            cases.append(dict(op = "all_reduce", id = cid, dtype = rng.choice([HALF, BF16, FP32]), numel = numel,
                              contrib = contrib))
        elif kind == "broadcast":
            nbytes = 2 * rng.choice([rng.randint(1, 1024), rng.randint(1024, 40000)])
            cases.append(dict(op = "broadcast", id = cid, dtype = torch.uint8, nbytes = nbytes,
                              src = rng.randrange(world), path = "backend"))
        elif kind == "gather":
            ldims = [64 * rng.randint(1, 8) for _ in range(world)]
            cases.append(dict(op = "gather", id = cid, dtype = HALF, batch = rng.randint(1, 40), ldims = ldims,
                              out = rng.randrange(world), small = False))
        elif kind == "gather_small":
            cases.append(dict(op = "gather", id = cid, dtype = torch.long, batch = rng.randint(1, 8),
                              ldims = [1] * world, out = rng.randrange(world), small = True))
        elif kind == "ring":
            cases.append(dict(op = "ring", id = cid, numel = 4 * rng.randint(1, 50000)))
        else:
            cases.append(dict(op = "barrier_plain", id = cid))
    return cases


def op_stress(ctx, c):
    outs = []
    for sc in stress_cases(c["seed"], c["n"], ctx.world):
        if sc["op"] == "barrier_plain":
            ctx.backend.fwd_barrier()
            outs.append(None)
        else:
            outs.append(OPS[sc["op"]](ctx, sc))
    return outs


OPS = dict(all_reduce = op_all_reduce, broadcast = op_broadcast, gather = op_gather, barrier = op_barrier,
           ring = op_ring, reject = op_reject, stress = op_stress)


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    return x


def suite(ctx, cases):
    """Rank program: run the cases in order, returning {id: output moved to the CPU}"""
    out = {}
    for c in cases:
        if c.get("sync"):
            ctx.sync()
        out[c["id"]] = to_cpu(OPS[c["op"]](ctx, c))
        # One CPU-reduce round per case, like one forward pass
        ctx.end_round()
        if c.get("sync"):
            ctx.sync()
    return out


# ---------------------------------------------------------------------------------------------------------------
# References

def bf16_rhaz(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 as the wire kernels and the CPU accumulate round: (bits + 0x8000) >> 16"""
    b = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    r = ((b + 0x8000) >> 16) & 0xFFFF
    return torch.where(r >= 0x8000, r - 0x10000, r).to(torch.int16).view(torch.bfloat16)


def all_reduce_ref(c: dict, world: int, fp16_wire: bool = True) -> torch.Tensor:
    """Two-rank reference of pg_all_reduce_cpu's result (contributions are summed pairwise, in arrival order beyond
    two ranks)"""
    assert world == 2
    dtype = c["dtype"]
    xs = [ar_input(c, r) for r in range(world) if c["contrib"][r]]
    n = c["numel"]
    if dtype == HALF and fp16_wire:
        if not xs:
            return torch.zeros(n, dtype = HALF)
        if len(xs) == 1:
            return xs[0]
        return (xs[0].float() + xs[1].float()).half()
    ws = [x if dtype == BF16 else bf16_rhaz(x.float()) for x in xs]
    if not ws:
        s = torch.zeros(n, dtype = BF16)
    elif len(ws) == 1:
        s = ws[0]
    else:
        s = bf16_rhaz(ws[0].float() + ws[1].float())
    return s.to(dtype) if dtype != HALF else s.float().half()


def with_guards(payload: torch.Tensor, offset: int = 0) -> torch.Tensor:
    p = payload.contiguous().reshape(-1).view(torch.uint8) if payload.numel() else torch.empty(0, dtype = torch.uint8)
    g = torch.full((GUARD + offset,), SENTINEL, dtype = torch.uint8)
    return torch.cat([g, p, torch.full((GUARD,), SENTINEL, dtype = torch.uint8)])


def sentinel_buffer(nbytes: int, offset: int = 0) -> torch.Tensor:
    return torch.full((2 * GUARD + offset + nbytes,), SENTINEL, dtype = torch.uint8)


def gather_ref(c: dict, world: int, devices) -> torch.Tensor:
    gd, _ = gather_plan(c, devices)
    parts = [gather_input(c, r) for r in range(world) if devices[r] in gd]
    return torch.cat(parts, dim = -1)


def mismatch(got: torch.Tensor, want: torch.Tensor) -> str:
    """Description of the first differing bytes (guards included)"""
    if got.shape != want.shape:
        return f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    d = (got != want).nonzero().flatten()
    if not d.numel():
        return ""
    i = d[0].item()
    region = "front guard" if i < GUARD else ("back guard" if i >= got.numel() - GUARD else "payload")
    return f"{d.numel()} bytes differ, first at byte {i} ({region}): got {got[i].item():#x}, want {want[i].item():#x}"


# ---------------------------------------------------------------------------------------------------------------
# Case tables

def ar_cases(world: int) -> list[dict]:
    full = (True,) * world
    cases = []
    for dtype in (HALF, BF16, FP32):
        name = {HALF: "f16", BF16: "bf16", FP32: "f32"}[dtype]
        for numel in (8, 1032, 65536, 65544, 3 * 65536 + 8, (1 << 21) + 8):
            cases.append(dict(op = "all_reduce", id = f"ar_{name}_{numel}", dtype = dtype, numel = numel,
                              contrib = full))
        cases.append(dict(op = "all_reduce", id = f"ar_{name}_special", dtype = dtype, numel = 16 * 64,
                          contrib = full, special = True))
        cases.append(dict(op = "all_reduce", id = f"ar_{name}_scaled", dtype = dtype, numel = 4096, contrib = full,
                          scale = 3000.0 if dtype == HALF else 1e20))
        for r in range(world):
            contrib = tuple(i != r for i in range(world))
            for numel in (1024, 200008):
                cases.append(dict(op = "all_reduce", id = f"ar_{name}_{numel}_skip{r}", dtype = dtype,
                                  numel = numel, contrib = contrib))
    cases.append(dict(op = "all_reduce", id = "ar_f32_big", dtype = FP32, numel = 6 << 20, contrib = full))
    return cases


def no_contrib_cases(world: int) -> list[dict]:
    """A reduce with no contributor after the R-buffer ring has gone round once (accumulator slots reused)"""
    full = (True,) * world
    none = (False,) * world
    cases = [dict(op = "all_reduce", id = f"nc_fill_{i}", dtype = HALF, numel = 1024, contrib = full)
             for i in range(20)]
    cases.append(dict(op = "all_reduce", id = "nc_none_single", dtype = HALF, numel = 1024, contrib = none))
    cases += [dict(op = "all_reduce", id = f"nc_fill_mc_{i}", dtype = FP32, numel = 200008, contrib = full)
              for i in range(3)]
    cases.append(dict(op = "all_reduce", id = "nc_none_multi", dtype = FP32, numel = 200008, contrib = none))
    return cases


BC_SIZES = [  # (nbytes, dtype, offset bytes, path)
    (2, HALF, 0, "backend"), (6, HALF, 0, "backend"), (2046, HALF, 2, "backend"), (2048, FP32, 0, "backend"),
    (2050, HALF, 0, "backend"), (16384, FP32, 0, "backend"), (16386, HALF, 0, "backend"),
    (16384 * 3 + 2, HALF, 2, "backend"), (1 << 20, BF16, 0, "backend"), (20 << 20, FP32, 0, "backend"),
    ((5 << 20) + 6, HALF, 2, "backend"), (0, HALF, 0, "backend"),
    (8192 + 6, HALF, 0, "ll"), (20000, HALF, 2, "ll"), (6, HALF, 2, "ll"), (4, torch.uint8, 0, "staged"),
    (100, torch.uint8, 0, "staged"),
]


def bc_cases(world: int) -> list[dict]:
    cases = []
    for src in range(world):
        for nbytes, dtype, offset, path in BC_SIZES:
            cases.append(dict(op = "broadcast", id = f"bc_{path}_{nbytes}_{offset}_from{src}", dtype = dtype,
                              nbytes = nbytes, offset = offset, src = src, path = path))
    # Alternating roots back to back on the LL path (sequence/epoch handoff between producers)
    for i in range(40):
        cases.append(dict(op = "broadcast", id = f"bc_alt_{i}", dtype = HALF, nbytes = 2 * (i + 1), offset = 0,
                          src = i % world, path = "backend"))
    return cases


def gather_cases(world: int) -> list[dict]:
    """Every rank takes part; only the output rank may hold an empty slice (the callers leave other empty ranks
    out of gather_devices, and never gather with a single participant)"""
    assert world == 2
    cases = []
    for out in range(world):
        empty = [0 if r == out else 128 for r in range(world)]
        for dtype, batch, ldims in ((HALF, 1, (128, 128)), (HALF, 7, (256, 64)), (FP32, 5, (32, 96)),
                                    (HALF, 3, empty), (HALF, 2200, (2048, 2304)), (FP32, 1, (64, 32))):
            cases.append(dict(op = "gather", id = f"g_{dtype}_{batch}_{tuple(ldims)}_to{out}".replace(" ", ""),
                              dtype = dtype, batch = batch, ldims = list(ldims), out = out, small = False))
        empty = [0 if r == out else 1 for r in range(world)]
        for dtype, batch, ldims in ((torch.long, 1, (1, 1)), (HALF, 1, (1, 1)), (torch.long, 5, empty),
                                    (HALF, 300, (1, 1)), (HALF, 9, (3, 5)), (torch.long, 1000, (1, 1))):
            cases.append(dict(op = "gather", id = f"gs_{dtype}_{batch}_{tuple(ldims)}_to{out}".replace(" ", ""),
                              dtype = dtype, batch = batch, ldims = list(ldims), out = out, small = True))
    return cases


def barrier_cases(world: int) -> list[dict]:
    return [dict(op = "barrier", id = f"barrier_sleep{r}_{i}", sleeper = r, marker = 1000 + 10 * i + r, sync = True)
            for i in range(3) for r in range(world)]


RING_SIZES = (4, 1000, 4096 * 3 + 4, 1 << 20, 9 << 20)

REJECTS = {   # what -> expected message fragment
    "all_reduce_numel": "multiple of 16",
    "all_reduce_dtype": "Unknown dtype",
    "broadcast_odd": "multiple of 2",
    "broadcast_ll_odd": "multiple of 2",
    "broadcast_misaligned": "aligned to 2 bytes",
    "gather_ldim": "multiple of 128",
    "gather_ldims_count": "one ldim per active device",
    "gather_small_capacity": "Shared buffer too small",
}


def ring_cases(world: int) -> list[dict]:
    cases = [dict(op = "ring", id = f"ring_{n}", numel = n) for n in RING_SIZES]
    cases += [dict(op = "ring", id = f"ring_b2b_{i}", numel = 1024 * (i % 5 + 1)) for i in range(40)]
    return cases


def full_suite(world: int) -> list[dict]:
    cases = barrier_cases(world)
    cases += ar_cases(world) + bc_cases(world) + gather_cases(world) + ring_cases(world)
    # Rejections, each followed by a reduce that must still line up across the ranks
    for what in REJECTS:
        cases.append(dict(op = "reject", id = f"reject_{what}", what = what))
        cases.append(dict(op = "all_reduce", id = f"after_reject_{what}", dtype = HALF, numel = 512,
                          contrib = (True,) * world))
    cases.append(dict(op = "stress", id = "stress", seed = 7, n = 300))
    cases += no_contrib_cases(world)
    return cases


# ---------------------------------------------------------------------------------------------------------------
# Runs

def other_device(device, devices):
    others = [d for d in devices if d != device]
    if not others:
        pytest.skip("needs a second visible device for the single-device ordinal mapping")
    return others[0]


@pytest.fixture(scope = "module")
def runs(device, devices):
    """{config: (ranks, results)}, each configuration run once"""
    out = {}

    def get(config):
        if config not in out:
            world = 2
            if config == "two_gpu":
                ranks = [devices[0].index, devices[1].index]
                results = run_ranks(suite, ranks, full_suite(world))
            elif config == "one_gpu":
                ranks, envs = single_device_ranks(device, other_device(device, devices))
                results = run_ranks(suite, ranks, full_suite(world), rank_envs = envs)
            elif config == "no_fp16_wire":
                ranks = [devices[0].index, devices[1].index]
                results = run_ranks(suite, ranks, ar_cases(world), env = {"EXL3_TP_NO_FP16_WIRE": "1"})
            elif config == "scrambled_context":
                ranks = [devices[0].index, devices[1].index]
                results = run_ranks(suite, ranks, scrambled_cases(world), scramble_context = True)
            out[config] = (ranks, results)
        return out[config]

    return get


CONFIGS = ("two_gpu", "one_gpu")


def results_for(runs, config, cid):
    ranks, results = runs(config)
    return ranks, [r[cid] for r in results]


def check_case(c: dict, outs: list, ranks: list, fp16_wire: bool = True):
    """Assert one case's per-rank outputs against its reference"""
    world = len(ranks)
    op, cid = c["op"], c["id"]
    if op == "all_reduce":
        want = with_guards(all_reduce_ref(c, world, fp16_wire))
        for r, got in enumerate(outs):
            m = mismatch(got, want)
            assert not m, f"{cid}: rank {r}: {m}"
    elif op == "broadcast":
        want = with_guards(bytes_input(cid, c["nbytes"]), c.get("offset", 0))
        for r, got in enumerate(outs):
            m = mismatch(got, want)
            assert not m, f"{cid}: rank {r}{' (source)' if r == c['src'] else ''}: {m}"
    elif op == "gather":
        want = with_guards(gather_ref(c, world, ranks))
        for r, got in enumerate(outs):
            if got is None:
                assert c["ldims"][r] == 0 and r != c["out"], f"{cid}: rank {r} did not take part"
                continue
            buf, input_kept = got
            assert input_kept, f"{cid}: rank {r} input modified"
            if r == c["out"]:
                m = mismatch(buf, want)
                assert not m, f"{cid}: output rank {r}: {m}"
            else:
                assert buf is None
    elif op == "barrier":
        for r, got in enumerate(outs):
            if r != c["sleeper"]:
                assert got.item() == c["marker"], \
                    f"{cid}: rank {r} passed the barrier before the sleeping rank {c['sleeper']} reached it " \
                    f"(witness {got.item()}, want {c['marker']})"
    elif op == "ring":
        want = with_guards(sum(ring_input(c, r) for r in range(world)))
        for r, got in enumerate(outs):
            m = mismatch(got, want)
            assert not m, f"{cid}: rank {r}: {m}"
    elif op == "reject":
        for r, msg in enumerate(outs):
            assert msg is not None and REJECTS[c["what"]] in msg, f"{cid}: rank {r}: {msg!r}"
    elif op == "barrier_plain":
        pass
    else:
        raise AssertionError(op)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", ar_cases(2), ids = lambda c: c["id"])
def test_all_reduce_cpu(runs, config, case):
    ranks, outs = results_for(runs, config, case["id"])
    check_case(case, outs, ranks)


@pytest.mark.parametrize("case", ar_cases(2), ids = lambda c: c["id"])
def test_all_reduce_cpu_bf16_wire_for_fp16(runs, case):
    """EXL3_TP_NO_FP16_WIRE=1: fp16 payloads take the bf16 wire"""
    ranks, outs = results_for(runs, "no_fp16_wire", case["id"])
    check_case(case, outs, ranks, fp16_wire = False)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("cid", ["nc_none_single", "nc_none_multi"])
def test_all_reduce_cpu_no_contributor(runs, config, cid):
    """A reduce no rank contributes to is the empty sum (zero), also once the accumulator slot it lands in held an
    earlier sum"""
    case = next(c for c in no_contrib_cases(2) if c["id"] == cid)
    ranks, outs = results_for(runs, config, cid)
    check_case(case, outs, ranks)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", bc_cases(2), ids = lambda c: c["id"])
def test_broadcast(runs, config, case):
    ranks, outs = results_for(runs, config, case["id"])
    check_case(case, outs, ranks)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", gather_cases(2), ids = lambda c: c["id"])
def test_gather(runs, config, case):
    ranks, outs = results_for(runs, config, case["id"])
    check_case(case, outs, ranks)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", barrier_cases(2), ids = lambda c: c["id"])
def test_barrier(runs, config, case):
    ranks, outs = results_for(runs, config, case["id"])
    check_case(case, outs, ranks)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("case", ring_cases(2), ids = lambda c: c["id"])
def test_ring_all_reduce(runs, config, case):
    ranks, outs = results_for(runs, config, case["id"])
    check_case(case, outs, ranks)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("what", list(REJECTS))
def test_rejected_inputs(runs, config, what):
    """Invalid inputs raise on every rank before anything is launched, and the next collective still lines up"""
    ranks, outs = results_for(runs, config, f"reject_{what}")
    check_case(dict(op = "reject", id = f"reject_{what}", what = what), outs, ranks)
    c = dict(op = "all_reduce", id = f"after_reject_{what}", dtype = HALF, numel = 512, contrib = (True, True))
    check_case(c, results_for(runs, config, c["id"])[1], ranks)


@pytest.mark.parametrize("config", CONFIGS)
def test_interleaved_back_to_back(runs, config):
    """300 mixed collectives enqueued with no host synchronization"""
    ranks, outs = results_for(runs, config, "stress")
    for i, sc in enumerate(stress_cases(7, 300, 2)):
        check_case(sc, [o[i] for o in outs], ranks)


def scrambled_cases(world: int) -> list[dict]:
    pick = lambda cs: cs[::5]
    return barrier_cases(world)[:2] + pick(ar_cases(world)) + pick(bc_cases(world)) + pick(gather_cases(world)) \
        + pick(ring_cases(world))


def test_context_init_alone(runs):
    """pg_init_context alone initializes every word the protocols read: the cases of a run whose context region
    was filled with random bytes before pg_init_context"""
    ranks, results = runs("scrambled_context")
    for c in scrambled_cases(2):
        check_case(c, [r[c["id"]] for r in results], ranks)


# ---------------------------------------------------------------------------------------------------------------
# Collectives over different device subsets in sequence

def subset_program(ctx):
    """Rank 1: (sleep) barrier over [rank 1] alone, write the witness, barrier over both ranks. Rank 0: barrier over
    both ranks, read the witness. The two-rank barrier must hold rank 0 until rank 1 arrives at it, after the witness
    write; the one-rank barrier in between is no part of it"""
    from exllamav3.ext import exllamav3_ext as ext
    b = ctx.backend
    w = b.tensor_b[-64:].view(torch.int32)
    both = list(ctx.devices)
    if ctx.rank == 1:
        m = torch.tensor([4242], dtype = torch.int32, device = ctx.device)
        torch.cuda._sleep(SLEEP_CYCLES)
        ext.pg_barrier(b.ptr_g, b.dev_g, [ctx.device], ctx.device, b.abort_flag)
        w[0:1].copy_(m, non_blocking = True)
        ext.pg_barrier(b.ptr_g, b.dev_g, both, ctx.device, b.abort_flag)
        torch.cuda.synchronize()
        return None, b.abort_flag.item()
    ext.pg_barrier(b.ptr_g, b.dev_g, both, ctx.device, b.abort_flag)
    got = torch.empty(1, dtype = torch.int32, device = ctx.device)
    got.copy_(w[0:1], non_blocking = True)
    torch.cuda.synchronize()
    return got.item(), b.abort_flag.item()


def test_barrier_after_subset_barrier(devices):
    """A barrier over a subset of the ranks (as pg_gather / pg_gather_small hold over their gather_devices) must not
    release a rank already waiting in the next barrier over all ranks. On failure rank 1 also waits out the sync
    timeout in its two-rank barrier"""
    ranks = [devices[0].index, devices[1].index]
    (witness, abort0), (_, abort1) = run_ranks(subset_program, ranks)
    assert witness == 4242 and not abort1, \
        f"rank 0 left the two-rank barrier before rank 1 reached it (witness {witness}, want 4242); rank 1 " \
        f"{'then timed out in it' if abort1 else 'did not time out'}"


def long_round_program(ctx, rounds: int, pause: float):
    """One CPU-reduce round of `rounds` small reduces `pause` seconds apart (a long forward pass)"""
    import time
    outs = []
    for i in range(rounds):
        x = torch.full((1024,), float(ctx.rank + i), dtype = HALF, device = ctx.device)
        ctx.backend.all_reduce(x)
        outs.append(x.cpu())
        time.sleep(pause)
    return outs


@pytest.mark.slow
def test_cpu_reduce_round_longer_than_timeout(devices):
    """run_cpu_reduce_jobs' wait timeout ("after 50 seconds of waiting for the queue to build") bounds the wait for
    the next job, not the length of a round that keeps delivering jobs"""
    ranks = [devices[0].index, devices[1].index]
    rounds, pause = 30, 2.0
    outs = run_ranks(long_round_program, ranks, rounds, pause, timeout = 300)
    for i in range(rounds):
        want = torch.full((1024,), float(2 * i + 1), dtype = HALF)
        assert torch.equal(outs[0][i], want) and torch.equal(outs[1][i], want), i


# ---------------------------------------------------------------------------------------------------------------
# Empty operands: every rank sees the same shape, so each collective is a no-op on all of them, and the next
# collective still lines up

def empty_program(ctx):
    b = ctx.backend
    out_dev = ctx.devices[0]
    is_out = ctx.device == out_dev
    for dtype in (HALF, torch.float):
        b.all_reduce(torch.empty(0, dtype = dtype, device = ctx.device))
    b.broadcast(torch.empty(0, dtype = HALF, device = ctx.device), ctx.devices[1])
    for fn in (b.gather, b.gather_small):
        # No rows
        x = torch.empty(0, 64, dtype = HALF, device = ctx.device)
        out = torch.empty(0, 64 * ctx.world, dtype = HALF, device = ctx.device) if is_out else None
        fn(x, out, list(ctx.devices), out_dev, [64] * ctx.world)
        # No columns on any rank (empty output tensor, null data pointer)
        x = torch.empty(3, 0, dtype = HALF, device = ctx.device)
        out = torch.empty(3, 0, dtype = HALF, device = ctx.device) if is_out else None
        fn(x, out, list(ctx.devices), out_dev, [0] * ctx.world)
    ctx.end_round()
    x = torch.full((1024,), float(ctx.rank + 1), dtype = HALF, device = ctx.device)
    b.all_reduce(x)
    return x.cpu()


def test_empty_collectives(devices):
    """Zero-element all-reduce (fp16 and fp32), broadcast and gathers (no rows, no columns) return on every rank
    without a hang or a launch error, and a following all-reduce is correct"""
    ranks = [devices[0].index, devices[1].index]
    outs = run_ranks(empty_program, ranks, timeout = 120)
    for r in outs:
        assert torch.equal(r, torch.full((1024,), 3.0, dtype = HALF))
