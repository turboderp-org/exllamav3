"""
Host-side validation of the native TP collectives (parallel/*.cu), in one process with a one-rank TPBackendNative
(real shared memory, registered and mapped, context initialized), so nothing here waits on another rank:

- Every TORCH_CHECK the bindings carry raises before anything is launched: pg_broadcast / pg_broadcast_ll (odd
  byte count, data or staging buffer misaligned), pg_gather (row bytes not a multiple of 128, ldims count, output
  last dim), pg_gather_small (ldims count, output last dim, staging capacity), pg_all_reduce (bytes % 16),
  pg_all_reduce_cpu (numel % 8, dtype).
- A synchronization timeout is sticky: once PGContext::sync_timeout is set (by any rank's kernel), every
  collective entry point raises "Synchronization timeout" instead of launching.
- One-rank collectives complete: pg_barrier, pg_broadcast(_ll) from self (tensor unchanged), pg_gather_small
  (the output is the input). (pg_gather with one rank is a no-op; OutputGather returns its input directly.)
"""

import uuid

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.model_tp_backend import SHBUF_SIZE_LL, SHBUF_SIZE_R, SHBUF_SIZE_S, TPBackendNative


@pytest.fixture
def backend(device):
    b = TPBackendNative(device.index, [device.index], device.index, "", master = True, uuid = "exl3t_" + uuid.uuid4().hex[:16])
    try:
        yield b
    finally:
        torch.cuda.synchronize(device)
        b.close()


def calls(b, dev):
    """name -> zero-argument call of each collective entry point with valid one-rank arguments"""
    d = dev.index
    t16 = torch.zeros(64, dtype = torch.half, device = dev)
    g = torch.zeros(2, 64, dtype = torch.half, device = dev)
    go = torch.zeros(2, 64, dtype = torch.half, device = dev)
    s = torch.zeros(2, 1, dtype = torch.long, device = dev)
    so = torch.zeros(2, 1, dtype = torch.long, device = dev)
    return {
        "pg_barrier": lambda: ext.pg_barrier(b.ptr_g, b.dev_g, [d], d, b.abort_flag),
        "pg_broadcast": lambda: ext.pg_broadcast(b.ptr_g, b.dev_g, [d], d, d, t16, b.dev_b, b.shbuf_size, b.abort_flag),
        "pg_broadcast_ll": lambda: ext.pg_broadcast_ll(b.ptr_g, b.dev_g, [d], d, d, t16, b.dev_ll, SHBUF_SIZE_LL,
                                                       b.abort_flag),
        "pg_gather": lambda: ext.pg_gather(b.ptr_g, b.dev_g, [d], d, d, g, go, [64], b.dev_b, b.shbuf_size, b.abort_flag),
        "pg_gather_small": lambda: ext.pg_gather_small(b.ptr_g, b.dev_g, [d], d, d, s, so, [1], b.dev_s, SHBUF_SIZE_S,
                                                       b.abort_flag),
        "pg_all_reduce": lambda: ext.pg_all_reduce(b.ptr_g, b.dev_g, [d], d, d, t16.float(), b.dev_b, b.shbuf_size,
                                                   b.abort_flag),
        "pg_all_reduce_cpu": lambda: ext.pg_all_reduce_cpu(b.ptr_g, b.dev_g, [d], d, d, t16, True, b.dev_r, SHBUF_SIZE_R,
                                                           False, b.abort_flag),
    }


def invalid_calls(b, dev):
    """(name, call, expected message fragment)"""
    d = dev.index
    h = lambda *shape: torch.zeros(*shape, dtype = torch.half, device = dev)
    u8 = torch.zeros(4096, dtype = torch.uint8, device = dev)
    out = []
    for fn, buf, size in (("pg_broadcast", b.dev_b, b.shbuf_size), ("pg_broadcast_ll", b.dev_ll, SHBUF_SIZE_LL)):
        f = getattr(ext, fn)
        out += [
            (f"{fn}_odd_bytes", lambda f = f, buf = buf, size = size:
                f(b.ptr_g, b.dev_g, [d], d, d, u8[:3], buf, size, b.abort_flag), "multiple of 2"),
            (f"{fn}_data_misaligned", lambda f = f, buf = buf, size = size:
                f(b.ptr_g, b.dev_g, [d], d, d, u8[1:5], buf, size, b.abort_flag), "data_ptr must be aligned to 2"),
        ]
    out += [
        ("pg_broadcast_shbuf_misaligned", lambda: ext.pg_broadcast(b.ptr_g, b.dev_g, [d], d, d, h(8), b.dev_b + 1,
                                                                   b.shbuf_size, b.abort_flag), "shbuf_ptr must be aligned to 2"),
        ("pg_broadcast_ll_shbuf_misaligned", lambda: ext.pg_broadcast_ll(b.ptr_g, b.dev_g, [d], d, d, h(8), b.dev_ll + 4,
                                                                         SHBUF_SIZE_LL, b.abort_flag), "shbuf_ptr must be aligned to 8"),
        ("pg_gather_row_bytes", lambda: ext.pg_gather(b.ptr_g, b.dev_g, [d], d, d, h(2, 72), h(2, 72), [72], b.dev_b,
                                                      b.shbuf_size, b.abort_flag), "multiple of 128"),
        ("pg_gather_ldims_count", lambda: ext.pg_gather(b.ptr_g, b.dev_g, [d], d, d, h(2, 64), h(2, 64), [64, 64],
                                                        b.dev_b, b.shbuf_size, b.abort_flag), "one ldim per active device"),
        ("pg_gather_out_dim", lambda: ext.pg_gather(b.ptr_g, b.dev_g, [d], d, d, h(2, 64), h(2, 128), [64], b.dev_b,
                                                    b.shbuf_size, b.abort_flag), "last dimension mismatch"),
        ("pg_gather_small_ldims_count", lambda: ext.pg_gather_small(b.ptr_g, b.dev_g, [d], d, d, h(2, 1), h(2, 1),
                                                                    [1, 1], b.dev_s, SHBUF_SIZE_S, b.abort_flag),
            "one ldim per active device"),
        ("pg_gather_small_out_dim", lambda: ext.pg_gather_small(b.ptr_g, b.dev_g, [d], d, d, h(2, 1), h(2, 3), [1],
                                                                b.dev_s, SHBUF_SIZE_S, b.abort_flag), "last dimension mismatch"),
        ("pg_gather_small_capacity", lambda: ext.pg_gather_small(b.ptr_g, b.dev_g, [d], d, d, h(4097, 2), h(4097, 2),
                                                                 [2], b.dev_s, SHBUF_SIZE_S, b.abort_flag), "Shared buffer too small"),
        ("pg_all_reduce_bytes", lambda: ext.pg_all_reduce(b.ptr_g, b.dev_g, [d], d, d, torch.zeros(6, device = dev),
                                                          b.dev_b, b.shbuf_size, b.abort_flag), "multiple of 16"),
        ("pg_all_reduce_cpu_numel", lambda: ext.pg_all_reduce_cpu(b.ptr_g, b.dev_g, [d], d, d, h(12), True, b.dev_r,
                                                                  SHBUF_SIZE_R, False, b.abort_flag), "multiple of 16"),
        ("pg_all_reduce_cpu_dtype", lambda: ext.pg_all_reduce_cpu(b.ptr_g, b.dev_g, [d], d, d,
                                                                  torch.zeros(16, dtype = torch.int32, device = dev), True,
                                                                  b.dev_r, SHBUF_SIZE_R, False, b.abort_flag), "Unknown dtype"),
    ]
    return out


def _invalid_names():
    return ["pg_broadcast_odd_bytes", "pg_broadcast_data_misaligned", "pg_broadcast_ll_odd_bytes",
            "pg_broadcast_ll_data_misaligned", "pg_broadcast_shbuf_misaligned", "pg_broadcast_ll_shbuf_misaligned",
            "pg_gather_row_bytes", "pg_gather_ldims_count", "pg_gather_out_dim", "pg_gather_small_ldims_count",
            "pg_gather_small_out_dim", "pg_gather_small_capacity", "pg_all_reduce_bytes", "pg_all_reduce_cpu_numel",
            "pg_all_reduce_cpu_dtype"]


@pytest.mark.parametrize("name", _invalid_names())
def test_rejected(backend, device, name):
    table = {n: (f, msg) for n, f, msg in invalid_calls(backend, device)}
    assert set(table) == set(_invalid_names())
    f, msg = table[name]
    with pytest.raises(RuntimeError, match = msg):
        f()
    torch.cuda.synchronize(device)
    # Nothing was launched that could have tripped the context
    assert backend.abort_flag.item() == 0
    assert backend.tensor_g[:4].view(torch.int32).item() == 0


@pytest.mark.parametrize("name", ["pg_barrier", "pg_broadcast", "pg_broadcast_ll", "pg_gather", "pg_gather_small",
                                  "pg_all_reduce", "pg_all_reduce_cpu"])
def test_sticky_timeout(backend, device, name):
    sync_timeout = backend.tensor_g[:4].view(torch.int32)
    sync_timeout[0] = 1
    try:
        with pytest.raises(RuntimeError, match = "Synchronization timeout"):
            calls(backend, device)[name]()
    finally:
        sync_timeout[0] = 0


def test_one_rank_collectives(backend, device):
    c = calls(backend, device)
    d = device.index
    b = backend
    c["pg_barrier"]()
    x = torch.randn(1000, device = device).half()
    x0 = x.clone()
    ext.pg_broadcast(b.ptr_g, b.dev_g, [d], d, d, x, b.dev_b, b.shbuf_size, b.abort_flag)
    ext.pg_broadcast_ll(b.ptr_g, b.dev_g, [d], d, d, x[:100], b.dev_ll, SHBUF_SIZE_LL, b.abort_flag)
    s = torch.randint(0, 1 << 40, (7, 3), device = device)
    so = torch.zeros_like(s)
    ext.pg_gather_small(b.ptr_g, b.dev_g, [d], d, d, s, so, [3], b.dev_s, SHBUF_SIZE_S, b.abort_flag)
    torch.cuda.synchronize(device)
    assert torch.equal(x, x0)
    assert torch.equal(so, s)
    assert b.abort_flag.item() == 0
