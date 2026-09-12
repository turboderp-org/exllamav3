"""Focused ROCm coverage for architecture paths whose CUDA-only bindings are absent."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm architecture binding-gate tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as native_ext
from exllamav3.modules import dsv4
from exllamav3.modules import hyperconnections as hc
from exllamav3.modules import ngram_embedding as ne
from exllamav3.modules import ple
from exllamav3.modules.attention_fn import bighead_scalar
from exllamav3.modules.attention_fn import torch as torch_attn
from exllamav3.modules.attention_fn.common import AttnArgs
from exllamav3.modules.quant.exl3_lib.ngram_codec import (
    ROW_DIM,
    dequant_rows,
    mul1_codebook,
    pack_rows,
)

DEVICE = "cuda"


def _ngram_module(*, trellis: bool):
    module = object.__new__(ne.NGramEmbedding)
    module.ngram_size = 2
    module.context_len = 1
    module.heads_per_ngram = 1
    module.num_heads = 1
    module.ple_embed_dim = ROW_DIM
    module.head_dim = ROW_DIM
    module.eos_token_id = 99
    module.head_offsets = torch.tensor([0], dtype = torch.long)
    module.head_vocab_sizes = torch.tensor([7], dtype = torch.long)
    module.layer_multipliers = torch.tensor([3, 5], dtype = torch.long)
    module.rows_per_shard = 7
    module.num_rows = 7
    module.device = torch.device(DEVICE)
    module.handles = None
    module.out_dtype = torch.half
    module._pin = None
    if trellis:
        module.mode = "trellis_ram"
        module.K = 1
        states = torch.arange(7 * ROW_DIM, dtype = torch.int64).view(7, ROW_DIM)
        scales = torch.linspace(0.25, 1.0, 7, dtype = torch.half)
        module.tables = [pack_rows(states, scales, module.K)]
        module.head_bias = torch.linspace(-0.2, 0.2, ROW_DIM, device = DEVICE,
                                          dtype = torch.half).unsqueeze(0)
        module.codebook = mul1_codebook(DEVICE)
        module._row_dtype = None
    else:
        module.mode = "fp16_ram"
        module.K = None
        module.tables = [torch.arange(7 * ROW_DIM, dtype = torch.float16).view(7, ROW_DIM)]
        module.head_bias = None
        module.codebook = None
        module._row_dtype = torch.float16
    return module


def test_rocm_ngram_bindings_match_torch_helpers():
    for name in ("ngram_hash_cpu", "ngram_gather_cpu", "ngram_dequant"):
        assert hasattr(native_ext, name), f"missing ROCm binding: {name}"

    module = _ngram_module(trellis = False)
    ids = torch.tensor([[99, 1, 2, 3]], dtype = torch.long)
    expected = module.compute_ngram_ids(ids, 3).reshape(-1)
    n = expected.numel()
    uids = torch.empty(n, dtype = torch.long)
    inverse = torch.empty(n, dtype = torch.long)
    heads = torch.empty(n, dtype = torch.int32)
    U = native_ext.ngram_hash_cpu(
        ids, 3, module.layer_multipliers, module.head_offsets, module.head_vocab_sizes,
        module.heads_per_ngram, module.eos_token_id, uids, inverse, heads,
    )
    assert torch.equal(uids[:U][inverse[:n]], expected)
    assert torch.equal(heads[:U], torch.zeros(U, dtype = torch.int32))

    K = 1
    packed = pack_rows(
        torch.arange(2 * ROW_DIM, device = DEVICE).view(2, ROW_DIM),
        torch.tensor([0.5, 1.0], device = DEVICE, dtype = torch.half),
        K,
    )
    bias = torch.linspace(-0.1, 0.1, ROW_DIM, device = DEVICE, dtype = torch.half).unsqueeze(0)
    row_heads = torch.zeros(2, device = DEVICE, dtype = torch.int32)
    out = torch.empty((2, ROW_DIM), device = DEVICE, dtype = torch.half)
    native_ext.ngram_dequant(packed, K, row_heads, bias, out)
    expected_rows = dequant_rows(packed, K, mul1_codebook(DEVICE), bias.expand(2, -1)).half()
    torch.testing.assert_close(out, expected_rows, rtol = 1e-3, atol = 1e-3)


def test_ngram_forward_falls_back_when_hash_or_dequant_is_missing(monkeypatch):
    ids = torch.tensor([[99, 1, 2]], dtype = torch.long)
    for module, missing_ext in (
        (_ngram_module(trellis = False), SimpleNamespace()),
        (_ngram_module(trellis = True), SimpleNamespace(ngram_hash_cpu = object())),
    ):
        expected = module.forward_reference(ids, {})
        monkeypatch.setattr(ne, "ext", missing_ext)
        actual = module.forward(ids, {})
        torch.testing.assert_close(actual, expected)
        monkeypatch.undo()


def test_ngram_disk_gather_has_read_rows_fallback(monkeypatch):
    table = torch.arange(8 * 3, dtype = torch.int16).view(8, 3)

    class Handle:
        def read_rows(self, rows):
            return table.index_select(0, rows)

    module = object.__new__(ne.NGramEmbedding)
    module.tables = None
    module.handles = [Handle()]
    module.rows_per_shard = 8
    uids = torch.tensor([1, 4, 7], dtype = torch.long)
    out = torch.empty((3, 3), dtype = torch.int16)
    monkeypatch.setattr(ne, "ext", SimpleNamespace())
    module._gather_rows(uids, out)
    assert torch.equal(out, table[uids])


def _run_compressor(kv, gate, chunks, overlapping):
    torch.manual_seed(13)
    m, hd = 2, 4
    ring_kv = torch.zeros((6, kv.shape[-1]), device = DEVICE, dtype = torch.half)
    ring_gate = torch.zeros_like(ring_kv)
    ovl = torch.zeros((4, 2, m, hd), device = DEVICE, dtype = torch.float) if overlapping else None
    ape = torch.randn((m, kv.shape[-1]), device = DEVICE, dtype = torch.float)
    norm_w = torch.linspace(0.8, 1.2, hd, device = DEVICE, dtype = torch.half)
    inv_freq = torch.tensor([0.2], device = DEVICE)
    dest_a = torch.zeros((4, 2), device = DEVICE, dtype = torch.half)
    dest_b = torch.zeros((4, 2), device = DEVICE, dtype = torch.half)
    pos = 0
    for size in chunks:
        dsv4._dsv4_compress(
            kv[pos:pos + size], gate[pos:pos + size], ring_kv, ring_gate, ovl,
            ape, norm_w, 1e-5, inv_freq, dest_a, dest_b, pos, None, m, None, None, 0,
        )
        pos += size
    return torch.cat((dest_a, dest_b), dim = -1), ring_kv, ring_gate


@pytest.mark.parametrize("overlapping", [False, True])
def test_dsv4_missing_compressor_binding_uses_chunk_complete_torch_path(monkeypatch, overlapping):
    torch.manual_seed(4)
    width = 8 if overlapping else 4
    kv = torch.randn((4, width), device = DEVICE, dtype = torch.half)
    gate = torch.randn_like(kv)
    monkeypatch.setattr(dsv4, "ext", SimpleNamespace())
    whole = _run_compressor(kv, gate, [4], overlapping)
    chunked = _run_compressor(kv, gate, [1, 3], overlapping)
    for actual, expected in zip(chunked, whole):
        torch.testing.assert_close(actual, expected, rtol = 2e-3, atol = 2e-3)


def test_dsv4_forward_fused_declines_missing_bc_and_compressor_bindings(monkeypatch):
    class Projection:
        def forward(self, x, params):
            return x

    module = object.__new__(dsv4.DSV4Compressor)
    module.fused_ready = False
    module.bc = None
    module.norm = SimpleNamespace(
        weight = SimpleNamespace(data = torch.ones(4, device = DEVICE, dtype = torch.half)),
        rms_norm_eps = 1e-5,
    )
    module.wkv = module.wgate = Projection()
    module.ape = torch.zeros((2, 4), device = DEVICE)
    module.fused_inv_freq = torch.tensor([0.2], device = DEVICE)
    module.fused_norm_w = None
    module.compress_rate = 2
    x = torch.randn((1, 4, 4), device = DEVICE, dtype = torch.half)
    ring_kv = torch.zeros((6, 4), device = DEVICE, dtype = torch.half)
    ring_gate = torch.zeros_like(ring_kv)
    dest_a = torch.zeros((2, 2), device = DEVICE, dtype = torch.half)
    dest_b = torch.zeros_like(dest_a)

    monkeypatch.setattr(dsv4, "ext", SimpleNamespace())
    module.forward_fused(x, {}, ring_kv, ring_gate, None, dest_a, dest_b, 0)
    assert module.fused_ready and module.bc is None
    assert torch.equal(ring_kv[:4], x[0])
    assert torch.isfinite(torch.cat((dest_a, dest_b), dim = -1)).all()


def test_dsv4_missing_ring_append_binding_updates_each_slot(monkeypatch):
    monkeypatch.setattr(dsv4, "ext", SimpleNamespace())
    kv = torch.arange(4 * 3, device = DEVICE, dtype = torch.half).view(4, 3)
    ring = torch.zeros((3, 5, 3), device = DEVICE, dtype = torch.half)
    pos = torch.tensor([2, 4], device = DEVICE, dtype = torch.int32)
    beg = torch.tensor([1, 3], device = DEVICE, dtype = torch.int32)
    slots = torch.tensor([2, 0], device = DEVICE, dtype = torch.int32)
    dsv4._dsv4_ring_append(kv, ring, pos, beg, slots)
    assert torch.equal(ring[2, 1:3], kv[:2])
    assert torch.equal(ring[0, 1:3], kv[2:])


class _Projection:
    def __init__(self, weight):
        self.quant_type = "fp16"
        self.inner = SimpleNamespace(weight = weight, bias = None)

    def forward(self, x, params):
        return F.linear(x, self.inner.weight)


class _IdentityNorm:
    def forward(self, x, params, out_dtype = None):
        return x.to(out_dtype) if out_dtype is not None else x


def test_ple_missing_bindings_selects_full_torch_reference(monkeypatch):
    torch.manual_seed(5)
    B, S, H, D, E = 1, 3, 2, 4, 5
    module = object.__new__(ple.PLELayer)
    emb = torch.randn((B, S, E), device = DEVICE, dtype = torch.half)
    module.ple_embedding = SimpleNamespace(forward = lambda *_args, **_kwargs: emb)
    module.key_proj = _Projection(torch.randn((H * D, E), device = DEVICE, dtype = torch.half))
    module.value_proj = _Projection(torch.randn((D, E), device = DEVICE, dtype = torch.half))
    module.norm_key = module.norm_query = module.norm_conv = _IdentityNorm()
    module.hc_mult = H
    module.hidden_size = D
    module.gate_scale = D ** -0.5
    module.conv_dilation = 2
    module.conv_state_len = 2
    module.conv_w = torch.zeros((H * D, 1, 2), device = DEVICE, dtype = torch.half)
    streams = torch.randn((B, S, H, D), device = DEVICE)

    key = F.linear(emb, module.key_proj.inner.weight).view(B, S, H, D).float()
    value = F.linear(emb, module.value_proj.inner.weight)
    gate = torch.bmm(streams.view(-1, 1, D), key.reshape(-1, D, 1)).view(B, S, H)
    expected = ple._ple_gate_torch(gate, value, module.gate_scale)

    monkeypatch.setattr(ple, "ext", SimpleNamespace())
    delta, conv_stream = module.forward_streams(streams, torch.zeros((B, S + 1), dtype = torch.long), {})
    torch.testing.assert_close(delta, expected, rtol = 2e-3, atol = 2e-3)
    assert conv_stream.shape == (B, H * D, module.conv_state_len + S)


def test_hyperconnection_missing_mix_and_apply_bindings_use_torch(monkeypatch):
    torch.manual_seed(6)
    B, S, H, D = 1, 2, 4, 8
    module = object.__new__(hc.HyperConnection)
    module.hc_mult = H
    module.hidden_size = D
    module.sinkhorn_iters = 2
    module.hc_eps = 1e-5
    module.rms_eps = 1e-5
    module.norm = _IdentityNorm()
    module.fn = torch.randn((2 * H + H * H, H * D), device = DEVICE)
    module.fn_h = None
    module.base = torch.randn((2 * H + H * H,), device = DEVICE)
    module.scale = torch.randn((3,), device = DEVICE)
    streams = torch.randn((B, S, H, D), device = DEVICE)

    monkeypatch.setattr(hc, "ext", SimpleNamespace())
    post, comb, collapsed = module.mix(streams, {})
    assert post.shape == (B, S, H)
    assert comb.shape == (B, S, H, H)
    assert collapsed.shape == (B, S, D)
    y = torch.randn((B, S, D), device = DEVICE, dtype = torch.half)
    expected = post.unsqueeze(-1) * y.float().unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), streams)
    torch.testing.assert_close(module.apply_(streams, y, post, comb, {}), expected)


def test_gated_residual_missing_gr_mix_uses_reference_at_decode_and_prefill(monkeypatch):
    torch.manual_seed(7)
    H, D, rank = 4, 8, 3
    module = object.__new__(hc.GatedResidual)
    module.hc_mult = H
    module.hidden_size = D
    module.rms_eps = 1e-5
    module.use_combine = True
    module.norm_w = torch.randn((H, D), device = DEVICE)
    module.down_h = torch.randn((rank, H * D), device = DEVICE, dtype = torch.half)
    module.up_h = torch.randn((H * D, rank), device = DEVICE, dtype = torch.half)
    module.inject_h = torch.randn((H, H * D), device = DEVICE, dtype = torch.half)
    monkeypatch.setattr(hc, "ext", SimpleNamespace())

    for rows in (1, module.FUSED_MAX_R + 1):
        streams = torch.randn((1, rows, H, D), device = DEVICE)
        expected_post, expected_mixed = module._mix_ref(streams)
        post, mixed = module._mix(streams)
        torch.testing.assert_close(post.view_as(expected_post), expected_post)
        torch.testing.assert_close(mixed.view_as(expected_mixed), expected_mixed.half())


def test_hyperhead_and_bighead_decline_missing_fast_bindings(monkeypatch):
    torch.manual_seed(8)
    B, S, H, D = 1, 2, 4, 8
    head = object.__new__(hc.HyperHead)
    head.mean = False
    head.norm = _IdentityNorm()
    head.fn = torch.randn((H, H * D), device = DEVICE)
    head.fn_h = None
    head.base = torch.randn((H,), device = DEVICE)
    head.scale = torch.randn((1,), device = DEVICE)
    head.rms_eps = 1e-5
    head.hc_eps = 1e-5
    x = torch.randn((B, S, H, D), device = DEVICE)
    expected_mix = F.linear(x.flatten(2), head.fn)
    expected = ((torch.sigmoid(expected_mix * head.scale + head.base) + head.hc_eps)
                .unsqueeze(-1) * x).sum(dim = 2)
    monkeypatch.setattr(hc, "ext", SimpleNamespace())
    torch.testing.assert_close(head.forward(x, {}), expected)

    q = torch.randn((1, 1, 1, 512), device = DEVICE, dtype = torch.half)
    cache = torch.zeros((1, 16, 1, 512), device = DEVICE, dtype = torch.half)
    args = AttnArgs(
        1, 1, 1, 512, 1, 1, q, q, q, cache, cache, True, 512 ** -0.5,
        None, None, None, 0.0,
        torch.zeros((1, 1), device = DEVICE, dtype = torch.int32),
        torch.zeros((1,), device = DEVICE, dtype = torch.int32),
    )
    monkeypatch.setattr(bighead_scalar, "ext", SimpleNamespace())
    assert bighead_scalar.fn_bighead_scalar_attn(args) is None
    baseline = torch_attn.fn_torch_sdpa_fallback_cache(args)
    assert baseline.shape == q.shape and torch.isfinite(baseline).all()
