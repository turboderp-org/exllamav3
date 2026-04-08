import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.constants import PAGE_SIZE
from exllamav3.ext import exllamav3_ext as ext


def _maybe_skip_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Gemma4 KV ext tests")


def _manual_gather_from_paged_cache(cache_tensor, total_lens, block_table):
    bsz = block_table.shape[0]
    max_total = int(total_lens.max().item())
    gathered = torch.zeros(
        (bsz, max_total, cache_tensor.shape[2], cache_tensor.shape[3]),
        dtype = cache_tensor.dtype,
        device = cache_tensor.device,
    )
    for batch_idx in range(bsz):
        total = int(total_lens[batch_idx].item())
        if total == 0:
            continue
        positions = torch.arange(total, device = cache_tensor.device, dtype = torch.long)
        pages = block_table[batch_idx].gather(0, torch.div(positions, PAGE_SIZE, rounding_mode = "floor"))
        page_pos = positions.remainder(PAGE_SIZE)
        gathered[batch_idx, :total] = cache_tensor[pages, page_pos]
    return gathered


def _manual_dequant_token_slots(qcache, scales, pages, page_pos, num_kv_heads, head_dim):
    q_tokens = qcache[pages, page_pos]
    s_tokens = scales[pages, page_pos]
    out = torch.empty((q_tokens.shape[0], num_kv_heads * head_dim), dtype = torch.half, device = qcache.device)
    ext.dequant_cache_cont(q_tokens, s_tokens, out)
    return out.view(q_tokens.shape[0], num_kv_heads, head_dim)


@torch.inference_mode()
def test_quant_cache_paged_delta_matches_quant_cache_paged():
    _maybe_skip_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda")

    bsz = 2
    pages = 8
    num_kv_heads = 2
    head_dim = 64
    bits = 4
    seq_len = 33

    block_table = torch.tensor([[7, 1, 0, 2], [6, 4, 5, 3]], dtype = torch.int, device = device)
    cache_seqlens = torch.tensor([5, 260], dtype = torch.int, device = device)
    k_delta = torch.randn((bsz, seq_len, num_kv_heads, head_dim), dtype = torch.half, device = device)
    v_delta = torch.randn_like(k_delta)

    qshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32 * bits)
    sshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32)

    qk_ref = torch.zeros(qshape, dtype = torch.int, device = device)
    qv_ref = torch.zeros_like(qk_ref)
    sk_ref = torch.zeros(sshape, dtype = torch.half, device = device)
    sv_ref = torch.zeros_like(sk_ref)

    qk_delta = torch.zeros_like(qk_ref)
    qv_delta = torch.zeros_like(qk_ref)
    sk_delta = torch.zeros_like(sk_ref)
    sv_delta = torch.zeros_like(sk_ref)

    staged_k = torch.zeros((pages, PAGE_SIZE, num_kv_heads, head_dim), dtype = torch.half, device = device)
    staged_v = torch.zeros_like(staged_k)
    positions = (
        cache_seqlens.to(dtype = torch.long).unsqueeze(1) +
        torch.arange(seq_len, dtype = torch.long, device = device).unsqueeze(0)
    )
    target_pages = block_table.gather(1, torch.div(positions, PAGE_SIZE, rounding_mode = "floor")).reshape(-1)
    target_pos = positions.remainder(PAGE_SIZE).reshape(-1)
    staged_k[target_pages, target_pos] = k_delta.reshape(-1, num_kv_heads, head_dim)
    staged_v[target_pages, target_pos] = v_delta.reshape(-1, num_kv_heads, head_dim)

    full_block_table = torch.arange(pages, dtype = torch.int, device = device).view(1, pages)
    full_cache_start = torch.zeros((1,), dtype = torch.int, device = device)
    full_seq_len = pages * PAGE_SIZE
    ext.quant_cache_paged(
        staged_k, qk_ref, sk_ref,
        staged_v, qv_ref, sv_ref,
        full_cache_start, full_block_table,
        PAGE_SIZE, full_seq_len,
    )
    ext.quant_cache_paged_delta(
        k_delta, qk_delta, sk_delta,
        v_delta, qv_delta, sv_delta,
        cache_seqlens, block_table,
        PAGE_SIZE, seq_len,
    )

    ref_k_tokens = _manual_dequant_token_slots(qk_ref, sk_ref, target_pages, target_pos, num_kv_heads, head_dim)
    ref_v_tokens = _manual_dequant_token_slots(qv_ref, sv_ref, target_pages, target_pos, num_kv_heads, head_dim)
    delta_k_tokens = _manual_dequant_token_slots(qk_delta, sk_delta, target_pages, target_pos, num_kv_heads, head_dim)
    delta_v_tokens = _manual_dequant_token_slots(qv_delta, sv_delta, target_pages, target_pos, num_kv_heads, head_dim)

    torch.testing.assert_close(delta_k_tokens, ref_k_tokens, atol = 0.08, rtol = 0.01)
    torch.testing.assert_close(delta_v_tokens, ref_v_tokens, atol = 0.08, rtol = 0.01)


@torch.inference_mode()
def test_dequant_cache_paged_gather_matches_reference_gather():
    _maybe_skip_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda")

    bsz = 2
    pages = 8
    num_kv_heads = 2
    head_dim = 64
    bits = 4
    total_lens = torch.tensor([270, 37], dtype = torch.int, device = device)
    max_total = int(total_lens.max().item())

    block_table = torch.tensor([[7, 1, 0, 2], [6, 4, 5, 3]], dtype = torch.int, device = device)
    cache_start = torch.zeros((bsz,), dtype = torch.int, device = device)
    k_full = torch.randn((bsz, max_total, num_kv_heads, head_dim), dtype = torch.half, device = device)
    v_full = torch.randn_like(k_full)

    qshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32 * bits)
    sshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32)
    qk = torch.zeros(qshape, dtype = torch.int, device = device)
    qv = torch.zeros_like(qk)
    sk = torch.zeros(sshape, dtype = torch.half, device = device)
    sv = torch.zeros_like(sk)

    ext.quant_cache_paged(
        k_full, qk, sk,
        v_full, qv, sv,
        cache_start, block_table,
        PAGE_SIZE, max_total,
    )

    gathered_k = torch.zeros((bsz, max_total, num_kv_heads, head_dim), dtype = torch.half, device = device)
    gathered_v = torch.zeros_like(gathered_k)
    ext.dequant_cache_paged_gather(
        qk, sk, gathered_k,
        qv, sv, gathered_v,
        total_lens, block_table,
        PAGE_SIZE, max_total,
    )

    for batch_idx in range(bsz):
        total = int(total_lens[batch_idx].item())
        positions = torch.arange(total, device = device, dtype = torch.long)
        pages_for_batch = block_table[batch_idx].gather(0, torch.div(positions, PAGE_SIZE, rounding_mode = "floor"))
        pos_for_batch = positions.remainder(PAGE_SIZE)
        ref_k = _manual_dequant_token_slots(qk, sk, pages_for_batch, pos_for_batch, num_kv_heads, head_dim)
        ref_v = _manual_dequant_token_slots(qv, sv, pages_for_batch, pos_for_batch, num_kv_heads, head_dim)
        torch.testing.assert_close(gathered_k[batch_idx, :total], ref_k, atol = 0.08, rtol = 0.01)
        torch.testing.assert_close(gathered_v[batch_idx, :total], ref_v, atol = 0.08, rtol = 0.01)


@torch.inference_mode()
def test_dequant_cache_paged_gather_delta_matches_reference_append():
    _maybe_skip_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda")

    bsz = 2
    pages = 8
    num_kv_heads = 2
    head_dim = 64
    bits = 4
    cache_seqlens = torch.tensor([270, 37], dtype = torch.int, device = device)
    delta_len = 5
    total_lens = cache_seqlens + delta_len
    max_total = int(total_lens.max().item())

    block_table = torch.tensor([[7, 1, 0, 2], [6, 4, 5, 3]], dtype = torch.int, device = device)
    cache_start = torch.zeros((bsz,), dtype = torch.int, device = device)
    k_full = torch.randn((bsz, max_total, num_kv_heads, head_dim), dtype = torch.half, device = device)
    v_full = torch.randn_like(k_full)
    k_delta = torch.randn((bsz, delta_len, num_kv_heads, head_dim), dtype = torch.half, device = device)
    v_delta = torch.randn_like(k_delta)

    qshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32 * bits)
    sshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32)
    qk = torch.zeros(qshape, dtype = torch.int, device = device)
    qv = torch.zeros_like(qk)
    sk = torch.zeros(sshape, dtype = torch.half, device = device)
    sv = torch.zeros_like(sk)

    ext.quant_cache_paged(
        k_full, qk, sk,
        v_full, qv, sv,
        cache_start, block_table,
        PAGE_SIZE, max_total,
    )

    gathered_k = torch.zeros((bsz, max_total, num_kv_heads, head_dim), dtype = torch.half, device = device)
    gathered_v = torch.zeros_like(gathered_k)
    ext.dequant_cache_paged_gather_delta(
        qk, sk, k_delta, gathered_k,
        qv, sv, v_delta, gathered_v,
        cache_seqlens, block_table,
        PAGE_SIZE, max_total, delta_len,
    )

    for batch_idx in range(bsz):
        cache_total = int(cache_seqlens[batch_idx].item())
        total = int(total_lens[batch_idx].item())
        positions = torch.arange(cache_total, device = device, dtype = torch.long)
        pages_for_batch = block_table[batch_idx].gather(0, torch.div(positions, PAGE_SIZE, rounding_mode = "floor"))
        pos_for_batch = positions.remainder(PAGE_SIZE)
        ref_k = _manual_dequant_token_slots(qk, sk, pages_for_batch, pos_for_batch, num_kv_heads, head_dim)
        ref_v = _manual_dequant_token_slots(qv, sv, pages_for_batch, pos_for_batch, num_kv_heads, head_dim)
        expected_k = torch.zeros((total, num_kv_heads, head_dim), dtype = torch.half, device = device)
        expected_v = torch.zeros_like(expected_k)
        expected_k[:cache_total] = ref_k
        expected_v[:cache_total] = ref_v
        expected_k[cache_total:total] = k_delta[batch_idx]
        expected_v[cache_total:total] = v_delta[batch_idx]
        torch.testing.assert_close(gathered_k[batch_idx, :total], expected_k, atol = 0.08, rtol = 0.01)
        torch.testing.assert_close(gathered_v[batch_idx, :total], expected_v, atol = 0.08, rtol = 0.01)


@torch.inference_mode()
def test_dequant_cache_paged_gather_heads_matches_reference_gather():
    _maybe_skip_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda")

    bsz = 2
    pages = 8
    num_kv_heads = 2
    head_dim = 64
    bits = 4
    total_lens = torch.tensor([270, 37], dtype = torch.int, device = device)
    max_total = int(total_lens.max().item())

    block_table = torch.tensor([[7, 1, 0, 2], [6, 4, 5, 3]], dtype = torch.int, device = device)
    cache_start = torch.zeros((bsz,), dtype = torch.int, device = device)
    k_full = torch.randn((bsz, max_total, num_kv_heads, head_dim), dtype = torch.half, device = device)
    v_full = torch.randn_like(k_full)

    qshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32 * bits)
    sshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32)
    qk = torch.zeros(qshape, dtype = torch.int, device = device)
    qv = torch.zeros_like(qk)
    sk = torch.zeros(sshape, dtype = torch.half, device = device)
    sv = torch.zeros_like(sk)

    ext.quant_cache_paged(
        k_full, qk, sk,
        v_full, qv, sv,
        cache_start, block_table,
        PAGE_SIZE, max_total,
    )

    gathered_k = torch.zeros((bsz, num_kv_heads, max_total, head_dim), dtype = torch.half, device = device)
    gathered_v = torch.zeros_like(gathered_k)
    ext.dequant_cache_paged_gather_heads(
        qk, sk, gathered_k,
        qv, sv, gathered_v,
        total_lens, block_table,
        PAGE_SIZE, max_total,
    )

    for batch_idx in range(bsz):
        total = int(total_lens[batch_idx].item())
        positions = torch.arange(total, device = device, dtype = torch.long)
        pages_for_batch = block_table[batch_idx].gather(0, torch.div(positions, PAGE_SIZE, rounding_mode = "floor"))
        pos_for_batch = positions.remainder(PAGE_SIZE)
        ref_k = _manual_dequant_token_slots(qk, sk, pages_for_batch, pos_for_batch, num_kv_heads, head_dim)
        ref_v = _manual_dequant_token_slots(qv, sv, pages_for_batch, pos_for_batch, num_kv_heads, head_dim)
        torch.testing.assert_close(gathered_k[batch_idx, :, :total].transpose(0, 1), ref_k, atol = 0.08, rtol = 0.01)
        torch.testing.assert_close(gathered_v[batch_idx, :, :total].transpose(0, 1), ref_v, atol = 0.08, rtol = 0.01)


@torch.inference_mode()
def test_dequant_cache_paged_gather_delta_heads_matches_reference_append():
    _maybe_skip_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda")

    bsz = 2
    pages = 8
    num_kv_heads = 2
    head_dim = 64
    bits = 4
    cache_seqlens = torch.tensor([270, 37], dtype = torch.int, device = device)
    delta_len = 5
    total_lens = cache_seqlens + delta_len
    max_total = int(total_lens.max().item())

    block_table = torch.tensor([[7, 1, 0, 2], [6, 4, 5, 3]], dtype = torch.int, device = device)
    cache_start = torch.zeros((bsz,), dtype = torch.int, device = device)
    k_full = torch.randn((bsz, max_total, num_kv_heads, head_dim), dtype = torch.half, device = device)
    v_full = torch.randn_like(k_full)
    k_delta = torch.randn((bsz, delta_len, num_kv_heads, head_dim), dtype = torch.half, device = device)
    v_delta = torch.randn_like(k_delta)

    qshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32 * bits)
    sshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32)
    qk = torch.zeros(qshape, dtype = torch.int, device = device)
    qv = torch.zeros_like(qk)
    sk = torch.zeros(sshape, dtype = torch.half, device = device)
    sv = torch.zeros_like(sk)

    ext.quant_cache_paged(
        k_full, qk, sk,
        v_full, qv, sv,
        cache_start, block_table,
        PAGE_SIZE, max_total,
    )

    gathered_k = torch.zeros((bsz, num_kv_heads, max_total, head_dim), dtype = torch.half, device = device)
    gathered_v = torch.zeros_like(gathered_k)
    ext.dequant_cache_paged_gather_delta_heads(
        qk, sk, k_delta, gathered_k,
        qv, sv, v_delta, gathered_v,
        cache_seqlens, block_table,
        PAGE_SIZE, max_total, delta_len,
    )

    for batch_idx in range(bsz):
        cache_total = int(cache_seqlens[batch_idx].item())
        total = int(total_lens[batch_idx].item())
        positions = torch.arange(cache_total, device = device, dtype = torch.long)
        pages_for_batch = block_table[batch_idx].gather(0, torch.div(positions, PAGE_SIZE, rounding_mode = "floor"))
        pos_for_batch = positions.remainder(PAGE_SIZE)
        ref_k = _manual_dequant_token_slots(qk, sk, pages_for_batch, pos_for_batch, num_kv_heads, head_dim)
        ref_v = _manual_dequant_token_slots(qv, sv, pages_for_batch, pos_for_batch, num_kv_heads, head_dim)
        expected_k = torch.zeros((num_kv_heads, total, head_dim), dtype = torch.half, device = device)
        expected_v = torch.zeros_like(expected_k)
        expected_k[:, :cache_total] = ref_k.transpose(0, 1)
        expected_v[:, :cache_total] = ref_v.transpose(0, 1)
        expected_k[:, cache_total:total] = k_delta[batch_idx].transpose(0, 1)
        expected_v[:, cache_total:total] = v_delta[batch_idx].transpose(0, 1)
        torch.testing.assert_close(gathered_k[batch_idx, :, :total], expected_k, atol = 0.08, rtol = 0.01)
        torch.testing.assert_close(gathered_v[batch_idx, :, :total], expected_v, atol = 0.08, rtol = 0.01)


@torch.inference_mode()
def test_dequant_cache_paged_select_delta_heads_matches_selected_reference():
    _maybe_skip_cuda()
    torch.manual_seed(0)
    device = torch.device("cuda")

    bsz = 2
    pages = 8
    num_kv_heads = 2
    head_dim = 64
    bits = 4
    cache_seqlens = torch.tensor([270, 37], dtype = torch.int, device = device)
    delta_len = 5
    total_lens = cache_seqlens + delta_len
    selected_positions = torch.tensor([[1, 269, 270, 274], [0, 10, 39, 41]], dtype = torch.int, device = device)
    selected_counts = torch.tensor([4, 3], dtype = torch.int, device = device)
    max_selected = selected_positions.size(1)
    max_total = int(total_lens.max().item())

    block_table = torch.tensor([[7, 1, 0, 2], [6, 4, 5, 3]], dtype = torch.int, device = device)
    cache_start = torch.zeros((bsz,), dtype = torch.int, device = device)
    k_full = torch.randn((bsz, max_total, num_kv_heads, head_dim), dtype = torch.half, device = device)
    v_full = torch.randn_like(k_full)
    k_delta = torch.randn((bsz, delta_len, num_kv_heads, head_dim), dtype = torch.half, device = device)
    v_delta = torch.randn_like(k_delta)

    qshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32 * bits)
    sshape = (pages, PAGE_SIZE, num_kv_heads * head_dim // 32)
    qk = torch.zeros(qshape, dtype = torch.int, device = device)
    qv = torch.zeros_like(qk)
    sk = torch.zeros(sshape, dtype = torch.half, device = device)
    sv = torch.zeros_like(sk)

    ext.quant_cache_paged(
        k_full, qk, sk,
        v_full, qv, sv,
        cache_start, block_table,
        PAGE_SIZE, max_total,
    )

    gathered_k = torch.zeros((bsz, num_kv_heads, max_selected, head_dim), dtype = torch.half, device = device)
    gathered_v = torch.zeros_like(gathered_k)
    ext.dequant_cache_paged_select_delta_heads(
        qk, sk, k_delta, gathered_k,
        qv, sv, v_delta, gathered_v,
        cache_seqlens, block_table,
        selected_positions, selected_counts,
        PAGE_SIZE, max_selected, delta_len,
    )

    for batch_idx in range(bsz):
        total = int(selected_counts[batch_idx].item())
        expected_k = torch.zeros((num_kv_heads, total, head_dim), dtype = torch.half, device = device)
        expected_v = torch.zeros_like(expected_k)
        cache_total = int(cache_seqlens[batch_idx].item())
        for i, pos in enumerate(selected_positions[batch_idx, :total].tolist()):
            if pos < cache_total:
                page = int(block_table[batch_idx, pos // PAGE_SIZE].item())
                page_pos = pos % PAGE_SIZE
                token_k = _manual_dequant_token_slots(
                    qk,
                    sk,
                    torch.tensor([page], dtype = torch.long, device = device),
                    torch.tensor([page_pos], dtype = torch.long, device = device),
                    num_kv_heads,
                    head_dim,
                )[0]
                token_v = _manual_dequant_token_slots(
                    qv,
                    sv,
                    torch.tensor([page], dtype = torch.long, device = device),
                    torch.tensor([page_pos], dtype = torch.long, device = device),
                    num_kv_heads,
                    head_dim,
                )[0]
                expected_k[:, i] = token_k
                expected_v[:, i] = token_v
            else:
                expected_k[:, i] = k_delta[batch_idx, pos - cache_total]
                expected_v[:, i] = v_delta[batch_idx, pos - cache_total]

        torch.testing.assert_close(gathered_k[batch_idx, :, :total], expected_k, atol = 0.08, rtol = 0.01)
        torch.testing.assert_close(gathered_v[batch_idx, :, :total], expected_v, atol = 0.08, rtol = 0.01)
