"""Olala BF16 MoE inference on gfx1100. Expert weights stay in EXL3 mul1 layout."""

import torch
import triton
import triton.language as tl

from .olala_moe import _project_inputs, _project_outputs


def codebook(device):
    # EXL3 mul1 first rounds the affine transform of the four-byte sum to half.
    sums = torch.arange(1021, dtype = torch.float32, device = device)
    return ((1024 + sums) * 0.00676727294921875 - 10.3828125).half().bfloat16()


@triton.jit
def _lookup(table, state):
    hashed = state * 0x83dcd12d
    index = (hashed & 255) + ((hashed >> 8) & 255) + ((hashed >> 16) & 255) + (hashed >> 24)
    return tl.load(table + index)


@triton.jit
def _decode(packed, table, k, n, N: tl.constexpr, BITS: tl.constexpr):
    # Map logical matrix coordinates back to the native EXL3 16x16 trellis tile.
    row = k % 16
    col = n % 16
    lane = (col % 8 // 2) * 8 + (col % 2) * 4 + row % 8 // 2
    position = lane * 8 + row % 2 + (row // 8) * 2 + (col // 8) * 4
    end = (position + 1) * BITS
    word = (end - 1) // 32
    shift = (-end) & 31
    base = packed + (k // 16 * (N // 16) + n // 16) * (8 * BITS)
    lo = tl.load(base + word)
    hi = tl.load(base + (word + 8 * BITS - 1) % (8 * BITS))
    state = ((lo >> shift) | tl.where(shift == 0, 0, hi << ((32 - shift) & 31))) & 65535
    return _lookup(table, state)


@triton.jit
def _row_gemv(X, Packed, Table, Ids, Y, K: tl.constexpr, N: tl.constexpr, BITS: tl.constexpr,
              KT: tl.constexpr):
    slot = tl.program_id(1)
    packed = tl.load(Packed + tl.load(Ids + slot)).to(tl.pointer_type(tl.uint32))
    tile = tl.program_id(0)
    # Consume each 16x16 tile in its packed lane order, then reduce the K
    # contributions for each column. Each lane decodes eight weights.
    tk = tl.arange(0, KT)
    pos = tl.arange(0, 256)
    base = packed + (tk[:, None] * (N // 16) + tile) * (8 * BITS)
    end = (pos + 1) * BITS
    word = (end - 1) // 32
    shift = (-end) & 31
    lo = tl.load(base + word[None, :], tk[:, None] < K // 16, 0)
    hi = tl.load(base + ((word + 8 * BITS - 1) % (8 * BITS))[None, :], tk[:, None] < K // 16, 0)
    state = ((lo >> shift) | tl.where(shift == 0, 0, hi << ((32 - shift) & 31))) & 65535
    w = _lookup(Table, state).to(tl.float32)
    k = tk[:, None] * 16 + ((pos // 8 % 4) * 2 + (pos % 2) + (pos // 2 % 2) * 8)[None, :]
    x = tl.load(X + slot * K + k, k < K, 0).to(tl.float32)
    partial = tl.sum(x * w, 0)
    partial = tl.permute(tl.reshape(partial, (8, 4, 2, 4)), (2, 0, 1, 3))
    col = tile * 16 + tl.arange(0, 16)
    tl.store(Y + slot * N + col, tl.sum(tl.reshape(partial, (16, 16)), 1))


@triton.jit
def _group_gemm(X, Packed, Table, Order, Ends, Tiles, Y, K: tl.constexpr, N: tl.constexpr,
                BITS: tl.constexpr, EXPERTS: tl.constexpr, M: tl.constexpr):
    tile = tl.program_id(1)
    if tile < tl.load(Tiles + EXPERTS - 1):
        expert = tl.sum((tl.load(Tiles + tl.arange(0, EXPERTS)) <= tile).to(tl.int32), 0)
        packed = tl.load(Packed + expert).to(tl.pointer_type(tl.uint32))
        first = tl.load(Ends + expert - 1, expert > 0, 0)
        last = tl.load(Ends + expert)
        block = tile - tl.load(Tiles + expert - 1, expert > 0, 0)
        row = first + block * M + tl.arange(0, M)
        slot = tl.load(Order + row, row < last, 0).to(tl.int32)
        col = tl.program_id(0) * 32 + tl.arange(0, 32)
        acc = tl.full((M, 32), 0, tl.float32)
        for j in range(K // 32):
            k = j * 32 + tl.arange(0, 32)
            x = tl.load(X + slot[:, None] * K + k[None, :], row[:, None] < last, 0)
            w = _decode(packed, Table, k[:, None], col[None, :], N, BITS)
            acc = tl.dot(x, w, acc)
        tl.store(Y + slot[:, None] * N + col[None, :], acc, row[:, None] < last)


@triton.jit
def _had(x, i):
    for s in tl.static_range(7):
        other = tl.gather(x, i ^ (1 << s), 0)
        x = tl.where((i & (1 << s)) != 0, other - x, x + other)
    return x * 0.08838834764831845


@triton.jit
def _rotate(X, Ids, Scales, NextScales, Y, WIDTH: tl.constexpr, TOP_K: tl.constexpr,
            INPUT: tl.constexpr):
    slot = tl.program_id(1)
    i = tl.arange(0, 128)
    col = tl.program_id(0) * 128 + i
    expert = tl.load(Ids + slot)
    scale = tl.load(tl.load(Scales + expert).to(tl.pointer_type(tl.float16)) + col)
    row = slot // TOP_K if INPUT else slot
    x = tl.load(X + row * WIDTH + col).to(tl.float32)
    if INPUT:
        x = _had(x * scale, i)
    else:
        x = _had(x, i) * scale
        x = tl.maximum(x, 0.)
        down = tl.load(tl.load(NextScales + expert).to(tl.pointer_type(tl.float16)) + col)
        x = _had(x * x * down, i)
    tl.store(Y + slot * WIDTH + col, x)


@triton.jit
def _finish(X, Ids, Scales, Weights, Y, WIDTH: tl.constexpr, TOP_K: tl.constexpr):
    row = tl.program_id(1)
    i = tl.arange(0, 128)
    col = tl.program_id(0) * 128 + i
    slots: tl.constexpr = triton.next_power_of_2(TOP_K)
    slot = tl.arange(0, slots)
    assignment = row * TOP_K + slot
    x = tl.load(X + assignment[:, None] * WIDTH + col[None, :], slot[:, None] < TOP_K, 0)
    for bit in tl.static_range(7):
        other = tl.gather(x, tl.broadcast_to((i ^ (1 << bit))[None, :], (slots, 128)), 1)
        x = tl.where((i & (1 << bit)) != 0, other - x, x + other)
    expert = tl.load(Ids + assignment, slot < TOP_K, 0)
    scale = tl.load(Scales + expert).to(tl.pointer_type(tl.float16))
    w = tl.load(Weights + assignment, slot < TOP_K, 0)
    x = x * 0.08838834764831845 * tl.load(scale[:, None] + col[None, :])
    tl.store(Y + row * WIDTH + col, tl.sum(x * w[:, None], 0))


@triton.jit
def _expert_tiles(Sorted, Ends, Tiles, SLOTS, step, EXPERTS: tl.constexpr, M: tl.constexpr):
    # Upper bounds give each expert's row range, including empty experts.
    i = tl.arange(0, EXPERTS)
    end = tl.full((EXPERTS,), 0, tl.int32)
    while step > 0:
        probe = end + step
        expert = tl.load(Sorted + probe - 1, probe <= SLOTS, EXPERTS)
        end = tl.where(expert <= i, probe, end)
        step = step // 2
    prev = tl.gather(end, tl.maximum(i - 1, 0), 0)
    count = end - tl.where(i > 0, prev, 0)
    tl.store(Ends + i, end)
    tl.store(Tiles + i, tl.cumsum((count + M - 1) // M))


def routed_experts(moe, latent, ids, weights):
    rows, top_k = ids.shape
    slots = rows * top_k
    up, down = moe.multi_up, moe.multi_down
    width, intermediate = up.in_features, up.out_features
    rotated = latent.new_empty((slots, width))
    _rotate[(width // 128, slots)](latent, ids, up.ptrs_suh, None, rotated, width, top_k, True)
    if rows > 1:
        # Reuse the token-to-expert assignment for both projections. Each expert
        # owns a contiguous range of tiles, including its partially filled last tile.
        sorted_ids, order = ids.flatten().sort()
        m_tile = 32
        ends = ids.new_empty(moe.num_experts, dtype = torch.int32)
        tiles = torch.empty_like(ends)
        _expert_tiles[(1,)](sorted_ids, ends, tiles, slots, triton.next_power_of_2(slots), moe.num_experts, m_tile)

    for linear in (up, down):
        k, n = linear.in_features, linear.out_features
        projected = latent.new_empty((slots, n), dtype = torch.float32)
        if rows == 1:
            _row_gemv[(n // 16, slots)](rotated, linear.ptrs_trellis, moe.bf16_codebook,
                                        ids, projected, k, n, int(linear.K), triton.next_power_of_2(k // 16))
        else:
            grid = (n // 32, moe.num_experts + triton.cdiv(slots, m_tile))
            _group_gemm[grid](rotated, linear.ptrs_trellis, moe.bf16_codebook, order, ends, tiles,
                              projected, k, n, int(linear.K), moe.num_experts, m_tile)
        if linear is up:
            rotated = latent.new_empty((slots, n))
            _rotate[(n // 128, slots)](projected, ids, up.ptrs_svh, down.ptrs_suh, rotated,
                                       n, top_k, False)
    routed = latent.new_empty((rows, width))
    _finish[(width // 128, rows)](projected, ids, down.ptrs_svh, weights, routed, width, top_k)
    return routed


@triton.jit
def _project(X, W, Y, Shared, M, N: tl.constexpr, K: tl.constexpr,
             LATENT: tl.constexpr, UP: tl.constexpr):
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    col = tl.program_id(1) * 64 + tl.arange(0, 64)
    acc = tl.full((32, 64), 0, tl.float32)
    for j in range(K // 32):
        k = j * 32 + tl.arange(0, 32)
        if UP:
            x = tl.load(X + row[:, None] * K + k[None, :], row[:, None] < M, 0)
        else:
            if j < LATENT // 32:
                x = tl.load(X + row[:, None] * LATENT + k[None, :], row[:, None] < M, 0)
            else:
                x = tl.load(Shared + row[:, None] * (K - LATENT) + k[None, :] - LATENT, row[:, None] < M, 0)
        w = tl.load(W + col[None, :] * K + k[:, None])
        acc = tl.dot(x.to(tl.bfloat16), w, acc)
    if UP:
        if tl.program_id(1) < LATENT // 64:
            tl.store(Y + row[:, None] * LATENT + col[None, :], acc, row[:, None] < M)
        else:
            positive = tl.maximum(acc.to(tl.bfloat16).to(tl.float32), 0.)
            tl.store(Shared + row[:, None] * (N - LATENT) + col[None, :] - LATENT, positive * positive, row[:, None] < M)
    else:
        tl.store(Y + row[:, None] * N + col[None, :], acc, row[:, None] < M)


def forward(moe, x, params):
    shape = x.shape
    x = x.view(-1, shape[-1])
    rows, hidden = x.shape
    ids, weights = moe.routing_fn(rows, moe.routing_cfg, x, params)
    up, down = moe.decode_weights
    width, shared_width = moe.config.latent_size, moe.config.shared_size
    latent = x.new_empty((rows, width), dtype = torch.bfloat16)
    shared = latent.new_empty((rows, shared_width))
    if rows == 1:
        _project_inputs[((width + shared_width) // 4,)](
            x, up, latent, shared, hidden, width, 4, enable_fp_fusion = False)
    else:
        _project[(triton.cdiv(rows, 32), (width + shared_width) // 64)](
            x, up, latent, shared, rows, width + shared_width, hidden, width, True)
    routed = routed_experts(moe, latent, ids, weights)
    out = x.new_empty((rows, hidden), dtype = torch.float32)
    if rows == 1:
        _project_outputs[(hidden // 4,)](routed, shared, down, None, out, width, shared_width, 4, enable_fp_fusion = False)
    else:
        _project[(triton.cdiv(rows, 32), hidden // 64)](
            routed, down, out, shared, rows, hidden, width + shared_width, width, False)
    return out.view(shape)
