import torch
import torch.nn.functional as F
import math
from ....ext import exllamav3_ext as ext
from ....util.progress import ProgressBar
from ....util.memory import free_mem
from ....util.hadamard import get_hadamard_dt
from ....util import cuda_sync_active
from functools import lru_cache

# Constant
had_k, had_n = 128, 128

@lru_cache
def tensor_core_perm(device):
    perm_a = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm_a[t * 8 + 0] = r0 * 16 + c0
        perm_a[t * 8 + 1] = r1 * 16 + c0
        perm_a[t * 8 + 2] = r2 * 16 + c0
        perm_a[t * 8 + 3] = r3 * 16 + c0
        perm_a[t * 8 + 4] = r0 * 16 + c1
        perm_a[t * 8 + 5] = r1 * 16 + c1
        perm_a[t * 8 + 6] = r2 * 16 + c1
        perm_a[t * 8 + 7] = r3 * 16 + c1
    return torch.tensor(perm_a, dtype = torch.int, device = device)


@lru_cache
def tensor_core_perm_i(device):
    return torch.argsort(tensor_core_perm(device))


@lru_cache
def get_temp_buffers(device, K: int):
    max_batch_size = 128
    temp_costs = torch.zeros((max_batch_size, 2, 65536 >> K), dtype = torch.float, device = device)
    temp_edges = torch.zeros((max_batch_size, 256, 65536 >> K), dtype = torch.short, device = device)
    return temp_costs, temp_edges


def quantize_tiles(tiles, quant_args: dict):
    tiles = tiles.contiguous()
    assert tiles.shape[1] == 256
    assert tiles.dtype == torch.float

    K = quant_args["K"]
    quantized_tiles = torch.zeros_like(tiles)
    quantized_idx = torch.zeros_like(tiles, dtype = torch.short)
    temp_costs, temp_edges = get_temp_buffers(tiles.device, K)
    ext.quantize_tiles(
        tiles,
        quantized_tiles,
        quantized_idx,
        temp_costs,
        temp_edges,
        K
    )
    return quantized_tiles, quantized_idx


@lru_cache
def get_quant_stream(device):
    return torch.cuda.Stream(device = device)


pinned_tiles: torch.Tensor | None = None
pinned_q_tiles: torch.Tensor | None = None
pinned_q_idx: torch.Tensor | None = None
def get_pinned(num_tiles: int):
    global pinned_tiles, pinned_q_tiles, pinned_q_idx
    if pinned_tiles is None or pinned_tiles.shape[0] < num_tiles:
        pinned_tiles = torch.empty((num_tiles, 256), device = "cpu", dtype = torch.float, pin_memory = True)
        pinned_q_tiles = torch.empty((num_tiles, 256), device = "cpu", dtype = torch.float, pin_memory = True)
        pinned_q_idx = torch.empty((num_tiles, 256), device = "cpu", dtype = torch.int16, pin_memory = True)
    return pinned_tiles[:num_tiles, :], pinned_q_tiles[:num_tiles, :], pinned_q_idx[:num_tiles, :]


def quantize_tiles_multigpu(tiles, quant_args: dict):
    devices = quant_args["devices"]
    if len(devices) == 1:
        return quantize_tiles(tiles, quant_args)

    # Get pinned buffers
    pin_tiles, pin_q_tiles, pin_q_idx = get_pinned(tiles.shape[0])

    # Copy input tiles to pinned memory. Input is always on the first device in the split
    copy_input_event = torch.cuda.Event(blocking = False)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):
        tiles = tiles.contiguous()
        pin_tiles.copy_(tiles, non_blocking = True)
        copy_input_event.record(main_stream)

    # Create split slices for input tiles, output tiles and output indices
    ratios = quant_args.get("device_ratios")
    if ratios:
        s = sum(ratios)
        split_sizes = [tiles.shape[0] * r / s for r in ratios]
        split_sizes = [round(s / 16) * 16 for s in split_sizes]
        split_sizes[-1] += tiles.shape[0] - sum(split_sizes)
    else:
        split_sizes = [tiles.shape[0] // len(devices)] * len(devices)
        split_sizes[-1] += tiles.shape[0] - sum(split_sizes)

    pin_split_tiles = torch.split(pin_tiles, split_sizes)
    pin_split_q_tiles = torch.split(pin_q_tiles, split_sizes)
    pin_split_q_idx = torch.split(pin_q_idx, split_sizes)

    slice_done_events = []
    for i, device in enumerate(devices):

        stream = get_quant_stream(device)
        with torch.cuda.stream(stream):

            # Wait for input in host memory
            if i > 0:
                stream.wait_event(copy_input_event)

            if split_sizes[i] > 0:

                # Asynchronously copy the slice from the pinned buffer to device memory
                dev_tiles = pin_split_tiles[i].to(device, non_blocking = True)

                # Preallocate output tensors on the device.
                dev_q_tiles = torch.empty_like(dev_tiles, device = device)
                dev_q_idx = torch.empty_like(dev_tiles, dtype = torch.short, device = device)

                # Work buffers
                K = quant_args["K"]
                temp_costs, temp_edges = get_temp_buffers(device, K)

                ext.quantize_tiles(
                    dev_tiles,
                    dev_q_tiles,
                    dev_q_idx,
                    temp_costs,
                    temp_edges,
                    K
                )

                # Async copy back to pinned memory
                pin_split_q_tiles[i].copy_(dev_q_tiles, non_blocking = True)
                pin_split_q_idx[i].copy_(dev_q_idx, non_blocking = True)

            # Finished slice
            evt = torch.cuda.Event(blocking = False)
            slice_done_events.append(evt)
            evt.record(stream)

    # Copy pinned buffers to original device
    with torch.cuda.stream(main_stream):
        for evt in slice_done_events:
            main_stream.wait_event(evt)
        q_tiles = torch.empty_like(tiles, device = devices[0])
        q_idx = torch.empty_like(tiles, dtype = torch.short, device = devices[0])
        q_tiles.copy_(pin_q_tiles, non_blocking = True)
        q_idx.copy_(pin_q_idx, non_blocking = True)

    return q_tiles, q_idx


def quantize_tiles_multigpu_sync(tiles, quant_args: dict):
    devices = quant_args["devices"]
    if len(devices) == 1:
        return quantize_tiles(tiles, quant_args)

    tiles = tiles.contiguous()

    split_sizes = [tiles.shape[0] // len(devices)] * len(devices)
    split_sizes[-1] += tiles.shape[0] - sum(split_sizes)
    split_tiles = torch.split(tiles, split_sizes)
    tiles_per_device = [chunk.to(device) for chunk, device in zip(split_tiles, devices)]
    torch.cuda.synchronize()

    q_tiles_per_device = []
    q_idx_per_device = []
    for dev_tiles, device in zip(tiles_per_device, devices):
        with torch.cuda.stream(get_quant_stream(device)):
            dev_q_tiles, dev_q_idx = quantize_tiles(dev_tiles, quant_args)
            q_tiles_per_device.append(dev_q_tiles)
            q_idx_per_device.append(dev_q_idx)

    for device in devices:
        torch.cuda.synchronize(device)

    q_tiles_per_device = [x.to(devices[0]) for x in q_tiles_per_device]
    q_idx_per_device = [x.to(devices[0]) for x in q_idx_per_device]
    quantized_tiles = torch.cat(q_tiles_per_device, dim = 0)
    quantized_idx = torch.cat(q_idx_per_device, dim = 0)
    return quantized_tiles, quantized_idx


def preapply_had_l(x: torch.Tensor, had_dim):
    k, n = x.shape
    xdt = x.dtype
    x = x.to(torch.float)
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    x = (had @ x.view(-1, had_dim, n)).view(k, n)
    x = x.to(xdt)
    return x


def preapply_had_r(x: torch.Tensor, had_dim):
    k, n = x.shape
    xdt = x.dtype
    x = x.to(torch.float)
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    x = (x.view(k, -1, had_dim) @ had).view(k, n)
    x = x.to(xdt)
    return x


def block_ldl(H: torch.Tensor, b: int):

    n, _ = H.shape
    assert (n % b == 0)
    m = n // b

    # Cholesky factorization: H = L @ L.T
    L = torch.linalg.cholesky(H)

    # Get blocks along diagonal of L: DL.shape = (m, b, b)
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1 = 0, dim2 = 2).permute(2, 0, 1)

    # Compute D as D[i] = DL[i] @ DL[i].T for each diagonal block i
    D = DL @ DL.transpose(1, 2)

    # Invert each diagonal block
    DL = torch.linalg.inv(DL)

    # Multiply each block's column with its inverse
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]  # TODO: Could maybe be L[m * b:, i, :]?
    L = L.reshape(n, n).contiguous()

    # Insert block identity matrices along the diagonal.
    # TODO: Figure out if this is necessary. Diagonal blocks should already be identities after previous step
    L_block = L.view(m, b, m, b).permute(0, 2, 1,3)
    dr = torch.arange(m)
    L_block[dr, dr] = torch.stack([torch.eye(b, device = L.device, dtype = H.dtype)] * m)

    return L, D.to(DL.device)


def ldlq(
    weight: torch.Tensor,
    L: torch.Tensor,
    quant_args: dict,
    pb: ProgressBar | None = None
):
    """
    :param weight:
        Input weights, shape (k, n). If device is "cpu", result is collected on CPU as well, saving a bunch of
        VRAM but adding a little PCIe overhead and many sync points

    :param L:
        LDL decomposition of regularized H

    :param quant_args:
        dict:
         - K: bitrate
         - buf_size_k: buffer size for LDLQ, along k

    :param pb:
        Optional ProgressPar to update, k // 16 steps

    :return:
        tuple:
         - quantized weight, shape (k, n)
         - indices (unpacked), shape (k // 16, n // 16, 256), uint16_t
    """

    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):

        devices = quant_args["devices"]
        device = L.device
        assert device == torch.device(devices[0])

        buffer_device = weight.device
        size_k, size_n = weight.shape  # Row-major
        assert size_k % 16 == 0
        assert size_n % 128 == 0
        tiles_k = size_k // 16
        tiles_n = size_n // 16

        buf_size_k = max(quant_args.get("buf_size_k", 128), 16)
        assert buf_size_k % 16 == 0
        assert size_n % buf_size_k == 0

        p_row = 0

        # Work buffers
        prod_cache = torch.zeros((size_k, size_n), dtype = torch.float, device = device)
        weight_q = torch.zeros((size_k, size_n), dtype = torch.float, device = buffer_device)
        encoded = torch.zeros((tiles_k, tiles_n, 256), dtype = torch.short, device = buffer_device)

        for j in range(size_k, 0, -buf_size_k):
            i = j - buf_size_k

            # Current span is rows i:j
            b_weight = weight[i:j].to(device)
            b_weight_q = weight_q[i:j] if device == buffer_device else \
                torch.zeros_like(weight_q[i:j], device = device)
            b_encoded = encoded[i // 16 : j // 16] if device == buffer_device else \
                torch.zeros_like(encoded[i // 16 : j // 16], device = device)
            b_prod_cache = prod_cache[i:j]
            b_L = L[i:j]

            # Iterate over rows of blocks in current span
            for bj in range(buf_size_k, 0, -16):
                bi = bj - 16

                # Error so far for the current span
                bb_err = b_weight[bj:] - b_weight_q[bj:]

                # Corresponding slice of LDL decomposition of H
                bb_L = b_L[bj:, i + bi:i + bj]

                # Input tiles for quantization
                compensation_term = b_prod_cache[bi:bj]
                compensation_term.addmm_(bb_L.T, bb_err,  alpha = 1.0, beta = 1.0)
                rows = b_weight[bi:bj] + compensation_term

                tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)

                # Pre-permute to tensor core layout
                tiles = tiles[:, tensor_core_perm(device)]

                # Quantize
                quant_w, quant_i = quantize_tiles_multigpu(tiles, quant_args)

                # Undo permutation on reconstructed tiles, but keep indices in tensor core layout
                quant_w = quant_w[:, tensor_core_perm_i(device)]

                # Store result
                quant_w = quant_w.reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, size_n)
                b_weight_q[bi:bj] = quant_w
                b_encoded[bi // 16 : bj // 16] = quant_i.unsqueeze(0)

                # Update progress
                if pb:
                    p_row += 1
                    pb.update(p_row)

            # Collect output
            if device != buffer_device:
                weight_q[i:j] = b_weight_q.to(buffer_device)
                encoded[i // 16 : j // 16] = b_encoded.to(buffer_device)

            # Cache error term for the rest of the matrix
            b_err = b_weight - b_weight_q
            prod_cache.addmm_(b_L.T, b_err, alpha = 1.0, beta = 1.0)

        for device in devices:
            torch.cuda.synchronize(device)

    return weight_q, encoded


def finalize_capture_H(H_data: dict, quant_args: dict):
    # Unswap H
    if "H_swap_device" in H_data:
        H_data["H"] = H_data["H"].to(H_data["H_swap_device"])
        del H_data["H_swap_device"]

    H = H_data["H"]
    if H_data["finalized"]:
        return H, H_data["L"], H_data["su"]

    # Mean of samples summed up during forward pass
    H /= H_data["count"]

    # Regularize diagonal
    diag_mean = torch.diag(H).mean()
    idx = torch.arange(H.shape[0])
    H[idx, idx] += quant_args.get("sigma_reg", 0.025) * diag_mean

    # Random sign flips for input channel, fixed for the first linear layer to quantize with this H
    k = H.shape[0]
    su = (torch.randn(k, device = H.device).sign() + 1e-5).sign().to(torch.float).unsqueeze(1)
    H_data["su"] = su

    # Input had
    H *= su.T
    H = preapply_had_r(H, had_k)  # Todo: in-place had kernels, to save some memory here
    H *= su
    H = preapply_had_l(H, had_k)

    # Get block LDL decomposition of H, zero diagonal
    L, _ = block_ldl(H, 16)
    dr = torch.arange(k)
    L[dr, dr] = 0
    H_data["L"] = L

    # H is no longer needed except to compute proxy error so move to CPU
    H = H.cpu()
    H_data["H"] = H.cpu()

    H_data["finalized"] = True
    return H, L, su


def pack_trellis(encoded: torch.Tensor, quant_args: dict) -> torch.Tensor:
    K = quant_args["K"]
    shape = encoded.shape
    assert len(shape) == 3 and shape[2] == 256
    assert encoded.dtype == torch.int16
    packed_shape = (shape[0], shape[1], 256 * K // 16)
    packed = torch.zeros(packed_shape, dtype = torch.int16, device = encoded.device)
    ext.pack_trellis(packed, encoded.contiguous(), K)
    # unpacked = torch.zeros_like(encoded)
    # ext.unpack_trellis(unpacked, packed, K)
    # assert torch.equal(unpacked, encoded)
    return packed


def pack_signs(signs: torch.Tensor, quant_args: dict) -> torch.Tensor:
    signs = signs.half().flatten().contiguous()
    assert signs.shape[0] % 16 == 0
    packed = torch.zeros(signs.shape[0] // 16, dtype = torch.int16, device = signs.device)
    ext.pack_signs(packed, signs)
    return packed


def g_scale_gss(
    weight_r: torch.Tensor,
    verbose: bool,
    quant_args: dict,
    width: int = 3,
    pb: ProgressBar = None
):
    # Select a sample of tiles along a wrapped diagonal (sampling from every row and column of tiles, hopefully
    # representative) and search for the global scale within given range that minimizes the direct quantization
    # error
    tiles = []
    tiles_k = weight_r.shape[0] // 16
    tiles_n = weight_r.shape[1] // 16
    for i in range(max(tiles_k, tiles_n)):
        for w in range(width):
            k = (i % tiles_k) * 16
            n = ((i + width) % tiles_n) * 16
            tile = weight_r[k : k + 16, n : n + 16].clone()
            tile = tile.view(256)
            tile = tile[tensor_core_perm(weight_r.device)]
            tiles.append(tile)
    tiles = torch.stack(tiles)

    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)

    main_stream = get_quant_stream(devices[0])
    # TODO: Figure out why Torch always initializes cuda:0 when exiting this CM, even when it's not used
    with torch.cuda.stream(main_stream):

        def test_scale(scale: float):
            quant_w, quant_i = quantize_tiles_multigpu(tiles * scale, quant_args)
            mse = ((quant_w / scale - tiles) ** 2).mean()
            return mse

        # Assume quantization error is a unimodal function of scale, golden section search to find minimum
        phi = (1 + math.sqrt(5)) / 2
        resphi = 2 - phi

        a, b = 0.1, 1.9
        tol = 0.01
        delta1 = abs(b - a)

        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)
        f1 = test_scale(x1)
        f2 = test_scale(x2)
        while abs(b - a) > tol:
            if verbose:
                print(f"     - gss: a = {a:.6f}, b = {b:.6f}")
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + resphi * (b - a)
                f1 = test_scale(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - resphi * (b - a)
                f2 = test_scale(x2)
            delta2 = abs(b - a)
            pb.update(100 - 100 * int(delta2 / delta1))

        best_scale = (a + b) / 2
        if verbose:
            print(f"     - gss: min = {best_scale:.6f}, mse: {(f1 + f2) / 2:.6f}")
        return best_scale


def block_rms(x: torch.Tensor, dim: int, keepdim: bool = False, blocksize: int = 32):
    # Compute blockwise x.square().mean(dim, keepdim).sqrt()
    n = x.size(dim)
    sq = None
    for block in torch.split(x, blocksize, dim = dim):
        block_sq = block.square().sum(dim = dim, keepdim = keepdim)
        if sq is None:
            sq = block_sq
        else:
            sq += block_sq
    mean_sq = sq / n
    return mean_sq.sqrt()


def block_rms_n(x: torch.Tensor, dim: int = 0, blocksize: int = 32):
    # Compute blockwise x.square().mean().sqrt()
    n = 0
    sq = None
    for block in torch.split(x, blocksize, dim = dim):
        block_sq = block.square().sum()
        n += block.numel()
        if sq is None:
            sq = block_sq
        else:
            sq += block_sq
    mean_sq = sq / n
    return mean_sq.sqrt()


def block_nmse(x: torch.Tensor, y: torch.Tensor, dim: int = 0, blocksize: int = 32):
    # Compute blockwise (x - y).square().mean().item() / y.square().mean().item()
    sq = None
    diff_sq = None
    for block_x, block_y in zip(torch.split(x, blocksize, dim = dim), torch.split(y, blocksize, dim = dim)):
        block_sq = block_y.square().sum()
        block_diff_sq = (block_x - block_y).square().sum()
        if sq is None:
            sq = block_sq
            diff_sq = block_diff_sq
        else:
            sq += block_sq
            diff_sq += block_diff_sq
    return diff_sq.item() / (sq.item() + 1e-20)


def quantize_exl3(
    weight: torch.Tensor,
    H_data: dict,
    quant_args: dict,
    return_weight_q: bool,
    progress_str: str | None = None,
    verbose: bool = False
):
    """
    :param weight:
        Input tensor, row major shape (in_features, out_features)

    :param H_data:
        Dictionary of hessian tensor and related data, as collected by Linear wrapper class. May be reused between
        linear layers (e.g. Q, K and V projections with the same input)

    :param quant_args:
        dict:
         - K: bitrate
         - seed: integer seed for random sign flips etc.
         - sigma_reg: regularization factor

    :param return_weight_q:
        Return quantized weight

    :param progress_str:
        Show progress bar during quantization

    :param verbose:
        Dump extra stats

    :return:
        tuple:
          - quantized weight
          - proxy error: trace(err @ H @ err.T) / (W @ H @ W.T)
          - quantized and packed tensors
    """

    progress_text = None if not progress_str else progress_str.replace("<step>", "Scaling")
    with ProgressBar(progress_text, 100) as pb:

        assert weight.dtype == torch.float
        tiles_k = weight.shape[0] // 16

        if "seed" in quant_args:
            torch.manual_seed(quant_args["seed"])

        device = weight.device
        k, n = weight.shape

        # Get H, LDL decomp. and input sign flips
        H, L, su = finalize_capture_H(H_data, quant_args)

        # Move stored L to CPU (if not already), move working L to device
        H_data["L"] = H_data["L"].cpu()
        L = L.to(device)

        codebook_scale = 1.24371088

        if verbose:
            weight_copy = weight.cpu()
        weight_r = weight
        weight = None

        if verbose:
            rms = block_rms_n(weight_r, dim = 0)
            print(f"     - input rms: {rms:.6f}")

        # Regularize output channels, then input channels with sign flips and scales. Output channel scales (sv)
        # are normalized to keep the global scale of the matrix incorporated in the input channel scales (su).
        # This reduces the risk of overflows during the GEMM and output hadamard transform in the linear layer.
        out_channel_scales = block_rms(weight_r, dim = 0, keepdim = True)
        out_channel_scales /= (out_channel_scales.abs() + 1e-10).mean()
        sv = (torch.randn(n, device = device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)
        sv = (sv * out_channel_scales + 1e-10).float()

        weight_r /= sv
        weight_r = preapply_had_r(weight_r, had_n)

        in_channel_scales = block_rms(weight_r, dim = 1, keepdim = True)
        su = (su * in_channel_scales / (-codebook_scale) + 1e-10).float()

        weight_r /= su
        weight_r = preapply_had_l(weight_r, had_k)

        # At this point the overall distribution of weight_r should be roughly matched to the codebook, but we may
        # still have too many outliers. For some tensors it is advantageous to dial in the global scale to keep
        # this number below a small threshold.
        g_scale = g_scale_gss(weight_r, verbose, quant_args, pb = pb)

        progress_text = None if not progress_str else progress_str.replace("<step>", "Quantizing")
        pb.update(0)
        pb.new_task(progress_text, tiles_k)

        weight_r *= g_scale
        su /= g_scale

        if verbose:
            rms = weight_r.square().mean().sqrt()
            print(f"     - global scale: {g_scale:.6f}")
            print(f"     - regularized rms:  {rms:.6f}")

        # Select device for work buffers (CPU is slower for small tensors but saves a lot of VRAM on big ones)
        # TODO: Use pynvml or mem_get_info to predict whether CPU buffer is needed
        if weight_r.numel() > 5e8:
            weight_r = weight_r.cpu()

        # Quantize
        # free_mem()
        weight_q, encoded_q = ldlq(weight_r, L, quant_args, pb)
        # free_mem()

        pb.update(tiles_k)

        # Metrics
        E = weight_r - weight_q  # may run on CPU
        W = weight_r
        Hd = H.to(device)
        weight_r = None
        E = E.to(device)
        num = torch.einsum("ik,ij,jk->", E, Hd, E).item()
        E = None
        W = W.to(device)
        den = torch.einsum("ik,ij,jk->", W, Hd, W).item()
        W = None
        Hd = None
        proxy_err = num / max(den, 1e-8)

        # free_mem()

        if return_weight_q or verbose:
            weight_q = weight_q.to(device)
            weight_q = preapply_had_l(weight_q, had_k)
            weight_q *= su
            weight_q = preapply_had_r(weight_q, had_n)
            weight_q *= sv

            if verbose:
                weight = weight_copy.to(device)
                nmse = block_nmse(weight_q, weight)
                print(f"     - quant nmse: {nmse:.6f}")

        # Compile packed tensor
        suh = su.flatten().contiguous().to(dtype = torch.half, copy = True)
        svh = sv.flatten().contiguous().to(dtype = torch.half, copy = True)
        trellis = pack_trellis(encoded_q.to(device), quant_args)

        out_tensors = {
            # "scale": weight_scale.to(dtype = torch.float, copy = True),
            # "su": pack_signs(su, quant_args),
            "suh": suh,
            # "sv": pack_signs(sv, quant_args),
            "svh": svh,
            "trellis": trellis,
        }

    return weight_q, proxy_err, out_tensors