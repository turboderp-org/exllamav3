import torch
import torch.nn.functional as F
import math
from ....ext import exllamav3_ext as ext
from ....util.progress import ProgressBar
from ....util.memory import free_mem, list_gpu_tensors
from ....util.hadamard import get_hadamard_dt
from ....util import cuda_sync_active
from ....util.tensor import save_tensor_image
from functools import lru_cache
import threading

# Constant
had_k, had_n = 128, 128
codebook_scale = 1.24371088

codebook_mcg_mult = 0xCBAC1FED
codebook_mul1_mult = 0x83DCD12D

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
    max_batch_size = 256
    temp_costs = torch.zeros((max_batch_size, 2, 65536 >> K), dtype = torch.half, device = device)
    temp_edges = torch.zeros((max_batch_size, 256, 65536 >> K), dtype = torch.short, device = device)
    return temp_costs, temp_edges


def quantize_tiles(tiles, quant_args: dict):
    tiles = tiles.contiguous()
    assert tiles.shape[1] == 256
    assert tiles.dtype == torch.float

    K = quant_args["K"]
    mcg = "mcg" in quant_args
    mul1 = "mul1" in quant_args
    quantized_tiles = torch.zeros_like(tiles)
    quantized_idx = torch.zeros_like(tiles, dtype = torch.short)
    temp_costs, temp_edges = get_temp_buffers(tiles.device, K)
    ext.quantize_tiles(
        tiles,
        quantized_tiles,
        quantized_idx,
        temp_costs,
        temp_edges,
        K,
        mcg,
        mul1,
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

    # Account for negative splits (edge case if too many GPUs and/or tensor too small)
    for i in range(len(split_sizes) - 2, -1, -1):
        if split_sizes[i + 1] < 0:
            split_sizes[i] += split_sizes[i + 1]
            split_sizes[i + 1] = 0

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
                mcg = "mcg" in quant_args
                mul1 = "mul1" in quant_args
                temp_costs, temp_edges = get_temp_buffers(device, K)

                ext.quantize_tiles(
                    dev_tiles,
                    dev_q_tiles,
                    dev_q_idx,
                    temp_costs,
                    temp_edges,
                    K,
                    mcg,
                    mul1
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
    x_dtype = x.dtype
    x = x.to(torch.float)
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    x = (had @ x.view(-1, had_dim, n)).view(k, n)
    x = x.to(x_dtype)
    return x


def preapply_had_r(x: torch.Tensor, had_dim):
    k, n = x.shape
    x_dtype = x.dtype
    x = x.to(torch.float)
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    x = (x.view(k, -1, had_dim) @ had).view(k, n)
    x = x.to(x_dtype)
    return x


def blockwise_preapply_had_l_(x: torch.Tensor, had_dim):
    k, n = x.shape
    assert k % had_dim == 0
    assert x.dtype == torch.float
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    num_blocks = k // had_dim
    for i in range(num_blocks):
        start = i * had_dim
        end = start + had_dim
        block = x[start:end, :]  # shape (had_dim, n)
        block_transformed = had @ block.view(had_dim, n)
        x[start:end, :] = block_transformed


def blockwise_preapply_had_r_(x: torch.Tensor, had_dim):
    k, n = x.shape
    assert n % had_dim == 0
    assert x.dtype == torch.float
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    num_blocks = n // had_dim
    for i in range(num_blocks):
        start = i * had_dim
        end = start + had_dim
        block = x[:, start:end]  # shape (k, had_dim)
        block_transformed = block @ had
        x[:, start:end] = block_transformed


def block_ldl(H: torch.Tensor, b: int, verbose: bool):

    n, _ = H.shape
    assert (n % b == 0)
    m = n // b

    # Cholesky factorization: H = L @ L.T
    # Try on GPU first
    try:
        retry_cpu = False
        L = torch.linalg.cholesky(H)
        # H is not needed after this, move to CPU. Then overwrite H's GPU storage with L, since we can't otherwise
        # free up that VRAM as the tensor is referenced by the parent frame
        H_cpu = H.cpu()
        H.copy_(L)  # VRAM copy, tiny overhead
        L = H
        H = H_cpu

    # Fall back on CPU factorization
    except Exception as e:
        if e.__class__.__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e) or "HIP out of memory" in str(e):
            retry_cpu = True
        else:
            raise e
    if retry_cpu:
        print(f" !! Out of memory on {str(H.device)}, trying CPU fallback")
        free_mem()
        H_cpu = H.cpu()
        L_cpu = torch.linalg.cholesky(H_cpu)
        # This is ugly, but overwrite H in VRAM to avoid allocating a new tensor, then replace reference with CPU copy
        H.copy_(L_cpu)
        del L_cpu
        L = H
        H = H_cpu

    # Get blocks along diagonal of L: DL.shape = (m, b, b)
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1 = 0, dim2 = 2).permute(2, 0, 1)

    # Compute D as D[i] = DL[i] @ DL[i].T for each diagonal block i (don't actually end up needing this)
    # D = DL @ DL.transpose(1, 2)

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

    return L, H  # , D.to(DL.device)


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


finalize_capture_H_mutex = threading.Lock()

def finalize_capture_H(H_data: dict, quant_args: dict, verbose: bool):
    with finalize_capture_H_mutex:

        # Unswap H
        if "H_swap_device" in H_data:
            H_data["H"] = H_data["H"].to(H_data["H_swap_device"])
            del H_data["H_swap_device"]

        H = H_data["H"]
        if H_data["finalized"]:
            return H, H_data["L"], H_data["su"], H_data["diag"]

        # Mean of samples summed up during forward pass
        H /= H_data["count"]

        # Regularize diagonal
        diag_mean = torch.diag(H).mean()
        idx = torch.arange(H.shape[0])
        H[idx, idx] += quant_args.get("sigma_reg", 0.025) * diag_mean

        # Some tests
        diag = H[idx, idx].clone()

        if verbose:
            print(f"     - H min/max: {H.min().item():.6f}   {H.max().item():.6f}")
            print(f"     - H mean/std: {H.mean().item():.6f}   {H.std().item():.6f}")
            print(f"     - H diag min/max: {diag.min():.6f}   {diag.max():.6f} ")

        # Random sign flips for input channel, fixed for the first linear layer to quantize with this H
        k = H.shape[0]
        su = (torch.randn(k, device = H.device).sign() + 1e-5).sign().to(torch.float).unsqueeze(1)
        H_data["su"] = su

        # Input had
        H *= su.T
        blockwise_preapply_had_r_(H, had_k)
        H *= su
        blockwise_preapply_had_l_(H, had_k)

        # Get block LDL decomposition of H, zero diagonal
        L, H = block_ldl(H, 16, verbose)
        dr = torch.arange(k)
        L[dr, dr] = 0
        H_data["L"] = L

        # H is no longer needed except to compute proxy error so move to CPU
        H = H.cpu()
        H_data["H"] = H.cpu()

        H_data["finalized"] = True
        H_data["diag"] = diag
        return H, L, su, diag


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
            n = ((i + w) % tiles_n) * 16
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
            # if verbose:
                # print(f"     - gss: a = {a:.6f}, b = {b:.6f}")
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
            if pb:
                pb.update(100 - 100 * int(delta2 / delta1))

        best_scale = (a + b) / 2
        if verbose:
            print(f"     - gss: min = {best_scale:.6f}, mse: {(f1 + f2) / 2:.6f}")

    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)

    return best_scale, (f1 + f2) / 2


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


def regularize(
    weight: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    quant_args: dict,
    verbose: bool,
    H_diag: torch.Tensor | None,
    pb: ProgressBar | None,
    skip_g_scale: bool = False
):
    force_out_scales = quant_args["apply_out_scales"]

    # dist_ref = torch.empty((512,), dtype = torch.float, device = weight.device)
    # dist_r = torch.empty_like(dist_ref)
    def jsd(h1, h2):
        m = (h1 + h2) / 2
        eps = 1e-12
        js = F.kl_div((h1 + eps).log(), m, reduction = "sum") + \
             F.kl_div((h2 + eps).log(), m, reduction = "sum")
        return js / 2

    # From experiments, it seems the deciding factor in when scaling output channels is beneficial is when
    # the input to the linear layer is very irregular. After some testing, set the cutoff at 15% of the RMS sum
    # on 2% of the channels
    # TODO: More science
    if H_diag is not None:
        diag = H_diag.sqrt()
        diag, _ = torch.sort(diag, descending = True)
        cutoff = diag.shape[0] // 50
        skew_factor = diag[:cutoff].sum() / diag.sum()
        if verbose:
            print(f"     - input state skew: {skew_factor.item():.6f}")

        if force_out_scales is None:
            apply_out_scales = skew_factor.item() < 0.15
        else:
            apply_out_scales = force_out_scales

    else:
        apply_out_scales = True if force_out_scales is None else force_out_scales

    # Apply output scales
    if apply_out_scales:
        out_channel_scales = block_rms(weight, dim = 0, keepdim = True)
        out_channel_scales /= out_channel_scales.mean()
        sv = (sv * out_channel_scales + 1e-10).float()
        if verbose:
            out_channel_std = out_channel_scales.std().item()
            out_channel_mean = out_channel_scales.mean().item()
            print(f"     - out ch scales std/mean: {out_channel_std:.6f}   {out_channel_mean:.6f}")

    # Output sign flips (and scales)
    weight /= sv

    # Output hadamard transform
    blockwise_preapply_had_r_(weight, had_n)

    # Input sign flips and scales
    in_channel_scales = block_rms(weight, dim = 1, keepdim = True)
    su = (su * in_channel_scales / (-codebook_scale) + 1e-10).float()  # mustn't be inplace
    weight /= su
    blockwise_preapply_had_l_(weight, had_k)

    # Determine best scale for matrix by test quantizing a sample of tiles along a wrapped diagonal
    if not skip_g_scale:
        g_scale, mse_scale = g_scale_gss(weight, False, quant_args, pb = pb)
    else:
        g_scale = 1.0
    weight *= g_scale
    su /= g_scale

    # ext.test_distribution(weight_os, dist_r, dist_ref, -3.8, 3.8)
    # js_os = jsd(dist_r, dist_ref)

    if verbose:
        print(f"     - su/sv std: {su.std().item():.6f}   {sv.std().item():.6f}")
        print(f"     - global scale: {g_scale:.6f}")
        print(f"     - sample mse: {mse_scale.item():.6f}")
        print(f"     - apply_out_scales: {str(apply_out_scales)}")

    return apply_out_scales, weight, g_scale, su, sv


def quantize_exl3(
    weight: torch.Tensor,
    H_data: dict,
    quant_args: dict,
    return_weight_q: bool,
    progress_str: str | None = None,
    verbose: bool = False,
    swap_to_device: torch.device | None = None,
    save_reg: str = None
):
    """
    :param weight:
        Input tensor, row major shape (in_features, out_features)

    :param H_data:
        Dictionary of hessian tensor and related data, as collected by Linear wrapper class. May be reused between
        linear layers with the same input (e.g. Q, K and V projections)

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

    :param swap_to_device:
        If input tensor is on CPU, move to this device before quantization

    :param save_reg:
        Save regularized tensor as image to the provided path

    :return:
        tuple:
          - quantized weight
          - proxy error: trace(err @ H @ err.T) / (W @ H @ W.T)
          - quantized and packed tensors
    """

    progress_text = None if not progress_str else progress_str.replace("<step>", "Preparing")
    with (ProgressBar(progress_text, 100) as pb):

        assert weight.dtype == torch.float
        tiles_k = weight.shape[0] // 16

        if "seed" in quant_args:
            torch.manual_seed(quant_args["seed"])

        devices = quant_args["devices"]
        if weight.device != torch.device(devices[0]):
            weight = weight.to(devices[0])

        device = weight.device if swap_to_device is None else swap_to_device
        k, n = weight.shape

        # Get H, LDL decomp. and input/output sign flips
        H, L, su, H_diag = finalize_capture_H(H_data, quant_args, verbose)
        if H.is_cuda: H = H.to(device)
        if L.is_cuda: L = L.to(device)
        if su.is_cuda: su = su.to(device)
        if H_diag.is_cuda: H_diag = H_diag.to(device)
        sv = (torch.randn(n, device = device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)

        # Move stored L to CPU (if not already), move working L to device
        H_data["L"] = H_data["L"].cpu()
        L = L.to(device)

        if swap_to_device is not None:
            weight = weight.to(swap_to_device)
        if verbose:
            weight_copy = weight.cpu()
        weight_r = weight
        del weight

        if verbose:
            rms = block_rms_n(weight_r, dim = 0)
            print(f"     - input tensor rms: {rms:.6f}")

        # Regularization
        apply_out_scales, weight_r, g_scale, su, sv = regularize(
            weight_r,
            su,
            sv,
            quant_args,
            verbose,
            H_diag,
            pb
        )

        if save_reg:
            save_tensor_image(weight_r, save_reg)

        if verbose:
            rms = weight_r.square().mean().sqrt()
            print(f"     - regularized rms:  {rms:.6f}")

        progress_text = None if not progress_str else progress_str.replace("<step>", "Quantizing")
        pb.update(0)
        pb.new_task(progress_text, tiles_k)

        # Select device for work buffers (CPU is slower for small tensors but saves a lot of VRAM on big ones)
        # TODO: Use pynvml or mem_get_info to predict whether CPU buffer is needed
        if weight_r.numel() > 5e8:
            weight_r = weight_r.cpu()

        # Quantize
        weight_q, encoded_q = ldlq(weight_r, L, quant_args, pb)
        del L

        pb.update(tiles_k)

        # Metrics
        try:
            def block_trace(A, B, block_size = 1024):
                total = 0.0
                for j_start in range(0, B.shape[1], block_size):
                    j_end = min(j_start + block_size, B.shape[1])
                    B_block = B[:, j_start:j_end]
                    A_j_block = A[j_start:j_end, :]
                    partial = torch.einsum("ik,ij,jk->", A, B_block, A_j_block)
                    total += partial.item()
                return total
            E = weight_r - weight_q  # may run on CPU
            W = weight_r
            Hd = H.to(device)
            weight_r = None
            E = E.to(device)
            num = block_trace(E, Hd)
            E = None
            W = W.to(device)
            den = block_trace(W, Hd)
            W = None
            Hd = None
            proxy_err = num / max(den, 1e-8)
        except torch.OutOfMemoryError:
            weight_r = None
            E = None
            W = None
            Hd = None
            proxy_err = -1.0

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

        # Safetensors doesn't know what to do with a torch.uint32 tensor. Anyway, since the multipliers are now
        # locked, the values in these tensors are never read, but they need to be present in the model files to
        # indicate which codebook to use during inference, per individual tensor.
        if quant_args.get("mcg"):
            out_tensors.update({
                "mcg": torch.tensor(codebook_mcg_mult, dtype = torch.uint32).view(torch.int)
            })
        if quant_args.get("mul1"):
            out_tensors.update({
                "mul1": torch.tensor(codebook_mul1_mult, dtype = torch.uint32).view(torch.int)
            })

        quant_args.update({
            "apply_out_scales": apply_out_scales,
            "g_scale": g_scale,
        })

    return weight_q, proxy_err, out_tensors