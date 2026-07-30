import torch
import torch.nn.functional as F
import math
import os
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
    """
    Return the 16x16 tile permutation expected by the EXL3 tensor-core quantization kernels.

    The cached index maps row-major tile elements into the lane/interleave order used by the CUDA encoder.
    """
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
    # The kernel runs one block per tile and caps each wave at min(temp_costs.size(0), 2 * SMs). At K >= 4 the
    # temp buffers are cheap enough to size for full occupancy on large GPUs (+17% throughput on GB202 at big
    # batches); at lower K, temp_edges is multiple GB already and 256 stays the cap
    max_batch_size = 256
    if K >= 4:
        mp_count = torch.cuda.get_device_properties(device).multi_processor_count
        max_batch_size = max(256, 2 * mp_count)
    temp_costs = torch.zeros((max_batch_size, 2, 65536 >> K), dtype = torch.half, device = device)
    temp_edges = torch.zeros((max_batch_size, 256, 65536 >> K), dtype = torch.short, device = device)
    return temp_costs, temp_edges


def quantize_tiles(tiles, quant_args: dict):
    """
    Quantize a batch of 16x16 tiles on the current device.

    tiles is shaped (num_tiles, 256) in the kernel's expected element order. The CUDA extension returns both the
    reconstructed float tile values and the short encoded indices used later for packing.
    """
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


@lru_cache
def arch_prior_speed(device: int) -> float:
    """Initial relative quant-throughput guess by architecture, used to seed device splits
    before any measurements exist. From measured conversion throughput: Ampere lands at about
    half of Ada, which is about 80% of Blackwell (Hopper assumed equal to Blackwell); anything
    older is pessimistically assumed half of Ampere."""
    major, minor = torch.cuda.get_device_capability(device)
    if major >= 9: return 5.0                        # Hopper, Blackwell
    if major == 8: return 4.0 if minor == 9 else 2.0  # Ada / Ampere
    return 1.0


class AutoSplit:
    """Learned per-device speed ratios for heterogeneous multi-GPU conversion splits. Each
    workload kind (parallel-quant threads, calibration row workers, tile fan-out slices) is
    tracked separately, since their per-unit costs differ; within a kind, only the ratios
    between devices matter. Speeds are EMA-damped so one noisy module (JIT warmup, thermal
    excursions) doesn't swing the split. Until every active device has a measurement for the
    kind, the split falls back to the per-architecture prior."""

    def __init__(self):
        self.lock = threading.Lock()
        self.speeds = {}   # kind -> {device: ema work/sec}

    def report(self, kind: str, device: int, work: float, elapsed_s: float):
        if work <= 0 or elapsed_s < 0.05:
            return
        s = work / elapsed_s
        with self.lock:
            d = self.speeds.setdefault(kind, {})
            prev = d.get(device)
            d[device] = s if prev is None else 0.7 * prev + 0.3 * s

    def ratios(self, kind: str, devices: list) -> list:
        with self.lock:
            d = self.speeds.get(kind)
            if d is not None and all(dev in d for dev in devices):
                return [d[dev] for dev in devices]
        # Incomplete measurements: architecture priors (mixing measured speeds with unit-less
        # priors would skew the split, so it is one or the other)
        return [arch_prior_speed(dev) for dev in devices]

    def has_measured(self, kind: str, devices: list) -> bool:
        with self.lock:
            d = self.speeds.get(kind)
            return d is not None and all(dev in d for dev in devices)

auto_split = AutoSplit()

# Pending timing events from the last measured quantize_tiles_multigpu call, and per-device
# accumulators: single calls are ~ms-scale, so busy time is aggregated over many calls and
# reported to the auto split once enough has been observed
_tiles_calls = 0
_tiles_timing = None
_tiles_acc = {}   # device -> [tiles, seconds]


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
    """
    Quantize tiles across multiple GPUs using pinned host memory for asynchronous fan-out/fan-in.

    The input starts on the first device, is copied once to pinned CPU memory, split by device_ratios or evenly, and
    each GPU quantizes its slice on a per-device stream. Results are copied back through pinned memory and gathered
    on the first device.
    """
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

    # Harvest the previous measured call's per-device busy times (the events have long
    # completed by the next call) into the accumulators, and report once every device has
    # enough observed time for a meaningful speed. Per-device event pairs, since elapsed_time
    # cannot cross devices; a new measured call is fenced periodically
    global _tiles_calls, _tiles_timing
    if _tiles_timing is not None:
        p_devs, p_sizes, p_starts, p_ends = _tiles_timing
        if all(e.query() for e in p_ends):
            for dv, sz, e0, e1 in zip(p_devs, p_sizes, p_starts, p_ends):
                acc = _tiles_acc.setdefault(dv, [0, 0.0])
                acc[0] += sz
                acc[1] += e0.elapsed_time(e1) / 1000
            _tiles_timing = None
            if all(_tiles_acc.get(dv, (0, 0.0))[1] > 0.25 for dv in devices):
                for dv in devices:
                    acc = _tiles_acc.pop(dv)
                    auto_split.report("quant_tiles", dv, acc[0], acc[1])
    _tiles_calls += 1
    measure = (
        _tiles_timing is None and _tiles_calls % 4 == 0 and
        all(s > 0 for s in split_sizes) and tiles.shape[0] >= 256
    )
    m_starts = []

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

            if measure:
                e0 = torch.cuda.Event(enable_timing = True)
                e0.record(stream)
                m_starts.append(e0)

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
            evt = torch.cuda.Event(blocking = False, enable_timing = measure)
            slice_done_events.append(evt)
            evt.record(stream)

    if measure:
        _tiles_timing = (list(devices), list(split_sizes), m_starts, slice_done_events)

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
    """
    Simpler synchronized multi-GPU tile quantization path.

    This variant explicitly copies tile chunks to each device, synchronizes around the work, then gathers results
    back to the first device. It is easier to reason about than the pinned asynchronous path but offers less overlap.
    """
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


def save_failed_cholesky(H: torch.Tensor, quant_args: dict, debug_info: dict | None):
    """Dump a Hessian that failed to decompose, before any retry damping is added, so the
    failure can be studied offline. H arrives sign-flipped and Hadamard-transformed with the
    base sigma_reg damping applied; su (in the dump) and the block Hadamard are orthogonal, so
    the captured matrix is exactly recoverable by applying the inverse transforms."""
    debug_dir = quant_args.get("debug_dir")
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok = True)
        key = (debug_info or {}).get("key") or "unknown"
        path = os.path.join(debug_dir, f"cholesky_fail_{key}.pt")
        torch.save({
            "H": H.detach().cpu().clone(),
            "sigma_reg": quant_args.get("sigma_reg", 0.025),
            **{k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
               for k, v in (debug_info or {}).items()},
        }, path)
        print(f" !! Saved failing Hessian to {path}")
    except Exception as e:
        print(f" !! Failed to save Hessian debug dump: {e}")


def block_ldl(H: torch.Tensor, b: int, quant_args: dict, verbose: bool, debug_info: dict | None = None):

    n, _ = H.shape
    assert (n % b == 0)
    m = n // b

    # Cholesky factorization: H = L @ L.T
    # Try on GPU first
    num_cholesky_retries = 0
    retry_cpu = False
    while True:
        try:
            L = torch.linalg.cholesky(H)
            # H is not needed after this, move to CPU. Then overwrite H's GPU storage with L, since we can't otherwise
            # free up that VRAM as the tensor is referenced by the parent frame
            H_cpu = H.cpu()
            H.copy_(L)  # VRAM copy, tiny overhead
            L = H
            H = H_cpu
            break

        except torch._C._LinAlgError as e:
            num_cholesky_retries += 1
            if num_cholesky_retries == 1:
                # Capture the matrix as it first failed (retry damping has never actually
                # recovered a failing decomposition, so the initial state is the interesting one)
                save_failed_cholesky(H, quant_args, debug_info)
            if num_cholesky_retries > 10:
                print(" ## Cholesky decomp. failed, number of retries exceeded")
                raise e
            print(f" !! Cholesky decomp. failed, increasing diagonal damping, attempt {num_cholesky_retries}/10")
            H.diagonal().add_(2.0 * quant_args.get("sigma_reg", 0.025) * H.diagonal().mean())
            continue

        # Fall back on CPU factorization
        except Exception as e:
            if e.__class__.__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e) or "HIP out of memory" in str(e):
                retry_cpu = True
                break
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


def fallback_quant(
    weight: torch.Tensor,
    q_device: torch.Tensor,
    quant_args: dict,
    pb: ProgressBar | None = None
):
    """
    Perform the same quantization as ldlq() but without an LDL decomposition

    :param weight:
        Input weights, shape (k, n). If device is "cpu", result is collected on CPU as well, saving a bunch of
        VRAM but adding a little PCIe overhead and many sync points

    :param q_device:
        Target device

    :param quant_args:
        dict:
         - K: bitrate
         - buf_size_k: buffer size for faux-LDLQ, along k

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
        device = weight.device
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
        weight_q = torch.zeros((size_k, size_n), dtype = torch.float, device = buffer_device)
        encoded = torch.zeros((tiles_k, tiles_n, 256), dtype = torch.short, device = buffer_device)

        # Accumulate sum of squared error on-device to avoid per-block CPU syncs
        mse_sum = torch.zeros((), dtype=torch.float64, device=device)

        for j in range(size_k, 0, -buf_size_k):
            i = j - buf_size_k

            # Current span is rows i:j
            b_weight = weight[i:j].to(device)
            b_weight_q = weight_q[i:j] if device == buffer_device else \
                torch.zeros_like(weight_q[i:j], device = device)
            b_encoded = encoded[i // 16 : j // 16] if device == buffer_device else \
                torch.zeros_like(encoded[i // 16 : j // 16], device = device)

            # Iterate over rows of blocks in current span
            for bj in range(buf_size_k, 0, -16):
                bi = bj - 16

                # Input tiles for quantization
                rows = b_weight[bi:bj]
                tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)

                # Pre-permute to tensor core layout
                tiles = tiles[:, tensor_core_perm(device)]

                # Quantize
                quant_w, quant_i = quantize_tiles_multigpu(tiles, quant_args)

                # Undo permutation on reconstructed tiles, but keep indices in tensor core layout
                quant_w = quant_w[:, tensor_core_perm_i(device)]

                # Restore row-major block layout
                quant_w = quant_w.reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, size_n)

                # Accumulate squared error for this block
                mse_sum += torch.sum(
                    (rows.float() - quant_w.float()).pow(2),
                    dtype=torch.float64,
                )

                # Store result
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

        for device in devices:
            torch.cuda.synchronize(device)

        mse = (mse_sum / weight.numel()).item()

    return weight_q, encoded, mse


def block_trace(A, B, block_size = 1024):
    """
    Compute trace(A.T @ B @ A) in column blocks of B to bound the temporary
    """
    total = 0.0
    for j_start in range(0, B.shape[1], block_size):
        j_end = min(j_start + block_size, B.shape[1])
        B_block = B[:, j_start:j_end]
        A_j_block = A[j_start:j_end, :]
        partial = torch.einsum("ik,ij,jk->", A, B_block, A_j_block)
        total += partial.item()
    return total


def ldlq_batched(
    weights: torch.Tensor,
    Ls: torch.Tensor,
    quant_args: dict,
    pb: ProgressBar | None = None
):
    """
    LDLQ over a stack of same-shape tensors with per-tensor L, batched so each 16-row step quantizes all
    tensors' tiles in one quantize_tiles call. The recursion is identical to ldlq() per batch element; only
    the matmuls become bmm. All buffers stay on-device (this path is only used for groups of small tensors).

    :param weights:
        Input weights, shape (B, k, n), float, on the quant device

    :param Ls:
        LDL decompositions of the regularized Hessians, shape (B, k, k), same device

    :return:
        tuple:
         - quantized weights, shape (B, k, n)
         - indices (unpacked), shape (B, k // 16, n // 16, 256), int16
    """
    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):

        device = weights.device
        assert device == torch.device(devices[0]) and Ls.device == device
        B, size_k, size_n = weights.shape
        assert Ls.shape == (B, size_k, size_k)
        assert size_k % 16 == 0
        assert size_n % 128 == 0
        tiles_k = size_k // 16
        tiles_n = size_n // 16

        buf_size_k = max(quant_args.get("buf_size_k", 128), 16)
        assert buf_size_k % 16 == 0
        assert size_k % buf_size_k == 0

        p_row = 0
        perm = tensor_core_perm(device)
        perm_i = tensor_core_perm_i(device)

        prod_cache = torch.zeros((B, size_k, size_n), dtype = torch.float, device = device)
        weight_q = torch.zeros_like(weights)
        encoded = torch.zeros((B, tiles_k, tiles_n, 256), dtype = torch.short, device = device)

        for j in range(size_k, 0, -buf_size_k):
            i = j - buf_size_k

            for bj in range(buf_size_k, 0, -16):
                bi = bj - 16
                gi, gj = i + bi, i + bj

                # Error so far for the remaining rows of the current span
                bb_err = weights[:, gj:j] - weight_q[:, gj:j]

                # Corresponding slice of the LDL decompositions
                bb_L = Ls[:, gj:j, gi:gj]

                # Input tiles for quantization; out-of-place equivalent of ldlq()'s in-place accumulation
                # (the mutated prod_cache rows are never read again there)
                compensation = torch.baddbmm(prod_cache[:, gi:gj], bb_L.transpose(1, 2), bb_err)
                rows = weights[:, gi:gj] + compensation

                tiles = rows.reshape(B, 16, tiles_n, 16).permute(0, 2, 1, 3).reshape(B * tiles_n, 256)
                tiles = tiles[:, perm]

                quant_w, quant_i = quantize_tiles_multigpu(tiles, quant_args)

                quant_w = quant_w[:, perm_i]
                quant_w = quant_w.reshape(B, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(B, 16, size_n)
                weight_q[:, gi:gj] = quant_w
                encoded[:, gi // 16] = quant_i.view(B, tiles_n, 256)

                if pb:
                    p_row += 1
                    pb.update(p_row)

            # Cache error term for the rest of the matrices
            b_err = weights[:, i:j] - weight_q[:, i:j]
            prod_cache.baddbmm_(Ls[:, i:j].transpose(1, 2), b_err)

        for device in devices:
            torch.cuda.synchronize(device)

    return weight_q, encoded


finalize_capture_H_mutex = threading.Lock()

def finalize_capture_H(H_data: dict, quant_args: dict, verbose: bool):
    with finalize_capture_H_mutex:

        if H_data["H"].is_meta:
            H_data["L"] = None
            H_data["finalized"] = True
            H_data["diag"] = None
            H_data["q_fallback"] = True

            H = H_data["H"]
            k = H.shape[0]
            su = (torch.randn(k, device = H_data["device"]).sign() + 1e-5).sign().to(torch.float).unsqueeze(1)
            H_data["su"] = su

            return True, None, None, su, None

        # Unswap H
        if "H_swap_device" in H_data:
            H_data["H"] = H_data["H"].to(H_data["H_swap_device"])
            del H_data["H_swap_device"]

        H = H_data["H"]
        if H_data["finalized"]:
            return H_data["q_fallback"], H, H_data["L"], H_data["su"], H_data["diag"]

        # Mean of samples summed up during forward pass
        # Switch to uncalibrated fallback if no input activations or diagonal is too small (few activations)
        count = H_data["count"]
        if count == 0:
            q_fallback = True
            diag_mean = 0.0
        else:
            H /= count
            diag_mean = torch.diag(H).mean().item()
            # A non-finite Hessian (e.g. fp16 activation overflow reaching the capture) cannot
            # be repaired by damping, and NaN would also defeat the < comparison below and turn
            # the whole diagonal non-finite through the regularization term
            if not math.isfinite(diag_mean):
                inf_nan = H_data.get("inf_nan")
                counts = f" (captured {inf_nan[0].item():,} inf, {inf_nan[1].item():,} NaN activation values)" \
                    if inf_nan is not None else ""
                print(f" !! Non-finite Hessian for {H_data.get('first_key')}, using uncalibrated fallback{counts}")
                q_fallback = True
                diag_mean = 0.0
            else:
                q_fallback = diag_mean < 1e-20

        # Regularize diagonal
        H.diagonal().add_(quant_args.get("sigma_reg", 0.025) * diag_mean)

        # Some tests
        diag = H.diagonal().clone()

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
        if q_fallback:
            L = None
        else:
            L, H = block_ldl(H, 16, quant_args, verbose, debug_info = {
                "key": H_data.get("first_key"),
                "count": H_data.get("count"),
                "num_total": H_data.get("num_total"),
                "inf_nan": H_data.get("inf_nan"),
                "su": su,
            })
            dr = torch.arange(k)
            L[dr, dr] = 0

        H_data["L"] = L

        # H is no longer needed except to compute proxy error so move to CPU
        H = H.cpu()
        H_data["H"] = H.cpu()

        H_data["finalized"] = True
        H_data["diag"] = diag
        H_data["q_fallback"] = q_fallback
        return q_fallback, H, L, su, diag


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


def sample_scale_tiles(weight_r: torch.Tensor, width: int = 3) -> torch.Tensor:
    """
    Sample tiles for the global scale search: a wrapped diagonal, guaranteeing every tile row and column is
    sampled at least once (outliers are typically whole input or output channels), plus the tiles with the
    highest and lowest RMS as explicit outlier insurance for anything a diagonal of the given width misses.
    """
    device = weight_r.device
    tiles_k = weight_r.shape[0] // 16
    tiles_n = weight_r.shape[1] // 16
    w4 = weight_r.view(tiles_k, 16, tiles_n, 16)

    diag_len = max(tiles_k, tiles_n)
    ii = torch.arange(diag_len, device = device).repeat_interleave(width)
    ww = torch.arange(width, device = device).repeat(diag_len)
    kk = ii % tiles_k
    nn = (ii + ww) % tiles_n

    # Tiles with extreme RMS, by flat tile index
    num_x = max(8, (diag_len * width) // 16)
    tile_ms = w4.square().mean(dim = (1, 3)).flatten()
    num_x = min(num_x, (tile_ms.shape[0] + 1) // 2)
    hi = torch.topk(tile_ms, num_x).indices
    lo = torch.topk(tile_ms, num_x, largest = False).indices
    xk = torch.cat((hi, lo)) // tiles_n
    xn = torch.cat((hi, lo)) % tiles_n

    tiles = w4[torch.cat((kk, xk)), :, torch.cat((nn, xn)), :].reshape(-1, 256)
    return tiles[:, tensor_core_perm(device)].contiguous()


def g_scale_search_batch(
    samples: list[torch.Tensor],
    quant_args: dict,
) -> list[tuple[float, torch.Tensor]]:
    """
    Global scale search over a group of tensors' tile samples, batching all evaluations into as few
    quantize_tiles calls as possible (two host syncs per group, regardless of group size).

    The sampled error curve is flat near the optimum but not perfectly unimodal, so a sequential golden-section
    search can converge on a local wiggle well away from the true minimum. Instead: a coarse grid over the full
    range on a subsample of the tiles, then a fine grid around the coarse minimum on the full sample, refined by
    parabolic interpolation; the curve's flatness makes the grid spacing precise enough.
    """
    n_t = len(samples)
    device = samples[0].device
    max_tiles = 65536

    def eval_pairs(pairs, tile_sets):
        # pairs: list of (tensor_idx, scale); one mse per pair, kernel calls chunked to the batch limit
        out = torch.empty(len(pairs), dtype = torch.float, device = device)
        i = 0
        while i < len(pairs):
            j, tot = i, 0
            while j < len(pairs) and tot + tile_sets[pairs[j][0]].shape[0] <= max_tiles:
                tot += tile_sets[pairs[j][0]].shape[0]
                j += 1
            j = max(j, i + 1)
            batch = torch.cat([tile_sets[t] * s for t, s in pairs[i:j]])
            quant_w, _ = quantize_tiles_multigpu(batch, quant_args)
            offset = 0
            for p, (t, s) in enumerate(pairs[i:j]):
                cnt = tile_sets[t].shape[0]
                out[i + p] = (quant_w[offset : offset + cnt] / s - tile_sets[t]).square().mean()
                offset += cnt
            i = j
        return out

    # Stage 1: coarse grid over the full search range, on a subsample of each tensor's tiles
    coarse = [0.1 + 0.2 * i for i in range(10)]  # 0.1 .. 1.9, same range as the old golden-section bracket
    subs = [s[::3] for s in samples]
    pairs1 = [(t, s) for t in range(n_t) for s in coarse]
    mse1 = eval_pairs(pairs1, subs).view(n_t, len(coarse))
    centers = [coarse[c] for c in mse1.argmin(dim = 1).tolist()]

    # Stage 2: fine grid around each tensor's coarse minimum, on its full sample
    step = 0.075
    fine = [[c + step * (i - 2) for i in range(5)] for c in centers]
    pairs2 = [(t, s) for t in range(n_t) for s in fine[t]]
    mse2 = eval_pairs(pairs2, samples).view(n_t, 5)
    mse2_h = mse2.tolist()

    results = []
    for t in range(n_t):
        best = min(range(5), key = lambda i: mse2_h[t][i])
        if 0 < best < 4:
            y0, y1, y2 = mse2_h[t][best - 1], mse2_h[t][best], mse2_h[t][best + 1]
            denom = y0 - 2.0 * y1 + y2
            offset = 0.5 * (y0 - y2) / denom if denom > 0 else 0.0
            offset = max(-0.5, min(0.5, offset))
        else:
            offset = 0.0
        best_scale = max(fine[t][best] + offset * step, 0.01)
        results.append((best_scale, mse2[t, best]))
    return results


def g_scale_gss(
    weight_r: torch.Tensor,
    verbose: bool,
    quant_args: dict,
    width: int = 3,
    pb: ProgressBar = None
):
    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)

    main_stream = get_quant_stream(devices[0])
    # TODO: Figure out why Torch always initializes cuda:0 when exiting this CM, even when it's not used
    with torch.cuda.stream(main_stream):
        tiles = sample_scale_tiles(weight_r, width)
        if pb:
            pb.update(50)
        best_scale, best_mse = g_scale_search_batch([tiles], quant_args)[0]
        if pb:
            pb.update(100)
        if verbose:
            print(f"     - scale search: min = {best_scale:.6f}, mse: {best_mse.item():.6f}")

    for device in devices:
        torch.cuda.synchronize(device)

    return best_scale, best_mse


def block_rms(x: torch.Tensor, dim: int, keepdim: bool = False, blocksize: int = 32):
    """
    Compute blockwise x.square().mean(dim, keepdim).sqrt()
    """
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
    """
    Compute blockwise x.square().mean().sqrt()
    """
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
    """
    Compute blockwise (x - y).square().mean().item() / y.square().mean().item()
    """
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
    skip_g_scale: bool = False,
    q_fallback: bool = False
):
    """
    Transform weights into the distribution expected by EXL3 tile quantization.

    The routine chooses whether to apply output-channel scaling, folds output and input sign/scale vectors into the
    matrix, applies blockwise Hadamard transforms, and optionally searches for a global scale that minimizes sample
    quantization error. It returns the transformed weight plus the scale metadata needed to reconstruct the original
    linear layer behavior after quantization.
    """
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
    if not q_fallback and H_diag is not None:
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

    if q_fallback:
        apply_out_scales = force_out_scales

    # Apply output scales
    out_channel_scales = block_rms(weight, dim = 0, keepdim = True)
    mean = out_channel_scales.mean().item()
    if mean > 1e-30:
        out_channel_scales /= mean
        quant_args["zeros"] = False
    else:
        quant_args["zeros"] = True
        if force_out_scales is not None:
            apply_out_scales = True
    zero_out_scales = out_channel_scales.abs() < 1e-30

    if apply_out_scales:
        out_channel_scales[zero_out_scales] = 0.1
        sv = (sv * out_channel_scales + 1e-10).float()
        if verbose:
            out_channel_std = out_channel_scales.std().item()
            out_channel_mean = out_channel_scales.mean().item()
            print(f"     - out ch scales std/mean: {out_channel_std:.6f}   {out_channel_mean:.6f}")

    # Output sign flips (and scales)
    weight /= sv

    # Force zero output channels to zero
    sv[zero_out_scales] = 0.0

    # Output hadamard transform
    blockwise_preapply_had_r_(weight, had_n)

    # Input sign flips and scales
    in_channel_scales = block_rms(weight, dim = 1, keepdim = True)
    in_channel_scales[in_channel_scales.abs() < 1e-30] = 0.1
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
        q_fallback, H, L, su, H_diag = finalize_capture_H(H_data, quant_args, verbose)
        if H is not None and H.is_cuda:
            H = H.to(device)
        if L is not None and L.is_cuda:
            L = L.to(device)
        if su.is_cuda:
            su = su.to(device)
        if H_diag is not None and H_diag.is_cuda:
            H_diag = H_diag.to(device)
        sv = (torch.randn(n, device = device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)

        # Move stored L to CPU (if not already), move working L to device
        if H_data["L"] is not None:
            H_data["L"] = H_data["L"].cpu()
        if L is not None:
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
            pb,
            q_fallback = q_fallback
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
        if not q_fallback:
            weight_q, encoded_q = ldlq(weight_r, L, quant_args, pb)  #zxc
            del L
        else:
            weight_q, encoded_q, mse_err = fallback_quant(weight_r, device, quant_args, pb)  # zxc

        pb.update(tiles_k)

        # Metrics
        if not q_fallback:
            try:
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
        else:
            proxy_err = mse_err

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
            "q_fallback": q_fallback,
        })

    return weight_q, proxy_err, out_tensors


def quantize_exl3_batch(
    weights: list[torch.Tensor],
    H_datas: list[dict],
    quant_args_list: list[dict],
    progress_str: str | None = None,
    verbose: bool = False,
):
    """
    Quantize a group of same-K linears together, batching the scale search and the LDLQ recursion so the
    quantize_tiles kernel sees large batches instead of one small tensor's worth of tiles per step.

    Two layouts, chosen by whether the H_data dicts are the same object:
      - shared Hessian (e.g. all gate/up projections of a block-sparse MLP, or fused q/k/v): the regularized
        weights are concatenated along out_features and run through a single ldlq() pass with the shared L.
        This is exact: LDLQ treats columns independently given L, and su (drawn per qmap) is identical
      - distinct Hessians over same-shape tensors (e.g. per-expert down projections): weights are stacked and
        run through ldlq_batched() with per-tensor L

    RNG is re-seeded per tensor (finalize's su draw on first call for the qmap, then the per-tensor sv draw),
    so per-tensor results should match quantize_exl3() in the wider matmuls. Tensors whose Hessian fails to
    finalize (q_fallback) are routed through quantize_exl3() individually.

    :return:
        list of (proxy_err, out_tensors) per input tensor; quant_args_list entries are updated in place like
        quantize_exl3 updates quant_args
    """
    n_t = len(weights)
    qa0 = quant_args_list[0]
    devices = qa0["devices"]
    device = torch.device(devices[0])
    shared_H = all(hd is H_datas[0] for hd in H_datas)
    assert shared_H or all(w.shape == weights[0].shape for w in weights)
    assert all(w.shape[0] == weights[0].shape[0] for w in weights)
    assert all(qa["K"] == qa0["K"] for qa in quant_args_list)

    size_k = weights[0].shape[0]
    tiles_k = size_k // 16
    results: list = [None] * n_t

    progress_text = None if not progress_str else progress_str.replace("<step>", "Preparing")
    with ProgressBar(progress_text, 100) as pb:

        # Finalize Hessians, replicating the serial path's per-tensor RNG stream. Fallback tensors are
        # handled individually by quantize_exl3
        finalized = []
        batch_idx = []
        for t in range(n_t):
            qa = quant_args_list[t]
            if "seed" in qa:
                torch.manual_seed(qa["seed"])
            q_fallback, H, L, su, H_diag = finalize_capture_H(H_datas[t], qa, verbose)
            if q_fallback:
                finalized.append(None)
            else:
                finalized.append((H, L, su, H_diag))
                batch_idx.append(t)

        for t in range(n_t):
            if finalized[t] is None:
                results[t] = quantize_exl3(weights[t], H_datas[t], quant_args_list[t], False, None, verbose)[1:]

        if not batch_idx:
            return results

        # Regularize each tensor with the scale search deferred, then search all scales in one batch
        regs = {}
        for t in batch_idx:
            qa = quant_args_list[t]
            if "seed" in qa:
                torch.manual_seed(qa["seed"])
            H, L, su, H_diag = finalized[t]
            weight = weights[t]
            if weight.device != device:
                weight = weight.to(device)
            if su.is_cuda:
                su = su.to(device)
            if H_diag is not None and H_diag.is_cuda:
                H_diag = H_diag.to(device)
            sv = (torch.randn(weight.shape[1], device = device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)
            apply_out_scales, weight_r, _, su, sv = regularize(
                weight, su, sv, qa, verbose, H_diag, None, skip_g_scale = True)
            regs[t] = [weight_r, su, sv, apply_out_scales]
            weights[t] = None

        samples = [sample_scale_tiles(regs[t][0]) for t in batch_idx]
        scales = g_scale_search_batch(samples, qa0)
        del samples
        g_scales = {}
        for t, (g_scale, _) in zip(batch_idx, scales):
            regs[t][0] *= g_scale
            regs[t][1] /= g_scale
            g_scales[t] = g_scale
        pb.update(100)

        progress_text = None if not progress_str else progress_str.replace("<step>", "Quantizing")
        pb.new_task(progress_text, tiles_k)

        # Quantize
        if shared_H:
            L = finalized[batch_idx[0]][1].to(device)
            widths = [regs[t][0].shape[1] for t in batch_idx]
            weight_r_cat = torch.cat([regs[t][0] for t in batch_idx], dim = 1)
            for t in batch_idx:
                regs[t][0] = None
            weight_q_cat, encoded_cat = ldlq(weight_r_cat, L, qa0, pb)
            del L
            weight_rs = list(torch.split(weight_r_cat, widths, dim = 1))
            weight_qs = list(torch.split(weight_q_cat, widths, dim = 1))
            encodeds = list(torch.split(encoded_cat, [w // 16 for w in widths], dim = 1))
        else:
            Ls = torch.stack([finalized[t][1].to(device) for t in batch_idx])
            weight_r_stack = torch.stack([regs[t][0] for t in batch_idx])
            for t in batch_idx:
                regs[t][0] = None
            weight_q_stack, encoded_stack = ldlq_batched(weight_r_stack, Ls, qa0, pb)
            del Ls
            weight_rs = list(weight_r_stack.unbind(0))
            weight_qs = list(weight_q_stack.unbind(0))
            encodeds = list(encoded_stack.unbind(0))

        pb.update(tiles_k)

        # Per-tensor metrics and packing
        Hd = None
        for bi, t in enumerate(batch_idx):
            qa = quant_args_list[t]
            _, su, sv, apply_out_scales = regs[t]
            if shared_H:
                if Hd is None:
                    Hd = finalized[t][0].to(device)
            else:
                Hd = finalized[t][0].to(device)
            try:
                E = weight_rs[bi] - weight_qs[bi]
                num = block_trace(E, Hd)
                E = None
                den = block_trace(weight_rs[bi], Hd)
                proxy_err = num / max(den, 1e-8)
            except torch.OutOfMemoryError:
                E = None
                proxy_err = -1.0
            weight_rs[bi] = None
            weight_qs[bi] = None

            suh = su.flatten().contiguous().to(dtype = torch.half, copy = True)
            svh = sv.flatten().contiguous().to(dtype = torch.half, copy = True)
            trellis = pack_trellis(encodeds[bi].contiguous(), qa)
            encodeds[bi] = None

            out_tensors = {
                "suh": suh,
                "svh": svh,
                "trellis": trellis,
            }
            if qa.get("mcg"):
                out_tensors.update({
                    "mcg": torch.tensor(codebook_mcg_mult, dtype = torch.uint32).view(torch.int)
                })
            if qa.get("mul1"):
                out_tensors.update({
                    "mul1": torch.tensor(codebook_mul1_mult, dtype = torch.uint32).view(torch.int)
                })

            qa.update({
                "apply_out_scales": apply_out_scales,
                "g_scale": g_scales[t],
                "q_fallback": False,
            })
            results[t] = (proxy_err, out_tensors)

    return results
