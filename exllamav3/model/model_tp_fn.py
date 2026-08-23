import torch
import traceback
import os
import signal
import sys
import threading
import time
from collections import deque
from .model_tp_shared import SMProducer, SMConsumer
from ..ext import exllamav3_ext as ext
from functools import lru_cache
from .model_tp_backend import TPBackendNCCL, TPBackendNative
from ..tokenizer.mm_embedding import recv_embeddings
from ..util import log_tp, set_t0

_no_fwd_barrier = os.environ.get("EXL3_TP_NO_FWD_BARRIER", "1") != "0"


from ..util.misc import install_parent_death_signal


def init_pg(device: int, active_devices: list[int], output_device: int, backend_args: dict, master: bool = False):
    rank = active_devices.index(device) if device >= 0 else -1
    output_rank = active_devices.index(output_device)
    world_size = len(active_devices)
    local_context = {
        "device": device,
        "modules": [],
        "kv_modules": [],
        "recurrent_modules": [],
        "recurrent_cache": {},
        "cpu_page_cache": None,
        "rank": rank,
        "world_size": world_size,
        "output_rank": output_rank,
        "active_devices": active_devices,
        "output_device": output_device,
    }

    torch.cuda.set_device(device)

    match backend_args["type"]:
        case "nccl":
            backend = TPBackendNCCL(
                device = device,
                active_devices = active_devices,
                output_device = output_device,
                init_method = backend_args["init_method"],
                master = master,
                uuid = backend_args["uuid"],
            )
        case "native":
            backend = TPBackendNative(
                device = device,
                active_devices = active_devices,
                output_device = output_device,
                init_method = backend_args["init_method"],  ##
                master = master,
                uuid = backend_args["uuid"],
                cpu = device < 0
            )
        case _:
            raise ValueError("Unknown backend type")

    local_context["backend"] = backend
    return local_context


def mp_model_worker(
    conn,
    device: int,
    active_devices: list[int],
    output_device: int,
    backend_args: dict,
    producer: dict,
    dbg_t0_: float
):
    # Terminal Ctrl-C is delivered to the whole foreground process group; shutdown is
    # orchestrated by the parent ("quit" command) or the kernel (PDEATHSIG), never by SIGINT
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    set_t0("TP", dbg_t0_)
    log_tp(device, f"Child process launched")
    if install_parent_death_signal():
        log_tp(device, f"Installed parent death signal")

    # EXL3_TP_SPIN_RECV=<ms>: after finishing a command, poll the pipe hot for this many ms before
    # falling back to a blocking recv. A blocking recv pays scheduler wake latency (tens to hundreds
    # of us depending on C-states) at the start of every forward pass; during decode the next command
    # arrives within a few ms of the previous ack, so a short spin window catches it with zero wake
    # cost, at the price of one busy core per rank for the window
    spin_recv_s = float(os.environ.get("EXL3_TP_SPIN_RECV", "0")) / 1e3

    with torch.inference_mode():
        local_context = init_pg(device, active_devices, output_device, backend_args)
        local_context["inf_consumer"] = SMConsumer(producer, device = device, pin_memory = True)

        # Dispatch loop
        while True:
            if spin_recv_s > 0 and not conn.poll(0):
                deadline = time.monotonic() + spin_recv_s
                while not conn.poll(0) and time.monotonic() < deadline:
                    pass
            msg = conn.recv()
            if msg == "quit":
                log_tp(device, f"Child worker exiting")
                torch.cuda.synchronize()
                local_context["inf_consumer"].close()
                local_context["backend"].close()
                break
            func, args = msg
            try:
                result = func(local_context, *args)
                conn.send(result)
            except Exception as e:
                tb = traceback.TracebackException.from_exception(e)
                print("-" * 40)
                print(" ## Exception in child process")
                print("".join(tb.format()))
                print("-" * 40)
                conn.send(e)


def mp_cpu_reduce(local_context: dict):
    backend = local_context["backend"]
    backend.run_cpu_reduce_jobs()


def mp_set_plan(local_context: dict, plan: dict, active_devices: list):
    """
    Used by TP loader, send (potentially large) plan dict once to avoid pickling with every message while loading
    the model
    """
    local_context["plan"] = plan
    local_context["active_devices"] = active_devices


def mp_set_consumer(local_context: dict, producer_exp: SMProducer | dict):
    """
    Used by TP loader
    """
    local_context["consumer"] = SMConsumer(
        producer_imp = producer_exp,
        device = local_context["device"],
        pin_memory = False
    )


def mp_close_consumer(local_context: dict):
    """
    Used by TP loader
    """
    local_context["consumer"].close()
    del local_context["consumer"]


def mp_model_append(local_context: dict, exported: dict):
    """
    Used by TP loader, append a partial module to the process's module list
    """
    modules = local_context["modules"]
    kv_modules = local_context["kv_modules"]
    recurrent_modules = local_context["recurrent_modules"]
    device = local_context["device"]
    cls = exported["cls"]
    plan = local_context["plan"]

    module = cls.tp_import(local_context, exported, plan[device])
    modules.append(module)
    kv_modules += module.all_cache_modules()
    recurrent_modules += module.all_recurrent_modules()
    if module.caps.get("logits_output"):
        local_context["logits_module"] = module
    return None


def mp_model_append_gather(local_context: dict):
    """
    Used by TP loader, append final logit gather module to the module list
    """
    from ..modules.gather import OutputGather
    modules = local_context["modules"]
    device = local_context["device"]
    plan = local_context["plan"]
    active_devices = local_context["active_devices"]
    output_device = local_context["output_device"]

    last_key = modules[-1].key
    gather_devices = []
    ldims = []
    for i, i_device in enumerate(sorted(active_devices)):
        ldim = plan[i_device][last_key][1] - plan[i_device][last_key][0]
        if ldim > 0 or i_device == output_device:
            gather_devices.append(i_device)
            ldims.append(ldim)

    module = OutputGather(
        config = None,
        key = "output_gather",
        device = device,
        output_device = output_device,
        gather_devices = gather_devices,
        ldims = ldims,
    )

    modules.append(module)
    return None


def mp_model_forward(
    local_context: dict,
    shared_input: dict,
    params: dict,
    last_kv_module_idx: int,
    prefill: bool,
    single_idx: int,
):
    """
    Forward pass for parallel slice of a model
    """
    backend = local_context["backend"]
    # The pass-start barrier aligns all rank streams before the first collective. The collectives
    # are individually ordered by their stage counters, so this is not required for correctness;
    # EXL3_TP_NO_FWD_BARRIER=1 skips it (experimental) to save one spin kernel per rank per pass
    if not _no_fwd_barrier:
        backend.fwd_barrier()

    modules = local_context["modules"] if single_idx is None else [local_context["modules"][single_idx]]
    consumer = local_context["inf_consumer"]

    for tensor_param in [
        "block_table",
        "cache_seqlens",
        "positions",
        "position_ids",
        "recurrent_slots",
        "inv_freq",
        "input_ids",     # hash-MoE routing (DeepSeek-V4 bootstrap layers)
    ]:
        p = params.get(tensor_param)
        if p is not None:
            params[tensor_param] = consumer.recv(p, cuda = True)

    p = params.get("indexed_embeddings")
    if p is not None:
        params["indexed_embeddings"] = recv_embeddings(consumer, p)

    params["backend"] = backend

    x = consumer.recv(shared_input)

    for idx, module in enumerate(modules):
        logits_layer = module.caps.get("logits_output")
        if logits_layer and (num := params.get("last_tokens_only")):
            x = x[..., -num:, :].contiguous()
        if prefill:
            params["prefill"] = (idx == last_kv_module_idx)
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
        if prefill and idx == last_kv_module_idx:
            backend.end_cpu_reduce_jobs()
            del params["prefill"]
            return None

    backend.end_cpu_reduce_jobs()
    return x


def mp_model_forward_embedding(
    local_context: dict,
    shared_input: dict,
    params: dict,
):
    consumer = local_context["inf_consumer"]
    module = local_context["modules"][0]
    x = consumer.recv(shared_input)
    x = module.forward(x, params)
    return x


def mp_model_forward_lm_head_argmax(
    local_context: dict,
    shared_input: dict,
    params: dict,
    offset: int,
    gather_devices: list[int] | None,
    ldims: list[int] | None,
):
    consumer = local_context["inf_consumer"]
    device = local_context["device"]
    output_device = local_context["output_device"]
    backend = local_context["backend"]

    x = consumer.recv(shared_input)

    if offset >= 0:
        module = local_context["logits_module"]
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
        v, i = x.max(dim = -1)
        i += offset
    else:
        v = torch.empty(*x.shape[:-1], dtype = x.dtype, device = x.device)
        i = torch.empty(*x.shape[:-1], dtype = torch.long, device = x.device)

    if gather_devices is None:
        return v, i

    v_dim = 1 if offset >= 0 else 0
    i_dim = 1 if offset >= 0 else 0
    vp = torch.empty(*v.shape, v_dim, dtype = v.dtype, device = v.device)
    ip = torch.empty(*i.shape, i_dim, dtype = i.dtype, device = i.device)
    if offset >= 0:
        vp[..., 0] = v
        ip[..., 0] = i

    if device == output_device:
        out_v_shape = list(vp.shape)
        out_i_shape = list(ip.shape)
        out_v_shape[-1] = sum(ldims)
        out_i_shape[-1] = sum(ldims)
        out_v = torch.empty(*out_v_shape, dtype = vp.dtype, device = vp.device)
        out_i = torch.empty(*out_i_shape, dtype = ip.dtype, device = ip.device)
    else:
        out_v = None
        out_i = None

    backend.gather_small(vp, out_v, gather_devices, output_device, ldims)
    backend.gather_small(ip, out_i, gather_devices, output_device, ldims)

    return (out_v, out_i) if device == output_device else None


# def mp_model_forward_lm_head_argmax_old(
#     local_context: dict,
#     shared_input: dict,
#     params: dict,
#     offset: int
# ):
#     consumer = local_context["inf_consumer"]
#     module = local_context["logits_module"]
#
#     x = consumer.recv(shared_input)
#     x = module.prepare_for_device(x, params)
#     x = module.forward(x, params)
#     v, i = x.max(dim = -1)
#     i += offset
#     return v, i


def mp_cache_page_copy(
    local_context: dict,
    cache_id: int,
    from_page: int,
    to_page: int,
    num_tokens: int
):
    """
    Copy (partial) cache page across all processes in a TP split cache
    """
    kv_modules = local_context["kv_modules"]
    for idx, module in enumerate(kv_modules):
        cache_layer = module.tp_cache_lookup[cache_id]
        cache_layer.copy_page(cache_layer, from_page, to_page, num_tokens)


def mp_rotate_cache_pages(
    local_context: dict,
    cache_id: int,
    all_rotations: dict,
):
    consumer = local_context["inf_consumer"]
    all_rotations = consumer.recv(all_rotations, cuda = True)
    kv_modules = local_context["kv_modules"]

    @lru_cache
    def get_buffer(shape, device, dtype):
        return torch.empty(shape, device = device, dtype = dtype)

    cache_tensors = []
    for idx, module in enumerate(kv_modules):
        cache_layer = module.tp_cache_lookup[cache_id]
        cache_tensors += cache_layer.get_tensors()

    for cache in cache_tensors:
        buffer = get_buffer(cache[0].shape, cache.device, cache.dtype)
        ext.cache_rotate(cache, all_rotations, buffer)


# CPU page cache in TP mode. The main process owns the page table, the slot table and the eviction policy but
# holds no cache tensors, so it names a slot by index and every rank keeps its own shard of that slot here. A
# slot is one whole page image of this rank's shard, across every attached cache, in ONE pinned slab with a
# view carved per cache tensor (never one allocation per tensor): a slot touches every layer's K and V, so
# per-tensor pinning is O(layers) cudaHostAlloc calls per slot, and across a real budget that overruns
# vm.max_map_count (~65k mappings by default) long before it runs out of memory, silently degrading the whole
# pool to pageable buffers.

class RankSlotPool:
    """
    This rank's half (or third, or...) of the CPU page cache: one pinned slab per slot index.

    Pinning host memory manages only ~2.5 GB/s and serializes with copy submission on the driver, so slabs
    are pinned ahead of demand by a background thread, mirroring what the single-process cache does. A store
    that outruns the thread pins synchronously and says so, since that stall lands on the generator's own
    dispatch path.
    """

    def __init__(self, cache_tensors: list, max_slots: int):
        # Segment layout of one slab, mirroring CPUPageCache._make_slab
        self.segments = []
        offset = 0
        for t in cache_tensors:
            nbytes = t[0].numel() * t.element_size()
            self.segments.append((offset, t.shape[1:], t.dtype, nbytes))
            offset = (offset + nbytes + 255) & ~255
        self.slab_size = (offset + 4095) & ~4095
        self.max_slots = max_slots
        self.slots = {}
        self.cold_allocs = 0
        self.pageable = False

        self._spare = deque()
        self._spare_cond = threading.Condition()
        self._alloc_thread = threading.Thread(target = self._alloc_worker, daemon = True)
        self._alloc_thread.start()


    def _make_buffers(self):
        try:
            slab = torch.empty((self.slab_size,), dtype = torch.uint8, pin_memory = True)
        except RuntimeError:
            # Out of lockable memory on this rank. Pageable buffers still work, at roughly half the transfer
            # bandwidth (and stores into them synchronize), which beats failing the store outright (but it
            # is a real degradation, so warn once)
            if not self.pageable:
                self.pageable = True
                print(" !! CPU page cache: rank out of pinnable memory, falling back to pageable buffers",
                      flush = True)
            slab = torch.empty((self.slab_size,), dtype = torch.uint8)
        return [slab[offset : offset + nbytes].view(dtype).view(shape)
                for offset, shape, dtype, nbytes in self.segments]


    def _alloc_worker(self):
        with torch.inference_mode():
            while True:
                with self._spare_cond:
                    while len(self.slots) + len(self._spare) >= self.max_slots:
                        self._spare_cond.wait()
                buffers = self._make_buffers()  # slow part, outside the lock
                with self._spare_cond:
                    self._spare.append(buffers)


    def get(self, slot: int):
        buffers = self.slots.get(slot)
        if buffers is not None:
            return buffers
        with self._spare_cond:
            if self._spare:
                buffers = self._spare.popleft()
                self.slots[slot] = buffers
                self._spare_cond.notify()
                return buffers
        buffers = self._make_buffers()
        self.cold_allocs += 1
        with self._spare_cond:
            self.slots[slot] = buffers
            self._spare_cond.notify()
        return buffers


def mp_cpu_cache_tensors(local_context: dict, cache_ids: list[int]):
    cache_tensors = []
    for cache_id in cache_ids:
        for module in local_context["kv_modules"]:
            cache_tensors += [t for t in module.tp_cache_lookup[cache_id].get_tensors() if t is not None]
    return cache_tensors


def mp_cpu_cache_init(local_context: dict, cache_ids: list[int], max_slots: int):
    """
    Size of one page of this rank's cache shard in bytes, and the slot pool to hold those pages. Called with
    max_slots = 0 to size the slot before the main process knows how many will fit.
    """
    cache_tensors = mp_cpu_cache_tensors(local_context, cache_ids)
    if max_slots:
        local_context["cpu_page_cache"] = RankSlotPool(cache_tensors, max_slots)
    return sum(t[0].numel() * t.element_size() for t in cache_tensors)


def mp_cpu_cache_store(local_context: dict, cache_ids: list[int], slot: int, page_index: int):
    """
    Copy this rank's shard of a page into the buffers held for slot (device-to-host, async on the current
    stream). Reusing a slot overwrites it in place, which is how the main process recycles an evicted entry.
    """
    pool = local_context["cpu_page_cache"]
    buffers = pool.get(slot)
    for buffer, tensor in zip(buffers, mp_cpu_cache_tensors(local_context, cache_ids)):
        buffer.copy_(tensor[page_index], non_blocking = True)
    return pool.cold_allocs


def mp_cpu_cache_fetch(local_context: dict, cache_ids: list[int], slot: int, page_index: int):
    """
    Copy a stored page back into this rank's shard of a page slot (host-to-device, async on the current stream)
    """
    buffers = local_context["cpu_page_cache"].slots[slot]
    for buffer, tensor in zip(buffers, mp_cpu_cache_tensors(local_context, cache_ids)):
        tensor[page_index].copy_(buffer, non_blocking = True)


class PseudoParentConn:
    """
    Standin for a Pipe to dispatch functions on the main device rather than a dedicated child process running
    `mp_model_worker`. This allows a tensor-parallel model to run partially in the main process, without additional
    IPC overhead when returning logits from forward(), and without needing two CUDA contexts on the main device.

    Unlike a real Pipe, send() executes the requested function synchronously. Callers therefore keep this
    output-device pseudo-worker last in TP fan-out order, so spawned workers are already running before the main
    process enters collectives or barriers on its own rank.
    """

    def __init__(
        self,
        device: int,
        active_devices: list[int],
        output_device: int,
        backend_args: dict,
        producer: SMProducer,
        dbg_t0_: float
    ):
        set_t0("TP", dbg_t0_)
        log_tp(None, f"Pseudoprocess created, device {device}")

        self.local_context = init_pg(device, active_devices, output_device, backend_args, master = True)
        self.local_context["inf_consumer"] = SMConsumer(producer, device = device, pin_memory = True)
        self.result = None
        self.device = device


    def send(self, msg):
        if msg == "quit":
            log_tp(self.device, f"Pseudoprocess worker quit message")
        else:
            fn, args = msg
            self.result = fn(self.local_context, *args)


    def poll(self, timeout):
        return True


    def recv(self):
        r = self.result
        self.result = None
        return r


    def close(self, *args, **kwargs):
        self.local_context["inf_consumer"].close()
        self.local_context = {}
        log_tp(self.device, f"Pseudoprocess closed")


    def quit(self):
        torch.cuda.synchronize()
        self.local_context["backend"].close()
        self.close()


class PseudoChildConn:
    def __init__(self):
        pass

    def close(self, *args, **kwargs):
        pass


class PseudoChild:
    def __init__(self):
        pass

    def is_alive(self):
        return True

    def join(self, *args, **kwargs):
        pass

    def terminate(self, *args, **kwargs):
        pass
