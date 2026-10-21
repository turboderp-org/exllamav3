import torch
import traceback
import os
import signal
import sys
import time
from .model_tp_shared import SMProducer, SMConsumer
from ..ext import exllamav3_ext as ext
from functools import lru_cache
from .model_tp_backend import TPBackendNCCL, TPBackendNative, TPBackendNull
from ..tokenizer.mm_embedding import recv_embeddings
from ..util import log_tp, set_t0

_no_fwd_barrier = os.environ.get("EXL3_TP_NO_FWD_BARRIER", "1") != "0"
_stream_hash_passes = int(os.environ.get("EXL3_TP_STREAM_HASH", "0") or "0")


from ..util.misc import install_parent_death_signal
from ..util.pinned_arena import PinnedArena
from ..constants import PAGE_SIZE


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
    # Gated-residual sites hand a weighted stream copy to their successor; rewire on each append
    from ..modules.hyperconnections import GatedResidual
    GatedResidual.link_sites(modules)
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
    # Warmup passes (Model.warmup) run every rank independently with collectives stubbed out
    warmup = bool(params.get("tp_warmup"))
    backend = TPBackendNull() if warmup else local_context["backend"]
    # The pass-start barrier aligns all rank streams before the first collective. The collectives
    # are individually ordered by their stage counters, so this is not required for correctness;
    # EXL3_TP_NO_FWD_BARRIER=1 skips it (experimental) to save one spin kernel per rank per pass
    if not _no_fwd_barrier and not warmup:
        backend.fwd_barrier()

    modules = local_context["modules"] if single_idx is None else [local_context["modules"][single_idx]]
    consumer = local_context["inf_consumer"]

    for tensor_param in [
        "block_table",
        "cache_seqlens",
        "positions",
        "position_ids",
        "recurrent_slots",
        "recurrent_slots_scan",
        "recurrent_slots_scan_in",
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

    # Modules that stage input-dependent data on a worker thread (PLE n-gram rows) go first, as
    # in the layer-split forward, so the gather overlaps the leading blocks
    if x.dtype == torch.long and single_idx is None:
        for module in modules:
            if module.caps.get("prefetch_ids"):
                module.prefetch(x, params)

    # EXL3_TP_STREAM_HASH: print a digest of the residual stream after every module on every rank
    # for the first forward passes, to locate where ranks stop agreeing bit for bit (debug)
    stream_hash = _stream_hash_passes > 0 and not warmup and local_context.get("_hash_passes", 0) < _stream_hash_passes
    if stream_hash:
        local_context["_hash_passes"] = local_context.get("_hash_passes", 0) + 1

    for idx, module in enumerate(modules):
        logits_layer = module.caps.get("logits_output")
        if logits_layer and (num := params.get("last_tokens_only")):
            x = x[..., -num:, :].contiguous()
        if prefill:
            params["prefill"] = (idx == last_kv_module_idx)
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
        if stream_hash and x is not None:
            import hashlib
            h = hashlib.sha1(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:12]
            print(f"stream-hash dev={local_context['device']} pass={local_context['_hash_passes']} {idx:3d} {module.key:<44} {h} {tuple(x.shape)} {x.dtype}", flush = True)
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
    vocab_size: int = -1,
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
        # The head is padded to the tile size; a shard reaching past the vocabulary must not let a
        # padding column (zero logit) win the local maximum
        if vocab_size >= 0 and offset + x.shape[-1] > vocab_size:
            x = x[..., :max(0, vocab_size - offset)]
        if x.shape[-1] > 0:
            v, i = x.max(dim = -1)
            i += offset
        else:
            v = torch.full(x.shape[:-1], -float("inf"), dtype = x.dtype, device = x.device)
            i = torch.zeros(x.shape[:-1], dtype = torch.long, device = x.device)
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


def mp_model_forward_lm_head_topk(
    local_context: dict,
    shared_input: dict,
    params: dict,
    offset: int,
    gather_devices: list[int] | None,
    ldims: list[int] | None,
    vocab_size: int,
    top_k: int,
    scale: float,
    softcap: float,
):
    """
    Per-row top-k of the logits through this rank's shard of the LM head, as (scores, vocabulary ids)
    k wide, gathered to the output device where the shards' candidates are merged (DFlash2's selector
    needs the top-k, not the logits). Same shard walk and padding crop as mp_model_forward_lm_head_argmax;
    the selector's scaling and softcap are applied here so the scores compare across shards. A shard
    narrower than k pads with -inf.
    """
    consumer = local_context["inf_consumer"]
    device = local_context["device"]
    output_device = local_context["output_device"]
    backend = local_context["backend"]

    x = consumer.recv(shared_input)

    if offset >= 0:
        module = local_context["logits_module"]
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
        if offset + x.shape[-1] > vocab_size:
            x = x[..., :max(0, vocab_size - offset)]
        x = x.float() * scale
        if softcap > 0.0:
            x = torch.tanh(x / softcap) * softcap
        k = min(top_k, x.shape[-1])
        if k > 0:
            v, i = torch.topk(x, k, dim = -1, sorted = False)
            i = i + offset
        else:
            v = torch.empty(*x.shape[:-1], 0, dtype = torch.float, device = x.device)
            i = torch.empty(*x.shape[:-1], 0, dtype = torch.long, device = x.device)
        if k < top_k:
            pad = top_k - k
            v = torch.cat([v, torch.full((*v.shape[:-1], pad), -float("inf"), dtype = v.dtype, device = v.device)], dim = -1)
            i = torch.cat([i, torch.zeros((*i.shape[:-1], pad), dtype = i.dtype, device = i.device)], dim = -1)
    else:
        v = torch.empty(*x.shape[:-1], 0, dtype = torch.float, device = torch.device("cuda", device))
        i = torch.empty(*x.shape[:-1], 0, dtype = torch.long, device = v.device)

    if gather_devices is None:
        return v, i

    if device == output_device:
        out_v = torch.empty(*v.shape[:-1], sum(ldims), dtype = v.dtype, device = v.device)
        out_i = torch.empty(*i.shape[:-1], sum(ldims), dtype = i.dtype, device = i.device)
    else:
        out_v = out_i = None

    backend.gather_small(v.contiguous(), out_v, gather_devices, output_device, ldims)
    backend.gather_small(i.contiguous(), out_i, gather_devices, output_device, ldims)
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
# view carved per cache tensor. The slabs are slices of one registered arena rather than pinned tensors of
# their own: torch's pinned allocator rounds each request up to a power of two, and per-tensor pinning is
# O(layers) mappings per slot, which across a real budget overruns vm.max_map_count (~65k by default) long
# before it runs out of memory.

class RankSlotPool:
    """
    This rank's half (or third, or...) of the CPU page cache: one pinned arena holding a slab per slot
    index, committed and locked when the tier is built.
    """

    def __init__(self, cache_tensors: list, max_slots: int):
        # Segment layout of one slab, mirroring CPUPageCache
        self.segments = []
        offset = 0
        for t in cache_tensors:
            nbytes = t[0].numel() * t.element_size()
            self.segments.append((offset, t.shape[1:], t.dtype, nbytes))
            offset = (offset + nbytes + 255) & ~255
        self.slab_size = (offset + 4095) & ~4095
        self.max_slots = max_slots
        self.slots = {}
        self.arena = PinnedArena(max_slots * self.slab_size, "CPU page cache (TP rank)")
        # With a disk tier: slot -> events after the last stream copies into or out of its slab, which the
        # rank's disk worker waits for before it reads or overwrites the slab from a host thread
        self.record_events = False
        self.events = {}


    def slab(self, slot: int):
        base = slot * self.slab_size
        return self.arena.tensor[base : base + self.slab_size]


    def note_copies(self, slot: int):
        if self.record_events:
            ev = torch.cuda.Event()
            ev.record()
            self.events[slot] = [ev]


    def close(self):
        """
        Drop the slot views and return the arena to the OS
        """
        self.slots = {}
        if self.arena is not None:
            self.arena.close()
            self.arena = None


    def get(self, slot: int):
        buffers = self.slots.get(slot)
        if buffers is None:
            assert 0 <= slot < self.max_slots
            base = slot * self.slab_size
            buffers = self.slots[slot] = [
                self.arena.tensor[base + offset : base + offset + nbytes].view(dtype).view(shape)
                for offset, shape, dtype, nbytes in self.segments
            ]
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
        # One pool per rank: a new Generator's tier replaces the previous Generator's
        if local_context.get("cpu_page_cache") is not None:
            local_context["cpu_page_cache"].close()
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
    pool.note_copies(slot)


def mp_cpu_cache_fetch(local_context: dict, cache_ids: list[int], slot: int, page_index: int):
    """
    Copy a stored page back into this rank's shard of a page slot (host-to-device, async on the current stream)
    """
    pool = local_context["cpu_page_cache"]
    buffers = pool.get(slot)        # a slot promoted from disk has not been stored into on this rank yet
    for buffer, tensor in zip(buffers, mp_cpu_cache_tensors(local_context, cache_ids)):
        tensor[page_index].copy_(buffer, non_blocking = True)
    pool.note_copies(slot)


# -- Disk tier (generator/disk_cache.py), rank side --------------------------------------------------------------
#
# A page on disk is one image in the canonical layout: every cache tensor's page slice with all kv heads in model
# order, the layout the non-TP tier writes. A rank holds a kv head range of each attention layer's cache, and in
# every cache format a head range is a contiguous sub-range of each token row (fp16 rows are heads x head_dim,
# the packed quantized rows and their scales are in head order), so a rank's shard maps onto the image as
# PAGE_SIZE fixed-stride chunks per tensor. Each rank reads and writes its own chunks straight from/into the file
# (memmap, the page cache assembles the image), so the split chosen at one load has no bearing on the file and a
# later load with another split, or no TP at all, restores the same page. Completion is signalled to the main
# process's writer thread through marker files, since only the main thread may use the dispatch pipes.

def mp_disk_layout(local_context: dict, cache_ids: list[int]) -> list[dict]:
    """
    This rank's shard geometry, one entry per slab segment (same order as RankSlotPool.segments): the cache,
    layer key and tensor index that identify the tensor across ranks, the local row and its offset and the
    canonical row width in bytes. Rows are tokens where the tensor is page-major by token, else the whole page.
    """
    pool = local_context["cpu_page_cache"]
    out = []
    i = 0
    for cache_id in cache_ids:
        for module in local_context["kv_modules"]:
            layer = module.tp_cache_lookup[cache_id]
            heads = module.num_kv_heads
            first = getattr(module, "tp_kv_first", 0)
            total = getattr(module, "tp_kv_total", heads)
            for idx, t in enumerate([t for t in layer.get_tensors() if t is not None]):
                nbytes = t[0].numel() * t.element_size()
                rows = PAGE_SIZE if t.dim() >= 2 and t.shape[1] == PAGE_SIZE else 1
                lrb = nbytes // rows
                if total != heads:
                    assert heads and lrb % heads == 0, \
                        f"cache rows of {module.key} do not split by kv head ({lrb} bytes, {heads} heads)"
                    unit = lrb // heads
                    row_offset, crb = first * unit, total * unit
                else:
                    row_offset, crb = 0, lrb
                out.append({
                    "cache_id": cache_id, "key": module.key, "idx": idx,
                    "slab_offset": pool.segments[i][0], "nbytes": nbytes, "rows": rows,
                    "local_row_bytes": lrb, "row_offset": row_offset, "canonical_row_bytes": crb,
                    "dtype": str(t.dtype),
                })
                i += 1
    return out


class DiskRankWorker:
    """One thread per rank moving this rank's shard of page images between its slabs and the page files"""

    def __init__(self, pool: RankSlotPool, layout: list[dict], image_size: int, device: int, local_context: dict):
        import threading
        from collections import deque
        self.pool = pool
        self.layout = layout
        self.image_size = image_size
        self.device = device
        self.local_context = local_context
        self.queue = deque()
        self.cv = threading.Condition()
        self.thread = threading.Thread(target = self._run, name = f"disk_cache_rank{device}", daemon = True)
        self.thread.start()

    def submit(self, kind: str, what, path: str, data_offset: int, plan = None):
        with self.cv:
            self.queue.append((kind, what, path, data_offset, plan))
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                kind, what, path, data_offset, plan = self.queue.popleft()
            marker = f"{path}.d{self.device}"
            try:
                if kind in ("write", "read"):
                    self._page_io(kind, what, path, data_offset)
                else:
                    self._state_io(kind, what, path, data_offset, plan)
            except Exception as e:
                marker += ".err"
                print(f" !! Disk cache tier (rank {self.device}): {kind} failed: {e}")
            open(marker, "w").close()

    def _page_io(self, kind, slot, path, data_offset):
        import numpy as np
        for ev in self.pool.events.pop(slot, []):
            ev.synchronize()
        slab = self.pool.slab(slot).numpy()
        mm = np.memmap(path, dtype = np.uint8, mode = "r+" if kind == "write" else "r",
                       offset = data_offset, shape = (self.image_size,))
        for seg in self.layout:
            rows, lrb, crb = seg["rows"], seg["local_row_bytes"], seg["canonical_row_bytes"]
            image = mm[seg["image_offset"] : seg["image_offset"] + rows * crb].reshape(rows, crb)
            image = image[:, seg["row_offset"] : seg["row_offset"] + lrb]
            local = slab[seg["slab_offset"] : seg["slab_offset"] + rows * lrb].reshape(rows, lrb)
            if kind == "write":
                image[...] = local
            else:
                local[...] = image
        if kind == "write":
            mm.flush()
        del mm

    def _state_io(self, kind, what, path, data_offset, plan):
        """Move this rank's pieces of every stashed tensor between the stash (pooled host tensors) and the
        canonical checkpoint image; a read allocates the stash first and files it under the handle"""
        import numpy as np
        from ..cache.recurrent import host_pool, wait_stashes
        cache_id, cp_handle = what
        lc = self.local_context
        write = kind == "state_write"
        if write:
            wait_stashes()  # the stash's copies to host may still be in flight
        image_size = max(e["offset"] + e["nbytes"] for layer in plan.values() for e in layer["tensors"])
        mm = np.memmap(path, dtype = np.uint8, mode = "r+" if write else "r", offset = data_offset, shape = (image_size,))
        stashed_in = lc["recurrent_cache"][cp_handle] if write else None
        stashed_out = []
        for i, module in enumerate(lc["recurrent_modules"]):
            layer = module.tp_recurrent_lookup[cache_id]
            entry = plan[module.key]
            if write:
                tensors = []
                pack_stash(stashed_in[i], tensors)
            else:
                tensors = []
            specs = _layer_pieces(layer, len(entry["tensors"]))
            for j, (info, spec) in enumerate(zip(entry["tensors"], specs)):
                dtype = getattr(torch, info["dtype"])
                itemsize = torch.empty((), dtype = dtype).element_size()
                shape = list(info["shape"])
                region = mm[info["offset"] : info["offset"] + info["nbytes"]].view(_raw_np(itemsize)).reshape(shape)
                if write:
                    t = tensors[j]
                else:
                    local_shape = list(shape)
                    if spec is not None:
                        local_shape[spec["dim"]] = sum(le - ls for ls, le, _ in spec["pieces"])
                    t = host_pool.take(local_shape, dtype)
                    tensors.append(t)
                raw = t.contiguous().view(_RAW_TORCH[itemsize]).numpy()
                if spec is None:
                    if write:
                        region[...] = raw
                    else:
                        raw[...] = region
                    continue
                d = spec["dim"]
                for ls, le, cs in spec["pieces"]:
                    li = [slice(None)] * raw.ndim
                    ci = [slice(None)] * raw.ndim
                    li[d] = slice(ls, le)
                    ci[d] = slice(cs, cs + (le - ls))
                    if write:
                        region[tuple(ci)] = raw[tuple(li)]
                    else:
                        raw[tuple(li)] = region[tuple(ci)]
            if not write:
                stashed_out.append(unpack_stash(entry["struct"], tensors))
        if write:
            mm.flush()
        else:
            lc["recurrent_cache"][cp_handle] = stashed_out
        del mm


def mp_disk_set_layout(local_context: dict, image_size: int, layout: list[dict]):
    """Attach the disk worker to this rank's slot pool with the canonical offsets the main process assigned"""
    pool = local_context["cpu_page_cache"]
    pool.record_events = True
    local_context["disk_worker"] = DiskRankWorker(pool, layout, image_size, local_context["device"], local_context)


# Checkpoints: a stash on a rank is a list, one entry per recurrent module, of tensors in whatever containers
# the layer's stash() returns. The disk tier stores a checkpoint in a canonical layout as well: each stashed
# tensor at full width, with the rank's part described by the layer state's tp_stash_pieces() (a split
# dimension and (local start, local end, canonical start) pieces; None for a tensor every rank holds whole).

def pack_stash(obj, tensors: list) -> object:
    """JSON-able description of a stash with its tensors replaced by indices into `tensors`; container types
    and dict key types (layer keys are ints, strings or tuples of them) are preserved"""
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
        return {"__t__": len(tensors) - 1, "shape": list(obj.shape), "dtype": str(obj.dtype).removeprefix("torch.")}
    if isinstance(obj, tuple):
        return {"__tuple__": [pack_stash(o, tensors) for o in obj]}
    if isinstance(obj, list):
        return [pack_stash(o, tensors) for o in obj]
    if isinstance(obj, dict):
        return {"__dict__": [[pack_stash(k, tensors), pack_stash(v, tensors)] for k, v in obj.items()]}
    return obj


def unpack_stash(obj, tensors: list) -> object:
    if isinstance(obj, dict):
        if "__t__" in obj:
            return tensors[obj["__t__"]]
        if "__tuple__" in obj:
            return tuple(unpack_stash(o, tensors) for o in obj["__tuple__"])
        if "__dict__" in obj:
            return {unpack_stash(k, tensors): unpack_stash(v, tensors) for k, v in obj["__dict__"]}
        return {k: unpack_stash(v, tensors) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unpack_stash(o, tensors) for o in obj]
    return obj


_RAW_TORCH = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

def _raw_np(itemsize: int):
    import numpy as np
    return {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[itemsize]


def _layer_pieces(layer, n: int):
    specs = layer.tp_stash_pieces() if hasattr(layer, "tp_stash_pieces") else None
    if specs is None:
        specs = [None] * n
    assert len(specs) == n, f"tp_stash_pieces of {type(layer).__name__} describes {len(specs)} tensors, stash has {n}"
    return specs


def mp_disk_state_describe(local_context: dict, cache_id: int, cp_handle: int) -> list[dict]:
    """This rank's part of a stashed checkpoint: per recurrent module its key, the stash's container
    structure and, per tensor, the local shape, dtype and split specification"""
    stashed = local_context["recurrent_cache"][cp_handle]
    out = []
    for module, s in zip(local_context["recurrent_modules"], stashed):
        layer = module.tp_recurrent_lookup[cache_id]
        tensors = []
        struct = pack_stash(s, tensors)
        specs = _layer_pieces(layer, len(tensors))
        out.append({
            "key": module.key,
            "struct": struct,
            "tensors": [
                {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."), "split": spec}
                for t, spec in zip(tensors, specs)
            ],
        })
    return out


def mp_disk_state_write(local_context: dict, cache_id: int, cp_handle: int, path: str, data_offset: int, plan: dict):
    """Queue this rank's pieces of checkpoint cp_handle for writing into the canonical image at data_offset;
    plan maps module key -> list of {offset, shape} per stashed tensor (image-relative offsets)"""
    local_context["disk_worker"].submit("state_write", (cache_id, cp_handle), path, data_offset, plan)


def mp_disk_state_read(local_context: dict, cache_id: int, cp_handle: int, path: str, data_offset: int, plan: dict):
    """Queue reading this rank's pieces of a checkpoint image into pooled host buffers, stored in the rank's
    recurrent cache under cp_handle; plan additionally carries each module's stash structure"""
    local_context["disk_worker"].submit("state_read", (cache_id, cp_handle), path, data_offset, plan)


def mp_disk_write(local_context: dict, slot: int, path: str, data_offset: int):
    local_context["disk_worker"].submit("write", slot, path, data_offset)


def mp_disk_read(local_context: dict, slot: int, path: str, data_offset: int):
    local_context["disk_worker"].submit("read", slot, path, data_offset)


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
        # This rank lives in the main process, so nothing else returns its arena on unload
        if self.local_context.get("cpu_page_cache") is not None:
            self.local_context["cpu_page_cache"].close()
            self.local_context["cpu_page_cache"] = None
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
