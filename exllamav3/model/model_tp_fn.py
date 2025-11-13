import torch
import traceback
from .model_tp_shared import SMProducer, SMConsumer
from ..ext import exllamav3_ext as ext
from functools import lru_cache
from .model_tp_backend import TPBackendNCCL, TPBackendNative
from ..util import log_tp, set_t0

def init_pg(device: int, active_devices: list[int], output_device: int, backend_args: dict, master: bool = False):
    rank = active_devices.index(device) if device >= 0 else -1
    output_rank = active_devices.index(output_device)
    world_size = len(active_devices)
    local_context = {
        "device": device,
        "modules": [],
        "kv_modules": [],
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
    set_t0("TP", dbg_t0_)
    log_tp(device, f"Child process launched")

    with torch.inference_mode():
        local_context = init_pg(device, active_devices, output_device, backend_args)
        local_context["inf_consumer"] = SMConsumer(producer, device = device, pin_memory = True)

        # Dispatch loop
        while True:
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
    device = local_context["device"]
    cls = exported["cls"]
    plan = local_context["plan"]

    module = cls.tp_import(local_context, exported, plan[device])
    modules.append(module)
    kv_modules += module.all_cache_modules()
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
):
    """
    Forward pass for parallel slice of a model
    """
    backend = local_context["backend"]
    backend.fwd_barrier()

    modules = local_context["modules"]
    consumer = local_context["inf_consumer"]

    for tensor_param in [
        "block_table",
        "cache_seqlens",
        "positions",
        "position_ids",
        "indexed_embeddings",
    ]:
        p = params.get(tensor_param)
        if p is not None:
            params[tensor_param] = consumer.recv(p, cuda = True)

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


class PseudoParentConn:
    """
    Standin for a Pipe to dispatch functions on the main device rather than a dedicated child process running
    `mp_model_worker`. This allows a tensor-parallel model to run partially in the main process, without additional
    IPC overhead when returning logits from forward(), and without needing two CUDA contexts on the main device.
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
