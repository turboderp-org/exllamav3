import torch
import torch.distributed as dist
import traceback
from .model_tp_shared import SMProducer, SMConsumer

def mp_warmup_nccl(device):
    """
    NCCL does lazy initialization which causes the first reduction operation to take an exceedingly long time
    (20+ seconds). This seems to lead to race conditions or timeouts if it happens during a forward pass. Called
    by TP loader as soon as processes are spawned and process group is initialized.
    """
    print(f" -- NCCL warmup, device {device}, please wait...")
    x = torch.ones((6,), device = device)
    dist.all_reduce(x)
    print(f" -- Finished NCCL warmup, device {device}")


def init_pg(device: int, rank: int, world_size: int, output_rank: int, init_method: str):
    local_context = {
        "device": device,
        "modules": [],
        "kv_modules": [],
        "rank": rank,
        "world_size": world_size,
        "output_rank": output_rank,
    }
    dist.init_process_group("nccl", rank = rank, world_size = world_size, init_method = init_method)
    torch.cuda.set_device(device)
    mp_warmup_nccl(device)
    return local_context


def mp_model_worker(
    conn,
    device: int,
    rank: int,
    world_size: int,
    output_rank: int,
    init_method: str,
):
    with torch.inference_mode():
        local_context = init_pg(device, rank, world_size, output_rank, init_method)

        # Dispatch loop
        while True:
            msg = conn.recv()
            if msg == "quit":
                torch.cuda.synchronize()
                dist.barrier()
                dist.destroy_process_group()
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

    last_key = modules[-1].key
    module = OutputGather(
        config = None,
        key = "output_gather",
        rank = local_context["rank"],
        world_size = local_context["world_size"],
        output_rank = local_context["output_rank"],
        splits = [
            (plan[active_devices[i]][last_key][0], plan[active_devices[i]][last_key][1])
            for i in range(local_context["world_size"])
        ],
    )
    modules.append(module)
    return None


def mp_model_forward(
    local_context: dict,
    shared_input: torch.tensor,
    params: dict,
    last_kv_module_idx: int,
    prefill: bool,
):
    """
    Forward pass for parallel slice of a model
    """
    # This seems to be needed (why?)
    dist.barrier()

    modules = local_context["modules"]
    x = shared_input
    for idx, module in enumerate(modules):
        logits_layer = module.caps.get("logits_output")
        if logits_layer and (num := params.get("last_tokens_only")):
            x = x[..., -num:, :].contiguous()
        if prefill:
            params["prefill"] = (idx == last_kv_module_idx)
        x = module.prepare_for_device(x, params)
        x = module.forward(x, params)
        if prefill and idx == last_kv_module_idx:
            return None
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


class PseudoParentConn:
    """
    Standin for a Pipe to dispatch functions on the main device rather than a dedicated child process running
    `mp_model_worker`. This allows a tensor-parallel model to run partially in the main process, without additional
    IPC overhead when returning logits from forward(), and without needing two CUDA contexts on the main device.
    """

    def __init__(
        self,
        device: int,
        rank: int,
        world_size: int,
        output_rank: int,
        init_method: str,
    ):
        self.local_context = init_pg(device, rank, world_size, output_rank, init_method)
        self.result = None


    def send(self, msg):
        fn, args = msg
        self.result = fn(self.local_context, *args)


    def recv(self):
        r = self.result
        self.result = None
        return r


    def close(self, *args, **kwargs):
        self.local_context = {}


    def quit(self):
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()


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
