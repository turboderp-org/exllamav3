from __future__ import annotations
import torch
import multiprocessing
from multiprocessing import Process, Pipe
import torch.distributed as dist
from ..util import find_free_port
from .model_tp_alloc import TPAllocator
import os
from typing import Callable
from ..util.memory import touch_device_measure_vram
from ..util.progress import ProgressBar
from .config import Config
from ..util.misc import Cleanupper
from .model_tp_shared import SMProducer
from .model_tp_fn import *

cleanupper = Cleanupper()

class Model_TPMixin:

    def __init__(self):
        self.mp_children = []
        self.mp_parent_conn = []
        self.mp_child_conn = []
        self.loaded_tp = False
        self.tp_output_device = None


    def create_tp_context(self):
        """
        Create child processes and pipes
        """

        # Must use spawn method to avoid CUDA errors. Docs say this should always be set by __main__ but seems
        # to work okay here
        multiprocessing.set_start_method('spawn')
        torch.multiprocessing.set_sharing_strategy('file_system')

        # Get target rank for the final logits gather
        for rank, device in enumerate(self.active_devices):
            if device == self.tp_output_device:
                output_rank = rank

        # Master address and port for the process group
        master_addr = os.environ.get("EXLLAMA_MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("EXLLAMA_MASTER_PORT", find_free_port())
        init_method = f"tcp://{master_addr}:{master_port}"

        # Spawn child processes, each running the mp_model_worker function
        num_devices = max(self.active_devices) + 1
        assert not self.mp_children
        assert not self.mp_parent_conn
        assert not self.mp_child_conn
        self.mp_children: list = [None] * num_devices
        self.mp_parent_conn: list = [None] * num_devices
        self.mp_child_conn: list = [None] * num_devices
        for rank, device in enumerate(self.active_devices):
            if self.tp_output_device == device:
                self.mp_parent_conn[device] = PseudoParentConn(
                    device,
                    rank,
                    len(self.active_devices),
                    output_rank,
                    init_method,
                )
                self.mp_child_conn[device] = PseudoChildConn()
                self.mp_children[device] = PseudoChild()
            else:
                self.mp_parent_conn[device], self.mp_child_conn[device] = Pipe()
                self.mp_children[device] = Process(
                    target = mp_model_worker, args = (
                        self.mp_child_conn[device],
                        device,
                        rank,
                        len(self.active_devices),
                        output_rank,
                        init_method,
                    )
                )
                self.mp_children[device].start()

        # Install exit hook to avoid child processes hanging if main process exits before unloading model
        cleanupper.register_atexit(self.destroy_tp_context)


    def destroy_tp_context(self):
        """
        Destroy child processes (when unloading TP model or atexit)
        """
        # Destroy process group in child processes
        for device, (parent_conn, child) in enumerate(zip(self.mp_parent_conn, self.mp_children)):
            if device == self.tp_output_device or child is None:
                continue
            if child.is_alive():
                try:
                    parent_conn.send("quit")
                except Exception:
                    pass

        # Destroy process group in main process. Called last since it blocks the main process
        self.mp_parent_conn[self.tp_output_device].quit()

        # Join child processes (terminate if hung), close connections
        for device, (parent_conn, child_conn, child) in \
            enumerate(zip(self.mp_parent_conn, self.mp_child_conn, self.mp_children)):
            if device == self.tp_output_device or child is None:
                continue
            child.join(timeout = 2)
            if child.is_alive():
                child.terminate()
            child_conn.close()
            parent_conn.close()

        self.mp_children = []
        self.mp_parent_conn = []
        self.mp_child_conn = []

        # Unregister exit hook
        cleanupper.unregister_atexit(self.destroy_tp_context)


    def tp_worker_dispatch_single(self, device, fn, args):
        """
        Dispatch single function call to child and get return value
        """
        conn = self.mp_parent_conn[device]
        conn.send((fn, args))
        result = conn.recv()
        if isinstance(result, Exception):
            raise result
        return result


    def tp_worker_dispatch(self, device, fn, args):
        """
        Dispatch function call to child
        """
        conn = self.mp_parent_conn[device]
        conn.send((fn, args))


    def tp_worker_result(self, device):
        """
        Await and return result from child function, and propagate any exceptions to main process
        """
        conn = self.mp_parent_conn[device]
        result = conn.recv()
        if isinstance(result, Exception):
            raise result
        return result


    def tp_worker_dispatch_multi(self, active_devices: list[int], fn, args, dev_args: list | None = None):
        for idx, device in enumerate(active_devices):
            d_args = args
            if dev_args is not None:
                d_args = d_args + dev_args[idx]
            conn = self.mp_parent_conn[device]
            conn.send((fn, d_args))


    def tp_worker_wait_multi(self, active_devices: list[int]):
        r = []
        for device in active_devices:
            r.append(self.tp_worker_result(device))
        return r


    def tp_worker_dispatch_wait_multi(self, active_devices: list[int], fn, args, dev_args: list | None = None):
        self.tp_worker_dispatch_multi(active_devices, fn, args, dev_args)
        return self.tp_worker_wait_multi(active_devices)


    def tp_cache_page_copy(self, cache_id: int, from_page: int, to_page: int, num_tokens: int):
        for device in self.active_devices:
            self.tp_worker_dispatch(device, mp_cache_page_copy, (
                cache_id,
                from_page,
                to_page,
                num_tokens
            ))
        for device in self.active_devices:
            self.tp_worker_result(device)


    def _load_tp(
        self,
        progressbar: bool,
        reserve_per_device: list[int] | None,
        use_per_device: list[int] | None,
        active_devices: list[int],
        max_chunk_size: int,
        max_output_size: int,
        max_output_factor: int,
        callback_sync: Callable[[int, int], None],
        generator: bool,
        tp_output_device: torch.device | int | str,
        config: Config,
        modules: list,
    ):
        assert use_per_device is None or reserve_per_device is None

        # Set output device
        if tp_output_device is None:
            tp_output_device = active_devices[0]
        self.tp_output_device = torch.device(tp_output_device).index

        # Move output device to end of active device list
        active_devices.remove(self.tp_output_device)
        active_devices.append(self.tp_output_device)

        # Create TP context
        self.active_devices = active_devices
        self.create_tp_context()

        # Split model
        num_devices = max(self.active_devices) + 1
        max_mem = [0] * num_devices
        free_total = self.tp_worker_dispatch_wait_multi(self.active_devices, touch_device_measure_vram, ())
        for device, (free, total) in zip(self.active_devices, free_total):
            # print(free / 1024**3)  snip
            if reserve_per_device is not None:
                free -= reserve_per_device[device]
            if use_per_device is not None:
                free = use_per_device[device]
            max_mem[device] = free

        # Define TP split
        components = []
        for m in modules:
            components += m.make_tp_allocation()
        allocator = TPAllocator(
            components,
            num_tokens = max_chunk_size,
            output_num_tokens = max_output_size,
            dev_limits = {
                # TODO: Args for max parallelism per component
                # "mlp": min(4, len(self.active_devices)),
                # "attn": min(3, len(self.active_devices)),
                # "linear": 1,
                # "moe": min(1, len(self.active_devices)),
            }
        )
        allocator.initial_split(max_mem)
        allocator.print_split()
        plan = allocator.compile_tp_plan()
        self.tp_worker_dispatch_wait_multi(self.active_devices, mp_set_plan, (plan, self.active_devices))

        # Distribution pipeline
        producer = SMProducer()
        self.tp_worker_dispatch_wait_multi(
            self.active_devices,
            mp_set_consumer,
            (),
            [(producer.export(d),) if d != tp_output_device else (producer,) for d in active_devices]
        )

        # Begin loading modules
        with (ProgressBar(f"Loading" if progressbar else None, len(modules)) as progress):
            for idx, module in enumerate(modules):
                last_module = module

                # Load module to CPU
                defer = module.can_defer_load()
                if defer:
                    config.stc.begin_deferred_load()
                module.load(torch.device("cpu"))
                if defer:
                    config.stc.end_deferred_load()

                # Do module-specific device/process split
                exported = module.tp_export(plan, producer)
                self.tp_worker_dispatch_wait_multi(self.active_devices, mp_model_append, (exported,))
                producer.clear()

                # Release loaded module
                module.unload()

                # Progress and callbacks per fully loaded module
                progress.update(idx + 1)
                if callback_sync: callback_sync(len(modules), len(modules))
                if generator: yield len(modules), len(modules)

            # Append final gather layer
            if last_module.caps["logits_output"]:
                self.tp_worker_dispatch_wait_multi(self.active_devices, mp_model_append_gather, ())

        # Distribution pipeline
        self.tp_worker_dispatch_wait_multi(self.active_devices, mp_close_consumer, ())
        producer.close()

        config.stc.close()
        self.loaded_tp = True

        if 'yield' in locals():
            yield


    def unload_tp(self):
        self.destroy_tp_context()
        self.loaded_tp = False
        self.tp_output_device = None
        cleanupper.unregister_atexit(self.destroy_tp_context)



    def prepare_inputs_for_tp(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        # Use ID of Cache object as reference to avoid having to pickle it
        if "cache" in params:
            params["cache"] = id(params["cache"])
        # Share memory of any additional CPU tensors
        for tensor_param in [
            "block_table",
            "cache_seqlens",
            "positions",
            "position_ids",
        ]:
            if params.get(tensor_param) is not None:
                params[tensor_param].share_memory_()
        x = x.clone()
        x.share_memory_()
        return x


    def prefill_tp(
        self,
        x: torch.Tensor,
        params: dict,
        last_kv_module_idx: int,
        modules: list,
    ):
        x = self.prepare_inputs_for_tp(x, params)
        for device in self.active_devices:
            # print("device", device, "launch prefill")
            self.tp_worker_dispatch(device, mp_model_forward, (
                x,
                params,
                last_kv_module_idx,
                True
            ))
        for device in self.active_devices:
            r = self.tp_worker_result(device)
            assert r is None, "TP logic error"
        return None


    def forward_tp(
        self,
        x: torch.Tensor,
        params: dict,
        last_kv_module_idx: int,
        modules: list,
    ):
        x = self.prepare_inputs_for_tp(x, params)
        for device in self.active_devices:
            # print("device", device, "launch forward")
            self.tp_worker_dispatch(device, mp_model_forward, (
                x,
                params,
                last_kv_module_idx,
                False
            ))
        return_tensors = []
        for device in self.active_devices:
            r = self.tp_worker_result(device)
            if r is not None:
                return_tensors.append(r)
        assert len(return_tensors) == 1, "TP logic error"
        return return_tensors[0]