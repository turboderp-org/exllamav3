from __future__ import annotations
from functools import lru_cache
from typing import Callable
import torch
from ..util.memory import (
    set_memory_fraction_reserve,
    set_memory_fraction_use,
    unset_memory_fraction,
    free_mem,
)
from ..util.progress import ProgressBar
from .config import Config


class Model_LSMixin:

    def __init__(self):
        pass


    def _load_single(
        self,
        progressbar: bool,
        device: torch.device,
        config: Config,
        modules: list,
        verbose: bool
    ):
        with ProgressBar(f"Loading" if progressbar else None, len(modules)) as progress:
            for idx, module in enumerate(modules):
                defer = module.can_defer_load()
                if defer:
                    config.stc.begin_deferred_load()
                module.load(torch.device("cpu") if module.caps.get("prefer_cpu") else device)
                if defer:
                    config.stc.end_deferred_load()
                progress.update(idx + 1)


    def default_load_shape_dtype(self, chunk_size):
        return (1, chunk_size), torch.long


    def default_load_params(self):
        return {}


    def _load_autosplit(
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
        config: Config,
        modules: list,
        verbose: bool
    ):
        current_device_i = 0
        backup_shape, backup_dtype = self.default_load_shape_dtype(max_chunk_size)
        dummy_state = None
        prev_load_device = None
        touched_devices = []
        params = self.default_load_params()

        with ProgressBar(f"Loading (LS)" if progressbar else None, len(modules)) as progress:

            for idx, module in enumerate(modules):

                if callback_sync: callback_sync(idx, len(modules))
                if generator: yield idx, len(modules)

                # Narrow state to max_output_size for logit output layer
                is_logits_layer = module.caps.get("logits_output")
                if is_logits_layer:
                    b, c, d = backup_shape
                    backup_shape = (b, min(max_output_size, c), d)
                    if dummy_state is not None:
                        dummy_state = dummy_state[:, :max_output_size, :]

                while True:
                    try:
                        # Select device
                        load_device = torch.device("cpu") if module.caps.get("prefer_cpu") else \
                            torch.device(active_devices[current_device_i])

                        # Set VRAM limit if new device
                        if load_device != torch.device("cpu") and load_device != prev_load_device:
                            prev_load_device = load_device
                            i = active_devices[current_device_i]
                            if reserve_per_device is not None:
                                set_memory_fraction_reserve(reserve_per_device[i], i)
                            elif use_per_device is not None:
                                 set_memory_fraction_use(use_per_device[i], i)
                            else:
                                raise RuntimeError("Logic error")
                            touched_devices.append(i)

                        # (Re)create or backup hidden state (metadata)
                        if dummy_state is None:
                            dummy_state = torch.zeros(backup_shape, dtype = backup_dtype, device = load_device)
                        else:
                            backup_shape = dummy_state.shape
                            backup_dtype = dummy_state.dtype

                        # Load module
                        defer = module.can_defer_load()
                        if defer:
                            config.stc.begin_deferred_load()
                        module.load(load_device)
                        if defer:
                            config.stc.end_deferred_load()

                        # Account for cache quant temporary tensors
                        qcache_overhead = []
                        for cm in module.all_cache_modules():
                            for cl in cm.cache_layers:
                                qcache_overhead.append(cl.get_kv_alloc_placeholder())

                        # Forward dummy state through module
                        dummy_state = module.prepare_for_device(dummy_state, params)
                        dummy_state = module.forward(dummy_state, params)

                        # Account for max_output_factor after last layer,
                        extra_dummy_states = None
                        if is_logits_layer:
                            extra_dummy_states = [
                                torch.empty_like(dummy_state)
                                for _ in range(max_output_factor - 1)
                            ]

                        # Dereference extra dummy tensors
                        extra_dummy_states = None
                        qcache_overhead = None

                        # We're good
                        fail = False
                        progress.update(idx + 1)

                    # We're not good
                    except Exception as e:
                        config.stc.abort_deferred_load()
                        if e.__class__.__name__ == "OutOfMemoryError" or \
                            "CUDA out of memory" in str(e) or \
                            "HIP out of memory" in str(e):
                            # Exception object will hold references to tensors so we can't free them here
                            fail = True
                        else:
                            raise

                    # Module failed to load with an OoM error, so advance to the next device if possible
                    if fail:
                        module.unload()
                        dummy_state = None
                        free_mem()
                        current_device_i += 1
                        if current_device_i >= len(active_devices):
                            raise RuntimeError("Insufficient VRAM in split for model and cache")
                        continue

                    # On to next module
                    break

            if callback_sync: callback_sync(len(modules), len(modules))
            if generator: yield len(modules), len(modules)

            dummy_state = None
            unset_memory_fraction(touched_devices)

        config.stc.close()
        self.active_devices = active_devices

        # Python will not run anything in an async function without at least one yield statement
        if 'yield' in locals():
            yield


    def prefill_ls(
        self,
        x: torch.Tensor,
        params: dict,
        last_kv_module_idx: int,
        modules: list,
    ):
        for idx, module in enumerate(modules):
            params["prefill"] = (idx == last_kv_module_idx)
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if idx == last_kv_module_idx:
                break
        del params["prefill"]
        return None


    def forward_ls(
        self,
        x: torch.Tensor,
        params: dict,
        last_kv_module_idx: int,
        modules: list,
    ):
        for idx, module in enumerate(modules):
            if module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                x = x[..., -num:, :].contiguous()
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x

