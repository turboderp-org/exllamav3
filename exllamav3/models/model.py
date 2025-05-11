from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
import os, json
from .config import Config
from ..util.progress import ProgressBar
from ..util.memory import set_memory_fraction_reserve, set_memory_fraction_use, unset_memory_fraction, free_mem

class Model:

    def __init__(
        self,
        config: Config,
        **kwargs,
    ):
        self.config = config

        self.modules = []

        # Index of last layer that affects KV cache, used during prefill
        self.last_kv_module_idx = None
        self.logit_layer_idx = None
        self.first_block_idx = None

        # Calibration options
        self.calibration_all_experts = False


    def __iter__(self):
        for module in self.modules:
            yield from module


    def find_module(self, key: str):
        for module in self:
            if module.key == key:
                return module


    @lru_cache
    def get_cache_layers(self):
        return [m for m in self if m.caps.get("kv_cache")]


    @staticmethod
    def from_config(config: Config, **kwargs):
        """
        Create model instance from config
        """
        model = config.model_class(config, **kwargs)
        return model


    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        return input_ids


    @torch.inference_mode
    def prefill(self, input_ids: torch.Tensor, params: dict):
        x = self.prepare_inputs(input_ids, params)
        for idx, module in enumerate(self.modules):
            params["prefill"] = (idx == self.last_kv_module_idx)
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if idx == self.last_kv_module_idx:
                break


    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        for idx, module in enumerate(self.modules):
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x


    def unload(self):
        for module in self.modules:
            module.unload()


    # Load to single device
    def _load_single(self, progressbar: bool, device: torch.device):
        with ProgressBar(f"Loading" if progressbar else None, len(self.modules)) as progress:
            for idx, module in enumerate(self.modules):
                defer = module.can_defer_load()
                if defer:
                    self.config.stc.begin_deferred_load()
                module.load(torch.device("cpu") if module.caps.get("prefer_cpu") else device)
                if defer:
                    self.config.stc.end_deferred_load()
                progress.update(idx + 1)


    # Load with split
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
        generator: bool
    ):
        current_device_i = 0
        backup_shape = (1, max_chunk_size)
        backup_dtype = torch.long
        dummy_state = None
        prev_load_device = None
        touched_devices = []

        with ProgressBar(f"Loading" if progressbar else None, len(self.modules)) as progress:

            for idx, module in enumerate(self.modules):

                if callback_sync: callback_sync(idx, len(self.modules))
                if generator: yield idx, len(self.modules)

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
                            self.config.stc.begin_deferred_load()
                        module.load(load_device)
                        if defer:
                            self.config.stc.end_deferred_load()

                        # Forward dummy state through module
                        dummy_state = module.prepare_for_device(dummy_state, {})
                        dummy_state = module.forward(dummy_state, {})

                        # Account for max_output_factor after last layer,
                        if is_logits_layer:
                            extra_dummy_states = [
                                torch.empty_like(dummy_state)
                                for _ in range(max_output_factor - 1)
                            ]

                        # We're good
                        fail = False
                        progress.update(idx + 1)

                    # We're not good
                    except Exception as e:
                        self.config.stc.abort_deferred_load()
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

            if callback_sync: callback_sync(len(self.modules), len(self.modules))
            if generator: yield len(self.modules), len(self.modules)

            dummy_state = None
            unset_memory_fraction(touched_devices)

        # Python will not run anything in an async function without at least one yield statement
        if 'yield' in locals():
            yield


    def load_gen(
        self,
        device: torch.device | str | int | None = None,
        reserve_per_device: list[float] | float | None = None,
        use_per_device: list[float] | float | None = None,
        tensor_p: bool = False,
        progressbar: bool = False,
        max_chunk_size: int = 2048,
        max_output_size: int = 32,
        max_output_factor: int = 1,
        callback: Callable[[int, int], None] | None = None,
        generator: bool = True
    ):
        """
        Load model, generator function. For regular function, call load() with the same arguments

        :param device:
            (optional) If specified, load to single device, e.g. "cuda:0"

        :param reserve_per_device:
            (optional) Amount of memory to reserve for any device. Either a value in GB to apply on all devices
            or a list of floats giving an individual reserve per device. Negative reserve excludes device from
            split. E.g.:

            # reserve 4.5 GB on cuda:0, 1 GB on each cuda:1 and on cuda:2
            model.load(reserve_per_device = [4.5, 1, 1])

            # reserve 1 GB on cuda:0 and cuda:2, exclude cuda:1
            model.load(reserve_per_device = [1, -1, 1])

            The default reserve per device is 0.25 GB. This applies to devices not included in reserve_per_device
            as well.

        :param use_per_device:
            (optional) Amount of memory to use per device.

            Does not account for memory allocated by other processes or by the calling process up to the call
            to model.load(), i.e. if cuda:0 currently has 3 GB in use and user_per_device = [12, ...], at the
            end of loading cuda:0 will have up to 15 GB of VRAM allocated, using up to 15 GB during a forward
            pass.

            Devices not included in use_per_device, or included with a value of 0, will not be used, e.g.:

            # use up to 23 GB on cuda:0 and cuda:2, do not load on cuda:1 and cuda:3 (if present)
            model.load(use_per_device = [23, 0, 23])

        :param tensor_p:
            Load in tensor-parallel mode (not implemented yet)  TODO

        :param max_chunk_size:
            The maximum number of tokens to expect in a single forward pass. Informs the layer split only, and
            makes no difference when loading on a single device.

        :param max_output_size:
            The maximum number of output tokens to expect in a single forward pass. Informs the estimate of the
            size of the output logits. Values larger than max_chunk_size have no effect.

        :param max_output_factor:
            When estimating the memory footprint of the output layer, scale the size of the output tensor by
            this factor. For instance, if the first thing you wish to do with a float16 output tensor is upcast
            to float32, a value of 3 here would (attempt to) make sure the output layer always ends up on a
            device where there is enough space for that.

        :param progressbar:
            Show rich progressbar while loading

        :param callback:
            If provided, called with (current_module, num_modules) for every module loaded. Don't specify a
            callback function when using the

        :param generator:
            Always true when using the _gen function directly
        """

        free_mem()

        assert not (bool(reserve_per_device) and bool(use_per_device)), \
            "Cannot specify both memory usage and memory reserve."

        assert max_chunk_size >= 1, "max_chunk_size must be positive"
        assert max_output_size >= 1, "max_output_size must be positive"
        assert max_output_factor >= 1, "max_output_factor must be positive"

        # Load to single device
        if device is not None:
            assert not bool(reserve_per_device) and not bool(use_per_device), \
                "Cannot specify reserve_per_device or use_per_device when loading to single device."
            assert not tensor_p, \
                "Cannot use tensor_p when loading to single device."
            self._load_single(progressbar, device)

        # Split load
        elif not tensor_p:
            rpd = reserve_per_device is not None
            upd = use_per_device is not None
            assert not (rpd and upd), \
                "Cannot specify both reserve_per_device or use_per_device."
            num_devices = torch.cuda.device_count()

            if not upd:
                if reserve_per_device is None:
                    reserve_per_device = [0.25] * num_devices
                elif any(isinstance(reserve_per_device, t) for t in [float, int]):
                    reserve_per_device = [reserve_per_device] * num_devices
                elif not isinstance(reserve_per_device, list):
                    raise ValueError("reserve_per_device must be float or list[float]")
                while len(reserve_per_device) < num_devices:
                    reserve_per_device.append(0.25)
                reserve_per_device = [int(x * 1024**3) for x in reserve_per_device]
                active_devices = [
                    i for i in range(num_devices)
                    if i >= len(reserve_per_device) or reserve_per_device[i] >= 0
                ]

            if upd:
                if any(isinstance(use_per_device, t) for t in [float, int]):
                    use_per_device = [use_per_device] * num_devices
                elif not isinstance(use_per_device, list):
                    raise ValueError("use_per_device must be float or list[float]")
                use_per_device = [int(x * 1024**3) for x in use_per_device]
                active_devices = [
                    i for i, x in enumerate(use_per_device)
                    if x > 0
                ]

            yield from self._load_autosplit(
                progressbar,
                reserve_per_device,
                use_per_device,
                active_devices,
                max_chunk_size,
                max_output_size,
                max_output_factor,
                callback,
                generator
            )

        # Tensor-p load
        else:
            raise NotImplementedError()

        self.config.stc.close()
        free_mem()


    @torch.inference_mode
    def load(self, *args, **kwargs):
        """
        Load as a regular function, see arguments for load_gen().
        """

        kwargs["generator"] = False
        f = self.load_gen(*args, **kwargs)
        for _ in f: pass


    def get_load_metrics(self):
        return self.config.stc.get_metrics()


    def get_layout_tree(self, pre_indent: int) -> str:
        def get_branch(module, b_indent) -> str:
            lines = [get_branch(m, b_indent + 4) for m in module.modules]
            dedup_lines = []
            count = 1
            for i in range(len(lines)):
                if i < len(lines) - 1 and lines[i] == lines[i + 1]:
                    count += 1
                else:
                    pref = ""
                    if count > 1:
                        pref = f"[{count}x] "
                        count = 1
                    dedup_lines.append(lines[i].replace("[]", pref))
            r = " " * (pre_indent + b_indent) + " - []" + module.get_name() + "\n"
            r += "".join(dedup_lines)
            return r
        return get_branch(self, 0).replace("[]", "").rstrip()


    def get_storage_info(self):
        from ..modules import Linear
        def get_tensor_size(tensors):
            return 8 * sum(t.element_size() * t.numel() for t in tensors.values())
        sum_bits = 0
        sum_numel = 0
        head_bpw = 0
        head_numel = 0
        for module in self:
            if module.key == "lm_head":
                head_bpw = get_tensor_size(module.get_tensors()) / module.weights_numel()
                head_numel = module.weights_numel()
            elif isinstance(module, Linear):
                sum_bits += get_tensor_size(module.get_tensors())
                sum_numel += module.weights_numel()
        vram_bits = head_numel * head_bpw + sum_bits
        return sum_bits / sum_numel, head_bpw, vram_bits


    def get_name(self):
        return self.__class__.__name__