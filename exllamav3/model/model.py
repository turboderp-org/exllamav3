from __future__ import annotations
from functools import cached_property
from typing import Callable
import torch
from .config import Config
from ..util.memory import free_mem
from .model_tp import Model_TPMixin
from .model_ls import Model_LSMixin

class Model(Model_TPMixin, Model_LSMixin):

    def __init__(
        self,
        config: Config,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        self.modules = []
        self.caps = {
            "supports_tp": True
        }
        self.active_devices = []
        self.output_device = None

        # Index of last layer that affects KV cache, used during prefill
        self.last_kv_module_idx = None
        self.logit_layer_idx = None
        self.first_block_idx = None

        # Calibration options
        self.calibration_all_experts = False

        # Modules dict
        self.modules_dict = None

        # Check compatibility
        self.check_compat()


    def __iter__(self):
        for module in self.modules:
            yield from module


    def find_module(self, key: str):
        if self.modules_dict is None:
            self.modules_dict = {module.key: module for module in self}
        return self.modules_dict[key]


    @cached_property
    def _get_cache_layers(self):
        return [m for m in self if m.caps.get("kv_cache")]
    def get_cache_layers(self):
        return self._get_cache_layers


    @cached_property
    def _get_recurrent_layers(self):
        return [m for m in self if m.caps.get("recurrent_cache")]
    def get_recurrent_layers(self):
        return self._get_recurrent_layers


    @staticmethod
    def from_config(
        config: Config,
        component: str = "text",
        **kwargs
    ):
        """
        Create model instance from config

        :param config:
            Config created with Config.from_directory()

        :param component:
            Which component model to load, for models with multiple component.
        """

        assert component in config.model_classes, \
            f"{config.architecture} does not define a '{component}' component model"

        model = config.model_classes[component](config, **kwargs)
        return model


    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Overridden by model arch class
        raise NotImplementedError()


    @torch.inference_mode
    def prefill(self, input_ids: torch.Tensor, params: dict | None = None):
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        if self.loaded_tp:
            return self.prefill_tp(x, params, self.last_kv_module_idx, self.modules)
        else:
            return self.prefill_ls(x, params, self.last_kv_module_idx, self.modules)


    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        if self.loaded_tp:
            return self.forward_tp(x, params, self.last_kv_module_idx, self.modules)
        else:
            return self.forward_ls(x, params, self.last_kv_module_idx, self.modules)


    def unload(self):
        for module in self.modules:
            module.unload()
        self.active_devices = []
        self.unload_tp()
        self.output_device = None


    def load_gen(
        self,
        device: torch.device | str | int | None = None,
        tp_output_device: torch.device | str | int | None = None,
        reserve_per_device: list[float] | float | None = None,
        use_per_device: list[float] | float | None = None,
        tensor_p: bool = False,
        progressbar: bool = False,
        max_chunk_size: int = 2048,
        max_output_size: int = 32,
        max_output_factor: int = 1,
        callback: Callable[[int, int], None] | None = None,
        generator: bool = True,
        tp_dev_limits: dict | None = None,
        tp_backend: str = "native",
        verbose: bool = False,
        tp_options: dict | None = None,
    ):
        """
        Load model, generator function. For regular function, call load() with the same arguments

        :param device:
            (optional) If specified, load to single device, e.g. "cuda:0"

        :param tp_output_device:
            (optional) If loading with tensor_p == True, device on which to gather output logits. Must be one of
            the active devices in the split. Default is first device in split

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
            Load in tensor-parallel mode. By default, attempt to split model according to available VRAM.
            Allocation can be overridden with use_per_device or modified by reserve_per_device.

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

        :param tp_dev_limits:
            (optional, TP only) Dictionary of module categories and max parallelism for each. Categories are
            "mlp", "attn", "moe", "linear" (i.e. output layer). Example:
            tp_dev_limits = {
                "attn": 2,  # Each attn layer uses at most two devices for tensor parallelism
                "moe": 3,  # etc.
            }

        :param tp_backend:
            str, either "nccl" (default) or "native"

        :param verbose:
            bool, more info while loading including full TP split

        :param tp_options:
            dict of optional values:
                "moe_tensor_split": bool - use tensor split rather than expert parallelism for MoE layers
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
            self._load_single(progressbar, device, self.config, self.modules, verbose)

        # Use/reserve
        else:
            rpd = reserve_per_device is not None
            upd = use_per_device is not None
            assert not (rpd and upd), \
                "Cannot specify both reserve_per_device or use_per_device."
            num_devices = torch.cuda.device_count()

            if not upd:
                if reserve_per_device is None:
                    reserve_per_device = [0.5] * num_devices
                elif any(isinstance(reserve_per_device, t) for t in [float, int]):
                    reserve_per_device = [reserve_per_device] * num_devices
                elif not isinstance(reserve_per_device, list):
                    raise ValueError("reserve_per_device must be float or list[float]")
                while len(reserve_per_device) < num_devices:
                    reserve_per_device.append(0.5)
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

            # Split load
            if not tensor_p:
                yield from self._load_autosplit(
                    progressbar,
                    reserve_per_device,
                    use_per_device,
                    active_devices,
                    max_chunk_size,
                    max_output_size,
                    max_output_factor,
                    callback,
                    generator,
                    self.config,
                    self.modules,
                    verbose,
                )
                self.output_device = self.modules[-1].device

            # Tensor-P load:
            else:
                if not self.caps.get("supports_tp"):
                    raise NotImplementedError(f"Tensor-parallel is not currently implemented for {self.config.architecture}")

                if tp_output_device is None:
                    tp_output_device = active_devices[0]
                else:
                    assert torch.device(tp_output_device).index in active_devices, \
                        "Output device must be part of split."

                if tp_options is None:
                    tp_options = {}

                yield from self._load_tp(
                    progressbar,
                    reserve_per_device,
                    use_per_device,
                    active_devices,
                    max_chunk_size,
                    max_output_size,
                    max_output_factor,
                    callback,
                    generator,
                    tp_output_device,
                    self.config,
                    self.modules,
                    tp_dev_limits,
                    tp_backend,
                    verbose,
                    tp_options,
                )
                self.output_device = tp_output_device

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
            if module.key.endswith("lm_head"):
                if module.device is not None:
                    head_bpw = get_tensor_size(module.get_tensors()) / module.weights_numel()
                else:
                    head_bpw = sum(self.config.stc.get_tensor_sizes(module.key)) / module.weights_numel() * 8
                head_numel = module.weights_numel()
            elif isinstance(module, Linear):
                if module.device is not None:
                    sum_bits += get_tensor_size(module.get_tensors())
                else:
                    sum_bits += sum(self.config.stc.get_tensor_sizes(module.key)) * 8
                sum_numel += module.weights_numel()
        vram_bits = head_numel * head_bpw + sum_bits
        return sum_bits / sum_numel, head_bpw, vram_bits


    def get_name(self):
        return self.__class__.__name__


    @staticmethod
    def get_additional_compiled_tensors(config: Config):
        return {}


    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        """
        Convenience function for formatting a single chat request with the default template associated with the
        model's architecture, to simplify example and test scripts. Doesn't consider the model's actual Jinja template.
        """
        raise NotImplementedError()


    def batch_recurrent_states(self):
        raise NotImplementedError()


    def check_compat(self):
        """
        Decide if any model-specific requirements are met when creating Model
        """
        pass