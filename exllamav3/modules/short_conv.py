from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

from ..cache import Cache
from ..model.config import Config
from ..util.tensor import get_for_device, to2
from . import Module, Linear
from ..cache.recurrent import (
    mp_cache_recurrent_stash,
    mp_cache_recurrent_unstash,
    mp_cache_recurrent_clear,
    new_checkpoint_handle,
)



def mp_cache_short_conv_recurrent_rewind(local_context: dict, cache_id: int, slot: int, last_history, num_tokens):
    recurrent_modules = local_context["recurrent_modules"]
    for module in recurrent_modules:
        l = module.tp_recurrent_lookup[cache_id]
        l.rewind(slot, last_history, num_tokens)


class ShortConvState:

    def __init__(
        self,
        cache: Cache,
        slot: int,
        position: int,
        clear: bool = True,
        stashed: dict = None,
        test_state: bool = False,
        exported: bool = False,
    ):
        self.slot = slot
        self.position = position
        self.cache = cache
        self.last_history = 0
        self.exported = exported

        if not exported:
            assert test_state or position == 0 or stashed is not None, \
                "State must be new, restored from checkpoint or marked as a test state."

            if clear and stashed is None:
                if not self.cache.model.loaded_tp:
                    for l in self.cache.get_all_recurrent_layers().values():
                        l.clear(slot)
                else:
                    self.cache.model.tp_dispatch_all(mp_cache_recurrent_clear, (id(self.cache), self.slot))

            if stashed is not None:
                self.unstash(stashed)

            self.checkpoint_size = sum(
                l.get_checkpoint_size()
                for l in self.cache.get_all_recurrent_layers().values()
            )


    def free(self):
        self.cache.release_state(self)


    def rewind(self, num_tokens: int):
        if not self.cache.model.loaded_tp:
            for l in self.cache.get_all_recurrent_layers().values():
                l.rewind(self.slot, self.last_history, num_tokens)
        else:
            self.cache.model.tp_dispatch_all(mp_cache_short_conv_recurrent_rewind, (id(self.cache), self.slot, self.last_history, num_tokens))
        self.position -= num_tokens
        self.last_history = 0


    def rollback_capacity(self):
        # The state advances destructively; rewinding is only possible immediately after a forward pass that
        # recorded per-token history (speculative decoding), never at an arbitrary later point
        return 0


    def stash(self):
        stashed = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size
        }
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                stashed[k] = l.stash(self.slot)
        else:
            cp_handle = new_checkpoint_handle()
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_stash, (id(self.cache), cp_handle, self.slot))
            stashed["tp_handle"] = cp_handle
        return stashed


    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                l.unstash(self.slot, stashed[k])
        else:
            cp_handle = stashed["tp_handle"]
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_unstash, (id(self.cache), cp_handle, self.slot))


    def post_advance(self):
        pass


    def tp_export(self):
        return ShortConvState(
            cache = id(self.cache),
            slot = self.slot,
            position = self.position,
            exported = True,
        )


    def reset(self):
        self.position = 0


class ShortConvLayerState:

    def __init__(
        self,
        module: ShortConv,
        max_batch_size: int,
        max_history: int,
        cache_id: int,
    ):
        self.module = module
        self.conv_state = torch.empty(
            (max_batch_size, module.hidden_size, module.conv_kernel_size + max_history),
            dtype = torch.half,
            device = "meta",
        )
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id


    def get_checkpoint_size(self):
        return self.module.hidden_size * self.module.conv_kernel_size * 2


    def storage_size(self):
        return self.conv_state.numel() * self.conv_state.element_size()


    def alloc(self, device):
        self.conv_state = torch.empty_like(self.conv_state, device = device)
        self.conv_state.zero_()
        self.device = device


    def free(self):
        self.conv_state = torch.empty_like(self.conv_state, device = "meta")
        self.device = None


    def clear(self, idx: int):
        if self.device is not None:
            self.conv_state[idx].zero_()


    def get_state_tensors(self):
        return (self.conv_state,)


    def rewind(self, slot: int, last_history: int, num_tokens: int):
        assert num_tokens <= last_history
        cdim = self.module.conv_kernel_size
        if last_history > 0:
            c_state = self.conv_state[slot, :, :cdim]
            p = self.conv_state.shape[-1] - num_tokens
            c_state_rewind = self.conv_state[slot, :, p - cdim : p]
            temp = c_state_rewind.clone()
            c_state.copy_(temp)


    def stash(self, slot, position: int = 0):
        cdim = self.module.conv_kernel_size
        return self.conv_state[slot, :, :cdim].cpu()


    def unstash(self, slot, stashed, position: int = 0):
        cdim = self.module.conv_kernel_size
        self.conv_state[slot, :, :cdim].copy_(stashed)


    def tp_export(self, plan):
        return {
            "cls": ShortConvLayerState,
            "args": {
                "cache_id": self.cache_id,
                "max_history": self.max_history,
                "max_batch_size": self.max_batch_size,
            }
        }


class ShortConv(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        conv_kernel_size: int,
        key_in: str,
        key_conv: str,
        key_out: str,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        select_hq_bits: int = 0,
    ):
        super().__init__(config, key, None)
        self.module_name = "ShortConv"
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.conv_kernel_size = conv_kernel_size
        self.out_dtype = out_dtype

        self.in_proj = Linear(
            config,
            f"{key}.{key_in}",
            hidden_size,
            hidden_size * 3,
            qmap = qmap + ".input" if qmap else None,
            out_dtype = torch.half,
            select_hq_bits = select_hq_bits,
            qgroup = key + ".in",
        )
        self.out_proj = Linear(
            config,
            f"{key}.{key_out}",
            hidden_size,
            hidden_size,
            qmap = qmap + ".output" if qmap else None,
            out_dtype = self.out_dtype,
            select_hq_bits = select_hq_bits,
            qgroup = key + ".out",
        )
        self.register_submodule(self.in_proj)
        self.register_submodule(self.out_proj)

        self.conv1d_weight = None
        self.conv1d_bias = None
        self.key_conv1d_weight = f"{key}.{key_conv}.weight"
        self.key_conv1d_bias = f"{key}.{key_conv}.bias"

        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = ShortConvLayerState

        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.tp_reduce = False
        self.has_split_cache = False


    @override
    def optimizer_targets(self):
        return [[self.in_proj.optimizer_targets(), self.out_proj.optimizer_targets()]]


    def load_local(self, device, **kwargs):
        for rl in self.recurrent_layers:
            rl.alloc(device)


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        self.conv1d_weight = self.config.stc.get_tensor(self.key_conv1d_weight, self.device, optional = False, allow_bf16 = True)
        self.conv1d_bias = self.config.stc.get_tensor(self.key_conv1d_bias, self.device, optional = True, allow_bf16 = True)
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        self.conv1d_weight = None
        self.conv1d_bias = None
        for cl in self.recurrent_layers:
            cl.free()
        super().unload()


    def _causal_conv1d(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
        recurrent_slots: torch.Tensor | None,
        history: bool,
        params: dict,
    ) -> torch.Tensor:
        bsz, dim, seqlen = x.shape
        weight = self.conv1d_weight.squeeze(1).contiguous()
        bias = self.conv1d_bias

        if conv_state is None:
            assert not history
            conv_state = torch.zeros((bsz, dim, self.conv_kernel_size), dtype = x.dtype, device = x.device)
            recurrent_slots_cpu = list(range(bsz))
        else:
            recurrent_slots_cpu = get_for_device(params, "recurrent_slots", "cpu").tolist()

        ys = []
        for i, slot in enumerate(recurrent_slots_cpu):
            state = conv_state[slot].unsqueeze(0)
            y = torch.cat([state[:, :, :self.conv_kernel_size], x[i].unsqueeze(0)], dim = -1)
            if history:
                write_size = min(conv_state.shape[-1], y.shape[-1])
                conv_state[slot, :, -write_size:].copy_(y[0, :, -write_size:])
            else:
                conv_state[slot, :, :self.conv_kernel_size].copy_(y[0, :, -self.conv_kernel_size:])
            y = F.conv1d(y.to(weight.dtype), weight.unsqueeze(1), bias, padding = 0, groups = dim)
            ys.append(y[:, :, -seqlen:].to(x.dtype))

        return torch.cat(ys, dim = 0).transpose(1, 2).contiguous()


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        save_history = params.get("recurrent_history", False)

        rsg = params.get("recurrent_states")
        if rsg:
            recurrent_slots = get_for_device(params, "recurrent_slots", self.device)
            layer_instance = (self.layer_idx, params.get("layer_instance", 0))
            if rsg[0].exported:
                rsl = self.tp_recurrent_lookup[rsg[0].cache]
            else:
                rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
            (conv_state,) = rsl.get_state_tensors()
            save_history = save_history
        else:
            recurrent_slots = None
            conv_state = None
            save_history = False

        bc_x = self.in_proj.forward(x, params)
        b, c, x = bc_x.chunk(3, dim = -1)
        conv_input = (b * x).transpose(1, 2).contiguous()
        conv_out = self._causal_conv1d(conv_input, conv_state, recurrent_slots, save_history, params)
        y = c * conv_out
        y = self.out_proj.forward(y.view(bsz, seqlen, self.hidden_size), params)

        if self.tp_reduce:
            params["backend"].all_reduce(y)

        return to2(y, out_dtype, self.out_dtype)


    @override
    def get_tensors(self):
        t = super().get_tensors()
        for x, k in [
            (self.conv1d_weight, self.key_conv1d_weight),
            (self.conv1d_bias, self.key_conv1d_bias),
        ]:
            if x is not None:
                t[k] = x
        return t
