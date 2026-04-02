from __future__ import annotations

from typing_extensions import override
import math
import torch
import torch.nn.functional as F

from ..cache import CacheableState
from ..model.config import Config
from ..model.model_tp_alloc import TPAllocation
from ..util.tensor import to2
from . import Linear, Module


class NemotronHMambaRecurrentState(CacheableState):

    def __init__(
        self,
        position: int | None = 0,
        positions: list[int] | None = None,
        last_conv_state: torch.Tensor | None = None,
        last_ssm_state: torch.Tensor | None = None,
        batched: bool = False,
    ):
        super().__init__()
        self.position = position
        self.positions = positions
        self.last_conv_state = last_conv_state
        self.last_ssm_state = last_ssm_state
        self.batched = batched

    @override
    def stash(self):
        return NemotronHMambaRecurrentState(
            self.position,
            self.positions,
            None if self.last_conv_state is None else self.last_conv_state.cpu(),
            None if self.last_ssm_state is None else self.last_ssm_state.cpu(),
            self.batched,
        )

    @override
    def unstash(self, device):
        return NemotronHMambaRecurrentState(
            self.position,
            self.positions,
            None if self.last_conv_state is None else self.last_conv_state.to(device, non_blocking = True),
            None if self.last_ssm_state is None else self.last_ssm_state.to(device, non_blocking = True),
            self.batched,
        )

    @override
    def get_size(self):
        total = 0
        if self.last_conv_state is not None:
            total += self.last_conv_state.element_size() * self.last_conv_state.numel()
        if self.last_ssm_state is not None:
            total += self.last_ssm_state.element_size() * self.last_ssm_state.numel()
        return total

    def collect_batch(self, batch: list[NemotronHMambaRecurrentState]):
        positions = [b.position for b in batch]
        if batch[0].last_conv_state is None or batch[0].last_ssm_state is None:
            return NemotronHMambaRecurrentState(None, positions, None, None, True)
        lcs = torch.cat([b.last_conv_state for b in batch], dim = 0)
        lss = torch.cat([b.last_ssm_state for b in batch], dim = 0)
        return NemotronHMambaRecurrentState(None, positions, lcs, lss, True)

    def distribute_batch(self, batch: list[NemotronHMambaRecurrentState]):
        for i, b in enumerate(batch):
            if self.last_conv_state is not None:
                b.last_conv_state.copy_(self.last_conv_state[i:i+1, ...])
            if self.last_ssm_state is not None:
                b.last_ssm_state.copy_(self.last_ssm_state[i:i+1, ...])
            b.position = self.positions[i]


class NemotronHMamba2(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        ssm_state_size: int,
        n_groups: int,
        conv_kernel_size: int,
        rms_norm_eps: float,
        time_step_limit: tuple[float, float] = (0.0, math.inf),
        key_in_proj: str = "in_proj",
        key_out_proj: str = "out_proj",
        key_a_log: str = "A_log",
        key_d: str = "D",
        key_dt_bias: str = "dt_bias",
        key_conv1d: str = "conv1d",
        key_norm: str = "norm",
        qmap: str | None = None,
        out_dtype: torch.dtype | None = torch.float,
    ):
        super().__init__(config, key, None)
        self.module_name = "NemotronHMamba2"

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ssm_state_size = ssm_state_size
        self.n_groups = n_groups
        self.conv_kernel_size = conv_kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.out_dtype = out_dtype
        self.time_step_limit = time_step_limit

        self.intermediate_size = self.num_heads * self.head_dim
        self.group_size = self.intermediate_size // self.n_groups
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.projection_size = self.intermediate_size + self.conv_dim + self.num_heads

        self.key_a_log = f"{key}.{key_a_log}"
        self.key_d = f"{key}.{key_d}"
        self.key_dt_bias = f"{key}.{key_dt_bias}"
        self.key_conv1d_weight = f"{key}.{key_conv1d}.weight"
        self.key_conv1d_bias = f"{key}.{key_conv1d}.bias"
        self.key_norm_weight = f"{key}.{key_norm}.weight"

        self.in_proj = Linear(
            config = config,
            key = f"{key}.{key_in_proj}",
            in_features = hidden_size,
            out_features = self.projection_size,
            qmap = qmap + ".input" if qmap else None,
        )
        self.out_proj = Linear(
            config = config,
            key = f"{key}.{key_out_proj}",
            in_features = self.intermediate_size,
            out_features = hidden_size,
            qmap = qmap + ".output" if qmap else None,
            out_dtype = out_dtype,
        )
        self.register_submodule(self.in_proj)
        self.register_submodule(self.out_proj)

        self.a_log = None
        self.d = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None
        self.norm_weight = None

        self.caps.update({"recurrent_cache": True})

    @override
    def optimizer_targets(self):
        return [[self.in_proj.optimizer_targets(), self.out_proj.optimizer_targets()]]

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.a_log = self.config.stc.get_tensor(self.key_a_log, self.device)
        self.d = self.config.stc.get_tensor(self.key_d, self.device)
        self.dt_bias = self.config.stc.get_tensor(self.key_dt_bias, self.device)
        self.conv1d_weight = self.config.stc.get_tensor(self.key_conv1d_weight, self.device)
        self.conv1d_bias = self.config.stc.get_tensor(self.key_conv1d_bias, self.device, optional = True)
        self.norm_weight = self.config.stc.get_tensor(self.key_norm_weight, self.device)

    @override
    def unload(self):
        super().unload()
        self.a_log = None
        self.d = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None
        self.norm_weight = None

    @override
    def weights_numel(self):
        total = super().weights_numel()
        for tensor in (
            self.a_log,
            self.d,
            self.dt_bias,
            self.conv1d_weight,
            self.conv1d_bias,
            self.norm_weight,
        ):
            if tensor is not None:
                total += tensor.numel()
        return total

    def _empty_conv_state(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        return torch.zeros(
            (batch_size, self.conv_dim, self.conv_kernel_size),
            dtype = dtype,
            device = device,
        )

    def _empty_ssm_state(self, batch_size: int, device: torch.device):
        return torch.zeros(
            (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
            dtype = torch.float,
            device = device,
        )

    def _conv_step(self, x_t: torch.Tensor, conv_state: torch.Tensor):
        conv_state = torch.cat([conv_state[..., 1:], x_t.unsqueeze(-1).to(conv_state.dtype)], dim = -1)
        weight = self.conv1d_weight.squeeze(1).unsqueeze(0)
        y = (conv_state * weight).sum(dim = -1)
        if self.conv1d_bias is not None:
            y = y + self.conv1d_bias
        y = F.silu(y)
        return y, conv_state

    def _repeat_groups(self, x: torch.Tensor):
        repeats = self.num_heads // self.n_groups
        if repeats == 1:
            return x
        return x.repeat_interleave(repeats, dim = 1)

    def _gated_group_rmsnorm(self, x: torch.Tensor, gate: torch.Tensor):
        input_dtype = x.dtype
        x = x.float() * F.silu(gate.float())
        prefix = x.shape[:-1]
        x = x.view(*prefix, self.n_groups, self.group_size)
        variance = x.square().mean(dim = -1, keepdim = True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        x = x.view(*prefix, self.intermediate_size)
        x = x * self.norm_weight.float()
        return x.to(input_dtype)

    def _ssm_step(
        self,
        hidden_t: torch.Tensor,
        b_t: torch.Tensor,
        c_t: torch.Tensor,
        dt_t: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        hidden_heads = hidden_t.view(-1, self.num_heads, self.head_dim).float()

        dt = F.softplus(dt_t.float().unsqueeze(-1) + self.dt_bias.float().view(1, self.num_heads, 1))
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

        b_t = b_t.view(-1, self.n_groups, self.ssm_state_size).float()
        c_t = c_t.view(-1, self.n_groups, self.ssm_state_size).float()
        b_t = self._repeat_groups(b_t)
        c_t = self._repeat_groups(c_t)

        a = -torch.exp(self.a_log.float()).view(1, self.num_heads, 1, 1)
        d_a = torch.exp(dt.unsqueeze(-1) * a)
        d_b = dt.unsqueeze(-1) * b_t.unsqueeze(2)

        ssm_state = ssm_state.float() * d_a + d_b * hidden_heads.unsqueeze(-1)
        y = torch.matmul(ssm_state, c_t.unsqueeze(-1)).squeeze(-1)
        y = y + hidden_heads * self.d.float().view(1, self.num_heads, 1)
        return y.reshape(-1, self.intermediate_size), ssm_state

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape
        projected_states = self.in_proj.forward(x, params)
        if projected_states.shape[-1] != self.projection_size:
            projected_states = projected_states[..., :self.projection_size].contiguous()
        _, _, gate, conv_input, dt = projected_states.split(
            [0, 0, self.intermediate_size, self.conv_dim, self.num_heads],
            dim = -1,
        )

        rs = None
        save_state = False
        recurrent_states = params.get("recurrent_states")
        if recurrent_states is not None:
            rs = recurrent_states.get(self.layer_idx)
            save_state = rs is not None

        if rs is not None and rs.last_conv_state is not None and rs.last_ssm_state is not None:
            conv_state = rs.last_conv_state
            ssm_state = rs.last_ssm_state
        else:
            conv_state = self._empty_conv_state(batch_size, x.dtype, x.device)
            ssm_state = self._empty_ssm_state(batch_size, x.device)

        outputs = []
        for i in range(seq_len):
            conv_out, conv_state = self._conv_step(conv_input[:, i, :], conv_state)
            hidden_t, b_t, c_t = torch.split(
                conv_out,
                [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
                dim = -1,
            )
            y_t, ssm_state = self._ssm_step(hidden_t, b_t, c_t, dt[:, i, :], ssm_state)
            y_t = self._gated_group_rmsnorm(y_t.to(x.dtype), gate[:, i, :])
            outputs.append(y_t)

        y = torch.stack(outputs, dim = 1)
        y = self.out_proj.forward(y, params)

        if save_state:
            rs.last_conv_state = conv_state
            rs.last_ssm_state = ssm_state
            if not rs.batched:
                rs.position += seq_len
            else:
                rs.positions = [r + seq_len for r in rs.positions]

        return to2(y, out_dtype, self.out_dtype)

    @override
    def get_tensors(self):
        t = super().get_tensors()
        for tensor, key in (
            (self.a_log, self.key_a_log),
            (self.d, self.key_d),
            (self.dt_bias, self.key_dt_bias),
            (self.conv1d_weight, self.key_conv1d_weight),
            (self.conv1d_bias, self.key_conv1d_bias),
            (self.norm_weight, self.key_norm_weight),
        ):
            if tensor is not None:
                t[key] = tensor
        return t

    def new_recurrent_state(self):
        return NemotronHMambaRecurrentState()

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError()

    def tp_export(self, plan, producer):
        raise NotImplementedError()

    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        raise NotImplementedError()
