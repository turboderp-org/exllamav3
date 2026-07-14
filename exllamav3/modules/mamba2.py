from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config
from ..util.tensor import get_for_device, buffered_arange, to2, g_tensor_cache
from . import Module, Linear
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation
from .gated_rmsnorm import GatedRMSNorm
from .gated_delta_net import GDNLayerState
from .gated_delta_net_fn import causal_conv1d_update

try:
    from fla.ops.simple_gla import chunk_simple_gla
except ModuleNotFoundError:
    chunk_simple_gla = None


def torch_recurrent_mamba2(
    mixed_xbc, dt, g, D, initial_state,
    num_k_heads, num_v_heads, k_head_dim, v_head_dim,
):
    """
    For reference, not used. mixed_xbc: [bsz, seqlen, v_dim + 2*k_dim] bf16 (conv output),
    dt/g: [bsz, seqlen, num_v_heads], D: [num_v_heads]
    """
    bsz, seqlen, _ = mixed_xbc.shape
    group = num_v_heads // num_k_heads
    v_dim = num_v_heads * v_head_dim
    k_dim = num_k_heads * k_head_dim

    x, B, C = torch.split(mixed_xbc.float(), [v_dim, k_dim, k_dim], dim = -1)
    x = x.view(bsz, seqlen, num_v_heads, v_head_dim)
    B = B.view(bsz, seqlen, num_k_heads, k_head_dim)
    C = C.view(bsz, seqlen, num_k_heads, k_head_dim)
    dt = dt.float()
    g = g.float()
    D = D.float()

    state = (
        torch.zeros(bsz, num_v_heads, k_head_dim, v_head_dim, device = mixed_xbc.device)
        if initial_state is None else initial_state.float().clone()
    )
    out = torch.zeros(bsz, seqlen, num_v_heads, v_head_dim, device = mixed_xbc.device)

    for s in range(seqlen):
        decay = g[:, s].exp()                                       # (b, H)
        B_h = B[:, s].repeat_interleave(group, dim = 1)             # (b, H, K)
        C_h = C[:, s].repeat_interleave(group, dim = 1)             # (b, H, K)
        u = x[:, s] * dt[:, s].unsqueeze(-1)                        # (b, H, V)
        state = state * decay[:, :, None, None] + B_h.unsqueeze(-1) * u.unsqueeze(-2)
        out[:, s] = torch.einsum("bhkv,bhk->bhv", state, C_h) + x[:, s] * D[None, :, None]

    return out, state


class Mamba2(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        state_size: int,
        rms_norm_eps: float,
        conv_kernel_size: int,
        dt_limit: tuple = (0.0, float("inf")),
        key_in: str | None = "in_proj",
        key_conv1d: str | None = "conv1d",
        key_a_log: str | None = "A_log",
        key_dt_bias: str | None = "dt_bias",
        key_d: str | None = "D",
        key_norm: str | None = "norm",
        key_o: str | None = "out_proj",
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        select_hq_bits: int = 0,
        in_proj: Linear | None = None,
        o_proj: Linear | None = None,
        norm: GatedRMSNorm | None = None,
        conv1d_weight: torch.Tensor | None = None,
        conv1d_bias: torch.Tensor | None = None,
        a_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        d_skip: torch.Tensor | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Mamba2"

        self.q_priority = 1 + select_hq_bits
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.dt_limit = dt_limit

        # The Mamba2 recurrence maps onto the gated delta rule with B->k, C->q, x->v: the SSM
        # state groups take the role of GQA k-heads. Attribute names shared with GatedDeltaNet
        # so GDNLayerState and the conv/state cache machinery apply unchanged
        self.num_v_heads = num_heads
        self.num_k_heads = num_groups
        self.k_head_dim = state_size
        self.v_head_dim = head_dim
        self.num_v_groups = num_heads // num_groups if num_groups else 0
        self.v_dim = num_heads * head_dim
        self.k_dim = num_groups * state_size

        # Conv covers [x, B, C] in checkpoint order
        self.fdim_qkv = self.v_dim + 2 * self.k_dim
        self.proj_dim = self.v_dim + self.fdim_qkv + num_heads

        self.out_dtype = out_dtype

        if self.num_k_heads == 0:
            return

        assert self.num_v_heads % self.num_k_heads == 0, \
            "Mamba2 num_heads must be divisible by n_groups"

        if in_proj is not None:
            self.in_proj = in_proj
        else:
            self.in_proj = Linear(
                config,
                f"{key}.{key_in}",
                hidden_size,
                self.proj_dim,
                qmap = qmap + ".input" if qmap else None,
                out_dtype = torch.float,
                trim_padded_out = True,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".in",
            )
        self.register_submodule(self.in_proj)

        if o_proj is not None:
            self.o_proj = o_proj
        else:
            self.o_proj = Linear(
                config,
                f"{key}.{key_o}",
                self.v_dim,
                hidden_size,
                qmap = qmap + ".output" if qmap else None,
                out_dtype = self.out_dtype,
                trim_padded_out = True,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".o",
            )
        self.register_submodule(self.o_proj)

        if norm is not None:
            self.norm = norm
        else:
            self.norm = GatedRMSNorm(
                config,
                f"{key}.{key_norm}",
                self.rms_norm_eps,
                out_dtype = torch.half,
                groups = self.num_k_heads,
                gate_first = True,
            )
        self.register_submodule(self.norm)

        if dt_bias is not None:
            self.a_log = a_log
            self.dt_bias = dt_bias
            self.d_skip = d_skip
            self.conv1d_weight = conv1d_weight
            self.conv1d_bias = conv1d_bias
            self.key_a_log = None
            self.key_dt_bias = None
            self.key_d = None
            self.key_conv1d_weight = None
            self.key_conv1d_bias = None
        else:
            self.a_log = None
            self.dt_bias = None
            self.d_skip = None
            self.conv1d_weight = None
            self.conv1d_bias = None
            self.key_a_log = f"{key}.{key_a_log}"
            self.key_dt_bias = f"{key}.{key_dt_bias}"
            self.key_d = f"{key}.{key_d}"
            self.key_conv1d_weight = f"{key}.{key_conv1d}.weight"
            self.key_conv1d_bias = f"{key}.{key_conv1d}.bias"

        self.a_log_f = None
        self.dt_bias_f = None
        self.d_skip_f = None
        self.conv1d_weight_flat = None

        self.caps.update({
            "recurrent_cache": True
        })
        self.layer_state_cls = GDNLayerState

        self.bc = None
        self.bsz1_pa_args = []
        self.derived_filled = False

        # TP: rank's head offset into the replicated dt section of the in_proj output
        self.tp_dt_first = 0

        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.tp_reduce = False
        self.has_split_cache = False


    @override
    def optimizer_targets(self):
        return [[self.in_proj.optimizer_targets()]]


    def load_local(self, device, **kwargs):
        if self.num_k_heads == 0:
            return
        for rl in self.recurrent_layers:
            rl.alloc(device)

        # Derived tensors are allocated here so the BC below can hold references, but filled on
        # the first forward pass (deferred loading: the source tensors aren't materialized yet)
        nh = self.num_v_heads
        self.a_log_f = torch.empty((nh,), dtype = torch.float, device = device)
        self.dt_bias_f = torch.empty((nh,), dtype = torch.float, device = device)
        self.d_skip_f = torch.empty((nh,), dtype = torch.float, device = device)
        self.conv1d_weight_flat = torch.empty(
            (self.fdim_qkv, self.conv_kernel_size), dtype = torch.bfloat16, device = device
        )
        self.derived_filled = False

        is_quantized = (
            device != torch.device("cpu") and
            self.in_proj.quant_type == "exl3" and self.in_proj.inner.bc is not None and
            self.o_proj.quant_type == "exl3" and self.o_proj.inner.bc is not None
        )

        if is_quantized:
            f = self.fdim_qkv
            hd = self.v_head_dim
            k_in = self.in_proj.in_features       # padded widths of the quantized projections
            n_in = self.in_proj.out_features
            n_out = self.o_proj.out_features
            o_dtype = self.out_dtype or torch.half

            self.bsz1_pa_args = [
                (device, (1, 1, n_in), torch.float, "m2_proj"),
                (device, (1, f, 1), torch.bfloat16, "m2_mx"),
                (device, (1, 1, nh), torch.bfloat16, "m2_dt"),
                (device, (1, 1, nh), torch.float, "m2_g"),
                (device, (1, 1, f), torch.bfloat16, "m2_co"),
                (device, (1, 1, nh, hd), torch.bfloat16, "m2_ca"),
                (device, (1, 1, nh * hd), torch.half, "m2_caf"),
            ]
            proj, mixed_xbc, dt, g, conv_out, core, core_f = \
                (g_tensor_cache.get(*arg) for arg in self.bsz1_pa_args)

            # Zero-padded staging statics for padded projection dims; the pad columns are never
            # written after this
            xp = None
            if k_in != self.hidden_size:
                xp = g_tensor_cache.get(device, (1, k_in), torch.half, "m2_xp")
                xp.zero_()
            yp = None
            if n_out != self.hidden_size:
                yp = g_tensor_cache.get(device, (1, n_out), o_dtype, "m2_yp")

            self.bc = ext.BC_Mamba2(
                xp = xp,
                proj = proj,
                mixed_xbc = mixed_xbc,
                dt = dt,
                g = g,
                conv_out = conv_out,
                core_attn_out = core,
                core_attn_out_f = core_f,
                yp = yp,
                in_proj = self.in_proj.inner.bc,
                o_proj = self.o_proj.inner.bc,
                dt_bias = self.dt_bias_f,
                a_log = self.a_log_f,
                d_skip = self.d_skip_f,
                dt_min = self.dt_limit[0],
                dt_max = self.dt_limit[1],
                num_k_heads = self.num_k_heads,
                num_v_heads = self.num_v_heads,
                k_head_dim = self.k_head_dim,
                v_head_dim = self.v_head_dim,
                hidden_size = self.hidden_size,
                conv1d_weight = self.conv1d_weight_flat,
                conv1d_bias = self.conv1d_bias,
                norm = self.norm.bc,
                dt_first = self.tp_dt_first,
            )


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        self.a_log = self.config.stc.get_tensor(self.key_a_log, self.device, optional = False, allow_bf16 = True)
        self.dt_bias = self.config.stc.get_tensor(self.key_dt_bias, self.device, optional = False, allow_bf16 = True)
        self.d_skip = self.config.stc.get_tensor(self.key_d, self.device, optional = False, allow_bf16 = True)
        self.conv1d_weight = self.config.stc.get_tensor(self.key_conv1d_weight, self.device, optional = False, allow_bf16 = True)
        self.conv1d_bias = self.config.stc.get_tensor(self.key_conv1d_bias, self.device, optional = True, allow_bf16 = True)
        # Derived copies (fp32 head params, flattened conv weight) are made lazily on the first
        # forward pass: with deferred loading the tensors aren't materialized yet
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        self.bc = None
        self.bsz1_pa_args = []
        self.derived_filled = False
        self.a_log = None
        self.dt_bias = None
        self.d_skip = None
        self.a_log_f = None
        self.dt_bias_f = None
        self.d_skip_f = None
        self.conv1d_weight = None
        self.conv1d_weight_flat = None
        self.conv1d_bias = None
        for cl in self.recurrent_layers:
            cl.free()
        super().unload()


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.num_k_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                params["backend"].all_reduce(x, False)
            return to2(x, out_dtype, self.out_dtype)

        bsz, seqlen, _ = x.shape
        save_history = params.get("recurrent_history", False)

        # Post load, fill the fp32 working copies for the dt/decay math and the flattened conv
        # weight (allocated in load_local, source tensors materialized by now)
        if not self.derived_filled:
            self.a_log_f.copy_(self.a_log)
            self.dt_bias_f.copy_(self.dt_bias)
            self.d_skip_f.copy_(self.d_skip)
            self.conv1d_weight_flat.copy_(self.conv1d_weight.squeeze(1))
            self.derived_filled = True

        # Previous state
        rsg = params.get("recurrent_states")
        if rsg:
            recurrent_slots = get_for_device(params, "recurrent_slots", self.device)
            layer_instance = (self.layer_idx, params.get("layer_instance", 0))
            if rsg[0].exported:
                rsl = self.tp_recurrent_lookup[rsg[0].cache]
            else:
                rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
            conv_state, recurrent_state = rsl.get_state_tensors()
            save_state = True
        else:
            recurrent_slots = None
            conv_state, recurrent_state = None, None
            save_state = False
            save_history = False  # no SD without prior state, for simplicity

        # Fused C++ path for single-token decode. Runs the entire layer in one call, replayed
        # through an internal CUDA graph from the third invocation on
        if (
            self.bc is not None and bsz == 1 and seqlen == 1 and
            save_state and not save_history and
            recurrent_slots is not None and
            x.dtype == torch.float16 and x.is_contiguous()
        ):
            y = torch.empty_like(x, dtype = self.out_dtype or torch.half)
            self.bc.run_bsz1(x, y, conv_state, recurrent_state, recurrent_slots)
            if self.tp_reduce:
                params["backend"].all_reduce(y)
            return to2(y, out_dtype, self.out_dtype)

        # Input projection, flat split [z, xBC, dt]. Under TP the dt section is replicated on
        # every rank (its per-group row count never tile-aligns); slice out this rank's heads
        proj = self.in_proj.forward(x, params)
        z = proj[..., : self.v_dim]
        xbc = proj[..., self.v_dim : self.v_dim + self.fdim_qkv]
        dt_base = self.v_dim + self.fdim_qkv + self.tp_dt_first
        dt_raw = proj[..., dt_base : dt_base + self.num_v_heads].contiguous()

        # Discretization: dt = clamp(softplus(dt_raw + dt_bias)), g = dt * A
        dt = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
        g = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)
        ext.mamba2_dt_op(dt_raw, self.dt_bias_f, self.a_log_f, dt, g, self.dt_limit[0], self.dt_limit[1])

        # Convolution over [x, B, C]
        mixed_xbc = xbc.transpose(1, 2).to(torch.bfloat16).contiguous()
        mixed_xbc = causal_conv1d_update(
            mixed_qkv = mixed_xbc,
            conv_state = conv_state,
            recurrent_slots = recurrent_slots,
            conv1d_weight = self.conv1d_weight_flat,
            conv1d_bias = self.conv1d_bias,
            history = save_history,
            params = params,
        )

        # SSM
        if seqlen >= self.num_v_heads and chunk_simple_gla is not None and not save_history:
            core_attn_out = self.ssd_chunked(mixed_xbc, dt, g, recurrent_state, save_state, params, bsz, seqlen)
        else:
            core_attn_out = torch.empty(
                (bsz, seqlen, self.num_v_heads, self.v_head_dim),
                dtype = torch.bfloat16,
                device = self.device,
            )
            if recurrent_state is None:
                recurrent_state = torch.zeros(
                    (bsz, 1, self.num_v_heads, self.k_head_dim, self.v_head_dim),
                    dtype = torch.float,
                    device = self.device
                )
            ext.cuda_recurrent_mamba2(
                mixed_xbc,
                g,
                dt,
                self.d_skip_f,
                recurrent_state,
                core_attn_out,
                self.num_k_heads,
                self.num_v_heads,
                self.k_head_dim,
                self.v_head_dim,
                recurrent_slots,
                save_history,
            )

        # Grouped, gated norm: y = groupnorm(y * silu(z)) * w
        gs = self.v_dim // self.num_k_heads
        core_attn_out = core_attn_out.view(bsz, seqlen, self.num_k_heads, gs)
        z = z.to(torch.bfloat16).contiguous().view(bsz, seqlen, self.num_k_heads, gs)
        core_attn_out = self.norm.forward(core_attn_out, params, gate = z)
        core_attn_out = core_attn_out.view(bsz, seqlen, self.v_dim)

        # Output projection
        x = self.o_proj.forward(core_attn_out, params)

        # TP reduction
        if self.tp_reduce:
            params["backend"].all_reduce(x)

        return to2(x, out_dtype, self.out_dtype)


    def ssd_chunked(self, mixed_xbc, dt, g, recurrent_state, save_state, params, bsz, seqlen):
        # Mamba2 (SSD) prefill: chunk_simple_gla computes the same recurrence without the
        # delta-rule correction. No GQA support, so B/C expand to all heads
        x_v, B, C = torch.split(mixed_xbc, [self.v_dim, self.k_dim, self.k_dim], dim = -1)
        x_v = x_v.view(bsz, seqlen, self.num_v_heads, self.v_head_dim)
        q = C.view(bsz, seqlen, self.num_k_heads, self.k_head_dim).repeat_interleave(self.num_v_groups, dim = 2)
        k = B.view(bsz, seqlen, self.num_k_heads, self.k_head_dim).repeat_interleave(self.num_v_groups, dim = 2)
        v = x_v * dt.unsqueeze(-1)

        recurrent_slots_cpu = get_for_device(params, "recurrent_slots", "cpu", None)
        if recurrent_slots_cpu is None:
            recurrent_slots_cpu = buffered_arange(bsz, mixed_xbc.device)
        core_attn_out = []
        for i, s in enumerate(recurrent_slots_cpu.tolist()):
            state = recurrent_state[s, 0].unsqueeze(0) if recurrent_state is not None else None
            core_attn, new_state = chunk_simple_gla(
                q[i:i + 1], k[i:i + 1], v[i:i + 1].contiguous(),
                g = g[i:i + 1],
                scale = 1.0,
                initial_state = state,
                output_final_state = save_state,
            )
            if save_state and state is not None:
                state.copy_(new_state)
            core_attn_out.append(core_attn)

        core_attn_out = torch.cat(core_attn_out, dim = 0)

        # Skip connection y += D * x, in fp32 like the fused kernels
        core_attn_out = (
            core_attn_out.float() + x_v.float() * self.d_skip_f.view(1, 1, -1, 1)
        ).to(torch.bfloat16)
        return core_attn_out


    @override
    def get_tensors(self):
        t = super().get_tensors()
        for x, k in [
            (self.a_log, self.key_a_log),
            (self.dt_bias, self.key_dt_bias),
            (self.d_skip, self.key_d),
            (self.conv1d_weight, self.key_conv1d_weight),
            (self.conv1d_bias, self.key_conv1d_bias),
        ]:
            if x is not None and k is not None:
                t[k] = x
        return t


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.in_proj.storage_size()
        storage += self.o_proj.storage_size()
        for cl in self.recurrent_layers:
            storage += cl.storage_size()
        overhead_d = 0
        overhead_d += self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        overhead_s += (self.proj_dim + self.fdim_qkv) * torch.float.itemsize
        recons = max(
            self.in_proj.recons_size(),
            self.o_proj.recons_size(),
        )
        # Split unit is the state group (B/C stay aligned with their heads); the split boundaries
        # must land on the 128-block Hadamard grid of the quantized in_proj in every section, so
        # groups pair up until both the v-side and k-side per-channel widths are 128-aligned.
        # (The dt section never aligns and is replicated instead.)
        gs_v = self.num_v_groups * self.v_head_dim
        channel_width = 1
        channels_to_split = self.num_k_heads
        while (channel_width * gs_v) % 128 != 0 or (channel_width * self.k_head_dim) % 128 != 0:
            assert channels_to_split % 2 == 0, \
                "Model's state groups cannot divide into 128-aligned channels"
            channel_width *= 2
            channels_to_split //= 2
        tpa = TPAllocation(
            key = self.key,
            channel_width = channel_width,
            channel_unit = "K-heads",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = channels_to_split,
            limit_key = "linear_attn"
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        from ..model.model_tp_shared import TPTensorWrapper

        def _export(child):
            nonlocal producer
            if child is None:
                return None
            if isinstance(child, torch.Tensor):
                return TPTensorWrapper.tp_export(child, plan, producer)
            else:
                return child.tp_export(plan, producer)

        return {
            "cls": Mamba2,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "hidden_size": self.hidden_size,
                "head_dim": self.v_head_dim,
                "state_size": self.k_head_dim,
                "rms_norm_eps": self.rms_norm_eps,
                "conv_kernel_size": self.conv_kernel_size,
                "dt_limit": self.dt_limit,
                "out_dtype": self.out_dtype,
            },
            "num_heads": self.num_v_heads,
            "num_groups": self.num_k_heads,
            **{name: _export(getattr(self, name, None)) for name in (
                "in_proj",
                "o_proj",
                "norm",
                "conv1d_weight",
                "conv1d_bias",
                "a_log",
                "dt_bias",
                "d_skip",
            )},
            "device": self.device,
            "recurrent_layers": [
                rl.tp_export(plan) for rl in self.recurrent_layers
            ]
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        from ..model.model_tp_shared import TPTensorWrapper
        key = exported["kwargs"]["key"]
        head_dim = exported["kwargs"]["head_dim"]
        state_size = exported["kwargs"]["state_size"]
        global_heads = exported["num_heads"]
        global_groups = exported["num_groups"]
        G = global_heads // global_groups
        device = local_context["device"]
        first, last, unit = plan[key]
        assert unit == "K-heads"
        num_groups = last - first
        num_heads = num_groups * G

        gs_v = G * head_dim                     # v-side channels per group
        ks = state_size                         # k-side channels per group
        d_inner = global_heads * head_dim
        k_dim = global_groups * state_size

        # in_proj output layout [z, x, B, C, dt]: z/x/B/C split by group, dt replicated (its
        # 8..12 rows per group never land on the quantized 128-block grid); each rank slices its
        # heads out of the full dt activation instead. Every slice must start and end on
        # 128-multiples of the original tensor, so the dt range runs through the padding columns
        # (dt + pad completes the tensor's final blocks)
        padded_out = exported["in_proj"]["kwargs"]["out_features"] if exported.get("in_proj") else 0
        z_split = (True, first * gs_v, last * gs_v) \
            if num_groups else None
        x_split = (True, d_inner + first * gs_v, d_inner + last * gs_v) \
            if num_groups else None
        b_split = (True, 2 * d_inner + first * ks, 2 * d_inner + last * ks) \
            if num_groups else None
        c_split = (True, 2 * d_inner + k_dim + first * ks, 2 * d_inner + k_dim + last * ks) \
            if num_groups else None
        dt_full = (True, 2 * d_inner + 2 * k_dim, padded_out) \
            if num_groups else None

        # conv covers [x, B, C]
        cx_split = (True, first * gs_v, last * gs_v) \
            if num_groups else None
        cb_split = (True, d_inner + first * ks, d_inner + last * ks) \
            if num_groups else None
        cc_split = (True, d_inner + k_dim + first * ks, d_inner + k_dim + last * ks) \
            if num_groups else None

        o_split = (False, first * gs_v, last * gs_v) \
            if num_groups else None
        h_split = (True, first * G, last * G) \
            if num_groups else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        def _import_split_n(name, splits):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split_n(local_context, exported[name], plan, splits) \
                if splits[0] and exported.get(name) else None

        norm = None
        if num_groups and exported.get("norm"):
            # The gated group norm splits per channel; the local module normalizes over its own
            # groups, and the BC must be built with the local group count
            exported["norm"]["kwargs"]["groups"] = num_groups
            norm = GatedRMSNorm.tp_import_split(
                local_context, exported["norm"], plan, (first * gs_v, last * gs_v))

        module = Mamba2(
            config = None,
            **exported["kwargs"],
            num_heads = num_heads,
            num_groups = num_groups,
            in_proj = _import_split_n("in_proj", [z_split, x_split, b_split, c_split, dt_full]),
            o_proj = _import_split("o_proj", o_split),
            norm = norm,
            conv1d_weight = _import_split_n("conv1d_weight", [cx_split, cb_split, cc_split]),
            conv1d_bias = _import_split_n("conv1d_bias", [cx_split, cb_split, cc_split]),
            a_log = _import_split("a_log", h_split),
            dt_bias = _import_split("dt_bias", h_split),
            d_skip = _import_split("d_skip", h_split),
        )
        module.tp_dt_first = first * G

        if num_groups:
            recurrent_layers = exported["recurrent_layers"]
            if len(recurrent_layers):
                module.has_split_cache = True
                for rl in exported["recurrent_layers"]:
                    rli = rl["cls"](module, **rl["args"])
                    module.recurrent_layers.append(rli)
                    module.tp_recurrent_lookup[rl["args"]["cache_id"]] = rli

        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True

        module.load_local(device)
        torch.cuda.synchronize()
        return module
