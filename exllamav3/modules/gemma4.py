from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from ..model.config import Config
from ..model.model_tp_alloc import TPAllocation
from ..util.rope import RoPE
from ..util.tensor import get_for_device
from ..constants import PAGE_SIZE
from ..util.tensor import to2
from . import Attention, GatedMLP, Linear, Module, RMSNorm, TransformerBlock


class Gemma4Attention(Attention):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        use_k_as_v: bool,
        v_norm: RMSNorm | None,
        **kwargs,
    ):
        key_v = kwargs.get("key_v")
        super().__init__(
            config=config,
            key=key,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            key_v=kwargs["key_k"] if use_k_as_v else key_v,
            **{k: v for k, v in kwargs.items() if k != "key_v"},
        )

        self.use_k_as_v = use_k_as_v
        if use_k_as_v:
            self.modules.remove(self.v_proj)
            self.v_proj = None

        self.v_norm = v_norm
        self.register_submodule(self.v_norm)


    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        k = self.k_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k, o]]


    def load_local(self, device, **kwargs):

        if self.num_kv_heads == 0:
            return

        for cl in self.cache_layers:
            cl.alloc(device)

        if self.rope_settings:
            self.rope = RoPE(
                device,
                self.rope_settings,
            )

        if self.q_norm and isinstance(self.q_norm, RMSNorm) and not self.q_norm.span_heads:
            self.q_norm_tensor = self.q_norm.weight.data
            self.k_norm_tensor = self.k_norm.weight.data


    def project_qkv(self, x: torch.Tensor, params: dict) -> tuple:
        bsz, q_len, _ = x.shape
        q = self.q_proj.forward(x, params)

        if self.interleaved_gate:
            q, g = torch.chunk(q.view(bsz, q_len, -1, self.head_dim * 2), 2, dim = -1)
            g = g.reshape(bsz, q_len, -1)
        elif self.g_proj:
            g = self.g_proj.forward(x, params)
        else:
            g = None

        k = self.k_proj.forward(x, params)
        v = k if self.v_proj is None else self.v_proj.forward(x, params)

        if self.v_norm is not None:
            v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)
            v = self.v_norm.forward(v, params, out_dtype = torch.half)
            v = v.view(bsz, q_len, self.num_kv_heads * self.head_dim)

        return q, k, v, g


    def _write_cache_pages(
        self,
        cache_tensor: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        bsz, seqlen, _, _ = values.shape
        for b in range(bsz):
            start = int(cache_seqlens[b])
            for t in range(seqlen):
                pos = start + t
                page = int(block_table[b, pos // PAGE_SIZE])
                page_pos = pos % PAGE_SIZE
                cache_tensor[page, page_pos].copy_(values[b, t], non_blocking = True)


    def _gather_cache_pages(
        self,
        cache_tensor: torch.Tensor,
        block_table: torch.Tensor,
        total_lens: torch.Tensor,
    ) -> torch.Tensor:
        bsz = block_table.shape[0]
        max_total = int(total_lens.max())
        gathered = torch.zeros(
            (bsz, max_total, self.num_kv_heads, self.head_dim),
            dtype = cache_tensor.dtype,
            device = cache_tensor.device,
        )
        for b in range(bsz):
            total = int(total_lens[b])
            if total == 0:
                continue
            num_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
            pages = block_table[b, :num_pages].long()
            flat = cache_tensor.index_select(0, pages).reshape(-1, self.num_kv_heads, self.head_dim)
            gathered[b, :total].copy_(flat[:total], non_blocking = True)
        return gathered


    def decode_flash_attn_fallback(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        cache = params.get("cache")
        block_table = get_for_device(params, "block_table", self.device)
        cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        causal = params.get("causal", True)

        q, k, v, g = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        if self.q_norm:
            if self.tp_span_heads_norm:
                q, k = self.apply_qk_norms_tp(q, k, params)
            elif not self.rope or self.q_norm_tensor is None:
                q = self.q_norm.forward(q, params, out_dtype = torch.half)
                k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(
                q, k,
                position,
                positions,
                position_ids,
                True,
                self.q_norm_tensor if not self.tp_span_heads_norm else None,
                self.k_norm_tensor if not self.tp_span_heads_norm else None,
                self.norm_eps,
                self.norm_constant_bias,
                inv_freq,
                self.post_rope_norm
            )

        cache_k, cache_v = cache.get_layer(self.layer_idx, cache_seqlens, block_table)
        self._write_cache_pages(cache_k, block_table, cache_seqlens, k)
        self._write_cache_pages(cache_v, block_table, cache_seqlens, v)

        total_lens = cache_seqlens + seqlen
        all_k = self._gather_cache_pages(cache_k, block_table, total_lens).transpose(1, 2)
        all_v = self._gather_cache_pages(cache_v, block_table, total_lens).transpose(1, 2)
        q = q.transpose(1, 2)

        max_total = all_k.shape[2]
        mask = torch.full(
            (bsz, 1, seqlen, max_total),
            torch.finfo(q.dtype).min,
            dtype = q.dtype,
            device = q.device,
        )
        for b in range(bsz):
            total = int(total_lens[b])
            if total == 0:
                continue
            if not causal:
                mask[b, 0, :, :total] = 0
                continue
            past = int(cache_seqlens[b])
            for qi in range(seqlen):
                upto = min(total, past + qi + 1)
                mask[b, 0, qi, :upto] = 0

        o = F.scaled_dot_product_attention(
            q,
            all_k,
            all_v,
            attn_mask = mask,
            enable_gqa = self.gqa,
            scale = self.sm_scale,
        )

        cache_layer = cache.layers[self.layer_idx]
        if hasattr(cache_layer, "qk"):
            cache.update_layer(self.layer_idx, cache_seqlens, block_table, k, v, seqlen)

        if self.headwise_gate:
            o *= g.sigmoid().unsqueeze(-1)
        o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.interleaved_gate:
            o *= g.sigmoid()

        return self.project_o(o, bsz, seqlen, params)


    def decode_sdpa_nc(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        causal = params.get("causal", True)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)

        q, k, v, g = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        if self.q_norm:
            if self.tp_span_heads_norm:
                q, k = self.apply_qk_norms_tp(q, k, params)
            elif not self.rope or self.q_norm_tensor is None:
                q = self.q_norm.forward(q, params, out_dtype = torch.half)
                k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(
                q, k,
                position,
                positions,
                position_ids,
                True,
                self.q_norm_tensor if not self.tp_span_heads_norm else None,
                self.k_norm_tensor if not self.tp_span_heads_norm else None,
                self.norm_eps,
                self.norm_constant_bias,
                inv_freq,
                self.post_rope_norm
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal = causal, enable_gqa = self.gqa, scale = self.sm_scale)

        if self.headwise_gate:
            o *= g.sigmoid().unsqueeze(-1)
        o = o.transpose(1, 2).contiguous().reshape((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.interleaved_gate:
            o *= g.sigmoid()

        return self.project_o(o, bsz, seqlen, params)


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if self.num_kv_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            return to2(x, out_dtype, self.out_dtype)

        bsz, seqlen, _ = x.shape
        attn_mode = params.get("attn_mode", "flash_attn_nc")

        if self.head_dim > 256:
            match attn_mode:
                case "flash_attn_nc":
                    x = self.decode_sdpa_nc(x, bsz, seqlen, params)
                case "flash_attn":
                    x = self.decode_flash_attn_fallback(x, bsz, seqlen, params)
                case "sdpa_nc":
                    x = self.decode_sdpa_nc(x, bsz, seqlen, params)
                case _:
                    raise ValueError(f"Unknown attn_mode: {attn_mode}")
            return to2(x, out_dtype, self.out_dtype)

        return super().forward(x, params, out_dtype)


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.q_proj.storage_size()
        storage += self.k_proj.storage_size()
        storage += self.o_proj.storage_size()
        for cl in self.cache_layers:
            storage += cl.storage_size()
        overhead_d = 0
        overhead_d += self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        for cl in self.cache_layers:
            overhead_s += cl.overhead_size()
        overhead_s += 2 * self.num_q_heads * self.head_dim * torch.half.itemsize
        overhead_s += 2 * self.num_kv_heads * self.head_dim * torch.half.itemsize
        recons = max(
            self.q_proj.recons_size(),
            self.k_proj.recons_size(),
            self.o_proj.recons_size(),
        )
        channel_width = 1
        channels_to_split = self.num_kv_heads
        while channel_width * self.head_dim < 128:
            assert channels_to_split % 2 == 0, \
                "Model's K/V heads cannot divide into 128-channel tensors"
            channel_width *= 2
            channels_to_split //= 2
        assert (channel_width * self.head_dim) % 128 == 0, \
            "Model's K/V heads cannot divide into 128-channel tensors"
        return [
            TPAllocation(
                key = self.key,
                channel_width = channel_width,
                channel_unit = "heads",
                storage_per_device = 0,
                storage_to_split = storage,
                overhead_per_device = overhead_d,
                overhead_to_split = overhead_s,
                recons_temp = recons,
                channels_to_split = channels_to_split,
                limit_key = "attn"
            )
        ]


class Gemma4TransformerBlock(TransformerBlock):

    def __init__(
        self,
        config: Config,
        key: str,
        **kwargs,
    ):
        super().__init__(config=config, key=key, **kwargs)
        self.layer_scalar_key = f"{key}.layer_scalar"
        self.layer_scalar = None
        self.layer_scalar_numel = 1


    def optimizer_targets(self):
        return super().optimizer_targets()


    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        layer_scalar = self.config.stc.get_tensor(
            self.layer_scalar_key,
            device,
            allow_bf16 = True,
            no_defer = True,
        )
        self.layer_scalar = nn.Parameter(layer_scalar, requires_grad = False)
        self.layer_scalar_numel = layer_scalar.numel()


    def unload(self):
        super().unload()
        self.layer_scalar = None


    def get_tensors(self):
        if self.layer_scalar is None:
            return {}
        return {
            self.layer_scalar_key: self.layer_scalar.data.contiguous(),
        }


    def weights_numel(self):
        return super().weights_numel() + self.layer_scalar_numel


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        x = super().forward(x, params, out_dtype = None)
        if self.layer_scalar is not None:
            x = x * self.layer_scalar.to(dtype = x.dtype)
        return to2(x, out_dtype, self.out_dtype)


class Gemma4Router(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scalar_root_size = hidden_size ** -0.5
        self.scale_key = f"{key}.scale"
        self.per_expert_scale_key = f"{key}.per_expert_scale"
        self.scale = None
        self.per_expert_scale = None
        self.extra_numel = 0

        self.norm = RMSNorm(
            config = config,
            key = f"{key}.norm",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
            unweighted = True,
        )
        self.proj = Linear(
            config = config,
            key = f"{key}.proj",
            in_features = hidden_size,
            out_features = num_experts,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.norm)
        self.register_submodule(self.proj)


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.scale = self.config.stc.get_tensor(
            self.scale_key,
            device,
            allow_bf16 = True,
            no_defer = True,
        )
        self.per_expert_scale = self.config.stc.get_tensor(
            self.per_expert_scale_key,
            device,
            allow_bf16 = True,
            no_defer = True,
        )
        self.extra_numel = self.scale.numel() + self.per_expert_scale.numel()


    @override
    def unload(self):
        super().unload()
        self.scale = None
        self.per_expert_scale = None
        self.extra_numel = 0


    @override
    def get_tensors(self):
        if self.scale is None or self.per_expert_scale is None:
            return {}
        return {
            self.scale_key: self.scale.data.contiguous(),
            self.per_expert_scale_key: self.per_expert_scale.data.contiguous(),
        }


    @override
    def weights_numel(self):
        return super().weights_numel() + self.extra_numel


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.norm.forward_torch(x, params, out_dtype = torch.float)
        y = y * self.scale.to(dtype = y.dtype)
        y *= self.scalar_root_size

        logits = self.proj.forward(y.half(), params, out_dtype = torch.float).float()
        probs = torch.softmax(logits, dim = -1)

        if params.get("activate_all_experts"):
            selected = (
                torch.arange(self.num_experts, dtype = torch.long, device = x.device)
                .repeat((x.shape[0], 1))
            )
            weights = probs * self.per_expert_scale.to(dtype = probs.dtype).unsqueeze(0)
            return selected, weights

        top_k_weights, top_k_index = torch.topk(
            probs,
            k = self.num_experts_per_tok,
            dim = -1,
        )
        top_k_weights /= top_k_weights.sum(dim = -1, keepdim = True)
        top_k_weights *= self.per_expert_scale[top_k_index].to(dtype = top_k_weights.dtype)
        return top_k_index, top_k_weights


class Gemma4Experts(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        qmap: str,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        self.gates = []
        self.ups = []
        self.downs = []

        fkey_gate_up = f"{key}.experts.gate_up_proj"
        fkey_down = f"{key}.experts.down_proj"

        for idx in range(num_experts):
            gate = Linear(
                config = config,
                key = f"{key}.experts.{idx}.gate_proj",
                fkey = fkey_gate_up,
                fidx = idx,
                frange = (0, intermediate_size),
                frange_dim = 1,
                in_features = hidden_size,
                out_features = intermediate_size,
                qmap = qmap + ".input",
                out_dtype = torch.half,
                transposed_load = True,
                transpose_fused_weights = True,
                ftranspose_after_load = False,
                qgroup = key + ".experts.gud",
            )
            up = Linear(
                config = config,
                key = f"{key}.experts.{idx}.up_proj",
                fkey = fkey_gate_up,
                fidx = idx,
                frange = (intermediate_size, intermediate_size * 2),
                frange_dim = 1,
                in_features = hidden_size,
                out_features = intermediate_size,
                qmap = qmap + ".input",
                out_dtype = torch.half,
                transposed_load = True,
                transpose_fused_weights = True,
                ftranspose_after_load = False,
                qgroup = key + ".experts.gud",
            )
            down = Linear(
                config = config,
                key = f"{key}.experts.{idx}.down_proj",
                fkey = fkey_down,
                fidx = idx,
                in_features = intermediate_size,
                out_features = hidden_size,
                qmap = qmap + f".{idx}.down",
                out_dtype = torch.float,
                allow_input_padding = True,
                transposed_load = True,
                transpose_fused_weights = True,
                ftranspose_after_load = False,
                qgroup = key + ".experts.gud",
            )

            self.gates.append(gate)
            self.ups.append(up)
            self.downs.append(down)
            self.register_submodule(gate)
            self.register_submodule(up)
            self.register_submodule(down)


    @override
    def optimizer_targets(self):
        g, u, d = [], [], []
        for m in self.gates:
            g += m.optimizer_targets()
        for m in self.ups:
            u += m.optimizer_targets()
        for m in self.downs:
            d += m.optimizer_targets()
        return [[g + u, d]]


    @override
    def forward(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        y = x.view(-1, self.hidden_size)
        final_hidden_states = torch.zeros_like(y, dtype = torch.float)

        num_tokens, top_k = selected_experts.shape
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1).to(dtype = torch.float)
        flat_tokens = torch.arange(num_tokens, device = y.device).repeat_interleave(top_k)

        order = flat_experts.argsort()
        expert_sorted = flat_experts[order]
        token_sorted = flat_tokens[order]
        weight_sorted = flat_weights[order]

        expert_count = torch.bincount(expert_sorted, minlength = self.num_experts)
        expert_ptr = torch.empty(self.num_experts + 1, dtype = torch.long, device = y.device)
        expert_ptr[0] = 0
        expert_ptr[1:] = expert_count.cumsum(0)

        for expert_idx in range(self.num_experts):
            start = int(expert_ptr[expert_idx])
            end = int(expert_ptr[expert_idx + 1])
            if start == end:
                continue

            top_x = token_sorted[start:end]
            current_state = y.index_select(0, top_x)
            gate = self.gates[expert_idx].forward(current_state, params)
            up = self.ups[expert_idx].forward(current_state, params)
            current_hidden_states = F.gelu(gate, approximate = "tanh") * up
            current_hidden_states = self.downs[expert_idx].forward(current_hidden_states, params)
            current_hidden_states *= weight_sorted[start:end].unsqueeze(1).to(dtype = current_hidden_states.dtype)
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        return to2(final_hidden_states.view(x.shape), out_dtype, torch.float)


class Gemma4MoEFeedForward(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size

        self.dense_mlp = GatedMLP(
            config = config,
            key = f"{key}.mlp",
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            key_up = "up_proj",
            key_gate = "gate_proj",
            key_down = "down_proj",
            qmap = "block.mlp",
            activation_fn = "gelu",
            interm_dtype = torch.half,
            out_dtype = torch.float,
        )
        self.dense_post_norm = RMSNorm(
            config = config,
            key = f"{key}.post_feedforward_layernorm_1",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
        )
        self.routed_pre_norm = RMSNorm(
            config = config,
            key = f"{key}.pre_feedforward_layernorm_2",
            rms_norm_eps = rms_norm_eps,
        )
        self.routed_post_norm = RMSNorm(
            config = config,
            key = f"{key}.post_feedforward_layernorm_2",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
        )
        self.router = Gemma4Router(
            config = config,
            key = f"{key}.router",
            hidden_size = hidden_size,
            num_experts = num_experts,
            num_experts_per_tok = num_experts_per_tok,
            rms_norm_eps = rms_norm_eps,
        )
        self.experts = Gemma4Experts(
            config = config,
            key = key,
            hidden_size = hidden_size,
            intermediate_size = moe_intermediate_size,
            num_experts = num_experts,
            qmap = "block.mlp",
        )

        self.register_submodule(self.dense_mlp)
        self.register_submodule(self.dense_post_norm)
        self.register_submodule(self.routed_pre_norm)
        self.register_submodule(self.routed_post_norm)
        self.register_submodule(self.router)
        self.register_submodule(self.experts)


    @override
    def optimizer_targets(self):
        return [self.dense_mlp.optimizer_targets(), self.experts.optimizer_targets()]


    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        dense = self.dense_mlp.forward(x, params)
        dense = self.dense_post_norm.forward(dense, params)

        selected_experts, routing_weights = self.router.forward(
            residual.view(-1, self.hidden_size),
            params,
        )
        routed_input = self.routed_pre_norm.forward(residual, params, out_dtype = torch.half)
        routed = self.experts.forward(routed_input, selected_experts, routing_weights, params)
        routed = self.routed_post_norm.forward(routed, params)

        y = dense + routed
        return to2(y, out_dtype, torch.float)


class Gemma4MoETransformerBlock(Gemma4TransformerBlock):

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.attn:
            y = self.attn_norm.forward(x, params, out_dtype = torch.half) if self.attn_norm else x.half()
            y = self.attn.forward(y, params)
            if params.get("prefill"):
                return x
            if self.attn_post_norm:
                y = self.attn_post_norm.forward(y, params)
            x = x + y

        if self.mlp:
            residual = x
            y = self.mlp_norm.forward(x, params, out_dtype = torch.half) if self.mlp_norm else x.half()
            y = self.mlp.forward(y, residual, params)
            if self.mlp_post_norm:
                y = self.mlp_post_norm.forward(y, params)
            x = residual + y

        if self.layer_scalar is not None:
            x = x * self.layer_scalar.to(dtype = x.dtype)

        return to2(x, out_dtype, self.out_dtype)
