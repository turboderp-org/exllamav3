from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ..activation_scaling import scale_activation
from ..block_sparse_mlp import BlockSparseMLP
from ..factorized_attention import FactorizedDifferentialAttention
from ..geodesic import geodesic_update
from ..linear import Linear
from ..mamba3 import Mamba3
from ..mlp import MLP
from ..tensor_bundle import TensorBundle
from ...util.backend import ROCM, gfx_arch
from . import olala_bf16
from .olala_moe import decode_moe, pack_projections


def projection(config, key, input_size, output_size, qmap, **kwargs):
    return Linear(
        config = config, key = key,
        in_features = input_size, out_features = output_size,
        trim_padded_out = True, qmap = qmap, **kwargs,
    )


class OlalaRouter(TensorBundle):
    def __init__(self, config, key):
        super().__init__(config, key, {"weight": (config.num_experts, config.hidden_size)})
        self.inner = None

    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        self.inner = SimpleNamespace(weight = self.tensors["weight"].T.contiguous())

    def unload(self):
        self.inner = None
        super().unload()

    def storage_size(self):
        return self.weights_numel() * torch.float32.itemsize

    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError


class OlalaLatentIn(Linear):
    def scale_input(self, latent, params, out_dtype = None):
        # Per-token scaling protects the half-precision expert activations and
        # routed sum, including outliers in otherwise small layers.
        return scale_activation(
            latent, self.config.moe_activation_multiplier, squared = True, out_dtype = out_dtype or torch.half,
            use_triton = "capture" not in params,
        )

    def forward(self, x, params, out_dtype = None):
        latent = super().forward(x, params, out_dtype = torch.float)
        scaled, scale = self.scale_input(latent, params, out_dtype)
        params["olala_latent_scale"] = scale
        return scaled


class OlalaLatentOut(Linear):
    def capture_H(self, x, params):
        # The scaled routed sum can contain FP16 subnormals. Capture its Gram
        # matrix in FP32 so half-precision tensor cores do not flush them.
        super().capture_H(x.float(), params)

    def forward(self, x, params, out_dtype = None):
        result = super().forward(x, params, out_dtype = torch.float)
        # The routed experts are bias-free and use ReLU squared. Restore the
        # degree-two scale in FP32 before the shared expert is added.
        scale = params.pop("olala_latent_scale")
        return (result * scale.reshape(*result.shape[:-1], 1)).to(out_dtype or torch.float)


class OlalaMoE(BlockSparseMLP):
    def __init__(self, config, key):
        latent_in = OlalaLatentIn(
            config = config, key = f"{key}.down_proj", qmap = "block.mlp.input",
            in_features = config.hidden_size, out_features = config.latent_size,
            qgroup = f"{key}.latent", select_hq_bits = 2,
        )
        latent_out = OlalaLatentOut(
            config = config, key = f"{key}.up_proj", qmap = "block.mlp.latent_out",
            in_features = config.latent_size, out_features = config.hidden_size,
            qgroup = f"{key}.latent", select_hq_bits = 2,
        )
        super().__init__(
            config = config, key = key,
            hidden_size = config.hidden_size, intermediate_size = config.expert_size,
            num_experts = config.num_experts, num_experts_per_tok = config.num_experts_per_tok,
            latent_size = config.latent_size, latent_in = latent_in, latent_out = latent_out,
            key_up = "experts.{expert_idx}.up_proj", key_down = "experts.{expert_idx}.down_proj",
            key_up_split = "experts.experts.weight", key_down_split = "experts.output_experts.weight",
            transpose_fused_weights = False,
            routing_gate = OlalaRouter(config, f"{key}.moe_gate"),
            key_e_score_bias = "expert_bias", router_type = "ds3", n_group = 1, topk_group = 1,
            routed_scaling_factor = config.expert_scale, activation_fn = "relu2",
            qmap = "block.mlp", out_dtype = torch.float,
            shared_experts = MLP(
                config = config, key = f"{key}.shared_experts",
                hidden_size = config.hidden_size, intermediate_size = config.shared_size,
                key_up = "fc_1", key_down = "fc_2", activation_fn = "relu2",
                qmap = "block.mlp", out_dtype = torch.float,
            ),
        )
        self.decode_linears = (latent_in, self.shared_experts.ups[0], latent_out, self.shared_experts.downs[0])
        self.bf16_codebook = None
        self.decode_weights = None

    def load(self, device, **kwargs):
        pack = (
            device is not None and torch.device(device).type == "cuda"
            and (not ROCM or gfx_arch(torch.device(device)) == "gfx1100")
            and os.environ.get("EXL3_OLALA_FUSED_MOE", "1") != "0"
            and not self.config.infer_params.no_reconstruct
            and (self.config.hidden_size, self.config.latent_size, self.config.shared_size) == (1536, 384, 1536)
            and self.shared_experts.num_slices == 1
        )
        self.bf16_codebook = None
        self.decode_weights = None
        for linear in self.decode_linears:
            linear.eager_exl3 = pack
        super().load(device, **kwargs)
        if (pack and self.bc is not None and not self.cpu_offload and self.cpu_split_first is None
                and not self.bc_sh_exp and all(
                    linear.quant_type == "exl3" and linear.inner.bias is None and not linear.softcap
                    and linear.pre_scale == linear.post_scale == 1 for linear in self.decode_linears
                )):
            with torch.cuda.device(device):
                if (ROCM and self.support_fused and not self.tp_reduce and self.num_experts == 256
                        and self.routing_device is None and self.num_local_experts == self.num_experts
                        and self.intermediate_size == 768 and self.num_experts_per_tok == 6
                        and all(linear.pre_scale == 1 for linear in self.ups + self.downs) and all(
                            linear.mul1 and linear.K in (2, 3, 4, 5, 6)
                            for linear in (self.multi_up, self.multi_down)
                        )):
                    self.bf16_codebook = olala_bf16.codebook(device)
                dtype = torch.bfloat16 if self.bf16_codebook is not None else torch.float16
                self.decode_weights = pack_projections(self.decode_linears, dtype)

    def unload(self):
        self.bf16_codebook = None
        self.decode_weights = None
        super().unload()

    def load_routing(self, **kwargs):
        bias = self.e_score_correction_bias
        if bias is None or bias.shape != (self.num_experts,):
            raise ValueError(f"{self.key}.expert_bias: expected one routing bias per expert")
        super().load_routing(**kwargs)
        self.routing_cfg.gate_tensor_t = self.routing_gate.tensors["weight"].contiguous()

    def forward(self, x, params, out_dtype = None):
        if (self.decode_weights is not None and x.ndim == 3 and x.shape[0] == 1
                and x.shape[1] > 0 and x.shape[2] == self.hidden_size and x.is_contiguous()
                and not any(key in params for key in (
                    "capture", "ovr", "reconstruct", "quant_preserve", "activate_all_experts",
                    "autosplit_measure", "tp_warmup", "capture_sens", "moe_affinity",
                ))
                and not any(linear.lora_a_tensors for linear in self.decode_linears)):
            with torch.cuda.device(x.device):
                if self.bf16_codebook is not None:
                    return olala_bf16.forward(self, x, params).to(out_dtype or torch.float32)
                if x.shape[1] == 1 and params.get("recurrent_slots") is not None and "prefill" not in params:
                    return decode_moe(self, x.half(), params).to(out_dtype or torch.float32)
        try:
            return super().forward(x.half(), params, out_dtype)
        finally:
            params.pop("olala_latent_scale", None)


class OlalaBlock(TensorBundle):
    def __init__(self, config, key, layer_idx, layer_type):
        super().__init__(config, key, {
            f"{component}.{name}": ()
            for component in ("geodesic_mixer", "geodesic_mlp")
            for name in ("scale", "bias", "prosres_scalar")
        })
        self.layer_idx = layer_idx
        if layer_type == "M":
            self.mixer = Mamba3(
                config, f"{key}.mixer", layer_idx,
                hidden_size = config.hidden_size, num_heads = 2 * config.hidden_size // config.mamba_head_dim,
                head_dim = config.mamba_head_dim, state_size = config.mamba_state_size, rank = config.mamba_rank,
                norm_eps = config.norm_eps, qmap = "block.mix.input",
            )
        else:
            self.mixer = FactorizedDifferentialAttention(
                config, f"{key}.mixer", layer_idx,
                hidden_size = config.hidden_size, num_q_heads = config.num_attention_heads,
                num_kv_heads = config.num_attention_heads - config.signal_heads, head_dim = config.head_dim,
                rank = config.attention_rank, sliding_window = config.sliding_window, softcap = config.attention_softcap,
                norm_eps = config.norm_eps, qmap = "block.mix.input",
            )
        self.gate = (
            projection(config, f"{key}.gate_proj", config.hidden_size, self.mixer.output_size, "block.mix.input")
            if layer_type == "V" else None
        )
        self.mixer_proj = projection(
            config, f"{key}.mixer_proj", self.mixer.output_size, config.hidden_size, "block.mix.output",
        )
        self.mlp = OlalaMoE(config, f"{key}.mlp")
        for module in (self.mixer, self.gate, self.mixer_proj, self.mlp):
            self.register_submodule(module)
        self.num_slices = self.mlp.num_slices

    def optimizer_targets(self):
        return [module.optimizer_targets() for module in self.modules]

    def forward(self, x, params, out_dtype = None):
        # Keep the reference arithmetic order for batched prefill and calibration.
        fused = x.shape[1] == 1 and params.get("recurrent_slots") is not None and "capture" not in params

        def residual_update(residual, update, site):
            return geodesic_update(
                residual, update, self.tensors[f"geodesic_{site}.scale"],
                self.tensors[f"geodesic_{site}.bias"], self.layer_idx + 1, use_triton = fused,
            )

        hidden = self.mixer.forward(x, params)
        if self.gate is not None:
            hidden = hidden * F.silu(self.gate.forward(x.half(), params, out_dtype = torch.float) + 1.15)
        # Some Mamba-3 mixer outputs exceed the FP16 range by orders of magnitude.
        # Divide before the bias-free projection and restore the scale afterward.
        scaled, scale = scale_activation(hidden, 1 / 64, use_triton = "capture" not in params)
        mixed = self.mixer_proj.forward(scaled, params, out_dtype = torch.float) * scale
        x = residual_update(x, mixed, "mixer")
        x = residual_update(x, self.mlp.forward(x, params), "mlp")
        return x.to(out_dtype or torch.float32)
