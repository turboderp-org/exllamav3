from __future__ import annotations
from typing_extensions import override
import torch.nn.functional as F
from .. import Module, Linear, LayerNorm, RMSNorm
from ...model import Config
import torch
from ...util.tensor import get_for_device, to2


class Gemma4VisionPatchEmbedder(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        patch_dim: int,
        position_embedding_size: int,
        out_dtype: torch.dtype = torch.float
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.position_embedding_size = position_embedding_size
        self.position_embedding_key = f"{key}.position_embedding_table"
        self.position_embedding_table = None
        self.position_embedding_numel = 0
        self.out_dtype = out_dtype

        self.input_proj = Linear(
            config = config,
            key = f"{key}.input_proj",
            in_features = patch_dim,
            out_features = hidden_size,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.input_proj)


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.position_embedding_table = self.config.stc.get_tensor(
            self.position_embedding_key,
            device,
            float2half = True,
            allow_bf16 = True,
        )
        self.position_embedding_numel = self.position_embedding_table.numel()


    @override
    def unload(self):
        super().unload()
        self.position_embedding_table = None
        self.position_embedding_numel = 0


    @override
    def weights_numel(self):
        return super().weights_numel() + self.position_embedding_numel


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        # Pixel values in range -1..1
        x = 2.0 * (x - 0.5)
        y = self.input_proj.forward(x.half(), params, out_dtype = torch.half)

        # Position IDs
        position_ids = get_for_device(params, "position_ids", self.device)
        pos_x = position_ids[..., 0].reshape(-1)
        pos_y = position_ids[..., 1].reshape(-1)

        # Table is 2x 1D learned embeddings (for x and y tile index, respectively)
        table = self.position_embedding_table
        pos_emb = table[0].index_select(0, pos_x) + table[1].index_select(0, pos_y)
        pos_emb = pos_emb.view(position_ids.shape[0], position_ids.shape[1], self.hidden_size)

        y = to2(y, out_dtype, self.out_dtype)
        y += pos_emb
        return y


class Gemma4VisionPooler(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        key_std_bias: str | None = None,
        key_std_scale: str | None = None,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.std_bias_key = f"{key}.{key_std_bias}" if key_std_bias else None
        self.std_scale_key = f"{key}.{key_std_scale}" if key_std_scale else None
        self.std_bias = None
        self.std_scale = None
        self.numel = 0
        assert bool(self.std_bias_key) == bool(self.std_scale_key), \
            "Must have both std_bias and std_scale or neither"
        self.has_bias_scale = bool(self.std_bias_key)


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.has_bias_scale:
            self.std_bias = self.config.stc.get_tensor(self.std_bias_key, device, allow_bf16 = True)
            self.std_scale = self.config.stc.get_tensor(self.std_scale_key, device, allow_bf16 = True)


    @override
    def weights_numel(self):
        return 2 * self.hidden_size if self.has_bias_scale else 0


    @override
    def unload(self):
        super().unload()
        self.std_bias = None
        self.std_scale = None


    @override
    def optimizer_targets(self):
        return []


    @override
    def get_tensors(self):
        if self.has_bias_scale:
            return {
                self.std_bias_key: self.std_bias.contiguous(),
                self.std_scale_key: self.std_scale.contiguous(),
            }
        else:
            return {}


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        position_ids = get_for_device(params, "position_ids", self.device)
        output_length = int(params["image_output_length"])
        if output_length > x.shape[1]:
            raise ValueError(f"Cannot pool {x.shape[1]} patches to {output_length} soft tokens.")

        if x.shape[1] != output_length:
            input_seq_len = x.shape[1]
            k = int((input_seq_len // output_length) ** 0.5)
            k_squared = k ** 2
            if k_squared * output_length != input_seq_len:
                raise ValueError(f"Cannot pool {x.shape} to {output_length}: {k=}^2 mismatch")
            max_x = position_ids[..., 0].max(dim = -1, keepdim = True)[0] + 1
            kernel_idxs = torch.div(position_ids, k, rounding_mode = "floor")
            kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
            weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared
            x = weights.transpose(1, 2) @ x

        x = x * (self.hidden_size ** 0.5)

        if self.has_bias_scale:
            x -= self.std_bias
            x *= self.std_scale

        return to2(x, out_dtype, torch.float)


class Gemma4UnifiedVisionEmbedder(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        patch_dim: int,
        mm_embed_dim: int,
        norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.mm_embed_dim = mm_embed_dim
        self.pos_embedding_key = f"{key}.pos_embedding"
        self.pos_embedding = None
        self.pos_embedding_numel = 0

        self.patch_ln1 = LayerNorm(config, f"{key}.patch_ln1", norm_eps, out_dtype = torch.half)
        self.patch_dense = Linear(
            config = config,
            key = f"{key}.patch_dense",
            in_features = patch_dim,
            out_features = mm_embed_dim,
            qmap = None,
            out_dtype = torch.float,
            pad_to = 1,
        )
        self.patch_ln2 = LayerNorm(config, f"{key}.patch_ln2", norm_eps, out_dtype = torch.float)
        self.pos_norm = LayerNorm(config, f"{key}.pos_norm", norm_eps, out_dtype = torch.half)

        self.register_submodule(self.patch_ln1)
        self.register_submodule(self.patch_dense)
        self.register_submodule(self.patch_ln2)
        self.register_submodule(self.pos_norm)


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.pos_embedding = self.config.stc.get_tensor(
            self.pos_embedding_key,
            device,
            float2half = True,
            allow_bf16 = True,
        )
        self.pos_embedding_numel = self.pos_embedding.numel()


    @override
    def unload(self):
        super().unload()
        self.pos_embedding = None
        self.pos_embedding_numel = 0


    @override
    def weights_numel(self):
        return super().weights_numel() + self.pos_embedding_numel


    @override
    def get_tensors(self):
        return {
            self.pos_embedding_key: self.pos_embedding.contiguous(),
        }


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        x = self.patch_ln1.forward(x.to(self.patch_dense.inner.weight.dtype), params)
        x = self.patch_dense.forward(x, params)
        x = self.patch_ln2.forward(x, params)

        position_ids = get_for_device(params, "position_ids", self.device)
        clamped = position_ids.clamp(min = 0).long()
        valid = (position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
        axes = torch.arange(2, device = position_ids.device)
        pos_embs = (self.pos_embedding[clamped, axes] * valid).sum(-2)

        x = self.pos_norm.forward(x + pos_embs, params)
        return x


class Gemma4PerLayerTokenEmbedding(Module):
    """
    Token-identity component of Gemma4 per-layer embeddings (PLE). Looks up input IDs in the packed
    embed_tokens_per_layer table, scaled by sqrt(ple_dim), and stores the result in params for the
    downstream projection module. Hidden states pass through unchanged. The table is large
    (vocab_size x num_layers * ple_dim), so like the main token embedding it prefers CPU placement.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        vocab_size: int,
        num_layers: int,
        ple_dim: int,
    ):
        super().__init__(config, key, None)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.embedding = None
        self._numel = vocab_size * num_layers * ple_dim
        # Multiplier survives the bf16 round trip the reference implementation applies to it
        self.multiplier = torch.tensor(ple_dim ** 0.5, dtype = torch.bfloat16).float().item()
        self.caps.update({
            "prefer_cpu": True,
            # Stays loaded during conversion so per_layer_quant_preamble can recompute the
            # combined per-layer inputs for every quantized module (same pattern as ValueEmbeddings)
            "retain_during_quant": True,
        })

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(self.key + ".weight", self.device, float2half = True, allow_bf16 = True)
        self._numel = weight.numel()
        self.embedding = torch.nn.Embedding(
            self.vocab_size,
            self.num_layers * self.ple_dim,
            device = "meta"
        )
        self.embedding.weight = torch.nn.Parameter(weight)

    @override
    def unload(self):
        self.device = None
        self.embedding = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.weight": self.embedding.weight.data.contiguous()
        }

    @override
    def weights_numel(self):
        return self._numel

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # g4_input_ids is set by prepare_inputs() on the normal forward path; the conversion and
        # autosplit pipelines call modules directly and supply plain input_ids instead
        input_ids = params.get("g4_input_ids")
        if input_ids is None:
            input_ids = params["input_ids"]
        input_ids = input_ids.to(self.embedding.weight.device)
        # Indexed multimodal embeddings use synthetic token IDs beyond the vocabulary; the PLE
        # table has no rows for them, so those positions look up the pad token (ID 0) instead,
        # matching the reference implementation
        if input_ids.numel() and input_ids.max().item() >= self.vocab_size:
            input_ids = torch.where(
                input_ids < self.vocab_size,
                input_ids,
                torch.zeros_like(input_ids),
            )
        tok = self.embedding(input_ids)
        tok = tok.view(*input_ids.shape, self.num_layers, self.ple_dim)
        tok = tok.half() * self.multiplier
        params["g4_ple_tok"] = tok
        return x


class Gemma4PerLayerProjection(Module):
    """
    Context-aware component of Gemma4 PLE. Projects the (scaled) input embeddings to per-layer space,
    normalizes, and combines with the token-identity component computed by Gemma4PerLayerTokenEmbedding.
    Stores the combined (bsz, seq_len, num_layers, ple_dim) tensor in params; hidden states pass
    through unchanged.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        key_norm: str,
        hidden_size: int,
        num_layers: int,
        ple_dim: int,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ple_dim = ple_dim

        self.caps.update({
            # Stays loaded during conversion so per_layer_quant_preamble can recompute the
            # combined per-layer inputs for every quantized module
            "retain_during_quant": True,
        })

        # The PLE projections (this one plus the per-layer gate/projection pairs, ~83M params /
        # ~165 MB at fp16 for E4B) are deliberately kept unquantized. Measured on E4B at 4.0 bpw,
        # giving them qmaps roughly doubled KL vs the bf16 reference (mean 0.034 -> 0.061, top-1
        # 90.5% -> 87.9%) while saving only ~1% of file size: the per-layer gates feed the
        # residual stream directly and their errors compound across all layers, and the extra
        # tensors dilute the bit budget for the attention/MLP weights.
        self.proj = Linear(
            config = config,
            key = key,
            in_features = hidden_size,
            out_features = num_layers * ple_dim,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.norm = RMSNorm(
            config = config,
            key = key_norm,
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.half,
        )
        self.register_submodule(self.proj)
        self.register_submodule(self.norm)

    @override
    def optimizer_targets(self):
        return []

    # The norm's tensor lives under a sibling key prefix (per_layer_projection_norm), which the
    # default per-module-key prefix scan in the compile step would miss
    @override
    def get_compile_sizes(self, stc):
        return stc.get_tensor_sizes(self.key) + stc.get_tensor_sizes(self.norm.key)

    @override
    def get_compile_tensors(self, stc):
        return stc.get_tensors(self.key, allow_bf16 = True) | stc.get_tensors(self.norm.key, allow_bf16 = True)

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        ctx = self.proj.forward(x.half(), params)
        ctx = ctx * (self.hidden_size ** -0.5)
        ctx = ctx.view(*x.shape[:-1], self.num_layers, self.ple_dim)
        ctx = self.norm.forward(ctx, params, out_dtype = torch.half)
        tok = get_for_device(params, "g4_ple_tok", ctx.device)
        ple = (ctx + tok) * (2 ** -0.5)
        params["g4_ple"] = ple
        return x


class Gemma4PerLayerInject(Module):
    """
    Per-decoder-layer PLE injection: gated projection of this layer's per-layer input added to the
    residual stream at the end of the transformer block (before the layer scalar is applied).
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        ple_dim: int,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.layer_idx = layer_idx

        # TODO: unquantized for now (see Gemma4PerLayerProjection); pending evaluation
        self.gate = Linear(
            config = config,
            key = f"{key}.per_layer_input_gate",
            in_features = hidden_size,
            out_features = ple_dim,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.proj = Linear(
            config = config,
            key = f"{key}.per_layer_projection",
            in_features = ple_dim,
            out_features = hidden_size,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.norm = RMSNorm(
            config = config,
            key = f"{key}.post_per_layer_input_norm",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
        )
        self.register_submodule(self.gate)
        self.register_submodule(self.proj)
        self.register_submodule(self.norm)

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        ple = get_for_device(params, "g4_ple", self.device)
        ple = ple[..., self.layer_idx, :]
        y = self.gate.forward(x.half(), params)
        y = F.gelu(y, approximate = "tanh")
        y = y * ple
        y = self.proj.forward(y, params)
        y = self.norm.forward(y, params, out_dtype = torch.float)
        x += y
        return x
