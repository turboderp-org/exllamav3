"""
Load EXL3-quantized checkpoints into Transformers models.

The quantizer registers itself with Transformers' own `register_quantization_config` / `register_quantizer`
(the `quant_method: "exl3"` entry the converter writes into config.json selects it), replaces every
`nn.Linear` that has EXL3 tensors in the checkpoint with `Exl3HfLinear`, and lets the regular weight loader
fill that module's buffers. The forward pass runs ExLlamaV3's kernels; the backward pass is the matmul
against the dequantized weight, so gradients flow to the input (the quantized weights themselves are frozen).

Limits: every quantized tensor must correspond to one `nn.Linear` in the Transformers model, so models whose
HF implementation fuses projections that the converter splits (fused qkv / gate_up, stacked MoE experts,
sliced wide MLPs) are not loadable this way.

Requires transformers >= 5.
"""
import os
import re
from typing import Optional

import torch
import torch.nn

from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING, AUTO_QUANTIZATION_CONFIG_MAPPING
from transformers.utils.quantization_config import QuantizationConfigMixin

from exllamav3.loader import SafetensorsCollection
from exllamav3.modules.quant.exl3 import LinearEXL3

QUANT_METHOD = "exl3"


class Exl3LinearFunction(torch.autograd.Function):
    """
    y = x W + b through the EXL3 kernels; d/dx through the dequantized W. No gradient for the weights
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, module: "Exl3HfLinear"):
        ctx.module = module
        return module.inner.forward(x.contiguous(), {})

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        w = ctx.module.dequantized_weight()
        grad_x = grad_out.contiguous().to(w.dtype) @ w.T
        return grad_x, None


class Exl3HfLinear(torch.nn.Module):
    """
    nn.Module standing in for an nn.Linear, holding the EXL3 tensors as buffers the weight loader fills
    """

    def __init__(self, in_features: int, out_features: int, exl3_tensors: dict):
        """
        :param exl3_tensors:
            The module's tensors as listed by SafetensorsCollection.list_tensors: {key: {"shape", "torch_dtype"}}
            for "trellis", "suh", "svh", the optional "bias", the codebook markers "mcg" / "mul1", and the legacy
            packed-sign "su" / "sv"
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        keys = list(exl3_tensors.keys())
        l = keys[0].rfind(".")
        self.key = keys[0][:l]
        assert all(k[:l] == self.key for k in keys), "All tensors must belong to the same module"

        self.optional = []
        for name in ("trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1"):
            meta = exl3_tensors.get(f"{self.key}.{name}")
            if meta is None:
                setattr(self, name, None)
                continue
            self.register_buffer(name, torch.empty(meta["shape"], dtype = meta["torch_dtype"], device = "meta"))
        assert self.trellis is not None, f"{self.key}: no trellis tensor"

        # Some model implementations read .weight.dtype / .weight.device; a stand-in keeps them happy
        self.weight = torch.zeros((1,), dtype = torch.float16, device = "meta")

        self.inner = None
        self.cache_dequantized = False
        self._dequantized = None


    def finalize(self):
        """
        Call once the buffers are loaded
        """
        for name in ("suh", "svh", "bias"):
            t = getattr(self, name)
            if t is not None and t.dtype != torch.half:
                setattr(self, name, t.half())
        self.weight = torch.zeros((1,), dtype = torch.float16, device = self.trellis.device)

        self.inner = LinearEXL3(
            config = None,
            in_features = self.in_features,
            out_features = self.out_features,
            scale = None,
            su = self.su,
            sv = self.sv,
            suh = self.suh,
            svh = self.svh,
            trellis = self.trellis,
            mcg = self.mcg,
            mul1 = self.mul1,
            bias = self.bias,
            out_dtype = torch.float16,
            transformers_fix = True,
            key = self.key,
        )


    def dequantized_weight(self) -> torch.Tensor:
        """
        W as (in_features, out_features) fp16, reconstructed from the trellis. Reconstructed on every call
        unless `cache_dequantized` is set, which keeps a full fp16 copy per layer
        """
        if self._dequantized is not None:
            return self._dequantized
        w = self.inner.get_weight_tensor()
        if self.cache_dequantized:
            self._dequantized = w
        return w


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The kernels take fp16; models that run in bf16 or fp32 get the cast around each layer
        dtype = x.dtype
        y = Exl3LinearFunction.apply(x.half(), self)
        return y.to(dtype)


# Per-expert projection names the converter leaves in the checkpoint, (gate, up, down)
EXPERT_PROJ_NAMES = (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2"))


class Exl3HfExperts(torch.nn.Module):
    """
    Stands in for a Transformers v5 experts container (3D gate_up_proj / down_proj parameters and the
    forward(hidden_states, top_k_index, top_k_weights) interface) with one EXL3 linear per expert and
    projection, registered under the checkpoint's own names (experts.{e}.gate_proj etc.) so the weight
    loader fills them directly. The forward is the eager per-expert loop
    """

    def __init__(self, experts: torch.nn.Module, key: str, exl3_tensors: dict, names: tuple):
        super().__init__()
        self.num_experts = experts.num_experts
        self.act_fn = experts.act_fn
        self.key = key
        self.names = names
        H, I = experts.hidden_dim, experts.intermediate_dim
        for e in range(self.num_experts):
            m = torch.nn.Module()
            for name, (k, n) in zip(names, ((H, I), (H, I), (I, H))):
                prefix = f"{key}.{e}.{name}"
                group = {t: meta for t, meta in exl3_tensors.items() if t.startswith(prefix + ".")}
                if not group:
                    raise ValueError(f"{prefix}: expert projection missing from the checkpoint")
                m.add_module(name, Exl3HfLinear(k, n, group))
            self.add_module(str(e), m)
        self.linears = [getattr(getattr(self, str(e)), name) for e in range(self.num_experts) for name in names]


    def forward(self, hidden_states, top_k_index, top_k_weights):
        gate_name, up_name, down_name = self.names
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes = self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim = (-1, -2)), 0).nonzero()
        for e in hit:
            e = int(e[0])
            expert = getattr(self, str(e))
            top_k_pos, token_idx = torch.where(mask[e])
            x = hidden_states[token_idx]
            h = self.act_fn(getattr(expert, gate_name)(x)) * getattr(expert, up_name)(x)
            y = getattr(expert, down_name)(h) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, y.to(final.dtype))
        return final


class Exl3Config(QuantizationConfigMixin):
    """
    The converter's config.json entry: {"quant_method": "exl3", "version": ..., "bits": ...}. Everything but
    quant_method is informational
    """

    def __init__(self, version: str | None = None, bits: float | None = None, **kwargs):
        self.quant_method = QUANT_METHOD
        self.version = version
        self.bits = bits
        self.extra = kwargs


    def to_dict(self):
        return {"quant_method": self.quant_method, "version": self.version, "bits": self.bits, **self.extra}


class Exl3HfQuantizer(HfQuantizer):

    requires_calibration = False
    required_packages = ["exllamav3"]
    requires_parameters_quantization = False

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.postprocess_modules = []


    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("EXL3 models need a CUDA device")


    def update_dtype(self, dtype):
        return dtype if dtype is not None else torch.float16

    # (pre-5.0 name)
    def update_torch_dtype(self, torch_dtype):
        return self.update_dtype(torch_dtype)


    @staticmethod
    def _checkpoint_dir(model, checkpoint_files):
        if checkpoint_files:
            return os.path.dirname(os.path.abspath(checkpoint_files[0]))
        path = getattr(model, "name_or_path", None)
        if path and os.path.isdir(path):
            return path
        raise ValueError("Could not locate the EXL3 checkpoint files")


    @staticmethod
    def _key_renames(model):
        """
        Checkpoint key -> model key, as the weight loader will rename them: the plain renaming entries of the
        model's conversion mapping (legacy prefixes such as Gemma3's language_model.model -> model.language_model).
        Structural conversions (merged expert lists etc.) are not representable here, and those models are out
        of scope anyway
        """
        try:
            from transformers.conversion_mapping import get_model_conversion_mapping, WeightRenaming
            transforms = [t for t in get_model_conversion_mapping(model) if type(t) is WeightRenaming]
        except Exception:
            return lambda key: key

        def rename(key):
            for t in transforms:
                scope = getattr(t, "scope_prefix", None)
                if scope and not key.startswith(scope + "."):
                    continue
                for src, tgt in zip(t.source_patterns, t.target_patterns):
                    new = re.sub(src, tgt, key, count = 1)
                    if new != key:
                        key = new
                        break
            return key
        return rename


    def get_modules_to_replace(self, model, checkpoint_dir):
        stc = SafetensorsCollection(checkpoint_dir)
        rename = self._key_renames(model)
        linears = {name: m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}

        all_modules = dict(model.named_modules())

        # Every module prefix with a trellis tensor in the checkpoint, mapped into the model's namespace.
        # Per-expert projections belong to an experts container (3D parameters in Transformers v5),
        # replaced as a whole
        modules = {}
        experts = {}
        skipped = []
        expert_re = re.compile(r"^(.*)\.(\d+)\.(" + "|".join(n for names in EXPERT_PROJ_NAMES for n in names) + r")$")
        for key in stc.tensor_file_map:
            if not key.endswith(".trellis"):
                continue
            ck = key[:-len(".trellis")]
            if stc.has_tensor(ck + ".tile_order"):
                raise ValueError(f"{ck}: colmajor tile order is not supported by the Transformers integration")
            name = rename(ck)
            module = linears.get(name)
            if module is not None:
                modules[name] = Exl3HfLinear(module.in_features, module.out_features, stc.list_tensors(ck))
                continue
            m = expert_re.match(ck)
            container = all_modules.get(rename(m.group(1))) if m else None
            if container is not None and hasattr(container, "num_experts") and hasattr(container, "act_fn"):
                experts.setdefault((rename(m.group(1)), m.group(1)), set()).add(m.group(3))
                continue
            skipped.append(ck)
        for (name, ck), found in experts.items():
            names = next((n for n in EXPERT_PROJ_NAMES if set(n) <= found), None)
            if names is None:
                raise ValueError(f"{ck}: expert projections {sorted(found)} are not a (gate, up, down) set")
            modules[name] = Exl3HfExperts(all_modules[name], ck, stc.list_tensors(ck), names)
        stc.close()
        if skipped:
            raise ValueError(
                f"{len(skipped)} EXL3 tensors have no nn.Linear in the Transformers model (first: {skipped[0]}). "
                f"Projections the converter splits or re-arranges relative to the HF implementation can't be loaded this way")
        return modules


    def _process_model_before_weight_loading(self, model, checkpoint_files = None, **kwargs):
        modules = self.get_modules_to_replace(model, self._checkpoint_dir(model, checkpoint_files))
        for name, new in modules.items():
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, new)
            self.postprocess_modules.append(new)

        # A quantized output layer can't share the embedding's weight: drop the tie and the tied-key
        # bookkeeping post_init already derived from the config
        tied = getattr(model, "all_tied_weights_keys", None) or {}
        replaced = set(modules)
        def owner(param_name):
            return param_name.rpartition(".")[0]
        stale = [k for k, v in tied.items() if owner(k) in replaced or owner(v) in replaced]
        if stale:
            for k in stale:
                tied.pop(k, None)
            for cfg in (model.config, model.config.get_text_config()):
                if getattr(cfg, "tie_word_embeddings", False):
                    cfg.tie_word_embeddings = False


    def _process_model_after_weight_loading(self, model, **kwargs):
        for module in self.postprocess_modules:
            for lin in (module.linears if isinstance(module, Exl3HfExperts) else [module]):
                lin.finalize()
        self.postprocess_modules = []
        return model


    @property
    def is_trainable(self):
        # Gradients flow through the layers (see Exl3LinearFunction); the quantized weights are frozen
        return True

    def is_serializable(self, safe_serialization = None):
        return False


def register():
    """
    Register the EXL3 quantizer with Transformers. Idempotent
    """
    from transformers.quantizers import register_quantization_config, register_quantizer
    if QUANT_METHOD not in AUTO_QUANTIZATION_CONFIG_MAPPING:
        register_quantization_config(QUANT_METHOD)(Exl3Config)
    if QUANT_METHOD not in AUTO_QUANTIZER_MAPPING:
        register_quantizer(QUANT_METHOD)(Exl3HfQuantizer)


# (earlier name)
patch_transformers = register
