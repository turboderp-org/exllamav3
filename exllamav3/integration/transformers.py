import os
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn
from transformers.quantizers.auto import HfQuantizer
from transformers.quantizers.base import QuantizationConfigMixin
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING, AUTO_QUANTIZATION_CONFIG_MAPPING
from transformers.utils.quantization_config import QuantizationMethod

from exllamav3.loader import SafetensorsCollection
from exllamav3.modules.quant.exl3 import LinearEXL3

class Exl3HfLinear(torch.nn.Module):
    """
    Basic nn.Module wrapping an EXL3 linear layer.
    """

    def __init__(self, in_features: int, out_features: int, exl3_tensors: dict):
        """
        :param in_features:
            Number of input features

        :param out_features:
            Number of output features

        :param exl3_tensors:
            Defines tensors to expect for the layer, as loaded from the model safetensors file. Example set:
            {
                # 1024x4096 layer in 16x16 tiles, 3 bits per weight = 48 uint16s per tile
                "model.layers.0.attn.v_proj.trellis": {
                    "shape": [256, 64, 48],
                    "torch_dtype": torch.int16
                }
                # 4096 input channels
                "model.layers.0.attn.q_proj.suh": {
                    "shape": [4096],
                    "torch_dtype": torch.float16
                }
                # 1024 output channels
                "model.layers.0.attn.q_proj.svh": {
                    "shape": [1024],
                    "torch_dtype": torch.float16
                }
                # 1024 channel bias (optional)
                "model.layers.0.attn.q_proj.bias": {
                    "shape": [1024],
                    "torch_dtype": torch.float16
                }
            }

            "su" and "sv" keys are packed signs bits supported for legacy reasons. They are expanded to "suh" and "svh"
            float16 tensors at load time, and models quantized since ~v0.0.2 will contain only "suh" and "svh" tensors
            with baked-in input/output channel scales.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Find prefix, only needed to extract the right keys from the exl3_tensors dict
        subkeys = list(exl3_tensors.keys())
        l = subkeys[0].rfind(".")
        self.key = subkeys[0][:l]
        assert all(key[:l] == self.key for key in subkeys), "All tensors must belong to the same module"

        # Metadata for each sub-tensor
        m_trellis = exl3_tensors.get(f"{self.key}.trellis")
        m_suh = exl3_tensors.get(f"{self.key}.suh")
        m_svh = exl3_tensors.get(f"{self.key}.svh")
        m_su = exl3_tensors.get(f"{self.key}.su")  # Legacy models
        m_sv = exl3_tensors.get(f"{self.key}.sv")  # Legacy models
        m_bias = exl3_tensors.get(f"{self.key}.bias")

        # Create empty meta tensors accordingly
        t_trellis = torch.empty(m_trellis["shape"], dtype = m_trellis["torch_dtype"], device = "meta")
        t_suh = torch.empty(m_suh["shape"], dtype = m_suh["torch_dtype"], device = "meta") if m_suh else None
        t_svh = torch.empty(m_svh["shape"], dtype = m_svh["torch_dtype"], device = "meta") if m_svh else None
        t_su = torch.empty(m_su["shape"], dtype = m_su["torch_dtype"], device = "meta") if m_su else None
        t_sv = torch.empty(m_sv["shape"], dtype = m_sv["torch_dtype"], device = "meta") if m_sv else None
        t_bias = torch.empty(m_bias["shape"], dtype = m_bias["torch_dtype"], device = "meta") if m_bias else None

        # Create buffers to load into
        self.register_buffer("trellis", t_trellis)
        if m_suh: self.register_buffer("suh", t_suh)
        else: self.suh = None
        if m_svh: self.register_buffer("svh", t_svh)
        else: self.svh = None
        if m_su: self.register_buffer("su", t_su)
        else: self.su = None
        if m_sv: self.register_buffer("sv", t_sv)
        else: self.sv = None
        if m_bias: self.register_buffer("bias", t_bias)
        else: self.bias = None

        # Inner LinearEXL3 module initialized after loading
        self.inner = None

        # Some implementations in transformers (Cohere2 at least) seem to reference .weight.dtype directly, so create
        # a dummy tensor keep them happy
        self.weight = torch.zeros((1,), dtype = torch.float16, device = "meta")


    def finalize(self):
        """
        Call once parameters are loaded.
        """

        # Some models seem to want to cast these to other types. Make sure they're float16 by the time the module
        # is fully loaded
        if self.suh is not None:
            self.suh = self.suh.half()
        if self.svh is not None:
            self.svh = self.svh.half()
        if self.bias is not None:
            self.bias = self.bias.half()

        self.inner = LinearEXL3(
            config = None,
            in_features = self.in_features,
            out_features = self.out_features,
            trellis = self.trellis,
            suh = self.suh,
            svh = self.svh,
            su = self.su,
            sv = self.sv,
            bias = self.bias,
            out_dtype = torch.float16,
            transformers_fix = True
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass works in float16 only, but some models (e.g. Gemma) require BF16 or FP32 precision, so allow
        # for that by temporarily downcasting just for individual linear layers
        dtype = x.dtype
        return self.inner.forward(x.half(), {}).to(dtype)


class Exl3HfQuantizer(HfQuantizer):

    requires_calibration = False
    required_packages = "exllamav3>=0.0.5"
    requires_parameters_quantization = False

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config
        self.postprocess_modules = []


    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run EXL3 model.")


    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        # Change the default datatype only
        if torch_dtype is None:
            torch_dtype = torch.float16
        return torch_dtype


    def get_modules_to_replace(self, model):

        # Ideally we would start from whatever directory is used by `_get_resolved_checkpoint_files`, to allow
        # instantiating model from a name on the Hub. For now, assume `model.name_or_path` is a path.
        # TODO: Figure this out
        path = model.name_or_path
        if not os.path.isdir(path):
            raise ValueError("EXL3 model must be initialized from local directory")

        # Use ExLlamaV3's own utility functions for indexing safetensors files. We only need this to determine which
        # tensor keys to mark as EXL3 tensors using the logic below. Models quantized since v0.0.2 also include a
        # `quantization_config.json` file that could be used instead (essentially just a compilation of the headers
        # from each individual .safetensors file in a mode.)
        stc = SafetensorsCollection(path)

        # Find modules to replace and create placeholders for them
        modules_to_replace = {}
        for name, module in tuple(model.named_modules()):
            if isinstance(module, torch.nn.Linear):

                # Kludge for models like Gemma3 where named_modules() don't match stored tensor keys.
                # TODO: Figure out the right way to handle this
                m_name = name
                if name.startswith("model.language_model."):
                    name = "language_model.model." + name[21:]

                if stc.has_tensor_group(name, [["sv", "svh"], ["su", "suh"], "trellis"]):
                    modules_to_replace[m_name] = Exl3HfLinear(
                        module.in_features,
                        module.out_features,
                        stc.list_tensors(name)
                    )

        # SafetensorsCollection may allocate managed resources. This ensures everything is freed
        stc.close()

        return modules_to_replace


    def replace_modules(self, module, path, modules_to_replace: dict):

        # Identify nodes by path
        children_to_replace = {}
        for name, child in module.named_children():
            key = f"{path}.{name}" if path else name
            if new_module := modules_to_replace.get(key):
                children_to_replace[name] = new_module
            else:
                self.replace_modules(child, key, modules_to_replace)

        # Replace after to avoid modifying any collections while iterating over them
        for name, new in children_to_replace.items():
            setattr(module, name, new)
            self.postprocess_modules.append(new)


    def _process_model_before_weight_loading(
        self,
        model,
        keep_in_fp32_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        # Get list of modules to replace and create their replacements (with meta tensors)
        modules_to_replace = self.get_modules_to_replace(model)

        # Recursively walk through model and insert new modules where needed
        self.replace_modules(model, None, modules_to_replace)

        # For models with tied embeddings, ExLlamaV3 always separates into an unquantized embedding layer (to be kept
        # in system RAM) and a quantized output layer that resides in VRAM. Make sure the HF config reflects this
        # TODO: Figure out if Transformers allows for keeping the embedding layer in system RAM
        config = kwargs.get("config")
        if config:
            config.tie_word_embeddings = False


    def _process_model_after_weight_loading(self, model, **kwargs):

        # Finalize modules after loading
        for module in self.postprocess_modules:
            module.finalize()

        # Don't hold on to any references
        self.postprocess_modules = []

        return model


    @property
    def is_trainable(self, model = None):
        return False

    def is_serializable(self, safe_serialization = None):
        return False


@dataclass
class Exl3Config(QuantizationConfigMixin):
    def __init__(
        self,
        **kwargs,
    ):
        # Unsure if this value is used anywhere?
        self.quant_method = QuantizationMethod.EXL3


def patch_transformers():

    # Inject EXL3 quantizer and config classes
    AUTO_QUANTIZER_MAPPING["exl3"] = Exl3HfQuantizer
    AUTO_QUANTIZATION_CONFIG_MAPPING["exl3"] = Exl3Config

    # Can't actually mutate this Enum, but code seems to work regardless
    # TODO: Something else
    QuantizationMethod.EXL3 = "exl3"