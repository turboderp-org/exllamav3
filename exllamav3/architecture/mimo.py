from typing_extensions import override
from .llama import LlamaConfig, LlamaModel
from .mimo_mtp import MiMoMTPModel

# Llama with an MTP layer, loadable as a separate draft model for
# self-speculative decoding (component = "mtp")

class MiMoConfig(LlamaConfig):
    arch_string = "MiMoForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            derived_model = {"text": MiMoModel, "mtp": MiMoMTPModel},
            **kwargs
        )

        # MTP (multi-token prediction) head — informational; head loaded as separate draft model
        self.num_nextn_predict_layers = self.read_cfg(int, "num_nextn_predict_layers", 0)
        if self.num_nextn_predict_layers == 0:
            del self.model_classes["mtp"]


class MiMoModel(LlamaModel):
    config_class = MiMoConfig

    def __init__(
        self,
        config: MiMoConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n"
            p += f"{system_prompt}<|im_end|>\n"
        p += f"<|im_start|>user\n"
        p += f"{prompt}<|im_end|>\n"
        p += f"<|im_start|>assistant\n"
        return p