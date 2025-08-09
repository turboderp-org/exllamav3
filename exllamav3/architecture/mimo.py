from typing_extensions import override
from .llama import LlamaConfig, LlamaModel

# Identical to Llama except for MTP layers, ignored for now (TODO:)

class MiMoConfig(LlamaConfig):
    arch_string = "MiMoForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            derived_model = {"text": MiMoModel},
            **kwargs
        )


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