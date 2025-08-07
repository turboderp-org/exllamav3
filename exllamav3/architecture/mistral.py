from typing_extensions import override
from .llama import LlamaConfig, LlamaModel

# Mistral is identical to Llama

class MistralConfig(LlamaConfig):
    arch_string = "MistralForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            derived_model = {"text": MistralModel},
            **kwargs
        )


class MistralModel(LlamaModel):
    config_class = MistralConfig

    def __init__(
        self,
        config: MistralConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<s>[INST]"
        if system_prompt:
            p += f" {system_prompt}\n\n"
        p += f" {prompt} [/INST]"
        return p