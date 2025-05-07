
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
