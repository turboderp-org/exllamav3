
from .llama import LlamaConfig, LlamaModel

# Mistral is identical to Llama

class MistralConfig(LlamaConfig):

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            arch_string = "MistralForCausalLM",
            **kwargs
        )


class MistralModel(LlamaModel):

    def __init__(
        self,
        config: MistralConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
