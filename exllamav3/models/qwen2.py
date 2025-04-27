
from .llama import LlamaConfig, LlamaModel

# Qwen2 is identical to Llama except for bias on Q, K and V projections, but Linear module automatically
# detects *.bias tensor

class Qwen2Config(LlamaConfig):
    arch_string = "Qwen2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            **kwargs
        )


class Qwen2Model(LlamaModel):
    config_class = Qwen2Config

    def __init__(
        self,
        config: Qwen2Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)
