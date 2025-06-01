
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
