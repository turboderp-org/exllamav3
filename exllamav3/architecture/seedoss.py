from typing_extensions import override
from .llama import LlamaConfig, LlamaModel

# Qwen2 is identical to Llama except for bias on Q, K and V projections, but Linear module automatically
# detects *.bias tensor

class SeedOssConfig(LlamaConfig):
    arch_string = "SeedOssForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            derived_model = {"text": SeedOssModel},
            **kwargs
        )


class SeedOssModel(LlamaModel):
    config_class = SeedOssConfig

    def __init__(
        self,
        config: SeedOssConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<seed:bos>system\n"
            p += f"{system_prompt}<seed:eos>\n"
        p += f"<seed:bos>user\n"
        p += f"{prompt}<seed:eos>\n"
        p += f"<seed:bos>assistant\n"
        return p