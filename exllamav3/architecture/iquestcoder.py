from typing_extensions import override
from .llama import LlamaConfig, LlamaModel

# The non-looping variant of IQuestCoder is identical to Llama

class IQuestCoderConfig(LlamaConfig):
    arch_string = "IQuestCoderForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            derived_model = {"text": IQuestCoderModel},
            **kwargs
        )


class IQuestCoderModel(LlamaModel):
    config_class = IQuestCoderConfig

    def __init__(
        self,
        config: IQuestCoderConfig,
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