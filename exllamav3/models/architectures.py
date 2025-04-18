
from .llama import LlamaConfig, LlamaModel
from .mistral import MistralConfig, MistralModel
from .qwen2 import Qwen2Config, Qwen2Model
from .phi3 import Phi3Config, Phi3Model
from .gemma import Gemma2Config, Gemma2Model
from .decilm import DeciLMConfig, DeciLMModel
from .glm4 import Glm4Config, Glm4Model

ARCHITECTURES = {
    "LlamaForCausalLM": {
        "architecture": "LlamaForCausalLM",
        "config_class": LlamaConfig,
        "model_class": LlamaModel,
    },
    "MistralForCausalLM": {
        "architecture": "MistralForCausalLM",
        "config_class": MistralConfig,
        "model_class": MistralModel,
    },
    "Qwen2ForCausalLM": {
        "architecture": "Qwen2ForCausalLM",
        "config_class": Qwen2Config,
        "model_class": Qwen2Model,
    },
    "Phi3ForCausalLM": {
        "architecture": "Phi3ForCausalLM",
        "config_class": Phi3Config,
        "model_class": Phi3Model,
    },
    "Gemma2ForCausalLM": {
        "architecture": "Gemma2ForCausalLM",
        "config_class": Gemma2Config,
        "model_class": Gemma2Model,
    },
    "DeciLMForCausalLM": {
        "architecture": "DeciLMForCausalLM",
        "config_class": DeciLMConfig,
        "model_class": DeciLMModel,
    },
    "Glm4ForCausalLM": {
        "architecture": "Glm4ForCausalLM",
        "config_class": Glm4Config,
        "model_class": Glm4Model,
    },
}

def get_architectures():
    return ARCHITECTURES