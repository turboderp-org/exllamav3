from .llama import LlamaModel
from .mistral import MistralModel
from .qwen2 import Qwen2Model
from .qwen3 import Qwen3Model
from .qwen3_moe import Qwen3MoeModel
from .phi3 import Phi3Model
from .gemma2 import Gemma2Model
from .decilm import DeciLMModel
from .glm4 import Glm4Model
from .cohere import CohereModel
from .cohere2 import Cohere2Model

ARCHITECTURES = {
    m.config_class.arch_string: {
        "architecture": m.config_class.arch_string,
        "config_class": m.config_class,
        "model_class": m,
    } for m in [
        LlamaModel,
        MistralModel,
        Qwen2Model,
        Qwen3Model,
        Qwen3MoeModel,
        Phi3Model,
        Gemma2Model,
        DeciLMModel,
        Glm4Model,
        CohereModel,
        Cohere2Model,
    ]
}

def get_architectures():
    return ARCHITECTURES