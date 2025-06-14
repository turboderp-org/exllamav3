from .cohere import CohereModel
from .cohere2 import Cohere2Model
from .decilm import DeciLMModel
from .dots1 import Dots1Model
from .gemma2 import Gemma2Model
from .gemma3 import Gemma3Model, Gemma3TextModel
from .glm4 import Glm4Model
from .llama import LlamaModel
from .mimo import MiMoModel
from .mistral import MistralModel
from .mistral3 import Mistral3Model
from .mixtral import MixtralModel
from .phi3 import Phi3Model
from .qwen2 import Qwen2Model
from .qwen3 import Qwen3Model
from .qwen3_moe import Qwen3MoeModel

ARCHITECTURES = {
    m.config_class.arch_string: {
        "architecture": m.config_class.arch_string,
        "config_class": m.config_class,
        "model_class": m,
    } for m in [
        CohereModel,
        Cohere2Model,
        DeciLMModel,
        Dots1Model,
        Gemma2Model,
        Gemma3Model,
        Gemma3TextModel,
        Glm4Model,
        LlamaModel,
        MiMoModel,
        MistralModel,
        Mistral3Model,
        MixtralModel,
        Phi3Model,
        Qwen2Model,
        Qwen3Model,
        Qwen3MoeModel,
    ]
}

def get_architectures():
    return ARCHITECTURES