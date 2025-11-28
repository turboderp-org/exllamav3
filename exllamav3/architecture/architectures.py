from .arcee import ArceeModel
from .apertus import ApertusModel
from .cohere import CohereModel
from .cohere2 import Cohere2Model
from .decilm import DeciLMModel
from .dots1 import Dots1Model
from .ernie4_5 import Ernie4_5Model
from .ernie4_5_moe import Ernie4_5MoEModel
from .exaone4 import Exaone4Model
from .gemma2 import Gemma2Model
from .gemma3 import Gemma3Model, Gemma3TextModel
from .glm4 import Glm4Model
from .glm4_moe import Glm4MoeModel
from .glm4v import Glm4VModel
from .glm4v_moe import Glm4VMoeModel
from .llama import LlamaModel
from .mimo import MiMoModel
from .minimax_m2 import MiniMaxM2Model
from .mistral import MistralModel
from .mistral3 import Mistral3Model
from .mixtral import MixtralModel
from .phi3 import Phi3Model
from .qwen2 import Qwen2Model
from .qwen3 import Qwen3Model
from .qwen3_moe import Qwen3MoeModel
from .qwen3_next import Qwen3NextModel
from .qwen3_vl import Qwen3VLModel
from .qwen3_vl_moe import Qwen3VLMoeModel
from .seedoss import SeedOssModel
from .smollm3 import SmolLM3Model

ARCHITECTURES = {
    m.config_class.arch_string: {
        "architecture": m.config_class.arch_string,
        "config_class": m.config_class,
        "model_class": m,
    } for m in [
        ArceeModel,
        ApertusModel,
        CohereModel,
        Cohere2Model,
        DeciLMModel,
        Dots1Model,
        Ernie4_5Model,
        Ernie4_5MoEModel,
        Exaone4Model,
        Gemma2Model,
        Gemma3Model,
        Gemma3TextModel,
        Glm4Model,
        Glm4MoeModel,
        Glm4VModel,
        Glm4VMoeModel,
        LlamaModel,
        MiMoModel,
        MiniMaxM2Model,
        MistralModel,
        Mistral3Model,
        MixtralModel,
        Phi3Model,
        Qwen2Model,
        Qwen3Model,
        Qwen3MoeModel,
        Qwen3NextModel,
        Qwen3VLModel,
        Qwen3VLMoeModel,
        SeedOssModel,
        SmolLM3Model,
    ]
}

def get_architectures():
    return ARCHITECTURES
