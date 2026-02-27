from .arcee import ArceeModel
from .apertus import ApertusModel
from .cohere import CohereModel
from .cohere2 import Cohere2Model
from .decilm import DeciLMModel
from .deepseek_v2 import DeepseekV2Model, DeepseekModel, DeepseekV3Model, GlmMoeDsaModel
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
from .hcxvisionv2 import HCXVisionV2Model
from .hyperclovax import HyperClovaxModel
from .iquestcoder import IQuestCoderModel
from .llama import LlamaModel
from .mimo import MiMoModel
from .minimax_m2 import MiniMaxM2Model
from .ministral3 import Ministral3Model
from .mistral import MistralModel
from .mistral3 import Mistral3Model
from .mixtral import MixtralModel
from .nanochat import NanoChatModel
from .olmo3 import Olmo3Model
from .phi3 import Phi3Model
from .qwen2 import Qwen2Model
from .qwen2_5_vl import Qwen2_5VLModel
from .qwen3 import Qwen3Model
from .qwen3_5 import Qwen3_5Model, Qwen3_5MoeModel
from .qwen3_moe import Qwen3MoeModel
from .qwen3_next import Qwen3NextModel
from .qwen3_vl import Qwen3VLModel
from .qwen3_vl_moe import Qwen3VLMoeModel
from .seedoss import SeedOssModel
from .smollm3 import SmolLM3Model
from .solar_open_moe import SolarOpenMoeModel

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
        DeepseekV2Model,
        DeepseekModel,
        DeepseekV3Model,
        GlmMoeDsaModel,
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
        HCXVisionV2Model,
        HyperClovaxModel,
        IQuestCoderModel,
        LlamaModel,
        MiMoModel,
        MiniMaxM2Model,
        Ministral3Model,
        MistralModel,
        Mistral3Model,
        MixtralModel,
        NanoChatModel,
        Olmo3Model,
        Phi3Model,
        Qwen2Model,
        Qwen2_5VLModel,
        Qwen3Model,
        Qwen3_5Model,
        Qwen3_5MoeModel,
        Qwen3MoeModel,
        Qwen3NextModel,
        Qwen3VLModel,
        Qwen3VLMoeModel,
        SeedOssModel,
        SmolLM3Model,
        SolarOpenMoeModel,
    ]
}

def get_architectures():
    return ARCHITECTURES
