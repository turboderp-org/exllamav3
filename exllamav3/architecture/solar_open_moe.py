from typing_extensions import override
from .glm4_moe import Glm4MoeConfig, Glm4MoeModel

class SolarOpenMoeConfig(Glm4MoeConfig):
    arch_string = "SolarOpenForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            derived_model={"text": SolarOpenMoeModel},
            **kwargs
        )

        # Override Glm4MoeConfig defaults specific to Solar
        self.first_k_dense_replace = self.read_cfg(int, "first_k_dense_replace", 0)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 2.0)


class SolarOpenMoeModel(Glm4MoeModel):
    config_class = SolarOpenMoeConfig