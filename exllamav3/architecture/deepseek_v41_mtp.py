from __future__ import annotations
from typing_extensions import override
import torch
from .deepseek_v4_mtp import DeepseekV4MTPModel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .deepseek_v41 import DeepseekV41Config

# DeepSeek-V4.1 MTP component: the DeepSeek-V4 DSpark drafter with its own routed expert count, every
# mHC site and the exit collapsing with the previous site's weights, no per-head q norm, and taps
# taken at the input of dspark_target_layer_ids (the output of the block before each).


class DeepseekV41MTPModel(DeepseekV4MTPModel):

    def __init__(
        self,
        config: DeepseekV41Config,
        **kwargs
    ):
        targets = config.dspark_target_layer_ids
        assert targets and len(set(targets)) == len(targets) and all(
            0 < t < config.num_hidden_layers and t not in config.engram_layer_ids for t in targets), \
            f"DeepseekV41 MTP: unsupported dspark_target_layer_ids {targets}"
        super().__init__(
            config,
            num_experts = config.dspark_num_experts,
            num_experts_per_tok = config.dspark_num_experts_per_tok,
            carry_pre = True,
            q_head_norm = False,
            **kwargs
        )
        self.draft_verifier_params["export_state_layers"] = {t - 1 for t in targets}

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["hc_pre"] = None
        return super().prepare_inputs(input_ids, params)
