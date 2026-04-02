from __future__ import annotations

from typing_extensions import override
import torch

from ..model.config import Config
from ..model.model_tp_alloc import TPAllocation
from ..util.tensor import to2
from . import Linear, MLP, Module


class NemotronHMoE(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        routed_scaling_factor: float,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        key_gate: str = "gate",
        key_shared_experts: str = "shared_experts",
        key_experts: str = "experts.{expert_idx}",
        key_e_score_bias: str = "gate.e_score_correction_bias",
        qmap: str | None = None,
        out_dtype: torch.dtype | None = torch.float,
    ):
        super().__init__(config, key, None)
        self.module_name = "NemotronHMoE"

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.out_dtype = out_dtype

        self.e_score_correction_bias_key = f"{key}.{key_e_score_bias}"
        self.e_score_correction_bias = None

        self.routing_gate = Linear(
            config = config,
            key = f"{key}.{key_gate}",
            in_features = hidden_size,
            out_features = num_experts,
            qmap = None,
            out_dtype = torch.float,
            pad_to = 1,
        )
        self.register_submodule(self.routing_gate)

        if shared_intermediate_size > 0:
            self.shared_experts = MLP(
                config = config,
                key = f"{key}.{key_shared_experts}",
                hidden_size = hidden_size,
                intermediate_size = shared_intermediate_size,
                key_up = "up_proj",
                key_down = "down_proj",
                activation_fn = "relu2",
                qmap = qmap + ".shared" if qmap else None,
                interm_dtype = torch.half,
                out_dtype = torch.float,
            )
            self.register_submodule(self.shared_experts)
        else:
            self.shared_experts = None

        self.experts = []
        for idx in range(num_experts):
            expert = MLP(
                config = config,
                key = f"{key}.{key_experts}".replace("{expert_idx}", str(idx)),
                hidden_size = hidden_size,
                intermediate_size = intermediate_size,
                key_up = "up_proj",
                key_down = "down_proj",
                activation_fn = "relu2",
                qmap = qmap + f".expert.{idx}" if qmap else None,
                interm_dtype = torch.half,
                out_dtype = torch.float,
            )
            self.experts.append(expert)
            self.register_submodule(expert)

    @override
    def optimizer_targets(self):
        targets = [self.routing_gate.optimizer_targets()]
        if self.shared_experts is not None:
            targets.append(self.shared_experts.optimizer_targets())
        targets += [expert.optimizer_targets() for expert in self.experts]
        return targets

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.e_score_correction_bias = self.config.stc.get_tensor(
            self.e_score_correction_bias_key,
            self.device,
            optional = True,
            allow_bf16 = True,
        )

    @override
    def unload(self):
        super().unload()
        self.e_score_correction_bias = None

    @override
    def weights_numel(self):
        total = super().weights_numel()
        if self.e_score_correction_bias is not None:
            total += self.e_score_correction_bias.numel()
        return total

    def _select_experts(self, scores: torch.Tensor, params: dict):
        activate_all_experts = params.get("activate_all_experts")
        if activate_all_experts:
            topk_indices = (
                torch.arange(self.num_experts, dtype = torch.long, device = scores.device)
                .repeat(scores.shape[0], 1)
            )
            topk_weights = scores
        else:
            scores_for_choice = scores
            if self.e_score_correction_bias is not None:
                scores_for_choice = scores_for_choice + self.e_score_correction_bias.float().unsqueeze(0)

            if self.n_group > 1:
                group_scores = (
                    scores_for_choice.view(-1, self.n_group, self.num_experts // self.n_group)
                    .topk(2, dim = -1)[0]
                    .sum(dim = -1)
                )
                group_idx = torch.topk(group_scores, k = self.topk_group, dim = -1, sorted = False)[1]
                group_mask = torch.zeros_like(group_scores)
                group_mask.scatter_(1, group_idx, 1)
                score_mask = (
                    group_mask.unsqueeze(-1)
                    .expand(-1, self.n_group, self.num_experts // self.n_group)
                    .reshape(-1, self.num_experts)
                )
                scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

            topk_indices = torch.topk(
                scores_for_choice,
                k = self.num_experts_per_tok,
                dim = -1,
                sorted = False,
            )[1]
            topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim = -1, keepdim = True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        orig_shape = x.shape
        hidden_states = x.view(-1, self.hidden_size)

        router_logits = self.routing_gate.forward(hidden_states, params)
        scores = router_logits.sigmoid()
        topk_indices, topk_weights = self._select_experts(scores, params)

        final_hidden_states = torch.zeros(
            hidden_states.shape,
            dtype = torch.float,
            device = hidden_states.device,
        )

        for expert_idx in topk_indices.unique(sorted = False).tolist():
            token_indices, weight_indices = torch.where(topk_indices == expert_idx)
            if token_indices.numel() == 0:
                continue
            expert_input = hidden_states[token_indices]
            expert_output = self.experts[expert_idx].forward(expert_input, params)
            expert_weights = topk_weights[token_indices, weight_indices].to(expert_output.dtype).unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, expert_output * expert_weights)

        if self.shared_experts is not None:
            final_hidden_states += self.shared_experts.forward(hidden_states, params)
        final_hidden_states = final_hidden_states.view(orig_shape)
        return to2(final_hidden_states, out_dtype, self.out_dtype)

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.e_score_correction_bias is not None:
            t[self.e_score_correction_bias_key] = self.e_score_correction_bias
        return t

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError()

    def tp_export(self, plan, producer):
        raise NotImplementedError()

    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        raise NotImplementedError()
