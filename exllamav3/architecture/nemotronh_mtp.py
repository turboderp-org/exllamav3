from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..model.config import Config
from ..model.model import Model
from ..modules import RMSNorm, Embedding, Linear
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .nemotronh import NemotronHConfig


class NemotronHMTPModel(Model):
    """
    Nemotron-3 Super MTP head (Megatron multi-token prediction): normed token embedding and normed
    trunk state concatenated (embedding first) and projected 2H -> H by eh_proj, then the blocks of
    mtp_hybrid_override_pattern ("*E": NoPE attention + latent MoE, the same block types as the
    trunk) and a final norm before the shared lm_head. Everything lives under mtp.layers.N; the
    input norms/projection sit on the first block, final_layernorm on the last. One native draft
    depth; deeper drafts iterate the head on its own output. Borrows the trunk's embeddings and
    lm_head. Mamba blocks in the MTP pattern would need recurrent draft states and are not
    supported (Super's pattern has none).
    """

    def __init__(
        self,
        config: NemotronHConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        from .nemotronh import nemotronh_block

        assert config.num_mtp_layers == 1, "NemotronH MTP: only one native draft step is supported"
        pattern = config.mtp_hybrid_override_pattern
        assert pattern and "M" not in pattern, \
            f"NemotronH MTP: unsupported mtp_hybrid_override_pattern {pattern!r}"

        self.input_layer = Qwen3_5MTPInputLayer(
            config = config,
            key = "mtp.layers.0.input",
            key_pre_fc_norm_hidden = "mtp.layers.0.hnorm",
            key_pre_fc_norm_embedding = "mtp.layers.0.enorm",
            key_fc = "mtp.layers.0.eh_proj",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.layer_norm_epsilon,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
            constant_bias = 0.0,
        )
        self.modules = [self.input_layer]
        self.first_block_idx = len(self.modules)

        for idx, block_type in enumerate(pattern):
            self.modules.append(nemotronh_block(
                config, f"mtp.layers.{idx}", idx, block_type, qbits_key = "mtp_bits"
            ))

        self.last_kv_module_idx = len(self.modules) - 1

        self.final_norm = RMSNorm(
            config = config,
            key = f"mtp.layers.{len(pattern) - 1}.final_layernorm",
            rms_norm_eps = config.layer_norm_epsilon,
            out_dtype = torch.half,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 2,
            "autosplit_load_fwd": False,
        })

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        # Which trunk state hnorm consumes: the post-final-norm state or the pre-norm residual.
        # Settled empirically (see attach_to)
        self.pre_norm_tap = False

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Embedding is handled by the input layer; no recurrent blocks in the head
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """
        Bind to the target model: borrow embeddings / lm_head and tell the target to export the
        state hnorm consumes (post-norm_f by default, the pre-norm residual with pre_norm_tap).
        """
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)

        target_embed = None
        for m in target.modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        if self.pre_norm_tap:
            self.draft_verifier_params = {
                "export_state_layers": {self.config.num_hidden_layers - 1},
            }
        else:
            target_norm = target.modules[target.logit_layer_idx - 1]
            assert isinstance(target_norm, RMSNorm), \
                "Expected target final RMSNorm immediately before lm_head"
            self.draft_verifier_params = {
                "export_state_norm_keys": {target_norm.key},
            }


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        ll = self.attached_model().logit_layer_idx
        lm = self.attached_model().modules[ll]
        logits = lm.prepare_for_device(state, params)
        logits = lm.forward(logits, params)
        if params.get("export_draft_conf"):
            logits = logits[..., :self.attached_model().config.vocab_size]
            conf, ids = torch.max(logits, dim = -1)
            params["draft_conf"] = conf
            return ids
        return torch.argmax(logits, dim = -1)
