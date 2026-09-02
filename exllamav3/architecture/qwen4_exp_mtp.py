from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..model.config import Config
from ..model.model import Model
from ..modules import Embedding, Linear, GatedResidual
from ..modules.module import Module
from ..modules.arch_specific.qwen4_exp_mtp import Qwen4ExpMTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qwen4_exp import Qwen4ExpConfig

"""
MTP (multi-token prediction) draft head for Qwen3.8-Flash-Next: input combine over the trunk's
PRE-collapse hyper-connection stream stack (exported by the trunk's final mixer) plus the next
token's embedding, one full qwen4_exp decoder block (QSA attention, MoE, gated-residual sites),
and its own combine-less mixer. Shares the trunk's embedding and lm_head.

No reference implementation exists for this head; the input-combine stream handling
(Qwen4ExpMTPInputLayer.stream_tap) is a semantic guess that must be confirmed by acceptance
rate on the full model.
"""


class Qwen4ExpMTPStackOut(Module):
    """
    Terminal module of the MTP draft chain: passes the decoder block's stream stack through
    FLATTENED (bsz, seq, hc_mult * hidden) instead of collapsing it, so the model's forward
    output can feed the next drafting step's target_hidden (symmetric with the trunk's pre-mixer
    stack export). The mixer is owned here as a submodule (so it loads with the model) and is
    applied by sample_from_state() before the shared lm_head.
    """

    def __init__(self, config, key: str, mixer: GatedResidual):
        super().__init__(config, key, None)
        self.mixer = mixer
        self.register_submodule(mixer)

    def optimizer_targets(self):
        return []

    # The compile step collects a top-level module's output tensors by ITS key prefix; this
    # module's own key ("mtp_stack_out") names no tensors, so it must hand the collection to
    # the owned mixer (prefix "mtp.hyper_connection_mixer."), or the mixer's tensors stay in
    # the qtensors files and never reach the compiled shards
    def get_compile_sizes(self, stc):
        return self.mixer.get_compile_sizes(stc)

    def get_compile_tensors(self, stc):
        return self.mixer.get_compile_tensors(stc)

    def forward(self, x, params, out_dtype = None):
        return x.flatten(-2).half()


class Qwen4ExpMTPModel(Model):

    def __init__(
        self,
        config: Qwen4ExpConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        from .qwen4_exp import build_qwen4_block

        self.input_layer = Qwen4ExpMTPInputLayer(
            config = config,
            key = "mtp",
            hidden_size = config.hidden_size,
            hc_mult = config.hc_mult,
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        )
        self.modules = [self.input_layer]
        self.first_block_idx = len(self.modules)

        for idx in range(config.mtp_num_hidden_layers):
            self.modules.append(
                build_qwen4_block(
                    config,
                    f"mtp.layers.{idx}",
                    idx,
                    "full_attention",
                    qbits_key = "mtp_bits",
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # The draft chain's output is the flattened PRE-mixer stream stack (it feeds the next
        # drafting step's target_hidden); sample_from_state applies the mixer + shared lm_head
        self.stack_out = Qwen4ExpMTPStackOut(
            config,
            "mtp_stack_out",
            GatedResidual(
                config = config,
                key = "mtp.hyper_connection_mixer",
                hc_mult = config.hc_mult,
                hidden_size = config.hidden_size,
                rms_norm_eps = config.rms_norm_eps,
                use_combine = False,
                out_dtype = torch.half,
            ),
        )
        self.modules.append(self.stack_out)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 4,
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")

    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and have the trunk's final mixer
        export the pre-collapse stream stack as the draft input state.
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

        target_mixer = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_mixer, GatedResidual) and not target_mixer.use_combine, \
            "Expected the trunk's combine-less mixer immediately before lm_head"
        self.draft_verifier_params.update({
            "export_state_norm_keys": {target_mixer.key},
        })

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, max_chunk_size):
        return {}

    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        # state is the flattened pre-mixer stream stack; collapse it before the shared head
        mixer = self.stack_out.mixer
        bsz, seq, _ = state.shape
        stack = state.to(mixer.device).view(bsz, seq, mixer.hc_mult, mixer.hidden_size)
        state = mixer.forward(stack, params)
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
