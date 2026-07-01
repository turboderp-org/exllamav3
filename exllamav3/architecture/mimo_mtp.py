from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..model.config import Config
from ..model.model import Model
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.arch_specific.mimo_mtp import MiMoMTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mimo import MiMoConfig


class MiMoMTPModel(Model):

    def __init__(
        self,
        config: MiMoConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        num_layers = config.num_nextn_predict_layers

        # Module list: token/hidden norms + input_proj, then MTP TransformerBlock(s), then final norm
        # module key must differ from the block's: the convert stage saves
        # per-module tensor files named by key, and identical keys clobber
        self.input_layer = MiMoMTPInputLayer(
            config = config,
            key = "model.mtp_layers.0.mtp_in",
            key_norm_hidden = "model.mtp_layers.0.hidden_layernorm",
            key_norm_embedding = "model.mtp_layers.0.token_layernorm",
            key_proj = "model.mtp_layers.0.input_proj",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        )

        self.modules = [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(num_layers):
            attn = Attention(
                config = config,
                key = f"model.mtp_layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_kv_heads,
                rope_settings = config.rope_settings,
                sm_scale = None,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                qmap = "block.attn",
                out_dtype = torch.float,
                qbits_key = "mtp_bits",
            )
            self.attn_modules.append(attn)

            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = f"model.mtp_layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"model.mtp_layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"model.mtp_layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"model.mtp_layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.float,
                        out_dtype = torch.float,
                        qbits_key = "mtp_bits",
                    ),
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # Final norm. Kept out of the forward chain: the drafting loop feeds this
        # model's output back as the next step's target_hidden, and MiMo chains
        # the pre-norm block output (the norm only feeds the head). Applied in
        # sample_from_state instead.
        self.final_norm = RMSNorm(
            config = config,
            key = f"model.mtp_layers.{num_layers - 1}.final_layernorm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
        )
        self.norm_module_idx = len(self.modules)
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 1,  # single depth-1 head; chained drafts decay hard
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
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):
        # Chain state stays pre-norm, see above
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        for module in self.modules[: self.norm_module_idx]:
            params["layer_instance"] = 0
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and tell target to export hidden state.

        MiMo's MTP layer consumes the trunk's pre-norm residual stream (DeepSeek-style), so the
        export hooks the last decoder block rather than the final norm.
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

        self.draft_verifier_params.update({
            "export_state_layers": {self.config.num_hidden_layers - 1},
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
        state = self.final_norm.forward(state.to(self.final_norm.device), params, out_dtype = torch.half)
        if not self.attached_model().loaded_tp:
            ll = self.attached_model().logit_layer_idx
            lm = self.attached_model().modules[ll]
            logits = lm.prepare_for_device(state, params)
            logits = lm.forward(logits, params)
            return torch.argmax(logits, dim = -1)
        else:
            state = self.attached_model().tp_producer.send(state)
            argmax = self.attached_model().tp_dispatch_lm_head_argmax((state, {}))
            return argmax
