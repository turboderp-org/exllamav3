from __future__ import annotations
from typing_extensions import override
import torch
import weakref
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Module, Linear, RMSNorm, GatedMLP, TransformerBlock, Attention, SlidingAttention
from ..modules.shared_kv_attn import SharedKVAttention
from ..modules.attn import prepare_for_attn
from ..util.device_copy import to_device
from ..util.tensor import get_for_device, to2
from ..util.rope import RopeStyle

"""
Gemma4 assistant drafters (google/gemma-4-*-it-assistant): a small Gemma4 text stack driven autoregressively
like an MTP head. Each step consumes the target's embedding of the last token and a backbone-width state
(the target's last decoder-layer output on the first step, the drafter's post-projected output after that),
and its layers attend with their own queries over the TARGET's K/V cache: the last full-attention layer's
keys for its full layer, the last sliding layer's window for its sliding layers. It writes no K/V, so it has
no cache, no prefill and nothing to rewind. Logits come from its own tied head.
"""


class Gemma4AssistantConfig(Config):
    arch_string = "Gemma4AssistantForCausalLM"

    def get_model_classes(self):
        return {"text": Gemma4AssistantModel}

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": Gemma4AssistantModel},
            **kwargs
        )
        pfx = "text_config->"
        self.backbone_hidden_size = self.read_cfg(int, "backbone_hidden_size", no_default)
        self.hidden_size = self.read_cfg(int, pfx + "hidden_size", no_default)
        self.intermediate_size = self.read_cfg(int, pfx + "intermediate_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, pfx + "num_hidden_layers", no_default)
        self.layer_types = self.read_cfg(list, pfx + "layer_types", no_default)
        self.head_dim = self.read_cfg(int, pfx + "head_dim", no_default)
        self.global_head_dim = self.read_cfg(int, pfx + "global_head_dim", self.head_dim)
        self.num_q_heads = self.read_cfg(int, pfx + "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, pfx + "num_key_value_heads", no_default)
        self.num_global_kv_heads = self.read_cfg(int, pfx + "num_global_key_value_heads", self.num_kv_heads)
        self.sliding_window = self.read_cfg(int, pfx + "sliding_window", 1024)
        self.rms_norm_eps = self.read_cfg(float, pfx + "rms_norm_eps", 1e-6)
        self.vocab_size = self.read_cfg(int, pfx + "vocab_size", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, pfx + "tie_word_embeddings", True)
        self.final_logit_softcapping = self.read_cfg(float, pfx + "final_logit_softcapping", None)
        self.attn_logit_softcapping = self.read_cfg(float, pfx + "attn_logit_softcapping", None) or 0.0
        self.use_ordered_embeddings = self.read_cfg(bool, "use_ordered_embeddings", False)
        assert not self.use_ordered_embeddings, "Gemma4 assistant: ordered (centroid) embeddings are not supported"
        # Default draft length: the checkpoint's own assisted-generation setting
        self.num_assistant_tokens = 6
        try:
            import json, os
            with open(os.path.join(directory, "generation_config.json"), encoding = "utf8") as f:
                self.num_assistant_tokens = int(json.load(f).get("num_assistant_tokens", 6) or 6)
        except (OSError, ValueError):
            pass
        self.rope_settings_local = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000.0,
            config_dict = self.read_cfg(dict, pfx + "rope_parameters->sliding_attention", {}),
            override_type = self.read_cfg(str, pfx + "rope_parameters->sliding_attention->rope_type", None),
        )
        self.rope_settings_global = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1000000.0,
            config_dict = self.read_cfg(dict, pfx + "rope_parameters->full_attention", {}),
            override_type = self.read_cfg(str, pfx + "rope_parameters->full_attention->rope_type", None),
            override_head_dim = self.global_head_dim,
        )


class Gemma4AssistantInputLayer(Module):
    """
    [target embedding of the token | backbone-width state] -> pre_projection -> drafter width. The embedding
    comes from the attached target's table (so an embedding row handed over by the sparse head is honoured),
    the state from params["target_hidden"]
    """

    def __init__(self, config: Gemma4AssistantConfig, key: str, qmap: str | None = None):
        super().__init__(config, key, None)
        self.backbone_hidden_size = config.backbone_hidden_size
        self.pre_projection = Linear(
            config = config,
            key = key,
            in_features = 2 * config.backbone_hidden_size,
            out_features = config.hidden_size,
            qmap = qmap,
            out_dtype = torch.float,
        )
        self.register_submodule(self.pre_projection)
        self.attached_model = None

    @override
    def optimizer_targets(self):
        return self.pre_projection.optimizer_targets()

    @override
    def weights_numel(self):
        return self.pre_projection.weights_numel()

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        target_hidden = params.get("target_hidden")
        assert target_hidden is not None, "Gemma4 assistant requires target_hidden"
        assert target_hidden.shape[:-1] == x.shape, \
            f"Gemma4 assistant token/state shape mismatch: {tuple(x.shape)} vs {tuple(target_hidden.shape)}"
        state = get_for_device(params, "target_hidden", self.device).half()
        emb = self.attached_model().modules[0].forward(x, params, out_dtype = torch.half)
        emb = to_device(emb, self.device)
        y = torch.cat((emb, state), dim = -1)
        return self.pre_projection.forward(y, params, out_dtype = torch.float)


class Gemma4AssistantHead(Module):
    """
    Terminal module owning the post-projection (next step's backbone-width state) and the drafter's own
    tied head, applied by sample_from_state(). Passes the normed hidden state through unchanged
    """

    def __init__(self, config: Gemma4AssistantConfig, key: str):
        super().__init__(config, key, None)
        self.post_projection = Linear(
            config = config,
            key = "post_projection",
            in_features = config.hidden_size,
            out_features = config.backbone_hidden_size,
            qmap = "post",
            out_dtype = torch.half,
        )
        self.lm_head = Linear(
            config = config,
            key = "lm_head",
            qbits_key = "head_bits",
            alt_key = "model.embed_tokens" if config.tie_word_embeddings else None,
            in_features = config.hidden_size,
            out_features = config.vocab_size,
            qmap = "head",
            softcap = config.final_logit_softcapping or 0.0,
            caps = {"logits_output": True},
        )
        self.register_submodule(self.post_projection)
        self.register_submodule(self.lm_head)

    @override
    def optimizer_targets(self):
        return []

    @override
    def weights_numel(self):
        return self.post_projection.weights_numel() + self.lm_head.weights_numel()

    # The compile step collects a top-level module's tensors by ITS key prefix; this module's own key names
    # nothing, so hand the collection to the owned linears
    def get_compile_sizes(self, stc):
        return self.post_projection.get_compile_sizes(stc) + self.lm_head.get_compile_sizes(stc)

    def get_compile_tensors(self, stc):
        a, b = self.post_projection.get_compile_tensors(stc), self.lm_head.get_compile_tensors(stc)
        return {**a, **b} if isinstance(a, dict) else a + b

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        return x


class Gemma4AssistantModel(Model):
    config_class = Gemma4AssistantConfig

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.input_layer = Gemma4AssistantInputLayer(config, "pre_projection", qmap = "target_hidden")
        self.modules += [self.input_layer]
        self.first_block_idx = len(self.modules)
        self.attn_modules = []
        for idx in range(config.num_hidden_layers):
            full = config.layer_types[idx] == "full_attention"
            attn = SharedKVAttention(
                config = config,
                key = f"model.layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.global_head_dim if full else config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_global_kv_heads if full else config.num_kv_heads,
                rope_settings = config.rope_settings_global if full else config.rope_settings_local,
                sliding_window = -1 if full else config.sliding_window,
                qmap = "block.attn",
                sm_scale = 1.0,
                logit_softcapping = config.attn_logit_softcapping,
                q_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
            )
            attn.is_full = full
            self.attn_modules.append(attn)
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"model.layers.{idx}",
                    layer_idx = idx,
                    key_layer_scalar = "layer_scalar",
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"model.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    attn_post_norm = RMSNorm(
                        config = config,
                        key = f"model.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"model.layers.{idx}.pre_feedforward_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"model.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        activation_fn = "gelu",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                    mlp_post_norm = RMSNorm(
                        config = config,
                        key = f"model.layers.{idx}.post_feedforward_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                )
            ]
        self.last_kv_module_idx = len(self.modules) - 1
        self.final_norm = RMSNorm(
            config = config,
            key = "model.norm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
        )
        self.head = Gemma4AssistantHead(config, "assistant_head")
        self.modules += [self.final_norm, self.head]
        self.logit_layer_idx = None
        self.caps.update({
            "uncalibrated_quantize": True,
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "draft_uses_target_cache": True,
            "default_draft_size": config.num_assistant_tokens,
            "autosplit_load_fwd": False,
        })
        self.attached_model = None


    def attach_to(self, target):
        """
        Bind to the target: embedding for the input layer, the K/V sources of its last full and last sliding
        layer (before any K/V sharing of its own), and the export of its last decoder layer's output, which is
        the state the first drafting step of a round consumes
        """
        # The drafter reads two of the target's cache layers directly; a tensor-parallel target shards those
        # across ranks (and keeps no embedding in this process), so it would need a TP split of its own
        assert not target.loaded_tp, \
            "Gemma4 assistant drafting requires a layer-split target: tensor-parallel targets shard the K/V cache the drafter reads"
        self.attached_model = weakref.ref(target)
        self.input_layer.attached_model = weakref.ref(target)
        tcfg = target.config
        n = tcfg.num_hidden_layers - getattr(tcfg, "num_kv_shared_layers", 0)
        types = list(tcfg.layer_types[:n])
        last_full = max(i for i, t in enumerate(types) if t == "full_attention")
        last_swa = max(i for i, t in enumerate(types) if t == "sliding_attention")
        target_attn = {}
        for m in target.modules:
            attn = getattr(m, "attn", None)
            if isinstance(attn, (Attention, SlidingAttention)):
                target_attn[attn.layer_idx] = attn
        for attn in self.attn_modules:
            src_idx = last_full if attn.is_full else last_swa
            tmod = target_attn[src_idx]
            kind = "ring" if isinstance(tmod, SlidingAttention) else "paged"
            attn.attach_source(kind, src_idx, tmod)
        # The exported block output is taken before the block's layer_scalar multiply, which is also where the
        # reference records hidden_states[-1] (measured: same scale), so it feeds the drafter as is
        self.draft_verifier_params.update({
            "export_state_layers": {tcfg.num_hidden_layers - 1},
        })


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)


    @override
    def prefill(self, input_ids: torch.Tensor, params: dict | None = None):
        # No K/V of its own: the drafter reads the target's cache, so there is nothing to prefill
        return


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("Draft model does not have its own chat template")


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def sparse_head_linear(self):
        """The drafter samples from its own head, so a sparse draft head attaches here, not to the target's"""
        return self.head.lm_head


    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        """
        Greedy token from the drafter's own head (or its sparse subset when the generator asks for it); the
        post-projected state for the next step goes back through params["next_target_hidden"], and the
        winning logit as draft confidence when asked
        """
        head = self.head
        state = to_device(state, head.device)
        params["next_target_hidden"] = head.post_projection.forward(state, params, out_dtype = torch.half)
        export = bool(params.get("export_draft_conf"))
        sh = head.lm_head.sparse_head
        flag = params.get("sparse_head") if sh is not None else None
        if flag is True and sh.ready:
            ids, conf, emb = sh.argmax(state, embedding = True)
            if export:
                params["draft_conf"] = conf
            if emb is not None:
                params["draft_embedding"] = emb
            return ids
        logits = head.lm_head.forward(state, params)[..., :self.config.vocab_size]
        if flag == "probe":
            sparse_ids, _ = sh.argmax(state)
            sh.record_probe(torch.argmax(logits, dim = -1), sparse_ids)
        if export:
            conf, ids = torch.max(logits, dim = -1)
            params["draft_conf"] = conf
            return ids
        return torch.argmax(logits, dim = -1)
