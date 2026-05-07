from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..cache import Cache
from ..ext import exllamav3_ext as ext
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn
from ..modules.module import no_p2p_copy
from ..util.tensor import get_for_device


# Synthetic architecture for the MTP head shipped alongside Qwen3.5/3.6 checkpoints.
#
# The MTP weights live inside the original (BF16) parent checkpoint under the "mtp." prefix
# and are *not* part of the main exl3 quant produced by convert.py today. To use this draft:
#
#   - Load the target model normally (e.g. an exl3 quant of Qwen3.6-27B with MTP tensors stripped)
#   - Load this draft model from the original BF16 directory using arch_override
#     ("Qwen3_5MTPDraftModel") so only the mtp.* tensors are picked up
#   - Pass the draft model to Generator(..., draft_model=draft, draft_cache=draft_cache)


class Qwen3_5MTPDraftConfig(Config):
    arch_string = "Qwen3_5MTPDraftModel"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Qwen3_5MTPDraftModel},
            **kwargs
        )

        # The parent config keeps the MTP fields under text_config (LM/VL flat — try both)
        text_cfg_dict = self.read_cfg(dict, "text_config", None)
        text_cfg = "text_config" if text_cfg_dict is not None else None

        def pfx(key):
            return key if text_cfg is None else f"{text_cfg}->{key}"

        # Attention params (same as Qwen3_5)
        self.head_dim = self.read_cfg(int, pfx("head_dim"), None)
        self.hidden_size = self.read_cfg(int, pfx("hidden_size"), no_default)
        self.num_q_heads = self.read_cfg(int, pfx("num_attention_heads"), no_default)
        self.num_kv_heads = self.read_cfg(int, pfx("num_key_value_heads"), self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Dense MLP params
        self.intermediate_size = self.read_cfg(int, pfx("intermediate_size"), no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, pfx("rms_norm_eps"), no_default)

        # MTP layer count + dedicated embeddings
        self.num_hidden_layers = self.read_cfg(int, pfx("mtp_num_hidden_layers"), 1)
        self.mtp_use_dedicated_embeddings = self.read_cfg(bool, pfx("mtp_use_dedicated_embeddings"), False)
        assert self.num_hidden_layers >= 1, "Qwen3_5MTPDraftConfig: mtp_num_hidden_layers must be >= 1"

        # RoPE — match parent (full_attention layers in Qwen3_5 use NEOX with partial rotary factor)
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000000,
            config_dict = self.read_cfg(dict, "text_config", no_default) if text_cfg else None,
        )

        # Vision placeholders (draft is text-only)
        self.vision = None
        self.vision_pp = None


class Qwen3_5MTPDraftModel(Model):
    config_class = Qwen3_5MTPDraftConfig

    def __init__(
        self,
        config: Qwen3_5MTPDraftConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        # Pre-fc norms applied to incoming hidden state and embeddings before they get concatenated
        self.pre_fc_norm_hidden = RMSNorm(
            config = config,
            key = "mtp.pre_fc_norm_hidden",
            rms_norm_eps = config.rms_norm_eps,
            constant_bias = 1.0,
            out_dtype = torch.half,
        )
        self.pre_fc_norm_embedding = RMSNorm(
            config = config,
            key = "mtp.pre_fc_norm_embedding",
            rms_norm_eps = config.rms_norm_eps,
            constant_bias = 1.0,
            out_dtype = torch.half,
        )

        # 2H -> H projection
        self.fc = Linear(
            config = config,
            key = "mtp.fc",
            in_features = config.hidden_size * 2,
            out_features = config.hidden_size,
            qmap = "block.mtp.fc",
            out_dtype = torch.half,
            pad_to = 1,
        )

        # Optional dedicated embedding (rare; default false for Qwen3.6)
        self.dedicated_embedding = None
        if config.mtp_use_dedicated_embeddings:
            self.dedicated_embedding = Embedding(
                config = config,
                key = "mtp.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )

        # Module list: optional embed, then pre_fc norms + fc, then num_mtp_layers * TransformerBlock, then norm
        # All modules are present here purely so the standard loader walks them and loads weights;
        # forward iteration is bypassed (we call step() manually from the generator).
        self.modules = []
        if self.dedicated_embedding is not None:
            self.modules.append(self.dedicated_embedding)
        self.modules.append(self.pre_fc_norm_hidden)
        self.modules.append(self.pre_fc_norm_embedding)
        self.modules.append(self.fc)

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.num_hidden_layers):
            attn = Attention(
                config = config,
                key = f"mtp.layers.{idx}.self_attn",
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
                q_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                out_dtype = torch.float,
                interleaved_gate = True,
            )
            self.attn_modules.append(attn)

            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = f"mtp.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"mtp.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # Final norm
        self.final_norm = RMSNorm(
            config = config,
            key = "mtp.norm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
            constant_bias = 1.0,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "qwen3_5_mtp_draft": True,
            "default_draft_size": 2,  # MTP-1 with 2-step recurrence
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None

    @property
    def _emb_module(self):
        """Embedding module to use: dedicated if present, else target's."""
        return self.dedicated_embedding if self.dedicated_embedding is not None else self.target_embed()


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # MTP doesn't take input_ids through Embedding here — embedding is handled in step()
        # But prepare_for_attn still wires up flash-attn params
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """Bind to target model: borrow embed_tokens / lm_head and tell target to export hidden state.

        We need the trunk's PRE-final-norm hidden state (residual stream after the last transformer
        block, before output_norm). MTP's pre_fc_norm_hidden does its own normalization on top of
        that — feeding post-norm here would double-normalize and hurt acceptance. See llama.cpp PR
        ggml-org/llama.cpp#22673 (`t_h_pre_norm` capture in qwen35.cpp).
        """
        target_modules = target.modules

        # Find the target's embedding (first module of class Embedding)
        target_embed = None
        for m in target_modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        # lm_head is the last module
        assert isinstance(target_modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target_modules[-1])

        # Find the last TransformerBlock and mark it to export its output residual stream
        # (= pre-final-norm hidden). Iterate from the end since vision variants may have
        # DeepstackEmbed/other modules between the last block and output_norm.
        last_block = None
        for m in reversed(target_modules):
            if isinstance(m, TransformerBlock):
                last_block = m
                break
        assert last_block is not None, "Could not locate a TransformerBlock in target to mark for export"
        last_block.export_state = True


    @override
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):
        """
        Convenience forward — caller provides:
          input_ids:    (bsz, num_steps)  ids to embed at each draft step
          params["target_hidden"]:  (bsz, 1, H)  starting hidden state — the trunk's residual stream
            captured BEFORE its final norm (= TransformerBlock.export_state on target's last block).
          params["block_table"], ["cache_seqlens"], ["cache"], ["attn_mode"="flash_attn"]

        Returns logits stacked along seq dim: (bsz, num_steps, vocab_size).
        Use generator's iterate_draftmodel_mtp_gen for the integrated drafting flow.
        """
        if params is None:
            params = {}
        target_hidden = params.get("target_hidden")
        assert target_hidden is not None, "MTP draft forward requires params['target_hidden']"

        bsz, num_steps = input_ids.shape
        last_hidden = target_hidden
        cache_seqlens = params.get("cache_seqlens")
        all_logits = []

        for step_idx in range(num_steps):
            step_params = dict(params)
            if cache_seqlens is not None:
                step_params["cache_seqlens"] = cache_seqlens
            ids_step = input_ids[:, step_idx:step_idx+1]
            new_hidden, logits = self.step(ids_step, last_hidden, step_params, layer_step = step_idx)
            all_logits.append(logits)
            last_hidden = new_hidden
            if cache_seqlens is not None:
                cache_seqlens = cache_seqlens + 1

        return torch.cat(all_logits, dim = 1)

    def _to_dev(self, t: torch.Tensor, dst: torch.device) -> torch.Tensor:
        """Move tensor to device, honoring EXLLAMA_NO_P2P_COPY (host bounce)."""
        if t.device == dst:
            return t
        return t.cpu().to(dst) if no_p2p_copy else t.to(dst)


    def step(
        self,
        last_token_ids: torch.Tensor,
        last_hidden: torch.Tensor,
        params: dict,
        layer_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one MTP step.

          last_token_ids: (bsz, 1) — id of the previously sampled / verified token
          last_hidden:    (bsz, 1, hidden_size) — PRE-final-norm hidden state (residual stream)
                            — for k=0 this is target.last_block's residual; for k>=1 it's the
                              previous step's returned new_hidden (MTP block's residual output).
                              Matches llama.cpp's `t_h_pre_norm` / `t_mtp_out` convention.
          params:         flash-attn params (cache, block_table, cache_seqlens, ...)
          layer_step:     which MTP layer to use (mod num_hidden_layers)

        Returns:
          new_hidden:  (bsz, 1, hidden_size)  MTP block's residual output (pre-MTP-final-norm),
                                              feed back as last_hidden on the next step.
          logits:      (bsz, 1, vocab_size)   computed via final_norm + lm_head on a separate path.
        """
        # Embed last token — keep ids on the embedding's own device (target embed has prefer_cpu=True)
        emb_module = self._emb_module
        last_token_ids = self._to_dev(last_token_ids, emb_module.device)
        embed = emb_module.forward(last_token_ids, params)

        # Pre-fc norms — RMSNorm.forward doesn't auto-move x, so explicitly land x on each norm's device
        # before running it, then bring outputs to fc.device for the cat. Honors EXLLAMA_NO_P2P_COPY.
        embed_for_norm = self.pre_fc_norm_embedding.prepare_for_device(embed, params).half()
        h_for_norm = self.pre_fc_norm_hidden.prepare_for_device(last_hidden, params).half()
        embed_n = self.pre_fc_norm_embedding.forward(embed_for_norm, params, out_dtype = torch.half)
        h_n = self.pre_fc_norm_hidden.forward(h_for_norm, params, out_dtype = torch.half)
        embed_n = self.fc.prepare_for_device(embed_n, params)
        h_n = self.fc.prepare_for_device(h_n, params)

        # Concatenate and project: [embed | hidden] -> hidden
        cat = torch.cat([embed_n, h_n], dim = -1)
        x = self.fc.forward(cat, params, out_dtype = torch.half)

        # Run MTP transformer block — TransformerBlock's submodules (attn_norm/attn/mlp_norm/mlp) assume
        # x is already on the block's device, so move x explicitly. (Mirrors what Model.forward_ls does
        # for top-level modules.)
        layer_idx = layer_step % len(self.attn_modules)
        block = self.modules[self.first_block_idx + layer_idx]
        x = block.prepare_for_device(x, params)
        x = block.forward(x, params)

        # Block output is the residual stream (pre-final-norm). This is what the NEXT AR step
        # consumes — pre_fc_norm_hidden expects pre-norm input. Matches llama.cpp's `t_mtp_out`
        # lm_head needs post-norm so we norm a copy.
        new_hidden = x

        # Apply final norm + lm_head on a separate path (don't propagate post-norm to next step)
        x_normed = self.final_norm.prepare_for_device(x, params)
        x_normed = self.final_norm.forward(x_normed, params, out_dtype = torch.half)
        lm_head = self.target_lm_head()
        x_for_head = lm_head.prepare_for_device(x_normed, params)
        logits = lm_head.forward(x_for_head, params)
        return new_hidden, logits


    def update_kv_from_target(
        self,
        shifted_tokens: torch.Tensor,
        shifted_hiddens: torch.Tensor,
        cache: Cache,
        params: dict,
    ) -> None:
        """
        Write MTP K/V at slots [start, start + n - 1] using already-aligned input pairs.

        K/V convention: slot p = fc(concat[embed(token@(p+1)), hidden@p]).
        Caller is responsible for the shift — pass:
          shifted_tokens[k]  = token at position (start + k + 1)
          shifted_hiddens[k] = hidden at position (start + k)
          params["cache_seqlens"] = (bsz,) tensor of `start` per batch element

        Both inputs have shape (bsz, n, ...).
        """
        assert len(self.attn_modules) == 1, \
            "update_kv_from_target currently supports a single MTP layer (Qwen3.6 default)"
        layer = self.attn_modules[0]

        device = self.fc.device
        bsz, n, hsz = shifted_hiddens.shape
        assert shifted_tokens.shape == (bsz, n), \
            f"shifted_tokens must be (bsz={bsz}, n={n}); got {tuple(shifted_tokens.shape)}"

        # Embed token ids — Embedding may live on CPU (prefer_cpu); land ids on its device first.
        ids_for_embed = self._to_dev(shifted_tokens, self._emb_module.device)
        embeds = self._emb_module.forward(ids_for_embed, {})

        # Pre-fc norms — RMSNorm.forward doesn't auto-move x, so land on each norm's device first.
        embeds_for_norm = self._to_dev(embeds, self.pre_fc_norm_embedding.device).half()
        hidden_for_norm = self._to_dev(shifted_hiddens, self.pre_fc_norm_hidden.device).half()
        embeds_n = self.pre_fc_norm_embedding.forward(embeds_for_norm, {}, out_dtype = torch.half)
        hidden_n = self.pre_fc_norm_hidden.forward(hidden_for_norm, {}, out_dtype = torch.half)
        embeds_n = self._to_dev(embeds_n, device)
        hidden_n = self._to_dev(hidden_n, device)

        # fc projection
        cat = torch.cat([embeds_n, hidden_n], dim = -1)
        x = self.fc.forward(cat, {}, out_dtype = torch.half)

        # Move to layer.device for k/v projections
        x = self._to_dev(x, layer.device)

        block_table = get_for_device(params, "block_table", layer.device)
        cache_seqlens = get_for_device(params, "cache_seqlens", layer.device)

        # k/v project + RoPE
        k = layer.k_proj.forward(x, {})
        v = layer.v_proj.forward(x, {})
        k = k.view(bsz, n, layer.num_kv_heads, layer.head_dim)
        v = v.view(bsz, n, layer.num_kv_heads, layer.head_dim)

        if layer.rope:
            k, _ = layer.rope.apply(
                k, None,
                0,
                cache_seqlens,
                None,
                True,
                layer.k_norm_tensor,
                None,
                layer.norm_eps,
                layer.norm_constant_bias,
                None,
                False,
            )

        cache_k, cache_v = cache.get_layer(layer.layer_idx, cache_seqlens, block_table, -1, 0)
        ext.paged_kv_cache_update(k, v, cache_k, cache_v, block_table, cache_seqlens)
        cache.update_layer(layer.layer_idx, cache_seqlens, block_table, cache_k, cache_v, n, 0)


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}
