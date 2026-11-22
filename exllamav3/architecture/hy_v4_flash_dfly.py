from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
import weakref
from ..model.model import Model
from ..modules import Module, RMSNorm, TransformerBlock, MLAttention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn
from ..util.device_copy import to_device
from ..util.tensor import get_for_device

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .hy_v4_flash import HyV4FlashConfig

# Hy4 flash MTP component: the DFly2 block drafter stored under model.dfly_layers. One round is a single
# forward over [last sampled token, mask x n] at the next n + 1 positions, followed by a short sequential
# head:
#
# - slot 0's embedding is fused with the target's final hidden state at the last accepted token
# - the layers are plain pre-norm blocks (no hyper-connections, no Engram, dense MLP) whose gated MLA
#   shares the latent K/V cache of one of the target's indexed layers each: a layer reads the positions
#   the target's indexer selected at the last accepted token plus the causal part of its own block,
#   through its own up-projection, and writes the block's rows into the same cache, at the positions
#   the verification pass overwrites
# - for slots 1 to n in turn, a small MLP corrects the hidden state with the embedding of the previous
#   token and the target's head picks the next one
# - a confidence head reads the same corrected state: a logit trained towards the probability that the
#   target accepts the slot's token, which is the score the dynamic draft length is cut by
#
# The drafter keeps no K/V of its own: no cache, no prefill, nothing to rewind. What it carries between
# rounds is the target's state and selections for the last accepted token, which the target exports.
# Reference: vLLM models/hy_v4/nvidia/dfly*.py; the drafter is AngelSpec's DFly2 (models/draft/dflash.py,
# hy4_mla.py, models/dspark.py for the confidence head, which vLLM leaves unused)


# The block is the seed and one mask per draft token: the checkpoint's own draft size by default, as
# many as the caller asks for up to this many rows (the masks all embed alike, and a row only sees the
# ones before it)
MAX_BLOCK_ROWS = 32


class HyV4DFlyInputLayer(Module):
    """
    Seed token -> (bsz, block, hidden) fp32: the target's embeddings of [seed, mask...], the seed's fused
    with the target state from params["target_hidden"]
    """

    def __init__(self, config: HyV4FlashConfig, key: str):
        super().__init__(config, key, None)
        self.hidden_size = config.hidden_size
        self.mask_token_id = config.dfly_mask_token_id
        self.block_size = config.dfly_block_size
        self.max_rows = MAX_BLOCK_ROWS
        self.enorm = RMSNorm(config, f"{key}.enorm", config.rms_norm_eps, out_dtype = torch.half)
        self.hnorm = RMSNorm(config, f"{key}.hnorm", config.rms_norm_eps, out_dtype = torch.half)
        self.eh_proj = Linear(
            config = config,
            key = f"{key}.eh_proj",
            in_features = 2 * config.hidden_size,
            out_features = config.hidden_size,
            qmap = "block.mtp.fc",
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
            select_hq_bits = 1,
        )
        self.register_submodule(self.enorm)
        self.register_submodule(self.hnorm)
        self.register_submodule(self.eh_proj)
        self.gate = None
        self.mask_emb = None
        self.target_embedding = None
        self.caps.update({"x_cpu": True})

    @override
    def optimizer_targets(self):
        return self.eh_proj.optimizer_targets()

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.gate = self.config.stc.get_tensor(
            f"{self.key}.gate", device, no_defer = True, allow_bf16 = True, arena = False
        ).float()

    @override
    def unload(self):
        super().unload()
        self.gate = None
        self.mask_emb = None

    @override
    def get_tensors(self):
        return {f"{self.key}.gate": self.gate.contiguous()}

    @override
    def weights_numel(self):
        return super().weights_numel() + self.hidden_size

    @override
    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        target_hidden = params.get("target_hidden")
        assert target_hidden is not None and target_hidden.shape[:-1] == x.shape, \
            "Hy4 DFly requires the target's state for each seed token"
        te = self.target_embedding()
        seed = to_device(te.forward(x, params, out_dtype = torch.half), self.device)
        params["dfly_seed_emb"] = seed
        if self.mask_emb is None:
            mask_id = torch.full((1, 1), self.mask_token_id, dtype = torch.long)
            self.mask_emb = to_device(te.forward(mask_id, {}, out_dtype = torch.half), self.device).float()
        state = get_for_device(params, "target_hidden", self.device).half()
        fused = self.eh_proj.forward(
            torch.cat((self.enorm.forward(seed, params), self.hnorm.forward(state, params)), dim = -1),
            params,
            out_dtype = torch.float,
        )
        # The block's rows land in the target's cache, so it is cut to what the caller has room for; rows only
        # see the ones before them, which leaves the rest as they are
        rows = min(params.get("draft_block_window", self.block_size) + 1, self.max_rows)
        y = self.mask_emb.repeat(x.shape[0], rows, 1)
        y[:, :1] = seed.float() + self.gate * fused
        return y


class HyV4DFlyHead(Module):
    """
    Terminal module owning the hidden correction, applied step by step in sample_from_state. Passes the
    normed block states through unchanged
    """

    def __init__(self, config: HyV4FlashConfig, key: str):
        super().__init__(config, key, None)
        h, w = config.hidden_size, config.dfly_correction_size
        self.hidden_norm = RMSNorm(config, f"{key}.hidden_norm", config.rms_norm_eps, out_dtype = torch.half)
        self.embed_norm = RMSNorm(config, f"{key}.embed_norm", config.rms_norm_eps, out_dtype = torch.half)
        def linear(name: str, in_features: int, out_features: int, qmap: str, out_dtype):
            return Linear(
                config = config,
                key = f"{key}.{name}",
                in_features = in_features,
                out_features = out_features,
                qmap = qmap,
                out_dtype = out_dtype,
                qbits_key = "mtp_bits",
                select_hq_bits = 1,
            )
        self.gate_proj = linear("gate_proj", 2 * h, w, "block.mlp.ug", torch.half)
        self.up_proj = linear("up_proj", 2 * h, w, "block.mlp.ug", torch.half)
        self.down_proj = linear("down_proj", w, h, "block.mlp.d", torch.float)
        for m in (self.hidden_norm, self.embed_norm, self.gate_proj, self.up_proj, self.down_proj):
            self.register_submodule(m)

    @override
    def optimizer_targets(self):
        return [[self.gate_proj.optimizer_targets(), self.up_proj.optimizer_targets()], self.down_proj.optimizer_targets()]

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        return x

    def correct(self, hidden: torch.Tensor, prev_emb: torch.Tensor, params: dict) -> torch.Tensor:
        """hidden, prev_emb: (bsz, 1, hidden) fp16, the block state of a slot and the embedding of the
        token before it"""
        c = torch.cat((self.hidden_norm.forward(hidden, params), self.embed_norm.forward(prev_emb, params)), dim = -1)
        a = F.silu(self.gate_proj.forward(c, params).float()) * self.up_proj.forward(c, params).float()
        d = self.down_proj.forward(a.half(), params, out_dtype = torch.float)
        return (hidden.float() + d).half()


class HyV4FlashDFlyModel(Model):

    def __init__(
        self,
        config: HyV4FlashConfig,
        key_prefix: str = "model.dfly_layers",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        n = config.dfly_num_layers
        assert n > 0 and len(config.dfly_kv_layers) == n and len(config.dfly_index_layers) == n and \
            len(set(config.dfly_kv_layers)) == n, \
            "HyV4Flash DFly: one target K/V layer and one target index layer per draft layer"
        for t in config.dfly_kv_layers + config.dfly_index_layers:
            assert 0 <= t < config.num_hidden_layers and config.layer_types[t] == "indexed", \
                f"HyV4Flash DFly: target layer {t} is not an indexed layer"
        assert not config.dfly_local_window, "HyV4Flash DFly: a local context window is not supported"

        self.block_size = config.dfly_block_size
        # The target's selection, as wide as its layers hand it over, followed by the block itself
        self.history_width = -(-config.index_topk // 32) * 32

        self.input_layer = HyV4DFlyInputLayer(config, f"{key_prefix}.mtp_fusion")
        self.modules += [self.input_layer]
        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(n):
            key = f"{key_prefix}.{idx}"
            attn = MLAttention(
                config = config,
                key = f"{key}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                num_q_heads = config.num_q_heads,
                kv_lora_rank = config.kv_lora_rank,
                qk_nope_head_dim = config.qk_nope_head_dim,
                qk_rope_head_dim = config.qk_rope_head_dim,
                v_head_dim = config.v_head_dim,
                rope_settings = config.rope_settings_indexed,
                q_lora_rank = config.q_lora_rank,
                rms_norm_eps = config.rms_norm_eps,
                qmap = "block.attn",
                out_dtype = torch.float,
                qbits_key = "mtp_bits",
                select_hq_bits = 2,
                key_out_gate = "linear_gate",
                key_sinks = "learnable_sink_param",
                indexer_mode = "external",
                index_topk = self.history_width + MAX_BLOCK_ROWS,
                cache_layer_idx = config.dfly_kv_layers[idx],
            )
            self.attn_modules.append(attn)
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(config, f"{key}.input_layernorm", config.rms_norm_eps),
                    attn = attn,
                    attn_post_norm = RMSNorm(config, f"{key}.self_attn.post_proj_norm", config.rms_norm_eps),
                    mlp_norm = RMSNorm(config, f"{key}.post_attention_layernorm", config.rms_norm_eps),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.dfly_intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        activation_fn = "silu",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        qbits_key = "mtp_bits",
                        select_hq_bits = 1,
                    ),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1
        self.final_norm = RMSNorm(config, f"{key_prefix}.final_norm", config.rms_norm_eps, out_dtype = torch.half)
        self.head = HyV4DFlyHead(config, f"{key_prefix}.hidden_correction")
        self.modules += [self.final_norm, self.head]
        self.fwd_end_idx = len(self.modules)

        # Confidence head, unquantized. A drafter converted without it scores by the winning logit
        self.confidence = None
        if config.dfly_confidence_head:
            self.confidence = Linear(
                config = config,
                key = f"{key_prefix}.confidence_head.proj",
                in_features = config.hidden_size,
                out_features = 1,
                qmap = None,
                pad_to = 1,
            )
            self.modules += [self.confidence]

        self.logit_layer_idx = None
        self.caps.update({
            "uncalibrated_quantize": True,
            "supports_tp": False,
            "attach_target": True,
            # Reads and writes the target's cache layers in place: loaded by the target's load (see
            # modules_with_target), never on its own
            "load_with_target": True,
            "mtp_draft": True,
            "draft_uses_target_cache": True,
            # One forward drafts the whole window
            "block_draft": True,
            "default_draft_size": self.block_size,
            "autosplit_load_fwd": False,
        })
        self.attached_model = None
        # Which of the target's exported selections each layer reads (see attach_to)
        self.selection_slots = None


    def attach_to(self, target):
        """
        Bind to the target: its embedding table for the input layer and the correction head, and the exports
        each verification (and the last prefill) forward leaves for the next round: the selections of the
        target's index layers, in layer order, then its post-final-norm state
        """
        config = self.config
        assert not target.loaded_tp, "HyV4Flash DFly: tensor-parallel targets are not supported"
        self.attached_model = weakref.ref(target)
        self.input_layer.target_embedding = weakref.ref(target.modules[0])
        target_norm = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_norm, RMSNorm), "Expected the target's final RMSNorm right before its head"

        target_attn = {m.attn.layer_idx: m.attn for m in target.modules if isinstance(getattr(m, "attn", None), MLAttention)}
        for attn, t in zip(self.attn_modules, config.dfly_kv_layers):
            src = target_attn[t]
            assert (src.kv_lora_rank, src.qk_rope_head_dim) == (attn.kv_lora_rank, attn.qk_rope_head_dim), \
                f"{attn.key}: latent K/V geometry differs from target layer {t}"
        index_layers = sorted(set(config.dfly_index_layers))
        for t in index_layers:
            src = target_attn[t]
            assert src.indexer_mode == "full" and -(-src.index_topk // 32) * 32 == self.history_width, \
                f"HyV4Flash DFly: target layer {t} makes no selection of the expected width"
        self.selection_slots = [index_layers.index(t) for t in config.dfly_index_layers]
        self.draft_verifier_params.update({
            "export_mla_selection": frozenset(index_layers),
            "export_state_norm_keys": {target_norm.key},
        })


    def target_source_keys(self) -> set[str]:
        """The target's attention modules whose cache the drafter shares, by key. Call after attach_to()"""
        target = self.attached_model()
        layers = set(self.config.dfly_kv_layers)
        return {
            m.attn.key for m in target.modules
            if isinstance(getattr(m, "attn", None), MLAttention) and m.attn.layer_idx in layers
        }


    def modules_with_target(self, target) -> dict[int, list]:
        """
        Placement for the target's loader: the index of the target module each of this drafter's modules loads
        right after, on the same device. Every block follows the target layer whose cache it shares (and whose
        selection reaches it on that device); the input layer goes with the first block, the final norm with
        the last, and the correction head with the target's head, which it alternates with. Call after
        attach_to()
        """
        assert self.attached_model is not None and self.attached_model() is target, \
            "HyV4Flash DFly: attach_to(target) before planning its placement"
        src_index = {
            m.attn.layer_idx: i for i, m in enumerate(target.modules)
            if isinstance(getattr(m, "attn", None), MLAttention)
        }
        block_targets = [src_index[t] for t in self.config.dfly_kv_layers]
        plan = {}
        def place(module, t):
            plan.setdefault(t, []).append(module)
        place(self.input_layer, block_targets[0])
        for i, t in enumerate(block_targets):
            place(self.modules[self.first_block_idx + i], t)
        place(self.final_norm, block_targets[-1])
        place(self.head, target.logit_layer_idx)
        if self.confidence is not None:
            place(self.confidence, target.logit_layer_idx)
        return plan


    def block_selection(self, layer: int, params: dict, bsz: int, rows: int) -> tuple:
        """
        The positions each row of a layer's block attends over: the target's selection for the last accepted
        token (params["target_aux"], (bsz, 1, history_width) per exported layer) and the block's own rows up
        to and including the row itself. Returns ((bsz * rows, K_pad) int32, columns in use)
        """
        attn = self.attn_modules[layer]
        device = attn.device
        hw = self.history_width
        history = to_device(params["target_aux"][self.selection_slots[layer]], device)
        assert history.shape == (bsz, 1, hw), "HyV4Flash DFly: unexpected selection carried from the target"
        key = ("dfly_block_sel", device)
        own = params.get(key)
        if own is None:
            anchor = get_for_device(params, "cache_seqlens", device).view(bsz, 1, 1).int()
            r = torch.arange(rows, dtype = torch.int32, device = device)
            own = torch.where(r.view(1, 1, rows) <= r.view(1, rows, 1), anchor + r.view(1, 1, rows), -1)
            params[key] = own
        k_pad = -(-attn.index_topk // 32) * 32
        sel = torch.full((bsz, rows, k_pad), -1, dtype = torch.int32, device = device)
        sel[:, :, :hw] = history
        sel[:, :, hw : hw + rows] = own
        return sel.view(bsz * rows, k_pad), hw + rows


    @override
    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict, **kwargs) -> torch.Tensor:
        """Seed token per row -> the normed block states (bsz, block + 1, hidden) fp16 for sample_from_state.
        params: the target's cache, block table and K/V positions, target_hidden and target_aux"""
        x = self.prepare_inputs(input_ids, params)
        bsz = input_ids.shape[0]
        for i, m in enumerate(self.modules[:self.fwd_end_idx]):
            x = m.prepare_for_device(x, params)
            layer = i - self.first_block_idx
            if 0 <= layer < len(self.attn_modules):
                params["mla_selection"] = self.block_selection(layer, params, bsz, x.shape[1])
            x = m.forward(x, params)
        return x


    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Greedy tokens for the block's slots, in sequence: each slot's state is corrected with the embedding of
        the token before it (the seed first) and goes through the target's head, or its sparse subset when the
        generator asks for it. Returns (bsz, window) ids, with the draft confidence per slot when asked:
        the confidence head's logit for the corrected state, or the winning logit without that head
        """
        target = self.attached_model()
        head = self.head
        te = self.input_layer.target_embedding()
        window = min(params.get("draft_window", self.block_size), state.shape[1] - 1)
        state = to_device(state, head.device)
        prev_emb = to_device(params["dfly_seed_emb"], head.device)
        params.pop("draft_embedding", None)
        export = bool(params.get("export_draft_conf"))
        ids, conf = [], []
        for step in range(window):
            h = head.correct(state[:, step + 1 : step + 2], prev_emb, params)
            step_ids = target.lm_head_argmax(h, params)
            ids.append(step_ids)
            if export and self.confidence is not None:
                conf.append(self.confidence.forward(h, params).float().view(-1, 1))
            elif export:
                conf.append(params["draft_conf"])
            if step + 1 < window:
                # The sparse head hands back the winning rows of the embedding table; otherwise the ids take
                # a host round trip to the table
                prev_emb = params.pop("draft_embedding", None)
                if prev_emb is None:
                    prev_emb = te.forward(to_device(step_ids, te.device), {}, out_dtype = torch.half)
                prev_emb = to_device(prev_emb, head.device)
        params.pop("draft_embedding", None)
        if export:
            params["draft_conf"] = torch.cat([to_device(c, conf[0].device) for c in conf], dim = 1)
        return torch.cat(ids, dim = 1)


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)


    @override
    def prefill(self, input_ids: torch.Tensor, params: dict | None = None):
        # No K/V of its own: the drafter shares the target's cache, so there is nothing to prefill
        return


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("Draft model does not have its own chat template")


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}
