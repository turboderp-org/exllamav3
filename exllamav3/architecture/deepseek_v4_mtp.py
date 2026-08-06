from __future__ import annotations
from typing_extensions import override
import torch

import weakref

from ..cache import Cache
from ..model.model import Model
from ..modules import RMSNorm, Linear, Embedding, GatedMLP, BlockSparseMLP, TransformerBlock, \
    HyperConnection, HyperHead
from ..modules.arch_specific.dspark import DSparkAttention, DSparkInputLayer, to_dev
from ..modules.attn import prepare_for_attn
from ..modules.module import no_p2p_copy
from ..util.tensor import get_for_device

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .deepseek_v4 import DeepseekV4Config

# DeepSeek-V4 MTP component = the DSPARK block drafter, stored under the mtp.* namespace:
# n_mtp_layers full transformer blocks (compressor-less DSA attention, 256-expert noaux MoE
# with shared expert, mHC residual streams), entered through main_proj/main_norm on block 0
# (projecting the concatenation of the trunk's stream-mean taps at dspark_target_layer_ids)
# and exited through block N-1's own hc_head collapse + final norm into the SHARED trunk
# head. The last block also carries a factorized-bigram markov head (per-token logit bias
# during the sequential sampling loop) and a confidence head (per-position score for
# dynamic draft length). Reference implementation ships in the checkpoint:
# hf/inference/model.py (DSparkBlock / DSparkAttention / forward_spec).


class DeepseekV4MTPModel(Model):

    def __init__(
        self,
        config: DeepseekV4Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        h = config.hidden_size
        n_taps = max(len(config.dspark_target_layer_ids), 1)

        # Entry: main_proj / main_norm (consumed by update_kv_from_target) + the noise-block
        # embedder for the draft forward
        self.input_layer = DSparkInputLayer(
            config = config,
            key = "mtp.input",
            hidden_size = h,
            n_taps = n_taps,
            hc_mult = config.hc_mult,
            noise_token_id = config.dspark_noise_token_id,
            block_size = config.dspark_block_size,
            rms_norm_eps = config.rms_norm_eps,
        )
        self.modules = [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.num_mtp_layers):
            layer_type = config.mtp_layer_types[idx]
            assert layer_type == "sliding", \
                f"DeepseekV4 MTP: expected compressor-less (sliding) blocks, got {layer_type}"
            key = f"mtp.{idx}"
            attn = DSparkAttention(
                config = config,
                key = f"{key}.attn",
                layer_idx = idx,
                layer_type = layer_type,
                hidden_size = h,
                num_q_heads = config.num_q_heads,
                head_dim = config.head_dim,
                rope_head_dim = config.qk_rope_head_dim,
                q_lora_rank = config.q_lora_rank,
                o_groups = config.o_groups,
                o_lora_rank = config.o_lora_rank,
                sliding_window = config.sliding_window,
                compress_rate = None,
                index_n_heads = config.index_n_heads,
                index_head_dim = config.index_head_dim,
                index_topk = config.index_topk,
                rope_theta = config.rope_theta,
                compress_rope_theta = config.compress_rope_theta,
                rope_scaling = config.rope_scaling,
                rms_norm_eps = config.rms_norm_eps,
                qmap = "block.attn",
                out_dtype = torch.float,
                qbits_key = "mtp_bits",
                select_hq_bits = 2,
            )
            self.attn_modules.append(attn)
            mlp = BlockSparseMLP(
                config = config,
                key = f"{key}.ffn",
                hidden_size = h,
                intermediate_size = config.moe_intermediate_size,
                num_experts = config.num_experts,
                num_experts_per_tok = config.num_experts_per_tok,
                key_up = "experts.{expert_idx}.w3",
                key_gate = "experts.{expert_idx}.w1",
                key_down = "experts.{expert_idx}.w2",
                key_routing_gate = "gate",
                key_e_score_bias = "gate.bias",
                qmap = "block.mlp",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                activation_fn = "silu",
                act_limit = config.swiglu_limit,
                router_type = "sqrtsp",
                routed_scaling_factor = config.routed_scaling_factor,
                qbits_key = "mtp_bits",
                shared_experts = GatedMLP(
                    config = config,
                    key = f"{key}.ffn.shared_experts",
                    hidden_size = h,
                    intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                    key_up = "w3",
                    key_gate = "w1",
                    key_down = "w2",
                    qmap = "block.mlp",
                    out_dtype = torch.float,
                    activation_fn = "silu",
                    act_limit = config.swiglu_limit,
                    qbits_key = "mtp_bits",
                    select_hq_bits = 2,
                ),
            )
            def _hc(tag: str):
                return HyperConnection(
                    config = config,
                    key = f"{key}.hc_{tag}",
                    hc_mult = config.hc_mult,
                    hidden_size = h,
                    sinkhorn_iters = config.hc_sinkhorn_iters,
                    hc_eps = config.hc_eps,
                    rms_norm_eps = config.rms_norm_eps,
                )
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(config, f"{key}.attn_norm", config.rms_norm_eps),
                    attn = attn,
                    attn_hc = _hc("attn"),
                    mlp_norm = RMSNorm(config, f"{key}.ffn_norm", config.rms_norm_eps),
                    mlp = mlp,
                    mlp_hc = _hc("ffn"),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1
        last = f"mtp.{config.num_mtp_layers - 1}"

        # Exit: the drafter's own stream collapse + final norm (logits via the trunk's
        # shared head at runtime). The draft forward runs modules[:fwd_end_idx]
        self.modules += [
            HyperHead(
                config = config,
                key = f"{last}.hc_head",
                hc_mult = config.hc_mult,
                rms_norm_eps = config.rms_norm_eps,
                hc_eps = config.hc_eps,
            ),
            RMSNorm(
                config = config,
                key = f"{last}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
        ]
        self.fwd_end_idx = len(self.modules)

        # Markov bigram head (per-token logit bias in the sampling loop) and confidence
        # head (per-position draft-length score), unquantized
        self.markov_w1 = Embedding(
            config = config,
            key = f"{last}.markov_head.markov_w1",
            vocab_size = config.vocab_size,
            hidden_size = config.dspark_markov_rank,
        )
        # Device-resident (~63 MB): the sampling loop stays on-stream with no host syncs
        self.markov_w1.caps["prefer_cpu"] = False
        self.markov_w2 = Linear(
            config = config,
            key = f"{last}.markov_head.markov_w2",
            in_features = config.dspark_markov_rank,
            out_features = config.vocab_size,
            qmap = None,
        )
        self.confidence = Linear(
            config = config,
            key = f"{last}.confidence_head.proj",
            in_features = h + config.dspark_markov_rank,
            out_features = 1,
            qmap = None,
            pad_to = 1,
        )
        self.modules += [self.markov_w1, self.markov_w2, self.confidence]

        # Draft length gate: keep the longest prefix with sigmoid(confidence) >= threshold
        import os
        self.draft_conf_threshold = float(os.environ.get("EXL3_DSPARK_CONF", "0.5"))
        self._conf_stats = [] if os.environ.get("EXL3_DSPARK_CONF_STATS") else None

        self.logit_layer_idx = None
        self.caps.update({
            "supports_tp": False,
            "autosplit_load_fwd": False,
            "attach_target": True,
            "dflash_draft": True,
            "default_draft_size": config.dspark_block_size,
        })
        self.attached_model = None
        # Private trunk embedding/head instances, loaded lazily by attach_to when the
        # target is tensor-parallel (its own embedding and head then live in the rank
        # worker processes and cannot be borrowed)
        self.own_embed = None
        self.own_head = None
        self.draft_verifier_params.update({
            "export_state_layers": set(config.dspark_target_layer_ids),
        })

        # Activate all experts during any calibrated capture pass
        self.calibration_all_experts = True


    def attach_to(self, target):
        self.attached_model = weakref.ref(target)
        self.input_layer.attached_model = weakref.ref(target)
        if getattr(target, "loaded_tp", False):
            self._load_own_embed_head()


    def _load_own_embed_head(self):
        """Load private copies of the trunk's embedding (system RAM, prefer_cpu) and lm
        head (drafter's device). Only needed when the target runs TP; keeping the local
        head also keeps the sequential markov/argmax draft loop free of collectives."""
        if self.own_head is not None:
            return
        cfg = self.config
        embed = Embedding(
            config = cfg,
            key = "embed",
            vocab_size = cfg.vocab_size,
            hidden_size = cfg.hidden_size,
        )
        embed.load(torch.device("cpu"))
        head_alt_key = None
        if cfg.tie_word_embeddings and not cfg.stc.has_tensor("head"):
            head_alt_key = "embed"
        head = Linear(
            config = cfg,
            key = "head",
            qbits_key = "head_bits",
            alt_key = head_alt_key,
            in_features = cfg.hidden_size,
            out_features = cfg.vocab_size,
            caps = {"logits_output": True},
        )
        head.load(self.modules[self.fwd_end_idx - 1].device)
        self.own_embed = embed
        self.own_head = head
        self.input_layer.own_embed = embed


    @override
    def forward(self, input_ids: torch.Tensor, params: dict, **kwargs) -> torch.Tensor:
        """Draft-block forward: seed token per row -> [seed, noise x (block-1)] -> input
        layer (embed + stream expand) -> blocks -> hc_head collapse -> norm. Returns the
        normed state (bsz, block, hidden) fp16 for sample_from_state."""
        x = self.prepare_inputs(input_ids, params)
        params["dspark_seed_ids"] = input_ids
        for i, m in enumerate(self.modules[:self.fwd_end_idx]):
            x = m.prepare_for_device(x, params)
            if i == self.fwd_end_idx - 1:
                params["dspark_prenorm"] = x.clone()    # confidence input is PRE-norm
            x = m.forward(x, params)
        return x


    def update_kv_from_target(
        self,
        target_hidden: list,
        cache: Cache,
        params: dict,
        lengths: list[int] = None,
    ):
        """Write main-kv rows for freshly accepted/prefilled target positions: concat the
        trunk tap states, project through main_proj/main_norm once, then per block derive
        and store that block's kv rows (paged, aligned with the target block tables)."""
        if lengths is not None:
            max_length = max(lengths)
            target_hidden = [t[:, :max_length] for t in target_hidden]
        device = self.input_layer.main_proj.device
        for i in range(len(target_hidden)):
            if target_hidden[i].device != device:
                if no_p2p_copy:
                    target_hidden[i] = target_hidden[i].cpu().to(device)
                else:
                    target_hidden[i] = target_hidden[i].to(device)
        th = torch.cat(target_hidden, dim = -1)
        params = dict(params)
        params["cache"] = cache
        params["dspark_main_x"] = self.input_layer.project_taps(th, {})
        for attn in self.attn_modules:
            mx = get_for_device(params, "dspark_main_x", attn.device)
            attn.update_kv_rows(mx, params)


    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """Trunk head over all block positions at once, then the sequential greedy loop
        with the markov bigram bias. Returns (bsz, block + 1) ids [seed, drafts...]; the
        generator crops the seed."""
        import torch.nn.functional as F
        if self.own_head is not None:
            lm = self.own_head
        else:
            am = self.attached_model()
            lm = am.modules[am.logit_layer_idx]
        logits = lm.forward(lm.prepare_for_device(state, params), params).float()
        b, s, V = logits.shape
        # Sequential in the sampled chain but fully on-device: embedding gather + bias
        # gemv + argmax per step, no host round trips (single sync when consumed)
        dev = self.markov_w2.device
        logits = to_dev(logits, dev)
        seed = to_dev(params["dspark_seed_ids"], dev)
        if getattr(self, "_markov_w1_dev", None) is None:
            self._markov_w1_dev = to_dev(self.markov_w1.embedding.weight.data, dev)
        w2 = self.markov_w2
        out = torch.empty((b, s + 1), dtype = torch.long, device = dev)
        out[:, 0] = seed[:, -1]
        embs = []
        for i in range(s):
            emb = F.embedding(out[:, i], self._markov_w1_dev).half()
            embs.append(emb)
            bias = w2.forward(emb.unsqueeze(1), params)
            logits[:, i] += bias[:, 0].float()
            out[:, i + 1] = torch.argmax(logits[:, i], dim = -1)

        # Confidence-capped draft length: proj(cat([pre-norm hidden, markov emb])) per
        # position; the generator clamps its window to the longest all-confident prefix
        # (batch max), 0 = skip drafting this round
        cdev = self.confidence.device
        xpre = to_dev(params["dspark_prenorm"], cdev).half()
        me = to_dev(torch.stack(embs, dim = 1), cdev).half()
        conf = self.confidence.forward(torch.cat((xpre, me), dim = -1), params)
        cs = torch.sigmoid(conf.float().squeeze(-1))
        if self._conf_stats is not None:
            self._conf_stats.append(cs[0].tolist())
        keep = cs >= self.draft_conf_threshold
        lens = torch.cumprod(keep.to(torch.int32), dim = 1).sum(dim = 1)
        params["draft_confidence_len"] = int(lens.max().item())
        return out


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}
