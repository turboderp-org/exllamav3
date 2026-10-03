from __future__ import annotations
from typing_extensions import override
import torch
from ..model.model import Model
from ..modules import RMSNorm, TransformerBlock, Attention, GatedMLP
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from .qwen3_5_mtp import Qwen3_5MTPModel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mimo_v2 import MiMoV2Config

class MiMoV2MTPModel(Qwen3_5MTPModel):
    """
    MiMo-V2 next-token prediction heads (model.mtp.layers.{k}): one chain per draft depth k of
    eh_proj over the concatenated normed token embedding and normed target state, a transformer
    block in the target's sliding-window configuration (window, sinks, SWA rope, asymmetric V)
    with a dense MLP, and final_layernorm; embedding and lm_head are the target's. Every head
    consumes the target's post-norm output at the round's anchor, so the Qwen3.5 MTP wiring
    (input layer, attach_to, sampling) is shared.

    The generator passes the draft step in params["draft_step"]; step k runs head min(k, n - 1)
    on the previous draft token, fed the anchor's target state rather than the previous head's
    output (feeding the chain of head outputs, as the reference proposer does with its single
    head, roughly halves the acceptance of the deeper heads on this checkpoint). Each head has
    its own cache layer: the post-verify refresh runs all heads over the accepted tokens, and a
    draft step also writes the deeper heads' entries at its position, so every head's history
    stays complete.
    """

    def __init__(
        self,
        config: MiMoV2Config,
        key_prefix: str = "model",
        **kwargs
    ):
        Model.__init__(self, config, **kwargs)
        self.use_moe = False
        self.num_depths = config.mtp_num_layers
        self.chains = []
        self.attn_modules = []

        for k in range(self.num_depths):
            key = f"{key_prefix}.mtp.layers.{k}"

            input_layer = Qwen3_5MTPInputLayer(
                config = config,
                key = f"{key}.input",
                key_pre_fc_norm_hidden = f"{key}.hnorm",
                key_pre_fc_norm_embedding = f"{key}.enorm",
                key_fc = f"{key}.eh_proj",
                hidden_size = config.hidden_size,
                rms_norm_eps = config.rms_norm_eps,
                native_draft_len = 1,
                out_dtype = torch.float,
                qbits_key = "mtp_bits",
            )

            attn = Attention(
                config = config,
                key = f"{key}.self_attn",
                layer_idx = k,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                v_head_dim = config.v_head_dim,
                num_q_heads = config.swa_num_q_heads,
                num_kv_heads = config.swa_num_kv_heads,
                rope_settings = config.rope_settings_swa,
                sm_scale = None,
                key_fused_qkv = "qkv_proj",
                key_o = "o_proj",
                key_sinks = "attention_sink_bias" if config.add_swa_attention_sink_bias else None,
                # HF window includes the query, the kernels count past keys
                sliding_window = config.sliding_window - 1,
                qmap = "block.attn",
                out_dtype = torch.float,
                qbits_key = "mtp_bits",
            )
            # Same fused-QKV checkpoint layout and V scale fold as the target's SWA layers
            qkv_reader = config.qkv_dequant_swa()
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                proj.fdequant = qkv_reader
            attn.o_proj.weight_scale = config.attention_value_scale
            self.attn_modules.append(attn)

            block = TransformerBlock(
                config = config,
                key = key,
                layer_idx = k,
                attn_norm = RMSNorm(
                    config = config,
                    key = f"{key}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                attn = attn,
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{key}.pre_mlp_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                mlp = GatedMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    qbits_key = "mtp_bits",
                ),
            )

            final_norm = RMSNorm(
                config = config,
                key = f"{key}.final_layernorm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            )

            self.modules += [input_layer, block, final_norm]
            self.chains.append((input_layer, block, final_norm))

        self.input_layer = self.chains[0][0]
        self.final_norm = self.chains[0][2]
        self.first_block_idx = 1
        self.last_kv_module_idx = len(self.modules) - 2

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "mtp_depths": self.num_depths,
            "default_draft_size": self.num_depths,
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None

        # The target state at the round's anchor per sequence (keyed by the sequence's first
        # cache page): the deeper heads' input on the later draft steps
        self._anchor = {}


    def attach_to(self, target):
        super().attach_to(target)
        for input_layer, _, _ in self.chains:
            input_layer.attached_model = self.attached_model


    # ---- Heads -------------------------------------------------------------------------------

    def _seq_keys(self, params: dict, bsz: int) -> list[int]:
        return [int(v) for v in params["block_table"][:bsz, 0].tolist()]

    def _run_chain(self, depth: int, x: torch.Tensor, params: dict, hidden: torch.Tensor) -> torch.Tensor:
        """One head over the prepared input ids x with `hidden` as the previous state; returns
        the post-norm state [bsz, q_len, H]"""
        p = {k: v for k, v in params.items() if k not in ("target_hidden", "dev_cache")}
        p["target_hidden"] = hidden
        p["layer_instance"] = 0
        for m in self.chains[depth]:
            x = m.prepare_for_device(x, p)
            x = m.forward(x, p)
        if params.get("export_draft_conf"):
            params["draft_conf"] = p.get("draft_conf")
        return x


    @override
    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):
        """Draft step params["draft_step"] (default 0) at the cache position: head min(step,
        n - 1) from the anchor's target state (params["target_hidden"] on step 0), then every
        deeper head at the same position from the same inputs, so their caches stay complete"""
        if params is None:
            params = {}
        bsz = input_ids.shape[0]
        step = params.get("draft_step", 0)
        depth = min(step, self.num_depths - 1)
        keys = self._seq_keys(params, bsz)
        x = self.prepare_inputs(input_ids, params)
        if step == 0:
            hidden = params["target_hidden"]
            for b, key in enumerate(keys):
                self._anchor[key] = hidden[b:b + 1, -1:]
        else:
            hidden = torch.cat([self._anchor[key] for key in keys], dim = 0)
        y = self._run_chain(depth, x, params, hidden)
        # Deeper heads' entries at this position: without them their windows hold stale rows
        # at the positions the round's draft steps stop short of
        for j in range(depth + 1, self.num_depths):
            self._run_chain(j, x, params, hidden)
        return y


    @override
    @torch.inference_mode
    def prefill(self, input_ids: torch.Tensor, params: dict | None = None):
        """Refresh every head over the given tokens from params["target_hidden"] (the target's
        states, shifted by one)"""
        if params is None:
            params = {}
        x = self.prepare_inputs(input_ids, params)
        hidden = params["target_hidden"]
        for k in range(self.num_depths):
            self._run_chain(k, x, params, hidden)
