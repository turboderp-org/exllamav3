from __future__ import annotations
import json
import os
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Module, Embedding, BlockSparseMLP, Linear
from ..modules.attn import prepare_for_attn
from .inkling2_loop_math import InklingMath, DT, rms, message_queries, softcapped_swiglu

# Inkling-2-Mini (thinkingmachines, model_type "inkling2_loop").
# Reference implementation: HF transformers models/inkling2_loop/modeling_inkling2_loop.py.
# Scope: EXL3 conversion and full-sequence (no KV cache) forward, e.g. for KL divergence checks. Serving the
# converted model is done elsewhere (vLLM); the vision tower is copied unquantized.
#
# Residual state: the model has no single residual stream (block attention residuals), and loop
# passes 2-3 restart from the pre-loop state and read the previous pass. The state passed between
# modules is a bf16 tensor (bsz, seq, NUM_SLOTS, hidden) with the static slot layout below.

SLOT_HIDDEN = 0             # running hidden state
SLOT_BLOCKS = 1             # 1..4: completed attention-residual block states
MAX_BLOCKS = 4
SLOT_PRE_LOOP_HIDDEN = 5    # hidden state at loop entry (passes restart from it)
SLOT_PREV_BLOCKS = 6        # 6..7: previous loop pass block states
SLOT_PREV_HIDDEN = 8        # previous loop pass final hidden state
NUM_SLOTS = 9


class Inkling2LoopConfig(Config):
    arch_string = "Inkling2LoopForConditionalGeneration"

    def __init__(self, directory: str, **kwargs):
        super().__init__(
            directory,
            {"text": Inkling2LoopModel},
            **kwargs
        )
        p = "text_config->"

        # Dimensions
        self.hidden_size = self.read_cfg(int, p + "hidden_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, p + "num_hidden_layers", no_default)
        self.vocab_size = self.read_cfg(int, p + "vocab_size", no_default)
        self.unpadded_vocab_size = self.read_cfg(int, p + "unpadded_vocab_size", self.vocab_size)
        self.rms_norm_eps = self.read_cfg(float, p + "rms_norm_eps", no_default)
        self.final_logit_softcapping = self.read_cfg(float, p + "final_logit_softcapping", None)

        # Global (full) attention
        self.num_q_heads = self.read_cfg(int, p + "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, p + "num_key_value_heads", no_default)
        self.head_dim = self.read_cfg(int, p + "head_dim", no_default)
        self.log_scaling_n_floor = self.read_cfg(int, p + "log_scaling_n_floor", no_default)
        self.log_scaling_alpha = self.read_cfg(float, p + "log_scaling_alpha", no_default)

        # Sliding-window attention
        self.local_layer_ids = set(self.read_cfg(list, p + "local_layer_ids", no_default))
        self.swa_num_q_heads = self.read_cfg(int, p + "swa_num_attention_heads", self.num_q_heads)
        self.swa_num_kv_heads = self.read_cfg(int, p + "swa_num_key_value_heads", self.num_kv_heads)
        self.swa_head_dim = self.read_cfg(int, p + "swa_head_dim", self.head_dim)
        self.sliding_window = self.read_cfg(int, p + "sliding_window_size", no_default)

        # Attention extras
        self.assert_cfg(bool, p + "q_bias", False, True)
        self.assert_cfg(bool, p + "o_bias", False, True)
        self.d_rel = self.read_cfg(int, p + "d_rel", no_default)
        self.rel_extent = self.read_cfg(int, p + "rel_extent", no_default)
        self.qk_l2_clip_threshold = self.read_cfg(float, p + "qk_l2_clip_threshold", no_default)
        self.rel_l2_clip_threshold = self.read_cfg(float, p + "rel_l2_clip_threshold", no_default)
        self.assert_cfg(bool, p + "use_value_norm_no_gamma", True, True)
        self.sconv_kernel_size = self.read_cfg(int, p + "sconv_kernel_size", no_default)

        # Block attention residuals
        self.layers_per_block = self.read_cfg(int, p + "n_layers_per_attn_res_block", no_default)
        self.assert_cfg(bool, p + "attn_res_normalize_values", True, True)

        # MoE
        self.num_experts = self.read_cfg(int, p + "n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, p + "num_experts_per_tok", no_default)
        self.assert_cfg(int, p + "n_shared_experts", 0, True)
        self.moe_intermediate_size = self.read_cfg(int, p + "intermediate_size", no_default)
        self.mlp_gate_softcap = self.read_cfg(float, p + "mlp_gate_softcap", no_default)
        self.mlp_up_softcap = self.read_cfg(float, p + "mlp_up_softcap", no_default)
        self.route_scale = self.read_cfg(float, p + "route_scale", no_default)
        self.assert_cfg(bool, p + "use_gate_bias", True, True)

        # Loop
        self.loop_start = self.read_cfg(int, p + "loop_config->core_start", no_default)
        self.loop_end = self.read_cfg(int, p + "loop_config->core_end", no_default)
        self.loop_passes = self.read_cfg(int, p + "loop_config->num_passes", no_default)
        self.loop_gate_rank = self.read_cfg(int, p + "loop_config->gate_rank", no_default)
        self.assert_cfg(bool, p + "loop_config->is_varloop", False, True)

        self.tie_word_embeddings = False
        self.text_cfg = json.load(open(os.path.join(directory, "config.json")))["text_config"]
        self.vision = None  # vision tower not ported: its tensors are copied unchanged (see get_additional_compiled_tensors)

        self._check_slot_layout()

    def is_global_layer(self, idx: int) -> bool:
        return idx not in self.local_layer_ids

    def is_block_boundary(self, idx: int) -> bool:
        # HF (modeling_inkling2_loop.py:861): is_block_boundary = layer_idx % n_layers_per_attn_res_block == 0.
        # At a boundary layer the model appends that layer's INPUT hidden state to block_states
        # (layers 0, 6, 12, 18 -> block 0 is the embedding output). Loop passes 2-3 read
        # block_states[num_pre_loop_blocks + 1:] of the previous pass (skips the loop-entry block,
        # which every pass shares), i.e. one block (layer 12 input) plus the pass's final hidden.
        return idx % self.layers_per_block == 0

    def _check_slot_layout(self):
        n_blocks = self.num_hidden_layers // self.layers_per_block
        assert n_blocks <= MAX_BLOCKS, f"Inkling2Loop: {n_blocks} attn-res blocks > {MAX_BLOCKS} slots"
        loop_blocks = (self.loop_end - self.loop_start) // self.layers_per_block
        assert loop_blocks <= SLOT_PREV_HIDDEN - SLOT_PREV_BLOCKS, "Inkling2Loop: loop blocks exceed slots"
        assert self.loop_start % self.layers_per_block == 0 and self.loop_end % self.layers_per_block == 0, \
            "Inkling2Loop: loop range must align with attn-res blocks"




# ----------------------------------------------------------------------------------------------------------------
# Modules. The math is inkling2_loop_math.InklingMath (validated against vLLM); these modules only own the
# weights, route the quantizable projections / experts through exllamav3 Linear / BlockSparseMLP, and keep the
# residual slots. No KV cache yet: every forward is a full sequence (conversion, KLD, prefill-only checks).
# ----------------------------------------------------------------------------------------------------------------

STATE_DTYPE = torch.bfloat16    # every slot holds values the HF model keeps in bf16, so this is exact
ATTN_PROJS = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "wo_ud")


def _no_cache(params: dict):
    if params.get("cache") is not None or params.get("past_len"):
        raise NotImplementedError("Inkling2Loop exllamav3 port: no KV cache yet (full-sequence forward only)")


class InklingExperts(BlockSparseMLP):
    """Routed experts with Inkling's softcapped SwiGLU, softcap(g) * sigmoid(g) * softcap(u).

    The fused/quantized fast paths only know silu/gelu, so this activation always takes the dense per-expert path
    (support_quant_paths is false for an unknown activation_fn): correct for conversion and full-sequence
    evaluation, not fast.
    """

    def __init__(self, config: Inkling2LoopConfig, key: str, **kwargs):
        super().__init__(config, key, activation_fn = "silu", **kwargs)
        gate_cap, up_cap = config.mlp_gate_softcap, config.mlp_up_softcap
        self.activation_fn = "inkling_softcap_swiglu"
        self.activation_fn_idx = -1

        def act(g, u, a, act_limit):
            a.copy_(softcapped_swiglu(g, u, gate_cap, up_cap).to(a.dtype))

        self.activation_fn_call = act


class _LayerMath(InklingMath):
    """InklingMath bound to one InklingLayer: projections and experts go through its exllamav3 modules."""

    def __init__(self, layer: "InklingLayer"):
        super().__init__(layer.config.text_cfg, layer.W)
        self.owner = layer
        self.params = None

    def proj(self, key, x):
        linear = self.owner.linears[key.rsplit(".", 1)[-1]]
        return linear.forward(x.half(), self.params, out_dtype = torch.half).to(DT)

    def moe(self, i, x):
        y = self.owner.mlp.forward(x.half().contiguous(), self.params, out_dtype = torch.float)
        return y.view(x.shape).to(DT)


class InklingLayer(Module):
    """One Inkling decoder layer (pass 1 of the loop for layers loop_start..loop_end-1).

    Owns the six attention projections and the experts (quantized) plus the small raw tensors under its key
    prefix (norms, short convs, clip scales, sink and rel-pos projections, loop controllers, router). The
    attention-residual queries live outside the layer prefix; they are read here and compiled through
    Inkling2LoopModel.get_additional_compiled_tensors.
    """

    def __init__(self, config: Inkling2LoopConfig, idx: int):
        key = f"model.llm.layers.{idx}"
        super().__init__(config, key, None)
        self.layer_idx = idx
        self.idx = idx
        c = config
        glob = c.is_global_layer(idx)
        nh = c.num_q_heads if glob else c.swa_num_q_heads
        nkv = c.num_kv_heads if glob else c.swa_num_kv_heads
        hd = c.head_dim if glob else c.swa_head_dim
        D = c.hidden_size
        # (in, out, qmap): q / rel-pos / output gate read the query input, k and v their own inputs (different
        # on global layers, equal on local ones; separate qmaps keep the Hessians right in both cases)
        shapes = {
            "wq_du": (D, nh * hd, "inkling.attn_q"),
            "wr_du": (D, nh * c.d_rel, "inkling.attn_q"),
            "gate_attn": (D, nh * hd, "inkling.attn_q"),
            "wk_dv": (D, nkv * hd, "inkling.attn_k"),
            "wv_dv": (D, nkv * hd, "inkling.attn_v"),
            "wo_ud": (nh * hd, D, "inkling.attn_o"),
        }
        self.linears = {}
        for name in ATTN_PROJS:
            fin, fout, qmap = shapes[name]
            # select_hq_bits = 2: -hq N raises the attention (small, run 3x in the loop) like other MoE ports
            lin = Linear(config, f"{key}.attn.{name}", in_features = fin, out_features = fout, qmap = qmap,
                         out_dtype = torch.half, select_hq_bits = 2)
            self.linears[name] = lin
            self.register_submodule(lin)
        self.mlp = InklingExperts(
            config, f"{key}.mlp",
            hidden_size = D,
            intermediate_size = c.moe_intermediate_size,
            num_experts = c.num_experts,
            num_experts_per_tok = c.num_experts_per_tok,
            key_up = "experts.{expert_idx}.up_proj",
            key_gate = "experts.{expert_idx}.gate_proj",
            key_down = "experts.{expert_idx}.down_proj",
            key_routing_gate = "gate",
            key_e_score_bias = "gate.bias",
            qmap = "block.mlp",
            interm_dtype = torch.half,
            out_dtype = torch.float,
            router_type = "dots",
            routed_scaling_factor = c.route_scale,
        )
        self.register_submodule(self.mlp)
        self.W = {}
        self.math = _LayerMath(self)
        self._raw_keys = None
        self._outside_keys = self._outside_prefix_keys()
        self._stash = None          # quantized tensors of this layer, for the later loop passes (conversion)
        self.msg_query = None

    def _outside_prefix_keys(self):
        c = self.config
        keys = [f"attn_res.mlp_queries.{self.idx}"]
        if self.idx > 0:
            keys.append(f"attn_res.attn_queries.{self.idx - 1}")
        if c.loop_start <= self.idx < c.loop_end:
            keys += ["middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"]
        return keys

    def _own_raw_keys(self):
        """Keys under this layer's prefix that no Linear / BlockSparseMLP owns."""
        stc = self.config.stc
        owned = set()
        for m in self:
            if isinstance(m, Linear):
                owned.add(m.key + ".")
        out = []
        for k in stc.list_tensors(prefix = self.key).keys():
            if ".mlp.experts." in k or k.startswith(self.key + ".mlp.gate."):
                continue
            if any(k.startswith(o) for o in owned):
                continue
            out.append(k)
        return out

    @override
    def can_defer_load(self):
        return False

    @override
    def load(self, device: torch.device, **kwargs):
        source = kwargs.get("source")
        super().load(device, **kwargs)
        stc = self.config.stc
        if self._raw_keys is None:
            self._raw_keys = self._own_raw_keys()
        self.W = {}
        for k in self._raw_keys:
            t = source[k] if source is not None and k in source else stc.get_tensor(k, device, allow_bf16 = True)
            self.W[k[len("model.llm."):]] = t.to(device)
        for k in self._outside_keys:
            self.W[k] = stc.get_tensor("model.llm." + k, device, allow_bf16 = True)
        # the router (gate.weight / gate.bias) is read by the math only through BlockSparseMLP; nothing else
        self.math.W = self.W
        if source is not None and self.config.loop_start <= self.idx < self.config.loop_end:
            self._stash = {k: v.cpu() for k, v in source.items()}
        # msg_query is derived from the loaded tensors on first use (run): under a deferred load (Model.load,
        # module-streamed evaluation) the tensors are only filled at end_deferred_load(), after this returns
        self.msg_query = None

    @override
    def unload(self):
        super().unload()
        self.W = {}
        self.math.W = self.W
        self.msg_query = None

    @override
    def get_tensors(self):
        return {"model.llm." + k: v.contiguous() for k, v in self.W.items() if k not in self._outside_keys}

    @override
    def weights_numel(self):
        return super().weights_numel() + sum(v.numel() for k, v in self.W.items() if k not in self._outside_keys)

    @override
    def optimizer_targets(self):
        q = [t for n in ("wq_du", "wr_du", "gate_attn") for t in self.linears[n].optimizer_targets()]
        k = self.linears["wk_dv"].optimizer_targets() + self.linears["wv_dv"].optimizer_targets()
        o = self.linears["wo_ud"].optimizer_targets()
        return [[q, k, o], self.mlp.optimizer_targets()]

    def run(self, rs: torch.Tensor, params: dict, loop_pass: int) -> torch.Tensor:
        _no_cache(params)
        c = self.config
        i = self.idx
        b, s = rs.shape[:2]
        pos = torch.arange(s, device = rs.device)[None]

        if i == c.loop_start:
            if loop_pass == 0:
                rs[:, :, SLOT_PRE_LOOP_HIDDEN] = rs[:, :, SLOT_HIDDEN]
            else:
                # restart from the loop entry; keep this pass's input to the next: the blocks after the
                # loop-entry block (here the layer-12 input) and the final hidden state of the previous pass
                n_loop_blocks = (c.loop_end - c.loop_start) // c.layers_per_block
                first = SLOT_BLOCKS + c.loop_start // c.layers_per_block + 1
                for j in range(n_loop_blocks - 1):
                    rs[:, :, SLOT_PREV_BLOCKS + j] = rs[:, :, first + j]
                rs[:, :, SLOT_PREV_HIDDEN] = rs[:, :, SLOT_HIDDEN]
                rs[:, :, SLOT_HIDDEN] = rs[:, :, SLOT_PRE_LOOP_HIDDEN]

        messages = None
        if loop_pass > 0:
            n_prev = (c.loop_end - c.loop_start) // c.layers_per_block - 1
            prev_b = rs[:, :, SLOT_PREV_BLOCKS:SLOT_PREV_BLOCKS + n_prev]
            prev_h = rs[:, :, SLOT_PREV_HIDDEN]
            if self.msg_query is None:
                self.msg_query = message_queries(self.W, c.hidden_size)[i - c.loop_start]
            messages = torch.stack(self.math.attn_res(self.msg_query, prev_b, prev_h), dim = -2)

        n_blocks = (i + c.layers_per_block - 1) // c.layers_per_block
        blocks = rs[:, :, SLOT_BLOCKS:SLOT_BLOCKS + n_blocks]
        h = rs[:, :, SLOT_HIDDEN].to(DT)
        self.math.params = params
        try:
            nh = self.math.layer(i, h, blocks, pos, messages)
        finally:
            self.math.params = None
        if c.is_block_boundary(i):
            rs[:, :, SLOT_BLOCKS + n_blocks] = h
        rs[:, :, SLOT_HIDDEN] = nh
        return rs

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.run(x, params, 0)


class InklingLoopPass(Module):
    """Loop pass >= 2 of a looped layer: no weights of its own, runs the pass-1 InklingLayer.

    During conversion the pass-1 layer is quantized first; this module reloads it from the quantized tensors the
    layer stashed, so passes 2-3 advance the calibration state through the quantized weights (the Hessians come
    from pass 1 only). Its key is outside the layer's prefix, so it saves and compiles nothing.
    """

    def __init__(self, config: Inkling2LoopConfig, base: InklingLayer, loop_pass: int):
        super().__init__(config, f"model.llm.loop_pass{loop_pass}.layers.{base.idx}", None)
        self.base = base
        self.loop_pass = loop_pass
        self.loaded_base = False

    @override
    def can_defer_load(self):
        return False

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        if self.base.device is None:
            if self.base._stash is not None:
                self.base.load(device, source = self.base._stash)
            else:
                self.base.load(device, **{k: v for k, v in kwargs.items() if k != "source"})
            self.loaded_base = True

    @override
    def unload(self):
        self.device = None
        if self.loaded_base:
            self.base.unload()
            self.loaded_base = False

    @override
    def get_qmaps(self):
        return set()

    @override
    def get_tensors(self):
        return {}

    @override
    def weights_numel(self):
        return 0

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.base.run(x, params, self.loop_pass)


class InklingEnter(Module):
    """Embedding norm, then the residual-slot state (bsz, seq, NUM_SLOTS, hidden)."""

    def __init__(self, config: Inkling2LoopConfig):
        super().__init__(config, "model.llm.embed_norm", None)
        self.weight = None

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        self.weight = self.config.stc.get_tensor(self.key + ".weight", device, allow_bf16 = True)

    @override
    def unload(self):
        self.device = None
        self.weight = None

    @override
    def get_tensors(self):
        return {self.key + ".weight": self.weight.contiguous()}

    @override
    def weights_numel(self):
        return self.weight.numel() if self.weight is not None else self.config.hidden_size

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        b, s, d = x.shape
        rs = torch.zeros((b, s, NUM_SLOTS, d), dtype = STATE_DTYPE, device = x.device)
        rs[:, :, SLOT_HIDDEN] = rms(x.to(DT), self.weight.to(DT), self.config.rms_norm_eps)
        return rs


class InklingExit(Module):
    """Output attention residual over all blocks + final norm -> half hidden state for the head."""

    def __init__(self, config: Inkling2LoopConfig):
        super().__init__(config, "model.llm.norm", None)
        self.math = InklingMath(config.text_cfg, {})

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        stc = self.config.stc
        self.math.W = {
            "norm.weight": stc.get_tensor("model.llm.norm.weight", device, allow_bf16 = True),
            "attn_res.output_query": stc.get_tensor("model.llm.attn_res.output_query", device, allow_bf16 = True),
        }

    @override
    def unload(self):
        self.device = None
        self.math.W = {}

    @override
    def get_tensors(self):
        return {"model.llm.norm.weight": self.math.W["norm.weight"].contiguous()}

    @override
    def weights_numel(self):
        return self.config.hidden_size

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        n = self.config.num_hidden_layers // self.config.layers_per_block
        h = self.math.output(x[:, :, SLOT_BLOCKS:SLOT_BLOCKS + n], x[:, :, SLOT_HIDDEN].to(DT))
        return h.half()


class Inkling2LoopModel(Model):
    config_class = Inkling2LoopConfig

    def __init__(self, config: Inkling2LoopConfig, **kwargs):
        super().__init__(config, **kwargs)
        c = config
        self.modules += [
            Embedding(config, "model.llm.embed", vocab_size = c.vocab_size, hidden_size = c.hidden_size),
            InklingEnter(config),
        ]
        self.first_block_idx = len(self.modules)
        layers = [InklingLayer(config, i) for i in range(c.num_hidden_layers)]
        self.modules += layers[:c.loop_start]
        self.modules += layers[c.loop_start:c.loop_end]
        for loop_pass in range(1, c.loop_passes):
            self.modules += [InklingLoopPass(config, layers[i], loop_pass) for i in range(c.loop_start, c.loop_end)]
        self.modules += layers[c.loop_end:]
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [
            InklingExit(config),
            Linear(config, "model.llm.unembed", qbits_key = "head_bits", in_features = c.hidden_size,
                   out_features = c.vocab_size, qmap = "block", softcap = c.final_logit_softcapping or 0.0,
                   caps = {"logits_output": True}),
        ]
        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True

    @staticmethod
    def get_additional_compiled_tensors(config: Inkling2LoopConfig) -> dict:
        # Raw tensors outside every module's key prefix: attention-residual queries, cross-pass message queries,
        # and the vision tower (not ported yet; kept so the converted model stays complete for vLLM)
        t = {}
        for prefix in ("model.llm.attn_res", "model.llm.middle_cycle_cross_pass_queries",
                       "model.llm.middle_cycle_cross_pass_kv_queries", "model.visual"):
            t.update(config.stc.list_tensors(prefix = prefix))
        return t

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError()
