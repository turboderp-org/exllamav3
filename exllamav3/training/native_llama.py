"""
Transformers-free differentiable Llama forward over native EXL3 weights.

Why this exists
---------------
The HF Transformers integration (``exllamav3/integration/transformers.py``)
turns an EXL3 model into a trainable graph by replacing only the linear
layers and letting stock Transformers supply norms/attention/RoPE. That works
in principle but couples training to a single Transformers version: the EXL3
Llama-3.2 weights were calibrated against transformers 4.45, and 5.x changed
``llama3`` RoPE handling, producing a correct-per-layer but garbage-overall
forward (see ``doc/qlora_handoff.md``).

This module sidesteps that entirely. It reconstructs the Llama decoder forward
in plain, autograd-friendly PyTorch directly on top of exllamav3's *own* loaded
modules -- the same weights, RoPE settings (``RoPE.inv_freq``), norms and
attention scale that the native (correct) inference forward uses. There is no
``transformers`` import anywhere in the path, so it cannot be broken by an
upstream version bump.

The quantized base weights stay frozen and are reconstructed on the fly from
the trellis via ``LinearEXL3.get_weight_tensor()`` (orientation ``[in, out]``,
``y = x @ W``). Only the low-rank ``lora_a`` / ``lora_b`` adapters train, routed
through the gradchecked :class:`EXL3LoRAFunction`. The LM head is handled by the
streaming :func:`fused_linear_cross_entropy`, so the ``[tokens, vocab]`` logit
tensor is never materialized during training.

Scope: pre-norm decoders. Every norm / activation / scale is read from the
loaded modules (see ``backbone.norm_spec``), so one block forward covers
Llama/Mistral/Qwen2 (plain), Qwen3 (q/k-norm), Gemma3/4 (q/k/v-norm + sandwich
post-norms + GeGLU + alternating sliding/full window + per-layer head dims +
logit softcapping), the Qwen3.5/3.6 hybrid: GatedDeltaNet layers (a
differentiable gated delta rule -- see ``training.gdn``; the fla
``chunk_gated_delta_rule`` Triton kernel on CUDA fp16/bf16, the sequential
fp32 reference elsewhere) and gated full-attention layers (interleaved output
gate), AFMoE / Trinity blocks (a separate full-width attention output gate
keyed ``self_attn.gate_proj``, NoPE full-attention layers -- RoPE only on
the sliding layers -- muP embedding multiplier, four-norm sandwich blocks,
dense-first-N-layers), MuseGlimmer text towers (the same full-width attention
gate, scaleless q/k-norm with a q scale factor folded into sm_scale, sandwich
norms, per-layer RoPE theta with NoPE full-attention layers, an unweighted
norm on the token embeddings, and a logit pre-scale on the softcapped head),
and BlockSparseMLP mixtures of experts with the std
softmax top-k
router (Qwen3-MoE, Qwen3.5-MoE incl. the shared expert + sigmoid shared
gate, and the Gemma4 MoE layout: routing + routed experts fed from the raw
post-attention residual through their own pre-norms, routed/shared post
norms, per-expert scale) or the "dots" sigmoid router (AFMoE: selection
bias, normalize-over-selected weights, routed scaling factor, ungated
shared expert) -- see ``_moe_out``. It reduces bit-identically to
the original Llama
path when those features are absent. Still rejected loudly
(``assert_block_supported``): fused-qkvz GatedDeltaNet (Qwen3-Next layout),
grouped ds3-router MoE, HEADWISE attention gating (per-head scalar g_proj),
and non-NeoX RoPE (mRoPE and partial
rotary are accepted for text-only training). Sample packing is not
supported on GatedDeltaNet
models (the recurrence would carry state across packed document boundaries);
train them unpacked.

MoE target semantics: the familiar ``gate_proj``/``up_proj``/``down_proj``
names adapt DENSE MLPs and the always-active shared expert only. The routed
experts are opt-in via the explicit ``expert_gate_proj``/``expert_up_proj``/
``expert_down_proj`` target names (one adapter pair per expert per layer --
on a many-expert model that is a large trainable surface with sparse
per-expert gradients; see ``_MOE_EXPERT_TARGET_ALIASES``). The router gate
and the shared-expert gate stay frozen always.

MTP head (``mtp_model`` / ``mtp_targets``): the Qwen3.5/3.6 multi-token
prediction head -- the generator's self-speculative draft, a separate
component model the trunk's targets never reach -- can be trained next to the
trunk or ALONE on a frozen trunk. Its block(s) get the same differentiable
block forward; its input layer (two pre-fc norms + ``fc`` over
``[embedding | trunk final-norm state]``) and the generator's one-position
shift are reproduced in ``_forward_mtp``; its loss is the same shifted CE
through the shared LM head. ``freeze_trunk`` is the two-stage recipe (fit the
head to a finished trunk adapter). See ``backbone.mtp_layout`` for the
accepted layout.
"""

from __future__ import annotations
from typing import Callable, Iterable, Optional
import contextlib
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone
from . import vision
from . import quant_aware as _qa
from .qlora_linear import EXL3LoRAFunction, EXL3LoRAHadFunction
from .fused_ce import (
    fused_linear_cross_entropy, fused_linear_cross_entropy_vocab_chunked,
    DEFAULT_CHUNK, DEFAULT_VOCAB_CHUNK, IGNORE_INDEX,
)
from .gdn import (
    gdn_delta_rule_reference, gdn_causal_conv1d_silu, gdn_gated_rmsnorm,
    gdn_beta_g,
)
from .short_conv import shortconv_mix


# FlashAttention-2 fast path. exllamav3's own FA2 usage is inference-only
# (@torch.inference_mode kernels), so we use the upstream `flash_attn` package's
# autograd-capable flash_attn_func instead -- it has a real backward. Imported
# lazily and cached: the import pulls a CUDA extension, so it must not run at
# module-import time (CPU tests / no-GPU boxes). Returns None when unavailable,
# in which case the forward falls back to eager attention.
_FLASH_FN = None
_FLASH_TRIED = False


def _flash_attn_func():
    global _FLASH_FN, _FLASH_TRIED
    if not _FLASH_TRIED:
        _FLASH_TRIED = True
        try:
            from flash_attn import flash_attn_func
            _FLASH_FN = flash_attn_func
        except Exception:
            _FLASH_FN = None
    return _FLASH_FN


# Liger Triton kernels (optional). Fused RMSNorm (in-place backward -> less activation
# memory) and SiLU*mul (fewer intermediates). Imported lazily/cached (Triton CUDA
# kernels; must not import at module load on a CPU box). Returns
# (LigerRMSNormFunction, LigerSiLUMulFunction) or None when unavailable. These are
# real autograd Functions, so they slot into the differentiable forward directly.
_LIGER = None
_LIGER_TRIED = False


def _liger_ops():
    global _LIGER, _LIGER_TRIED
    if not _LIGER_TRIED:
        _LIGER_TRIED = True
        try:
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction
            _LIGER = (LigerRMSNormFunction, LigerSiLUMulFunction)
        except Exception:
            _LIGER = None
    return _LIGER


# fla's chunked gated delta rule (flash-linear-attention). The autograd-capable
# Triton kernel behind Qwen3.5/3.6's GatedDeltaNet layers -- the SAME kernel
# exllamav3's own inference prefill dispatches to (modules/gated_delta_net_fn/
# gated_delta_rule.py), so training and serving share numerics. Imported
# lazily/cached (Triton CUDA; must not import at module load on a CPU box).
# When unavailable, GDN layers fall back to the sequential fp32 reference in
# training.gdn -- correct but slow and memory-heavy at long sequence lengths.
_FLA_CHUNK_FN = None
_FLA_TRIED = False


def _fla_chunk_fn():
    global _FLA_CHUNK_FN, _FLA_TRIED
    if not _FLA_TRIED:
        _FLA_TRIED = True
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            _FLA_CHUNK_FN = chunk_gated_delta_rule
        except Exception:
            _FLA_CHUNK_FN = None
    return _FLA_CHUNK_FN


# Big-head (head_dim > 256) layers -- the Gemma global layers -- have no
# memory-efficient kernel in the training forward: FA2 caps at head_dim 256,
# exllamav3's bighead kernel is inference-only, and at head_dim > 256 torch SDPA
# can only pick the math backend (full [b, nq, t, t] score matrix -> the O(t^2)
# long-context OOM). FlexAttention was tried but its Triton kernel exceeds the
# shared-memory limit of a 24GB consumer card at this head dim (~512), and the
# backward needs more still. So under sample packing these layers run SDPA
# PER DOCUMENT (see _block_forward): each document attends only to itself, so the
# materialized scores are bounded by the longest document, not the full packed
# block -- the head-dim-agnostic O(sum t_doc^2) fallback.


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

# Leaf name -> the role we expect it to play in the differentiable forward.
_ROLE_BY_LEAF = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
    "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
}

# GatedDeltaNet leaf -> the target names that select it. Lets the familiar
# Llama-style target list (q_proj/k_proj/v_proj/o_proj) adapt the analogous
# GDN projections on Qwen3.5/3.6 hybrid layers without per-arch target lists:
# the fused in_proj_qkv plays the q/k/v role, out_proj plays o, and the output
# gate in_proj_z rides with v_proj (it is a value-shaped projection; PEFT
# "all-linear" recipes adapt it too). The tiny scalar-per-head b/a projections
# are only adapted when named explicitly.
_GDN_TARGET_ALIASES = {
    "in_proj_qkv": ("in_proj_qkv", "qkv_proj", "q_proj", "k_proj", "v_proj"),
    "in_proj_z": ("in_proj_z", "z_proj", "v_proj"),
    "in_proj_b": ("in_proj_b", "b_proj"),
    "in_proj_a": ("in_proj_a", "a_proj"),
    "out_proj": ("out_proj", "o_proj"),
}

# ShortConv leaf -> the target names that select it (LFM2 / LFM2-MoE conv
# layers). Same idea as the GDN aliases: the fused in_proj ([hidden, 3*hidden]
# emitting b|c|x) plays the q/k/v role, out_proj plays o, so the familiar
# Llama-style target list adapts the conv mixer too, and LFM2's own
# checkpoint names (in_proj/out_proj) select them directly. conv_in_proj /
# conv_out_proj pick the conv layers alone (out_proj also names the
# attention output projection on LFM2's self_attn layers, which the plain
# attention path adapts under o_proj).
_SHORTCONV_TARGET_ALIASES = {
    "in_proj": ("in_proj", "conv_in_proj", "q_proj", "k_proj", "v_proj"),
    "out_proj": ("out_proj", "conv_out_proj", "o_proj"),
}

# Routed-expert leaf -> the target names that select it. The routed experts
# deliberately do NOT alias the plain gate_proj/up_proj/down_proj names: on a
# many-expert model (Qwen3.5-MoE class) that would silently attach one adapter
# pair per expert per layer -- hundreds of thousands of tiny adapters with
# sparse gradients (each expert sees only its routed tokens) -- which is
# exactly the trap the mainstream MoE-LoRA recipes avoid by training
# attention + the dense/shared paths only. Adapting the routed experts is
# opt-in via these explicit names. The router gate and shared-expert gate are
# never adapted (top-k selection is discontinuous; standard recipes freeze
# the router).
_MOE_EXPERT_TARGET_ALIASES = {
    "gate_proj": ("expert_gate_proj",),
    "up_proj": ("expert_up_proj",),
    "down_proj": ("expert_down_proj",),
}


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    # Matches exllamav3.util.rope._rotate_half_neox: split the head dim in half.
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class DiffLinear(nn.Module):
    """
    Differentiable linear over a frozen native ``Linear`` module.

    The frozen effective weight is reconstructed on every call from the EXL3
    trellis (or read directly for an fp16 inner layer) via ``get_weight_tensor``
    and treated as a constant -- no gradient ever flows into the quantized base.
    Optionally carries trainable LoRA ``a``/``b`` (fp32 master weights); when
    absent the layer is a pure frozen projection. Either way the forward/backward
    runs through the gradchecked :class:`EXL3LoRAFunction`, which recomputes the
    weight in the backward pass instead of stashing it (activation-memory win).

    Shapes: ``a`` is ``[in, r]``, ``b`` is ``[r, out]`` (``y = x @ W + s·x@a@b``).

    ``adapter_enabled`` (default True) gates the whole trainable surface at
    call time: when False the forward is the PURE frozen base projection --
    no LoRA term AND no PiSSA offset (the offset exists only to cancel the
    pissa-initialized adapter; dropping them together reproduces the exact
    base weight ``W_q``). This is how a preference-training reference model
    (DPO/KTO) is realized without a second model copy -- see
    ``NativeLlamaQLoRA.adapters_disabled``.
    """

    # Class default so headless instances (tests build nets via __new__) and
    # pre-toggle checkpoints behave as before.
    adapter_enabled = True

    # Class default for the same reason: a headless instance has no wrapped
    # linear to take trellis parts from, so it uses the legacy closure path.
    _trellis_parts = None
    _had = None

    # Quantization-aware training mode (training.quant_aware): "" = off (the
    # default; eval/validate paths and pre-feature checkpoints are exact).
    # configure_quant_aware() sets these per adapted wrapper; nothing here is
    # persisted -- it is run configuration, reapplied by the trainer.
    qa_mode = ""
    qa_sigma = None       # [out] fp32 per-channel error scale (incl. scale knob)
    qa_state = None       # shared {"tick", "seed"} dict owned by the net
    qa_layer_id = 0       # stable per-layer seed offset

    # LoRA-branch dropout (PEFT's lora_dropout). Class default so headless
    # instances and pre-feature checkpoints behave exactly as before.
    lora_drop = None

    def __init__(
        self,
        linear: nn.Module,
        r: int = 0,
        alpha: float = 16.0,
        use_rslora: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert backbone.is_loaded(linear), \
            "native Linear must be loaded (have .inner) before wrapping"
        self.linear = linear                 # frozen; holds trellis / fp16 weight
        self.key = linear.key
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.compute_dtype = compute_dtype
        self.r = r
        self.lora_alpha = float(alpha)
        self.use_rslora = use_rslora

        if r > 0:
            denom = (r ** 0.5) if use_rslora else r
            self.scale = float(alpha) / float(denom)
            dev = backbone.linear_device(self.linear)
            # B starts at zero so the adapter is a no-op at init (training begins
            # from the exact base model); A uses the PEFT kaiming init. An SVD init
            # (lora_init.apply_init_lora) may overwrite both afterwards.
            self.lora_a = nn.Parameter(torch.empty(self.in_features, r, dtype=torch.float32, device=dev))
            self.lora_b = nn.Parameter(torch.zeros(r, self.out_features, dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
            # PEFT-style lora_dropout: dropout on the LoRA branch's input only,
            # the frozen base path always sees the undropped x. Train-time only
            # (nn.Dropout is identity in eval; forward also gates on training).
            self.lora_drop = nn.Dropout(dropout) if dropout > 0.0 else None
        else:
            self.scale = 1.0
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

        # Frozen PiSSA offset (see lora_init): when set, the frozen weight is
        # served as W_q - scale·(a0@b0), realizing the PiSSA residual base
        # without touching the trellis. Not persisted in state_dict -- the
        # adapter save/load path round-trips it via pissa_init.safetensors.
        # The on-device copies live in the COMPUTE dtype (the closure casts to
        # it on every reconstruction regardless, so this is bit-identical for
        # training while halving the offsets' VRAM); the exact fp32 factors are
        # kept on CPU (init_a0_master/init_b0_master) for the sidecar, the
        # converted export and apply_to_native, where fp16/bf16-level
        # cancellation against the fp32 adapter would poison the tiny delta.
        self.register_buffer("init_a0", None, persistent=False)
        self.register_buffer("init_b0", None, persistent=False)
        self.init_a0_master: Optional[torch.Tensor] = None
        self.init_b0_master: Optional[torch.Tensor] = None

        # The wrapped native Linear (an exllamav3 ABC Module, not an nn.Module)
        # holds its weights as plain tensors / buffers, never nn.Parameters, so
        # there is nothing to freeze: no gradient can ever reach the base.

        # Activation-side-transform pieces (audit A1): (inner_fn, suh, svh)
        # for a standard trellis linear, None for fp16/nonstandard inners.
        # When present (and quant-aware is off / mode isn't "legacy") the
        # forward takes EXL3LoRAHadFunction, which skips the four full-weight
        # transform passes + cast of get_weight_tensor on every dequant.
        # Everything lives in the compute dtype (S35 Tier-2 item 1): the
        # reconstruct kernel emits it directly and suh/svh/had are pre-cast
        # once, so the Function's per-call .to() casts are all no-ops.
        self._trellis_parts = backbone.frozen_trellis_parts(linear, compute_dtype)
        self._had = None
        if self._trellis_parts is not None:
            suh = self._trellis_parts[1]
            self._had = backbone.hadamard_128(suh.device, suh.dtype)

    @torch.no_grad()
    def set_init_offset(self, a0: torch.Tensor, b0: torch.Tensor) -> None:
        """Install the frozen PiSSA offset factors (fp32, adapter shapes)."""
        assert self.r > 0
        dev = backbone.linear_device(self.linear)
        assert a0.shape == self.lora_a.shape and b0.shape == self.lora_b.shape
        self.init_a0_master = a0.detach().to("cpu", torch.float32).contiguous()
        self.init_b0_master = b0.detach().to("cpu", torch.float32).contiguous()
        self.init_a0 = a0.detach().to(dev, self.compute_dtype).contiguous()
        self.init_b0 = b0.detach().to(dev, self.compute_dtype).contiguous()

    def _weight_closure(self):
        wfn = backbone.frozen_weight_closure(self.linear, self.compute_dtype)
        if self.init_a0 is None:
            return wfn
        # PiSSA residual base: subtract the frozen principal component on every
        # reconstruction (forward AND the backward recompute -- the closure is
        # what EXL3LoRAFunction re-invokes, so grads see the residual too). The
        # product runs in the compute dtype: the result is stored at that
        # precision regardless, and the rank-r matmul is noise next to the
        # trellis dequant inside wfn (which --profile-dequant still times).
        # Fused out-of-place addmm: never materializes the [in, out] a0@b0
        # product as a separate temporary (in-place on w is unsafe -- for an
        # fp16 inner layer wfn can return the stored weight itself).
        a0, b0, s = self.init_a0, self.init_b0, self.scale

        def residual_fn():
            w = wfn()
            if a0.dtype != w.dtype:      # off-dtype path (e.g. float64 tests)
                return w - s * (a0.to(w.dtype) @ b0.to(w.dtype))
            return torch.addmm(w, a0, b0, alpha=-s)

        return residual_fn

    def _weight_closure_qa(self):
        """The frozen-weight closure with the quantization-aware perturbation
        (training.quant_aware) composed on top when a mode is active; the
        plain closure otherwise (off / eval -- exact weights always)."""
        return _qa.wrap_weight_closure(self, self._weight_closure())

    def _qa_active(self) -> bool:
        return bool(self.qa_mode) and self.qa_sigma is not None and self.training

    def _lora_drop_term(self, xc: torch.Tensor) -> torch.Tensor:
        """The LoRA branch on the DROPPED input, through plain autograd
        (grads reach the fp32 masters through the .to casts, same as the
        fused Functions' cast-back)."""
        a = self.lora_a.to(xc.dtype)
        b = self.lora_b.to(xc.dtype)
        return self.scale * ((self.lora_drop(xc) @ a) @ b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.to(self.compute_dtype)

        # lora_dropout: the fused Functions share ONE input between the frozen
        # base matmul and the LoRA branch, but PEFT dropout must hit only the
        # branch input. So with dropout live, run the fused op adapter-free
        # (keeping bias and the pissa offset -- those belong to the effective
        # frozen base) and add the low-rank term on the dropped input outside.
        # Taken only while training with p > 0: eval and p=0 keep the fused
        # path, so parity/gradcheck numerics are untouched. Incompatible with
        # quant_aware=ste (rejected at configure time): ste's closure relies
        # on the un-dropped branch cancelling its -delta term.
        drop_lora = (self.lora_drop is not None and self.training
                     and self.adapter_enabled and self.lora_a is not None)

        # Fast path (audit A1): trellis inner + activation-side transforms.
        # Quant-aware runs need the full-weight closure to perturb, so they
        # (and --dequant-mode legacy) fall through to the original path.
        parts = self._trellis_parts
        if parts is not None and not self._qa_active() \
                and backbone.dequant_mode() == "fast":
            inner_fn, suh, svh = parts
            had = self._had if self._had is not None \
                else backbone.hadamard_128(suh.device, suh.dtype)
            bias = backbone.frozen_bias(self.linear, self.compute_dtype)
            if not self.adapter_enabled:
                # Reference view: pure frozen base -- no LoRA, no pissa offset.
                return EXL3LoRAHadFunction.apply(
                    xc, None, None, bias, 1.0,
                    inner_fn, suh, svh, had, None, None, 1.0)
            if drop_lora:
                y = EXL3LoRAHadFunction.apply(
                    xc, None, None, bias, 1.0,
                    inner_fn, suh, svh, had,
                    self.init_a0, self.init_b0, self.scale)
                return y + self._lora_drop_term(xc)
            return EXL3LoRAHadFunction.apply(
                xc, self.lora_a, self.lora_b, bias, self.scale,
                inner_fn, suh, svh, had,
                # PiSSA residual base as a low-rank activation term instead of
                # the per-reconstruction full-weight addmm of _weight_closure.
                self.init_a0, self.init_b0, self.scale)

        if not self.adapter_enabled:
            # Reference view: pure frozen base -- no LoRA term, no pissa offset
            # (backbone.frozen_weight_closure directly, bypassing the residual).
            return EXL3LoRAFunction.apply(
                xc, None, None,
                backbone.frozen_bias(self.linear, self.compute_dtype),
                1.0,
                backbone.frozen_weight_closure(self.linear, self.compute_dtype),
            )
        if drop_lora:
            # Closure includes the pissa residual and any quant-aware NOISE
            # perturbation; only the trainable low-rank term moves outside.
            y = EXL3LoRAFunction.apply(
                xc, None, None,
                backbone.frozen_bias(self.linear, self.compute_dtype),
                1.0,
                self._weight_closure_qa(),
            )
            return y + self._lora_drop_term(xc)
        return EXL3LoRAFunction.apply(
            xc, self.lora_a, self.lora_b,
            backbone.frozen_bias(self.linear, self.compute_dtype),
            self.scale,
            self._weight_closure_qa(),
        )

    def extra_repr(self) -> str:
        return (f"key={self.key}, in={self.in_features}, out={self.out_features}, "
                f"r={self.r}, compute_dtype={self.compute_dtype}"
                f"{', pissa_offset' if self.init_a0 is not None else ''}"
                f"{f', dropout={self.lora_drop.p}' if self.lora_drop is not None else ''}")


class NativeLlamaQLoRA(nn.Module):
    """
    Differentiable Llama decoder built on a loaded exllamav3 ``Model``.

    Construct from a model already loaded with native exllamav3 (the path that
    forwards correctly on the quantized weights), then train LoRA adapters with
    a plain PyTorch loop -- no HuggingFace Transformers anywhere.

    Example::

        from exllamav3 import Config, Model
        from exllamav3.training.native_llama import NativeLlamaQLoRA

        config = Config.from_directory(model_dir)
        model = Model.from_config(config)
        model.load(device="cuda:0")
        net = NativeLlamaQLoRA(model, r=16, alpha=32,
                               target_modules=["q_proj", "v_proj"])
        loss = net.compute_loss(input_ids, labels)   # fused-CE, frozen head
        loss.backward()
        ...
        net.save_adapter("out/pirate")               # PEFT format
    """

    # Class-level defaults for the optional feature flags so the helper methods are
    # safe on a headless instance built via __new__ (the CPU tests construct nets that
    # way to exercise _norm/_block_forward without a real model). Instance __init__
    # overrides these. NOTE: only non-parameter flags get class defaults -- a class
    # attribute that shadows a registered nn.Parameter (embed_lora_a, ...) would hide
    # it from nn.Module's __getattr__, so those are read via getattr() in the accessor.
    use_liger = False
    offload_activations = False
    offload_mode = "async"
    _offloader = None
    init_lora = "default"
    lora_dropout = 0.0
    # Quant-aware training (training.quant_aware): shared {"tick", "seed"}
    # state when a mode is enabled via set_quant_aware(), else None (exact
    # weights; the default -- eval/validate and old checkpoints unaffected).
    _qa_state = None
    quant_aware = "none"
    quant_aware_scale = 1.0
    # When True (via the adapters_disabled() context manager) the forward runs
    # as the PURE frozen base model: per-linear LoRA + pissa offsets off, the
    # trainable/LoRA embedding and LM head ignored. Used for the reference-model
    # logprobs in preference training (DPO/KTO) -- the PEFT "disable adapter"
    # trick, so no second model copy is ever loaded.
    _adapters_off = False
    # EBFT feature taps (collect_hidden): (set[int] block idxs, [b,P] positions
    # or None) while active, else None. Class default so older pickles/subclasses
    # without the attribute still run the forward untapped.
    _collect_spec = None

    def __init__(
        self,
        model: nn.Module,
        r: int = 16,
        alpha: float = 16.0,
        target_modules: Optional[Iterable[str]] = None,
        use_rslora: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = True,
        train_embeddings: bool = False,
        train_head: bool = False,
        attn_impl: str = "auto",
        head_vocab_chunk: int = 0,
        modules_to_save_dtype: torch.dtype = torch.float32,
        lora_embed: bool = False,
        lora_head: bool = False,
        offload_activations: bool = False,
        offload_mode: str = "async",
        use_liger: bool = False,
        expert_r: Optional[int] = None,
        lora_dropout: float = 0.0,
        mtp_model: Optional[nn.Module] = None,
        mtp_targets: Optional[Iterable[str]] = None,
        mtp_loss_weight: float = 1.0,
        freeze_trunk: bool = False,
    ):
        """
        ``mtp_model`` (optional): the model's loaded MTP (multi-token
        prediction) head, ``Model.from_config(config, component="mtp")``, to
        train NEXT TO the trunk -- see :meth:`_forward_mtp`. ``mtp_targets``
        are the leaf names adapted inside the head (same vocabulary as
        ``target_modules`` plus ``fc``, the head's input projection; None =
        the default list + ``fc``); the trunk's ``target_modules`` never reach
        into the head and vice versa, so ``target_modules=[]`` with
        ``mtp_targets`` set trains the MTP tensors ALONE. ``mtp_loss_weight``
        scales the head's next-token loss when it is summed with the trunk's.
        ``freeze_trunk`` freezes every trunk-side trainable (per-linear LoRA,
        embed/head adapters) after construction -- the two-stage recipe: load
        a finished trunk adapter with ``load_adapter`` and fit the head to the
        tuned trunk it will draft for, exporting both in one adapter dir.
        """
        super().__init__()
        # Offload the (grad-checkpointed) saved activations to CPU, and/or route
        # RMSNorm + SwiGLU through Liger Triton kernels. Both are CUDA + fp16/bf16
        # memory levers; eager fp32 / CPU / the validate path ignore them (so the
        # correctness gate is unaffected). Liger changes numerics slightly -- run
        # qlora_validate_native.py with --use-liger before trusting a Liger run.
        # offload_mode: "async" (S36: double-buffered side-stream copies,
        # value-exact, overlaps compute) or "sync" (torch save_on_cpu, the S9
        # behavior, kept for A/B and as the retain_graph-safe fallback).
        self.offload_activations = bool(offload_activations)
        assert offload_mode in ("async", "sync"), f"bad offload_mode {offload_mode!r}"
        self.offload_mode = offload_mode
        self._offloader = None  # lazy AsyncActivationOffload (holds pinned pool)
        self.use_liger = bool(use_liger)
        if self.use_liger and _liger_ops() is None:
            raise SystemExit(
                "use_liger=True but liger_kernel is not importable. Install it "
                "(pip install liger-kernel) or drop --use-liger.")
        self.model = model
        self.compute_dtype = compute_dtype
        self.gradient_checkpointing = gradient_checkpointing
        # Attention implementation. "auto": use FlashAttention-2 when the
        # flash_attn package imports AND the run is on CUDA in fp16/bf16 (decided
        # per-forward), else eager. "flash": require it. "eager": never (the
        # reference path; fp32 / CPU / gradcheck always land here).
        self.attn_impl = attn_impl
        _fn = _flash_attn_func()
        if attn_impl == "flash":
            assert _fn is not None, \
                "attn_impl='flash' but the flash_attn package is not importable"
            self._flash_ok = True
        elif attn_impl == "eager":
            self._flash_ok = False
        else:
            self._flash_ok = _fn is not None
        targets_defaulted = target_modules is None
        targets = set(target_modules) if target_modules is not None else set(DEFAULT_TARGET_MODULES)
        self.target_modules = sorted(targets)
        self.r = r
        self.lora_alpha = float(alpha)
        self.use_rslora = use_rslora
        # Rank for ROUTED-expert adapters (expert_gate/up/down_proj targets)
        # when it should differ from the net-wide r. On a many-expert model the
        # per-expert adapters dominate the trainable parameter count, so PEFT's
        # MoE recipe divides the rank by the expert count (rank_pattern); this
        # is that knob. None = same rank as everything else.
        self.expert_r = int(expert_r) if expert_r is not None else None
        # PEFT-style lora_dropout on every per-linear adapter's branch input
        # (incl. routed experts). The embed/head adapters (lora_embed/lora_head
        # and the fully-trained modules_to_save) are NOT dropped -- they're
        # separate whole-module surfaces, not low-rank branches on a linear.
        self.lora_dropout = float(lora_dropout)
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError(f"lora_dropout must be in [0, 1), got {lora_dropout}")

        # All reach into exllamav3's internal module layout goes through backbone:
        # embedding first, final RMSNorm + LM head last, transformer blocks between.
        self.embed, blocks, self.final_norm, self.lm_head = backbone.split_decoder(model)
        # Optional norm on the token embeddings, between the Embedding and the
        # first block (MuseGlimmer's unweighted embed_norm). None on every arch
        # that doesn't have one -> no-op.
        self.embed_norm_spec = backbone.norm_spec(backbone.embed_norm(model))

        self.blocks = nn.ModuleList()
        self._block_meta = []
        self._block_devices = []   # per-block device (differs under layer-autosplit)
        wrappers: list[DiffLinear] = []
        satisfied_targets: set[str] = set()   # requested names that matched a linear
        self.has_gdn = False
        self.has_shortconv = False
        self.has_moe = False

        def make_wrap(targets, satisfied_targets):
            def wrap(linear, leaf, aliases=None, r_override=None):
                # `aliases` (GDN / MoE-expert layers) lets several requested
                # target names select one physical linear (see
                # _GDN_TARGET_ALIASES / _MOE_EXPERT_TARGET_ALIASES; an empty
                # tuple = never targetable); plain layers match on their own
                # leaf name only. `r_override` (MoE routed experts) trains the
                # hit at a different rank than the net-wide r -- PEFT's
                # rank_pattern recipe for many-expert models.
                names = aliases if aliases is not None else (leaf,)
                hit = targets.intersection(names)
                satisfied_targets.update(hit)
                r_eff = (r_override if r_override is not None else r) if hit else 0
                w = DiffLinear(
                    linear,
                    r=r_eff,
                    alpha=alpha,
                    use_rslora=use_rslora,
                    compute_dtype=compute_dtype,
                    dropout=lora_dropout,
                )
                wrappers.append(w)
                return w
            return wrap

        # The decoder blocks to build entries for: the trunk's, then (when an
        # MTP head is attached) the head's own block(s), which are ordinary
        # TransformerBlocks and get the SAME entry/forward as a trunk block --
        # only their target set and destination lists differ.
        wrap = make_wrap(targets, satisfied_targets)
        plan = [(blk, wrap, self.blocks, self._block_meta, self._block_devices)
                for blk in blocks]
        self.mtp = None
        self.mtp_target_modules: list[str] = []
        self.mtp_loss_weight = float(mtp_loss_weight)
        self._mtp_wrappers: list[DiffLinear] = []
        mtp_targets_defaulted = mtp_targets is None
        if mtp_model is not None:
            mtp_in, mtp_blocks, mtp_final_norm = backbone.mtp_layout(mtp_model)
            mtargets = (set(mtp_targets) if mtp_targets is not None
                        else set(DEFAULT_TARGET_MODULES) | {"fc"})
            mtp_satisfied: set[str] = set()
            mwrap = make_wrap(mtargets, mtp_satisfied)
            self.mtp = nn.Module()
            self.mtp.blocks = nn.ModuleList()
            self._mtp_block_meta: list[dict] = []
            self._mtp_block_devices: list = []
            plan += [(blk, mwrap, self.mtp.blocks, self._mtp_block_meta,
                      self._mtp_block_devices) for blk in mtp_blocks]

        has_mrope = False
        n_trunk_wrappers = 0
        for blk, wrap, out_blocks, out_meta, out_devs in plan:
            backbone.assert_block_supported(blk)
            has_mrope = has_mrope or backbone.attn_has_mrope(blk)
            meta = backbone.block_metadata(blk)
            is_moe = meta.get("mlp_kind", "dense") == "moe"
            if is_moe:
                # MoE block: entry.gates/ups/downs hold the SHARED expert (a
                # dense, always-active GatedMLP -- absent on Qwen3-MoE), so it
                # rides the plain gate/up/down_proj target names like any dense
                # MLP; the routed experts get their own lists below.
                shared = backbone.moe_shared_projections(blk)
                gate_lins, up_lins, down_lins = shared if shared is not None else ([], [], [])
            else:
                gate_lins, up_lins, down_lins = backbone.mlp_projections(blk)
            attn_norm, mlp_norm = backbone.block_norms(blk)
            attn_post, mlp_post = backbone.block_post_norms(blk)        # Gemma sandwich

            # MLP may be sliced across intermediate dim for very wide models;
            # wrap every slice and sum the down-projections (mirrors GatedMLP).
            gates = [wrap(g, "gate_proj") for g in gate_lins]
            ups = [wrap(u, "up_proj") for u in up_lins]
            downs = [wrap(d, "down_proj") for d in down_lins]

            entry = nn.Module()
            entry.is_moe = is_moe
            if is_moe:
                self.has_moe = True
                eg, eu, ed = backbone.moe_expert_projections(blk)
                # One wrapper per routed expert; adapted only via the explicit
                # expert_* target names (see _MOE_EXPERT_TARGET_ALIASES).
                entry.expert_gates = nn.ModuleList(
                    [wrap(g, "gate_proj", _MOE_EXPERT_TARGET_ALIASES["gate_proj"],
                          r_override=self.expert_r) for g in eg])
                entry.expert_ups = nn.ModuleList(
                    [wrap(u, "up_proj", _MOE_EXPERT_TARGET_ALIASES["up_proj"],
                          r_override=self.expert_r) for u in eu])
                entry.expert_downs = nn.ModuleList(
                    [wrap(d, "down_proj", _MOE_EXPERT_TARGET_ALIASES["down_proj"],
                          r_override=self.expert_r) for d in ed])
                # Router + shared-expert gate: frozen projections (aliases=() so
                # no target name can select them), still DiffLinears so the
                # routing weights stay differentiable w.r.t. the hidden state.
                entry.router = wrap(backbone.moe_router_linear(blk), "router_gate", ())
                sh_gate = backbone.moe_shared_gate_linear(blk)
                entry.shared_gate = wrap(sh_gate, "shared_gate", ()) if sh_gate is not None else None
                # Gemma4-layout extra norms (None on Qwen-family MoE): router
                # pre-norm (constant_scale = hidden**-0.5 read from the
                # module), routed pre/post norms, shared-expert post norm.
                rp, ep, po, sp = backbone.moe_extra_norms(blk)
                entry.router_pre_spec = backbone.norm_spec(rp)
                entry.routed_pre_spec = backbone.norm_spec(ep)
                entry.routed_post_spec = backbone.norm_spec(po)
                entry.shared_post_spec = backbone.norm_spec(sp)
            # Norm specs (plain dicts read from the modules; see backbone.norm_spec),
            # so the block forward reproduces each arch's exact RMSNorm without
            # reaching into module internals. None means "no such norm here".
            entry.attn_norm_spec = backbone.norm_spec(attn_norm)
            entry.mlp_norm_spec = backbone.norm_spec(mlp_norm)
            entry.attn_post_spec = backbone.norm_spec(attn_post)
            entry.mlp_post_spec = backbone.norm_spec(mlp_post)
            if meta["kind"] == "gdn":
                # GatedDeltaNet layer (Qwen3.5/3.6 linear attention). The five
                # split projections are all native Linears, so LoRA / pissa /
                # quant-aware compose exactly as on softmax layers.
                self.has_gdn = True
                qkv_l, z_l, b_l, a_l, o_l = backbone.gdn_projections(blk)
                entry.gdn_norm_spec = backbone.gdn_norm_spec(blk)
                entry.qkv_proj = wrap(qkv_l, "in_proj_qkv",
                                      _GDN_TARGET_ALIASES["in_proj_qkv"])
                entry.z_proj = wrap(z_l, "in_proj_z",
                                    _GDN_TARGET_ALIASES["in_proj_z"])
                entry.b_proj = wrap(b_l, "in_proj_b",
                                    _GDN_TARGET_ALIASES["in_proj_b"])
                entry.a_proj = wrap(a_l, "in_proj_a",
                                    _GDN_TARGET_ALIASES["in_proj_a"])
                entry.o_proj = wrap(o_l, "out_proj",
                                    _GDN_TARGET_ALIASES["out_proj"])
            elif meta["kind"] == "shortconv":
                # ShortConv layer (LFM2 / LFM2-MoE gated causal conv). Two
                # native Linears (in_proj / out_proj), so LoRA / pissa /
                # quant-aware compose exactly as on softmax layers; the
                # depthwise conv tensor itself stays frozen (meta).
                self.has_shortconv = True
                in_l, out_l = backbone.short_conv_projections(blk)
                entry.in_proj = wrap(in_l, "in_proj",
                                     _SHORTCONV_TARGET_ALIASES["in_proj"])
                entry.out_proj = wrap(out_l, "out_proj",
                                      _SHORTCONV_TARGET_ALIASES["out_proj"])
            else:
                q_proj, k_proj, v_proj, o_proj = backbone.attn_projections(blk)
                q_norm, k_norm, v_norm = backbone.attn_qkv_norms(blk)   # Qwen3 / Gemma
                entry.q_norm_spec = backbone.norm_spec(q_norm)
                entry.k_norm_spec = backbone.norm_spec(k_norm)
                entry.v_norm_spec = backbone.norm_spec(v_norm)
                entry.q_proj = wrap(q_proj, "q_proj")
                entry.k_proj = wrap(k_proj, "k_proj")
                # Some Gemma layers reuse K as V (no v_proj); the block forward then
                # takes V from the raw K projection.
                entry.v_proj = wrap(v_proj, "v_proj") if v_proj is not None else None
                # The attention output projection answers to o_proj and, when
                # the checkpoint keys it differently (LFM2's
                # self_attn.out_proj), to its own leaf name as well -- so a
                # target list written in the model's own names works.
                o_leaf = str(getattr(o_proj, "key", "o_proj")).split(".")[-1]
                entry.o_proj = wrap(o_proj, "o_proj",
                                    ("o_proj", o_leaf) if o_leaf != "o_proj" else None)
                # AFMoE full-width attention output gate (a separate linear the
                # checkpoint keys as self_attn.gate_proj). It answers to the
                # plain gate_proj target -- PEFT's suffix matching would adapt
                # it under that name too -- and to attn_gate_proj for
                # adapting it alone.
                g_lin = backbone.attn_gate_linear(blk)
                entry.g_proj = (wrap(g_lin, "gate_proj",
                                     ("gate_proj", "attn_gate_proj"))
                                if g_lin is not None else None)
            entry.gates = nn.ModuleList(gates)
            entry.ups = nn.ModuleList(ups)
            entry.downs = nn.ModuleList(downs)
            out_blocks.append(entry)
            out_meta.append(meta)
            out_devs.append(backbone.block_device(blk))
            if out_blocks is self.blocks:
                n_trunk_wrappers = len(wrappers)   # plan is trunk-then-MTP

        if self.mtp is not None:
            # The head's input layer: two pre-fc norms + the [2d -> d] fc over
            # [embedding | trunk state] (backbone.mtp_layout), then its final
            # norm. The block(s) were built by the loop above. Everything
            # lives on one device: the head is loaded whole, next to the
            # trunk's output device where the final-norm state it consumes
            # is produced.
            nh, ne, fc = backbone.mtp_input_parts(mtp_in)
            self.mtp.fc = mwrap(fc, "fc")
            self.mtp.norm_hidden_spec = backbone.norm_spec(nh)
            self.mtp.norm_embed_spec = backbone.norm_spec(ne)
            self.mtp.final_norm_spec = backbone.norm_spec(mtp_final_norm)
            devs = set(map(str, self._mtp_block_devices))
            devs.add(str(backbone.linear_device(fc)))
            assert len(devs) == 1, \
                f"MTP head modules span devices {sorted(devs)}; load it on one device"
            self._mtp_device = self._mtp_block_devices[0]
            self._mtp_wrappers = wrappers[n_trunk_wrappers:]
            missing = mtargets - mtp_satisfied
            if missing and mtp_targets_defaulted:
                print(f" -- note: default MTP target(s) {sorted(missing)} matched "
                      f"no linear in the MTP head; training the rest.")
                mtargets -= missing
                missing = set()
            if missing:
                raise ValueError(
                    f"mtp_targets {sorted(missing)} matched no linear in the MTP "
                    f"head (available leaves: "
                    f"{sorted({w.key.split('.')[-1] for w in self._mtp_wrappers})})")
            self.mtp_target_modules = sorted(mtargets)

        self._wrappers = wrappers
        if self.has_moe:
            expert_targets = targets.intersection(
                n for names in _MOE_EXPERT_TARGET_ALIASES.values() for n in names)
            if not expert_targets:
                print(" -- note: MoE model; routed experts stay FROZEN (plain "
                      "gate/up/down_proj targets adapt the dense/shared-expert "
                      "paths only). Opt in with --targets ... expert_gate_proj "
                      "expert_up_proj expert_down_proj (consider a small "
                      "--expert-r; one adapter pair per expert per layer).")
            elif self.expert_r is None:
                n_exp = next(m["num_experts"] for m in self._block_meta
                             if m.get("mlp_kind") == "moe")
                print(f" -- note: adapting routed experts at rank {r} "
                      f"({n_exp} experts/layer). If the adapter count is too "
                      f"large, set --expert-r (PEFT's MoE recipe uses "
                      f"~r/num_experts, min 1).")
        # Vision-language plumbing of the TEXT tower (the vision tower itself is
        # a separate frozen component; see training.vision). has_mrope: the
        # attention layers rotate 3-D positions when image tokens are present
        # (text-only input reduces exactly to 1-D RoPE). _deepstack: which
        # decoder blocks are followed by a DeepstackEmbed (Qwen3-VL /
        # Qwen3.5-VL add vision intermediate features at the image positions
        # there). _bidir_mm: image tokens attend bidirectionally within their
        # span (Gemma4). All three are no-ops without an image batch.
        self.has_mrope = has_mrope
        self._deepstack = backbone.deepstack_layout(model)
        self._bidir_mm = backbone.uses_noncausal_mm_spans(model)
        if has_mrope:
            print(" -- note: mRoPE detected (Qwen-VL text tower). Text-only "
                  "batches reduce exactly to 1D NeoX RoPE; image batches "
                  "(--vision) rotate the 3D [t, h, w] positions per band.")
        if self.has_gdn and _fla_chunk_fn() is None:
            print(" -- note: GatedDeltaNet layers present but the fla package "
                  "(flash-linear-attention) is not importable; the delta rule "
                  "will run the sequential torch reference -- correct but slow "
                  "and memory-heavy at long seq-len. pip install "
                  "flash-linear-attention for the chunked Triton kernel.")
        self.final_norm_spec = backbone.norm_spec(self.final_norm)
        # Final-logit tanh softcapping (Gemma2; 0 = none). Applied by logits(),
        # by the materialized supervised-position loss, and by both fused-CE
        # heads (softcap arg -- the cap is elementwise, so it chunks cleanly).
        self.final_softcap = backbone.head_softcap(self.lm_head)
        # Logit pre-scale (MuseGlimmer's output_multiplier; 1.0 on every other
        # arch). Applied by folding it into the head INPUT at the end of
        # forward() -- see the note there.
        self.head_pre_scale = backbone.head_pre_scale(self.lm_head)
        # Output device = where the final norm + LM head live. Under a single
        # load this is the one decoder device; under layer-autosplit it is the
        # last device in the split (modules[-1].device). The embedding is loaded
        # on CPU (prefer_cpu) regardless.
        self.device = self.final_norm.weight.device
        self._head_device = backbone.linear_device(self.lm_head)

        # Optional chunked-vocab head loss: reconstruct + matmul the head in
        # vocab-column tiles so the full [hidden, vocab] weight (and its fp32 upcast
        # + [tokens, vocab] logits) is never held at once -- the dominant memory
        # spike on the output device for big vocabs (e.g. Gemma's 262k head). 0 =
        # off (the single-shot fused head). Resolved to None if the head can't slice.
        self.head_vocab_chunk = int(head_vocab_chunk)
        self._head_slice = None
        if self.head_vocab_chunk > 0:
            self._head_slice = backbone.head_weight_slice_closure(self.lm_head)
        # A *trainable* head (train_head/lora_head) forces compute_loss down the
        # materialized supervised-position path (the fused heads are frozen-head
        # only: they emit no weight/LoRA gradient), so --head-vocab-chunk is
        # silently a no-op there. Warn once: on a big-vocab model the [tokens,
        # vocab] spike the chunk flag is meant to avoid is NOT avoided. (Final-
        # logit softcapping alone no longer forces this path -- both fused heads
        # apply the tanh cap in-tile since Session 12.)
        if self.head_vocab_chunk > 0 and (train_head or lora_head):
            print(" -- note: --head-vocab-chunk is ignored when the LM head trains "
                  "(--train-head / --lora-head); the chunked-vocab CE is frozen-head "
                  "only, so compute_loss uses the materialized supervised-position "
                  "head loss. Watch head memory on a big-vocab base.")

        # Sanity: every requested target name actually matched something
        # (directly by leaf name, or via a GDN / MoE alias). When the DEFAULT
        # list was used (no explicit target_modules), unmatched names are
        # dropped with a note instead -- e.g. a pure-MoE model with no shared
        # expert has no plain gate/up/down_proj, and the default should still
        # train its attention without the user hand-picking targets.
        missing = targets - satisfied_targets
        if missing and targets_defaulted:
            print(f" -- note: default target(s) {sorted(missing)} matched no "
                  f"linear on this architecture; training the rest.")
            targets -= missing
            self.target_modules = sorted(targets)
            missing = set()
        if missing:
            raise ValueError(
                f"target_modules {sorted(missing)} matched no linear in the model "
                f"(available leaves: {sorted({w.key.split('.')[-1] for w in wrappers})})"
            )

        # --- optional full-trained input/output embeddings (modules_to_save) ---
        # LoRA normally freezes embed_tokens / lm_head; these flags fully train
        # them (a trainable fp32 copy reconstructed once from the frozen base),
        # PEFT-`modules_to_save` style, and save them alongside the adapter.
        #
        # exllamav3 always loads a *separate* embedding and lm_head, even for a
        # tied model (the tied weight is materialized into both), so we train
        # whichever is requested independently -- no shared-parameter special
        # case. (tie_word_embeddings is recorded in the saved config only as a
        # hint for a downstream merge/re-quantize step.)
        self.train_embeddings = bool(train_embeddings)
        self.train_head = bool(train_head)
        # dtype of the trainable embed/head master copies. fp32 by default; bf16 when
        # the optimizer does bf16 stochastic-rounding updates (CPU-offload path), which
        # keeps the master at bf16 without losing small updates (so the GPU holds a 2-
        # byte param instead of 4, and the CPU-offloaded optimizer halves its master).
        self.modules_to_save_dtype = modules_to_save_dtype
        self.tie_word_embeddings = bool(
            getattr(getattr(model, "config", None), "tie_word_embeddings", False))
        self.embed_weight = None    # [vocab, hidden], trainable, or None
        self.head_weight = None     # [hidden, vocab], trainable, or None

        # The base embedding is often loaded on CPU (prefer_cpu); put the trainable
        # copies on the GPU compute devices so training isn't bottlenecked on a CPU
        # optimizer / matmul. First-block device for the input embedding, head
        # device for the output projection (identical under single-device / DDP).
        # Reconstruct in fp32, then cast to the chosen master dtype.
        if self.train_embeddings:
            w = backbone.embed_weight(self.embed).detach().to(torch.float32)   # [V, d]
            self.embed_weight = nn.Parameter(
                w.clone().to(self._block_devices[0]).to(modules_to_save_dtype))
        if self.train_head:
            hw = backbone.head_weight_closure(self.lm_head)().detach().to(torch.float32)  # [d, V]
            self.head_weight = nn.Parameter(
                hw.clone().to(self._head_device).to(modules_to_save_dtype))

        # --- optional LoRA on the embedding / LM head (the low-rank alternative
        # to fully training them above) ---------------------------------------
        # A rank-r *shift* of the whole embedding / head: far cheaper than the full
        # modules_to_save matrices (r*(vocab+hidden) params vs vocab*hidden), trained
        # through ordinary autograd (no custom Function -- correctness rests on the
        # forward formula). Mutually exclusive with the full-train flag for the same
        # module. The frozen base embedding/head stays frozen; only a/b train. fp32
        # masters like the per-linear LoRA. B=0 at init so the adapter is a no-op and
        # training begins from the exact base model.
        self.lora_embed = bool(lora_embed)
        self.lora_head = bool(lora_head)
        if (self.lora_embed or self.lora_head) and r <= 0:
            raise ValueError("lora_embed/lora_head need r > 0")
        if self.lora_embed and self.train_embeddings:
            raise ValueError("lora_embed and train_embeddings both train the embedding; pick one")
        if self.lora_head and self.train_head:
            raise ValueError("lora_head and train_head both train the LM head; pick one")
        denom = (r ** 0.5) if use_rslora else r
        self._module_lora_scale = float(alpha) / float(denom)
        self.embed_lora_a = self.embed_lora_b = None   # a:[vocab,r] (token-indexed), b:[r,hidden]
        self.head_lora_a = self.head_lora_b = None      # a:[hidden,r], b:[r,vocab]
        if self.lora_embed:
            vocab, hid = backbone.embed_weight(self.embed).shape           # [V, d]
            dev = self._block_devices[0]
            self.embed_lora_a = nn.Parameter(torch.empty(vocab, r, dtype=torch.float32, device=dev))
            self.embed_lora_b = nn.Parameter(torch.zeros(r, hid, dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.embed_lora_a, a=5 ** 0.5)
        if self.lora_head:
            hid, vocab = self.lm_head.in_features, self.lm_head.out_features   # [d, V]
            dev = self._head_device
            self.head_lora_a = nn.Parameter(torch.empty(hid, r, dtype=torch.float32, device=dev))
            self.head_lora_b = nn.Parameter(torch.zeros(r, vocab, dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.head_lora_a, a=5 ** 0.5)

        # --- freeze the trunk (MTP-only stage 2) ------------------------------
        # Every trunk-side trainable keeps its VALUE (a resumed trunk adapter
        # still applies in the forward, and save_adapter still exports it) but
        # stops training: requires_grad off, and the parameter accessors below
        # (lora_parameters / param_groups / num_trainable) skip it, so the
        # optimizer, grad clip and the trainable count see only the head.
        # With nothing trainable upstream the trunk forward then runs under
        # no_grad (see forward), which is the whole point: a frozen trunk
        # stores no activations and re-runs no blocks in backward.
        self.freeze_trunk = bool(freeze_trunk)
        if self.freeze_trunk:
            if self.mtp is None:
                raise ValueError("freeze_trunk without an MTP head leaves nothing to train")
            mtp_ids = {id(w) for w in self._mtp_wrappers}
            for w in self._wrappers:
                if w.r > 0 and id(w) not in mtp_ids:
                    w.lora_a.requires_grad_(False)
                    w.lora_b.requires_grad_(False)
            for p in (self.embed_weight, self.head_weight, self.embed_lora_a,
                      self.embed_lora_b, self.head_lora_a, self.head_lora_b):
                if p is not None:
                    p.requires_grad_(False)

    def trunk_trainable(self) -> bool:
        """True when anything OUTSIDE the MTP head can receive a gradient: a
        trunk per-linear adapter, or an embed/head trainable. False = the trunk
        is a pure frozen feature extractor for the head (freeze_trunk, or
        target_modules=[] with mtp_targets set), and forward() runs it under
        no_grad."""
        mtp_ids = {id(w) for w in getattr(self, "_mtp_wrappers", [])}
        for w in self._wrappers:
            if w.r > 0 and id(w) not in mtp_ids and w.lora_a.requires_grad:
                return True
        for p in self.module_lora_parameters() + self.modules_to_save_parameters():
            if p.requires_grad:
                return True
        return False

    # --- forward -----------------------------------------------------------

    def _norm(self, x: torch.Tensor, spec: dict) -> torch.Tensor:
        # Reproduces RMSNorm.forward_torch exactly for any arch: normalize in fp32,
        # scale by constant_scale, then multiply by (weight + constant_bias). spec
        # comes from backbone.norm_spec; weight None = unweighted (Gemma v-norm).
        # Gemma's (1 + weight) convention is just bias = 1.0 read from the module.
        # Normalize in fp32 for stability, then return in the INPUT dtype (bf16 in
        # a training run, fp32 in the validate/reference path) so the residual
        # stream stays in compute_dtype rather than being silently upcast to fp32.
        # Mirrors HF RMSNorm (fp32 internals, cast back to the activation dtype).
        #
        # Liger fast path (opt-in, CUDA): the fused RMSNorm kernel with fused
        # backward cuts activation memory. Only for the 2D/3D norms (attn/mlp/
        # post/final) -- the 4D per-head q/k/v norm falls through to torch. Requires
        # constant_scale == 1.0 (Liger has no scale param). casting_mode="gemma" =
        # full-fp32 normalize, matching this module's fp32-internal numerics for every
        # arch; offset = constant_bias reproduces Gemma's (1 + weight). fp32 is
        # allowed (the gemma-mode upcasts are no-ops there): pointless for VRAM but
        # required by the parity gate's fp32 math tier, which separates wrong-math
        # from half-precision reassociation noise.
        # An UNWEIGHTED norm (weight None -- MuseGlimmer's embed_norm and qk_norm)
        # has no weight tensor for Liger's kernel to take, and its Function saves
        # W for backward, so it can't run there: those fall through to the torch
        # path below. (Gemma's unweighted v-norm never reached this branch either
        # -- it's the 4D per-head case.)
        if (self.use_liger and x.is_cuda and x.dim() in (2, 3)
                and spec["scale"] == 1.0 and spec["weight"] is not None
                and self.compute_dtype in (torch.float16, torch.bfloat16,
                                           torch.float32)):
            ops = _liger_ops()
            if ops is not None:
                w = spec["weight"]
                if w is not None:
                    # Move the norm weight to x's device (under --parallel split a
                    # block's weights and activations share a card, but the Triton
                    # kernel launches on the *current* CUDA device), keeping the
                    # weight's OWN dtype: casting_mode="gemma" upcasts W to fp32
                    # inside the kernel, so the storage dtype reproduces the torch
                    # path's w.float() exactly for any base (a cast to x.dtype
                    # would round fp16-normed bases when compute is bf16).
                    w = w.to(device=x.device)
                    # The frozen base weight is an inference tensor (the EXL3 model
                    # loads under @torch.inference_mode), and Liger's autograd Function
                    # saves W for backward -- which forbids inference tensors. When the
                    # .to() above already copied (dtype/device change) the result is a
                    # normal tensor; when it was a no-op (weight already in compute
                    # dtype on this card) clone to get one. The torch path below sidesteps
                    # this for free via w.float(). Negligible: w is a [hidden] vector.
                    if w.is_inference():
                        w = w.clone()
                # Triton launches on torch.cuda.current_device(), not x.device, so
                # for a tail block on cuda:1 (split mode) it would otherwise emit a
                # cuda:1 pointer from a cuda:0 launch ("cannot be accessed from
                # Triton (cpu tensor?)"). Pin the launch to x's device.
                with torch.cuda.device(x.device):
                    # in_place=False (6th arg). Liger defaults in_place=True, whose
                    # backward writes dX into the grad-output buffer; under
                    # use_reentrant=False checkpointing + the residual that also consumes
                    # this norm's input that reuse corrupts gradients (forward fine, grad
                    # norm explodes ~1e16). A fresh dX buffer costs one [tokens, hidden]
                    # allocation per norm and fixes it.
                    return ops[0].apply(x, w, spec["eps"], spec["bias"], "gemma", False)
        # fp32 internals for fidelity on half/bf16/fp32 inputs; fp64 inputs
        # stay fp64 -- that path exists only for the fp64 gradchecks (e.g. the
        # Gemma4 MoE norms), where a .float() round-trip would corrupt the
        # double-precision Jacobian. No real training run uses fp64.
        xf = x if x.dtype == torch.float64 else x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True) + spec["eps"]
        xn = xf * torch.rsqrt(var) * spec["scale"]
        w = spec["weight"]
        if w is None:
            return xn.to(x.dtype)
        w = w.to(xf.dtype)
        b = spec["bias"]
        out = xn * (w + b) if b != 0.0 else xn * w
        return out.to(x.dtype)

    def _apply_rope(self, x: torch.Tensor, inv_freq: torch.Tensor, attn_factor: float,
                    position_ids: torch.Tensor, mrope_section=None) -> torch.Tensor:
        # x: [b, t, n_heads, head_dim]. position_ids: [b, t], or [3, b, t] for
        # 3-D mRoPE positions (image+text on a Qwen-VL tower). Rotation math
        # runs in fp32 (position-dependent precision) and the result is cast
        # back to x's dtype, so a bf16 activation stays bf16 instead of being
        # upcast to fp32.
        in_dtype = x.dtype
        rot = 2 * inv_freq.numel()          # rotary_dim; == head_dim for full rotary
        if rot < x.shape[-1]:
            # Partial rotary (Qwen-VL partial_rotary_factor): rotate only the
            # leading `rot` dims of each head, pass the trailing dims through
            # unrotated. exllamav3 builds inv_freq over the rotary slice only
            # (util/rope.py "default"/"mrope" case), so 2*inv_freq.numel() IS the
            # rotary width. The recursive call hits the full-rotary branch below
            # (rot == x_rot.shape[-1]); for full-rotary models rot == head_dim so
            # there is no split and behavior is byte-identical to before.
            x_rot = self._apply_rope(x[..., :rot], inv_freq, attn_factor, position_ids,
                                     mrope_section)
            return torch.cat((x_rot, x[..., rot:]), dim=-1)
        xf = x.float()
        if position_ids.dim() == 3:
            # mRoPE: [3, b, t] (t/h/w) positions. Text tokens carry the same
            # value on all three axes, so a text-only batch reduces to the 1-D
            # rotation below; image tokens spread over the axes and each
            # frequency band picks its axis per mrope_section (the interleaved
            # split RoPE.get_mrope_freqs applies at inference). A block without
            # a section (1-D RoPE arch fed 3-D positions) uses the t axis.
            if mrope_section is not None:
                freqs = vision.mrope_freqs(position_ids, inv_freq, mrope_section)  # [b,t,rot/2]
            else:
                freqs = position_ids[0].float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
        else:
            freqs = position_ids.float().unsqueeze(-1) * inv_freq.float().unsqueeze(0).unsqueeze(0)  # [b,t,rot/2]
        emb = torch.cat((freqs, freqs), dim=-1)                  # [b,t,hd]  (NeoX layout)
        cos = (emb.cos() * attn_factor).unsqueeze(2)             # [b,t,1,hd]
        sin = (emb.sin() * attn_factor).unsqueeze(2)
        return (xf * cos + _rotate_half_neox(xf) * sin).to(in_dtype)

    def _attn_bias(self, attention_mask: Optional[torch.Tensor], t: int,
                   device, dtype, window: int = -1,
                   seg_ids: Optional[torch.Tensor] = None,
                   bidir_spans: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Additive attention bias: causal (upper triangle masked) AND, if given, a
        # [b, t] key-padding mask. Assumes right-padding (pads at the end), so no
        # real-token query row is ever fully masked -> softmax can't produce NaN.
        # When window > 0, also mask keys older than the sliding window (query i
        # attends only to j with i - window < j <= i), matching Gemma's local
        # layers; window <= 0 is full causal.
        #
        # seg_ids ([b, t]) is the sample-packing block-diagonal mask: query i may
        # only attend to key j in the SAME document (seg_ids[i] == seg_ids[j]), so
        # packed documents never bleed across boundaries. This is the eager
        # *reference* path (CUDA fp16/bf16 packed runs use flash-varlen instead);
        # at the small t of CPU/fp32/gradcheck/validate the [b,1,t,t] size is fine.
        # Pad positions carry the last real document's seg id (see pack_examples),
        # so a pad query still attends back into a real doc -> never fully masked.
        #
        # bidir_spans ([b, t]; -1 = text, k = image span k) lifts the causal /
        # window masks INSIDE each image span: query i may attend to key j
        # whenever both sit in the same span, in either direction. This is the
        # Gemma4 multimodal rule (inference runs each image span non-causally
        # over itself -- attention_fn's non_causal_spans, built by the arch's
        # _prepare_noncausal_mm_spans); only used when
        # backbone.uses_noncausal_mm_spans says the arch does it.
        neg = float("-inf")
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)
        blocked = causal[None, None].clone()                     # [1,1,t,t]
        if window and window > 0:
            too_old = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device),
                                 diagonal=-window)
            blocked = blocked | too_old[None, None]
        if bidir_spans is not None:
            sp = bidir_spans.to(device)
            same = (sp[:, :, None] == sp[:, None, :]) & (sp[:, :, None] >= 0)  # [b,t,t]
            blocked = blocked & ~same[:, None]                   # [b,1,t,t]
        bias = torch.zeros(blocked.shape, dtype=dtype, device=device)
        bias = bias.masked_fill(blocked, neg)
        if seg_ids is not None:
            # [b,1,t,t]: True where query i and key j are in different documents.
            # Add -inf there (additive, so it composes with the causal/window
            # masks above); broadcasts the [1,1,t,t] bias up to [b,1,t,t].
            cross = (seg_ids[:, :, None] != seg_ids[:, None, :])[:, None]
            bias = bias + torch.zeros_like(cross, dtype=dtype).masked_fill(cross, neg)
        if attention_mask is not None:
            key_pad = (attention_mask == 0)[:, None, None, :]    # [b,1,1,t]
            bias = bias.masked_fill(key_pad, neg)
        return bias

    def _block_forward(self, meta, entry, hidden, position_ids, attn_bias,
                       attn_mode="eager", pack=None):
        bsz, t, _ = hidden.shape
        nq, nkv, hd = meta["num_q_heads"], meta["num_kv_heads"], meta["head_dim"]

        # --- attention ---
        # Projections already run in compute_dtype (the QLoRA linear casts input and
        # reconstructs the frozen weight in compute_dtype), so q/k/v come out bf16 in
        # a training run / fp32 in the validate path -- DON'T upcast them to fp32 here
        # (that is what doubled the activation footprint). Per-head norm and RoPE
        # below keep their internals in fp32 but return in this dtype.
        normed = self._norm(hidden, entry.attn_norm_spec)
        if meta.get("interleaved_gate"):
            # Qwen3.5 gated attention: q_proj emits [q | gate] interleaved per
            # head ([..., nq, 2*hd]); the gate multiplies the attention output
            # (sigmoid) before o_proj. Split exactly as the inference
            # project_qkv does (chunk on the per-head 2*hd slice).
            qg = entry.q_proj(normed).view(bsz, t, nq, 2 * hd)
            q, out_gate = torch.chunk(qg, 2, dim=-1)             # [b,t,nq,hd] each
            out_gate = out_gate.reshape(bsz, t, nq * hd)
        elif meta.get("full_gate"):
            # AFMoE gated attention: the gate comes from its own full-width
            # linear on the block input; the sigmoid multiply below is the
            # same as the interleaved case (inference: g = g_proj(x), then
            # ext.mul_sigmoid_(o, g) on the flattened context before o_proj).
            q = entry.q_proj(normed).view(bsz, t, nq, hd)
            out_gate = entry.g_proj(normed)                      # [b,t,nq*hd]
        else:
            q = entry.q_proj(normed).view(bsz, t, nq, hd)
            out_gate = None
        k = entry.k_proj(normed).view(bsz, t, nkv, hd)
        # use_k_as_v: V is the raw K projection (taken before k-norm/RoPE).
        v = k if entry.v_proj is None else entry.v_proj(normed).view(bsz, t, nkv, hd)

        # Per-head q/k/v RMSNorm, applied to the raw projections before RoPE
        # (Qwen3: q/k; Gemma: q/k/v unweighted). No-ops when the spec is None.
        if entry.q_norm_spec is not None:
            q = self._norm(q, entry.q_norm_spec)
        if entry.k_norm_spec is not None:
            k = self._norm(k, entry.k_norm_spec)
        if entry.v_norm_spec is not None:
            v = self._norm(v, entry.v_norm_spec)

        # NoPE layers (AFMoE full-attention layers) carry no RoPE table at all
        # -- skip the rotation exactly as the inference forward's `if self.rope:`.
        if meta["inv_freq"] is not None:
            sec = meta.get("mrope_section")
            q = self._apply_rope(q, meta["inv_freq"], meta["attn_factor"], position_ids, sec)
            k = self._apply_rope(k, meta["inv_freq"], meta["attn_factor"], position_ids, sec)

        if attn_mode == "flash":
            # FlashAttention-2 (autograd-capable upstream flash_attn): O(t) memory,
            # never materializes the [b, nq, t, t] score matrix. Takes [b, t, nh, hd]
            # (no transpose), handles GQA (nkv < nq) and the causal / sliding-window
            # mask and softcap via flags -- so the eager [t, t] bias is not built at
            # all. Right-padding (collate) means a real-token query never attends to
            # trailing pad keys under `causal`, matching the eager mask for every
            # loss-bearing position. FA2 supports head_dim <= 256; larger heads take
            # the SDPA branch. Runs in compute_dtype (fp16/bf16).
            if pack is not None:
                # Sample packing: flatten the batch's non-pad tokens into one
                # [total, nh, hd] stream and run variable-length flash, isolating
                # documents by cu_seqlens with NO [t,t] mask (the O(t) packing
                # primitive). RoPE was applied above with per-document position
                # resets; here we just gather, attend and scatter back.
                # keep / cu_seqlens are built on the first block's device; under a
                # layer-autosplit load this block may be on another card, so localize.
                keep = pack["keep"].to(hidden.device)           # [b, t] bool, non-pad
                cu = pack["cu_seqlens"].to(hidden.device)
                of = backbone.attn_varlen(
                    q.to(self.compute_dtype)[keep],             # [total, nq, hd]
                    k.to(self.compute_dtype)[keep],             # [total, nkv, hd]
                    v.to(self.compute_dtype)[keep],
                    cu, pack["max_seqlen"],
                    meta["sm_scale"], window=meta["sliding_window"],
                    softcap=float(meta["softcap"] or 0.0),
                )                                               # [total, nq, hd]
                o = q.new_zeros(bsz, t, nq, hd, dtype=of.dtype)
                o[keep] = of
                ctx = o.reshape(bsz, t, nq * hd)
            else:
                window = meta["sliding_window"]
                ws = (window - 1, 0) if window and window > 0 else (-1, -1)
                o = _flash_attn_func()(
                    q.to(self.compute_dtype), k.to(self.compute_dtype), v.to(self.compute_dtype),
                    softmax_scale=meta["sm_scale"], causal=True,
                    window_size=ws, softcap=float(meta["softcap"] or 0.0),
                )                                               # [b, t, nq, hd]
                ctx = o.reshape(bsz, t, nq * hd)
        elif attn_mode == "sdpa":
            # head_dim > 256 (Gemma global layers): FA2 can't (caps at 256), and
            # torch's mem-efficient / flash SDPA backends ALSO cap at head_dim 256 --
            # so at this head dim SDPA can only pick the MATH backend, which
            # materializes the [nq, L, L] score matrix in fp32 (it upcasts regardless
            # of input dtype). This is the same O(t^2) math HF/Axolotl run on these
            # layers too. (An earlier comment here claimed the mem-efficient backend
            # could be kept "eligible" by hand-expanding GQA and dropping the mask --
            # that premise was wrong; see the Session 8 correction in the handoff doc.)
            # We bound the spike by running PER DOCUMENT under packing with is_causal,
            # so the score matrix scales with the longest *document*, not the full
            # packed block. The hand repeat_interleave of K/V to nq heads below is now
            # pure overhead (the math backend handles GQA via enable_gqa) and should be
            # dropped in the query-tiled rewrite -- left as-is here to avoid an
            # untested change to the attention path. The real O(t) fix for this head
            # dim is query-tiled attention (module header / handoff doc) -- not built.
            # Non-packed: one full is_causal pass.
            cd = self.compute_dtype
            rep = nq // nkv
            if pack is not None:
                keep = pack["keep"].to(hidden.device)           # [b, t] bool
                cu = pack["cu_seqlens"].to(hidden.device)       # [num_docs + 1]
                qf = q[keep].to(cd)                             # [total, nq, hd]
                kf = k[keep].to(cd)                             # [total, nkv, hd]
                vf = v[keep].to(cd)
                if rep > 1:                                     # expand GQA -> nq heads
                    kf = kf.repeat_interleave(rep, dim=1)
                    vf = vf.repeat_interleave(rep, dim=1)
                of = qf.new_zeros(qf.shape[0], nq, hd)
                cul = cu.tolist()
                for s, e in zip(cul[:-1], cul[1:]):
                    # [1, nq, L, hd] for SDPA; isolate this document.
                    od = F.scaled_dot_product_attention(
                        qf[s:e].transpose(0, 1).unsqueeze(0),
                        kf[s:e].transpose(0, 1).unsqueeze(0),
                        vf[s:e].transpose(0, 1).unsqueeze(0),
                        is_causal=True, scale=meta["sm_scale"],
                    )                                           # [1, nq, L, hd]
                    of[s:e] = od.squeeze(0).transpose(0, 1)
                o = q.new_zeros(bsz, t, nq, hd, dtype=of.dtype)
                o[keep] = of
                ctx = o.reshape(bsz, t, nq * hd)
            else:
                qh = q.transpose(1, 2).to(cd)
                kh = k.transpose(1, 2).to(cd)
                vh = v.transpose(1, 2).to(cd)
                if rep > 1:
                    kh = kh.repeat_interleave(rep, dim=1)
                    vh = vh.repeat_interleave(rep, dim=1)
                o = F.scaled_dot_product_attention(
                    qh, kh, vh, is_causal=True, scale=meta["sm_scale"],
                )                                               # [b, nq, t, hd]
                ctx = o.transpose(1, 2).reshape(bsz, t, nq * hd)
        else:
            # Eager reference (validate / CPU / fp32 / gradcheck): materializes the
            # [b, nq, t, t] scores. Keep the score+softmax math in fp32 for fidelity
            # regardless of the activation dtype, then return ctx in compute_dtype.
            q = q.transpose(1, 2).float()
            k = k.transpose(1, 2).float()
            v = v.transpose(1, 2).float()
            if nq != nkv:                                       # GQA: expand KV groups
                rep = nq // nkv
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)

            scores = torch.matmul(q, k.transpose(-1, -2)) * meta["sm_scale"]  # [b,nq,t,t]
            softcap = meta["softcap"]
            if softcap:                                         # tanh logit softcap (Gemma2)
                scores = softcap * torch.tanh(scores / softcap)
            scores = scores + attn_bias.to(scores.dtype)
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v)                        # [b,nq,t,hd]
            ctx = ctx.transpose(1, 2).reshape(bsz, t, nq * hd)  # fp32 reference result
        # Output gate (Qwen3.5 interleaved / AFMoE full-width g_proj):
        # o *= sigmoid(gate), applied to the flattened context exactly as the
        # inference mul_sigmoid_ does, before o_proj. sigmoid in fp32 for
        # fidelity, back to ctx dtype.
        if out_gate is not None:
            ctx = ctx * torch.sigmoid(out_gate.float()).to(ctx.dtype)
        # o_proj re-casts its input to compute_dtype and outputs in it, so attn_out is
        # bf16 in a training run regardless of ctx's dtype -- no fp32 upcast of the
        # residual stream here (that is what the old `.float()` was costing).
        attn_out = entry.o_proj(ctx)
        # Sandwich post-norm (Gemma): x = x + post_norm(attn_out). Plain pre-norm
        # archs have no post-norm -> straight residual add.
        if entry.attn_post_spec is not None:
            hidden = hidden + self._norm(attn_out, entry.attn_post_spec)
        else:
            hidden = hidden + attn_out

        # --- gated MLP (SwiGLU / GeGLU) / MoE ---
        # `hidden` here is the post-attention residual stream -- exactly what
        # the inference TransformerBlock stashes as params["residual"] for the
        # Gemma4 MoE's alt residual channel.
        normed2 = self._norm(hidden, entry.mlp_norm_spec)
        mlp_out = self._mlp_out(meta, entry, normed2, residual=hidden)
        if entry.mlp_post_spec is not None:
            hidden = hidden + self._norm(mlp_out, entry.mlp_post_spec)
        else:
            hidden = hidden + mlp_out

        # Gemma's learned per-layer scalar on the whole residual stream (block end).
        # None for plain archs -> no-op. Compounds over layers, so omitting it on
        # Gemma produces garbage even though each block is individually close.
        ls = meta.get("layer_scalar")
        if ls is not None:
            # Cast the result back to hidden's dtype: a fp32 layer_scalar tensor would
            # otherwise promote the whole bf16 residual to fp32 at every block end.
            hidden = (hidden * ls).to(hidden.dtype)
        return hidden

    def _mlp_out(self, meta, entry, normed2, residual=None):
        # MLP half of a block, shared by the softmax-attention and
        # GatedDeltaNet block forwards: dense gated MLP, or the block-sparse
        # mixture of experts (meta["mlp_kind"] == "moe"). ``residual`` is the
        # RAW post-attention residual stream, needed only by the Gemma4 MoE
        # layout (alt_residual_channel routes from it); dense MLPs ignore it.
        if meta.get("mlp_kind", "dense") == "moe":
            return self._moe_out(meta, entry, normed2, residual)
        return self._gated_mlp(entry.gates, entry.ups, entry.downs,
                               meta["activation"], normed2)

    def _gated_mlp(self, gates, ups, downs, activation, x):
        # Gated MLP (SwiGLU / GeGLU) over the already-normed input. The MLP may
        # be sliced across the intermediate dim (very wide models) -- sum the
        # downs. Also runs a MoE block's shared expert (its slices, its
        # activation).
        is_silu = activation == "silu"
        act = F.silu if is_silu else (lambda z: F.gelu(z, approximate="tanh"))
        # Liger fused SiLU*mul (CUDA, silu only -- GeGLU stays torch): fuses the
        # activation+multiply into one kernel, saving an intermediate. fp32 is
        # allowed for the parity gate's math tier (see _norm).
        liger_silu = (self.use_liger and is_silu and x.is_cuda
                      and self.compute_dtype in (torch.float16, torch.bfloat16,
                                                 torch.float32))
        liger_ops = _liger_ops() if liger_silu else None
        mlp_out = None
        for gate, up, down in zip(gates, ups, downs):
            if liger_ops is not None:
                # Pin the Triton launch to the block's device (split mode: a tail
                # block on cuda:1 while current_device is cuda:0 would otherwise
                # fault on an inaccessible pointer). See _norm for the full note.
                with torch.cuda.device(x.device):
                    a = liger_ops[1].apply(gate(x), up(x))       # silu(gate)*up
            else:
                a = act(gate(x)) * up(x)
            d = down(a)                                          # compute_dtype (no fp32 upcast)
            mlp_out = d if mlp_out is None else mlp_out + d
        return mlp_out

    def _moe_out(self, meta, entry, normed2, residual=None):
        """Block-sparse mixture of experts (BlockSparseMLP -- Qwen3-MoE /
        Qwen3.5-MoE / Gemma4 MoE / AFMoE) over the already-normed input.

        Routing reproduces the inference kernels. "std" (``ext.routing_std``):
        logits from the frozen router gate, top-k over the logits, softmax
        over the selected k in fp32 -- mathematically identical to HF's
        softmax-over-all + renormalize-top-k with ``norm_topk_prob=True``
        (which the Qwen MoE configs assert), since restricting a softmax to a
        subset and renormalizing equals the softmax over that subset's
        logits. "dots" (``ext.routing_ds3_nogroup`` -- AFMoE): sigmoid
        scores, the frozen ``e_score_correction_bias`` added for top-k
        SELECTION only, the selected experts weighted by their unbiased
        scores normalized over the selected set and scaled by
        ``routed_scaling_factor``. The
        routing weights stay differentiable w.r.t. the hidden state (gradients
        flow through the weighted sum into the softmax and router matmul; the
        arg-top-k selection itself is piecewise-constant, exactly as in HF /
        PEFT training). The router itself is frozen (r=0) -- no mainstream
        MoE-LoRA recipe trains it, and no auxiliary load-balancing loss is
        added (HF leaves it off by default for fine-tuning; with a frozen
        router it could not rebalance routing anyway).

        Experts run as a gather -> gated-MLP -> weighted index_add scatter per
        expert with assigned tokens (the HF Qwen3MoeSparseMoeBlock shape),
        each projection through its DiffLinear wrapper so LoRA / pissa /
        quant-aware compose. Accumulation is fp32, matching the inference
        path's float output buffer. The optional shared expert (Qwen3.5-MoE)
        adds ``sigmoid(shared_gate(x)) * shared_mlp(x)``, exactly the
        inference ``add_sigmoid_gate`` kernel. Per-token math throughout, so
        sample packing needs no special case (Qwen3.5-MoE still rejects
        packing at the net level, for its GDN layers).

        Gemma4 layout (``meta["alt_residual_channel"]``, mirroring
        ``BlockSparseMLP.forward``'s ``params["residual"]`` read): routing and
        the routed experts consume the RAW post-attention ``residual`` stream
        through their own pre-norms (``router_pre`` -- an RMSNorm additionally
        scaled by ``hidden**-0.5`` via its ``constant_scale`` -- and
        ``routed_pre``), while the shared expert consumes ``normed2`` (the
        block's ``pre_feedforward_layernorm`` output) as usual. The routed sum
        then passes ``routed_post`` and the shared output ``shared_post``
        before the two are added; the block's outer ``mlp_post_norm`` sandwich
        and ``layer_scalar`` apply on top in ``_block_forward``. All four
        norms are optional per the module (``None`` on Qwen-family MoE, which
        reduces this to the plain path)."""
        bsz, t, d = normed2.shape
        y = normed2.reshape(-1, d)
        nt = y.shape[0]
        num_experts = meta["num_experts"]
        top_k = meta["num_experts_per_tok"]
        # Routing weights + output accumulation promote to >= fp32 (like the
        # fused-CE head): fp32 for half/bf16/fp32 compute, fp64 preserved for
        # the fp64 gradcheck path.
        acc = torch.float64 if y.dtype == torch.float64 else torch.float32

        # Routing / routed-expert input channel: the raw residual (Gemma4) or
        # the normed block input (everything else). getattr defaults keep
        # pre-Gemma4 entries (and the CPU test namespaces) working unchanged.
        if meta.get("alt_residual_channel"):
            assert residual is not None, \
                "alt_residual_channel MoE needs the block's residual stream"
            yr = residual.reshape(-1, d)
        else:
            yr = y
        router_pre = getattr(entry, "router_pre_spec", None)
        routed_pre = getattr(entry, "routed_pre_spec", None)
        z = self._norm(yr, router_pre) if router_pre is not None else yr
        ye = self._norm(yr, routed_pre) if routed_pre is not None else yr

        logits = entry.router(z).to(acc)                        # [nt, E]
        if meta.get("router_type", "std") == "dots":
            # Ungrouped sigmoid router (AFMoE / dots.llm1), mirroring the
            # inference routing_dots / ext.routing_ds3_nogroup: sigmoid
            # scores; e_score_correction_bias added for the top-k SELECTION
            # only; the selected experts weighted by their UNBIASED scores,
            # normalized over the selected set (+1e-20, the kernel's guard),
            # scaled by routed_scaling_factor. Differentiable w.r.t. the
            # hidden state through sigmoid/gather/normalize; the arg-top-k
            # selection (and the bias, which only shifts it) is
            # piecewise-constant, exactly as in the HF reference.
            scores = torch.sigmoid(logits)                      # [nt, E] >=fp32
            e_bias = meta.get("e_score_bias")
            sel_scores = scores if e_bias is None \
                else scores + e_bias.to(acc).to(scores.device)
            _, top_idx = torch.topk(sel_scores, top_k, dim=-1)
            weights = scores.gather(1, top_idx)                 # [nt, k]
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            weights = weights * meta.get("routed_scaling_factor", 1.0)
        else:
            top_logits, top_idx = torch.topk(logits, top_k, dim=-1)
            weights = torch.softmax(top_logits, dim=-1)         # [nt, k] >=fp32
            pes = meta.get("per_expert_scale")
            if pes is not None:
                weights = weights * pes.to(acc).to(weights.device)[top_idx]

        act = F.silu if meta["activation"] == "silu" \
            else (lambda z: F.gelu(z, approximate="tanh"))

        # Group token assignments by expert: one sort, then per-expert
        # contiguous slices (the inference forward's batched layout). The
        # bincount round-trip to host is one small sync per MoE layer --
        # unavoidable for data-dependent expert batching (inference pays the
        # same). stable sort so the grad-checkpoint recompute reproduces the
        # exact fp reduction order.
        flat_idx = top_idx.reshape(-1)                          # [nt*k]
        flat_w = weights.reshape(-1)                            # [nt*k] fp32
        flat_tok = torch.arange(nt, device=y.device).repeat_interleave(top_k)
        order = torch.argsort(flat_idx, stable=True)
        tok_sorted = flat_tok[order]
        w_sorted = flat_w[order]
        counts = torch.bincount(flat_idx, minlength=num_experts).tolist()

        out = torch.zeros(nt, d, dtype=acc, device=y.device)
        start = 0
        for e in range(num_experts):
            end = start + counts[e]
            if end == start:
                continue                                        # no tokens routed here
            toks = tok_sorted[start:end]
            w = w_sorted[start:end].unsqueeze(1)                # [c, 1] >=fp32
            xe = ye.index_select(0, toks)                       # [c, d]
            h = act(entry.expert_gates[e](xe)) * entry.expert_ups[e](xe)
            de = entry.expert_downs[e](h)                       # [c, d] compute dtype
            out.index_add_(0, toks, de.to(acc) * w)
            start = end

        # Gemma4: the routed output sum is normed before the shared expert is
        # added (post_feedforward_layernorm_2 in the checkpoint).
        routed_post = getattr(entry, "routed_post_spec", None)
        if routed_post is not None:
            out = self._norm(out, routed_post)

        # Shared expert (always active), gated by sigmoid(shared_gate(x)).
        # Gemma4 has no shared gate but norms the shared output
        # (post_feedforward_layernorm_1) before adding it to the routed sum.
        if len(entry.gates):
            sh = self._gated_mlp(entry.gates, entry.ups, entry.downs,
                                 meta.get("shared_activation") or meta["activation"], y)
            shared_post = getattr(entry, "shared_post_spec", None)
            if shared_post is not None:
                sh = self._norm(sh, shared_post)
            if entry.shared_gate is not None:
                g = torch.sigmoid(entry.shared_gate(y).to(acc))  # [nt, 1]
                out = out + g * sh.to(acc)
            else:
                out = out + sh.to(acc)

        return out.view(bsz, t, d).to(normed2.dtype)

    # --- GatedDeltaNet (linear attention) block forward --------------------

    def _gdn_delta_rule(self, q, k, v, g, beta):
        """Dispatch the gated delta rule: fla's chunked Triton kernel (the same
        one exllamav3's inference prefill uses -- autograd-capable, O(t) memory)
        on CUDA fp16/bf16, else the sequential fp32 reference (CPU / fp32
        validate / gradcheck). q/k arrive RAW; both paths L2-normalize per head
        internally. Shapes: q/k [b,t,nk,dk], v [b,t,nv,dv], g/beta [b,t,nv];
        returns [b,t,nv,dv]."""
        fla_fn = _fla_chunk_fn()
        use_fla = (fla_fn is not None and q.is_cuda and self.attn_impl != "eager"
                   and self.compute_dtype in (torch.float16, torch.bfloat16))
        if use_fla:
            cd = self.compute_dtype
            core, _ = fla_fn(
                q.to(cd), k.to(cd), v.to(cd),
                g=g.float(), beta=beta.to(cd),
                initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            return core
        return gdn_delta_rule_reference(q, k, v, g, beta)

    def _gdn_forward(self, meta, entry, hidden):
        """One GatedDeltaNet decoder block (Qwen3.5/3.6 linear-attention layer):
        pre-norm -> split q/k/v + z + b/a projections -> causal depthwise conv +
        SiLU -> gated delta rule -> gated RMSNorm * silu(z) -> o_proj residual,
        then the standard gated-MLP half. Stateless (sequences start at position
        0), so the inference path's zero conv/recurrent state is reproduced
        exactly; no RoPE, no attention mask (the recurrence is causal by
        construction). Right-padding is safe: pad positions produce garbage that
        never feeds back into real positions and is masked from the loss."""
        bsz, t, _ = hidden.shape
        nk, nv = meta["num_k_heads"], meta["num_v_heads"]
        dk, dv = meta["k_head_dim"], meta["v_head_dim"]
        k_dim, v_dim = nk * dk, nv * dv

        normed = self._norm(hidden, entry.attn_norm_spec)
        qkv = entry.qkv_proj(normed)                       # [b, t, 2*k_dim + v_dim]
        z = entry.z_proj(normed).view(bsz, t, nv, dv)      # output gate
        beta, g = gdn_beta_g(
            entry.b_proj(normed), entry.a_proj(normed),
            meta["a_log"], meta["dt_bias"], meta["beta_scale"],
        )                                                  # [b, t, nv] fp32 each

        # Causal depthwise conv + SiLU over the packed q|k|v channels, in the
        # activation dtype (fp32 on the validate path, compute dtype in training;
        # the inference kernel runs bf16).
        x = qkv.transpose(1, 2)                            # [b, dim, t]
        x = gdn_causal_conv1d_silu(x, meta["conv1d_weight"], meta["conv1d_bias"])
        x = x.transpose(1, 2)                              # [b, t, dim]

        q, k, v = torch.split(x, [k_dim, k_dim, v_dim], dim=-1)
        q = q.view(bsz, t, nk, dk)
        k = k.view(bsz, t, nk, dk)
        v = v.view(bsz, t, nv, dv)

        core = self._gdn_delta_rule(q, k, v, g, beta)      # [b, t, nv, dv]
        core = gdn_gated_rmsnorm(core, entry.gdn_norm_spec, z)
        attn_out = entry.o_proj(core.reshape(bsz, t, v_dim))
        if entry.attn_post_spec is not None:
            hidden = hidden + self._norm(attn_out, entry.attn_post_spec)
        else:
            hidden = hidden + attn_out

        normed2 = self._norm(hidden, entry.mlp_norm_spec)
        mlp_out = self._mlp_out(meta, entry, normed2, residual=hidden)
        if entry.mlp_post_spec is not None:
            hidden = hidden + self._norm(mlp_out, entry.mlp_post_spec)
        else:
            hidden = hidden + mlp_out

        ls = meta.get("layer_scalar")
        if ls is not None:
            hidden = (hidden * ls).to(hidden.dtype)
        return hidden

    # --- ShortConv (LFM2 gated causal conv) block forward ------------------

    def _shortconv_forward(self, meta, entry, hidden):
        """One ShortConv decoder block (LFM2 / LFM2-MoE ``conv`` layer):
        pre-norm -> in_proj -> ``b | c | x`` split -> causal depthwise conv over
        ``b * x`` -> ``c`` gate -> out_proj residual, then the standard MLP
        half (dense or MoE). Stateless (sequences start at position 0), so the
        inference path's zero conv state is reproduced exactly by a left pad of
        ``kernel - 1`` zeros; no RoPE, no attention mask (the conv is causal
        by construction). Right-padding is safe: pad positions only ever feed
        later pad positions, which are masked from the loss."""
        normed = self._norm(hidden, entry.attn_norm_spec)
        bcx = entry.in_proj(normed)                        # [b, t, 3*hidden]
        y = shortconv_mix(bcx, meta["conv1d_weight"], meta["conv1d_bias"])
        attn_out = entry.out_proj(y)
        if entry.attn_post_spec is not None:
            hidden = hidden + self._norm(attn_out, entry.attn_post_spec)
        else:
            hidden = hidden + attn_out

        normed2 = self._norm(hidden, entry.mlp_norm_spec)
        mlp_out = self._mlp_out(meta, entry, normed2, residual=hidden)
        if entry.mlp_post_spec is not None:
            hidden = hidden + self._norm(mlp_out, entry.mlp_post_spec)
        else:
            hidden = hidden + mlp_out

        ls = meta.get("layer_scalar")
        if ls is not None:
            hidden = (hidden * ls).to(hidden.dtype)
        return hidden

    # --- attention backend selection -------------------------------------

    def _attn_mode_for(self, meta, mem_eff: bool) -> str:
        """Per-block attention backend: "flash" (FA2, head_dim<=256), "sdpa"
        (head_dim>256 full-causal no-softcap -- torch's mem-efficient backend, O(t),
        per-document under packing for isolation), else "eager". mem_eff gates the
        memory-efficient kernels (CUDA + fp16/bf16)."""
        if not mem_eff:
            return "eager"
        hd = meta["head_dim"]
        if hd <= 256 and hd % 8 == 0:
            return "flash"
        # head_dim > 256 (Gemma global layers): FA2 unsupported, but torch's
        # mem-efficient SDPA backend handles big heads at O(t). Only the plain
        # full-causal case goes there (per-document under packing for isolation);
        # windowed/softcap big-head stays eager.
        if meta["sliding_window"] <= 0 and not meta["softcap"]:
            return "sdpa"
        return "eager"

    def _build_pack_context(self, seg_ids, attention_mask, bsz, t, device):
        """Precompute the variable-length packing descriptor for the flash path.

        Returns ``{keep, cu_seqlens, max_seqlen}`` or ``None`` if there are no real
        tokens. ``keep`` ([b, t] bool) selects non-pad tokens; flattening the batch
        in row-major order yields one token stream whose per-document boundaries are
        ``cu_seqlens`` (int32, ``[num_docs + 1]`` cumulative lengths) -- exactly the
        input ``flash_attn_varlen_func`` wants. A document is a maximal run of equal
        ``seg_ids`` within a row (pads, which carry the last doc's seg id, are
        dropped by ``keep``); a row change also starts a new document, since seg ids
        are block-local (each packed sequence numbers its docs from 0)."""
        keep = (attention_mask.to(device) > 0) if attention_mask is not None \
            else torch.ones(bsz, t, dtype=torch.bool, device=device)
        seg = seg_ids.to(device)
        seg_flat = seg[keep]                                    # [total], row-major
        if seg_flat.numel() == 0:
            return None
        rows = torch.arange(bsz, device=device)[:, None].expand(bsz, t)[keep]
        new_doc = torch.ones_like(seg_flat, dtype=torch.bool)
        new_doc[1:] = (seg_flat[1:] != seg_flat[:-1]) | (rows[1:] != rows[:-1])
        doc_id = torch.cumsum(new_doc.long(), 0) - 1            # 0..num_docs-1
        counts = torch.bincount(doc_id)                        # tokens per document
        cu = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.cumsum(counts, 0).to(torch.int32)
        return {"keep": keep, "cu_seqlens": cu, "max_seqlen": int(counts.max().item())}

    def describe_attn(self) -> str:
        """One-line summary of the attention plan for the loaded model + dtype, so
        a run can confirm flash/SDPA actually engage (vs a silent eager fallback)."""
        from collections import Counter
        # block.device may be a torch.device OR the string passed to load
        # ("cuda:0"); match the forward's intent (hidden lands on this device)
        # with a string test rather than a .type attribute that strings lack.
        dev = self._block_devices[0]
        is_cuda = "cuda" in str(dev)
        mem_eff = (self._flash_ok and is_cuda
                   and self.compute_dtype in (torch.float16, torch.bfloat16))
        counts = Counter(
            m["kind"] if m.get("kind", "attn") in ("gdn", "shortconv")
            else self._attn_mode_for(m, mem_eff)
            for m in self._block_meta)
        plan = ", ".join(f"{counts[k]}×{k}" for k in
                         ("flash", "sdpa", "eager", "gdn", "shortconv")
                         if counts[k])
        avail = "available" if _flash_attn_func() is not None else "NOT importable"
        why = "" if mem_eff else "  (eager: needs CUDA + fp16/bf16" + \
            ("" if self._flash_ok else " + flash_attn") + ")"
        line = f"attn: {plan}  [impl={self.attn_impl}, flash_attn {avail}]{why}"
        moe = [m for m in self._block_meta if m.get("mlp_kind") == "moe"]
        if moe:
            line += (f"  [moe: {len(moe)} layers x {moe[0]['num_experts']} experts, "
                     f"top-{moe[0]['num_experts_per_tok']}]")
        if counts["gdn"]:
            gdn_fast = (is_cuda and self.attn_impl != "eager"
                        and self.compute_dtype in (torch.float16, torch.bfloat16)
                        and _fla_chunk_fn() is not None)
            line += ("  [gdn: fla chunked kernel]" if gdn_fast
                     else "  [gdn: torch reference -- SLOW, install "
                          "flash-linear-attention + use bf16/fp16 CUDA]")
        vl = []
        if getattr(self, "has_mrope", False):
            vl.append("mrope")
        if getattr(self, "_deepstack", None):
            vl.append(f"deepstack after blocks {sorted(self._deepstack)}")
        if getattr(self, "_bidir_mm", False):
            vl.append("bidirectional image spans (eager attention on image batches)")
        if vl:
            line += "  [vision: " + ", ".join(vl) + "]"
        return line

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
        mm: Optional[dict] = None,
        mtp: bool = False,
    ):
        """Return final-norm hidden states ``[b, t, d]`` in fp32 -- the LM head's
        INPUT: on an arch with a head logit pre-scale (MuseGlimmer's
        ``output_multiplier``) that scalar is already folded in, so every head
        path can matmul this straight against the head weight. 1.0 everywhere
        else, so this is the plain final-norm output for every other base.

        ``seg_ids`` ([b, t]) enables sample packing: it marks each token's
        document within its packed sequence, so attention is restricted to the
        same document (flash-varlen via cu_seqlens on CUDA fp16/bf16; the
        block-diagonal eager/SDPA mask otherwise). ``position_ids`` must then be
        per-document resets (built by ``pack_examples`` / ``collate``).

        ``mm`` (``training.vision.build_mm_batch``) is the image splice for an
        image+text batch: the frozen vision features replace the token
        embeddings at the image positions (``index`` / ``embeds``), the
        deepstack maps are added after the blocks the arch marks
        (``deepstack``), and on a bidirectional-image arch the ``spans`` lift
        the causal mask inside each image. On an mRoPE tower ``position_ids``
        is then the 3-D ``[3, b, t]`` tensor (``vision.mrope_position_ids``).
        Not combinable with sample packing.

        ``mtp=True`` (an MTP head must be attached) ALSO runs the head on the
        trunk's output and returns ``(hidden, mtp_hidden)`` -- the head's own
        final-norm state, same shape/dtype contract, whose position ``i`` is the
        head's prediction of token ``i+1`` (see :meth:`_forward_mtp`). When
        nothing outside the head trains (``trunk_trainable()`` False) the trunk
        runs under ``no_grad``: no checkpointing, no saved activations -- it is
        a frozen feature extractor for the head."""
        if mtp and self.mtp is None:
            raise ValueError("forward(mtp=True) but no MTP head is attached")
        if mtp and mm is not None:
            raise ValueError("MTP head training is not supported on image batches (mm)")
        trunk_grad = torch.is_grad_enabled() and not (mtp and not self.trunk_trainable())
        with torch.set_grad_enabled(trunk_grad):
            hidden, ctx = self._forward_trunk(input_ids, attention_mask, position_ids,
                                              seg_ids, mm)
        if not mtp:
            return hidden
        return hidden, self._forward_mtp(input_ids, ctx)

    def _embed(self, input_ids: torch.Tensor, device) -> torch.Tensor:
        """The token embedding as the trunk's first block sees it -- the frozen
        table (or the trainable ``embed_weight``) with the module's multiplier /
        normalize applied, plus the embedding LoRA shift -- on ``device`` in the
        compute dtype. NOT including the image splice or the MuseGlimmer
        embed_norm (those follow in ``_forward_trunk``). Shared by the trunk and
        the MTP head, which borrows the trunk's embedding at inference."""
        if self.embed_weight is not None and not self._adapters_off:
            # Trainable input embedding: lookup against the fp32 master weight,
            # then the same multiplier/normalize the frozen path applies.
            ew = self.embed_weight
            looked_up = F.embedding(input_ids.to(ew.device), ew)
            hidden = backbone.embed_apply(self.embed, looked_up).to(device).to(self.compute_dtype)
        else:
            hidden = backbone.embed_tokens(self.embed, input_ids).to(device).to(self.compute_dtype)

        # Embedding LoRA: add a rank-r, token-indexed shift to the (frozen) embedding
        # output. Only the rows for tokens present in the batch get a gradient (the
        # F.embedding lookup is sparse over a), so this is cheap. Added after the
        # base embed scaling; the from-zero B absorbs any constant factor, so this is
        # as expressive as adding before the scale. requires_grad carries back to a/b
        # (so the grad-checkpoint detach below leaves the path intact).
        if self.embed_lora_a is not None and not self._adapters_off:
            ea = self.embed_lora_a.to(self.compute_dtype)
            eb = self.embed_lora_b.to(self.compute_dtype)
            delta = F.embedding(input_ids.to(ea.device), ea) @ eb       # [b, t, hid]
            hidden = hidden + self._module_lora_scale * delta.to(hidden.device).to(hidden.dtype)
        return hidden

    def _run_block(self, meta, entry, hidden, position_ids, dev, ctx, ckpt):
        """One decoder block (trunk or MTP head) on ``hidden``, dispatching on
        the block kind, with the attention backend / bias / packing descriptor
        drawn from the per-forward ``ctx`` (built by ``_forward_trunk``)."""
        if meta.get("kind", "attn") == "gdn":
            # GatedDeltaNet block: no RoPE, no attention bias, no packing
            # (packing is rejected up front -- the recurrence would carry
            # state across packed document boundaries).
            if ckpt:
                return torch.utils.checkpoint.checkpoint(
                    self._gdn_forward, meta, entry, hidden,
                    use_reentrant=False,
                )
            return self._gdn_forward(meta, entry, hidden)
        if meta.get("kind", "attn") == "shortconv":
            # ShortConv block (LFM2): causal conv, no RoPE, no
            # attention bias, no packing (rejected up front).
            if ckpt:
                return torch.utils.checkpoint.checkpoint(
                    self._shortconv_forward, meta, entry, hidden,
                    use_reentrant=False,
                )
            return self._shortconv_forward(meta, entry, hidden)
        mode = self._attn_mode_for(meta, ctx["mem_eff"])
        # eager: seg-aware additive bias. flash + sdpa both isolate
        # documents via pack_ctx (cu_seqlens for flash-varlen; per-
        # document SDPA loop for big-head), so neither builds the
        # [t, t] bias.
        attn_bias = ctx["get_bias"](meta["sliding_window"], dev) if mode == "eager" else None
        pack_ctx = ctx["pack_ctx"]
        pack = pack_ctx if (mode in ("flash", "sdpa") and pack_ctx is not None) else None
        if ckpt:
            return torch.utils.checkpoint.checkpoint(
                self._block_forward, meta, entry, hidden, position_ids, attn_bias,
                mode, pack, use_reentrant=False,
            )
        return self._block_forward(meta, entry, hidden, position_ids, attn_bias, mode, pack)

    def _forward_trunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
        mm: Optional[dict] = None,
    ):
        """The trunk half of :meth:`forward`: returns ``(hidden, ctx)`` where
        ``hidden`` is the (head-pre-scaled) final-norm state and ``ctx`` the
        per-forward attention context (positions, mask, packing descriptor,
        bias builder, the UNSCALED final-norm state) that the MTP head's
        forward consumes."""
        # Quant-aware noise tick: advance once per grad-enabled training
        # forward so each micro-batch draws fresh noise while the (up to 3)
        # weight reconstructions within its forward + backward all see the
        # same draw. No-grad forwards (eval, the DPO/KTO reference / KL
        # passes) do not advance it -- see quant_aware's determinism contract.
        qa_state = getattr(self, "_qa_state", None)
        if (qa_state is not None and self.training and torch.is_grad_enabled()
                and not self._adapters_off):
            qa_state["tick"] += 1
        if seg_ids is not None and (getattr(self, "has_gdn", False)
                                    or getattr(self, "has_shortconv", False)):
            raise ValueError(
                "sample packing (seg_ids) is not supported on GatedDeltaNet / "
                "ShortConv models: the recurrence and causal conv would carry "
                "state across packed document boundaries. Train unpacked "
                "(drop --pack).")
        if mm is not None and seg_ids is not None:
            raise ValueError("image batches (mm) cannot be sample-packed (seg_ids)")
        # Image batch: only counts when at least one image token is present.
        has_img = mm is not None and bool((mm["index"] >= 0).any())
        # Bidirectional image spans (Gemma4) need the explicit [t, t] mask, so
        # such a batch runs every attention block on the eager path.
        bidir = (has_img and getattr(self, "_bidir_mm", False)
                 and bool((mm["spans"] >= 0).any()))
        bsz, t = input_ids.shape
        # Under a layer-autosplit load each block lives on its own device; the
        # hidden state is migrated across boundaries below (mirroring exllamav3's
        # forward_ls). Start on the first block's device. The embedding is loaded
        # on CPU (prefer_cpu); backbone.embed_tokens runs the lookup there.
        first_device = self._block_devices[0]
        hidden = self._embed(input_ids, first_device)

        # Image splice: the frozen vision features take the image positions,
        # AFTER the text embedding scaling and adapters (they are never scaled
        # or adapted -- modules.Embedding inserts them raw after scaling the
        # standard rows) and BEFORE any embedding norm (module order).
        if has_img:
            mm = vision.move_mm(mm, first_device)
            hidden = vision.splice_embeddings(hidden, mm)

        # Norm on the embeddings (MuseGlimmer embed_norm), when the arch has one.
        # AFTER the embedding adapters, mirroring the module order: the adapter
        # replaces/shifts the Embedding's output, and the norm module then runs on
        # whatever the Embedding emitted.
        if self.embed_norm_spec is not None:
            hidden = self._norm(hidden, self.embed_norm_spec)

        if attention_mask is not None:
            attention_mask = attention_mask.to(first_device)
        if seg_ids is not None:
            seg_ids = seg_ids.to(first_device)
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.clamp_min(0)
            else:
                position_ids = torch.arange(t, device=first_device).unsqueeze(0).expand(bsz, t)
        elif position_ids.dim() == 3:
            assert position_ids.shape == (3, bsz, t), \
                f"3-D (mRoPE) position_ids must be [3, b, t], got {tuple(position_ids.shape)}"
        position_ids = position_ids.to(first_device)

        # Gradient checkpointing needs at least one input that requires grad. With
        # a frozen embedding the hidden state doesn't, so detach to a leaf and flag
        # it (no gradient is lost: the embedding is frozen). But when the embedding
        # is TRAINABLE, hidden already requires grad and carries the path back to
        # embed_weight -- detaching would sever it, so only detach when needed.
        # (Under no_grad -- eval, or a frozen trunk feeding the MTP head --
        # there is nothing to checkpoint: skip the recompute machinery.)
        ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        if ckpt and not hidden.requires_grad:
            hidden = hidden.detach().requires_grad_(True)

        # Memory-efficient attention is used only on CUDA in fp16/bf16 (the kernels
        # need both); fp32 / CPU / gradcheck fall back to eager. The mode is chosen
        # PER BLOCK because Gemma mixes head sizes: FA2 (head_dim <= 256) covers the
        # sliding/full layers, SDPA's mem-efficient backend covers head_dim > 256
        # full-causal layers at O(t) (per-document under packing for isolation),
        # and eager is the fallback. Only eager blocks build the [t, t] bias --
        # flash and sdpa both isolate documents via the packing descriptor instead.
        mem_eff = (self._flash_ok and hidden.is_cuda
                   and self.compute_dtype in (torch.float16, torch.bfloat16)
                   and not bidir)

        # Attention bias per (window, device). Built for eager blocks (always) and,
        # under packing, also for SDPA big-head blocks (which take an explicit
        # block-diagonal mask). Gemma alternates sliding/full layers, so each
        # distinct window needs its own mask; plain archs have a single window (-1).
        # When packing, the bias is seg-aware (block-diagonal). Cached per key.
        bias_cache: dict = {}

        def get_bias(window, dev):
            key = (int(window), str(dev))
            b = bias_cache.get(key)
            if b is None:
                am = attention_mask.to(dev) if attention_mask is not None else None
                sg = seg_ids.to(dev) if seg_ids is not None else None
                sp = mm["spans"].to(dev) if bidir else None
                b = self._attn_bias(am, t, dev, torch.float32, window, seg_ids=sg,
                                    bidir_spans=sp)
                bias_cache[key] = b
            return b

        # Sample-packing context for the flash-varlen path: flatten the batch's
        # non-pad tokens into one stream and mark per-document boundaries with
        # cu_seqlens, so flash isolates documents with NO [t,t] mask. Built once
        # (constant across layers); only needed when flash actually runs (CUDA
        # fp16/bf16). On CPU/fp32 the eager block-diagonal bias handles packing.
        pack_ctx = self._build_pack_context(seg_ids, attention_mask, bsz, t,
                                            first_device) if (seg_ids is not None
                                                              and mem_eff) else None
        # The per-forward attention context, shared with the MTP head's blocks
        # (same mask, positions, packing and backend selection as the trunk's).
        ctx = {"mem_eff": mem_eff, "get_bias": get_bias, "pack_ctx": pack_ctx,
               "attention_mask": attention_mask, "seg_ids": seg_ids}

        # Activation offload: park the (grad-checkpointed) block-boundary activations
        # saved for backward in CPU RAM, trading CPU<->GPU copies for GPU memory.
        # Only meaningful with checkpointing on CUDA; wraps just the block loop (the
        # final norm/head stay GPU-resident). "async" (default) double-buffers the
        # copies on a side stream so they overlap compute (training.offload);
        # "sync" is torch's blocking save_on_cpu, kept for A/B + retain_graph.
        offload = (self.offload_activations and ckpt and "cuda" in str(first_device))
        if not offload:
            save_ctx = contextlib.nullcontext()
        elif self.offload_mode == "async":
            if self._offloader is None:
                from .offload import AsyncActivationOffload
                self._offloader = AsyncActivationOffload()
            save_ctx = self._offloader
        else:
            save_ctx = torch.autograd.graph.save_on_cpu(pin_memory=True)
        cur_device = first_device
        with save_ctx:
            for bi, (meta, entry, dev) in enumerate(
                    zip(self._block_meta, self.blocks, self._block_devices)):
                # Cross the device boundary if this block sits on another card. All
                # no-ops under a single-device load (dev == cur_device throughout).
                if dev != cur_device:
                    hidden = backbone.to_device(hidden, dev)
                    position_ids = backbone.to_device(position_ids, dev)
                    cur_device = dev
                hidden = self._run_block(meta, entry, hidden, position_ids, dev, ctx, ckpt)
                # Deepstack (Qwen3-VL / Qwen3.5-VL): the vision tower's
                # intermediate feature map for this depth is added onto the
                # image positions after the block -- the DeepstackEmbed module
                # that follows this block in the inference module list. No-op
                # on a text batch.
                if has_img and bi in self._deepstack:
                    hidden = vision.add_deepstack(hidden, mm, self._deepstack[bi])
                # EBFT feature taps: stash the residual stream after selected
                # blocks (see collect_hidden). The stream is the raw pre-norm
                # residual, matching HF's output_hidden_states[bi+1] convention
                # that the EBFT reference implementation reads its features from.
                if self._collect_spec is not None:
                    want, pos = self._collect_spec
                    if bi in want:
                        h = hidden.detach().float()
                        if pos is not None:
                            p = pos.to(h.device).unsqueeze(-1).expand(-1, -1, h.shape[-1])
                            h = h.gather(1, p)
                        self.collected[bi] = h

        # Final norm + head live on the output device (the last split device).
        # Return fp32 hidden so the head / cross-entropy path keeps its existing
        # dtype contract (the per-block residual stream above is bf16, but this is a
        # single [b, t, d] tensor, so the fp32 cast here is negligible).
        hidden = backbone.to_device(hidden, self.device)
        hidden = self._norm(hidden, self.final_norm_spec).float()
        # The raw final-norm state is what the MTP head consumes (inference
        # exports exactly this module's output as target_hidden), BEFORE the
        # head pre-scale below.
        ctx["trunk_state"] = hidden
        ctx["position_ids"] = position_ids
        # Head logit pre-scale (MuseGlimmer output_multiplier), folded into the
        # head INPUT: inference computes `logits = (head(x) [+ lora]) * pre_scale`
        # before the softcap, and the native head is a bias-free `x @ W` (+ a
        # low-rank delta, also linear in x), so scaling x here is exactly the same
        # logits. Done at the single point every head path reads -- materialized
        # logits(), the trainable/LoRA head, both fused-CE heads, and the EBFT
        # sampler all take their hidden state from this return. 1.0 -> no-op.
        if self.head_pre_scale != 1.0:
            hidden = hidden * self.head_pre_scale
        return hidden, ctx

    def _forward_mtp(self, input_ids: torch.Tensor, ctx: dict) -> torch.Tensor:
        """The MTP head's differentiable forward, reproducing the inference
        draft path (``Qwen3_5MTPInputLayer.forward`` + the generator's
        prefill wiring in ``job.py``) over a whole sequence at once:

        * position ``i`` pairs token ``i``'s embedding with the trunk's
          final-norm state at ``i-1`` -- the state that PRODUCED token ``i`` --
          and predicts token ``i+1``; the generator feeds exactly this
          ``cat(carry, target_hidden[:, :-1])`` shift, with a zero carry at the
          very first position (and here at every packed document start);
        * each half is RMS-normed by its own pre-fc norm, the two are
          concatenated ``[embedding | state]`` and projected by ``fc``
          (``2d -> d``), which is the head's residual-stream input;
        * the head's block(s) run at the trunk's position ids (the draft cache
          continues the trunk's positions) with the same mask / packing, then
          its own final norm. The trunk's LM head (borrowed at inference via
          ``attach_to``) turns the result into logits, so the returned state
          follows :meth:`forward`'s contract (fp32, head pre-scale folded in)
          and every head-loss path applies unchanged with ``shift=True``.

        The embedding is the trunk's, adapters included (``_embed``): at
        inference the head reads the attached model's embedding module."""
        dev = self._mtp_device
        state = ctx["trunk_state"]                                  # [b, t, d] fp32
        b, t, d = state.shape
        h = backbone.to_device(state, dev).to(self.compute_dtype)
        prev = torch.cat([h.new_zeros(b, 1, d), h[:, :-1]], dim=1)  # state i-1 at i
        seg_ids = ctx["seg_ids"]
        if seg_ids is not None:
            # Packed documents: each starts from a zero carry like a fresh
            # sequence, never from the previous document's last state.
            seg = seg_ids.to(dev)
            start = torch.ones(b, t, dtype=torch.bool, device=dev)
            start[:, 1:] = seg[:, 1:] != seg[:, :-1]
            prev = prev.masked_fill(start.unsqueeze(-1), 0)
        y = self._norm(prev, self.mtp.norm_hidden_spec)
        e = self._norm(self._embed(input_ids, dev), self.mtp.norm_embed_spec)
        x = self.mtp.fc(torch.cat([e, y], dim=-1))
        # Same checkpointing contract as the trunk: the checkpointed block
        # needs an input that requires grad; with a frozen trunk + frozen fc
        # nothing upstream does, so detach to a leaf (no gradient is lost).
        ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        if ckpt and not x.requires_grad:
            x = x.detach().requires_grad_(True)
        position_ids = backbone.to_device(ctx["position_ids"], dev)
        for meta, entry in zip(self._mtp_block_meta, self.mtp.blocks):
            x = self._run_block(meta, entry, x, position_ids, dev, ctx, ckpt)
        x = self._norm(x, self.mtp.final_norm_spec).float()
        if self.head_pre_scale != 1.0:
            x = x * self.head_pre_scale
        return x

    # --- heads -------------------------------------------------------------

    def lm_head_weight_fn(self) -> Callable[[], torch.Tensor]:
        """Frozen LM-head weight closure in ``[hidden, vocab]`` orientation."""
        return backbone.head_weight_closure(self.lm_head)

    def logits(self, input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               position_ids: Optional[torch.Tensor] = None,
               seg_ids: Optional[torch.Tensor] = None,
               mm: Optional[dict] = None) -> torch.Tensor:
        """Materialize full logits ``[b, t, vocab]`` (validation / small batches)."""
        hidden = self.forward(input_ids, attention_mask, position_ids, seg_ids, mm=mm)
        return self._head_logits(hidden)

    def mtp_logits(self, input_ids: torch.Tensor,
                   attention_mask: Optional[torch.Tensor] = None,
                   position_ids: Optional[torch.Tensor] = None,
                   seg_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Materialize the MTP head's logits ``[b, t, vocab]`` through the
        (borrowed) LM head: position ``i`` is the head's draft for token
        ``i+1`` -- the validate gate compares this against the inference draft
        forward. Frozen head only (the LoRA/trainable-head deltas are a
        training-loss concern; see ``_head_loss``)."""
        _, mtp_hidden = self.forward(input_ids, attention_mask, position_ids, seg_ids,
                                     mtp=True)
        return self._head_logits(mtp_hidden)

    def _head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        w = self.lm_head_weight_fn()()
        logits = backbone.to_device(hidden, w.device).to(w.dtype) @ w
        if self.final_softcap:
            logits = self.final_softcap * torch.tanh(logits / self.final_softcap)
        return logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk: int = DEFAULT_CHUNK,
        ignore_index: int = IGNORE_INDEX,
        position_ids: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
        mm: Optional[dict] = None,
    ) -> torch.Tensor:
        """Shifted causal-LM cross-entropy.

        Frozen head: the streaming fused head (never materializes ``[tokens,
        vocab]``), which also applies any final-logit softcap (Gemma2) in-tile.
        Trainable head (``train_head``/``lora_head``): standard autograd so the
        head weight gets a gradient -- but logits are computed only at the
        *supervised* positions (labels != ignore_index), so memory scales with
        the number of supervised tokens, not the full sequence.

        ``position_ids`` / ``seg_ids`` enable sample packing (per-document position
        resets + block-diagonal attention). The shifted CE needs no packing special
        case: at a document boundary the shift predicts the next document's first
        token, which is always a masked (-100) prompt token, so it contributes no
        loss. ``mm`` is the image splice (see :meth:`forward`).

        With an MTP head attached the head's next-token loss (its state at
        ``i`` vs label ``i+1`` -- the same shifted CE through the same LM head)
        is added: ``trunk + mtp_loss_weight * mtp`` while the trunk trains,
        the head's loss ALONE when it doesn't (``trunk_trainable()`` False --
        the trunk's loss is then a constant, and skipping it saves a head
        pass). The components land in ``last_losses`` for logging."""
        if self.mtp is None:
            hidden = self.forward(input_ids, attention_mask, position_ids, seg_ids, mm=mm)
            return self._head_loss(hidden, labels, chunk, ignore_index)
        hidden, mtp_hidden = self.forward(input_ids, attention_mask, position_ids,
                                          seg_ids, mm=mm, mtp=True)
        mtp_loss = self._head_loss(mtp_hidden, labels, chunk, ignore_index)
        if not self.trunk_trainable():
            self.last_losses = {"mtp": float(mtp_loss.detach())}
            return mtp_loss
        trunk_loss = self._head_loss(hidden, labels, chunk, ignore_index)
        self.last_losses = {"trunk": float(trunk_loss.detach()),
                            "mtp": float(mtp_loss.detach())}
        return trunk_loss + self.mtp_loss_weight * mtp_loss.to(trunk_loss.device)

    def _head_loss(self, hidden: torch.Tensor, labels: torch.Tensor,
                   chunk: int = DEFAULT_CHUNK,
                   ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
        """Shifted CE of a final-norm state ``[b, t, d]`` through the LM head
        (frozen fused / chunked heads, or the trainable/LoRA head) -- the
        trunk's state or the MTP head's, which share the head."""
        # Materialize logits only at supervised positions when the head trains
        # (train_head OR lora_head) -- the fused heads are frozen-head only (no
        # weight/LoRA gradient). Memory scales with the supervised token count.
        # A frozen softcapped head (Gemma) stays on the fused paths below, which
        # apply the tanh cap in-tile. Under adapters_disabled() the trainable/
        # LoRA head is part of the adapter, so the frozen fused path is used.
        if (self.train_head or self.lora_head) and not self._adapters_off:
            d = hidden.shape[-1]
            hs = hidden[:, :-1, :].reshape(-1, d)                 # shift
            lbl = labels[:, 1:].reshape(-1).to(hs.device)
            valid = lbl != ignore_index
            # train_head -> the trainable full head; otherwise the frozen head weight
            # (a lora_head delta is added to its logits below).
            w = self.head_weight if self.train_head else self.lm_head_weight_fn()()  # [d, V]
            # The frozen base loads under inference_mode, so its head weight is an
            # inference tensor. hs requires grad, so `hs @ w` would save w for backward
            # -- forbidden for inference tensors. Clone to a normal tensor when the head
            # is frozen (train_head already owns a normal Parameter). Negligible: a
            # [d, V] clone once per step. Same fix as the Liger RMSNorm weight (#106).
            if torch.is_inference(w):
                w = w.clone()
            if not bool(valid.any()):
                # No supervised tokens: keep the graph alive with a 0 grad for every
                # trainable surface touched here (full head and/or head-LoRA a/b).
                z = hs.sum() * 0.0
                if self.train_head:
                    z = z + w.sum() * 0.0
                if self.lora_head:
                    z = z + self.head_lora_a.sum() * 0.0 + self.head_lora_b.sum() * 0.0
                return z
            hs = backbone.to_device(hs[valid], w.device).to(w.dtype)
            # Keep logits in the head's dtype: an extra fp32 copy of [supervised
            # tokens, vocab] doubles the single biggest allocation on the head device
            # (~6.7 GB on a 262k-vocab base) and OOMs. The matmul already accumulates
            # in fp32 internally; F.cross_entropy is stable on the resulting logits.
            logits = hs @ w
            if self.lora_head:
                # Low-rank head shift: logits += scale * (hs @ A) @ B.
                # a/b are fp32 masters (like every other LoRA here) but hs was cast
                # to w.dtype above, so they must be cast to match or the matmul
                # raises. Cast to hs.dtype, NOT compute_dtype (the lora_embed path's
                # convention): the delta is added to `logits`, which is w.dtype --
                # an exl3 head reconstructs fp16 even in a bf16 run, and a bf16 delta
                # would type-promote that add to fp32, i.e. a [supervised_tok, vocab]
                # fp32 buffer (~2 GB at 202k vocab) on the head device.
                ha = self.head_lora_a.to(hs.dtype)
                hb = self.head_lora_b.to(hs.dtype)
                logits = logits + self._module_lora_scale * ((hs @ ha) @ hb)
            if self.final_softcap:
                # In place: the out-of-place chain (`cap * tanh(logits / cap)`)
                # holds three [supervised, vocab] buffers at its peak; div_/tanh_
                # reuse the logits buffer so the peak is two. Safe: matmul/add
                # save their inputs, not outputs; div-by-scalar saves nothing;
                # tanh_'s backward uses its own output, which the final
                # out-of-place scalar mul leaves unmodified.
                logits = logits.div_(self.final_softcap).tanh_() * self.final_softcap
            return F.cross_entropy(logits, lbl[valid].to(w.device))
        # The fused head matmuls hidden against the frozen head weight, which
        # lives on the head's device; co-locate them (no-op single-device).
        hidden = backbone.to_device(hidden, self._head_device)
        labels = labels.to(hidden.device)
        # Chunked-vocab head: reconstruct + matmul the head in vocab tiles so the
        # full [hidden, vocab] weight is never built at once (frees the output card
        # for big vocabs). Falls through to the single-shot fused head when off or
        # when the head can't slice.
        if self.head_vocab_chunk > 0 and self._head_slice is not None:
            slice_fn, vocab_size, granularity = self._head_slice
            return fused_linear_cross_entropy_vocab_chunked(
                hidden, slice_fn, labels, vocab_size,
                vocab_chunk=self.head_vocab_chunk, token_chunk=chunk,
                ignore_index=ignore_index, granularity=granularity, shift=True,
                softcap=self.final_softcap,
            )
        return fused_linear_cross_entropy(
            hidden, self.lm_head_weight_fn(), labels,
            chunk=chunk, ignore_index=ignore_index, shift=True,
            softcap=self.final_softcap,
        )

    # --- EBFT feature taps ---------------------------------------------------

    def feature_block_indices(self, fracs=(0.25, 0.50, 0.75)) -> list[int]:
        """0-based block indices for EBFT feature extraction at fractional
        depths. Matches the reference implementation's HF-convention pick
        ``hidden_states[clamp(floor(L*frac), 1, L)]`` where ``hidden_states[k]``
        is the residual stream after block ``k-1`` -- hence the ``-1`` here."""
        L = len(self.blocks)
        idxs = sorted({max(1, min(L, math.floor(L * f))) - 1 for f in fracs})
        return idxs

    @contextlib.contextmanager
    def collect_hidden(self, block_indices, positions=None):
        """Collect the residual stream after selected blocks during forwards
        inside this context (the EBFT feature-network taps).

        ``block_indices``: 0-based indices into ``self.blocks``; index ``i``
        stores the residual stream AFTER block ``i`` (HF's
        ``output_hidden_states[i+1]``, pre final-norm). ``positions``: optional
        ``[b, P]`` long tensor of token positions to gather per row, keeping
        the stored tensors at ``[b, P, d]`` instead of the full ``[b, t, d]``
        stream (the full stream at fp32 is hundreds of MB per tap on real
        batches -- always pass positions in training loops).

        Collected tensors are detached fp32, left on their block's device, in
        ``self.collected`` (dict block index -> tensor), overwritten by each
        forward inside the context. Not reentrant (one active spec at a time),
        and composes with ``adapters_disabled()`` for the frozen feature
        network phi."""
        assert self._collect_spec is None, "collect_hidden is not reentrant"
        self._collect_spec = (set(int(i) for i in block_indices), positions)
        self.collected: dict[int, torch.Tensor] = {}
        try:
            yield self
        finally:
            self._collect_spec = None

    # --- preference training (DPO / KTO) ------------------------------------

    @contextlib.contextmanager
    def adapters_disabled(self):
        """Run the forward as the PURE frozen base model (the reference model
        for DPO/KTO), the PEFT disable-adapter trick -- no second model copy:

          * every per-linear LoRA term is skipped AND any pissa offset with it
            (they cancel at init, so dropping both is exactly the base ``W_q``);
          * the trainable / LoRA embedding and LM head are ignored.

        For ``init_lora`` default/eva/pissa the reference equals the step-0
        policy exactly. For ``qerr`` the step-0 policy is the error-repaired
        model while the reference is the raw quantized base, so rewards start
        slightly nonzero -- the trainer prints a note.

        Restores the previous state on exit (exception-safe); reentrant."""
        prev_net = self._adapters_off
        prev = [w.adapter_enabled for w in self._wrappers]
        self._adapters_off = True
        for w in self._wrappers:
            w.adapter_enabled = False
        try:
            yield self
        finally:
            self._adapters_off = prev_net
            for w, p in zip(self._wrappers, prev):
                w.adapter_enabled = p

    def compute_logps(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk: int = DEFAULT_CHUNK,
        ignore_index: int = IGNORE_INDEX,
        position_ids: Optional[torch.Tensor] = None,
        mm: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sequence summed completion log-probabilities (DPO / KTO).

        Returns ``(logps, counts)``: ``logps[b]`` = sum over the supervised
        (shifted-label != ignore_index) positions of ``log p(label | prefix)``,
        fp32, differentiable; ``counts[b]`` = the number of supervised tokens
        (for length-normalized variants like IPO). Streams through the fused
        heads with ``reduction="none"``, so the ``[tokens, vocab]`` logits are
        never materialized -- same memory story as :meth:`compute_loss`.

        Matches TRL's convention (sum over completion tokens, prompt masked).
        One sequence per batch ROW: sample packing is not supported here (a
        packed block would sum across its documents), so no ``seg_ids``.
        The trainable/LoRA head paths are not wired (preference training runs
        frozen-head); use plain LoRA targets.
        """
        assert not ((self.train_head or self.lora_head)
                    and not self._adapters_off), \
            "compute_logps supports the frozen LM head only (no --train-head/--lora-head)"
        hidden = self.forward(input_ids, attention_mask, position_ids, mm=mm)
        hidden = backbone.to_device(hidden, self._head_device)
        labels = labels.to(hidden.device)
        if self.head_vocab_chunk > 0 and self._head_slice is not None:
            slice_fn, vocab_size, granularity = self._head_slice
            nll = fused_linear_cross_entropy_vocab_chunked(
                hidden, slice_fn, labels, vocab_size,
                vocab_chunk=self.head_vocab_chunk, token_chunk=chunk,
                ignore_index=ignore_index, granularity=granularity, shift=True,
                softcap=self.final_softcap, reduction="none",
            )
        else:
            nll = fused_linear_cross_entropy(
                hidden, self.lm_head_weight_fn(), labels,
                chunk=chunk, ignore_index=ignore_index, shift=True,
                softcap=self.final_softcap, reduction="none",
            )
        b, t = labels.shape
        # Ignored positions carry an exact 0 in the per-token NLL, so the row
        # sum is the completion logprob with the prompt contributing nothing.
        logps = -nll.view(b, t - 1).float().sum(dim=-1)
        counts = (labels[:, 1:] != ignore_index).sum(dim=-1)
        return logps, counts

    # --- adapter parameters / IO ------------------------------------------

    @torch.no_grad()
    def apply_init_lora(self, mode: str, ref_model_dir: Optional[str] = None,
                        svd_niter: int = 16, verbose: bool = True,
                        eva_batches=None) -> None:
        """
        Replace the default (kaiming/zeros) adapter init with an SVD-based one:
        ``"pissa"`` (principal components of the frozen base, trained against a
        frozen-offset residual), ``"qerr"`` (top-r of the quantization error
        vs the original model at ``ref_model_dir``) or ``"eva"`` (top-r
        right-singular vectors of each target's input activations, streamed
        from ``eva_batches`` -- an iterable of :meth:`forward` kwargs dicts of
        real training data). See ``training.lora_init``.
        Call after construction and BEFORE ``load_adapter`` -- a resume restores
        the exact saved tensors (including pissa offsets) over this.
        """
        from .lora_init import apply_init_lora as _apply
        _apply(self, mode, ref_model_dir=ref_model_dir,
               svd_niter=svd_niter, verbose=verbose, eva_batches=eva_batches)

    def set_quant_aware(self, mode: str, scale: float = 1.0,
                        ref_model_dir: Optional[str] = None, seed: int = 0,
                        verbose: bool = True) -> None:
        """
        Enable a quantization-aware training mode on the adapted linears:
        ``"noise"`` (fresh per-micro-batch pseudo-quantization noise on the
        frozen weight) or ``"ste"`` (the effective adapter delta snapped to a
        quant-floor grid with a straight-through gradient); ``"none"``
        disables. σ per layer is measured against ``ref_model_dir`` (the
        original unquantized HF dir) when given, else estimated from the
        trellis bitrate. Run configuration, not learned state -- call after
        construction / resume each run (nothing is persisted). See
        ``training.quant_aware`` for the design and determinism contract.
        """
        _qa.configure_quant_aware(self, mode, scale=scale,
                                  ref_model_dir=ref_model_dir, seed=seed,
                                  verbose=verbose)
        self.quant_aware = mode if mode not in (None, "") else "none"
        self.quant_aware_scale = float(scale)

    # Every accessor below returns only params that still TRAIN: freeze_trunk
    # (MTP stage 2) turns requires_grad off on the trunk side while the
    # wrappers keep their (resumed) values, and the optimizer / grad clip /
    # trainable count must not see those. Without freeze_trunk nothing is
    # filtered (every adapter param is created with requires_grad=True).
    def lora_parameters(self) -> list[nn.Parameter]:
        ps: list[nn.Parameter] = []
        for w in self._wrappers:
            if w.r > 0:
                ps += [w.lora_a, w.lora_b]
        # Embed/head LoRA adapters are small and GPU-resident (never offloaded), so
        # they ride with the per-linear LoRA: optimized by the main optimizer and
        # included in the grad clip.
        ps += self.module_lora_parameters()
        return [p for p in ps if p.requires_grad]

    def module_lora_parameters(self) -> list[nn.Parameter]:
        """LoRA adapters on the embedding / LM head (lora_embed / lora_head), if any.
        Read via getattr so a headless instance (no __init__) returns []."""
        ps: list[nn.Parameter] = []
        if getattr(self, "embed_lora_a", None) is not None:
            ps += [self.embed_lora_a, self.embed_lora_b]
        if getattr(self, "head_lora_a", None) is not None:
            ps += [self.head_lora_a, self.head_lora_b]
        return [p for p in ps if p.requires_grad]

    def modules_to_save_parameters(self) -> list[nn.Parameter]:
        """Trainable full embed/head params (PEFT ``modules_to_save``), if any.
        For a tied model this is the single shared embedding weight."""
        ps: list[nn.Parameter] = []
        if self.embed_weight is not None:
            ps.append(self.embed_weight)
        if self.head_weight is not None:
            ps.append(self.head_weight)
        return [p for p in ps if p.requires_grad]

    def mtp_parameters(self) -> list[nn.Parameter]:
        """The MTP head's adapter params (empty without a head)."""
        ps: list[nn.Parameter] = []
        for w in getattr(self, "_mtp_wrappers", []):
            if w.r > 0:
                ps += [w.lora_a, w.lora_b]
        return ps

    def trainable_parameters(self) -> list[nn.Parameter]:
        """All trainable params: LoRA adapters + any full-trained embed/head."""
        return self.lora_parameters() + self.modules_to_save_parameters()

    def lora_param_groups(self, weight_decay: float, base_lr: float | None = None,
                          module_lora_lr_mul: float = 1.0) -> list[dict]:
        """LoRA-only param groups. With ``module_lora_lr_mul != 1`` the embed/head
        LoRA adapters get their OWN group at ``base_lr * module_lora_lr_mul``
        (S64: at the shared LR the head LoRA lands ~8x short of its optimum on
        its own descent direction). The per-linear group stays group 0, whose LR
        the log line and the offload-optimizer mirror read."""
        per_linear: list[nn.Parameter] = []
        for w in self._wrappers:
            if w.r > 0 and w.lora_a.requires_grad:
                per_linear += [w.lora_a, w.lora_b]
        mod = self.module_lora_parameters()
        if not mod or module_lora_lr_mul == 1.0:
            return [{"params": per_linear + mod, "weight_decay": weight_decay}]
        assert base_lr is not None, "module_lora_lr_mul != 1 needs base_lr"
        return [{"params": per_linear, "weight_decay": weight_decay},
                {"params": mod, "weight_decay": weight_decay,
                 "lr": base_lr * module_lora_lr_mul}]

    def param_groups(self, weight_decay: float, base_lr: float | None = None,
                     module_lora_lr_mul: float = 1.0) -> list[dict]:
        """Optimizer param groups: weight decay on the LoRA params, but NONE on
        the full embed/head (weight-decaying a whole embedding table is harmful)."""
        groups = self.lora_param_groups(weight_decay, base_lr, module_lora_lr_mul)
        ms = self.modules_to_save_parameters()
        if ms:
            groups.append({"params": ms, "weight_decay": 0.0})
        return groups

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    @torch.no_grad()
    def apply_to_native(self, scaling: float = 1.0) -> None:
        """
        Push the *current* adapter weights into the underlying native ``Linear``
        modules' runtime LoRA slots, so that ``model.forward`` / generation
        reflects the adapter. Independent of the training forward (which reads
        only the frozen base weight), so this is safe to call mid-training to
        sample progress. Call :meth:`remove_from_native` to revert to base.

        The native ``Linear.apply_lora`` computes ``x @ A @ B`` with no extra
        scale, so the LoRA scale is folded into B here.

        MoE caveat: runtime LoRA fires only through ``Linear.forward``, and the
        quantized ``BlockSparseMLP`` inference forward dispatches its routed
        experts through fused kernels (bc / mgemm / exl3_moe) that bypass it --
        so ROUTED-expert adapters will NOT be reflected in native generation
        (live samples / qlora_infer_native). They still train correctly; deploy
        them by merge-and-requantize (the recommended path per the handoff's
        finding C anyway). A one-time warning is printed when that applies.
        """
        if not getattr(self, "_warned_moe_apply", False) and any(
                w.r > 0 and ".experts." in w.key for w in self._wrappers):
            self._warned_moe_apply = True
            print(" -- warning: routed-expert LoRA adapters are installed in the "
                  "runtime slots, but the fused MoE inference kernels bypass "
                  "them: native generation will NOT reflect the expert "
                  "adapters. Merge-and-requantize to deploy them.")
        for w in self._wrappers:
            if w.r <= 0:
                continue
            a = w.lora_a.detach()
            b = w.lora_b.detach() * (w.scale * scaling)
            if w.init_a0 is not None:
                # PiSSA: the effective delta is s·(AB - A0B0); the runtime slot
                # gets the rank-2r concatenation so generation matches training.
                # Concatenate from the exact fp32 masters, not the compute-dtype
                # device copies: A ≈ A0 early in training, and a bf16-rounded A0
                # against an fp32 A leaves a spurious residual comparable to the
                # trained delta itself.
                a = torch.cat([a, w.init_a0_master.to(a.device)], dim=1)
                b = torch.cat([b, w.init_b0_master.to(b.device)
                               * (-w.scale * scaling)], dim=0)
            backbone.set_runtime_lora(w.linear, self, a.to(torch.float16),
                                      b.to(torch.float16))

    @torch.no_grad()
    def remove_from_native(self) -> None:
        """Remove this adapter from the native modules' runtime LoRA slots."""
        for w in self._wrappers:
            if w.r <= 0:
                continue
            backbone.clear_runtime_lora(w.linear, self)

    # --- idle offload (serve-only VRAM squeeze) -----------------------------

    # Class default (same pattern as the other non-parameter flags): a net that
    # never offloads -- including headless test instances built via __new__ --
    # is simply "not offloaded".
    _training_state_homes = None

    @torch.no_grad()
    def offload_training_state(self) -> bool:
        """
        Move every tensor that exists only for TRAINING -- the fp32 LoRA /
        embed / head masters (plus their gradients, if any are live) and the
        compute-dtype PiSSA offset copies -- to system memory, remembering each
        tensor's home device so :meth:`restore_training_state` can undo it
        exactly. Value-exact: these are device moves, not casts.

        The inference side is untouched: generation reads the frozen base
        weights and the fp16 runtime LoRA slots installed by
        :meth:`apply_to_native`, which are separate tensors (and
        ``apply_to_native`` / ``save_adapter`` still work while offloaded --
        both normalize devices themselves). The training forward is what
        becomes unavailable until restore.

        Parameter *identity* is preserved (only ``.data`` moves), so optimizer
        ``param_groups`` built from these params stay valid; the caller owns
        moving any optimizer state (see ``RealtimeQLoRA``).

        Returns True if anything actually moved. Idempotent: a second call
        while offloaded is a no-op returning False.
        """
        if self._training_state_homes is not None:
            return False
        homes: list = []
        for p in self.trainable_parameters():
            if p.device.type == "cpu":
                continue
            homes.append(("param", p, p.device))
            p.data = p.data.to("cpu")
            if p.grad is not None:
                p.grad = p.grad.to("cpu")
        for w in self._wrappers:
            for name in ("init_a0", "init_b0"):
                t = getattr(w, name, None)
                if t is not None and t.device.type != "cpu":
                    homes.append(("buffer", (w, name), t.device))
                    setattr(w, name, t.to("cpu"))
        self._training_state_homes = homes
        return bool(homes)

    @torch.no_grad()
    def restore_training_state(self) -> None:
        """Undo :meth:`offload_training_state`: move everything back to its
        recorded home device. No-op if not currently offloaded."""
        homes, self._training_state_homes = self._training_state_homes, None
        if not homes:
            return
        for kind, ref, dev in homes:
            if kind == "param":
                ref.data = ref.data.to(dev)
                if ref.grad is not None:
                    ref.grad = ref.grad.to(dev)
            else:
                w, name = ref
                setattr(w, name, getattr(w, name).to(dev))

    def save_adapter(self, directory: str,
                     base_model_name_or_path: Optional[str] = None) -> None:
        """
        Write the trained adapters in PEFT format, keyed by the native module
        path so they load with both PEFT and exllamav3's ``LoRA.from_directory``
        (and hence ``training/qlora_infer_native.py``).

        Internal tensors are ``a=[in, r]`` / ``b=[r, out]``; PEFT stores
        ``lora_A=[r, in]`` / ``lora_B=[out, r]``, so we transpose on save and
        emit the *unscaled* B (the loader reapplies alpha/r).
        """
        from safetensors.torch import save_file
        os.makedirs(directory, exist_ok=True)

        # PiSSA (init_lora="pissa") trains against a frozen-offset residual base;
        # its true delta is s·(AB - A0B0). Exported adapters must be correct for
        # ANY consumer (PEFT, LoRA.from_directory, merge scripts), so the main
        # adapter_model.safetensors gets the rank-2r standard-LoRA conversion
        # [A | A0] / [s·B ; -s·B0] with alpha'=2r (loader scale 1.0), and the
        # exact fp32 training state (A/B/A0/B0) goes to a pissa_init.safetensors
        # sidecar that load_adapter prefers on resume.
        offsets = [w.init_a0 is not None for w in self._wrappers if w.r > 0]
        has_offset = any(offsets)
        assert not has_offset or all(offsets), \
            "mixed pissa/non-pissa wrappers can't share one adapter export"

        state: dict[str, torch.Tensor] = {}
        sidecar: dict[str, torch.Tensor] = {}
        target_leaves: set[str] = set()
        r = alpha = None
        expert_ranks: set[int] = set()   # routed-expert ranks (may differ: expert_r)
        use_rslora = self.use_rslora
        for w in self._wrappers:
            if w.r <= 0:
                continue
            if ".experts." in w.key:
                expert_ranks.add(w.r)
            else:
                r, alpha = w.r, w.lora_alpha
            target_leaves.add(w.key.split(".")[-1])
            key = f"base_model.model.{w.key}"
            if has_offset:
                # Convert on CPU from the exact fp32 masters (the on-device
                # offsets are compute-dtype; rounding them against the fp32
                # adapter would corrupt the exported delta).
                a_cpu = w.lora_a.detach().float().cpu()
                b_cpu = w.lora_b.detach().float().cpu()
                a = torch.cat([a_cpu, w.init_a0_master], dim=1)             # [in, 2r]
                b = torch.cat([b_cpu * w.scale,
                               w.init_b0_master * (-w.scale)], dim=0)       # [2r, out]
                sidecar[f"{w.key}.lora_a"] = a_cpu
                sidecar[f"{w.key}.lora_b"] = b_cpu
                sidecar[f"{w.key}.init_a0"] = w.init_a0_master
                sidecar[f"{w.key}.init_b0"] = w.init_b0_master
            else:
                a = w.lora_a.detach()
                b = w.lora_b.detach()
            state[f"{key}.lora_A.weight"] = a.t().contiguous().to(torch.float16).cpu()
            state[f"{key}.lora_B.weight"] = b.t().contiguous().to(torch.float16).cpu()

        if r is None:
            # No non-expert per-linear LoRA (e.g. only --lora-embed/--lora-head
            # or only expert_* targets requested); the saved config still needs
            # a rank/alpha -- use the configured ones.
            r, alpha = self.r, self.lora_alpha
        init_lora_r = r
        # Routed experts trained at a different rank (expert_r): record a PEFT
        # rank_pattern so external consumers scale their B by alpha/expert_r,
        # not alpha/r. (LoRA.from_directory derives each module's rank from the
        # tensor shapes and needs no pattern.)
        rank_pattern = {}
        alpha_pattern = {}
        odd_expert_ranks = expert_ranks - {r}
        if odd_expert_ranks:
            assert len(odd_expert_ranks) == 1, \
                f"mixed routed-expert ranks {sorted(expert_ranks)} can't be " \
                f"described by one rank_pattern"
            er = next(iter(odd_expert_ranks))
            # Match routed-expert modules on any arch: Qwen keys them under
            # ...mlp.experts.<i>..., Gemma4 under ...layers.<n>.experts.<i>...
            # (no .mlp segment). The \d+ segment keeps Qwen's shared_expert
            # (and any other "expert"-named sibling) out of the pattern.
            pat = r".*\.experts\.\d+\..*"
            rank_pattern = {pat: 2 * er if has_offset else er}
            if has_offset:
                # The pissa conversion folds the scale into B (loader scale must
                # be 1.0), so the pattern-matched modules need alpha == rank too.
                alpha_pattern = {pat: float(2 * er)}
        if has_offset:
            # The exported tensors are rank 2r with the scale folded in.
            r, alpha, use_rslora = 2 * r, float(2 * r), False
        if state:
            save_file(state, os.path.join(directory, "adapter_model.safetensors"))
        if sidecar:
            save_file(sidecar, os.path.join(directory, "pissa_init.safetensors"))

        # Fully-trained embed/head (PEFT modules_to_save). Kept in a SEPARATE file
        # so the LoRA loader (which expects only lora_A/lora_B) is undisturbed;
        # they are large full matrices, applied by merging into the base, not via
        # the runtime LoRA slots. Saved in HF orientation ([vocab, hidden]) under
        # the standard module names.
        modules_to_save: list[str] = []
        ms_state: dict[str, torch.Tensor] = {}
        if self.embed_weight is not None:
            ms_state["model.embed_tokens.weight"] = \
                self.embed_weight.detach().to(torch.float16).cpu()           # [V, d]
            modules_to_save.append("embed_tokens")
        if self.head_weight is not None:
            ms_state["lm_head.weight"] = \
                self.head_weight.detach().t().contiguous().to(torch.float16).cpu()  # [d,V]->[V,d]
            modules_to_save.append("lm_head")
        if ms_state:
            save_file(ms_state, os.path.join(directory, "modules_to_save.safetensors"))

        # Embed/head LoRA (the low-rank alternative to modules_to_save). Also a
        # SEPARATE file: like modules_to_save, the runtime per-linear LoRA loader
        # doesn't apply these -- they're a merge-path artifact. Internal orientation
        # (embed a:[V,r] b:[r,d]; head a:[d,r] b:[r,V]); rank/alpha come from config.
        module_lora: list[str] = []
        ml_state: dict[str, torch.Tensor] = {}
        if self.embed_lora_a is not None:
            ml_state["embed_tokens.lora_a"] = self.embed_lora_a.detach().to(torch.float16).cpu()
            ml_state["embed_tokens.lora_b"] = self.embed_lora_b.detach().to(torch.float16).cpu()
            module_lora.append("embed_tokens")
        if self.head_lora_a is not None:
            ml_state["lm_head.lora_a"] = self.head_lora_a.detach().to(torch.float16).cpu()
            ml_state["lm_head.lora_b"] = self.head_lora_b.detach().to(torch.float16).cpu()
            module_lora.append("lm_head")
        if ml_state:
            save_file(ml_state, os.path.join(directory, "lora_modules.safetensors"))

        if not (state or ms_state or ml_state):
            raise ValueError("No trainable adapters to save.")

        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": r,
            "lora_alpha": alpha,
            "use_rslora": use_rslora,
            "lora_dropout": getattr(self, "lora_dropout", 0.0),
            "bias": "none",
            "fan_in_fan_out": False,
            "target_modules": sorted(target_leaves),
            **({"rank_pattern": rank_pattern} if rank_pattern else {}),
            **({"alpha_pattern": alpha_pattern} if alpha_pattern else {}),
            "modules_to_save": modules_to_save,
            "lora_embed": self.lora_embed,
            "lora_head": self.lora_head,
            # The embed/head LoRAs in lora_modules.safetensors are saved RAW (no
            # scale folded in) and need alpha/denom applied at load time. That
            # value is NOT recoverable from r/lora_alpha above: a pissa export
            # rewrites those to the converted 2r/2r/rslora=False triple so the
            # per-linear loader computes 1.0, which is wrong for these. Record
            # the module scale explicitly so a loader never has to guess.
            **({"module_lora_scale": self._module_lora_scale} if module_lora else {}),
            "tie_word_embeddings": self.tie_word_embeddings,
            "base_model_name_or_path": base_model_name_or_path,
            # Provenance: how the adapter was initialized. For pissa the
            # exported r/alpha above describe the CONVERTED rank-2r tensors;
            # init_lora_r is the rank that actually trained.
            "init_lora": getattr(self, "init_lora", "default"),
            "init_lora_r": init_lora_r,
            # Provenance only (not consumed at load): the quant-aware training
            # mode the adapter was trained under (training.quant_aware).
            "quant_aware": getattr(self, "quant_aware", "none"),
            "quant_aware_scale": getattr(self, "quant_aware_scale", 1.0),
            # MTP head adapters ride in the same adapter_model.safetensors under
            # the head's own module keys (mtp.layers.N..., mtp.fc); a loader
            # given the trunk skips them and one given the head skips the
            # trunk's, so ONE dir serves both models. Provenance only.
            **({"mtp_target_modules": list(self.mtp_target_modules)}
               if getattr(self, "mtp", None) is not None else {}),
        }
        with open(os.path.join(directory, "adapter_config.json"), "w", encoding="utf8") as f:
            json.dump(config, f, indent=2)
        extra = f" + modules_to_save {modules_to_save}" if modules_to_save else ""
        extra += f" + module_lora {module_lora}" if module_lora else ""
        print(f" -- saved native QLoRA adapter ({len(target_leaves)} target types"
              f"{extra}) to {directory}")

    def load_adapter(self, directory: str) -> int:
        """
        Load adapter weights previously written by :meth:`save_adapter` back into
        the trainable wrappers, to *continue* training from a checkpoint. Inverts
        the save transpose (PEFT ``lora_A=[r, in]`` / ``lora_B=[out, r]`` ->
        internal ``a=[in, r]`` / ``b=[r, out]``).

        Only the adapter weights are restored, NOT optimizer state -- AdamW
        resumes cold (a brief, harmless re-warmup for LoRA). The target modules /
        rank must match the current model (a shape mismatch raises). Returns the
        number of wrappers loaded.
        """
        from safetensors.torch import load_file

        # A pissa checkpoint's adapter_model.safetensors holds the CONVERTED
        # rank-2r export (correct for external consumers, wrong shapes for
        # resume); the exact fp32 training state lives in the sidecar. Restore
        # from it when present -- including the frozen offsets, which cannot be
        # recomputed (randomized SVD is not deterministic across runs).
        sidecar_path = os.path.join(directory, "pissa_init.safetensors")
        loaded = 0
        # MTP head wrappers may legitimately be absent from the checkpoint:
        # the two-stage recipe resumes a finished TRUNK adapter (freeze_trunk)
        # and fits a fresh head to it. Those start from init (B = 0) with a
        # note; a missing TRUNK tensor is still an error.
        mtp_ids = {id(w) for w in getattr(self, "_mtp_wrappers", [])}
        fresh_mtp = 0
        if os.path.exists(sidecar_path):
            sc = load_file(sidecar_path)
            with torch.no_grad():
                for w in self._wrappers:
                    if w.r <= 0:
                        continue
                    if id(w) in mtp_ids and f"{w.key}.lora_a" not in sc:
                        fresh_mtp += 1
                        continue
                    try:
                        a = sc[f"{w.key}.lora_a"]
                        b = sc[f"{w.key}.lora_b"]
                        a0 = sc[f"{w.key}.init_a0"]
                        b0 = sc[f"{w.key}.init_b0"]
                    except KeyError as e:
                        raise KeyError(f"pissa checkpoint missing tensors for "
                                       f"{w.key}") from e
                    if a.shape != w.lora_a.shape or b.shape != w.lora_b.shape:
                        raise ValueError(
                            f"pissa adapter shape mismatch for {w.key}: "
                            f"checkpoint a{tuple(a.shape)}/b{tuple(b.shape)} vs "
                            f"model a{tuple(w.lora_a.shape)}/"
                            f"b{tuple(w.lora_b.shape)} -- do --r/--targets "
                            f"match the checkpoint?")
                    w.lora_a.copy_(a.to(w.lora_a.dtype).to(w.lora_a.device))
                    w.lora_b.copy_(b.to(w.lora_b.dtype).to(w.lora_b.device))
                    w.set_init_offset(a0, b0)
                    loaded += 1
            self.init_lora = "pissa"
            print(f" -- resumed {loaded} pissa adapters (+frozen offsets) from "
                  f"{directory}")
        elif getattr(self, "init_lora", "default") == "pissa":
            raise SystemExit(
                f"resuming a pissa run but {directory} has no "
                f"pissa_init.safetensors -- the frozen offsets cannot be "
                f"recovered from the converted adapter. Was this checkpoint "
                f"written by an --init-lora pissa run?")
        else:
            path = os.path.join(directory, "adapter_model.safetensors")
            # adapter_model.safetensors holds the per-linear LoRA; it may be absent
            # for a checkpoint that only trained the embed/head (--lora-embed/
            # --lora-head or --train-embeddings/--train-head). The per-linear loop
            # below still raises if this model HAS per-linear targets but the
            # file/tensors are missing.
            state = load_file(path) if os.path.exists(path) else {}

            for w in self._wrappers:
                if w.r <= 0:
                    continue
                key = f"base_model.model.{w.key}"
                ak, bk = f"{key}.lora_A.weight", f"{key}.lora_B.weight"
                if ak not in state or bk not in state:
                    if id(w) in mtp_ids:
                        fresh_mtp += 1
                        continue
                    raise KeyError(f"checkpoint missing tensors for {w.key} ({ak})")
                a = state[ak].t()  # [r, in] -> [in, r]
                b = state[bk].t()  # [out, r] -> [r, out]
                if a.shape != w.lora_a.shape or b.shape != w.lora_b.shape:
                    raise ValueError(
                        f"adapter shape mismatch for {w.key}: checkpoint "
                        f"a{tuple(a.shape)}/b{tuple(b.shape)} vs model "
                        f"a{tuple(w.lora_a.shape)}/b{tuple(w.lora_b.shape)} "
                        f"-- do --r/--targets match the checkpoint?"
                    )
                with torch.no_grad():
                    w.lora_a.copy_(a.to(w.lora_a.dtype).to(w.lora_a.device))
                    w.lora_b.copy_(b.to(w.lora_b.dtype).to(w.lora_b.device))
                loaded += 1

        # Restore fully-trained embed/head if present and currently enabled.
        restored_extra = 0
        ms_path = os.path.join(directory, "modules_to_save.safetensors")
        if os.path.exists(ms_path) and (self.embed_weight is not None
                                        or self.head_weight is not None):
            ms = load_file(ms_path)
            with torch.no_grad():
                if self.embed_weight is not None and "model.embed_tokens.weight" in ms:
                    e = ms["model.embed_tokens.weight"]            # [V, d]
                    self.embed_weight.copy_(e.to(self.embed_weight.dtype).to(self.embed_weight.device))
                    restored_extra += 1
                if self.head_weight is not None and "lm_head.weight" in ms:
                    h = ms["lm_head.weight"].t()                   # [V,d] -> [d,V]
                    self.head_weight.copy_(h.to(self.head_weight.dtype).to(self.head_weight.device))
                    restored_extra += 1
            print(f" -- resumed modules_to_save from {directory}")

        # Restore embed/head LoRA if present and currently enabled.
        ml_path = os.path.join(directory, "lora_modules.safetensors")
        if os.path.exists(ml_path) and (self.embed_lora_a is not None
                                        or self.head_lora_a is not None):
            ml = load_file(ml_path)
            with torch.no_grad():
                if self.embed_lora_a is not None and "embed_tokens.lora_a" in ml:
                    self.embed_lora_a.copy_(ml["embed_tokens.lora_a"].to(
                        self.embed_lora_a.dtype).to(self.embed_lora_a.device))
                    self.embed_lora_b.copy_(ml["embed_tokens.lora_b"].to(
                        self.embed_lora_b.dtype).to(self.embed_lora_b.device))
                    restored_extra += 1
                if self.head_lora_a is not None and "lm_head.lora_a" in ml:
                    self.head_lora_a.copy_(ml["lm_head.lora_a"].to(
                        self.head_lora_a.dtype).to(self.head_lora_a.device))
                    self.head_lora_b.copy_(ml["lm_head.lora_b"].to(
                        self.head_lora_b.dtype).to(self.head_lora_b.device))
                    restored_extra += 1
            print(f" -- resumed embed/head LoRA from {directory}")

        if loaded == 0 and restored_extra == 0:
            raise ValueError("No trainable adapters matched the checkpoint.")

        if fresh_mtp:
            print(f" -- note: {directory} carries no MTP head adapters; {fresh_mtp} "
                  f"MTP wrappers start from init (the head is fitted fresh to "
                  f"the resumed trunk).")
        print(f" -- resumed {loaded} adapters from {directory} (optimizer state not restored)")
        return loaded
