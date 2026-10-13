"""
The single seam between the native training forward and exllamav3 internals.

``exllamav3/training/native_llama.py`` reconstructs a differentiable
Llama/Mistral decoder on top of an already-loaded ``exllamav3.Model``. Doing so
requires reading exllamav3's *internal* module layout: the ``..modules`` types,
the loaded RoPE table, RMSNorm epsilons, and the trellis weight reconstruction.
Every such reach lives here and nowhere else, so the training code above depends
on a small, named surface instead of scattered attribute access.

Why isolate it: this is the natural API boundary for the work. If it is ever
promoted into exllamav3 as a supported training entry point, this is the file
that moves (or becomes a thin shim); ``native_llama.py`` would be unaffected.
And a standalone trainer that pins exllamav3 would depend on exactly this
surface and nothing deeper.

``..modules`` is imported lazily, inside the functions that need it, so that
importing this module never triggers the CUDA extension build.
"""

from __future__ import annotations
from typing import Callable, Optional
import torch


# --- top-level decoder layout ----------------------------------------------

def _decoder_layout(model):
    """
    Index a loaded ``Model``'s module list and validate the overall layout,
    returning ``(mods, first_block_idx, last_block_idx)``. The layout the native
    forward reproduces is::

        Embedding  [pre-block norm]  TransformerBlock ... TransformerBlock  RMSNorm  Linear

    where the optional pre-block norm is a norm applied to the token embeddings
    before the first block (MuseGlimmer's unweighted
    ``embed_tokens.embed_norm``; see ``embed_norm``).

    The second layout understood is the Qwen3.8-Flash-Next (``qwen4_exp``)
    stream-stack decoder::

        Embedding  ExpandStreams  TransformerBlock ... [PLELayer] ... TransformerBlock
                   GatedResidual(use_combine=False)  Linear

    where ``ExpandStreams`` broadcasts the embedding into ``hc_mult`` fp32
    residual streams, every block mixes them through its ``attn_hc`` /
    ``mlp_hc`` ``GatedResidual`` sites (see ``block_hc_sites``), a ``PLELayer``
    may sit between two blocks (``ple_layout``), and the final combine-less
    ``GatedResidual`` mixer stands in the final-norm slot (there is no final
    RMSNorm; ``is_gated_residual`` tells the two apart).

    The third layout is the mHC stream-stack decoder (DeepSeek-V4,
    GLM-5.3-Flash)::

        Embedding  ExpandStreams  TransformerBlock ... TransformerBlock
                   HyperHead  RMSNorm  Linear

    where the blocks' sites are ``HyperConnection`` modules (Sinkhorn combine;
    see ``mhc_spec``) next to ordinary pre-norms, and a ``HyperHead`` collapses
    the stack (``hyper_head_spec``) in front of the usual final RMSNorm.

    Everything else is rejected HERE rather than dropped. This selection used to
    be by TYPE alone (``[m for m in mods if isinstance(m, TransformerBlock)]``,
    plus positional picks for the embedding / final norm / head), so any module
    the layout didn't anticipate -- a norm on the embeddings, a stream-expansion
    module, a second head -- silently vanished from the training forward while
    the inference forward still ran it: a wrong frozen base, with no error and
    no NaN, only an adapter that doesn't reproduce at inference.
    """
    from ..modules import Embedding, TransformerBlock, RMSNorm, Linear
    mods = list(model.modules)
    assert isinstance(mods[0], Embedding), \
        f"expected Embedding as first module, got {type(mods[0]).__name__}"
    hc = _is_expand_streams(mods[1]) if len(mods) > 1 else False
    # mHC layout (DeepSeek-V4 / GLM-5.3-Flash): the stream stack is collapsed
    # by a HyperHead module in front of an ordinary final RMSNorm + head.
    mhc = hc and len(mods) > 3 and _is_hyper_head(mods[-3])
    if hc and not mhc:
        assert _is_gated_residual(mods[-2]) and not mods[-2].use_combine, \
            f"stream-stack decoder (ExpandStreams) without a final combine-less " \
            f"GatedResidual mixer as penultimate module: got {type(mods[-2]).__name__}"
    else:
        assert isinstance(mods[-2], RMSNorm), \
            f"expected final RMSNorm as penultimate module, got {type(mods[-2]).__name__}"
    assert isinstance(mods[-1], Linear), \
        f"expected Linear LM head as last module, got {type(mods[-1]).__name__}"
    idxs = [i for i, m in enumerate(mods) if isinstance(m, TransformerBlock)]
    assert idxs, "no TransformerBlock modules found; unsupported architecture"
    first, last = idxs[0], idxs[-1]
    # The non-block modules allowed between decoder blocks: Qwen-VL's
    # DeepstackEmbed (Qwen3-VL / Qwen3.5-VL text towers build one after each of
    # the first N blocks). It holds no weights and is a pure no-op on text-only
    # input; under image input it adds the vision tower's intermediate-layer
    # ("deepstack") features onto the image positions of the residual stream.
    # The native forward reproduces it (see ``deepstack_layout``), so it is
    # accepted here rather than rejected. And, on a stream-stack decoder only,
    # the PLELayer (Qwen3.8-Flash-Next's hashed n-gram injection; see
    # ``ple_layout``). Everything else stays rejected.
    between = [i for i in range(first, last + 1)
               if not isinstance(mods[i], TransformerBlock)]
    bad = [i for i in between if not (_is_deepstack_embed(mods[i])
                                      or (hc and _is_ple_layer(mods[i])))]
    assert not bad, \
        f"non-TransformerBlock module(s) interleaved between the decoder " \
        f"blocks: {[type(mods[i]).__name__ for i in bad]}; unsupported architecture"
    tail = 4 if mhc else 3
    assert last == len(mods) - tail, \
        f"unexpected module(s) between the last decoder block and the final " \
        f"norm: {[type(m).__name__ for m in mods[last + 1:-(tail - 1)]]}"
    pre = mods[1:first]
    if hc:
        assert len(pre) == 1, \
            f"unexpected module(s) between ExpandStreams and the first decoder " \
            f"block: {[type(m).__name__ for m in pre[1:]]}"
    else:
        assert len(pre) <= 1 and all(isinstance(m, RMSNorm) for m in pre), \
            f"unexpected module(s) between the embedding and the first decoder " \
            f"block: {[type(m).__name__ for m in pre]} (only a single RMSNorm on " \
            f"the embeddings is understood -- see backbone.embed_norm)"
    return mods, first, last


def _is_deepstack_embed(module) -> bool:
    """True for the Qwen-VL ``DeepstackEmbed`` module (imported lazily: it lives
    under ``modules.arch_specific`` and is absent on old library versions)."""
    try:
        from ..modules.arch_specific.qwen3_vl import DeepstackEmbed
    except ImportError:            # pragma: no cover - library without VL support
        return False
    return isinstance(module, DeepstackEmbed)


def _is_expand_streams(module) -> bool:
    """True for the hyper-connection ``ExpandStreams`` module (the embedding ->
    stream-stack broadcast of mHC / Qwen3.8-Flash-Next decoders)."""
    try:
        from ..modules.hyperconnections import ExpandStreams
    except ImportError:            # pragma: no cover - library without HC support
        return False
    return isinstance(module, ExpandStreams)


def is_gated_residual(module) -> bool:
    """True for the Qwen3.8-Flash-Next ``GatedResidual`` (low-rank gated
    hyper-connection site, or the final combine-less mixer)."""
    try:
        from ..modules.hyperconnections import GatedResidual
    except ImportError:            # pragma: no cover
        return False
    return isinstance(module, GatedResidual)


def _is_gated_residual(module) -> bool:
    return is_gated_residual(module)


def is_hyper_connection(module) -> bool:
    """True for the mHC ``HyperConnection`` site (DeepSeek-V4 / GLM-5.3-Flash:
    Sinkhorn-normalized combine matrix; see ``mhc_spec``)."""
    try:
        from ..modules.hyperconnections import HyperConnection
    except ImportError:            # pragma: no cover
        return False
    return isinstance(module, HyperConnection)


def _is_hyper_head(module) -> bool:
    """True for the mHC ``HyperHead`` (final stream collapse before the model norm)."""
    try:
        from ..modules.hyperconnections import HyperHead
    except ImportError:            # pragma: no cover
        return False
    return isinstance(module, HyperHead)


def hyper_head(model):
    """The mHC ``HyperHead`` module of a stream-stack decoder (the module in
    front of the final RMSNorm), or ``None`` on every other layout."""
    mods, _, last = _decoder_layout(model)
    m = mods[last + 1]
    return m if _is_hyper_head(m) else None


def _is_ple_layer(module) -> bool:
    """True for the Qwen3.8-Flash-Next ``PLELayer`` (hashed n-gram injection)."""
    try:
        from ..modules.ple import PLELayer
    except ImportError:            # pragma: no cover
        return False
    return isinstance(module, PLELayer)


def hc_mult(model) -> int:
    """Number of parallel residual streams of a stream-stack decoder
    (``ExpandStreams.hc_mult``; Qwen3.8-Flash-Next: 4), or 0 for an ordinary
    ``[b, t, d]`` residual."""
    mods, first, _ = _decoder_layout(model)
    return int(mods[1].hc_mult) if first > 1 and _is_expand_streams(mods[1]) else 0


def split_decoder(model):
    """
    Return ``(embed, blocks, final_norm, lm_head)`` from a loaded exllamav3
    ``Model``, validating the overall module layout. ``blocks`` is the list of
    ``TransformerBlock`` modules, in order (any interleaved ``DeepstackEmbed`` /
    ``PLELayer`` modules are skipped here; see ``deepstack_layout`` /
    ``ple_layout``). On a stream-stack decoder ``final_norm`` is the final
    ``GatedResidual`` mixer, not an ``RMSNorm`` (``is_gated_residual``).

    NOTE: an architecture may also carry a norm on the token embeddings, which
    this tuple does NOT include -- get it from ``embed_norm(model)``.
    """
    from ..modules import TransformerBlock
    mods, first, last = _decoder_layout(model)
    blocks = [m for m in mods[first:last + 1] if isinstance(m, TransformerBlock)]
    return mods[0], blocks, mods[-2], mods[-1]


def deepstack_layout(model) -> dict:
    """
    ``{block_index: deepstack_index}`` for every ``DeepstackEmbed`` module that
    follows decoder block ``block_index`` (0-based over the TransformerBlocks),
    empty on every architecture without them. ``deepstack_index`` selects which
    of the vision tower's deepstack feature maps the module adds
    (``MMEmbedding.deepstack_embeddings[deepstack_index]``); the inference
    module adds ``params["deepstack_emb"][deepstack_index]`` -- a [b, t, hidden]
    tensor that is zero everywhere except the image token positions -- onto
    the residual stream in place. The native forward mirrors that add.
    """
    from ..modules import TransformerBlock
    mods, first, last = _decoder_layout(model)
    layout, bi = {}, -1
    for m in mods[first:last + 1]:
        if isinstance(m, TransformerBlock):
            bi += 1
        elif _is_deepstack_embed(m):
            assert bi >= 0, "DeepstackEmbed before the first decoder block"
            assert bi not in layout, f"two DeepstackEmbed modules after block {bi}"
            layout[bi] = int(m.deepstack_index)
    return layout


def ple_layout(model) -> dict:
    """
    ``{block_index: PLELayer}`` for every ``PLELayer`` module that runs BEFORE
    decoder block ``block_index`` (0-based over the TransformerBlocks); empty
    on every architecture without one. Qwen3.8-Flash-Next builds the layer in
    front of the block whose 1-based index is in ``ple_layer_ids`` (``[2]``:
    between blocks 0 and 1). The inference module adds its delta into the
    stream stack in place (``streams <- streams + ple(streams, ids)``); the
    native forward mirrors that add (``training.ple``).
    """
    from ..modules import TransformerBlock
    mods, first, last = _decoder_layout(model)
    layout, bi = {}, 0
    for m in mods[first:last + 1]:
        if isinstance(m, TransformerBlock):
            bi += 1
        elif _is_ple_layer(m):
            assert bi not in layout, f"two PLELayer modules before block {bi}"
            assert not getattr(m, "stub", False), \
                "tensor-parallel PLE stub in the module list; load the model layer-split"
            layout[bi] = m
    return layout


def uses_noncausal_mm_spans(model) -> bool:
    """
    True when this architecture's INFERENCE forward lets image tokens attend
    bidirectionally within their own image span (Gemma4: ``prepare_inputs``
    builds ``params["non_causal_spans"]`` from the multimodal token mask, and
    the attention kernels run each image span non-causally over itself). The
    native forward must mirror that for image+text parity; every other VL arch
    in exllamav3 (Qwen-VL, Gemma3, Mistral3) prefills image tokens causally.
    Detected the same way the inference path does it -- by the arch module
    defining the span preparer that its ``prepare_inputs`` calls -- so it can't
    drift from inference by a hand-kept list.
    """
    import sys
    arch_mod = sys.modules.get(type(model).__module__)
    return callable(getattr(arch_mod, "_prepare_noncausal_mm_spans", None))


def embed_norm(model):
    """
    The optional norm applied to the token embeddings before the first decoder
    block, or ``None``.

    MuseGlimmer normalizes its embeddings with an unweighted RMSNorm
    (``model.language_model.embed_tokens.embed_norm``, eps = ``rms_norm_eps``),
    a separate module sitting between the ``Embedding`` and the first block --
    NOT the ``Embedding``'s own ``normalize`` flag (Gemma's muP-style scaling,
    which ``embed_apply`` handles). Skipping it feeds the first block an
    embedding of entirely the wrong magnitude, so it is read here and applied by
    the native forward right after the embedding lookup (and after any embedding
    adapter, mirroring the module order).
    """
    mods, first, _ = _decoder_layout(model)
    pre = [m for m in mods[1:first] if not _is_expand_streams(m)]
    return pre[0] if pre else None


def output_device(final_norm):
    """The device of the final-norm slot's weights (= the output device under a
    layer-autosplit load): an ``RMSNorm``'s weight, or a stream-stack decoder's
    final ``GatedResidual`` mixer (its ``norm_w`` table)."""
    if is_gated_residual(final_norm):
        return final_norm.norm_w.device
    return final_norm.weight.device


# --- gated-residual hyper-connections (Qwen3.8-Flash-Next) -----------------

def block_hc_sites(block):
    """The ``(attn_hc, mlp_hc)`` hyper-connection site modules of one block, or
    ``(None, None)`` on an ordinary pre-norm block. Both are ``GatedResidual``
    sites (Qwen3.8-Flash-Next) or both ``HyperConnection`` sites (mHC:
    DeepSeek-V4 / GLM-5.3-Flash) when present; ``hc_site_spec`` builds the
    matching spec."""
    return getattr(block, "attn_hc", None), getattr(block, "mlp_hc", None)


def hc_site_spec(module) -> dict:
    """The spec of one hyper-connection site of either kind, tagged with
    ``kind``: ``"gated"`` (``gated_residual_spec``) or ``"mhc"`` (``mhc_spec``)."""
    if is_gated_residual(module):
        return {"kind": "gated", **gated_residual_spec(module)}
    assert is_hyper_connection(module), \
        f"expected a GatedResidual or HyperConnection site, got {type(module).__name__}"
    return mhc_spec(module)


def mhc_spec(module) -> dict:
    """
    Plain-data description of one mHC ``HyperConnection`` site for
    ``training.hyperconnections.mhc_mix`` / ``mhc_apply``: ``hc_mult``,
    ``hidden_size``, ``eps`` (the site's unweighted RMSNorm, = the model's
    rms_norm_eps), ``hc_eps``, ``sinkhorn_iters`` and the frozen fp32 tables
    ``fn`` ``[2H + H*H, H*d]``, ``base`` ``[2H + H*H]``, ``scale`` ``[3]``
    (the checkpoint's ``{key}_fn`` / ``_base`` / ``_scale``), laundered out of
    inference mode (``_frozen_normal``).
    """
    assert is_hyper_connection(module), \
        f"expected a HyperConnection, got {type(module).__name__}"
    assert module.fn is not None and module.base is not None and module.scale is not None, \
        f"HyperConnection {module.key}: not loaded (load the model first)"
    H = int(module.hc_mult)
    assert tuple(module.fn.shape) == (2 * H + H * H, H * module.hidden_size), \
        f"HyperConnection {module.key}: fn shape {tuple(module.fn.shape)} != " \
        f"{(2 * H + H * H, H * module.hidden_size)}"
    assert module.base.numel() == 2 * H + H * H and module.scale.numel() == 3, \
        f"HyperConnection {module.key}: base/scale shape mismatch"
    return {
        "kind": "mhc",
        "key": module.key,
        "hc_mult": H,
        "hidden_size": int(module.hidden_size),
        "eps": float(module.rms_eps),
        "hc_eps": float(module.hc_eps),
        "sinkhorn_iters": int(module.sinkhorn_iters),
        "fn": _frozen_normal(module.fn),
        "base": _frozen_normal(module.base),
        "scale": _frozen_normal(module.scale),
    }


def hyper_head_spec(module) -> dict:
    """Spec of the mHC ``HyperHead`` for ``training.hyperconnections.mhc_head``:
    ``mean`` (GLM-5.3-Flash: parameterless mean, no tables) or the DeepSeek-V4
    gate tables ``fn`` ``[H, H*d]``, ``base`` ``[H]``, ``scale`` ``[1]`` fp32
    with the norm ``eps`` and ``hc_eps``."""
    assert _is_hyper_head(module), f"expected a HyperHead, got {type(module).__name__}"
    spec = {"mean": bool(module.mean), "hc_mult": int(module.hc_mult),
            "eps": float(module.rms_eps), "hc_eps": float(module.hc_eps)}
    if not module.mean:
        assert module.fn is not None and module.base is not None and module.scale is not None, \
            f"HyperHead {module.key}: not loaded (load the model first)"
        H = int(module.hc_mult)
        assert module.fn.shape[0] == H and module.base.numel() == H \
            and module.scale.numel() == 1, \
            f"HyperHead {module.key}: table shapes {tuple(module.fn.shape)} / " \
            f"{module.base.numel()} / {module.scale.numel()}"
        spec.update({"fn": _frozen_normal(module.fn), "base": _frozen_normal(module.base),
                     "scale": _frozen_normal(module.scale)})
    return spec


def gated_residual_spec(module) -> dict:
    """
    Plain-data description of one ``GatedResidual`` (a block's site, or the
    final mixer) for ``training.hyperconnections.gated_residual_mix``:
    ``hc_mult``, ``hidden_size``, ``rank``, ``eps``, and the frozen tables
    ``norm_w`` ``[H, d]`` fp32 (the module's ``1 + w``), ``down`` ``[rank,
    H*d]`` and ``up`` ``[H*d, rank]`` (fp16, the checkpoint orientation) and,
    for a site, ``inject`` ``[H, H*d]`` fp16 (``None`` on the final mixer).

    Read from the loaded module's RESIDENT tables, not the source weights:
    ``down`` / ``inject`` live as views of ``proj_h`` and ``up`` is recovered
    from the always-kept fused-kernel repack ``upx_h`` (``[H, d/4, rank, 4]``,
    a pure permutation of ``up``; ``_tiled_tables`` restores it the same way),
    so the model needs no ``keep_source_weights`` load -- inference releases
    the checkpoint-layout ``up_h`` when its tiled kernel is active. Laundered
    out of inference mode (``_frozen_normal``) since they reach autograd ops raw.
    """
    assert is_gated_residual(module), \
        f"expected a GatedResidual, got {type(module).__name__}"
    assert module.norm_w is not None and module.proj_h is not None, \
        f"GatedResidual {module.key}: not loaded (load the model first)"
    H, d, rank = module.hc_mult, module.hidden_size, module.rank
    if module.up_h is not None:
        up = module.up_h
    else:
        up = module.upx_h.permute(0, 1, 3, 2).reshape(H * d, rank)
    inject = module.inject_h if module.use_combine else None
    assert (inject is not None) == bool(module.use_combine)
    return {
        "key": module.key,
        "hc_mult": H,
        "hidden_size": d,
        "rank": rank,
        "eps": float(module.rms_eps),
        "norm_w": _frozen_normal(module.norm_w),                 # [H, d] fp32, 1 + w
        "down": _frozen_normal(module.down_h),                   # [rank, H*d] half
        "up": _frozen_normal(up.contiguous()),                   # [H*d, rank] half
        "inject": _frozen_normal(inject.contiguous()) if inject is not None else None,
    }


# --- PLE layer (Qwen3.8-Flash-Next hashed n-gram injection) ----------------

def ple_projections(module):
    """The ``(key_proj, value_proj)`` native ``Linear``s of a ``PLELayer``
    (``[ple_embed_dim, hc_mult*hidden]`` and ``[ple_embed_dim, hidden]``).
    Frozen by the training forward: they project a frozen hashed lookup."""
    return module.key_proj, module.value_proj


def ple_spec(module) -> dict:
    """
    Plain-data description of a ``PLELayer`` for ``training.ple.ple_delta``:
    ``hc_mult``, ``hidden_size``, ``gate_scale`` (``hidden**-0.5``),
    ``dilation`` (= ``ngram_size``), ``kernel``, the frozen depthwise conv
    weight ``conv_w`` ``[H*d, 1, kernel]`` (half), and the three grouped
    ``norm_spec``s ``norm_key`` / ``norm_query`` / ``norm_conv`` (weight
    viewed ``[H, d]``, bias 1.0 -- the module's ``1 + w``).
    """
    assert _is_ple_layer(module), f"expected a PLELayer, got {type(module).__name__}"
    assert not getattr(module, "stub", False), "PLE stub (TP) has no weights"
    assert module.conv_w is not None, \
        f"PLELayer {module.key}: not loaded (load the model first)"
    w = module.conv_w
    if w.dim() == 2:
        w = w.unsqueeze(1)
    H, d = module.hc_mult, module.hidden_size
    assert tuple(w.shape) == (H * d, 1, module.conv_kernel_size), \
        f"PLELayer {module.key}: conv weight shape {tuple(w.shape)} != " \
        f"({H * d}, 1, {module.conv_kernel_size})"
    return {
        "key": module.key,
        "hc_mult": H,
        "hidden_size": d,
        "gate_scale": float(module.gate_scale),
        "dilation": int(module.conv_dilation),
        "kernel": int(module.conv_kernel_size),
        "state_len": int(module.conv_state_len),
        "conv_w": _frozen_normal(w),
        "norm_key": norm_spec(module.norm_key),
        "norm_query": norm_spec(module.norm_query),
        "norm_conv": norm_spec(module.norm_conv),
    }


def ple_ngram_embed(module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    The frozen n-gram embedding ``[b, t, ple_embed_dim]`` (half, on the PLE
    layer's device) for whole sequences starting at position 0 -- the
    inference layer's stateless path: ids prepared as ``PLELayer._prepare_ids``
    (CPU int64; multimodal alias ids -> the placeholder token), the hashing
    history eos-padded by ``ngram_size - 1`` in front (``_history``, the
    fresh-sequence convention ``PLELayerState.clear`` restores), then
    ``NGramEmbedding.forward`` (host-side hashing + row gather from RAM /
    disk, GPU dequant). No gradient: the table is frozen and the gather is
    not differentiable anyway.
    """
    with torch.no_grad():
        ids = module._prepare_ids(input_ids)
        history = module._history(ids)
        emb = module.ple_embedding.forward(history, {})
    return emb


# --- MTP (multi-token prediction) head ---------------------------------------

def mtp_layout(mtp_model):
    """
    Index a loaded MTP component model (``Model.from_config(config,
    component="mtp")``) and validate its layout, returning
    ``(input_layer, blocks, final_norm)``. The layout the native MTP forward
    reproduces is the Qwen3.5/3.6 one::

        MTPInputLayer  TransformerBlock ... TransformerBlock  RMSNorm

    where the input layer holds two pre-fc RMSNorms (one on the trunk's
    post-final-norm hidden state, one on the token embedding) and a
    ``[2*hidden -> hidden]`` fc projection over their concatenation
    ``[embedding | hidden]``; the embedding and LM head are BORROWED from the
    trunk at inference (``attach_to``), so the MTP model carries neither.

    Everything else is rejected here rather than silently dropped -- the
    DeepSeek/GLM-style heads consume a pre-norm residual, and the Qwen3.8
    (``Qwen4ExpMTPInputLayer``) head splits the projection and runs hyper
    connections; neither is what this forward computes.
    """
    from ..modules import TransformerBlock, RMSNorm
    mods = list(mtp_model.modules)
    assert len(mods) >= 3, \
        f"MTP model has {len(mods)} modules; expected input layer + block(s) + final norm"
    inp = mods[0]
    name = getattr(inp, "module_name", type(inp).__name__)
    assert name == "Qwen3_5MTPInputLayer" and all(
        hasattr(inp, a) for a in ("pre_fc_norm_hidden", "pre_fc_norm_embedding", "fc")), \
        f"unsupported MTP input layer {name}: only the Qwen3.5/3.6 layout " \
        f"(pre_fc_norm_hidden + pre_fc_norm_embedding + fc over [embedding | " \
        f"trunk final-norm state]) is reproduced by the native MTP forward"
    assert isinstance(mods[-1], RMSNorm), \
        f"expected final RMSNorm as last MTP module, got {type(mods[-1]).__name__}"
    blocks = mods[1:-1]
    bad = [type(m).__name__ for m in blocks if not isinstance(m, TransformerBlock)]
    assert not bad, \
        f"non-TransformerBlock module(s) inside the MTP head: {bad}; unsupported layout"
    return inp, blocks, mods[-1]


def mtp_input_parts(input_layer):
    """``(pre_fc_norm_hidden, pre_fc_norm_embedding, fc)`` of a Qwen3.5-style
    MTP input layer (see ``mtp_layout``). ``fc`` is a native ``Linear``
    (``[2*hidden, hidden]``), so it takes a LoRA like any block projection."""
    return (input_layer.pre_fc_norm_hidden, input_layer.pre_fc_norm_embedding,
            input_layer.fc)


# --- per-block structure ---------------------------------------------------

def is_gated_delta_net(attn) -> bool:
    """True when a block's ``attn`` slot holds a ``GatedDeltaNet`` (linear /
    recurrent attention -- Qwen3.5/3.6, Qwen3-Next, OLMo-hybrid) rather than
    softmax attention."""
    from ..modules import GatedDeltaNet
    return isinstance(attn, GatedDeltaNet)


def is_mla(attn) -> bool:
    """True for exllamav3's ``MLAttention`` (multi-head latent attention:
    DeepSeek-V2/V3 family, GLM-5.x; the latent ``kv_a`` path with the
    ``kv_b`` up-projection held as the module's ``w_uk_flat`` / ``w_uv_flat``
    tables). ``mla_spec`` reads it; the native forward attends densely over
    the up-projected per-head K/V."""
    try:
        from ..modules import MLAttention
    except ImportError:            # pragma: no cover
        return False
    return isinstance(attn, MLAttention)


def is_dsv4_attention(attn) -> bool:
    """True for exllamav3's ``DSV4Attention`` (DeepSeek-V4 hybrid sparse
    attention: sliding / CSA / HCA layers with gated-pool compressors, sinks,
    MQA with V == K, grouped low-rank output projection). ``dsv4_spec`` reads
    it; ``native_llama._dsv4_forward`` is the differentiable transcription."""
    try:
        from ..modules.dsv4 import DSV4Attention
    except ImportError:            # pragma: no cover
        return False
    return isinstance(attn, DSV4Attention)


def is_short_conv(attn) -> bool:
    """True when a block's ``attn`` slot holds a ``ShortConv`` (LFM2 /
    LFM2-MoE gated short causal convolution -- the ``conv`` layers of
    LFM2.5-8B-A1B) rather than softmax attention."""
    from ..modules import ShortConv
    return isinstance(attn, ShortConv)


def is_block_sparse_mlp(mlp) -> bool:
    """True when a block's ``mlp`` slot holds a ``BlockSparseMLP`` (mixture of
    experts -- Qwen3-MoE, Qwen3.5-MoE, Mixtral, ...) rather than a dense
    ``GatedMLP``."""
    from ..modules import BlockSparseMLP
    return isinstance(mlp, BlockSparseMLP)


def _assert_moe_supported(key: str, mlp) -> None:
    """
    The differentiable MoE forward covers two router types. "std": the
    softmax router (top-k over the router logits, softmax over the selected
    k -- identical to HF's softmax-all + renormalize with
    ``norm_topk_prob=True``, which the Qwen3/3.5-MoE configs assert) with
    optional per-expert scale. "dots": the ungrouped sigmoid router (AFMoE /
    dots.llm1): sigmoid scores, ``e_score_correction_bias`` added for the
    top-k *selection* only, the selected experts weighted by their unbiased
    scores normalized over the selected set, then multiplied by
    ``routed_scaling_factor`` (``routing_dots`` in the inference module).
    Both cover the optional shared expert behind a sigmoid shared gate
    (Qwen3.5-MoE) or ungated (AFMoE), plus the Gemma4 MoE layout:
    ``alt_residual_channel`` (routing and the routed experts read the RAW
    post-attention residual through their own pre-norms while the shared
    expert reads the block's normed input) and the four extra RMSNorms
    (router pre / routed pre / routed post / shared post -- each must be an
    ``RMSNorm``; ``norm_spec`` rejects anything else at construction).
    Everything else -- the grouped ds3 router and expert-parallel TP splits
    -- is rejected loudly here.
    """
    from ..modules import GatedMLP
    assert mlp.router_type in ("std", "dots", "sqrtsp", "sqrtsp_hash"), \
        f"{key}: only the 'std' softmax, 'dots' sigmoid and DeepSeek-V4 'sqrtsp' / " \
        f"'sqrtsp_hash' top-k routers are supported, got router_type " \
        f"{mlp.router_type!r} (grouped ds3 MoE routing not wired up)"
    if mlp.router_type == "sqrtsp_hash":
        # DeepSeek-V4 hash-MoE bootstrap layers: expert ids come from the
        # frozen tid2eid table indexed by the input token ids (the trunk
        # forward hands input_ids down to the block for this).
        assert getattr(mlp, "tid2eid", None) is not None, \
            f"{key}: sqrtsp_hash router without a loaded tid2eid table"
        assert mlp.tid2eid.shape[-1] == mlp.num_experts_per_tok, \
            f"{key}: tid2eid width {mlp.tid2eid.shape[-1]} != top-k {mlp.num_experts_per_tok}"
    assert mlp.routing_gate is not None, f"{key}: MoE block has no routing gate"
    # The inference side's CPU expert offload (EXL3_MOE_CPU_OFFLOAD: whole
    # layers registered with the CPU MoE worker, their Linears never loaded;
    # EXL3_MOE_CPU_SPLIT: the tail experts removed from the module tree) runs
    # the experts as fp16 GEMMs in a worker process -- not differentiable, and
    # the weights are not on the GPU for the training forward to read. The
    # training path's own offload is ``--stream-experts`` (expert_stream.py),
    # offline trainer only; the realtime (tabbyAPI) path needs resident experts.
    assert not getattr(mlp, "cpu_offload", False) and \
        getattr(mlp, "cpu_split_first", None) is None, \
        f"{key}: the inference CPU expert offload (EXL3_MOE_CPU_OFFLOAD / " \
        f"EXL3_MOE_CPU_SPLIT) is not supported for training: the offloaded " \
        f"experts run in the CPU worker and are not differentiable. Unset it; " \
        f"to train with experts in host RAM use the offline trainer's " \
        f"--stream-experts (not available on the realtime path)"
    assert mlp.num_local_experts == mlp.num_experts, \
        f"{key}: expert/tensor-parallel MoE split ({mlp.num_local_experts} of " \
        f"{mlp.num_experts} experts local) is not supported for training"
    assert len(mlp.gates) == len(mlp.ups) == len(mlp.downs) == mlp.num_experts, \
        f"{key}: expected one gate/up/down linear per expert"
    # n_group/topk_group = 1/1 is the trivial grouping (one group holding
    # every expert, and it is always selected): GLM-4.5/4.7 (Glm4MoeForCausalLM)
    # declares its "dots" router that way and the inference side runs the
    # same ungrouped ext.routing_ds3_nogroup kernel for it. Real grouping
    # (DeepSeek-V3's n_group > 1) is a different top-k and stays rejected.
    assert mlp.n_group in (None, 1) and mlp.topk_group in (None, 1), \
        f"{key}: grouped expert routing (n_group={mlp.n_group}, " \
        f"topk_group={mlp.topk_group}) is not supported"
    if mlp.router_type == "std":
        assert mlp.routed_scaling_factor in (None, 1.0), \
            f"{key}: routed_scaling_factor is a sigmoid-router feature, " \
            f"unsupported on the std router"
        # e_score_correction_bias is a ds3/dots-router input; the std routing
        # path ignores it, so a loaded one would silently change nothing --
        # reject.
        assert mlp.e_score_correction_bias is None, \
            f"{key}: e_score_correction_bias is not consumed by the std router"
    if mlp.shared_experts is not None:
        assert isinstance(mlp.shared_experts, GatedMLP), \
            f"{key}: only a GatedMLP shared expert is supported, got " \
            f"{type(mlp.shared_experts).__name__}"
        assert mlp.shared_experts.activation_fn in ("silu", "gelu"), \
            f"{key}: unsupported shared-expert activation " \
            f"{mlp.shared_experts.activation_fn!r}"
        assert float(getattr(mlp.shared_experts, "act_limit", 0.0) or 0.0) >= 0.0, \
            f"{key}: negative shared-expert act_limit"
    if mlp.shared_gate is not None:
        assert mlp.shared_experts is not None, \
            f"{key}: shared gate without shared experts"


def assert_block_supported(block):
    """
    Reject architectures the native forward can't faithfully reproduce, loudly,
    so a mismatch is an explicit error rather than a silently wrong forward.

    The native block forward reproduces a pre-norm decoder and reads every
    norm / activation / scale from the loaded modules, so it covers
    Llama/Mistral/Qwen2 (plain), Qwen3 (q/k-norm), Gemma3/4 (q/k/v-norm +
    sandwich post-norms + GeGLU + sliding/full window + per-layer head dims),
    the Qwen3.5/3.6 hybrid layers: GatedDeltaNet (linear/recurrent
    attention, split in_proj_qkv/z/b/a projection layout) and gated softmax
    attention (interleaved output gate OR a separate full-width g_proj --
    AFMoE), NoPE layers (AFMoE full-attention layers carry no RoPE at all),
    MuseGlimmer text towers (full-width gate + scaleless q/k-norm with the q
    scale factor folded into sm_scale + sandwich norms + per-layer RoPE theta
    with NoPE full-attention layers; its embedding norm and head logit
    pre-scale are handled outside the block, see ``embed_norm`` /
    ``head_pre_scale``), LFM2 / LFM2-MoE ShortConv layers (gated short causal
    depthwise conv in the attention slot -- LFM2.5-8B-A1B builds ~3/4 of its
    layers this way, the rest as q/k-normed softmax attention),
    and BlockSparseMLP mixtures of experts with the "std" softmax or "dots"
    sigmoid top-k router incl. the optional shared expert + sigmoid shared
    gate (Qwen3-MoE, Qwen3.5-MoE), the ungated shared expert (AFMoE) and the
    Gemma4 MoE layout (alt residual channel + router/routed/shared extra
    norms), and the HEADWISE sigmoid attention output gate (Spark-X2.5: a
    per-head scalar ``g_proj`` ``[hidden, nq]`` broadcast over head_dim).
    Qwen3.8-Flash-Next blocks are accepted too: GatedResidual hyper-connection
    sites in place of the pre-norms (the residual is the fp32 stream stack --
    ``block_hc_sites`` / ``gated_residual_spec``), the sigmoid-gated GDN
    output norm (``gdn_norm_spec``) and QSA full attention, trained as dense
    attention up to the indexer's sparse threshold (``block_metadata``'s
    ``qsa_threshold``).
    What it still cannot do is rejected here: fused-qkvz GatedDeltaNet (the
    Qwen3-Next layout), grouped ds3-router MoE, the softplus headwise gate
    (Laguna), mHC (Sinkhorn) hyper-connections, and non-NeoX RoPE. mRoPE and
    partial rotary (Qwen-VL text towers, Spark-X2.5 full-attention layers)
    are ACCEPTED for text-only training -- see the notes at the assertions
    below.
    """
    from ..modules import GatedMLP, Attention, SlidingAttention
    key = getattr(block, "key", "?")
    attn = getattr(block, "attn", None)
    mlp = getattr(block, "mlp", None)
    assert attn is not None and mlp is not None, \
        f"{key}: block must have both attention and MLP (parallel/no-op blocks unsupported)"
    # Hyper-connection sites (Qwen3.8-Flash-Next): the block's residual is the
    # fp32 stream stack and each sublayer site is a GatedResidual mix / inject
    # in place of the pre-norm / residual add (training.hyperconnections). The
    # mHC HyperConnection kind (DeepSeek-V4 / GLM5.3: Sinkhorn combine matrix)
    # is a different computation and is rejected here.
    attn_hc, mlp_hc = block_hc_sites(block)
    if attn_hc is not None or mlp_hc is not None:
        assert attn_hc is not None and mlp_hc is not None, \
            f"{key}: hyper-connection site on one sublayer only"
        mhc = is_hyper_connection(attn_hc)
        for site in (attn_hc, mlp_hc):
            if mhc:
                # mHC (DeepSeek-V4 / GLM-5.3-Flash): the Sinkhorn mix feeds the
                # block's ordinary pre-norm (mix_norm), so the pre-norms MUST
                # be present here; the site tables are checked by mhc_spec.
                assert is_hyper_connection(site), \
                    f"{key}: mixed hyper-connection kinds on one block"
                assert site.fn is not None, \
                    f"{key}: HyperConnection {site.key} not loaded (load the model first)"
            else:
                assert is_gated_residual(site), \
                    f"{key}: only the GatedResidual (Qwen3.8-Flash-Next) and mHC " \
                    f"HyperConnection (DeepSeek-V4 / GLM-5.3-Flash) hyper-connections " \
                    f"are supported, got {type(site).__name__}"
                assert site.use_combine, f"{key}: site-form GatedResidual expected"
                assert site.norm_w is not None, \
                    f"{key}: GatedResidual {site.key} not loaded (load the model first)"
        assert attn_hc.hc_mult == mlp_hc.hc_mult, f"{key}: hc_mult differs between sites"
        if mhc:
            assert getattr(block, "attn_norm", None) is not None \
                and getattr(block, "mlp_norm", None) is not None, \
                f"{key}: mHC sites without the block pre-norms are not supported"
        else:
            assert getattr(block, "attn_norm", None) is None \
                and getattr(block, "mlp_norm", None) is None, \
                f"{key}: pre-norms alongside GatedResidual sites are not supported"
        assert getattr(block, "attn_post_norm", None) is None \
            and getattr(block, "mlp_post_norm", None) is None \
            and getattr(block, "layer_scalar_f", None) is None, \
            f"{key}: post-norms / layer scalars alongside hyper-connection sites " \
            f"are not supported"
        assert not (is_block_sparse_mlp(mlp) and getattr(mlp, "alt_residual_channel", False)), \
            f"{key}: alt_residual_channel MoE on a stream-stack residual is not supported"
    if is_block_sparse_mlp(mlp):
        _assert_moe_supported(key, mlp)
    else:
        assert isinstance(mlp, GatedMLP), \
            f"{key}: only GatedMLP or BlockSparseMLP is supported, got {type(mlp).__name__}"
    assert mlp.activation_fn in ("silu", "gelu"), \
        f"{key}: only SiLU/GeLU gated MLP is supported, got activation {mlp.activation_fn!r}"
    # act_limit (DeepSeek-V4 / GLM-5.3-Flash swiglu_limit): the clamped gated
    # activation min(act(g), L) * clamp(u, -L, L) -- reproduced by the native
    # forward exactly as the inference act_mul kernel orders it (activation
    # first, then the clamp on its output; see native_llama._gated_act).
    assert float(getattr(mlp, "act_limit", 0.0) or 0.0) >= 0.0, \
        f"{key}: negative act_limit"
    if is_gated_delta_net(attn):
        # Differentiable GatedDeltaNet: supported for the SPLIT projection
        # layout (Qwen3.5/3.6: in_proj_qkv / in_proj_z / in_proj_b /
        # in_proj_a) and the KDA layout (GLM-5.3-Flash: qkv_proj + b_proj +
        # the low-rank f_a/f_b forget gate and g_a/g_b output gate; see
        # training.gdn's KDA section). The fused qkvz/ba layout (Qwen3-Next)
        # interleaves heads inside one tensor and is not wired up.
        assert attn.num_k_heads > 0, \
            f"{key}: GatedDeltaNet with no local K heads (TP shard?) unsupported"
        assert attn.qkvz_proj is None and attn.ba_proj is None, \
            f"{key}: fused qkvz/ba GatedDeltaNet projections (Qwen3-Next " \
            f"layout) are not supported; only the split in_proj_qkv/z/b/a " \
            f"layout (Qwen3.5/3.6) and the KDA layout (GLM-5.3-Flash) are"
        if getattr(attn, "kda", False):
            for name in ("qkv_proj", "b_proj", "f_a_proj", "f_b_proj",
                         "g_a_proj", "g_b_proj", "o_proj"):
                assert getattr(attn, name, None) is not None, \
                    f"{key}: KDA GatedDeltaNet missing {name}"
            assert attn.z_proj is None and attn.a_proj is None, \
                f"{key}: KDA layer with z/a projections (unexpected layout)"
            assert attn.num_k_heads == attn.num_v_heads and attn.k_head_dim == attn.v_head_dim, \
                f"{key}: KDA requires uniform head geometry"
            assert str(getattr(attn.norm, "gate_activation", "silu")) == "sigmoid", \
                f"{key}: KDA gated norm must use the sigmoid output gate"
        else:
            for name in ("qkv_proj", "z_proj", "b_proj", "a_proj", "o_proj"):
                assert getattr(attn, name, None) is not None, \
                    f"{key}: GatedDeltaNet missing {name}"
        assert attn.norm is not None, f"{key}: GatedDeltaNet missing gated norm"
        assert attn.a_log is not None and attn.dt_bias is not None, \
            f"{key}: GatedDeltaNet a_log/dt_bias not loaded (load the model first)"
        return
    if is_dsv4_attention(attn):
        # DeepSeek-V4 attention (training.native_llama._dsv4_forward): sliding
        # window + compressed-window entries + sink column, exactly as the
        # stateless inference forward (_forward_nc) up to the CSA indexer's
        # selection budget: a CSA layer's top-k over its T = S // 4 compressed
        # entries is a no-op while T <= index_topk, i.e. S <= 4*index_topk+3
        # (2051 on V4-Flash); block_metadata exposes that as the sparse
        # threshold and the trunk refuses longer batches. HCA / sliding
        # layers have no indexer and are exact at any length.
        assert attn.num_q_heads > 0, f"{key}: DSV4 layer with no local heads (TP shard?)"
        assert attn.layer_type in ("sliding", "csa", "hca"), \
            f"{key}: unknown DSV4 layer type {attn.layer_type!r}"
        assert not getattr(attn, "tp_reduce", False) and not getattr(attn, "tp_mode", False), \
            f"{key}: tensor-parallel DSV4 attention is not supported for training"
        for name in ("q_a", "q_norm", "q_b", "wkv", "kv_norm", "wo_b"):
            assert getattr(attn, name, None) is not None, f"{key}: DSV4 layer missing {name}"
        assert len(attn.wo_a) == attn.o_groups and attn.num_q_heads % attn.o_groups == 0, \
            f"{key}: o_groups {attn.o_groups} vs {len(attn.wo_a)} wo_a slices / " \
            f"{attn.num_q_heads} heads"
        assert attn.sinks is not None and attn.inv_freq_main is not None \
            and attn.inv_freq_compress is not None, \
            f"{key}: DSV4 sinks / rope tables not loaded (load the model first)"
        if attn.layer_type in ("csa", "hca"):
            comp = attn.compressor
            assert comp is not None and comp.ape is not None, \
                f"{key}: DSV4 {attn.layer_type} layer without a loaded compressor"
            assert comp.overlapping == (attn.layer_type == "csa"), \
                f"{key}: compressor overlap flag disagrees with the layer type"
            assert int(comp.compress_rate) > 0
        if attn.layer_type == "csa":
            assert int(attn.index_topk) > 0, f"{key}: CSA layer with a zero index_topk"
        assert int(attn.sliding_window) > 0, f"{key}: DSV4 layer without a sliding window"
        return
    if is_mla(attn):
        # Multi-head latent attention (GLM-5.3-Flash NoPE MLA): the native
        # forward up-projects the normalized latent to per-head K/V through
        # the module's w_uk/w_uv tables and attends densely. Only the NoPE
        # form is wired up (qk_rope_head_dim == 0); the interleaved-RoPE MLA
        # of DeepSeek-V3 / GLM-5.x-big stays rejected here. DSA indexer
        # layers are exact up to index_topk tokens (the inference forward
        # itself attends densely below the selection budget -- MLAttention.
        # _attend: ``sparse = ... > index_topk``); block_metadata exposes the
        # budget as ``qsa_threshold`` and the trunk refuses longer batches.
        assert attn.num_q_heads > 0, f"{key}: MLA layer with no local heads (TP shard?)"
        assert int(attn.qk_rope_head_dim) == 0 and attn.rope_settings is None, \
            f"{key}: only NoPE MLA (qk_rope_head_dim == 0) is supported; this layer " \
            f"carries a {attn.qk_rope_head_dim}-dim RoPE slice (DeepSeek-V3 / GLM " \
            f"MLA with interleaved RoPE is not wired up)"
        assert attn.kv_a_proj_with_mqa is not None and attn.kv_a_layernorm is not None \
            and attn.o_proj is not None, f"{key}: MLA layer missing kv_a / o projections"
        assert attn.q_proj is not None, f"{key}: MLA layer missing the q projection"
        assert (attn.q_a_proj is None) == (attn.q_a_layernorm is None), \
            f"{key}: q_a_proj / q_a_layernorm must come together"
        assert attn.w_uk_flat is not None and attn.w_uv_flat is not None, \
            f"{key}: MLA kv_b tables not loaded (load the model first)"
        assert not float(getattr(attn, "l4_beta", 0.0) or 0.0), \
            f"{key}: Llama-4 position scaling on MLA is not supported"
        assert attn.indexer_mode in (None, "full", "shared"), \
            f"{key}: unknown DSA indexer mode {attn.indexer_mode!r}"
        if attn.indexer_mode is not None:
            assert int(attn.index_topk) > 0, \
                f"{key}: DSA indexer with a zero selection budget cannot be trained densely"
        assert not getattr(attn, "tp_reduce", False), \
            f"{key}: tensor-parallel MLA is not supported for training"
        return
    if is_short_conv(attn):
        # Differentiable ShortConv (LFM2 / LFM2-MoE): in_proj -> b|c|x split ->
        # causal depthwise conv over b*x -> c gate -> out_proj. Two native
        # Linears plus one small depthwise conv tensor; no norm, no RoPE, no
        # state beyond the conv's kernel-1 history (zero for a fresh sequence).
        for name in ("in_proj", "out_proj"):
            assert getattr(attn, name, None) is not None, \
                f"{key}: ShortConv missing {name}"
        assert attn.conv1d_weight is not None, \
            f"{key}: ShortConv conv weight not loaded (load the model first)"
        w = attn.conv1d_weight
        assert w.dim() in (2, 3) and w.shape[0] == attn.hidden_size, \
            f"{key}: ShortConv conv weight shape {tuple(w.shape)} does not " \
            f"match hidden_size {attn.hidden_size} (depthwise [hidden, 1, kernel])"
        assert w.shape[-1] == attn.conv_kernel_size, \
            f"{key}: ShortConv conv weight kernel {w.shape[-1]} != " \
            f"conv_kernel_size {attn.conv_kernel_size}"
        assert not getattr(attn, "tp_reduce", False), \
            f"{key}: tensor-parallel ShortConv is not supported for training"
        return
    # Softmax attention path.
    assert isinstance(attn, (Attention, SlidingAttention)), \
        f"{key}: only softmax Attention/SlidingAttention, GatedDeltaNet or " \
        f"ShortConv is supported, got {type(attn).__name__}"
    # q/k/v norms and output gating ARE supported (read from the modules):
    # the interleaved gate (Qwen3.5 full-attn layers, folded into q_proj), the
    # separate FULL-width g_proj (AFMoE: sigmoid over the whole flattened
    # context, applied before o_proj) and the HEADWISE g_proj (Spark-X2.5: one
    # scalar per q head, sigmoid, broadcast over head_dim -- the inference
    # mul_sigmoid_broadcast_ kernel). The softplus headwise variant (Laguna's
    # mul_softplus_broadcast_) is not wired up: the native forward applies
    # sigmoid, so it must refuse rather than silently gate with the wrong
    # function.
    if getattr(attn, "g_proj", None) is not None:
        assert not getattr(attn, "gate_softplus", False), \
            f"{key}: softplus headwise attention gating is not supported; " \
            f"only the sigmoid gates (headwise / full-width / interleaved) are"
    # QSA (Qwen3.8-Flash-Next sparse full attention): the indexer selects
    # 4-token key blocks for query positions ABOVE its sparse threshold
    # (``4 * block_topk + 3``; 2051 on the released model); at or below it the
    # inference forward attends densely and exactly (``Attention.forward``:
    # ``if seqlen > sparse_threshold()``). The native forward does dense
    # attention only, so it is exact for sequences up to the threshold and
    # must REFUSE longer ones (block_metadata carries the threshold; the
    # trainer checks every batch against it). The indexer's own projections
    # are then dead weight: never targets, never run.
    qsa = getattr(attn, "qsa_indexer", None)
    if qsa is not None:
        assert callable(getattr(qsa, "sparse_threshold", None)), \
            f"{key}: QSA indexer without a sparse_threshold(); unsupported"
        assert int(qsa.sparse_threshold()) > 0, \
            f"{key}: QSA indexer with a zero sparse threshold (sparse from " \
            f"position 0) cannot be trained as dense attention"
    # NoPE layers (AFMoE full-attention layers) are built with no rope_settings
    # and skip RoPE entirely -- accepted (block_metadata carries inv_freq=None
    # and the native forward skips the rotation, mirroring the inference
    # forward's `if self.rope:`). A MISSING table on a layer that should have
    # one (rope_settings set but rope not built) is still a loading error.
    if attn.rope_settings is None:
        return
    assert attn.rope is not None and attn.rope.inv_freq is not None, \
        f"{key}: model loaded without a RoPE table; cannot build positional encoding"
    # mRoPE (Qwen2/3-VL text towers) is ACCEPTED for text-only training. mRoPE
    # only differs from ordinary 1D NeoX RoPE by assigning DIFFERENT position
    # indices to different frequency bands (temporal/height/width sections) --
    # a spread that exists only for image/video tokens. For a pure-text
    # sequence every section shares the same position, so mRoPE collapses
    # EXACTLY to 1D RoPE, which is precisely what native_llama._apply_rope
    # computes (one position_id per token against the loaded inv_freq). All our
    # text-only training needs no forward-math change, and the native validate
    # gate confirms the forward still matches the (mRoPE-aware) inference
    # oracle. Image+text training passes true 3-D [t, h, w] positions (built
    # by training.vision.mrope_position_ids from the image grids) and the
    # block forward then applies the per-band split (block_metadata's
    # mrope_section; native_llama._apply_rope) exactly as the inference
    # get_mrope_freqs does.
    assert attn.rope.rope_settings.rope_style.name == "NEOX", \
        f"{key}: only NeoX-style RoPE is supported, got {attn.rope.rope_settings.rope_style.name}"
    # Partial rotary (rotary_dim < head_dim, e.g. Qwen-VL partial_rotary_factor)
    # is supported: native_llama._apply_rope rotates the leading rotary_dim dims
    # and passes the rest through. inv_freq is built over the rotary slice only
    # (util/rope.py), so its width is rotary_dim; it must not exceed head_dim.
    assert attn.rope.inv_freq.numel() * 2 <= attn.head_dim, \
        f"{key}: rotary_dim ({attn.rope.inv_freq.numel() * 2}) exceeds " \
        f"head_dim ({attn.head_dim})"


def attn_has_mrope(block) -> bool:
    """True if the block's softmax-attention RoPE carries an mRoPE section
    (Qwen-VL text tower). Trained as text-only 1D RoPE -- see the note in
    ``assert_block_supported``. False for GDN blocks (no rope) and plain RoPE."""
    attn = getattr(block, "attn", None)
    rope = getattr(attn, "rope", None)
    return rope is not None and getattr(rope, "mrope_section", None) is not None


def _mlp_metadata(block) -> dict:
    """
    The MLP half of a block's metadata, shared by the ``attn`` and ``gdn``
    block kinds. ``mlp_kind`` is ``"dense"`` (GatedMLP) or ``"moe"``
    (BlockSparseMLP); the MoE keys describe the std softmax top-k router.
    """
    mlp = block.mlp
    meta = {
        # gated-MLP activation ("silu" or "gelu"/GeGLU) -- routed experts and
        # dense MLP alike; the shared expert's own activation is read below.
        "activation": mlp.activation_fn,
        # Clamped gated activation (DeepSeek-V4 / GLM-5.3-Flash swiglu_limit):
        # min(act(g), L) * clamp(u, -L, L); 0.0 = unclamped.
        "act_limit": float(getattr(mlp, "act_limit", 0.0) or 0.0),
    }
    if not is_block_sparse_mlp(mlp):
        meta["mlp_kind"] = "dense"
        return meta
    meta.update({
        "mlp_kind": "moe",
        # "std" (softmax top-k) or "dots" (sigmoid + selection bias +
        # normalize-over-selected * routed_scaling_factor -- AFMoE).
        "router_type": mlp.router_type,
        "num_experts": mlp.num_experts,
        "num_experts_per_tok": mlp.num_experts_per_tok,
        # Post-softmax per-expert scale (bf16 tensor or None); multiplied onto
        # the selected routing weights exactly as routing_std does.
        "per_expert_scale": mlp.per_expert_scale,
        # dots-router inputs (None / unused on the std router). The bias
        # enters expert SELECTION only, never the weights; laundered out of
        # inference-mode since it reaches autograd ops raw (see
        # _frozen_normal).
        "routed_scaling_factor": (float(mlp.routed_scaling_factor)
                                  if mlp.routed_scaling_factor is not None else 1.0),
        "e_score_bias": _frozen_normal(mlp.e_score_correction_bias),
        # DeepSeek-V4 hash layers: [vocab, top_k] expert-id table (CPU/long).
        "tid2eid": (_frozen_normal(mlp.tid2eid) if mlp.router_type == "sqrtsp_hash" else None),
        "shared_activation": (mlp.shared_experts.activation_fn
                              if mlp.shared_experts is not None else None),
        "shared_act_limit": (float(getattr(mlp.shared_experts, "act_limit", 0.0) or 0.0)
                             if mlp.shared_experts is not None else 0.0),
        # Gemma4 layout: routing and the routed experts read the RAW
        # post-attention residual (params["residual"] in the inference
        # forward) through their own pre-norms, NOT the block's normed MLP
        # input (which feeds only the shared expert).
        "alt_residual_channel": bool(mlp.alt_residual_channel),
    })
    return meta


def _frozen_normal(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Materialize a frozen loaded tensor as a NORMAL (non-inference) tensor.

    The native model is loaded under ``torch.inference_mode``, so its weights are
    "inference tensors" that autograd refuses to save for backward. Most reach
    the differentiable forward through arithmetic (casts / adds) that happens to
    launder them, but that laundering is a NO-OP whenever the op is a no-op --
    ``w.to(compute_dtype)`` when the stored dtype already equals it,
    ``w.float()`` when the weight is already fp32 -- and the raw inference tensor
    then reaches an autograd op and dies ("Inference tensors cannot be saved for
    backward"). A one-time ``detach().clone()`` at spec-build makes a normal,
    still-frozen tensor regardless of dtype; cheap (norm/conv tensors are small)
    and dtype-independent, so it can't silently regress on a future dtype combo."""
    return None if t is None else t.detach().clone()


def block_metadata(block) -> dict:
    """
    Plain-data description of one decoder block's attention / RoPE / norm config,
    consumed by the differentiable block forward. Tensors are referenced, not
    copied (``inv_freq`` is the loaded RoPE table itself). ``kind`` is
    ``"attn"`` (softmax attention), ``"gdn"`` (GatedDeltaNet) or
    ``"shortconv"`` (LFM2 ShortConv); the kinds carry different keys. ``mlp_kind`` is ``"dense"`` or ``"moe"`` (see
    ``_mlp_metadata``).
    """
    attn = block.attn
    if is_short_conv(attn):
        # Depthwise causal conv weight, stored [hidden, 1, kernel]; squeezed to
        # [hidden, kernel] for F.conv1d's groups=hidden layout. Laundered out
        # of inference-mode like the GDN conv (see _frozen_normal).
        w = attn.conv1d_weight
        if w.dim() == 3:
            w = w.squeeze(1)
        return {
            "kind": "shortconv",
            "hidden_size": attn.hidden_size,
            "conv_kernel_size": attn.conv_kernel_size,
            "conv1d_weight": _frozen_normal(w),               # [hidden, kernel]
            "conv1d_bias": _frozen_normal(attn.conv1d_bias),  # [hidden] or None
            # MLP half (activation + dense/moe description).
            **_mlp_metadata(block),
            "layer_scalar": getattr(block, "layer_scalar_f", None),
        }
    if is_gated_delta_net(attn):
        # Depthwise causal conv weight: one fused [dim, 1, kernel] tensor, or
        # (older checkpoints) separate q/k/v parts that concatenate along the
        # channel dim -- exactly the fusion the inference forward performs.
        w = attn.conv1d_weight
        if w is None:
            w = torch.cat([attn.conv1d_q_weight, attn.conv1d_k_weight,
                           attn.conv1d_v_weight], dim=0)
        if w.dim() == 3:
            w = w.squeeze(1)
        # Launder the frozen GDN tensors out of inference-mode (see
        # _frozen_normal): conv1d_weight/bias reach F.conv1d raw, and
        # weight.to(x.dtype) is a no-op when the loaded dtype == the compute
        # dtype (bf16), so without this the inference tensor reaches conv1d and
        # backward dies. a_log/dt_bias are laundered defensively too.
        _normal = _frozen_normal
        return {
            "kind": "gdn",
            # KDA (GLM-5.3-Flash): per-k-channel decay from the low-rank
            # forget gate, "safe gate" lower bound, low-rank sigmoid output
            # gate -- see training.gdn.kda_beta_g. False on Qwen3.5/3.6.
            "kda": bool(getattr(attn, "kda", False)),
            "gate_lower_bound": (float(attn.gate_lower_bound)
                                 if getattr(attn, "gate_lower_bound", None) is not None
                                 else None),
            "num_k_heads": attn.num_k_heads,
            "num_v_heads": attn.num_v_heads,
            "k_head_dim": attn.k_head_dim,
            "v_head_dim": attn.v_head_dim,
            "conv_kernel_size": attn.conv_kernel_size,
            "beta_scale": float(attn.beta_scale),
            "a_log": _normal(attn.a_log),        # [nv]
            "dt_bias": _normal(attn.dt_bias),    # [nv] (KDA: [nv*dk])
            "conv1d_weight": _normal(w),         # [2*k_dim + v_dim, kernel]
            "conv1d_bias": _normal(attn.conv1d_bias),  # [2*k_dim + v_dim] or None
            # MLP half (activation + dense/moe description).
            **_mlp_metadata(block),
            "layer_scalar": getattr(block, "layer_scalar_f", None),
        }
    if is_dsv4_attention(attn):
        lt = attn.layer_type
        comp = attn.compressor if lt in ("csa", "hca") else None
        m = int(comp.compress_rate) if comp is not None else 1
        return {
            "kind": "dsv4",
            "layer_type": lt,
            "num_q_heads": attn.num_q_heads,
            "num_kv_heads": 1,
            "head_dim": attn.head_dim,
            "v_head_dim": attn.head_dim,
            "rope_head_dim": attn.rope_head_dim,
            "o_groups": attn.o_groups,
            "sm_scale": attn.sm_scale,
            "sliding_window": int(attn.sliding_window),
            "sinks": _frozen_normal(attn.sinks),                   # [H] fp32
            # This layer's rope table (main theta on sliding layers, the
            # yarn-ramped compress theta on CSA/HCA) and the compressor's.
            "inv_freq": _frozen_normal(attn.inv_freq_main if lt == "sliding"
                                       else attn.inv_freq_compress),
            "inv_freq_compress": _frozen_normal(attn.inv_freq_compress),
            "compress_rate": m,
            "compressor": (None if comp is None else {
                "overlapping": bool(comp.overlapping),
                "head_dim": int(comp.head_dim),
                "rope_dim": int(comp.rope_dim),
                "ape": _frozen_normal(comp.ape),                   # [m, W] fp32
                "norm": norm_spec(comp.norm),
            }),
            "index_topk": int(attn.index_topk or 0),
            "attn_factor": 1.0,
            "mrope_section": None,
            "softcap": 0.0,
            **_mlp_metadata(block),
            "use_k_as_v": False,
            "interleaved_gate": False,
            "full_gate": False,
            "headwise_gate": False,
            "layer_scalar": getattr(block, "layer_scalar_f", None),
            # CSA: dense over all visible compressed entries is exact while
            # T = S // m <= index_topk (DSV4Attention._forward_nc: the top-k
            # only runs past it) -> S <= m * index_topk + m - 1.
            "qsa_threshold": (m * int(attn.index_topk) + m - 1) if lt == "csa" else None,
        }
    if is_mla(attn):
        H = attn.num_q_heads
        return {
            "kind": "mla",
            "num_q_heads": H,
            # K/V are up-projected per q head from the shared latent: no GQA.
            "num_kv_heads": H,
            "head_dim": attn.qk_head_dim,
            "v_head_dim": attn.v_head_dim,
            "kv_lora_rank": attn.kv_lora_rank,
            "qk_nope_head_dim": attn.qk_nope_head_dim,
            "qk_rope_head_dim": attn.qk_rope_head_dim,
            "sm_scale": attn.sm_scale,
            # The kv_b up-projection, in the module's flat layout: [D_c, H*nope]
            # and [D_c, H*v] fp16 (frozen; never a LoRA target -- it is not a
            # Linear on the inference side).
            "w_uk": _frozen_normal(attn.w_uk_flat),
            "w_uv": _frozen_normal(attn.w_uv_flat),
            "inv_freq": None,            # NoPE only (asserted above)
            "attn_factor": 1.0,
            "mrope_section": None,
            "sliding_window": -1,
            "softcap": 0.0,
            **_mlp_metadata(block),
            "use_k_as_v": False,
            "interleaved_gate": False,
            "full_gate": False,
            "headwise_gate": False,
            "layer_scalar": getattr(block, "layer_scalar_f", None),
            # DSA indexer layers: dense attention is exact up to index_topk
            # visible tokens (MLAttention._attend's sparse switch); the trunk
            # refuses longer sequences through the same guard as QSA.
            "qsa_threshold": (int(attn.index_topk) if attn.indexer_mode is not None else None),
        }
    sw = getattr(attn, "sliding_window", -1)
    return {
        "kind": "attn",
        "num_q_heads": attn.num_q_heads,
        "num_kv_heads": attn.num_kv_heads,
        "head_dim": attn.head_dim,
        "sm_scale": attn.sm_scale,
        # RoPE: the llama3-scaled inv_freq lives on the loaded RoPE object.
        # None on a NoPE layer (AFMoE full-attention layers) -- the native
        # forward then skips the rotation, mirroring inference.
        "inv_freq": attn.rope.inv_freq if attn.rope is not None else None,
        "attn_factor": attn.rope.attn_factor if attn.rope is not None else 1.0,
        # mRoPE (Qwen-VL text towers): the per-axis frequency-band split
        # ([t, h, w] counts over the inv_freq entries) that the multimodal
        # forward applies to 3-D [t, h, w] position ids -- see
        # training.vision.mrope_freqs. None on 1-D RoPE archs; also unused
        # whenever the positions passed in are 1-D (text-only training, where
        # mRoPE collapses exactly to 1-D RoPE).
        "mrope_section": (list(attn.rope.mrope_section)
                          if attn.rope is not None
                          and getattr(attn.rope, "mrope_section", None) is not None
                          else None),
        # Per-layer attention window: >0 means sliding (band) attention, else full
        # causal (Gemma / Spark-X2.5 alternate local-sliding / global-full
        # layers). CONVENTION: the module's ``sliding_window`` is the number of
        # PREVIOUS tokens a query may see -- the inference forward hands it to
        # FlashAttention as ``window_size=(sliding_window, 0)`` (attention_fn
        # common.get_window_size), i.e. itself + ``sliding_window`` past keys
        # (Gemma3's arch subtracts 1 from HF's query-inclusive count for this
        # reason). The native forward's window is the QUERY-INCLUSIVE count
        # (eager mask ``-window`` diagonal; flash ``(window - 1, 0)``; see
        # attn_varlen), so convert here: +1, and the training block attends to
        # exactly the keys the inference block does. Matters at any sequence
        # longer than the window (Spark-X2.5: 512).
        "sliding_window": int(sw) + 1 if sw not in (None, 0) and int(sw) > 0 else -1,
        # tanh logit softcapping on the attention scores (Gemma2; 0 = none).
        "softcap": float(getattr(attn, "logit_softcapping", 0.0) or 0.0),
        # MLP half (activation + dense/moe description).
        **_mlp_metadata(block),
        # Some Gemma layers reuse the K projection as V (no separate v_proj).
        "use_k_as_v": bool(getattr(attn, "use_k_as_v", False)),
        # Qwen3.5 full-attention layers: q_proj emits [q | gate] interleaved
        # per head (out_features = 2*nq*hd); the attention output is multiplied
        # by sigmoid(gate) before o_proj.
        "interleaved_gate": bool(getattr(attn, "interleaved_gate", False)),
        # AFMoE: a separate full-width gate projection ([hidden, nq*hd]) on the
        # block input; same sigmoid multiply on the flattened context before
        # o_proj as the interleaved gate, just sourced from its own linear
        # (attn_gate_linear).
        "full_gate": bool(getattr(attn, "full_gate", False)
                          and getattr(attn, "g_proj", None) is not None),
        # Spark-X2.5: a headwise gate projection ([hidden, nq], fp16) on the
        # block input; the per-head context [b, t, nq, hd] is multiplied by
        # sigmoid(gate)[..., None] before the flatten + o_proj (inference:
        # ext.mul_sigmoid_broadcast_(o, g)). Sourced from attn_gate_linear.
        "headwise_gate": bool(getattr(attn, "headwise_gate", False)
                              and getattr(attn, "g_proj", None) is not None),
        # Gemma applies a learned per-layer scalar to the whole residual stream at
        # block end (TransformerBlock.forward: x *= layer_scalar_f). None elsewhere.
        "layer_scalar": getattr(block, "layer_scalar_f", None),
        # QSA sparse-attention threshold (Qwen3.8-Flash-Next): the longest
        # sequence this layer attends to DENSELY at inference. The native
        # forward is exact up to it and refuses anything longer. None on every
        # layer without an indexer.
        "qsa_threshold": (int(attn.qsa_indexer.sparse_threshold())
                          if getattr(attn, "qsa_indexer", None) is not None else None),
    }


def norm_spec(norm) -> Optional[dict]:
    """
    Plain-data description of an ``RMSNorm`` module for the native forward:
    ``{weight, eps, bias, scale}`` (``weight`` is the frozen tensor, or ``None``
    when unweighted). Reproduces ``RMSNorm.forward_torch`` exactly --
    ``y = (x / rms(x)) * scale * (weight + bias)`` -- so Gemma's ``(1 + weight)``
    convention and unweighted v-norm are handled by reading the module's own
    ``constant_bias`` / ``constant_scale`` / ``unweighted`` rather than hardcoding.
    A GROUPED norm (``groups > 1``: the PLE layer's per-stream norms over a
    ``[..., groups, dim]`` input, weight row selected per stream) gets its
    weight viewed ``[groups, dim]`` so the same ``x * (w + bias)`` broadcasts
    per stream, exactly as ``forward_torch`` views it. Returns ``None`` for a
    missing norm.
    """
    if norm is None:
        return None
    from ..modules import RMSNorm
    assert isinstance(norm, RMSNorm), \
        f"native forward only supports RMSNorm, got {type(norm).__name__}"
    w = None if getattr(norm, "unweighted", False) else _frozen_normal(norm.weight)
    groups = int(getattr(norm, "groups", 1) or 1)
    if w is not None and groups > 1:
        assert w.numel() % groups == 0, \
            f"{norm.key}: grouped norm weight ({w.numel()}) not divisible by groups ({groups})"
        w = w.view(groups, -1)
    return {
        "weight": w,
        "eps": norm.rms_norm_eps,
        "bias": float(getattr(norm, "constant_bias", 0.0)),
        "scale": float(getattr(norm, "constant_scale", 1.0)),
    }


def block_norms(block):
    """Return the ``(attn_norm, mlp_norm)`` modules of one block."""
    return block.attn_norm, block.mlp_norm


def block_post_norms(block):
    """Return the optional ``(attn_post_norm, mlp_post_norm)`` modules of one
    block (Gemma sandwich norms). Either is ``None`` for a plain pre-norm block."""
    return getattr(block, "attn_post_norm", None), getattr(block, "mlp_post_norm", None)


def attn_qkv_norms(block):
    """Return the optional ``(q_norm, k_norm, v_norm)`` modules of one block's
    attention (Qwen3: q/k; Gemma: q/k/v; Llama/Mistral/Qwen2: all ``None``)."""
    a = block.attn
    return getattr(a, "q_norm", None), getattr(a, "k_norm", None), getattr(a, "v_norm", None)


def head_softcap(lm_head) -> float:
    """Final-logit tanh softcapping on the LM head (Gemma2; 0 = none)."""
    return float(getattr(lm_head, "softcap", 0.0) or 0.0)


def head_pre_scale(lm_head) -> float:
    """
    The LM head's logit PRE-scale (MuseGlimmer's ``output_multiplier``; 1.0 =
    none). ``Linear.forward`` applies it to the head output before the softcap
    (``logits = cap * tanh(x * pre_scale / cap)``), after any runtime LoRA -- so
    it scales the whole logit vector, and the native forward reproduces it by
    scaling the head INPUT instead, which is exact for a bias-free head and
    covers every head path (materialized logits, trainable/LoRA head, both fused
    CE heads) in one place. Hence the no-bias assertion: the native head is a
    plain ``hidden @ W`` and has always ignored a head bias, but with a scale
    folded into the input an ignored bias would be wrong in a second way.

    ``post_scale`` (applied AFTER the softcap, so it cannot be folded the same
    way) is unused by every architecture here and rejected loudly.
    """
    scale = float(getattr(lm_head, "pre_scale", 1.0) or 1.0)
    post = float(getattr(lm_head, "post_scale", 1.0) or 1.0)
    assert post == 1.0, \
        f"LM head post_scale ({post}) is not supported by the native forward"
    if scale != 1.0:
        assert frozen_bias(lm_head, torch.float32) is None, \
            "LM head with both a bias and a pre_scale is not supported"
    return scale


def block_device(block):
    """
    The device a block's weights live on (set at load; differs per block under a
    layer-autosplit load, identical for a single-device load).
    """
    return block.device


def to_device(x: torch.Tensor, device) -> torch.Tensor:
    """
    Migrate ``x`` to ``device`` the way exllamav3's own layer-split forward does
    (``Module.prepare_for_device``, ``modules/module.py``): a direct copy, or a
    bounce through host memory for device pairs whose peer copies corrupt data.
    A no-op when already on ``device``.

    Unlike the native forward (``@torch.inference_mode``), this runs inside the
    training graph. ``util.device_copy.to_device`` uses only ``.to`` / ``.cpu``,
    both autograd-friendly, so gradients flow back across the boundary.

    Upstream b4010c2 replaced the old module-level ``no_p2p_copy`` flag with a
    lazy per-pair probe behind that helper (``EXLLAMA_NO_P2P_COPY`` still forces
    the bounce). This function read the removed global until the v1.4.9 sync
    surfaced it as an AttributeError on the first cross-device hidden-state
    transfer, which broke EVERY parallel: split training run -- SFT, EBFT and
    preference alike.
    """
    if x.device == device:
        return x
    # Lazy import: only reached at runtime on a real multi-device model, never in
    # the single-device CPU tests (which return above).
    from ..util.device_copy import to_device as _to_device
    return _to_device(x, device)


def attn_projections(block):
    """Return the ``(q_proj, k_proj, v_proj, o_proj)`` linears of one block."""
    a = block.attn
    return a.q_proj, a.k_proj, a.v_proj, a.o_proj


def attn_gate_linear(block):
    """The separate attention output gate ``Linear`` of one block, or ``None``:
    the full-width gate (AFMoE: ``self_attn.gate_proj``, ``[hidden, nq*hd]``;
    inference multiplies the flattened context by ``sigmoid(g)`` before
    ``o_proj`` -- ``ext.mul_sigmoid_``) or the headwise gate (Spark-X2.5:
    ``self_attn.g_proj``, ``[hidden, nq]``, unquantized; inference multiplies
    each head's context by its own ``sigmoid(g)`` -- ``ext.mul_sigmoid_broadcast_``).
    ``block_metadata``'s ``full_gate`` / ``headwise_gate`` say which."""
    a = block.attn
    g = getattr(a, "g_proj", None)
    if g is None:
        return None
    if getattr(a, "full_gate", False) or getattr(a, "headwise_gate", False):
        return g
    return None


def gdn_projections(block):
    """Return the ``(qkv_proj, z_proj, b_proj, a_proj, o_proj)`` linears of a
    GatedDeltaNet block (split projection layout -- Qwen3.5/3.6)."""
    a = block.attn
    return a.qkv_proj, a.z_proj, a.b_proj, a.a_proj, a.o_proj


def kda_projections(block):
    """Return the ``(qkv_proj, b_proj, f_a_proj, f_b_proj, g_a_proj, g_b_proj,
    o_proj)`` linears of a KDA GatedDeltaNet block (GLM-5.3-Flash)."""
    a = block.attn
    return a.qkv_proj, a.b_proj, a.f_a_proj, a.f_b_proj, a.g_a_proj, a.g_b_proj, a.o_proj


def mla_projections(block):
    """Return the ``(q_a_proj, q_b_proj, kv_a_proj_with_mqa, o_proj)`` linears
    of an MLA block. ``q_a_proj`` is ``None`` on a direct-q layout, in which
    case ``q_b_proj`` IS the direct ``q_proj`` (the inference module aliases
    ``q_proj = q_b_proj`` on the low-rank layout)."""
    a = block.attn
    return a.q_a_proj, a.q_proj, a.kv_a_proj_with_mqa, a.o_proj


def dsv4_projections(block):
    """The ``(q_a, q_b, wkv, wo_a_slices, wo_b)`` linears of a DSV4 block
    (``wo_a_slices`` is the list of ``o_groups`` group slices)."""
    a = block.attn
    return a.q_a, a.q_b, a.wkv, list(a.wo_a), a.wo_b


def dsv4_norms(block):
    """The ``(q_norm, kv_norm)`` weighted RMSNorms of a DSV4 block (on the
    q_a latent and the shared kv row). The per-head q norm after q_b is
    unweighted (spec ``weight: None``, eps = rms_norm_eps)."""
    a = block.attn
    return a.q_norm, a.kv_norm


def dsv4_compressor_projections(block):
    """The ``(wkv, wgate)`` linears of a CSA/HCA layer's compressor, or
    ``(None, None)`` on a sliding layer. The CSA indexer's own compressor and
    projections are dead weight for training (dense-exact below the budget)."""
    comp = getattr(block.attn, "compressor", None)
    if comp is None:
        return None, None
    return comp.wkv, comp.wgate


def mla_norms(block):
    """The ``(q_a_layernorm, kv_a_layernorm)`` RMSNorms of an MLA block
    (``q_a_layernorm`` is ``None`` on a direct-q layout)."""
    a = block.attn
    return a.q_a_layernorm, a.kv_a_layernorm


def short_conv_projections(block):
    """Return the ``(in_proj, out_proj)`` linears of a ShortConv block (LFM2 /
    LFM2-MoE ``conv`` layers). ``in_proj`` is ``[hidden, 3*hidden]`` (the
    ``b | c | x`` split), ``out_proj`` is ``[hidden, hidden]``."""
    a = block.attn
    return a.in_proj, a.out_proj


def gdn_norm_spec(block) -> dict:
    """``norm_spec``-shaped description of a GatedDeltaNet block's gated
    RMSNorm (applied per value head over ``v_head_dim``, then multiplied by
    ``act(z)`` -- see ``training.gdn.gdn_gated_rmsnorm``), plus
    ``gate_activation``: the module's ``"silu"`` (Qwen3.5/3.6) or
    ``"sigmoid"`` (Qwen3.8-Flash-Next ``output_gate_type``). Read from the
    module, never assumed -- the two differ at every token, silently."""
    from ..modules import GatedRMSNorm
    norm = block.attn.norm
    assert isinstance(norm, GatedRMSNorm), \
        f"expected GatedRMSNorm on GatedDeltaNet, got {type(norm).__name__}"
    act = str(getattr(norm, "gate_activation", "silu"))
    assert act in ("silu", "sigmoid"), \
        f"{norm.key}: unsupported GatedRMSNorm gate activation {act!r}"
    assert not getattr(norm, "gate_first", False), \
        f"{norm.key}: gate-first (Mamba2 group norm) GatedRMSNorm is not supported"
    assert int(getattr(norm, "groups", 1) or 1) == 1, \
        f"{norm.key}: grouped GatedRMSNorm is not supported"
    return {
        "weight": _frozen_normal(norm.weight),
        "eps": norm.rms_norm_eps,
        "bias": float(getattr(norm, "constant_bias", 0.0)),
        "scale": 1.0,
        "gate_activation": act,
    }


def mlp_projections(block):
    """
    Return ``(gates, ups, downs)`` linear lists of one block's gated MLP. Each is
    a list because a very wide MLP may be sliced across the intermediate dim.
    For a BlockSparseMLP block use the ``moe_*`` accessors below instead (there
    the lists are per-EXPERT, not intermediate-dim slices).
    """
    m = block.mlp
    return m.gates, m.ups, m.downs


def moe_expert_projections(block):
    """Return the per-expert ``(gates, ups, downs)`` linear lists of a
    BlockSparseMLP block -- one entry per routed expert, index == expert id."""
    m = block.mlp
    return m.gates, m.ups, m.downs


def moe_shared_projections(block):
    """Return the shared expert's ``(gates, ups, downs)`` slice lists of a
    BlockSparseMLP block (same shape as ``mlp_projections``), or ``None`` when
    the architecture has no shared expert (Qwen3-MoE)."""
    sh = block.mlp.shared_experts
    if sh is None:
        return None
    return sh.gates, sh.ups, sh.downs


def moe_router_linear(block):
    """The router gate ``Linear`` (``[hidden, num_experts]``, fp16) of a
    BlockSparseMLP block. Kept frozen by the training forward: adapting the
    router under a top-k discontinuity destabilizes expert selection, and no
    mainstream MoE-LoRA recipe trains it."""
    return block.mlp.routing_gate


def moe_shared_gate_linear(block):
    """The sigmoid shared-expert gate ``Linear`` (``[hidden, 1]``) of a
    BlockSparseMLP block (Qwen3.5-MoE), or ``None``. The inference kernel adds
    ``shared_out * sigmoid(shared_gate(x))`` to the routed output."""
    return block.mlp.shared_gate


def moe_extra_norms(block):
    """The optional Gemma4-layout norms of a BlockSparseMLP block, as the
    ``(router_pre, routed_pre, routed_post, shared_post)`` module tuple
    (each ``RMSNorm`` or ``None``). In the inference forward: ``router_pre``
    normalizes the routing input (Gemma4 also scales it by
    ``hidden_size**-0.5`` via the module's ``constant_scale``), ``routed_pre``
    normalizes the routed experts' input, ``routed_post`` the routed output
    sum, and ``shared_post`` the shared expert's output -- all read from the
    module so ``norm_spec`` reproduces their exact epsilon/scale/bias."""
    m = block.mlp
    return (m.router_pre_norm, m.routed_pre_norm,
            m.routed_post_norm, m.shared_experts_post_norm)


def rms_norm_eps(norm) -> float:
    """The epsilon of an exllamav3 ``RMSNorm`` module."""
    return norm.rms_norm_eps


# --- variable-length (packed) attention ------------------------------------

def attn_varlen(q, k, v, cu_seqlens, max_seqlen, sm_scale,
                window: int = -1, softcap: float = 0.0):
    """
    Variable-length (packed) attention for the training forward, via exllamav3's
    own autograd-capable flash wrapper -- the O(t) primitive that lets sample
    packing isolate documents without ever building a ``[t, t]`` mask.

    ``q`` / ``k`` / ``v`` are ``[total_tokens, num_heads, head_dim]``: every
    document of a packed batch concatenated into one token stream, with
    ``cu_seqlens`` (int32, shape ``[num_docs + 1]``, cumulative document lengths)
    marking the per-document boundaries so a document never attends across one.
    ``max_seqlen`` is the longest document length.

    Routed through ``attention_fn.attn_dispatch`` with no cache, so it skips every
    paged/cache backend and lands on ``fn_flash_attn_varlen_func`` -- the upstream
    ``flash_attn_varlen_func``, which (unlike exllamav3's inference kernels) is NOT
    wrapped in ``inference_mode`` and has a real backward. Requires
    ``head_dim <= 256`` (FA2 limit; the caller routes larger heads elsewhere).

    ``window > 0`` is the native forward's QUERY-INCLUSIVE sliding window (a
    token attends to itself + ``window - 1`` previous tokens = ``window`` total;
    ``block_metadata`` derives it as the module's past-tokens-only
    ``sliding_window`` + 1). We hand ``attn_dispatch`` ``window - 1`` because its
    ``get_window_size`` wraps the value as the FA2 left-window ``(w, 0)`` -- so the
    result matches the eager reference's ``-window`` diagonal exactly. ``softcap``
    applies tanh logit softcapping. Returns ``[total_tokens, num_heads, head_dim]``.
    """
    from ..modules.attention_fn.dispatch import attn_dispatch
    # attn_dispatch reads [bsz, q_len, num_heads, head_dim]; the varlen backend
    # asserts bsz == 1 and squeezes it, reading boundaries from cu_seqlens. Present
    # the flattened stream as a single batch row; the result is [total, nh, hd].
    window_size = (int(window) - 1) if (window and window > 0) else None
    return attn_dispatch(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        cache=None,
        causal=True,
        sm_scale=float(sm_scale),
        cu_seqlens=cu_seqlens,
        max_seqlen=int(max_seqlen),
        window_size=window_size,
        softcap=float(softcap or 0.0),
    )


# --- frozen quantized linears ----------------------------------------------

# Optional dequant profiling (``--profile-dequant``). When enabled, every
# frozen-weight reconstruction (block linears, LM head, head slices) times
# itself into this mutable dict -- answering "how much of a training step is
# trellis reconstruction", the load-bearing question for the dequant-count
# optimizations (see doc/qlora_optimization_audit.md A1). Costs a device sync
# either side of each reconstruction while enabled (a diagnostic mode; the
# measured share of wall time is still representative). Disabled = one global
# read per call, negligible next to the matmul it precedes.
_DEQUANT_PROFILE: Optional[dict] = None


def profile_dequant(state: Optional[dict]) -> None:
    """Enable (pass a dict with ``calls``/``s`` keys) or disable (pass None)
    reconstruction timing for all frozen-weight closures."""
    global _DEQUANT_PROFILE
    _DEQUANT_PROFILE = state


def _timed_reconstruct(fn):
    """Wrap a weight-producing closure so it accumulates into the profile dict
    when profiling is on. The check runs at call time, so enabling/disabling
    mid-run needs no closure rebuild."""
    def wrapped(*a):
        p = _DEQUANT_PROFILE
        if p is None:
            return fn(*a)
        import time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        w = fn(*a)
        if w.is_cuda:
            torch.cuda.synchronize(w.device)
        p["calls"] += 1
        p["s"] += time.perf_counter() - t0
        return w
    return wrapped


def is_loaded(linear) -> bool:
    """True once a native ``Linear`` has its inner (trellis / fp16) weight."""
    return getattr(linear, "inner", None) is not None


def linear_device(linear):
    """Device a linear's weights live on (the trellis device when quantized)."""
    try:
        return linear.inner.trellis.device
    except Exception:
        return getattr(linear, "device", None)


# --- backward-phase dequant cache (audit A1; OPT-IN via --dequant-cache) ----
#
# Under (unconditional) gradient checkpointing every frozen weight is
# reconstructed three times per step: outer forward, checkpoint-recompute
# forward, and the Function backward. The recompute and the backward run
# back-to-back per block, so caching the weight between exactly those two
# calls removes one reconstruction per step at the memory cost of one block's
# frozen weights held live at a time.
#
# Box-measured (Session 30): under the FAST dequant path this trade is a net
# loss -- inner-only reconstructions are so cheap (0.06-0.23 ms) that the
# cache's bookkeeping and allocator pressure cost ~1-4% tok/s AND +0.5-1.5 GB
# peak VRAM (worst on many-expert MoE). It therefore defaults OFF; it can pay
# only under --dequant-mode legacy, where each avoided reconstruction is a
# 5-7 ms full get_weight_tensor.
#
# The trainer wraps ``loss.backward()`` in ``backward_dequant_cache()``; while
# the phase is active a closure's first call stores its result and the second
# call returns-and-evicts it (so a block's weights free as its backward
# consumes them). Outside the phase every call is a plain reconstruction --
# eval, validation and the outer forward are untouched. Correct for any call
# count: an unpaired store is dropped when the phase ends, a third call is
# just a fresh miss. Closures whose result is consumed only once per backward
# (the fused-CE head) are deliberately NOT cache-wrapped -- they would hold
# the weight for the whole phase for no reuse.
_BWD_WEIGHT_CACHE: Optional[dict] = None


class backward_dequant_cache:
    """Context manager arming the recompute->backward weight cache (no-op when
    constructed with ``enable=False``, so call sites stay unconditional)."""

    def __init__(self, enable: bool = True):
        self.enable = enable

    def __enter__(self):
        global _BWD_WEIGHT_CACHE
        self.prev = _BWD_WEIGHT_CACHE
        if self.enable:
            _BWD_WEIGHT_CACHE = {}
        return self

    def __exit__(self, *exc):
        global _BWD_WEIGHT_CACHE
        _BWD_WEIGHT_CACHE = self.prev
        return False


def _cached_weight(key, fn):
    """Wrap a weight closure with the backward-phase store/evict-on-hit cache.
    Sits OUTSIDE the profiling wrapper so cache hits cost (and count) nothing
    toward the ``--profile-dequant`` reconstruction share."""
    def wrapped():
        c = _BWD_WEIGHT_CACHE
        if c is None:
            return fn()
        w = c.pop(key, None)
        if w is None:
            w = fn()
            c[key] = w
        return w
    return wrapped


# --- dequant mode (audit A1, the cheap-per-reconstruction half) -------------

# "fast": DiffLinear runs trellis linears through EXL3LoRAHadFunction --
# reconstruct only the inner weight and apply the Hadamard/sign transforms to
# the activations (the same math as inference's ``reconstruct_hgemm``),
# skipping the four full-weight transform passes + dtype cast that
# ``get_weight_tensor`` performs per reconstruction. "legacy": the original
# full-weight closure path, kept for A/B measurement and as the fallback
# (fp16 inners, quant-aware runs, float64 gradchecks use it regardless).
_DEQUANT_MODE = "fast"


def set_dequant_mode(mode: str) -> None:
    assert mode in ("fast", "legacy"), f"unknown dequant mode {mode!r}"
    global _DEQUANT_MODE
    _DEQUANT_MODE = mode


def dequant_mode() -> str:
    return _DEQUANT_MODE


_HAD_128_CACHE: dict = {}


def hadamard_128(device, dtype: torch.dtype) -> torch.Tensor:
    """The normalized 128x128 Hadamard matrix used by the EXL3 transforms
    (orthogonal and symmetric: H^-1 = H^T = H), cached per device/dtype.
    (util.hadamard.get_hadamard_dt copies to device on EVERY call -- fine for
    one-off inference preapplies, a per-linear-per-step alloc+cast here.)"""
    key = (str(device), dtype)
    had = _HAD_128_CACHE.get(key)
    if had is None:
        from ..util.hadamard import get_hadamard_dt
        had = get_hadamard_dt(128, device, dtype, 128 ** -0.5)
        _HAD_128_CACHE[key] = had
    return had


def frozen_trellis_parts(linear, dtype: Optional[torch.dtype] = None):
    """
    The pieces of a standard trellis linear needed for activation-side
    transforms: ``(inner_fn, suh, svh)`` where ``inner_fn()`` reconstructs the
    INNER ``[in, out]`` weight (no Hadamard/sign transforms) and ``suh``/
    ``svh`` are the input/output sign vectors, such that

        get_weight_tensor() == diag(suh) @ H_128 @ inner @ H_128 @ diag(svh)

    (H block-diagonal at 128). Returns ``None`` when the layer can't take the
    activation-side path (fp16 inner, or feature dims not 128-aligned) --
    callers fall back to ``frozen_weight_closure``.

    ``dtype`` (fp16/bf16) is the dtype the reconstruct kernel EMITS and the
    sign vectors are returned in -- pass the compute dtype so the training
    Function's per-call ``.to()`` casts (full-weight in backward, activations
    both ways in forward) all become no-ops. The kernel dequantizes in fp16
    regardless and rounds once at the output store, bit-identical to
    reconstructing in half and casting. ``None`` keeps the stored fp16.
    ``suh``/``svh`` are cast ONCE here (call at setup time, not per step).
    """
    inner = getattr(linear, "inner", None)
    if getattr(inner, "quant_type", None) != "exl3":
        return None
    if inner.in_features % 128 or inner.out_features % 128:
        return None
    suh, svh = inner.suh, inner.svh
    if suh is None or svh is None:
        return None
    # Only fp16/bf16 can be emitted by the kernel; for any other compute
    # dtype (fp32 debug runs) keep the fp16 inner and let the Function's
    # per-call casts handle it, exactly the pre-S36 behavior.
    wdtype = dtype if dtype in (torch.half, torch.bfloat16) else torch.half
    suh, svh = suh.to(wdtype), svh.to(wdtype)
    inner_fn = _cached_weight(
        (id(inner), "inner", wdtype),
        _timed_reconstruct(lambda: inner.get_inner_weight_tensor(out_dtype = wdtype)),
    )
    return inner_fn, suh, svh


def frozen_weight_closure(linear, dtype: torch.dtype) -> Callable[[], torch.Tensor]:
    """
    Closure that reconstructs the frozen effective weight (``[in, out]``) from the
    EXL3 trellis on every call, cast to ``dtype``. Recomputing rather than caching
    is what lets the backward pass avoid stashing the dense weight (the
    backward-phase cache above then removes the recompute->backward duplicate).
    """
    inner = linear.inner
    return _cached_weight(
        (id(inner), dtype),
        _timed_reconstruct(lambda: inner.get_weight_tensor().to(dtype)),
    )


def linear_quant_bits(linear) -> Optional[float]:
    """
    Bits-per-weight of a linear's frozen storage, or ``None`` when the layer is
    not trellis-quantized (an fp16/bf16 inner has no quantization error to be
    aware of). Reads ``LinearEXL3.K`` (trellis bits per weight); behind the
    seam so the quant-aware training modes never touch exllamav3 internals.
    """
    k = getattr(getattr(linear, "inner", None), "K", None)
    return float(k) if k is not None else None


def frozen_bias(linear, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """The linear's frozen bias cast to ``dtype``, or ``None`` if it has none."""
    get_bias = getattr(linear.inner, "get_bias_tensor", None)
    if get_bias is None:
        return None
    b = get_bias()
    return b.to(dtype) if b is not None else None


def head_weight_closure(lm_head) -> Callable[[], torch.Tensor]:
    """
    Closure for the frozen LM-head weight in ``[hidden, vocab]`` orientation (no
    dtype cast; the fused-CE head promotes to >=fp32 internally).
    """
    inner = lm_head.inner
    return _timed_reconstruct(lambda: inner.get_weight_tensor())


def head_weight_slice_closure(lm_head):
    """
    For chunked-vocab head loss: return ``(slice_fn, out_features, granularity)``
    where ``slice_fn(n_start, n_features) -> [hidden, n_features]`` reconstructs only
    those output columns, or ``None`` if the head can't slice efficiently.

    An EXL3 head reconstructs just the requested columns (``get_weight_tensor_slice``)
    so the fused CE never materializes the full ``[hidden, vocab]`` weight + its fp32
    upcast -- the dominant memory spike on the output device for large vocabularies.
    Other head types (e.g. an unquantized ``[hidden, vocab]`` tensor) fall back to a
    plain column index, which still avoids the full-vocab fp32 logits/softmax.
    """
    inner = lm_head.inner
    sliced = getattr(inner, "get_weight_tensor_slice", None)
    if sliced is not None:
        gran = getattr(inner, "RECONSTRUCT_SLICE_GRANULARITY_N", None)
        if gran is None:
            # Module-level constant on the EXL3 linear's module.
            import exllamav3.modules.quant.exl3 as _exl3
            gran = _exl3.RECONSTRUCT_SLICE_GRANULARITY_N
        return _timed_reconstruct(lambda s, n: sliced(s, n)), inner.out_features, gran
    # Generic fallback: index the full (already-resident) weight. No reconstruction
    # spike to avoid, but the chunked CE still bounds the logits/softmax memory.
    get_full = getattr(inner, "get_weight_tensor", None)
    if get_full is None:
        return None
    out_features = getattr(inner, "out_features", None)
    if out_features is None:
        return None
    return _timed_reconstruct(lambda s, n: get_full()[:, s:s + n]), out_features, 1


# --- token embedding -------------------------------------------------------

def embed_weight(embed) -> torch.Tensor:
    """The input-embedding weight tensor (``[vocab, hidden]``) of an ``Embedding``
    module -- the tensor to clone when fully training the embeddings."""
    return embed.embedding.weight


def embed_apply(embed, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the ``Embedding`` module's optional multiplier / normalization to
    already-looked-up hidden states (shared by the frozen and trainable paths)."""
    if getattr(embed, "multiplier", 1.0) != 1.0:
        hidden = hidden * embed.multiplier
    if getattr(embed, "normalize", False):
        hidden = hidden * (hidden.shape[-1] ** 0.5)
    return hidden


def embed_tokens(embed, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Look up token embeddings via exllamav3's ``Embedding`` module, applying its
    optional multiplier / normalization. Returns hidden states on the embedding's
    own device (which may be CPU even when the decoder is on GPU).
    """
    table = embed.embedding
    hidden = table(input_ids.to(table.weight.device))
    return embed_apply(embed, hidden)


# --- runtime LoRA slots (so native generation reflects the adapter) --------

def set_runtime_lora(linear, owner, a: torch.Tensor, b: torch.Tensor) -> None:
    """
    Install adapter tensors into a native ``Linear``'s runtime LoRA slots, keyed
    by ``owner``, so ``model.forward`` / generation applies them. ``a`` / ``b``
    are moved to the linear's device.
    """
    linear.lora_a_tensors[owner] = a.to(linear.device)
    linear.lora_b_tensors[owner] = b.to(linear.device)


def clear_runtime_lora(linear, owner) -> None:
    """Remove ``owner``'s adapter tensors from a native ``Linear``'s LoRA slots."""
    linear.lora_a_tensors.pop(owner, None)
    linear.lora_b_tensors.pop(owner, None)
