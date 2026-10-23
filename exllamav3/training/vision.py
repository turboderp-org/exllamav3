"""
Image+text (vision) support for the native QLoRA forward.

The design follows how exllamav3 itself serves a VLM at inference, and how
Axolotl fine-tunes one: the vision tower + projector stay FROZEN and only the
language model's adapters train. That makes the image features a constant of
each training example, so they are produced ONCE by exllamav3's own vision
component (``Model.from_config(config, component="vision")`` --
``get_image_embeddings``, the exact forward the generator runs) and spliced
into the text tower's embedding stream at the image token positions, the same
way ``modules.Embedding.forward`` does with ``params["indexed_embeddings"]``.
Nothing about the vision math is re-implemented here; the differentiable side
only has to reproduce what the text tower does with the features:

  * the embedding splice (``native_llama.forward``, ``mm=``);
  * mRoPE (Qwen-VL): 3-D [t, h, w] position ids for image tokens
    (``mrope_position_ids`` -- a Python mirror of the ``gen_mrope_pos_ids``
    kernel) and the per-frequency-band split (``mrope_freqs`` -- a mirror of
    ``util.rope.RoPE.get_mrope_freqs``);
  * deepstack (Qwen3-VL / Qwen3.5-VL): the vision tower's intermediate feature
    maps added onto the image positions after the first N decoder blocks
    (``backbone.deepstack_layout``);
  * bidirectional attention within an image span (Gemma4:
    ``backbone.uses_noncausal_mm_spans``).

``VisionEncoder`` wraps the frozen vision component with an in-RAM feature
cache (fp16, byte-budgeted) so a dataset's images are encoded once at build
time and re-encoded only on a cache miss; ``build_mm_batch`` assembles the
per-batch splice tensors the forward consumes.

This module imports exllamav3 lazily (inside ``VisionEncoder``) so the pure
math helpers stay importable on CPU without the compiled extension.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import torch

# Placeholder inside an ``ImageFeatures.token_ids`` layout for a position that
# takes a row of the image features instead of a text-token embedding. Same
# convention as the architectures' raw ``token_string`` before MMEmbedding
# allocates runtime ids for it.
IMAGE_SLOT = -1


# --- mRoPE ------------------------------------------------------------------

def mrope_position_ids(seq_len: int, images: list, merge_size: int) -> torch.Tensor:
    """
    3-D mRoPE position ids ``[3, seq_len]`` (t / h / w rows) for one sequence.

    ``images`` lists ``(start, n_tokens, grid_thw)`` per image in sequence
    order: the image's feature rows occupy positions ``start .. start +
    n_tokens - 1`` (a contiguous run on every mRoPE arch) and ``grid_thw`` is
    the vision tower's patch grid ``(t, h, w)`` in PATCH units (``h``/``w`` are
    divided by ``merge_size`` to get the token grid, as the kernel does).

    A Python mirror of ``exllamav3_ext gen_mrope_pos_ids`` (which the generator
    runs through ``RoPE.get_mrope_freqs``) -- and of HF's ``get_rope_index``:
    text tokens advance one shared position on all three axes; an image's
    tokens take ``base + (t_idx, h_idx, w_idx)`` where ``base`` is the position
    the next text token would have had, and the text after it resumes at
    ``max(image positions) + 1``. Vectorized per segment rather than per token.
    """
    pos = torch.empty(3, seq_len, dtype=torch.long)
    spans = sorted((int(s), int(n), tuple(int(g) for g in grid)) for s, n, grid in images)
    # Kernel state: `base` is the offset the NEXT image's grid indexes from and
    # is only advanced by text tokens; `next_base` is one past the largest
    # position emitted so far, which the next text token resumes from. (So two
    # images with no text between them index from the same base -- the
    # kernel's rule, mirrored exactly; in practice every arch wraps an image
    # in start/end text tokens, so it never arises.)
    base = 0
    next_base = 0
    cur = 0           # next sequence index to fill
    for start, n, (gt, gh, gw) in spans:
        assert start >= cur, "overlapping image spans"
        if start > cur:                                  # text run before the image
            L = start - cur
            base = next_base
            pos[:, cur:start] = torch.arange(base, base + L).unsqueeze(0)
            base += L
            next_base = base
        ghm, gwm = gh // merge_size, gw // merge_size
        assert gt * ghm * gwm == n, \
            f"image span of {n} tokens does not match grid {(gt, gh, gw)} / " \
            f"merge {merge_size} = {gt * ghm * gwm} tokens"
        k = torch.arange(n)
        pos[0, start:start + n] = base + (k // (gwm * ghm)) % gt
        pos[1, start:start + n] = base + (k // gwm) % ghm
        pos[2, start:start + n] = base + k % gwm
        next_base = max(next_base, base + max(gt, ghm, gwm))
        cur = start + n
    if cur < seq_len:                                    # trailing text
        base = next_base
        pos[:, cur:] = torch.arange(base, base + (seq_len - cur)).unsqueeze(0)
    return pos


def mrope_freqs(position_ids: torch.Tensor, inv_freq: torch.Tensor,
                mrope_section) -> torch.Tensor:
    """
    Per-token rotation angles ``[b, t, n]`` (``n = inv_freq.numel()``) from 3-D
    positions ``[3, b, t]`` under Qwen's INTERLEAVED mRoPE: frequency band
    ``i`` rotates by the temporal position, except bands ``i = 1, 4, 7, ...``
    below ``3 * section[1]`` (height) and ``i = 2, 5, 8, ...`` below
    ``3 * section[2]`` (width). A mirror of ``RoPE.get_mrope_freqs`` -- the
    interleaving exllamav3's generator applies to every mRoPE arch it serves.
    """
    pf = position_ids.float()                                     # [3, b, t]
    f = pf.unsqueeze(-1) * inv_freq.float().view(1, 1, 1, -1)    # [3, b, t, n]
    out = f[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = int(mrope_section[dim]) * 3
        idx = slice(offset, length, 3)
        out[..., idx] = f[dim][..., idx]
    return out


# --- image features ---------------------------------------------------------

@dataclass
class ImageFeatures:
    """One encoded image: its token layout in the text stream plus the frozen
    features that fill the ``IMAGE_SLOT`` positions."""
    token_ids: list             # arch token string; IMAGE_SLOT where a feature row goes
    n_tokens: int               # number of feature rows (== token_ids.count(IMAGE_SLOT))
    grid_thw: Optional[tuple]   # vision patch grid (mRoPE archs) or None
    merge_size: Optional[int]   # spatial merge (mRoPE archs) or None
    embeds: Optional[torch.Tensor]              # [n_tokens, hidden] (CPU) or None if evicted
    deepstack: Optional[list]                   # per-deepstack-layer [n_tokens, hidden], or None
    nbytes: int = 0

    @property
    def slot_offsets(self) -> list:
        """Offsets (within ``token_ids``) of the feature-row positions, in
        feature-row order."""
        return [i for i, t in enumerate(self.token_ids) if t == IMAGE_SLOT]

    def has_embeds(self) -> bool:
        return self.embeds is not None


def downscale_image(image, max_pixels: int):
    """Shrink ``image`` (PIL) so ``w*h <= max_pixels``, keeping aspect ratio;
    returns it unchanged when it already fits or ``max_pixels`` is 0. Applied
    BEFORE the architecture's own preprocessing (which re-fits the size to its
    patch grid / pixel bounds), so it only ever lowers the image token count --
    the ``image_size`` lever Axolotl exposes for the same reason."""
    if not max_pixels or max_pixels <= 0:
        return image
    w, h = image.size
    if w * h <= max_pixels:
        return image
    from PIL import Image
    s = (max_pixels / float(w * h)) ** 0.5
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return image.resize((nw, nh), resample=Image.Resampling.LANCZOS)


def features_from_mme(mme, store_dtype: torch.dtype = torch.float16) -> ImageFeatures:
    """``ImageFeatures`` from an inference ``MMEmbedding`` (the object
    ``get_image_embeddings`` returns): the token layout with the runtime ids
    MMEmbedding allocated mapped back to ``IMAGE_SLOT``, and the features
    (plus deepstack maps) copied to CPU in ``store_dtype``."""
    from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
    ids = mme.token_string[0].tolist()
    # MMEmbedding replaced the layout's -1 placeholders with a run of runtime
    # ids [first_index, first_index + n); map them back to the position-
    # independent IMAGE_SLOT layout (and check they are in row order, which
    # the splice relies on).
    rows = [t - mme.first_index for t in ids if t >= FIRST_MM_EMBEDDING_INDEX]
    assert rows == list(range(mme.mm_length)), \
        "image token string does not list the feature rows in order"
    layout = [IMAGE_SLOT if t >= FIRST_MM_EMBEDDING_INDEX else int(t) for t in ids]
    emb = mme.embeddings.detach().to("cpu", store_dtype).clone()
    ds = None
    if mme.deepstack_embeddings is not None:
        ds = [d.detach().to("cpu", store_dtype).clone() for d in mme.deepstack_embeddings]
    nbytes = emb.numel() * emb.element_size()
    if ds:
        nbytes += sum(d.numel() * d.element_size() for d in ds)
    return ImageFeatures(
        token_ids=layout, n_tokens=int(mme.mm_length),
        grid_thw=tuple(int(g) for g in mme.grid_thw) if mme.grid_thw is not None else None,
        merge_size=(int(mme.mrope_merge_size)
                    if mme.mrope_merge_size is not None else None),
        embeds=emb, deepstack=ds, nbytes=nbytes,
    )


class VisionEncoder:
    """
    The frozen vision component of an exllamav3 VLM, as a feature service for
    training. ``encode(key, loader)`` returns the ``ImageFeatures`` for an
    image (``loader()`` -> PIL image, called only on a cache miss), running
    exllamav3's own ``get_image_embeddings`` -- the exact forward inference
    uses, so the features the adapter trains against are the features it will
    see when served.

    Features are cached in CPU RAM (``store_dtype``, default fp16) up to
    ``cache_bytes``; past the budget an image's LAYOUT is still remembered
    (token count / grid -- what the dataset builder needs) but its features are
    dropped and recomputed on the next request, which needs the tower to stay
    loaded. ``all_cached`` tells the trainer whether it can ``unload()`` the
    tower after the dataset build to free its VRAM.
    """

    def __init__(self, config, tokenizer, device="cuda:0", max_pixels: int = 0,
                 cache_bytes: int = 8 << 30, store_dtype: torch.dtype = torch.float16,
                 progressbar: bool = True):
        from ..model import Model
        assert "vision" in getattr(config, "model_classes", {}), \
            f"{getattr(config, 'architecture', '?')} has no vision component; " \
            f"this is a text-only model"
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.max_pixels = int(max_pixels or 0)
        self.cache_bytes = int(cache_bytes)
        self.store_dtype = store_dtype
        self.model = Model.from_config(config, component="vision")
        self.model.load(device=device, progressbar=progressbar)
        self.loaded = True
        self._cache: dict = {}          # key -> ImageFeatures (embeds may be None)
        self._cached_bytes = 0
        self.n_encoded = 0              # forward passes run
        self.n_evicted = 0              # features dropped for the byte budget

    # -- lifecycle --

    def unload(self):
        """Free the vision tower (its VRAM); cached features stay usable."""
        if self.loaded:
            self.model.unload()
            self.loaded = False

    @property
    def all_cached(self) -> bool:
        return all(f.has_embeds() for f in self._cache.values())

    @property
    def hidden_size(self) -> Optional[int]:
        for f in self._cache.values():
            if f.embeds is not None:
                return int(f.embeds.shape[-1])
        return None

    def stats(self) -> dict:
        n = len(self._cache)
        toks = [f.n_tokens for f in self._cache.values()]
        return {
            "images": n,
            "encoded": self.n_encoded,
            "cached": sum(1 for f in self._cache.values() if f.has_embeds()),
            "cache_gb": self._cached_bytes / float(1 << 30),
            "tokens_mean": (sum(toks) / n) if n else 0.0,
            "tokens_max": max(toks) if toks else 0,
        }

    # -- encoding --

    def _run(self, image) -> ImageFeatures:
        """Run the frozen vision forward on one PIL image."""
        assert self.loaded, "vision tower was unloaded; cannot encode a cache miss"
        image = downscale_image(image, self.max_pixels)
        with torch.inference_mode():
            mme = self.model.get_image_embeddings(tokenizer=self.tokenizer, image=image)
        self.n_encoded += 1
        return features_from_mme(mme, self.store_dtype)

    def encode(self, key, loader: Callable) -> ImageFeatures:
        """Features for image ``key`` (any hashable). ``loader()`` must return
        the PIL image; it is called only when the features are not cached."""
        hit = self._cache.get(key)
        if hit is not None and hit.has_embeds():
            return hit
        return self._store(key, self._run(loader()), hit)

    def register_mme(self, key, mme) -> ImageFeatures:
        """Seed the cache from an inference ``MMEmbedding`` the host already
        computed for serving (one tower forward feeds both generation and
        training -- the realtime chat's ``/image``). Returns the features."""
        hit = self._cache.get(key)
        if hit is not None and hit.has_embeds():
            return hit
        return self._store(key, features_from_mme(mme, self.store_dtype), hit)

    def _store(self, key, feats: ImageFeatures, hit: Optional[ImageFeatures]) -> ImageFeatures:
        if hit is not None:
            assert hit.token_ids == feats.token_ids, \
                f"image {key!r} re-encoded to a different token layout"
        if self._cached_bytes + feats.nbytes <= self.cache_bytes:
            self._cache[key] = feats
            self._cached_bytes += feats.nbytes
            return feats
        # Over budget: keep the layout, drop the features (recomputed next time).
        self.n_evicted += 1
        self._cache[key] = ImageFeatures(
            token_ids=feats.token_ids, n_tokens=feats.n_tokens,
            grid_thw=feats.grid_thw, merge_size=feats.merge_size,
            embeds=None, deepstack=None, nbytes=0)
        return feats

    def layout(self, key) -> Optional[ImageFeatures]:
        """The cached layout entry for ``key`` (features possibly evicted), or
        None if never encoded."""
        return self._cache.get(key)


# --- batch assembly -----------------------------------------------------------

def build_mm_batch(rows: list, seq_len: int, dtype: torch.dtype,
                   device=None) -> Optional[dict]:
    """
    Assemble the multimodal splice for one padded batch.

    ``rows`` has one entry per batch row: a list of ``(start, features)`` with
    ``start`` the index (in that row's ``input_ids``) of the first token of the
    image's token layout and ``features`` its ``ImageFeatures`` WITH embeds.
    Returns None when no row carries an image, else::

        {"index":     LongTensor [b, seq_len]  -1 for text, else row into "embeds"
         "embeds":    Tensor [N, hidden]       all image feature rows of the batch
         "deepstack": [Tensor [N, hidden], ...] or None   (Qwen3-VL deepstack maps)
         "spans":     LongTensor [b, seq_len]  -1 for text, else the id of the
                                              contiguous image-feature run the
                                              position belongs to (bidirectional
                                              attention spans; Gemma4)}

    ``spans`` follows the inference rule (Gemma4 ``_prepare_noncausal_mm_spans``):
    a span is a maximal run of consecutive feature positions, so an arch whose
    layout interleaves text tokens inside an image (Mistral3's ``[IMG_BREAK]``)
    gets one span per row of the image -- exactly what inference would build.
    """
    if not any(rows):
        return None
    bsz = len(rows)
    index = torch.full((bsz, seq_len), -1, dtype=torch.long)
    embeds, deepstack, n = [], None, 0
    for b, imgs in enumerate(rows):
        for start, f in sorted(imgs, key=lambda x: x[0]):
            assert f.embeds is not None, "build_mm_batch needs features with embeds"
            offs = torch.tensor(f.slot_offsets, dtype=torch.long) + int(start)
            assert int(offs.max()) < seq_len, "image span exceeds the padded sequence"
            index[b, offs] = torch.arange(n, n + f.n_tokens)
            embeds.append(f.embeds)
            if f.deepstack is not None:
                if deepstack is None:
                    deepstack = [[] for _ in f.deepstack]
                assert len(deepstack) == len(f.deepstack), "deepstack depth varies"
                for lst, d in zip(deepstack, f.deepstack):
                    lst.append(d)
            else:
                assert deepstack is None, "mixed deepstack / no-deepstack images"
            n += f.n_tokens
    is_img = index >= 0
    # span id = cumulative count of run starts, masked to image positions
    starts = is_img.clone()
    starts[:, 1:] &= ~is_img[:, :-1]
    spans = torch.where(is_img, starts.long().cumsum(-1) - 1, torch.full_like(index, -1))
    out = {
        "index": index,
        "embeds": torch.cat(embeds, 0).to(dtype),
        "deepstack": ([torch.cat(lst, 0).to(dtype) for lst in deepstack]
                      if deepstack is not None else None),
        "spans": spans,
    }
    if device is not None:
        out = move_mm(out, device)
    return out


def move_mm(mm: Optional[dict], device) -> Optional[dict]:
    """``mm`` with every tensor on ``device``."""
    if mm is None:
        return None
    return {
        "index": mm["index"].to(device),
        "embeds": mm["embeds"].to(device),
        "deepstack": ([d.to(device) for d in mm["deepstack"]]
                      if mm.get("deepstack") is not None else None),
        "spans": mm["spans"].to(device),
    }


def splice_embeddings(hidden: torch.Tensor, mm: dict) -> torch.Tensor:
    """Replace the rows of ``hidden`` ([b, t, d]) at image positions with the
    image feature rows (``mm["index"]`` selects the row of ``mm["embeds"]``).
    Text rows are untouched, so any scaling/normalization the embedding
    applied to text tokens does NOT touch the image features -- mirroring
    ``modules.Embedding.forward``, which scales the standard embeddings before
    inserting the indexed ones. Out-of-place (autograd-safe)."""
    index = mm["index"].to(hidden.device)
    sel = index >= 0
    if not bool(sel.any()):
        return hidden
    rows = mm["embeds"].to(hidden.device, hidden.dtype)[index[sel]]
    return hidden.masked_scatter(sel.unsqueeze(-1), rows)


def add_deepstack(hidden: torch.Tensor, mm: dict, layer: int) -> torch.Tensor:
    """``hidden`` with deepstack feature map ``layer`` added at the image
    positions (the ``DeepstackEmbed`` module's ``x += deepstack_emb[layer]``,
    where that tensor is zero off the image positions)."""
    ds = mm.get("deepstack")
    if ds is None:
        return hidden
    index = mm["index"].to(hidden.device)
    sel = index >= 0
    if not bool(sel.any()):
        return hidden
    rows = ds[layer].to(hidden.device, hidden.dtype)[index[sel]]
    return hidden.index_put(tuple(sel.nonzero(as_tuple=True)), rows, accumulate=True)
