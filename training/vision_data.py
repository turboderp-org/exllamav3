"""
Image+text rows for the SFT trainer (``--vision``).

Dataset shape (the Axolotl / OpenAI multimodal convention): a ``messages``
column whose turn contents may be content-parts lists::

    {"messages": [
       {"role": "user", "content": [
           {"type": "image", "path": "/data/cat.png"},
           {"type": "text",  "text": "What is in this picture?"}]},
       {"role": "assistant", "content": "A cat on a sofa."}]}

An image part names its pixels one of these ways (checked in this order):
``path`` (local file; relative to the dataset file's directory), ``url``,
``base64``, ``image`` (a PIL image, or HF datasets' ``{"bytes", "path"}``
dict, or a path string), ``image_url`` (OpenAI style, ``{"url": ...}`` or a
string; a ``data:`` URL is decoded). A bare ``{"type": "image"}`` part takes
the next entry of the row's ``images`` column (``--images-key``) -- the
layout of most HF VLM datasets (LLaVA-style, TheCauldron, ...).

How a row becomes tokens. The parts list is flattened to ONE string per turn
with a sentinel (``<$EXL3_IMAGE_k$>``) standing in for each image, so the
existing chat renderers (the hardcoded prompt formats, the model's Jinja
template) render it like any text row; after rendering, the sentinel is
replaced by the architecture's own image token layout (``<|vision_start|>``
+ N feature slots + ``<|vision_end|>`` on Qwen-VL, ``<start_of_image>`` +
256 slots + ``<end_of_image>`` on Gemma3, ``[IMG]``/``[IMG_BREAK]`` rows on
Mistral3, ...) -- the SAME token string exllamav3's inference tokenizer
splices in for an ``MMEmbedding`` alias, so what the adapter trains against is
exactly what the generator will feed it. Text pieces around a sentinel are
tokenized separately (as the inference tokenizer does with aliases), so the
image boundaries are exact. Image tokens are always masked (-100), including
inside a supervised assistant turn.

The frozen vision features come from ``exllamav3.training.vision.VisionEncoder``
(exllamav3's own vision component); per example we keep only ``(start, key)``
per image and re-fetch the features at batch time (cache hit, or a recompute
on the still-loaded tower), so a dataset never has to hold its pixels.
"""

from __future__ import annotations
import base64
import hashlib
import io
import os
import re
from typing import Callable, Optional

import torch

from exllamav3.training.vision import (
    IMAGE_SLOT, ImageFeatures, build_mm_batch, mrope_position_ids,
)

SENTINEL_FMT = "<$EXL3_IMAGE_{}$>"
SENTINEL_RE = re.compile(r"(<\$EXL3_IMAGE_\d+\$>)")
_SENTINEL_K = re.compile(r"<\$EXL3_IMAGE_(\d+)\$>")

IMAGE_PART_TYPES = ("image", "image_url", "input_image")
# Placeholder-token pieces the known VL tokenizers register for image slots;
# what a feature-slot position shows as when an example is decoded. Cosmetic
# only -- the slot's embedding is replaced by the vision feature regardless.
_PLACEHOLDER_PIECES = ("<|image_pad|>", "<image_soft_token>", "[IMG]", "<|image|>",
                       "<image>")


def resolve_placeholder_id(config, tokenizer) -> int:
    """Token id written into ``input_ids`` at image feature-slot positions."""
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v >= 0:
            return int(v)
    for piece in _PLACEHOLDER_PIECES:
        tid = tokenizer.extended_piece_to_id.get(piece)
        if tid is not None:
            return int(tid)
    pad = tokenizer.pad_token_id
    if pad is None or pad < 0:
        pad = tokenizer.eos_token_id or 0
    return int(pad)


# --- image references ----------------------------------------------------------

class ImageRef:
    """A lazily-loadable image: a stable cache ``key`` plus ``load() -> PIL``."""
    __slots__ = ("key", "load")

    def __init__(self, key, load: Callable):
        self.key = key
        self.load = load


def _open_bytes(b: bytes):
    from PIL import Image
    im = Image.open(io.BytesIO(b))
    im.load()
    return im


def _open_path(path: str):
    from PIL import Image
    im = Image.open(path)
    im.load()
    return im


def _open_url(url: str, timeout: float = 30.0):
    if url.startswith("data:"):
        head, _, payload = url.partition(",")
        if ";base64" in head:
            return _open_bytes(base64.b64decode(payload))
        raise ValueError("only base64 data: URLs are supported")
    try:
        import requests
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        return _open_bytes(r.content)
    except ImportError:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as f:
            return _open_bytes(f.read())


def _pil_of(obj):
    """PIL image from a PIL image / HF ``{"bytes","path"}`` dict / path str."""
    from PIL import Image
    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, dict):
        if obj.get("bytes"):
            return _open_bytes(obj["bytes"])
        if obj.get("path"):
            return _open_path(obj["path"])
        raise ValueError("image dict carries neither bytes nor path")
    if isinstance(obj, (bytes, bytearray)):
        return _open_bytes(bytes(obj))
    if isinstance(obj, str):
        if obj.startswith(("http://", "https://", "data:")):
            return _open_url(obj)
        return _open_path(obj)
    raise ValueError(f"unsupported image object {type(obj).__name__}")


def is_image_part(part) -> bool:
    return isinstance(part, dict) and (part.get("type") or "text") in IMAGE_PART_TYPES


def has_image_parts(messages) -> bool:
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, (list, tuple)) and any(is_image_part(p) for p in c):
            return True
    return False


class ImageResolver:
    """Turns image parts into ``ImageRef``s. ``base_dir`` resolves relative
    ``path`` entries (the dataset file's directory); ``row_fetch(row_idx)``
    re-reads a dataset row so PIL images (``image`` parts, the ``images``
    column) are decoded on demand instead of retained."""

    def __init__(self, base_dir: Optional[str], images_key: str = "images",
                 messages_key: str = "messages",
                 row_fetch: Optional[Callable] = None, name: str = "ds"):
        self.base_dir = base_dir
        self.images_key = images_key
        self.messages_key = messages_key
        self.row_fetch = row_fetch
        self.name = name

    def _abs(self, p: str) -> str:
        p = os.path.expanduser(p)
        if not os.path.isabs(p) and self.base_dir and not p.startswith(("http://", "https://")):
            cand = os.path.join(self.base_dir, p)
            if os.path.exists(cand):
                return cand
        return p

    def refs_for_row(self, row: dict, row_idx: int, messages) -> tuple:
        """``(messages_with_sentinels, [ImageRef, ...])``: every turn whose
        content is a parts list becomes a plain string with one sentinel per
        image part (in order); text parts are kept verbatim. Raises
        ValueError on a part type it cannot handle (video / audio)."""
        refs: list = []
        col = row.get(self.images_key)
        col = list(col) if isinstance(col, (list, tuple)) else ([col] if col is not None else [])
        col_cursor = 0
        out = []
        for m_idx, m in enumerate(messages or []):
            if not isinstance(m, dict):
                out.append(m)
                continue
            c = m.get("content")
            if not isinstance(c, (list, tuple)):
                out.append(m)
                continue
            pieces = []
            for p_idx, p in enumerate(c):
                if isinstance(p, str):
                    pieces.append(p)
                    continue
                if not isinstance(p, dict):
                    raise ValueError(f"unsupported content part {type(p).__name__}")
                ptype = p.get("type") or "text"
                if ptype == "text":
                    pieces.append(p.get("text") or "")
                    continue
                if ptype not in IMAGE_PART_TYPES:
                    raise ValueError(f"unsupported content part type {ptype!r} "
                                     f"(images only)")
                k = len(refs)
                ref = self._ref_for_part(p, row_idx, m_idx, p_idx, col, col_cursor)
                if ref is None:
                    raise ValueError(
                        f"image part {k} has no source (path/url/base64/image) and "
                        f"the row's {self.images_key!r} column has no entry left")
                if ref == "column":
                    ref = self._ref_for_column(row_idx, col_cursor, col)
                    col_cursor += 1
                refs.append(ref)
                pieces.append(SENTINEL_FMT.format(k))
            mm = dict(m)
            mm["content"] = "".join(pieces)
            out.append(mm)
        return out, refs

    def _ref_for_part(self, p, row_idx, m_idx, p_idx, col, col_cursor):
        if p.get("path"):
            path = self._abs(str(p["path"]))
            return ImageRef(("path", path), lambda path=path: _pil_of(path))
        if p.get("url"):
            url = str(p["url"])
            return ImageRef(("url", url), lambda url=url: _open_url(url))
        iu = p.get("image_url")
        if iu:
            url = iu.get("url") if isinstance(iu, dict) else str(iu)
            if url:
                key = ("url", url) if not url.startswith("data:") else \
                    ("b64", hashlib.sha1(url.encode()).hexdigest())
                return ImageRef(key, lambda url=url: _open_url(url))
        if p.get("base64"):
            b64 = str(p["base64"])
            key = ("b64", hashlib.sha1(b64.encode()).hexdigest())
            return ImageRef(key, lambda b64=b64: _open_bytes(base64.b64decode(b64)))
        img = p.get("image")
        if img is not None:
            if isinstance(img, str):
                path = self._abs(img)
                return ImageRef(("path", path), lambda path=path: _pil_of(path))
            if isinstance(img, dict) and img.get("path") and not img.get("bytes"):
                path = self._abs(str(img["path"]))
                return ImageRef(("path", path), lambda path=path: _pil_of(path))
            # In-row pixels (PIL / bytes): identify by position, re-read the row
            # on demand so the dataset build never pins decoded images.
            key = (self.name, int(row_idx), "msg", int(m_idx), int(p_idx))
            if self.row_fetch is not None:
                fetch, mk = self.row_fetch, self.messages_key
                return ImageRef(key, lambda: _pil_of(
                    fetch(row_idx)[mk][m_idx]["content"][p_idx]["image"]))
            return ImageRef(key, lambda img=img: _pil_of(img))
        if col_cursor < len(col):
            return "column"
        return None

    def _ref_for_column(self, row_idx, j, col):
        key = (self.name, int(row_idx), "col", int(j))
        if self.row_fetch is not None:
            fetch, ik = self.row_fetch, self.images_key
            return ImageRef(key, lambda: _pil_of(_col_image(fetch(row_idx), ik, j)))
        obj = col[j]
        return ImageRef(key, lambda obj=obj: _pil_of(obj))


def _col_image(row, images_key, j):
    col = row[images_key]
    if not isinstance(col, (list, tuple)):
        col = [col]
    return col[j]


# --- tokenization with image layouts ------------------------------------------

def _strip_bos(ids, bos, first):
    if bos is None or not ids:
        return ids
    if first:
        while len(ids) >= 2 and ids[0] == bos and ids[1] == bos:
            ids = ids[1:]
    elif ids[0] == bos:
        ids = ids[1:]
    return ids


def encode_segments_with_images(tokenizer, segments, feats: dict, placeholder_id: int):
    """Tokenize ``(text, supervised)`` segments whose text may carry image
    sentinels. Returns ``(input_ids, labels, images)`` where ``images`` is
    ``[(start, k), ...]`` -- the index of the first token of image ``k``'s
    layout. Text pieces are encoded separately around every sentinel and
    every segment boundary (exact masks); image tokens are never supervised.
    BOS is normalized exactly as chat_turns.encode_segments does."""
    bos = tokenizer.bos_token_id
    input_ids, labels, images = [], [], []
    first = True
    for text, supervised in segments:
        for piece in SENTINEL_RE.split(text):
            if piece == "":
                continue
            m = _SENTINEL_K.fullmatch(piece)
            if m:
                k = int(m.group(1))
                f = feats[k]
                ids = [placeholder_id if t == IMAGE_SLOT else int(t) for t in f.token_ids]
                images.append((len(input_ids), k))
                input_ids += ids
                labels += [-100] * len(ids)
                first = False
                continue
            ids = tokenizer.encode(piece, add_bos=False,
                                   encode_special_tokens=True)[0].tolist()
            ids = _strip_bos(ids, bos, first)
            first = False
            input_ids += ids
            labels += list(ids) if supervised else [-100] * len(ids)
    return input_ids, labels, images


# --- per-example assembly ------------------------------------------------------

class VisionData:
    """Trainer-side glue: an ``ImageResolver`` per dataset, the ``VisionEncoder``,
    and the example / batch assembly around them."""

    def __init__(self, encoder, tokenizer, placeholder_id: int, mrope: bool,
                 images_key: str = "images"):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.placeholder_id = int(placeholder_id)
        self.mrope = bool(mrope)
        self.images_key = images_key
        self.loaders: dict = {}          # key -> load()
        self.n_rows = 0                  # rows that carried images
        self.n_images = 0                # image occurrences
        self.image_tokens = 0            # feature-slot tokens over all built rows

    # -- dataset build --

    def resolver(self, dataset_name: str, ds, messages_key: str = "messages") -> ImageResolver:
        path = os.path.expanduser(dataset_name)
        base = os.path.dirname(os.path.abspath(path)) if os.path.exists(path) else None
        return ImageResolver(base, self.images_key, messages_key,
                             row_fetch=(lambda i: ds[int(i)]) if ds is not None else None,
                             name=str(dataset_name))

    def prepare(self, resolver: ImageResolver, row: dict, row_idx: int, messages):
        """``(messages, refs)`` with image parts flattened to sentinels; ``refs``
        is empty when the row carries no image."""
        if not has_image_parts(messages):
            return messages, []
        return resolver.refs_for_row(row, row_idx, messages)

    def flatten(self, messages, base_dir: Optional[str] = None):
        """:meth:`prepare` for an in-memory conversation (the realtime
        coordinator's ``vision.flatten`` contract): image parts name their
        pixels inline (path / url / base64 / image / image_url), relative
        paths resolve against ``base_dir`` (else the working directory)."""
        resolver = ImageResolver(base_dir, self.images_key, name="realtime")
        return self.prepare(resolver, {}, 0, messages)

    def features(self, refs: list) -> dict:
        """``{k: ImageFeatures}`` (with embeds) for the row's refs, encoding on
        a cache miss. Registers each ref's loader for batch-time refetch."""
        out = {}
        for k, ref in enumerate(refs):
            self.loaders.setdefault(ref.key, ref.load)
            out[k] = self.encoder.encode(ref.key, ref.load)
        return out

    def encode_segments(self, segments, refs):
        feats = self.features(refs)
        ids, labels, images = encode_segments_with_images(
            self.tokenizer, segments, feats, self.placeholder_id)
        return ids, labels, images, feats

    def finalize(self, input_ids, labels, images, feats, refs, seq_len):
        """Truncate / validate and attach the image bookkeeping. Returns
        ``(example, None)`` or ``(None, reason)``. A row is dropped rather
        than cut through an image (Axolotl likewise doesn't truncate
        multimodal rows), and when no supervised token survives."""
        if len(input_ids) > seq_len:
            last_end = max(s + len(feats[k].token_ids) for s, k in images) if images else 0
            if last_end > seq_len:
                return None, "too_long"
            input_ids, labels = input_ids[:seq_len], labels[:seq_len]
        if all(l == -100 for l in labels):
            return None, "truncated"
        ex = {
            "input_ids": input_ids, "labels": labels,
            "images": [(int(s), refs[k].key) for s, k in images],
        }
        if self.mrope:
            spans = []
            for s, k in images:
                f = feats[k]
                assert f.grid_thw is not None and f.merge_size is not None, \
                    "mRoPE tower but the vision features carry no grid"
                spans.append((s + f.slot_offsets[0], f.n_tokens, f.grid_thw))
            merge = feats[images[0][1]].merge_size if images else 1
            ex["mrope_position_ids"] = mrope_position_ids(
                len(input_ids), spans, merge).tolist()
        self.n_rows += 1
        self.n_images += len(images)
        self.image_tokens += sum(feats[k].n_tokens for _, k in images)
        return ex, None

    # -- batch --

    def collate_mm(self, batch: list, seq_len: int, dtype: torch.dtype) -> Optional[dict]:
        """The ``mm`` splice for a collated batch of examples (padded to
        ``seq_len``), or None when none carries an image."""
        rows = []
        for ex in batch:
            imgs = []
            for start, key in ex.get("images", ()):
                loader = self.loaders.get(key)
                if loader is None:
                    raise KeyError(f"no loader registered for image {key!r}")
                imgs.append((start, self.encoder.encode(key, loader)))
            rows.append(imgs)
        return build_mm_batch(rows, seq_len, dtype)

    def describe(self) -> str:
        st = self.encoder.stats()
        s = (f"{self.n_rows} rows with images, {self.n_images} image occurrences, "
             f"{st['images']} distinct images ({st['tokens_mean']:.0f} tokens/image "
             f"mean, {st['tokens_max']} max), {self.image_tokens} image tokens total; "
             f"feature cache {st['cache_gb']:.2f} GB "
             f"({st['cached']}/{st['images']} cached")
        s += ")" if self.encoder.n_evicted == 0 else \
            f", {self.encoder.n_evicted} over budget -> recomputed per batch)"
        return s
