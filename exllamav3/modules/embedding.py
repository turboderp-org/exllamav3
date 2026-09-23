from __future__ import annotations
from typing_extensions import override
import torch
from torch import nn
from ..model.config import Config
from ..util.tensor import to2
from ..ext import exllamav3_ext as ext
from .quant.exl3_lib import embed_trellis
from . import Module
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from ..model.model_tp_alloc import TPAllocation
import logging

logger = logging.getLogger(__name__)


class Embedding(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        vocab_size: int,
        hidden_size: int,
        out_dtype: torch.dtype | None = torch.float,
        qmap: str | None = None,
        normalize: bool = False,
        multiplier: float = 1.0
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for Embedding"

        self.key = key
        self.embedding = None
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self._pinned_staging = {}
        self.trellis = None
        self._numel = vocab_size * hidden_size
        self.normalize = normalize
        self.multiplier = multiplier

        self.caps.update({
            "prefer_cpu": True,
        })

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        # a reload without unload() must not leak the previous device-mapped registration
        # (model.load()/load_gen() reuse module instances: load() -> load()); the fp16
        # branch below would otherwise leave self.trellis populated and _gather would
        # silently prefer the stale table over the freshly loaded weight
        self._release_trellis()
        stc = self.config.stc
        if stc.has_tensor(self.key + ".weight_trellis"):
            self._load_trellis(stc, device, kwargs.get("compute_device"))
            return
        weight = stc.get_tensor(self.key + ".weight", self.device, float2half = True, allow_bf16 = True)
        self._numel = weight.numel()
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            device = "meta"
        )
        self.embedding.weight = nn.Parameter(weight)

    def _load_trellis(self, stc, device, compute_device: torch.device | None = None):
        # Trellis-quantized token embedding (exl3_trellis_embed, util/convert_embedding.py):
        # the packed table stays in host RAM, page-locked and device-mapped; the fused GPU
        # kernel (ext.trellis_embed_gather) reads scattered rows zero-copy over PCIe and
        # writes the fp32 output on-device. Only the mul1 codebook LUT (128 KB) and the
        # fp32 column scales (D * 4 B) are VRAM-resident.
        key = self.key + ".weight_trellis"
        # variant-aware owning collection (config.stc is a VariantSafetensorsCollection
        # under --override / optimize_model, which has no tensor_file_map of its own)
        owner = stc.find_stc(key)
        filename = owner.tensor_file_map.get(key)
        assert filename is not None, \
            f"{key}: trellis table must be backed by a *.safetensors file (in-memory tensor?)"
        meta = owner.file_headers[filename].get("__metadata__", {})
        assert meta.get("format") == embed_trellis.FORMAT, \
            f"{key}: not an {embed_trellis.FORMAT} table (format = {meta.get('format')!r})"
        assert meta.get("version") == embed_trellis.FORMAT_VERSION, f"{key}: unsupported format version {meta.get('version')}"
        assert meta.get("codebook") == "mul1" and meta.get("transform") == "qtip1", \
            f"{key}: unsupported codebook/transform in metadata"
        K = int(meta["K"])
        G = int(meta["G"])
        seed = int(meta["seed"])
        rows = int(meta["rows"])
        hidden = int(meta["hidden"])
        assert hidden == self.hidden_size, \
            f"{key}: table hidden {hidden} does not match model hidden {self.hidden_size}"
        assert rows >= self.vocab_size, \
            f"{key}: table has {rows} rows but the model vocab_size is {self.vocab_size}"
        if rows > (self.vocab_size + 127) // 128 * 128:
            # padded tables (rows > vocab_size, e.g. 128-alignment) are fine: _gather
            # bounds-checks against the table's own row count. A table padded far beyond
            # any plausible alignment is more likely a mismatched table than padding
            logger.warning(
                "%s: trellis table has %d rows but model vocab_size is %d (> 128-aligned "
                "padding): rows above vocab_size are never addressed; verify the table "
                "matches this model", self.key, rows, self.vocab_size)
        D = hidden
        assert D % embed_trellis.GROUP == 0 and G == D // embed_trellis.GROUP, \
            f"{key}: metadata G {G} inconsistent with hidden {D}"
        assert K in (6, 7, 8), f"{key}: unsupported trellis K {K} (fused kernel supports 6/7/8)"

        # no_defer: the buffer must hold its data before it is copied into the pinned
        # registration (a deferred load would fill the original after this copy)
        table = owner.get_tensor(key, torch.device("cpu"), no_defer = True)
        assert table.dtype == torch.int16 and table.dim() == 2 and table.shape == (rows, embed_trellis.words_per_row(D, K, G)), \
            f"{key}: expected int16 ({rows}, {embed_trellis.words_per_row(D, K, G)}) packed table, got {tuple(table.shape)} {table.dtype}"
        # resolve through the variant-aware collection (each key to its own owner): a
        # --override glob that matches *.weight_trellis but not *.col_scales would
        # otherwise make this lookup raise although the tensor exists in the base dir
        col_scales = stc.find_stc(self.key + ".col_scales").get_tensor(
            self.key + ".col_scales", torch.device("cpu"), no_defer = True)
        assert col_scales.dtype == torch.float16 and col_scales.shape == (D,), \
            f"{key}: expected fp16 ({D},) col_scales"
        # source-table element count (rows * D), not the packed footprint (rows *
        # (G + D*K/16) int16 words): the table is host-resident, so no VRAM budget
        # consumer sees this; code that sums weights_numel as a size estimate
        # overestimates a trellis table by the packing ratio
        self._numel = rows * D

        cuda = torch.cuda.is_available()
        if cuda:
            # pin_memory() makes a second full copy of the table: transient 2x host RAM
            # (loaded source copy + pinned copy) until the load copy is dropped here
            pinned = table.pin_memory()
            del table
            # prefer the loader's compute device (this module always loads on CPU, so
            # `device` is never cuda); current_device() only as last resort. Callers may
            # pass str/int devices (Model.load("cuda:0")), so normalize before .type
            dev = None
            for cand in (compute_device, device):
                if cand is None:
                    continue
                cand = torch.device(cand)
                if cand.type == "cuda":
                    dev = cand
                    break
            if dev is None or dev.index is None:
                # no compute device given (or a bare "cuda"): current_device() only as
                # last resort; resolve the index now so t["dev"] is fully indexed for the
                # module's whole lifetime (a later current-device change must not silently
                # move the registration)
                dev = torch.device("cuda", torch.cuda.current_device())
            col_scales_f32 = col_scales.float().to(dev)
            codebook = embed_trellis.mul1_codebook(dev)
            # register LAST: a raise above must not leave the pinned region device-mapped
            # with self.trellis unset (_release_trellis would be a no-op and a later
            # pinned alloc reusing the address would inherit the stale alias)
            table_ptr = ext.trellis_embed_register(pinned, dev.index if dev.index is not None else -1)
            self.trellis = {
                "table": pinned,
                "table_ptr": table_ptr,
                "col_scales": col_scales,
                "col_scales_f32": col_scales_f32,
                "codebook": codebook,
                "K": K, "G": G, "seed": seed, "D": D,
                "dev": dev,
            }
        else:
            # Correctness-only fallback (no CUDA): CPU torch reference codec, bit-exact with
            # the fused kernel but ~6.5x slower than q8_0 CPU; never the primary path
            self.trellis = {
                "table": table,
                "table_ptr": None,
                "col_scales": col_scales,
                "col_scales_f32": None,
                "codebook": None,
                "K": K, "G": G, "seed": seed, "D": D,
                "dev": None,
            }
        self.embedding = None

    @override
    def unload(self):
        self.device = None
        self.embedding = None
        self._release_trellis()

    def _release_trellis(self):
        # Drop the device-mapped registration (explicit unload or module destruction).
        # Idempotent: a late __del__ after unload() finds self.trellis already released.
        t = self.trellis
        if t is None:
            return
        self.trellis = None
        if t["table_ptr"] is not None:
            try:
                ext.trellis_embed_unregister(t["table"])
            except Exception:
                pass


    def retarget_trellis(self, device: torch.device):
        # Autosplit fixup: the load-time compute-device guess (the device active when
        # this module loaded) can miss the consumer block's final placement. The packed
        # table never moves (host RAM); only the registration and the device-resident
        # state (codebook LUT, fp32 column scales) are re-created on the new device.
        t = self.trellis
        if t is None or t["dev"] is None:
            return
        device = torch.device(device)
        if t["dev"] == device:
            return
        self._release_trellis()
        pinned = t["table"]
        col_scales_f32 = t["col_scales"].float().to(device)
        codebook = embed_trellis.mul1_codebook(device)
        # register LAST: a raise above must not leave the pinned region device-mapped
        # with self.trellis unset (same convention as _load_trellis)
        table_ptr = ext.trellis_embed_register(pinned, device.index if device.index is not None else -1)
        self.trellis = {
            "table": pinned,
            "table_ptr": table_ptr,
            "col_scales": t["col_scales"],
            "col_scales_f32": col_scales_f32,
            "codebook": codebook,
            "K": t["K"], "G": t["G"], "seed": t["seed"], "D": t["D"],
            "dev": device,
        }

    def __del__(self):
        # Release the device-mapped registration when the module is destroyed without an
        # explicit unload() (refcount or cyclic GC). _release_trellis is idempotent, so
        # the unload() -> __del__ double path is safe.
        try:
            self._release_trellis()
        except Exception:
            pass

    @override
    def get_tensors(self):
        if self.trellis is not None:
            # Nothing to export: the packed table is host-resident and the device-resident
            # state (codebook LUT, fp32 col scales) is runtime-only - emitting it would
            # inject a stray fp32 tensor under the format's own fp16 {key}.col_scales key
            # into the conversion pipeline (convert_model collects get_tensors per module)
            # while dropping the table itself; the embedding-trellis file is produced by
            # util/convert_embedding.py (same convention as NGramEmbedding.get_tensors)
            return {}
        return {
            f"{self.key}.weight": self.embedding.weight.data.contiguous()
        }

    @override
    def weights_numel(self):
        return self._numel

    def _gather(self, ids: torch.Tensor) -> torch.Tensor:
        # Row gather for the current storage format. The trellis branch returns fp32 on the
        # compute device (fused kernel; multiplier/normalize/out_dtype cast happen in
        # forward, in-place on the on-device output - no staging).
        if self.trellis is None:
            return self.embedding.forward(ids)
        t = self.trellis
        shape = ids.shape
        ids = ids.reshape(-1)
        rows = t["table"].shape[0]
        if t["dev"] is None:
            # CPU reference path (correctness-only, slow): bit-exact with the fused kernel
            if ids.numel() and (bool(ids.min() < 0) or bool(ids.max() >= rows)):
                raise IndexError(f"trellis embed: row id out of range [0, {rows})")
            rows_t = t["table"][ids.to(t["table"].device)]
            x = embed_trellis.dequant_rows_transformed(
                rows_t, t["col_scales"], t["K"], t["seed"], t["D"], t["G"], row_ids = ids)
            return x.view(*shape, t["D"])
        # CUDA: ids still on the host (the generator's decode loop) get a loud bound
        # check for free - no stream sync; ids that arrive on CUDA skip it and rely on
        # the kernel's clamp (an out-of-range id decodes the last, all-zero pad row
        # instead of faulting the device-mapped table)
        if ids.device.type != "cuda" and ids.numel() and \
                (bool(ids.min() < 0) or bool(ids.max() >= rows)):
            raise IndexError(f"trellis embed: row id out of range [0, {rows})")
        ids_d = ids if (ids.device.type == "cuda" and ids.device == t["dev"]) else ids.to(t["dev"])
        if ids_d.dtype != torch.int64:
            ids_d = ids_d.to(torch.int64)
        out = torch.empty((ids_d.shape[0], t["D"]), dtype = torch.float32, device = t["dev"])
        ext.trellis_embed_gather(t["table_ptr"], t["codebook"], t["col_scales_f32"],
                                 ids_d, t["K"], t["seed"], rows, out)
        return out.view(*shape, t["D"])

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        # Ensure input IDs in params
        if "input_ids" not in params:
            params["input_ids"] = x

        indexed_emb = params.get("indexed_embeddings")
        input_ids = x
        out_dtype = out_dtype or self.out_dtype or x.dtype

        # Indexed embedding masks
        if indexed_emb:
            standard_mask = input_ids < FIRST_MM_EMBEDDING_INDEX
            indexed_masks = [
                (input_ids >= e.first_index) & (input_ids < (e.first_index + e.mm_length))
                for e in indexed_emb
            ]
            indexed_act = [im.any() for im in indexed_masks]
            use_indexed_emb = any(indexed_act)

        # Mixed embeddings when needed
        if indexed_emb and use_indexed_emb:
            bsz, seq_len = input_ids.shape
            combined_emb = torch.empty((bsz, seq_len, self.hidden_size), device = self.device, dtype = out_dtype)

            # Prepare deepstack embedding tensors
            if any(ie.deepstack_embeddings is not None for ie in indexed_emb) and indexed_act:
                assert all(ie.deepstack_embeddings is not None for ie in indexed_emb)
                num_layers = len(indexed_emb[0].deepstack_embeddings)
                assert all(num_layers == len(ie.deepstack_embeddings) is not None for ie in indexed_emb)
                deepstack_emb = [torch.zeros_like(combined_emb) for _ in range(num_layers)]
            else:
                deepstack_emb = None

            # Insert standard embeddings: one gather + one move for the whole batch, not
            # per row - a trellis gather lands fp32 on the compute device while
            # combined_emb is on self.device (CPU for prefer_cpu), so the per-row loop
            # paid a kernel launch + a D2H copy per row
            if standard_mask.any():
                standard_ids = input_ids[standard_mask]
                standard_emb = self._gather(standard_ids).to(combined_emb.device, out_dtype)
                combined_emb[standard_mask] = standard_emb

            # Only normalize standard embeddings
            if self.normalize:
                combined_emb *= combined_emb.shape[-1] ** 0.5

            # Also only scale standard embeddings
            if self.multiplier != 1.0:
                combined_emb *= self.multiplier

            # Insert indexed embeddings
            for im, ie, act in zip(indexed_masks, indexed_emb, indexed_act):
                if not act:
                    continue
                for i in range(bsz):
                    indexed_ids_row = input_ids[i][im[i]] - ie.first_index
                    combined_emb[i][im[i]] = ie.embeddings[indexed_ids_row].to(out_dtype)

                    # Prepare deepstack embeddings
                    if ie.deepstack_embeddings is not None:
                        for layer, de in enumerate(ie.deepstack_embeddings):
                            deepstack_emb[layer][i][im[i]] = de[indexed_ids_row].to(out_dtype)

            # Save deepstack embeddings to params
            if deepstack_emb is not None:
                params["deepstack_emb"] = deepstack_emb

            return combined_emb

        # No indexed embeddings, or none in current batch
        else:
            x = self._gather(x)
            if self.multiplier != 1.0:
                x *= self.multiplier
            x = to2(x, out_dtype, self.out_dtype)
            if self.normalize:
                x *= x.shape[-1] ** 0.5
            # When the embedding resides on the CPU, its output is uploaded to the first
            # device layer; staging it through a reused pinned buffer makes that upload
            # asynchronous. Only callers that guarantee a sync point between forward passes
            # (the generator's decode loop) may set the pinned_staging flag.
            if params.get("pinned_staging") and x.device.type == "cpu":
                key = (x.shape, x.dtype)
                buf = self._pinned_staging.get(key)
                if buf is None:
                    if len(self._pinned_staging) > 8:
                        self._pinned_staging.clear()
                    buf = torch.empty_like(x, pin_memory = True)
                    self._pinned_staging[key] = buf
                buf.copy_(x)
                x = buf
            return x

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        # fail during planning, before the TP loader loads, pins and device-maps the full
        # trellis table only for tp_export to reject it (multi-GB of work for a rejected config)
        assert not self.config.stc.has_tensor(self.key + ".weight_trellis"), \
            "Trellis-quantized embedding is not supported with tensor parallelism"
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        assert self.trellis is None, "Trellis-quantized embedding is not supported with tensor parallelism"
        return {
            "cls": Embedding,
            "kwargs": {
                "key": self.key,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "out_dtype": self.out_dtype,
                "normalize": self.normalize,
                "multiplier": self.multiplier,
            },
            "embedding.weight": producer.send(self.embedding.weight),
            "device": self.device
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = Embedding(
            config = None,
            **exported["kwargs"],
        )
        module.device = exported["device"]
        module.embedding = nn.Embedding(
            module.vocab_size,
            module.hidden_size,
            device = "meta"
        )
        emb = consumer.recv(exported["embedding.weight"], cuda = False)
        module.embedding.weight = nn.Parameter(emb)
        return module