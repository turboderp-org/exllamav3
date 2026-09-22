from __future__ import annotations
from typing_extensions import override
import torch
from .. import LayerNorm
from ...constants import PAGE_SIZE
from ...model.config import Config
from ...model.model_tp_fn import mp_model_forward_embedding
from ...modules import Module, Linear, RMSNorm
from ...modules.attn import Attention
from ...util.rope import RopeSettings, RoPE
from ...util.tensor import get_for_device, to2

class DFlashInputLayer(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        key_norm: str,
        hidden_size: int,
        target_state_size: int,
        mask_token_id: int,
        rms_norm_eps: float,
        native_draft_len: int,
        out_dtype: torch.dtype | None = torch.float,
        qmap: str | None = None,
        key_aux_norms: str | None = None,
        num_aux_norms: int = 0,
        input_embedding_scale: float = 1.0,
        key_mask_embedding: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "DFlashInputLayer"
        self.qmap = qmap
        self.key = key
        self.hidden_size = hidden_size
        self.target_state_size = target_state_size
        self.out_dtype = out_dtype
        self.native_draft_len = native_draft_len

        self.proj = Linear(
            config = config,
            key = f"{key}",
            in_features = self.target_state_size,
            out_features = self.hidden_size,
            qmap = (qmap + ".input") if qmap else None,
            out_dtype = out_dtype,
            pad_to = 1
        )

        self.norm = RMSNorm(
            config = config,
            key = f"{key_norm}",
            rms_norm_eps = rms_norm_eps,
            out_dtype = out_dtype,
        )

        self.register_submodule(self.proj)
        self.register_submodule(self.norm)

        # Per-tap norms (Laguna DFlash): each captured target hidden state is RMS-normed
        # individually before the taps are concatenated and projected by fc
        self.aux_norms = []
        if key_aux_norms is not None:
            for i in range(num_aux_norms):
                aux_norm = RMSNorm(
                    config = config,
                    key = f"{key_aux_norms}.{i}",
                    rms_norm_eps = rms_norm_eps,
                    out_dtype = torch.half,
                )
                self.aux_norms.append(aux_norm)
                self.register_submodule(aux_norm)

        self.mask_token_id = mask_token_id
        self.input_embedding_scale = input_embedding_scale

        # Learned mask embedding. The original DFlash release looks mask_token_id up in the
        # target's embedding table; some checkpoints ship their own vector instead, because
        # the id is not in the target's trained vocabulary at all (MiMo-V2.6-Flash-RL:
        # mask_token_id 151675 is past the end of the tokenizer and the target's embedding row
        # for it is an untrained padding row, L2 norm 2e-5 against the shipped vector's 0.76).
        # When present it replaces the embedding of every mask position in the block.
        self.key_mask_embedding = key_mask_embedding
        self.mask_embedding = None

        # Populated by attach_to()
        self.attached_model = None

        self.caps.update({"x_cpu": True})


    def optimizer_targets(self):
        raise NotImplementedError()


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.key_mask_embedding:
            self.mask_embedding = self.config.stc.get_tensor(
                self.key_mask_embedding, device, allow_bf16 = True, no_defer = True
            ).view(-1)


    @override
    def unload(self):
        self.mask_embedding = None
        super().unload()


    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ):
        bsz, seqlen = x.shape
        noise_mask = torch.full((bsz, self.native_draft_len - 1), self.mask_token_id, dtype = torch.long)
        x = torch.cat((x, noise_mask), dim = -1)
        if not self.attached_model().loaded_tp:
            x = self.attached_model().modules[0].forward(x, params)
        else:
            x = self.attached_model().tp_producer.send(x)
            x = self.attached_model().tp_dispatch_master(mp_model_forward_embedding, (x, params))
        if self.input_embedding_scale != 1.0:
            x = x * self.input_embedding_scale
        mask_embedding = getattr(self, "mask_embedding", None)
        if mask_embedding is not None:
            # The trailing native_draft_len - 1 positions are the mask tokens appended above;
            # everything before them keeps its real embedding
            x[:, -(self.native_draft_len - 1):, :] = mask_embedding.to(x.dtype)
        return x


class DFlashRing:
    """
    Fixed-size K/V ring for DFlash draft layers with a sliding window.

    A DFlash drafter's K/V context is one row per TARGET position, written by
    update_kv_from_target and never revised, and a layer declared `sliding_attention` only ever
    reads the last `sliding_window` of them (plus the drafted block itself, which the
    bidirectional right window makes visible). Storing that in a cache sized for the whole
    context is pure waste: at 20 KiB/token for MiMo-V2.6-Flash-RL's drafter it is 2.5 GiB at
    128k and 7.5 GiB at 384k, on top of the target's own KV.

    The ring stores absolute position p at ring offset `p % num_tokens`, which needs no state
    at all beyond the per-slot validity range: page tables address the ring modulo its page
    count, so a window that wraps is just a rotated block table and the ordinary paged kernels
    handle it unchanged. Because the mapping is a pure function of the absolute position,
    a rejected draft needs no rollback -- positions below the target's new sequence length are
    still in place, and everything above is rewritten before it can be read again. The ring is
    overprovisioned by `overprovision` tokens beyond window + block so that rewinds (rejected
    drafts, banned-string rewinds) up to that depth stay in range, mirroring SWAState's
    guaranteed_rollback.
    """

    def __init__(
        self,
        sliding_window: int,
        block_size: int,
        overprovision: int = 2 * PAGE_SIZE,
    ):
        assert sliding_window > 0
        self.sliding_window = sliding_window
        self.block_size = block_size
        self.num_tokens = -(-(sliding_window + block_size + overprovision) // PAGE_SIZE) * PAGE_SIZE
        self.num_pages = self.num_tokens // PAGE_SIZE
        # Longest write that can be expressed as one contiguous span in the ring, whatever the
        # page offset the write starts at
        self.max_write = self.num_tokens - PAGE_SIZE
        self.num_slots = 1
        self.beg = [0]
        self.end = [0]
        self._bt_host = None
        self._sl_host = None
        self._bt_dev = {}
        self._sl_dev = {}


    def guaranteed_rollback(self):
        """Tokens the ring can rewind and still serve a full window from storage."""
        return max(0, self.num_tokens - self.sliding_window - self.block_size - (PAGE_SIZE - 1))


    def set_num_slots(self, num_slots: int):
        self.num_slots = max(1, num_slots)
        self.beg = [0] * self.num_slots
        self.end = [0] * self.num_slots
        self._bt_host = None
        self._sl_host = None
        self._bt_dev = {}
        self._sl_dev = {}


    def reset_slot(self, slot: int):
        self.beg[slot] = 0
        self.end[slot] = 0


    def reset(self):
        self.beg = [0] * self.num_slots
        self.end = [0] * self.num_slots


    def storage_tokens(self):
        return self.num_pages * PAGE_SIZE * self.num_slots


    def note_write(self, slots: list[int], positions: list[int], length: int):
        """Record that `length` context rows were written at absolute `positions`."""
        for r, slot in enumerate(slots):
            p = int(positions[r])
            if p > self.end[slot]:
                # A gap (prefix-cache reuse skipped a span of target states): nothing below the
                # gap can be trusted any more
                self.beg[slot] = p
            # The write frontier is the truth even when it moves backwards (a rewind re-feeds
            # positions the ring already holds), so beg stays as generous as the storage allows
            self.end[slot] = p + length
            self.beg[slot] = max(self.beg[slot], self.end[slot] - self.num_tokens)


    def _rows(self, slots: list[int], base_pages: list[int]):
        """Rotated page list per row: ring page (base + j) mod num_pages inside the slot."""
        np_ = self.num_pages
        return [
            [slot * np_ + (base_pages[r] + j) % np_ for j in range(np_)]
            for r, slot in enumerate(slots)
        ]


    def _tables(self, slots, base_pages, seqlens, device, persistent: bool = True):
        """Per-row rotated block table over the slot's ring pages, plus in-ring past lengths.
        The read path reuses persistent buffers so the graph-captured decode step sees stable
        pointers; the write path allocates, since several chunks can be in flight at once."""
        bsz = len(slots)
        rows = self._rows(slots, base_pages)
        if not persistent:
            bt = torch.tensor(rows, dtype = torch.int32, device = device)
            sl = torch.tensor(seqlens, dtype = torch.int32, device = device)
            return bt, sl
        if self._bt_host is None or self._bt_host.shape[0] < bsz:
            n = max(bsz, self.num_slots)
            self._bt_host = torch.empty((n, self.num_pages), dtype = torch.int32, pin_memory = True)
            self._sl_host = torch.empty((n,), dtype = torch.int32, pin_memory = True)
            self._bt_dev = {}
            self._sl_dev = {}
        self._bt_host[:bsz].copy_(torch.tensor(rows, dtype = torch.int32))
        self._sl_host[:bsz].copy_(torch.tensor(seqlens, dtype = torch.int32))
        key = str(device)
        bt = self._bt_dev.get(key)
        if bt is None:
            bt = torch.empty(self._bt_host.shape, dtype = torch.int32, device = device)
            sl = torch.empty(self._sl_host.shape, dtype = torch.int32, device = device)
            self._bt_dev[key] = bt
            self._sl_dev[key] = sl
        else:
            sl = self._sl_dev[key]
        bt[:bsz].copy_(self._bt_host[:bsz], non_blocking = True)
        sl[:bsz].copy_(self._sl_host[:bsz], non_blocking = True)
        return bt[:bsz], sl[:bsz]


    def read_view(self, slots: list[int], positions, q_len: int, device):
        """Block table and in-ring past-lengths for a draft block of q_len queries whose first
        position is `positions[r]`. The span starts on a page boundary at or before
        position - sliding_window, so the attention kernel's own window mask still decides
        which keys count; the extra keys in the leading page are masked out exactly as they are
        for the target's SWA ring."""
        base_pages, seqlens = [], []
        max_hot = self.num_tokens - q_len
        for r, slot in enumerate(slots):
            p = int(positions[r])
            lo = max(self.beg[slot], p - self.sliding_window, 0)
            if p > self.end[slot]:
                # Nothing was written for [end, p): the generator skipped that span of prefill
                # because the target's pages were already cached, and a DFlash drafter cannot
                # reconstruct target states it never saw. Attend to the block alone rather than
                # to whatever the ring happens to hold; acceptance recovers as tokens are
                # accepted and written, and correctness never depended on the drafter anyway
                lo = p
            wpos = lo // PAGE_SIZE * PAGE_SIZE
            hot = p - wpos
            if hot > max_hot:
                # Cannot happen while num_tokens >= window + block + PAGE_SIZE, but clamping
                # keeps a misconfigured ring reading valid memory
                wpos = -(-(p - max_hot) // PAGE_SIZE) * PAGE_SIZE
                hot = p - wpos
            base_pages.append(wpos // PAGE_SIZE)
            seqlens.append(hot)
        return self._tables(slots, base_pages, seqlens, device)


    def write_views(self, slots: list[int], positions, length: int, device):
        """Yield (offset, chunk_len, block_table, cache_seqlens) covering a write of `length`
        context rows starting at absolute `positions[r]` on every row. Writes longer than the
        ring are split; the tail chunks simply overwrite the earlier ones, leaving the last
        num_tokens positions in place, which is all the window can ever read."""
        t = 0
        while t < length:
            c = min(self.max_write, length - t)
            base_pages = [(int(positions[r]) + t) // PAGE_SIZE for r in range(len(slots))]
            seqlens = [(int(positions[r]) + t) % PAGE_SIZE for r in range(len(slots))]
            bt, sl = self._tables(slots, base_pages, seqlens, device, persistent = False)
            yield t, c, bt, sl
            t += c


def dflash_ring_slots(params: dict, bsz: int) -> list[int]:
    """Per-row ring slot for the current batch. The generator assigns a stable slot to every
    live sequence; anything driving the model directly (tests, examples) gets the identity
    mapping, which is correct for any batch whose row order does not change."""
    slots = params.get("dflash_ring_slots")
    if slots is None:
        return list(range(bsz))
    return [int(s) for s in slots]


class DFlashRingAttention(Attention):
    """
    DFlash draft attention layer backed by a DFlashRing instead of a full-length paged cache.

    Everything except the cache geometry is the base Attention: the same projections, the same
    sinks, the same (left, right) window. Only two things change, and both are pure address
    arithmetic: the cache layer is sized for the ring rather than for the job page pool, and
    the block table / past-length pair handed to the kernels addresses the ring. The model's
    prepare_inputs() rewrites those once per forward for every ring layer at once (they share
    one ring geometry), so the per-layer path is untouched.
    """

    def __init__(self, *args, ring: DFlashRing, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.sliding_window > 0, "DFlashRingAttention requires a sliding window"
        self.ring = ring


    def cache_layer_num_tokens(self, max_num_tokens: int, max_batch_size: int) -> int:
        self.ring.set_num_slots(max_batch_size)
        return self.ring.storage_tokens()
