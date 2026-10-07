import torch

# Slot index tensors are tiny and recur across forward passes; cache them with persistent device
# copies so each decode step doesn't rebuild and re-upload one
_slot_tensors = {}

# `key` distinguishes cached tensors over the same slot tuple: the scan-side tensors are the
# pool slots doubled and offset by each state's base/scratch parity (Path A), so the parity has
# to be part of the cache key.
def _get_slot_tensor(slots: tuple, key: tuple = ()) -> torch.Tensor:
    ck = (slots, key)
    t = _slot_tensors.get(ck)
    if t is None:
        if len(_slot_tensors) > 4096:
            _slot_tensors.clear()
        t = torch.tensor(list(slots), dtype = torch.int32)
        t._static_dev_cache = True
        _slot_tensors[ck] = t
    return t


def prepare_for_recurrence(input_ids: torch.Tensor, params: dict, model) -> torch.Tensor:
    """
    Add linear attn/SWA/recurrent parameters to state

    batch_shape: tuple of (bsz, _)
    past_len: int (default: 0)

    *OR*

    cache_seqlens: shape (bsz)
    """
    batch_shape = params.get("batch_shape")
    cache_seqlens = params.get("cache_seqlens")
    rs = params.get("recurrent_states")

    # Rectangular batch
    if batch_shape is not None:
        bsz, _ = batch_shape
        past_len = params.get("past_len", 0)
        if past_len > 0:
            if rs is None:
                raise ValueError(f"Past length given, but no previous state for recurrence in params")
            if not isinstance(rs, list):
                rs = [rs]
                params["recurrent_states"] = rs
            assert all(r.position == past_len for r in rs), "recurrent states don't match input past_len"
        else:
            if rs is None:
                rs = [params["cache"].get_new_state() for _ in range(bsz)]
                params["recurrent_states"] = rs
            else:
                assert all(r.position == 0 for r in rs), "recurrent states don't match input past_len"

    # Paged attn batch
    elif cache_seqlens is not None:
        # (Empty) states must be provided with cache_seqlens
        pass

    # Neither
    else:
        if rs is not None:
            raise ValueError(f"recurrent_states given without bsz and seqlens")

    # Create slot index tensors. `recurrent_slots` indexes conv_state (cache pool slots);
    # `recurrent_slots_scan[_in]` index the doubled recurrent_state pool (Path A): flat row
    # 2*slot + parity is the live base state, 2*slot + (1 - parity) the scratch buffer. The
    # scan-side tensors name the kernel's WRITE (`_scan`) and READ (`_scan_in`) state rows:
    # a spec pass advances scratch from base (out = 2s+1-p, in = 2s+p; scan-in only exists
    # on spec passes), a non-spec pass runs in place on the live row (out = 2s+p, no scan-in).
    if rs is not None:
        pool = tuple(r.slot for r in rs)
        params["recurrent_slots"] = _get_slot_tensor(pool)
        parity = tuple(int(getattr(r, "parity", 0)) for r in rs)
        spec = bool(params.get("recurrent_history"))
        params["recurrent_slots_scan"] = _get_slot_tensor(
            tuple(2 * s + ((1 - p) if spec else p) for s, p in zip(pool, parity)),
            ("scan", parity, spec))
        if spec:
            params["recurrent_slots_scan_in"] = _get_slot_tensor(
                tuple(2 * s + p for s, p in zip(pool, parity)), ("scan_in", parity))


def advance_recurrent_states(input_ids: torch.Tensor, params: dict, model):
    rs = params.get("recurrent_states")
    history = params.get("recurrent_history")
    if rs:
        bsz, seqlen = input_ids.shape
        assert len(rs) == bsz
        for i, r in enumerate(rs):
            r.position += seqlen
            r.last_history = (seqlen - 1) if history else 0
            if history:
                # [Path A] rewind() replays this pass's staged scan inputs (keyed by shape,
                # batch-indexed) for the accepted prefix; i is the batch row this state held
                r.spec_row = i
                r.spec_shape = (bsz, seqlen)
            r.post_advance()
