import torch

# Slot index tensors are tiny and recur across forward passes; cache them with persistent device
# copies so each decode step doesn't rebuild and re-upload one
_slot_tensors = {}

def _get_slot_tensor(slots: tuple) -> torch.Tensor:
    t = _slot_tensors.get(slots)
    if t is None:
        if len(_slot_tensors) > 4096:
            _slot_tensors.clear()
        t = torch.tensor(list(slots), dtype = torch.int32)
        t._static_dev_cache = True
        _slot_tensors[slots] = t
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

    # Slot index tensors: the cache slot (conv rings, SWA / short-conv states), and for the
    # recurrent-state pools the row the scan writes plus, on a speculative pass (history), the
    # base row it starts from. Both come from the state's slot and parity (GDNState); state
    # types without a parity run in place on their slot
    if rs is not None:
        history = bool(params.get("recurrent_history"))
        slots = tuple(r.slot for r in rs)
        params["recurrent_slots"] = _get_slot_tensor(slots)
        base = tuple(2 * r.slot + getattr(r, "parity", 0) for r in rs)
        if history:
            params["recurrent_slots_scan"] = _get_slot_tensor(tuple(2 * r.slot + 1 - getattr(r, "parity", 0) for r in rs))
            params["recurrent_slots_scan_in"] = _get_slot_tensor(base)
        else:
            params["recurrent_slots_scan"] = _get_slot_tensor(base)
            params.pop("recurrent_slots_scan_in", None)


def advance_recurrent_states(input_ids: torch.Tensor, params: dict, model):
    rs = params.get("recurrent_states")
    history = params.get("recurrent_history")
    if rs:
        bsz, seqlen = input_ids.shape
        assert len(rs) == bsz
        for i, r in enumerate(rs):
            r.position += seqlen
            r.last_history = (seqlen - 1) if history else 0
            # Where this state's staged scan inputs sit for a rewind's replay
            if hasattr(r, "spec_row"):
                r.spec_row = i if history else None
                r.spec_shape = (bsz, seqlen) if history else None
            r.post_advance()
