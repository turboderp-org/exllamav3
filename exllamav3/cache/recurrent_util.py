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

    # Create slot index tensor
    if rs is not None:
        params["recurrent_slots"] = _get_slot_tensor(tuple(r.slot for r in rs))


def advance_recurrent_states(input_ids: torch.Tensor, params: dict, model):
    rs = params.get("recurrent_states")
    history = params.get("recurrent_history")
    if rs:
        bsz, seqlen = input_ids.shape
        assert len(rs) == bsz
        for r in rs:
            r.position += seqlen
            r.last_history = (seqlen - 1) if history else 0
            r.post_advance()
