import torch

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
        recurrent_slots = torch.tensor([r.slot for r in rs], dtype = torch.int32)
        params["recurrent_slots"] = recurrent_slots


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
