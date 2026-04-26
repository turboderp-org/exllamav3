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

    if batch_shape is not None:
        bsz, _ = batch_shape
        past_len = params.get("past_len", 0)
        if past_len > 0:
            rs = params.get("recurrent_states")
            if rs is None:
                raise ValueError(f"Past length given, but no previous state for recurrence in params")
            for k, v in rs.items():
                if not v.batched and v.position != past_len:
                    raise ValueError(f"recurrent states don't match input past_len")
        else:
            rl = model.get_recurrent_layers()
            rs = {attn.layer_idx: attn.new_recurrent_state() for attn in rl}
            params["recurrent_states"] = rs

    elif cache_seqlens is not None:
        # (Empty) states must be provided with cache_seqlens
        pass

    else:
        if "recurrent_states" in params:
            raise ValueError(f"recurrent_states given without bsz and seqlens")
