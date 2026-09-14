from __future__ import annotations


def hf_device_map_from_split(model_dir, split, dtype = None, trust_remote_code = None):
    """
    Explicit per-layer device map for a Transformers model from a -gs style GiB budget list. Built with
    accelerate's planner on a meta-device instance: Transformers' own device_map = "auto" path plans
    more conservatively (it declined a 91.5 GiB model with 122 GiB of budget, insisting on disk offload)
    and does not reserve room for its conversion transients (fused expert tensors)
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    from accelerate.utils import infer_auto_device_map
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code = trust_remote_code)
    dtype = dtype or getattr(cfg, "dtype", None) or torch.bfloat16
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, dtype = dtype, trust_remote_code = trust_remote_code)
    max_memory = {i: f"{float(g)}GiB" for i, g in enumerate(split) if float(g) > 0}
    # accelerate wraps anything that is not a list/tuple into a one-element list, so a set (what
    # Transformers stores) would silently disable the no-split check and split a layer across GPUs
    dm = infer_auto_device_map(m, max_memory = max_memory, no_split_module_classes = list(m._no_split_modules or []), dtype = dtype)
    assert all(isinstance(v, int) or str(v).startswith("cuda") for v in dm.values()), f"budget too small: {dm}"
    return dm
