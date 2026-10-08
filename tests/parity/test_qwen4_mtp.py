"""
Qwen3.8-Flash-Next MTP head mechanics on the 4-layer stub (registry role qwen4-exp-stub). There is no reference
implementation for this head (HF ignores the mtp.* tensors), so the test validates structure rather than semantics:
the trunk's final mixer exports the pre-collapse stream stack, every mtp.* checkpoint tensor is consumed by a module,
the full draft chain (input combine -> qwen4 block with QSA -> mixer -> shared lm_head) produces finite logits, and
the two candidate input-combine variants (stream_tap True / False) genuinely differ (which one is right is settled by
acceptance rate on the full model).
"""

import re

import pytest
import torch

pytestmark = pytest.mark.model("qwen4-exp-stub")

S = 96


def _canon(keys):
    """BlockSparseMLP reads stacked expert tensors but reports per-expert names; map both to the stacked names"""
    out = set()
    for k in keys:
        m = re.match(r"(.*\.experts)\.\d+\.(?:gate|up)_proj\.weight$", k)
        if m:
            out.add(f"{m.group(1)}.gate_up_proj")
            continue
        m = re.match(r"(.*\.experts)\.\d+\.down_proj\.weight$", k)
        if m:
            out.add(f"{m.group(1)}.down_proj")
            continue
        out.add(k)
    return out


def _load(config, module, device, **kwargs):
    config.stc.begin_deferred_load()
    module.load(device, **kwargs)
    config.stc.end_deferred_load()


def _release(config, module):
    from exllamav3.util.memory import free_mem
    module.unload()
    config.stc.close()
    free_mem()


@pytest.fixture(scope = "module")
def trunk(model_registry, device):
    """(config, model, input ids, exported pre-mixer stack, mixer input stack, trunk logits)"""
    from exllamav3 import Config, Model
    from exllamav3.modules import GatedResidual
    config = Config.from_directory(model_registry.get("qwen4-exp-stub").path)
    config.override_dynamic_seq_len(S)
    model = Model.from_config(config)
    mixer = model.modules[model.logit_layer_idx - 1]
    assert isinstance(mixer, GatedResidual) and not mixer.use_combine
    input_ids = torch.randint(0, 200000, (1, S), generator = torch.Generator().manual_seed(11))
    params = {"export_state_norm_keys": {mixer.key}}
    state = model.prepare_inputs(input_ids, params)
    exported = pre_mixer = None
    with torch.inference_mode():
        for module in model.modules:
            _load(config, module, device)
            state = module.prepare_for_device(state, params)
            if module is mixer:
                pre_mixer = state.flatten(-2).half().cpu()
            state = module.forward(state, params)
            if module is mixer:
                exported = params["export_states"][-1].cpu()
            _release(config, module)
    # the draft borrows the trunk's embedding, which the streaming loop unloaded
    model.modules[0].load(torch.device("cpu"))
    return config, model, mixer, input_ids, exported, pre_mixer, state.float().cpu()


@torch.inference_mode()
def _mtp_forward(trunk, stream_tap, device):
    from exllamav3 import Model
    config, model, mixer, input_ids, exported, _, _ = trunk
    mtp = Model.from_config(config, component = "mtp")
    mtp.input_layer.stream_tap = stream_tap
    mtp.attach_to(model)
    assert mtp.draft_verifier_params["export_state_norm_keys"] == {mixer.key}
    p = {"target_hidden": exported[:, :-1].to(device)}
    x = mtp.prepare_inputs(input_ids[:, 1:], p)
    consumed = set()
    for module in mtp.modules:
        # get_tensors() reports the checkpoint tensors a module consumed; gated residuals only keep their fp16
        # sources when asked (conversion does)
        _load(config, module, device, keep_source_weights = True)
        for m in module:
            consumed |= set(m.get_tensors().keys())
        x = module.prepare_for_device(x, p)
        x = module.forward(x, p)
        _release(config, module)
    # the draft chain's terminal module returns the flattened PRE-mixer stack (it feeds the next drafting step);
    # collapse it like sample_from_state does, then the shared lm_head
    mtp_mixer = mtp.stack_out.mixer
    _load(config, mtp_mixer, device)
    x = mtp_mixer.forward(x.view(x.shape[0], x.shape[1], mtp_mixer.hc_mult, mtp_mixer.hidden_size)
                          .float().contiguous(), p)
    _release(config, mtp_mixer)
    lm = model.modules[model.logit_layer_idx]
    _load(config, lm, device)
    logits = lm.forward(lm.prepare_for_device(x, p), p).float().cpu()
    _release(config, lm)
    return logits, consumed


def test_trunk_export(trunk):
    _, _, _, _, exported, pre_mixer, _ = trunk
    assert torch.equal(exported, pre_mixer), "export != mixer input stack"


def test_tensor_coverage_and_logits(trunk, device):
    config = trunk[0]
    logits, consumed = _mtp_forward(trunk, True, device)
    all_mtp = {k for k in config.stc.tensor_file_map if k.startswith("mtp.")}
    missing = sorted(all_mtp - _canon(consumed))
    assert not missing, f"unconsumed mtp tensors: {missing}"
    assert logits.isfinite().all(), "non-finite MTP logits"


def test_stream_tap_variants_differ(trunk, device):
    logits_a, _ = _mtp_forward(trunk, True, device)
    logits_b, _ = _mtp_forward(trunk, False, device)
    tv = (logits_a.softmax(-1) - logits_b.softmax(-1)).abs().sum(-1).mean().item()
    assert tv > 1e-3, f"stream_tap True vs False: mean TV distance {tv:.3e} (variants must differ)"
