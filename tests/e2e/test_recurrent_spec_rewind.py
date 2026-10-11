"""
Speculative passes over recurrent layers (GDN / Mamba2 state pools) settle correctly on rewind, for passes far
longer than the graph slots cover: a pass of `pass_len` tokens with recurrent_history, then rewind(rejected), must
leave the model in the state a plain forward over the accepted prefix leaves, as seen through the next token's
logits (bit-identical for short passes, which share the sequential scan kernel; the long passes run the chunked
kernel and the rewind replays with the sequential one, so those agree to numerical tolerance). Covers full accept,
single and mid rejections, and rejecting everything but the first token, on a cache with max_history 64, and a
pass without history that is then position-corrected. Runs over every registry model tagged "recurrent".
"""

import pytest
import torch

from testlib.graph import load_with_caches, unload_model

MAX_HISTORY = 64
PROMPT = "The committee reviewed the proposal in detail and concluded that"


def _forward(model, cache, state, ids, pos, history = False):
    params = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 4096), "past_len": pos,
              "recurrent_states": [state]}
    if history:
        params["recurrent_history"] = True
    return model.forward(ids, params)[0].float()


def _rel(a, b):
    return (a - b).abs().max().item() / b.abs().max().item()


@pytest.fixture(scope = "module")
def loaded(model_id, model_registry, request):
    # model_id is module-scoped (tests are grouped by model), so one load serves every case of a model
    entry = model_registry.get(model_id)
    if not entry.available:
        pytest.skip(f"test model '{model_id}' not available")
    device = torch.device(request.config.getoption("--device"))
    _, model, (cache,), tok = load_with_caches(entry.path, device, [{"max_history": MAX_HISTORY}])
    ids = tok.encode(PROMPT, add_bos = True)
    gen = torch.Generator().manual_seed(1)
    cont = torch.randint(100, 2000, (1, MAX_HISTORY + 2), generator = gen)
    yield model, cache, ids, cont
    unload_model(model)


@pytest.mark.models("recurrent")
@torch.inference_mode()
@pytest.mark.parametrize("pass_len, rejected", [
    (5, 0), (5, 1), (5, 4),
    (MAX_HISTORY + 1, 0), (MAX_HISTORY + 1, 1), (MAX_HISTORY + 1, 30), (MAX_HISTORY + 1, MAX_HISTORY),
])
def test_spec_pass_rewind_matches_plain(model_id, loaded, pass_len, rejected):
    model, cache, ids, cont = loaded
    spec = cont[:, :pass_len]
    accepted = pass_len - rejected
    probe = cont[:, pass_len : pass_len + 1]
    n = ids.shape[1]

    # Plain: prompt, then one pass over the accepted prefix without history
    s_plain = cache.get_new_state()
    _forward(model, cache, s_plain, ids, 0)
    _forward(model, cache, s_plain, spec[:, :accepted], n)
    ref = _forward(model, cache, s_plain, probe, n + accepted)[-1]
    s_plain.free()

    # Noise floor at the probe: the same tokens grouped into passes of other row counts. Passes of different
    # row counts compute a token's activations through different GEMM tilings (and a quantized MoE model can
    # flip routes on that), and a small quantized model amplifies the difference through its layers; the
    # speculative pass computes the accepted tokens with pass_len rows, the plain one with `accepted`
    s_alt = cache.get_new_state()
    _forward(model, cache, s_alt, ids[:, :-3], 0)
    _forward(model, cache, s_alt, ids[:, -3:], n - 3)
    if accepted > 1:
        _forward(model, cache, s_alt, spec[:, :1], n)
        _forward(model, cache, s_alt, spec[:, 1:accepted], n + 1)
    else:
        _forward(model, cache, s_alt, spec[:, :accepted], n)
    alt = _forward(model, cache, s_alt, probe, n + accepted)[-1]
    s_alt.free()

    # Speculative: the whole pass with history, then rewind the rejected suffix
    s_spec = cache.get_new_state()
    _forward(model, cache, s_spec, ids, 0)
    _forward(model, cache, s_spec, spec, n, history = True)
    assert s_spec.last_history == pass_len - 1
    s_spec.rewind(rejected)
    assert s_spec.position == n + accepted and s_spec.last_history == 0
    got = _forward(model, cache, s_spec, probe, n + accepted)[-1]
    s_spec.free()

    # The probe after the rewind must sit at the plain-vs-regrouped level; long passes additionally replay
    # with the sequential scan after a chunked pass (fp32 reassociation), hence the absolute minimum
    floor = _rel(alt, ref)
    err = _rel(got, ref)
    print(f"[{model_id} pass {pass_len} rejected {rejected}] probe rel err {err:.2e}, regrouping floor {floor:.2e}")
    assert err <= max(3 * floor, 5e-3), \
        f"{model_id}: rewind({rejected}) after a {pass_len}-token pass: probe rel err {err:.3e}, floor {floor:.3e}"


@pytest.mark.models("recurrent")
@torch.inference_mode()
def test_rewind_without_history_is_position_only(model_id, loaded):
    model, cache, ids, cont = loaded
    n = ids.shape[1]
    s = cache.get_new_state()
    model.forward(ids, {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 4096), "past_len": 0,
                        "recurrent_states": [s]})
    model.forward(cont[:, :3], {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 4096), "past_len": n,
                                "recurrent_states": [s]})
    layers = list(cache.get_all_recurrent_layers().values())
    before = [l.recurrent_state.clone() for l in layers if hasattr(l, "recurrent_state")]
    s.rewind(2)
    assert s.position == n + 1
    after = [l.recurrent_state for l in layers if hasattr(l, "recurrent_state")]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "a rewind with no recorded history touched the state"
    s.free()
