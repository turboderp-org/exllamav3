"""
Qwen3.8-Flash-Next gated residual (GatedResidual, the low-rank hyper-connection) against the HF reference module
(Qwen4ExpTextGatedResidual) with real weights from the 4-layer stub (registry role qwen4-exp-stub; the gated
residual tables are stored unquantized), at all eight block sites and the final combine-less mixer:

  - _mix_ref (fp32 torch reference) must match fp32 HF to float rounding (1e-5): the anchor proving the math
  - the prefill path (R > FUSED_MAX_R rows: the tiled int8 tensor-core kernel, and the cuBLAS fallback it replaces
    where the shape or architecture does not fit) and the fused decode kernel (R <= FUSED_MAX_R) are gated against
    fp32 HF with a tolerance calibrated by a bf16-HF control (HF runs this model in bf16), plus a mutual consistency
    bound between the two paths
  - apply_ (ext.hc_apply without a comb) through the site output
"""

import pytest
import torch

from testlib.compare import rel_err
from testlib.parity import Gate, control_check

pytestmark = [pytest.mark.hf, pytest.mark.model("qwen4-exp-stub")]

B, S = 2, 33
SITES = [(layer, site) for layer in range(4) for site in ("attn_hyper_connection", "mlp_hyper_connection")]


@pytest.fixture(scope = "module")
def stub(model_registry):
    from transformers import AutoConfig
    from exllamav3.loader.safetensors import SafetensorsCollection
    model_dir = model_registry.get("qwen4-exp-stub").path
    stc = SafetensorsCollection(model_dir)
    yield AutoConfig.from_pretrained(model_dir).text_config, stc
    stc.close()


def hf_site(cfg, stc, key, use_combine, dtype):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual
    mod = Qwen4ExpTextGatedResidual(cfg, use_combine = use_combine)
    names = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight") + \
        (("block_inject_weight.weight",) if use_combine else ())
    mod.load_state_dict({n: stc.get_tensor(f"{key}.{n}", "cpu").float() for n in names})
    return mod.to(dtype)


def exl_site(cfg, stc, key, use_combine, device, path):
    from types import SimpleNamespace
    from exllamav3.modules import GatedResidual
    mod = GatedResidual(SimpleNamespace(stc = stc), key, cfg.hc_count, cfg.hidden_size, cfg.rms_norm_eps, use_combine)
    mod.load(device, keep_source_weights = True)     # the reference and cuBLAS paths read the fp16 sources
    if path == "cublas":
        mod.tiled = False
    elif not mod.tiled:
        pytest.skip("the tiled kernel does not apply to this shape / device")
    return mod


def _rel(ref, got):
    return rel_err(got.float().cpu(), ref.float())


@pytest.mark.parametrize("path", ["tiled", "cublas"])
@pytest.mark.parametrize("layer, site", SITES)
@torch.inference_mode()
def test_site(stub, layer, site, path, device):
    from exllamav3.modules import GatedResidual
    cfg, stc = stub
    H, D = cfg.hc_count, cfg.hidden_size
    key = f"model.language_model.layers.{layer}.{site}"
    hf = hf_site(cfg, stc, key, True, torch.float32)
    hf16 = hf_site(cfg, stc, key, True, torch.bfloat16)
    ex = exl_site(cfg, stc, key, True, device, path)
    g = torch.Generator().manual_seed(layer * 2 + (site == "mlp_hyper_connection"))
    hyper = torch.randn(B, S, H * D, generator = g) * 0.7 + 0.05
    y = torch.randn(B, S, D, generator = g)          # stand-in sublayer output

    mixed_hf, hyper_out, inj_hf = hf(hyper)
    out_hf = hyper_out + (y.unsqueeze(-2) * inj_hf.unsqueeze(-1)).flatten(-2)
    m16, _, i16 = hf16(hyper.to(torch.bfloat16))
    streams = hyper.view(B, S, H, D).to(device).float().contiguous()
    gate = Gate(f"{key} ({path})")

    # fp32 reference path: float-rounding parity with fp32 HF
    post_r, mixed_r = ex._mix_ref(streams)
    gate.lt("_mix_ref mixed", _rel(mixed_hf, mixed_r), 1e-5)
    gate.lt("_mix_ref inject", _rel(inj_hf, post_r), 1e-5)

    # prefill path (R = B * S rows) and the fused decode kernel (a small-R slice)
    assert B * S > GatedResidual.FUSED_MAX_R
    post_g, comb, mixed_g = ex.mix(streams, {})
    gate.true("no comb", comb is None)
    out_g = ex.apply_(streams.clone(), y.to(device).half(), post_g, comb, {}).flatten(-2)
    sf = GatedResidual.FUSED_MAX_R // B
    post_f, _, mixed_f = ex.mix(streams[:, :sf].contiguous(), {})

    # bf16-HF control calibrates the half-precision tolerance
    for ref, got, ctrl, name in (
        (mixed_hf, mixed_g, m16, f"mixed ({path})"),
        (inj_hf, post_g, i16, f"inject ({path})"),
        (mixed_hf[:, :sf], mixed_f, m16[:, :sf], "mixed (fused)"),
        (inj_hf[:, :sf], post_f, i16[:, :sf], "inject (fused)"),
    ):
        control_check(gate, name, _rel(ref, got), _rel(ref, ctrl))
    control_check(gate, "site out", _rel(out_hf, out_g), _rel(mixed_hf, m16))

    # the two kernel paths agree tightly with each other (same weights, both half internals)
    gate.lt("fused vs prefill path", _rel(mixed_g[:, :sf].float().cpu(), mixed_f), 3e-3)
    gate.assert_passes()


@pytest.mark.parametrize("path", ["tiled", "cublas"])
@torch.inference_mode()
def test_final_mixer(stub, path, device):
    from exllamav3.modules import GatedResidual
    cfg, stc = stub
    H, D = cfg.hc_count, cfg.hidden_size
    key = "model.language_model.hyper_connection_mixer"
    hf = hf_site(cfg, stc, key, False, torch.float32)
    hf16 = hf_site(cfg, stc, key, False, torch.bfloat16)
    ex = exl_site(cfg, stc, key, False, device, path)
    hyper = torch.randn(B, S, H * D, generator = torch.Generator().manual_seed(100))
    ref = hf(hyper)
    ctrl = hf16(hyper.to(torch.bfloat16))
    streams = hyper.view(B, S, H, D).to(device).float().contiguous()
    _, mixed_r = ex._mix_ref(streams)
    sf = GatedResidual.FUSED_MAX_R // B
    out_g = ex.forward(streams, {})
    out_f = ex.forward(streams[:, :sf].contiguous(), {})
    gate = Gate(f"{key} ({path})")
    gate.lt("_mix_ref", _rel(ref, mixed_r), 1e-5)
    c = _rel(ref, ctrl)
    control_check(gate, f"mixed ({path})", _rel(ref, out_g), c)
    control_check(gate, "mixed (fused)", _rel(ref[:, :sf], out_f), c)
    gate.assert_passes()
