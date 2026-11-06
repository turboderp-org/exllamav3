"""
LinearFP16Packed (conversion-time packed storage of natively quantized checkpoint tensors, selected by
infer_params.packed_fp16_source) against the eager fp16 load of the same synthetic checkpoint, through the real
loader: the quantizer-facing weight (get_weight_tensor), the forward, get_tensors and the swap round trip must be
identical, and the packed storage must be smaller. Formats, as the loader branches route them:

- fp8 e4m3 `weight` + `weight_scale_inv` 128 x 128 block grid, including a partial last block (DeepSeek-V3 style)
- fp8 `weight` + per-row and per-tensor `weight_scale` (Transformers fp8), with a bias
- int8 fp4 nibbles `.weight` + E8M0 `.scale` (1 x 32 blocks, DeepSeek-V4 / MiMo experts)
- fp8 e4m3 `.weight` + E8M0 `.scale` 128 x 128 (DeepSeek-V4 dense tensors)
- alt_key part lists, fp8 + per-part block scales concatenated along the output dimension (Ling 3.x)
- bf16 weights are not packed (LinearFP16 as before)

Padded declarations (out_features above the stored rows) must zero the pad columns in both forms.
"""

import pytest
import torch

from exllamav3.modules import Linear
from exllamav3.modules.quant.fp16 import LinearFP16, LinearFP16Packed

from testlib.checkpoint import module_config

KEY = "model.layers.0.mlp.up_proj"
pytestmark = pytest.mark.cuda_only


def fp8_bytes(shape, gen, dtype = torch.float8_e4m3fn):
    return (torch.randn(shape, generator = gen) * 0.5).to(dtype)


def e8m0(shape, gen):
    return torch.randint(118, 134, shape, dtype = torch.uint8, generator = gen)


def checkpoints(k: int, n: int):
    """(name, tensors, in_features, out_features, expect_packed); weights stored (out, in) as in the checkpoints"""
    gen = torch.Generator().manual_seed(0)
    cases = []
    # fp8 + 128-block scale_inv, partial last block on the output side (n - 64 rows stored)
    w = fp8_bytes((n - 64, k), gen)
    si = torch.rand(((n - 64 + 127) // 128, (k + 127) // 128), generator = gen) * 0.1
    cases.append(("fp8_block_scale_inv_partial", {f"{KEY}.weight": w, f"{KEY}.weight_scale_inv": si}, k, n, True))
    # fp8 + per-row scale + bias
    w = fp8_bytes((n, k), gen)
    cases.append(("fp8_row_scale_bias", {
        f"{KEY}.weight": w, f"{KEY}.weight_scale": torch.rand((n, 1), generator = gen) * 0.1,
        f"{KEY}.bias": torch.randn((n,), generator = gen).half()}, k, n, True))
    # fp8 e5m2 + per-tensor scale
    w = fp8_bytes((n, k), gen, torch.float8_e5m2)
    cases.append(("fp8_e5m2_tensor_scale", {f"{KEY}.weight": w, f"{KEY}.weight_scale": torch.tensor(0.0371)}, k, n, True))
    # packed fp4 + E8M0 (1 x 32)
    w = torch.randint(-128, 128, (n, k // 2), dtype = torch.int8, generator = gen)
    cases.append(("fp4_e8m0_32", {f"{KEY}.weight": w, f"{KEY}.scale": e8m0((n, k // 32), gen)}, k, n, True))
    # fp8 + E8M0 (128 x 128)
    w = fp8_bytes((n, k), gen)
    cases.append(("fp8_e8m0_128", {f"{KEY}.weight": w, f"{KEY}.scale": e8m0((n // 128, k // 128), gen)}, k, n, True))
    # bf16: stays a plain LinearFP16
    cases.append(("bf16_plain", {f"{KEY}.weight": torch.randn((n, k), generator = gen).bfloat16()}, k, n, False))
    return cases


def build(tensors, tmp_path, k, n, packed: bool, **kwargs):
    module = Linear(module_config(tensors, tmp_path / ("packed" if packed else "plain"), packed_fp16_source = packed),
                    KEY, k, n, **kwargs)
    return module


def compare(plain: Linear, packed: Linear, device, expect_packed: bool):
    plain.load(device)
    packed.load(device)
    assert isinstance(plain.inner, LinearFP16) and not isinstance(plain.inner, LinearFP16Packed)
    assert isinstance(packed.inner, LinearFP16Packed) == expect_packed
    w0 = plain.inner.get_weight_tensor()
    w1 = packed.inner.get_weight_tensor()
    assert w0.shape == w1.shape == (plain.in_features, plain.out_features)
    assert torch.equal(w0, w1)
    x = torch.randn((3, 5, plain.in_features), device = device).half()
    with torch.inference_mode():
        assert torch.equal(plain.forward(x, {}), packed.forward(x, {}))
    t0, t1 = plain.inner.get_tensors(KEY), packed.inner.get_tensors(KEY)
    assert t0.keys() == t1.keys() and all(torch.equal(t0[k], t1[k]) for k in t0)
    if expect_packed:
        assert packed.storage_size() < plain.inner.weight.numel() * 2, "packed storage not smaller than fp16"
        packed.inner.swap_cpu()
        assert packed.inner.segments[0].packed.device.type == "cpu"
        assert torch.equal(packed.inner.get_weight_tensor().to(device), w0)
        packed.inner.unswap_cpu()
        assert torch.equal(packed.inner.get_weight_tensor(), w0)
        with pytest.raises(NotImplementedError):
            packed.inner.set_weight(w0)
    plain.unload()
    packed.unload()


@pytest.mark.parametrize("case", checkpoints(256, 384), ids = lambda c: c[0])
def test_packed_matches_eager(tmp_path, device, case):
    name, tensors, k, n, expect_packed = case
    compare(build(tensors, tmp_path, k, n, False), build(tensors, tmp_path, k, n, True), device, expect_packed)


def test_alt_key_parts(tmp_path, device):
    """Fused parts (Ling 3.x gate/up stored separately) with their own block scales"""
    gen = torch.Generator().manual_seed(1)
    k, n1, n2 = 256, 64, 192     # parts sum to a 128-multiple (as the real fused parts do); each has a partial block
    keys = [f"{KEY}.part_a", f"{KEY}.part_b"]
    tensors = {}
    for key, n in zip(keys, (n1, n2)):
        tensors[f"{key}.weight"] = fp8_bytes((n, k), gen)
        tensors[f"{key}.weight_scale_inv"] = torch.rand(((n + 127) // 128, (k + 127) // 128), generator = gen) * 0.1
    plain = build(tensors, tmp_path, k, n1 + n2, False, alt_key = keys)
    packed = build(tensors, tmp_path, k, n1 + n2, True, alt_key = keys)
    compare(plain, packed, device, True)
    packed.load(device)
    assert len(packed.inner.segments) == 2 and packed.inner.segments[1].col0 == n1
    packed.unload()
