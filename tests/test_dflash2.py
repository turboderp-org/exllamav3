from types import SimpleNamespace
from unittest.mock import patch

import torch

from exllamav3.architecture.architectures import ARCHITECTURES
from exllamav3.architecture.dflash import DFlashModel
from exllamav3.architecture.dflash2 import DFlash2Model
from exllamav3.modules.arch_specific.dflash import DFlashInputLayer
from exllamav3.modules.arch_specific.dflash2 import (
    DFlash2Block,
    DFlash2DynConv,
    DFlash2Selector,
    _grouped_dynamic_convolve,
    _grouped_dynamic_convolve_torch,
    has_triton,
)


def test_architecture_is_registered():
    assert "DFlash2DraftModel" in ARCHITECTURES


def test_prepare_inputs_sets_anchor_and_uses_dflash_attention_setup():
    model = object.__new__(DFlash2Model)
    input_ids = torch.tensor([[42]])
    params = {}
    prepared = object()

    with patch.object(DFlashModel, "prepare_inputs", return_value = prepared) as prepare:
        actual = DFlash2Model.prepare_inputs(model, input_ids, params)

    assert actual is prepared
    assert params["dflash2_anchor_ids"] is input_ids
    prepare.assert_called_once_with(input_ids, params)


def test_input_embedding_scale_is_applied():
    class Embedding:
        def forward(self, ids, params):
            return ids.unsqueeze(-1).float()

    target = SimpleNamespace(loaded_tp = False, modules = [Embedding()])
    layer = object.__new__(DFlashInputLayer)
    layer.native_draft_len = 2
    layer.mask_token_id = 9
    layer.input_embedding_scale = 2.0
    layer.attached_model = lambda: target

    actual = DFlashInputLayer.forward(layer, torch.tensor([[3]]), {})

    assert actual.tolist() == [[[6.0], [18.0]]]


def test_tp_target_is_rejected_during_attach():
    model = object.__new__(DFlash2Model)
    target = SimpleNamespace(loaded_tp = True)

    try:
        model.attach_to(target)
    except NotImplementedError as exc:
        assert "tensor-parallel targets" in str(exc)
    else:
        raise AssertionError("DFlash2 attached to a tensor-parallel target")


def test_target_taps_are_reordered_at_dflash_projection():
    class Projection:
        def forward(self, x, params, out_dtype = None):
            return x

    class Norm:
        def forward(self, x, params, out_dtype = None):
            return x

    model = object.__new__(DFlashModel)
    model.config = SimpleNamespace(target_layer_ids = [5, 2])
    model.input_layer = SimpleNamespace(
        device = torch.device("cpu"),
        proj = Projection(),
        norm = Norm(),
    )
    model.attn_modules = []
    params = {}

    state_2 = torch.full((1, 1, 1), 2.0)
    state_5 = torch.full((1, 1, 1), 5.0)
    model.update_kv_from_target([state_2, state_5], None, params)

    assert params["target_hidden_cc"].tolist() == [[[5.0, 2.0]]]


def test_exact_projection_widths_are_preserved():
    config = SimpleNamespace()
    conv = DFlash2DynConv(
        config, "conv", hidden_size = 96, kernel_size = 3, group_size = 16)
    selector = DFlash2Selector(
        config, "selector", vocab_size = 32, hidden_size = 96, rank = 64, top_k = 4)

    assert conv.proj.out_features_unpadded == 36
    assert conv.proj.out_features == 128
    assert conv.proj.trim_padded_out is True
    assert selector.hidden_proj.out_features_unpadded == 64
    assert selector.hidden_proj.out_features == 128
    assert selector.hidden_proj.trim_padded_out is True


def _dynamic_conv_reference(hidden, dynamic, base, group_size):
    expected = torch.zeros_like(hidden)
    for batch in range(hidden.shape[0]):
        for position in range(hidden.shape[1]):
            for channel in range(hidden.shape[2]):
                group = channel // group_size
                for offset in range(base.shape[0]):
                    if position >= offset:
                        weight = base[offset, channel] + dynamic[batch, position, offset, group]
                        expected[batch, position, channel] += weight * hidden[batch, position - offset, channel]
    return expected


def test_grouped_dynamic_convolve_torch_matches_reference():
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 8)
    dynamic = torch.randn(2, 5, 3, 2)
    base = torch.randn(3, 8)

    actual = _grouped_dynamic_convolve_torch(hidden, dynamic, base, group_size = 4)
    expected = _dynamic_conv_reference(hidden, dynamic, base, group_size = 4)

    torch.testing.assert_close(actual, expected)


def test_grouped_dynamic_convolve_triton_matches_torch():
    if not torch.cuda.is_available() or not has_triton:
        return

    torch.manual_seed(1)
    hidden = torch.randn(2, 8, 96, dtype = torch.float16, device = "cuda")
    packed = torch.randn(2, 8, 2, 3, 6, dtype = torch.float16, device = "cuda")
    dynamic = packed[:, :, 1]
    base = torch.randn(3, 96, dtype = torch.bfloat16, device = "cuda")
    assert not dynamic.is_contiguous()

    actual = _grouped_dynamic_convolve(hidden, dynamic, base, group_size = 16)
    expected = _grouped_dynamic_convolve_torch(hidden, dynamic, base, group_size = 16)

    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol = 2e-3, atol = 2e-3)


def test_grouped_dynamic_convolve_triton_handles_odd_geometry():
    if not torch.cuda.is_available() or not has_triton:
        return

    torch.manual_seed(2)
    hidden = torch.randn(1, 17, 192, dtype = torch.float32, device = "cuda")
    packed = torch.randn(1, 17, 2, 5, 8, dtype = torch.float16, device = "cuda")
    dynamic = packed[:, :, 0]
    base = torch.randn(5, 192, dtype = torch.bfloat16, device = "cuda")

    actual = _grouped_dynamic_convolve(hidden, dynamic, base, group_size = 24)
    expected = _grouped_dynamic_convolve_torch(hidden, dynamic, base, group_size = 24)

    torch.testing.assert_close(actual, expected, rtol = 1e-5, atol = 1e-5)


def test_grouped_dynamic_convolve_triton_preserves_fp32_finish():
    if not torch.cuda.is_available() or not has_triton:
        return

    hidden = torch.full((1, 8, 96), 1.5e5, dtype = torch.float32, device = "cuda")
    dynamic = torch.zeros(1, 8, 2, 6, dtype = torch.float16, device = "cuda")
    base = torch.ones(2, 96, dtype = torch.bfloat16, device = "cuda")

    actual = _grouped_dynamic_convolve(hidden, dynamic, base, group_size = 16)
    expected = _grouped_dynamic_convolve_torch(hidden, dynamic, base, group_size = 16)

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert actual.max() > 65504
    torch.testing.assert_close(actual, expected)


class _Norm:
    def __init__(self):
        self.input_dtypes = []

    def forward(self, x, params, out_dtype = None):
        self.input_dtypes.append(x.dtype)
        assert out_dtype == torch.half
        return x.half()


class _Conv:
    def __init__(self):
        self.prepare_dtypes = []
        self.finish_dtypes = []

    def prepare(self, x, params):
        self.prepare_dtypes.append(x.dtype)
        return x, None

    def finish(self, x, dynamic):
        self.finish_dtypes.append(x.dtype)
        return x.float()


class _Branch:
    def forward(self, x, params):
        return x.float()


def test_dflash2_block_keeps_fp32_residual_and_fp16_branches():
    block = object.__new__(DFlash2Block)
    block.attn_norm = _Norm()
    block.mlp_norm = _Norm()
    block.attn_conv = _Conv()
    block.mlp_conv = _Conv()
    block.attn = _Branch()
    block.mlp = _Branch()

    actual = block.forward(torch.ones(1, 2, 4, dtype = torch.float32), {})

    assert actual.dtype == torch.float32
    assert block.attn_norm.input_dtypes == [torch.float32]
    assert block.mlp_norm.input_dtypes == [torch.float32]
    assert block.attn_conv.prepare_dtypes == [torch.float16]
    assert block.mlp_conv.prepare_dtypes == [torch.float16]
    assert block.attn_conv.finish_dtypes == [torch.float32]
    assert block.mlp_conv.finish_dtypes == [torch.float32]


def test_candidate_logit_scale_and_softcap_are_applied():
    class LMHead:
        def prepare_for_device(self, state, params):
            return state

        def forward(self, state, params):
            return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    class Selector:
        device = torch.device("cpu")

        def walk(self, hidden, logits, anchor, return_confidence = False):
            self.logits = logits
            return torch.zeros((1, 1), dtype = torch.long)

    target = SimpleNamespace(
        loaded_tp = False,
        logit_layer_idx = 0,
        modules = [LMHead()],
        config = SimpleNamespace(vocab_size = 2),
    )
    selector = Selector()
    model = object.__new__(DFlash2Model)
    model.config = SimpleNamespace(output_multiplier = 2.0, final_logit_softcapping = 3.0)
    model.attached_model = lambda: target
    model.selector = selector
    params = {"dflash2_anchor_ids": torch.tensor([[1]])}

    DFlash2Model.sample_from_state(model, torch.zeros(1, 2, 4), params)

    assert "draft_conf" not in params
    raw = torch.tensor([[3.0, 4.0]]) * 2.0
    expected = torch.tanh(raw / 3.0) * 3.0
    torch.testing.assert_close(selector.logits[0], expected)


def test_sample_exports_selector_confidence_for_dynamic_drafting():
    class LMHead:
        def prepare_for_device(self, state, params):
            return state

        def forward(self, state, params):
            return torch.zeros(1, 3, 4)

    class Selector:
        device = torch.device("cpu")

        def walk(self, hidden, logits, anchor, return_confidence = False):
            assert return_confidence
            return torch.tensor([[2, 3]]), torch.tensor([[7.5, 4.25]])

    target = SimpleNamespace(
        loaded_tp = False,
        logit_layer_idx = 0,
        modules = [LMHead()],
        config = SimpleNamespace(vocab_size = 4),
    )
    model = object.__new__(DFlash2Model)
    model.config = SimpleNamespace(output_multiplier = 1.0, final_logit_softcapping = 0.0)
    model.attached_model = lambda: target
    model.selector = Selector()
    params = {
        "dflash2_anchor_ids": torch.tensor([[1]]),
        "export_draft_conf": True,
    }

    ids = DFlash2Model.sample_from_state(model, torch.zeros(1, 3, 4), params)

    assert ids.tolist() == [[1, 2, 3]]
    assert params["draft_conf"].tolist() == [[0.0, 7.5, 4.25]]


class _Projection:
    def forward(self, hidden, params):
        return torch.ones((*hidden.shape[:-1], 1), device = hidden.device)


def test_selector_chains_candidates_from_anchor_and_returns_scores():
    selector = object.__new__(DFlash2Selector)
    selector.top_k = 2
    selector.hidden_proj = _Projection()
    selector.pred_codebook = torch.tensor([[1.0], [0.0], [-1.0], [0.0]])
    selector.succ_codebook = torch.tensor([[0.0], [0.0], [2.0], [2.0]])

    hidden = torch.zeros(1, 2, 4)
    logits = torch.tensor([[[0.0, 3.0, 2.0, -1.0], [0.0, 3.0, -1.0, 2.0]]])
    path, confidence = selector.walk(
        hidden, logits, torch.tensor([0]), return_confidence = True)

    assert path.tolist() == [[2, 1]]
    assert confidence.tolist() == [[4.0, 3.0]]


if __name__ == "__main__":
    test_architecture_is_registered()
    test_prepare_inputs_sets_anchor_and_uses_dflash_attention_setup()
    test_input_embedding_scale_is_applied()
    test_tp_target_is_rejected_during_attach()
    test_target_taps_are_reordered_at_dflash_projection()
    test_exact_projection_widths_are_preserved()
    test_grouped_dynamic_convolve_torch_matches_reference()
    test_grouped_dynamic_convolve_triton_matches_torch()
    test_grouped_dynamic_convolve_triton_handles_odd_geometry()
    test_grouped_dynamic_convolve_triton_preserves_fp32_finish()
    test_dflash2_block_keeps_fp32_residual_and_fp16_branches()
    test_candidate_logit_scale_and_softcap_are_applied()
    test_sample_exports_selector_confidence_for_dynamic_drafting()
    test_selector_chains_candidates_from_anchor_and_returns_scores()
    print("DFlash2 tests passed")
