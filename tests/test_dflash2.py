from types import SimpleNamespace
from unittest.mock import patch

import torch

from exllamav3.architecture.architectures import ARCHITECTURES
from exllamav3.modules.attention_fn.common import AttnArgs, get_non_causal_span_arglist
from exllamav3.modules.attention_fn.torch import (
    _torch_sdpa_nocache,
    fn_torch_sdpa_fallback_nocache,
)
from exllamav3.modules.attention_fn.triton_paged import has_triton, paged_attn_triton
from exllamav3.architecture.dflash import DFlashModel
from exllamav3.architecture.dflash2 import DFlash2Model
from exllamav3.generator.generator import Generator
from exllamav3.modules import TransformerBlock
from exllamav3.modules.arch_specific.dflash import DFlashInputLayer
from exllamav3.modules.arch_specific.dflash2 import (
    DFlash2DynConv,
    DFlash2Selector,
    _DFlash2Norm,
    _grouped_dynamic_convolve,
)


def test_architecture_is_registered():
    assert "DFlash2DraftModel" in ARCHITECTURES


def test_prepare_inputs_sets_anchor_and_bilateral_span():
    model = object.__new__(DFlash2Model)
    model.config = SimpleNamespace(block_size = 8, draft_causal = False)
    input_ids = torch.tensor([[42]])
    params = {}
    prepared = object()

    with patch.object(DFlashModel, "prepare_inputs", return_value = prepared):
        actual = DFlash2Model.prepare_inputs(model, input_ids, params)

    assert actual is prepared
    assert params["dflash2_anchor_ids"] is input_ids
    assert params["non_causal_spans"] == [(0, 8, True)]
    assert params["causal"] is False


def test_prepare_inputs_respects_causal_checkpoint():
    model = object.__new__(DFlash2Model)
    model.config = SimpleNamespace(block_size = 8, draft_causal = True)
    input_ids = torch.tensor([[42]])
    params = {"non_causal_spans": [(0, 3, True)]}

    with patch.object(DFlashModel, "prepare_inputs", return_value = input_ids):
        DFlash2Model.prepare_inputs(model, input_ids, params)

    assert "non_causal_spans" not in params
    assert params["causal"] is True


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


def test_target_taps_are_exported_in_checkpoint_order():
    def block(layer_idx):
        module = object.__new__(TransformerBlock)
        module.layer_idx = layer_idx
        module.attn = None
        module.mlp = None
        module.attn_hc = None
        module.layer_scalar_f = None
        module.out_dtype = None
        return module

    params = {
        "export_state_layers": {2, 5},
        "export_state_order": {5: 0, 2: 1},
    }
    state_2 = torch.tensor([[[2.0]]])
    state_5 = torch.tensor([[[5.0]]])
    block(2).forward(state_2, params)
    block(5).forward(state_5, params)

    assert [state.item() for state in params["export_states"]] == [5.0, 2.0]


def test_generator_rejects_oversized_draft_window():
    model = SimpleNamespace(
        config = SimpleNamespace(vocab_size = 100),
        caps = {},
    )
    cache = SimpleNamespace(max_num_tokens = 256)
    draft_model = SimpleNamespace(
        caps = {"default_draft_size": 7, "max_draft_size": 7},
    )
    draft_cache = SimpleNamespace(max_num_tokens = 256)
    page_table = SimpleNamespace(max_pages = 1)

    with patch("exllamav3.generator.generator.PageTable", return_value = page_table):
        try:
            Generator(
                model, cache, None,
                draft_model = draft_model,
                draft_cache = draft_cache,
                num_draft_tokens = 8,
            )
        except ValueError as exc:
            assert "at most 7 draft tokens" in str(exc)
        else:
            raise AssertionError("Generator accepted an oversized DFlash2 draft window")


def test_generator_rejects_reduced_fixed_draft_window():
    model = SimpleNamespace(
        config = SimpleNamespace(vocab_size = 100),
        caps = {},
    )
    cache = SimpleNamespace(max_num_tokens = 256)
    draft_model = SimpleNamespace(
        caps = {
            "default_draft_size": 7,
            "max_draft_size": 7,
            "required_draft_size": 7,
        },
    )
    draft_cache = SimpleNamespace(max_num_tokens = 256)
    page_table = SimpleNamespace(max_pages = 1)

    with patch("exllamav3.generator.generator.PageTable", return_value = page_table):
        try:
            Generator(
                model, cache, None,
                draft_model = draft_model,
                draft_cache = draft_cache,
                num_draft_tokens = 4,
            )
        except ValueError as exc:
            assert "requires exactly 7 draft tokens" in str(exc)
        else:
            raise AssertionError("Generator accepted a reduced fixed DFlash2 draft window")


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


def test_grouped_dynamic_convolve_matches_reference():
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 8)
    dynamic = torch.randn(2, 5, 3, 2)
    base = torch.randn(3, 8)

    actual = _grouped_dynamic_convolve(hidden, dynamic, base, group_size = 4)
    expected = torch.zeros_like(hidden)
    for batch in range(hidden.shape[0]):
        for position in range(hidden.shape[1]):
            for channel in range(hidden.shape[2]):
                group = channel // 4
                for offset in range(base.shape[0]):
                    if position >= offset:
                        weight = base[offset, channel] + dynamic[batch, position, offset, group]
                        expected[batch, position, channel] += weight * hidden[batch, position - offset, channel]

    torch.testing.assert_close(actual, expected)


def test_cacheless_non_causal_span_keeps_prefix_and_right_window():
    q = torch.randn(1, 4, 2, 16)
    k = torch.randn(1, 4, 2, 16)
    v = torch.randn(1, 4, 2, 16)
    args = AttnArgs(
        bsz = 1,
        q_len = 4,
        num_q_heads = 2,
        dim = 16,
        kv_len = 4,
        num_kv_heads = 2,
        q = q,
        k = k,
        v = v,
        k_cache = None,
        v_cache = None,
        causal = False,
        sm_scale = 0.25,
        cu_seqlens = None,
        max_seqlen = None,
        window_size = 2,
        softcap = 0.0,
        block_table = None,
        cache_seqlens = None,
        non_causal_spans = [(0, 2, True), (2, 4, True)],
    )

    spans = get_non_causal_span_arglist(args)

    assert spans[0]["q"].shape[1] == 2
    assert spans[0]["k"].shape[1] == 2
    assert spans[1]["q"].shape[1] == 2
    assert spans[1]["k"].shape[1] == 4
    assert spans[1]["cache_seqlens"] is None
    assert spans[1]["causal"] is False
    assert spans[1]["window_size"] == (2, 1)


def test_standard_causal_sdpa_uses_optimized_maskless_path():
    q = torch.randn(1, 4, 2, 16)
    with patch("exllamav3.modules.attention_fn.torch.F.scaled_dot_product_attention") as sdpa:
        sdpa.return_value = q.transpose(1, 2)
        _torch_sdpa_nocache(q, q, q, True, 0.25, (-1, -1))

    assert sdpa.call_args.kwargs["attn_mask"] is None
    assert sdpa.call_args.kwargs["is_causal"] is True


def test_cached_noncausal_span_uses_inclusive_window_boundary():
    q = torch.randn(1, 2, 1, 16)
    args = AttnArgs(
        bsz = 1,
        q_len = 2,
        num_q_heads = 1,
        dim = 16,
        kv_len = 2,
        num_kv_heads = 1,
        q = q,
        k = q,
        v = q,
        k_cache = torch.empty(3, 4, 1, 16),
        v_cache = torch.empty(3, 4, 1, 16),
        causal = False,
        sm_scale = 0.25,
        cu_seqlens = None,
        max_seqlen = None,
        window_size = 3,  # Configured four-token window, converted to inclusive distance.
        softcap = 0.0,
        block_table = torch.tensor([[2, 0, 1]], dtype = torch.int32),
        cache_seqlens = torch.tensor([7], dtype = torch.int32),
        non_causal_spans = [(0, 2, True)],
    )

    [span] = get_non_causal_span_arglist(args)

    assert span["cache_seqlens"].tolist() == [7]
    assert span["window_size"] == (3, 1)


def test_triton_cached_bilateral_window_matches_dense_reference():
    if not torch.cuda.is_available() or not has_triton:
        return

    device = torch.device("cuda")
    page_size = 4
    block_table = torch.tensor([[2, 0, 3]], dtype = torch.int32, device = device)
    cache_seqlens = torch.tensor([7], dtype = torch.int32, device = device)
    k_cache = torch.zeros(4, page_size, 1, 16, dtype = torch.float16, device = device)
    v_cache = torch.zeros_like(k_cache)

    # Populate seven cached logical positions through a deliberately noncontiguous page table.
    for pos in range(7):
        page = block_table[0, pos // page_size]
        v_cache[page, pos % page_size] = pos

    q = torch.zeros(1, 2, 1, 16, dtype = torch.float16, device = device)
    k = torch.zeros_like(q)
    v = torch.empty_like(q)
    v[:, 0] = 7
    v[:, 1] = 8

    actual = paged_attn_triton(
        q, k, v, k_cache, v_cache, block_table, cache_seqlens,
        causal = False, softmax_scale = 0.25, window_size = (3, 1),
    )

    # Query positions 7 and 8 may attend distances <= 3 plus the future synthetic row.
    expected = torch.tensor([6.0, 6.5], dtype = torch.float16, device = device)
    torch.testing.assert_close(actual[0, :, 0, 0], expected, rtol = 1e-3, atol = 1e-3)


def test_cacheless_draft_block_is_bilateral():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 2, 16)
    k = torch.randn(1, 4, 2, 16)
    v = torch.randn(1, 4, 2, 16)
    args = AttnArgs(
        bsz = 1,
        q_len = 4,
        num_q_heads = 2,
        dim = 16,
        kv_len = 4,
        num_kv_heads = 2,
        q = q,
        k = k,
        v = v,
        k_cache = None,
        v_cache = None,
        causal = False,
        sm_scale = 0.25,
        cu_seqlens = None,
        max_seqlen = None,
        window_size = 2,
        softcap = 0.0,
        block_table = None,
        cache_seqlens = None,
        non_causal_spans = [(0, 4, True)],
    )

    actual = fn_torch_sdpa_fallback_nocache(args)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal = False, scale = 0.25,
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected)


def test_candidate_logit_scale_and_softcap_are_applied():
    class LMHead:
        def prepare_for_device(self, state, params):
            return state

        def forward(self, state, params):
            return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    class Selector:
        device = torch.device("cpu")

        def walk(self, hidden, logits, anchor):
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

    raw = torch.tensor([[3.0, 4.0]]) * 2.0
    expected = torch.tanh(raw / 3.0) * 3.0
    torch.testing.assert_close(selector.logits[0], expected)


def test_norm_preserves_large_bfloat16_residuals():
    norm = object.__new__(_DFlash2Norm)
    norm.weight = torch.ones(4, dtype = torch.bfloat16)
    norm.eps = 1e-6
    x = torch.tensor([[[1.5e5, -1.5e5, 7.5e4, -7.5e4]]], dtype = torch.bfloat16)

    actual = norm.forward(x, {})

    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    expected = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim = True) + norm.eps)
    torch.testing.assert_close(actual.float(), expected, rtol = 5e-3, atol = 5e-3)


class _Projection:
    def forward(self, hidden, params):
        return torch.ones((*hidden.shape[:-1], 1), device = hidden.device)


def test_selector_chains_candidates_from_anchor():
    selector = object.__new__(DFlash2Selector)
    selector.top_k = 2
    selector.hidden_proj = _Projection()
    selector.pred_codebook = torch.tensor([[1.0], [0.0], [-1.0], [0.0]])
    selector.succ_codebook = torch.tensor([[0.0], [0.0], [2.0], [2.0]])

    hidden = torch.zeros(1, 2, 4)
    logits = torch.tensor([[[0.0, 3.0, 2.0, -1.0], [0.0, 3.0, -1.0, 2.0]]])
    path = selector.walk(hidden, logits, torch.tensor([0]))

    assert path.tolist() == [[2, 1]]


if __name__ == "__main__":
    test_architecture_is_registered()
    test_prepare_inputs_sets_anchor_and_bilateral_span()
    test_prepare_inputs_respects_causal_checkpoint()
    test_input_embedding_scale_is_applied()
    test_tp_target_is_rejected_during_attach()
    test_target_taps_are_exported_in_checkpoint_order()
    test_generator_rejects_oversized_draft_window()
    test_generator_rejects_reduced_fixed_draft_window()
    test_exact_projection_widths_are_preserved()
    test_grouped_dynamic_convolve_matches_reference()
    test_cacheless_non_causal_span_keeps_prefix_and_right_window()
    test_standard_causal_sdpa_uses_optimized_maskless_path()
    test_cached_noncausal_span_uses_inclusive_window_boundary()
    test_triton_cached_bilateral_window_matches_dense_reference()
    test_cacheless_draft_block_is_bilateral()
    test_candidate_logit_scale_and_softcap_are_applied()
    test_norm_preserves_large_bfloat16_residuals()
    test_selector_chains_candidates_from_anchor()
    print("DFlash2 tests passed")
