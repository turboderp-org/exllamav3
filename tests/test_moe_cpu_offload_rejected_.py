"""The inference CPU expert offload (EXL3_MOE_CPU_OFFLOAD / EXL3_MOE_CPU_SPLIT)
must be rejected up front by the training support check, with a message that
names it -- not deep inside DiffLinear as "native Linear must be loaded"."""
import pytest
from exllamav3.training.backbone import _assert_moe_supported


class _StubMoE:
    router_type = "std"
    routing_gate = object()
    num_experts = 512
    num_local_experts = 512
    gates = ups = downs = [object()] * 512
    n_group = topk_group = None
    routed_scaling_factor = None
    e_score_correction_bias = None
    shared_experts = None
    shared_gate = None
    cpu_offload = False
    cpu_split_first = None


def test_resident_layer_passes():
    _assert_moe_supported("layers.0", _StubMoE())


def test_whole_layer_offload_rejected():
    m = _StubMoE()
    m.cpu_offload = True
    with pytest.raises(AssertionError, match="EXL3_MOE_CPU_OFFLOAD"):
        _assert_moe_supported("layers.0", m)


def test_split_offload_rejected():
    m = _StubMoE()
    m.cpu_split_first = 117
    m.num_local_experts = 117
    with pytest.raises(AssertionError, match="EXL3_MOE_CPU_SPLIT"):
        _assert_moe_supported("layers.0", m)


def test_trivial_grouping_accepted():
    # GLM-4.5/4.7 (Glm4MoeForCausalLM) builds its dots router with
    # n_group = topk_group = 1: one group holding every expert, always
    # selected, which the inference side runs through the ungrouped kernel.
    m = _StubMoE()
    m.router_type = "dots"
    m.routed_scaling_factor = 2.5
    m.n_group = m.topk_group = 1
    _assert_moe_supported("layers.3", m)


def test_real_grouping_rejected():
    m = _StubMoE()
    m.router_type = "dots"
    m.routed_scaling_factor = 2.5
    m.n_group, m.topk_group = 8, 4
    with pytest.raises(AssertionError, match="n_group=8"):
        _assert_moe_supported("layers.3", m)
