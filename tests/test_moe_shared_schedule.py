"""Shared work may precede CPU collect; nonlinear norms and TP collectives may not."""
from types import SimpleNamespace
import pytest
import torch
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP, MAX_BSZN


@pytest.mark.parametrize('rows,pending', [(1, True), (1, False), (MAX_BSZN + 1, True)])
@pytest.mark.parametrize('tp', [False, True])
@pytest.mark.parametrize('offload', [False, True])
def test_shared_schedule(rows, pending, tp, offload):
    # offload: whole-layer CPU offload (cpu_offload_issue) instead of the per-layer split
    events = []
    x = torch.arange(rows * 4, dtype=torch.float32).reshape(1, rows, 4) / 100

    def module(name, fn):
        def forward(x, params):
            events.append(name)
            return fn(x)
        return SimpleNamespace(forward=forward)

    def submit(*args):
        events.append('submit')
        return None, object() if pending else None

    def offload_issue(shape, *args):
        events.append('submit')
        return (None, object()) if pending else (torch.zeros(shape), None)

    def combine(result, cpu_partial, cpu_pending, shape):
        events.append('collect')
        return (torch.zeros(shape) if result is None else result) + 2

    def collect(*args):
        events.append('tp')

    layer = SimpleNamespace(
        alt_residual_channel=False, hidden_size=4, expert_size=4, bc=None,
        router_pre_norm=None, routed_pre_norm=None, latent_in=None, latent_out=None,
        routing_gate=object(), routing_cfg=None,
        routing_fn=lambda *args: (torch.zeros(rows, 1, dtype=torch.long), torch.ones(rows, 1)),
        routing_device=None, cpu_split_first=None if offload else 0, cpu_split_submit=submit,
        cpu_offload_issue=offload_issue,
        cpu_split_combine=combine, cpu_offload=offload, intermediate_size=0,
        num_local_experts=0, tp_reduce=tp, tp_collect=collect, shared_gate=None,
        shared_experts=module('shared', lambda a: a * 3),
        shared_experts_post_norm=module('shared_norm', lambda a: a.square()),
        routed_post_norm=module('routed_norm', lambda a: a.square()),
    )
    actual = BlockSparseMLP.forward(layer, x, {'backend': None})
    torch.testing.assert_close(actual, 4 + (x * 3).square())
    assert events.count('shared') == 1
    assert (events.index('shared') < events.index('collect')) == (pending and rows <= MAX_BSZN)
    assert events.index('collect') < events.index('routed_norm') < events.index('shared_norm')
    if tp:
        assert [e for e in events if e != 'shared'] == [
            'submit', 'collect', 'tp', 'routed_norm', 'tp', 'shared_norm']


def test_fused_shared_expert_is_not_run_twice():
    events = []
    output = torch.zeros(1, 4)

    def fused(*args):
        events.append('fused')
        output.fill_(7)  # GPU routed + shared contribution

    def collect(result, *args):
        events.append('collect')
        return result + 2  # CPU routed contribution

    def unexpected(*args):
        pytest.fail('shared expert was already evaluated by the fused path')

    layer = SimpleNamespace(
        alt_residual_channel=False, hidden_size=4, expert_size=4,
        bc=SimpleNamespace(run_bszN=fused), bc_sh_exp=True,
        experts_cfg=SimpleNamespace(out_bszn=output), f_threshold=128,
        is_quantized=True, config=SimpleNamespace(infer_params=SimpleNamespace(no_reconstruct=False)),
        support_quant_paths=True, router_pre_norm=None, routed_pre_norm=None,
        latent_in=None, latent_out=None, routing_gate=object(), routing_cfg=None,
        routing_fn=lambda *args: (torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1)),
        routing_device=None, cpu_split_first=0,
        cpu_split_submit=lambda *args: (None, object()), cpu_split_combine=collect,
        cpu_offload=False, intermediate_size=4, num_local_experts=1, tp_reduce=False,
        shared_experts=SimpleNamespace(forward=unexpected), shared_experts_post_norm=None,
        routed_post_norm=None, shared_gate=None,
    )
    result = BlockSparseMLP.forward(layer, torch.ones(1, 1, 4), {})
    torch.testing.assert_close(result, torch.full((1, 1, 4), 9.))
    assert events == ['fused', 'collect']
