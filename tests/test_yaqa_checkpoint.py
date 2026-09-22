"""Verify checkpointed frozen blocks preserve gradients and release streamed weights."""

import copy

import torch
from accelerate import cpu_offload
from torch import nn

from util.yaqa_hessians import checkpoint_layers


def test_checkpoint_preserves_keyword_gradients_without_retaining_weights() -> None:
    """Keep keyword-call gradients exact while retaining block inputs instead of weight copies."""
    torch.manual_seed(7)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(64, 64, bias=False) for _ in range(4)])
    model.requires_grad_(False)
    wrapped = copy.deepcopy(model)
    checkpoint_layers(wrapped)
    inputs = torch.randn(3, 64)
    results = []
    retained_bytes = []
    for candidate in (model, wrapped):
        x = inputs.clone().requires_grad_()
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor, saved=saved: (
                saved.append(tensor.numel() * tensor.element_size()),
                tensor,
            )[1],
            lambda tensor: tensor,
        ):
            y = x
            for layer in candidate.get_submodule("layers").children():
                y = layer(input=y)
        y.square().sum().backward()
        results.append((y.detach(), x.grad))
        retained_bytes.append(sum(saved))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    # Four block inputs, not four 64x64 streamed weight matrices.
    assert retained_bytes[1] <= 4 * inputs.numel() * inputs.element_size()
    assert retained_bytes[0] >= 4 * 64 * 64 * inputs.element_size()


def test_checkpoint_releases_accelerate_weights_after_recomputation() -> None:
    """Run Accelerate cleanup after backward replay, returning execution weights to meta storage."""
    torch.manual_seed(11)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(3)])
    model.requires_grad_(False)
    inputs = torch.randn(2, 8, requires_grad=True)
    expected = inputs
    for layer in model.get_submodule("layers").children():
        expected = layer(expected)
    expected.sum().backward()
    assert inputs.grad is not None
    expected_grad = inputs.grad.clone()
    inputs.grad = None
    cpu_offload(model, execution_device=torch.device("cpu"))
    checkpoint_layers(model)
    actual = inputs
    for layer in model.get_submodule("layers").children():
        actual = layer(actual)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, expected_grad, rtol=0, atol=0)
    # Accelerate must release the execution copies after recomputation, not just after forward.
    assert all(weight.is_meta for weight in model.parameters())
