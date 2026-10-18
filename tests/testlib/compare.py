"""Tolerance helpers and logit metrics shared by kernel, module and model tests."""

import torch


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """max |a - e| / max |e|, computed in fp32"""
    a, e = actual.float(), expected.float()
    return (a - e).abs().max().item() / max(e.abs().max().item(), 1e-6)


def assert_rel_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, msg: str = ""):
    assert actual.shape == expected.shape, f"{msg} shape {tuple(actual.shape)} vs {tuple(expected.shape)}"
    assert torch.isfinite(actual).all(), f"{msg} non-finite values in output"
    err = rel_err(actual, expected)
    assert err < tol, f"{msg} relative error {err:.3e} >= {tol:.1e}"


def assert_close_mr(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    mismatch_ratio: float = 0.0,
    check_device: bool = True,
    check_dtype: bool = True,
    msg: str | None = None,
):
    """torch.isclose with an allowed fraction of out-of-tolerance elements"""
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
    if check_device and actual.device != expected.device:
        raise AssertionError(f"Device mismatch: {actual.device} vs {expected.device}")
    if check_dtype and actual.dtype != expected.dtype:
        raise AssertionError(f"Dtype mismatch: {actual.dtype} vs {expected.dtype}")
    close = torch.isclose(actual, expected, rtol = rtol, atol = atol)
    total = close.numel()
    mismatched = total - close.sum().item()
    if mismatched / total > mismatch_ratio:
        detail = (
            f"Too many values are out of tolerance:\n"
            f"  Mismatch ratio = {mismatched / total:.6f} (allowed <= {mismatch_ratio:.6f})\n"
            f"  Mismatched elements = {mismatched} / {total}\n"
            f"  rtol={rtol}, atol={atol}"
        )
        raise AssertionError(f"{msg}\n{detail}" if msg else detail)


def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Per-row KL(P || Q) of two logit tensors (..., vocab), fp32"""
    lp = torch.log_softmax(logits_p.float(), dim = -1)
    lq = torch.log_softmax(logits_q.float(), dim = -1)
    return (lp.exp() * (lp - lq)).sum(dim = -1)


def top1_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    return (logits_a.argmax(dim = -1) == logits_b.argmax(dim = -1)).float().mean().item()


def confident_top1_agreement(ref_logits: torch.Tensor, logits: torch.Tensor, margin: float = 1.0) -> tuple[float, int]:
    """(top-1 agreement, count) over the positions where the reference's top logit leads its runner-up by more than
    margin. Raw top-1 agreement over a few dozen positions mostly counts the text's near-ties, which two correct
    paths are free to resolve differently; a confident position can only flip through an actual error"""
    top2 = ref_logits.float().topk(2, dim = -1).values
    confident = (top2[..., 0] - top2[..., 1]) > margin
    n = int(confident.sum().item())
    if n == 0:
        return 1.0, 0
    agree = ref_logits.argmax(dim = -1)[confident] == logits.argmax(dim = -1)[confident]
    return agree.float().mean().item(), n
