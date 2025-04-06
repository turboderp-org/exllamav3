import torch

def assert_close_mr(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        mismatch_ratio: float = 0.0,
        check_device: bool = True,
        check_dtype: bool = True,
        msg: str = None,
):

    # 1) Check shape
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: {actual.shape} vs {expected.shape}"
        )

    # 2) (Optional) Check device
    if check_device and (actual.device != expected.device):
        raise AssertionError(
            f"Device mismatch: {actual.device} vs {expected.device}"
        )

    # 3) (Optional) Check dtype
    if check_dtype and (actual.dtype != expected.dtype):
        raise AssertionError(
            f"Dtype mismatch: {actual.dtype} vs {expected.dtype}"
        )

    # 4) Compare element-wise closeness
    #    close_mask[i] = True if actual[i] ~ expected[i] within rtol/atol
    close_mask = torch.isclose(actual, expected, rtol = rtol, atol = atol)

    # 5) Compute fraction of elements that are out of tolerance
    total_elements = close_mask.numel()
    mismatch_count = total_elements - close_mask.sum().item()
    fraction_mismatched = mismatch_count / total_elements

    if fraction_mismatched > mismatch_ratio:
        default_msg = (
            f"Too many values are out of tolerance:\n"
            f"  Mismatch ratio = {fraction_mismatched:.6f} "
            f"(allowed <= {mismatch_ratio:.6f})\n"
            f"  Mismatched elements = {mismatch_count} / {total_elements}\n"
            f"  rtol={rtol}, atol={atol}"
        )
        error_msg = f"{msg}\n{default_msg}" if msg else default_msg
        raise AssertionError(error_msg)