import pytest
import torch

# Tests written for a particular GPU of a multi-GPU box (device = "cuda:N" at module level) take
# EXL3_TEST_DEVICE in its place, e.g. EXL3_TEST_DEVICE=cuda:0 on a single-GPU machine


@pytest.fixture(autouse = True)
def _restore_current_cuda_device():
    # Several test modules pin tensors to "cuda:0" but launch Triton kernels on the *current* device,
    # while others call torch.cuda.set_device() for their own device and never restore it. Reset the
    # current device after every test so the suite's result doesn't depend on collection order
    yield
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
