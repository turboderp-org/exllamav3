"""
Shared pytest configuration and fixtures for TQ3 tests.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action = "store",
        default = "cuda:0",
        help = "CUDA device to run tests on (default: cuda:0)"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks (deselect with '-m \"not benchmark\"')"
    )


@pytest.fixture
def cuda_device(request):
    """Return the CUDA device string from --device flag."""
    import torch
    device = request.config.getoption("--device")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device


@pytest.fixture
def ext():
    """Return the compiled exllamav3_ext module, or skip if not compiled."""
    try:
        from exllamav3.ext import exllamav3_ext as _ext
        return _ext
    except ImportError:
        pytest.skip("exllamav3_ext not compiled")
