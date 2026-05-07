"""
Pytest configuration and shared fixtures.
"""
import pytest
import torch


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def _cuda_actually_works():
    """`torch.cuda.is_available()` only checks driver presence, not whether
    the driver is recent enough for the installed PyTorch wheel. Probe
    with a real op so we skip GPU tests cleanly on hosts whose driver is
    too old for the wheel (the typical container/PyPI mismatch).
    """
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() == False"
    try:
        torch.zeros(1, device="cuda") + 1
        return True, ""
    except Exception as exc:
        return False, f"CUDA op failed: {type(exc).__name__}: {exc}"


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    cuda_ok, cuda_reason = _cuda_actually_works()
    if not cuda_ok:
        skip_gpu = pytest.mark.skip(reason=f"GPU unavailable ({cuda_reason})")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)