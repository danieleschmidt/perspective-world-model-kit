"""Global pytest configuration for PWMK."""
import pytest
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ["PWMK_TEST_MODE"] = "true"
os.environ["PWMK_LOG_LEVEL"] = "WARNING"  # Reduce log noise in tests


def pytest_configure(config):
    """Called after command line options have been parsed."""
    # Register custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "unity: marks tests that require Unity")
    config.addinivalue_line("markers", "prolog: marks tests that require Prolog backend")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on file location
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance/" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "benchmarks/" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add markers based on test name patterns
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        if "unity" in item.name:
            item.add_marker(pytest.mark.unity)
        
        if "prolog" in item.name or "belief" in item.name:
            item.add_marker(pytest.mark.prolog)


def pytest_runtest_setup(item):
    """Called to perform the setup phase for a test item."""
    # Skip GPU tests if CUDA is not available
    if "gpu" in [marker.name for marker in item.iter_markers()]:
        pytest.importorskip("torch")
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    
    # Skip Unity tests if Unity environment variable is not set
    if "unity" in [marker.name for marker in item.iter_markers()]:
        if not os.environ.get("UNITY_ENV_AVAILABLE"):
            pytest.skip("Unity environment not available")
    
    # Skip Prolog tests if backend is not available
    if "prolog" in [marker.name for marker in item.iter_markers()]:
        try:
            pytest.importorskip("pyswip")
        except ImportError:
            pytest.skip("Prolog backend not available")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup global test environment."""
    # Ensure consistent random seeds
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass
    
    yield
    
    # Cleanup after all tests
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass