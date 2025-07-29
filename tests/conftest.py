"""Pytest configuration and fixtures for PWMK test suite."""
import pytest
import torch
import numpy as np
from typing import Dict, Any, Generator
from unittest.mock import Mock

# Set random seeds for reproducible tests
torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_observation() -> torch.Tensor:
    """Sample observation tensor for testing."""
    return torch.randn(4, 64)  # batch_size=4, obs_dim=64


@pytest.fixture
def sample_actions() -> torch.Tensor:
    """Sample action tensor for testing."""
    return torch.randint(0, 4, (4,))  # batch_size=4, 4 possible actions


@pytest.fixture
def mock_environment() -> Mock:
    """Mock multi-agent environment for testing."""
    env = Mock()
    env.num_agents = 3
    env.observation_space.shape = (64,)
    env.action_space.n = 4
    env.reset.return_value = [torch.randn(64) for _ in range(3)]
    env.step.return_value = (
        [torch.randn(64) for _ in range(3)],  # observations
        [0.1, 0.2, 0.3],  # rewards
        False,  # done
        {}  # info
    )
    return env


@pytest.fixture
def sample_belief_facts() -> Dict[str, Any]:
    """Sample belief facts for testing belief reasoning."""
    return {
        "agent_0": [
            "has(agent_1, key)",
            "at(treasure, room_3)",
            "believes(agent_1, location(agent_0, room_1))"
        ],
        "agent_1": [
            "has(agent_0, map)",
            "at(agent_1, room_2)",
            "believes(agent_0, has(agent_1, key))"
        ]
    }


@pytest.fixture
def world_model_config() -> Dict[str, Any]:
    """Standard configuration for world model testing."""
    return {
        "obs_dim": 64,
        "action_dim": 4,
        "hidden_dim": 128,
        "num_agents": 3,
        "num_layers": 2,
        "learning_rate": 1e-4,
        "batch_size": 32
    }


@pytest.fixture
def belief_store_config() -> Dict[str, Any]:
    """Configuration for belief store testing."""
    return {
        "backend": "memory",  # Use in-memory backend for tests
        "max_depth": 3,
        "predicates": ["has", "at", "believes", "knows"]
    }


@pytest.fixture(autouse=True)
def reset_torch_state() -> Generator[None, None, None]:
    """Reset PyTorch state between tests."""
    torch.manual_seed(42)
    yield
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Markers for different test categories
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]