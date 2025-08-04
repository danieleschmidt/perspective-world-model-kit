"""Pytest configuration and fixtures for PWMK test suite."""
import pytest
import torch
import numpy as np
from typing import Dict, Any, Generator
from unittest.mock import Mock

# Set random seeds for reproducible tests
torch.manual_seed(42)
np.random.seed(42)

# Import PWMK components for real fixtures
try:
    from pwmk import PerspectiveWorldModel, BeliefStore, EpistemicPlanner, ToMAgent
    from pwmk.envs import SimpleGridWorld
    from pwmk.planning.epistemic import Goal
    PWMK_AVAILABLE = True
except ImportError:
    PWMK_AVAILABLE = False


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


# Real PWMK fixtures when available
if PWMK_AVAILABLE:
    @pytest.fixture
    def simple_env() -> SimpleGridWorld:
        """Simple grid world environment for testing."""
        return SimpleGridWorld(grid_size=6, num_agents=2, view_radius=2)

    @pytest.fixture 
    def world_model() -> PerspectiveWorldModel:
        """Basic world model for testing."""
        return PerspectiveWorldModel(
            obs_dim=32,
            action_dim=4,
            hidden_dim=64,
            num_agents=2
        )

    @pytest.fixture
    def belief_store() -> BeliefStore:
        """Basic belief store for testing."""
        store = BeliefStore()
        # Add some test beliefs
        store.add_belief("agent_0", "has(agent_1, key)")
        store.add_belief("agent_0", "at(treasure, room_3)")
        store.add_belief("agent_1", "at(agent_0, room_1)")
        return store

    @pytest.fixture
    def epistemic_planner(world_model, belief_store) -> EpistemicPlanner:
        """Epistemic planner for testing."""
        return EpistemicPlanner(
            world_model=world_model,
            belief_store=belief_store,
            search_depth=5
        )

    @pytest.fixture
    def tom_agent(world_model) -> ToMAgent:
        """Theory of Mind agent for testing."""
        return ToMAgent(
            agent_id="agent_0", 
            world_model=world_model,
            tom_depth=2
        )

    @pytest.fixture
    def sample_goal() -> Goal:
        """Sample planning goal."""
        return Goal(
            achievement="has(agent_0, treasure)",
            epistemic=["believes(agent_1, at(agent_0, room_2))"]
        )


# Markers for different test categories
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]