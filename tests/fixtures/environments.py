"""Test fixtures for environments."""
import pytest
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, MagicMock


@pytest.fixture
def basic_env_config() -> Dict[str, Any]:
    """Basic environment configuration for testing."""
    return {
        "num_agents": 3,
        "grid_size": 10,
        "max_steps": 100,
        "partial_observability": True,
        "view_radius": 3,
        "seed": 42
    }


@pytest.fixture
def mock_gym_env() -> Mock:
    """Mock Gym-style environment."""
    env = Mock()
    env.observation_space = Mock()
    env.observation_space.shape = (64,)
    env.observation_space.n = None
    env.action_space = Mock()
    env.action_space.n = 4
    env.action_space.shape = None
    
    # Mock reset method
    def mock_reset():
        return torch.randn(64).numpy()
    env.reset = Mock(side_effect=mock_reset)
    
    # Mock step method
    def mock_step(action):
        obs = torch.randn(64).numpy()
        reward = np.random.random()
        done = np.random.random() < 0.1  # 10% chance of episode end
        info = {"step": np.random.randint(0, 100)}
        return obs, reward, done, info
    env.step = Mock(side_effect=mock_step)
    
    return env


@pytest.fixture
def mock_multi_agent_env() -> Mock:
    """Mock multi-agent environment."""
    env = Mock()
    env.num_agents = 3
    env.observation_spaces = [Mock() for _ in range(3)]
    env.action_spaces = [Mock() for _ in range(3)]
    
    for i in range(3):
        env.observation_spaces[i].shape = (64,)
        env.action_spaces[i].n = 4
    
    # Mock reset method
    def mock_reset():
        return [torch.randn(64).numpy() for _ in range(3)]
    env.reset = Mock(side_effect=mock_reset)
    
    # Mock step method
    def mock_step(actions):
        observations = [torch.randn(64).numpy() for _ in range(3)]
        rewards = [np.random.random() for _ in range(3)]
        dones = [False] * 3
        infos = [{"agent_id": i} for i in range(3)]
        return observations, rewards, dones, infos
    env.step = Mock(side_effect=mock_step)
    
    return env


@pytest.fixture
def unity_env_mock() -> Mock:
    """Mock Unity ML-Agents environment."""
    env = Mock()
    env.behavior_specs = Mock()
    env.behavior_names = ["TestBehavior"]
    
    # Mock behavior spec
    behavior_spec = Mock()
    behavior_spec.observation_specs = [Mock()]
    behavior_spec.observation_specs[0].shape = (64,)
    behavior_spec.action_spec = Mock()
    behavior_spec.action_spec.discrete_size = 4
    behavior_spec.action_spec.continuous_size = 0
    
    env.behavior_specs["TestBehavior"] = behavior_spec
    
    # Mock decision steps and terminal steps
    def mock_get_steps(behavior_name):
        decision_steps = Mock()
        decision_steps.obs = [torch.randn(3, 64).numpy()]  # 3 agents
        decision_steps.reward = np.array([0.1, 0.2, 0.3])
        decision_steps.agent_id = np.array([0, 1, 2])
        decision_steps.action_mask = None
        
        terminal_steps = Mock()
        terminal_steps.obs = [torch.randn(0, 64).numpy()]  # No terminal agents
        terminal_steps.reward = np.array([])
        terminal_steps.agent_id = np.array([])
        terminal_steps.interrupted = np.array([])
        
        return decision_steps, terminal_steps
    
    env.get_steps = Mock(side_effect=mock_get_steps)
    env.set_actions = Mock()
    env.step = Mock()
    env.reset = Mock()
    env.close = Mock()
    
    return env


@pytest.fixture
def sample_trajectory_data() -> Dict[str, torch.Tensor]:
    """Sample trajectory data for testing."""
    batch_size = 32
    seq_len = 10
    obs_dim = 64
    action_dim = 4
    num_agents = 3
    
    return {
        "observations": torch.randn(batch_size, seq_len, num_agents, obs_dim),
        "actions": torch.randint(0, action_dim, (batch_size, seq_len, num_agents)),
        "rewards": torch.randn(batch_size, seq_len, num_agents),
        "dones": torch.zeros(batch_size, seq_len, num_agents, dtype=torch.bool),
        "next_observations": torch.randn(batch_size, seq_len, num_agents, obs_dim),
        "agent_ids": torch.arange(num_agents).expand(batch_size, seq_len, -1),
        "timesteps": torch.arange(seq_len).expand(batch_size, num_agents, -1).transpose(1, 2)
    }


@pytest.fixture
def environment_factory():
    """Factory for creating different types of test environments."""
    def create_env(env_type: str, **kwargs) -> Mock:
        if env_type == "gym":
            return mock_gym_env()
        elif env_type == "multi_agent":
            return mock_multi_agent_env()
        elif env_type == "unity":
            return unity_env_mock()
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    return create_env


@pytest.fixture
def partially_observable_env() -> Mock:
    """Mock partially observable environment."""
    env = Mock()
    env.num_agents = 3
    env.full_state_dim = 128
    env.partial_obs_dim = 64
    env.view_radius = 3
    
    # Mock partial observations
    def mock_get_partial_obs(agent_id: int, full_state: np.ndarray) -> np.ndarray:
        # Simulate partial observability by masking parts of the state
        mask = np.random.random(env.full_state_dim) > 0.5
        partial_obs = full_state.copy()
        partial_obs[mask] = 0
        return partial_obs[:env.partial_obs_dim]
    
    env.get_partial_observation = Mock(side_effect=mock_get_partial_obs)
    
    # Mock full state
    env.get_full_state = Mock(return_value=torch.randn(env.full_state_dim).numpy())
    
    return env


@pytest.fixture
def belief_ground_truth() -> Dict[str, List[str]]:
    """Ground truth beliefs for testing belief tracking accuracy."""
    return {
        "agent_0": [
            "has(agent_1, key)",
            "at(agent_0, room_1)",
            "at(agent_1, room_2)",
            "at(treasure, room_3)",
            "knows(agent_1, at(treasure, room_3))",
            "believes(agent_1, has(agent_0, map))"
        ],
        "agent_1": [
            "has(agent_0, map)",
            "at(agent_0, room_1)",
            "at(agent_1, room_2)",
            "knows(agent_0, has(agent_1, key))",
            "believes(agent_0, at(treasure, room_3))"
        ],
        "agent_2": [
            "at(agent_2, room_3)",
            "sees(agent_2, treasure)",
            "knows(agent_2, at(treasure, room_3))",
            "believes(agent_0, has(agent_1, key))",
            "believes(agent_1, has(agent_0, map))"
        ]
    }


@pytest.fixture
def communication_channels() -> Dict[str, Any]:
    """Mock communication channels between agents."""
    channels = {}
    
    for i in range(3):
        for j in range(3):
            if i != j:
                channel_name = f"agent_{i}_to_agent_{j}"
                channels[channel_name] = Mock()
                channels[channel_name].send = Mock()
                channels[channel_name].receive = Mock(return_value=None)
                channels[channel_name].is_connected = Mock(return_value=True)
    
    return channels