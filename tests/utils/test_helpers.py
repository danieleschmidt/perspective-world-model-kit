"""Test utility functions and helpers."""
import torch
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager
import pytest


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: Optional[str] = None
):
    """Assert that two tensors are approximately equal."""
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        error_msg = f"Tensors not close enough:\nActual: {actual}\nExpected: {expected}"
        if msg:
            error_msg = f"{msg}\n{error_msg}"
        raise AssertionError(error_msg)


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, msg: Optional[str] = None):
    """Assert that a tensor has the expected shape."""
    if tensor.shape != expected_shape:
        error_msg = f"Shape mismatch: got {tensor.shape}, expected {expected_shape}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def create_dummy_trajectory(
    num_agents: int = 3,
    seq_len: int = 10,
    obs_dim: int = 64,
    action_dim: int = 4,
    batch_size: int = 1
) -> Dict[str, torch.Tensor]:
    """Create dummy trajectory data for testing."""
    return {
        "observations": torch.randn(batch_size, seq_len, num_agents, obs_dim),
        "actions": torch.randint(0, action_dim, (batch_size, seq_len, num_agents)),
        "rewards": torch.randn(batch_size, seq_len, num_agents),
        "dones": torch.zeros(batch_size, seq_len, num_agents, dtype=torch.bool),
        "next_observations": torch.randn(batch_size, seq_len, num_agents, obs_dim),
        "beliefs": torch.randn(batch_size, seq_len, num_agents, 10),  # 10 belief predicates
        "agent_ids": torch.arange(num_agents).expand(batch_size, seq_len, -1),
    }


def create_belief_facts(num_agents: int = 3) -> Dict[str, List[str]]:
    """Create sample belief facts for testing."""
    facts = {}
    objects = ["key", "treasure", "map"]
    locations = ["room_1", "room_2", "room_3"]
    
    for i in range(num_agents):
        agent_name = f"agent_{i}"
        facts[agent_name] = []
        
        # Basic facts
        facts[agent_name].append(f"agent({agent_name})")
        facts[agent_name].append(f"at({agent_name}, {locations[i % len(locations)]})")
        
        # Object possession
        if i < len(objects):
            facts[agent_name].append(f"has({agent_name}, {objects[i]})")
        
        # Beliefs about other agents
        for j in range(num_agents):
            if i != j:
                other_agent = f"agent_{j}"
                facts[agent_name].append(f"believes({agent_name}, agent({other_agent}))")
    
    return facts


@contextmanager
def temporary_directory():
    """Context manager for creating a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir)


@contextmanager
def temporary_file(suffix: str = "", content: Optional[str] = None):
    """Context manager for creating a temporary file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
    try:
        if content:
            temp_file.write(content)
        temp_file.close()
        yield Path(temp_file.name)
    finally:
        Path(temp_file.name).unlink(missing_ok=True)


def save_test_config(config: Dict[str, Any], filename: str = "test_config.json") -> Path:
    """Save a test configuration to a temporary file."""
    temp_dir = Path(tempfile.mkdtemp())
    config_path = temp_dir / filename
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def load_test_data(data_path: Union[str, Path]) -> Dict[str, Any]:
    """Load test data from a file."""
    data_path = Path(data_path)
    
    if data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            return json.load(f)
    elif data_path.suffix == '.pt':
        return torch.load(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def generate_test_dataset(
    num_episodes: int = 10,
    episode_length: int = 100,
    num_agents: int = 3,
    obs_dim: int = 64,
    action_dim: int = 4
) -> List[Dict[str, torch.Tensor]]:
    """Generate a test dataset of episodes."""
    dataset = []
    
    for _ in range(num_episodes):
        episode = create_dummy_trajectory(
            num_agents=num_agents,
            seq_len=episode_length,
            obs_dim=obs_dim,
            action_dim=action_dim,
            batch_size=1
        )
        
        # Remove batch dimension
        episode = {k: v.squeeze(0) for k, v in episode.items()}
        dataset.append(episode)
    
    return dataset


def mock_model_output(
    batch_size: int,
    output_dim: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate mock model output for testing."""
    return torch.randn(batch_size, output_dim, device=device)


def create_test_environment_config() -> Dict[str, Any]:
    """Create a standard test environment configuration."""
    return {
        "env_type": "multi_agent_box_world",
        "num_agents": 3,
        "grid_size": 10,
        "max_steps": 100,
        "partial_observability": True,
        "view_radius": 3,
        "communication": False,
        "seed": 42
    }


def create_test_model_config() -> Dict[str, Any]:
    """Create a standard test model configuration."""
    return {
        "obs_dim": 64,
        "action_dim": 4,
        "hidden_dim": 128,
        "num_agents": 3,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "batch_size": 32
    }


def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


def skip_if_no_unity():
    """Skip test if Unity is not available."""
    try:
        import mlagents_envs
    except ImportError:
        pytest.skip("Unity ML-Agents not available")


def skip_if_no_prolog():
    """Skip test if Prolog backend is not available."""
    try:
        import pyswip
    except ImportError:
        pytest.skip("Prolog backend not available")


class TensorComparator:
    """Utility class for comparing tensors with different tolerances."""
    
    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        self.rtol = rtol
        self.atol = atol
    
    def __call__(self, actual: torch.Tensor, expected: torch.Tensor, msg: Optional[str] = None):
        """Compare two tensors."""
        assert_tensors_close(actual, expected, self.rtol, self.atol, msg)
    
    def close(self, rtol: float, atol: float):
        """Return a new comparator with different tolerances."""
        return TensorComparator(rtol, atol)


# Global tensor comparator with default tolerances
tensor_compare = TensorComparator()


class MockLogger:
    """Mock logger for testing."""
    
    def __init__(self):
        self.logs = []
    
    def info(self, msg: str):
        self.logs.append(("INFO", msg))
    
    def warning(self, msg: str):
        self.logs.append(("WARNING", msg))
    
    def error(self, msg: str):
        self.logs.append(("ERROR", msg))
    
    def debug(self, msg: str):
        self.logs.append(("DEBUG", msg))
    
    def clear(self):
        self.logs.clear()
    
    def get_logs(self, level: Optional[str] = None) -> List[tuple]:
        if level:
            return [log for log in self.logs if log[0] == level]
        return self.logs.copy()


def parametrize_device():
    """Pytest parametrize decorator for testing on different devices."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    
    return pytest.mark.parametrize("device", devices)