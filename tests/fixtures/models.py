"""Test fixtures for models and neural networks."""
import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock


@pytest.fixture
def model_configs() -> Dict[str, Dict[str, Any]]:
    """Various model configurations for testing."""
    return {
        "small": {
            "obs_dim": 32,
            "action_dim": 4,
            "hidden_dim": 64,
            "num_agents": 2,
            "num_layers": 1,
            "dropout": 0.0
        },
        "medium": {
            "obs_dim": 64,
            "action_dim": 4,
            "hidden_dim": 128,
            "num_agents": 3,
            "num_layers": 2,
            "dropout": 0.1
        },
        "large": {
            "obs_dim": 128,
            "action_dim": 8,
            "hidden_dim": 256,
            "num_agents": 5,
            "num_layers": 3,
            "dropout": 0.2
        }
    }


@pytest.fixture
def mock_world_model() -> Mock:
    """Mock world model for testing."""
    model = Mock()
    model.obs_dim = 64
    model.action_dim = 4
    model.hidden_dim = 128
    model.num_agents = 3
    
    # Mock forward pass
    def mock_forward(obs, actions, agent_ids):
        batch_size = obs.shape[0]
        next_obs = torch.randn(batch_size, model.obs_dim)
        beliefs = torch.randn(batch_size, 10)  # 10 belief predicates
        return next_obs, beliefs
    
    model.forward = Mock(side_effect=mock_forward)
    model.predict = Mock(side_effect=mock_forward)
    
    # Mock training methods
    model.train = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.randn(10, 10, requires_grad=True)])
    
    return model


@pytest.fixture
def mock_perspective_encoder() -> Mock:
    """Mock perspective encoder for testing."""
    encoder = Mock()
    encoder.obs_dim = 64
    encoder.hidden_dim = 128
    encoder.num_perspectives = 3
    
    def mock_encode(obs, perspective_id):
        batch_size = obs.shape[0]
        return torch.randn(batch_size, encoder.hidden_dim)
    
    encoder.forward = Mock(side_effect=mock_encode)
    encoder.encode = Mock(side_effect=mock_encode)
    
    return encoder


@pytest.fixture
def mock_dynamics_model() -> Mock:
    """Mock dynamics model for testing."""
    model = Mock()
    model.hidden_dim = 128
    model.action_dim = 4
    
    def mock_predict_next(hidden_state, actions):
        batch_size = hidden_state.shape[0]
        return torch.randn(batch_size, model.hidden_dim)
    
    model.forward = Mock(side_effect=mock_predict_next)
    model.predict_next = Mock(side_effect=mock_predict_next)
    
    return model


@pytest.fixture
def mock_belief_extractor() -> Mock:
    """Mock belief extractor for testing."""
    extractor = Mock()
    extractor.hidden_dim = 128
    extractor.num_predicates = 10
    
    def mock_extract(hidden_state):
        batch_size = hidden_state.shape[0]
        return torch.randn(batch_size, extractor.num_predicates)
    
    extractor.forward = Mock(side_effect=mock_extract)
    extractor.extract_beliefs = Mock(side_effect=mock_extract)
    
    return extractor


@pytest.fixture
def sample_model_weights() -> Dict[str, torch.Tensor]:
    """Sample model weights for testing serialization."""
    return {
        "encoder.weight": torch.randn(128, 64),
        "encoder.bias": torch.randn(128),
        "dynamics.weight": torch.randn(128, 132),  # hidden_dim + action_dim
        "dynamics.bias": torch.randn(128),
        "belief_head.weight": torch.randn(10, 128),
        "belief_head.bias": torch.randn(10)
    }


@pytest.fixture
def mock_optimizer() -> Mock:
    """Mock optimizer for testing."""
    optimizer = Mock()
    optimizer.param_groups = [{"lr": 1e-4}]
    optimizer.state_dict = Mock(return_value={"state": {}, "param_groups": optimizer.param_groups})
    optimizer.load_state_dict = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    
    return optimizer


@pytest.fixture
def mock_loss_function() -> Mock:
    """Mock loss function for testing."""
    loss_fn = Mock()
    
    def mock_compute_loss(pred, target, reduction="mean"):
        if reduction == "mean":
            return torch.tensor(0.5, requires_grad=True)
        else:
            batch_size = pred.shape[0] if hasattr(pred, 'shape') else 1
            return torch.tensor([0.5] * batch_size, requires_grad=True)
    
    loss_fn.forward = Mock(side_effect=mock_compute_loss)
    loss_fn.__call__ = Mock(side_effect=mock_compute_loss)
    
    return loss_fn


@pytest.fixture
def training_metrics() -> Dict[str, float]:
    """Sample training metrics for testing."""
    return {
        "loss": 0.45,
        "state_prediction_error": 0.12,
        "belief_accuracy": 0.87,
        "dynamics_loss": 0.23,
        "belief_loss": 0.22,
        "learning_rate": 1e-4,
        "gradient_norm": 2.3,
        "epoch": 10,
        "step": 1000
    }


@pytest.fixture
def mock_data_loader() -> Mock:
    """Mock data loader for testing."""
    loader = Mock()
    
    # Sample batch data
    batch_data = {
        "observations": torch.randn(32, 64),
        "actions": torch.randint(0, 4, (32,)),
        "next_observations": torch.randn(32, 64),
        "rewards": torch.randn(32),
        "dones": torch.zeros(32, dtype=torch.bool),
        "agent_ids": torch.randint(0, 3, (32,))
    }
    
    # Mock iteration
    loader.__iter__ = Mock(return_value=iter([batch_data] * 5))  # 5 batches
    loader.__len__ = Mock(return_value=5)
    
    return loader


@pytest.fixture
def model_checkpoint() -> Dict[str, Any]:
    """Sample model checkpoint for testing."""
    return {
        "epoch": 50,
        "step": 5000,
        "model_state_dict": {
            "encoder.weight": torch.randn(128, 64),
            "encoder.bias": torch.randn(128)
        },
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{"lr": 1e-4}]
        },
        "loss": 0.35,
        "metrics": {
            "state_prediction_error": 0.08,
            "belief_accuracy": 0.92
        },
        "config": {
            "obs_dim": 64,
            "hidden_dim": 128,
            "num_agents": 3
        }
    }


@pytest.fixture
def attention_weights() -> torch.Tensor:
    """Sample attention weights for testing attention mechanisms."""
    batch_size, num_heads, seq_len = 4, 8, 10
    return torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)


@pytest.fixture
def mock_tokenizer() -> Mock:
    """Mock tokenizer for text-based components."""
    tokenizer = Mock()
    tokenizer.vocab_size = 1000
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = Mock(return_value="sample text")
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    
    return tokenizer


class SimpleTestModel(nn.Module):
    """Simple model for integration testing."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.encoder(x))
        return self.decoder(h)


@pytest.fixture
def simple_test_model() -> SimpleTestModel:
    """Simple test model for integration tests."""
    return SimpleTestModel()


@pytest.fixture
def device_aware_model():
    """Factory for creating device-aware models."""
    def create_model(device: torch.device) -> SimpleTestModel:
        model = SimpleTestModel()
        return model.to(device)
    
    return create_model