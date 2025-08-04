"""Input validation and error handling for PWMK."""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Union, Optional


class PWMKValidationError(Exception):
    """Custom exception for PWMK validation errors."""
    pass


def validate_tensor_shape(
    tensor: torch.Tensor, 
    expected_shape: Tuple[int, ...], 
    name: str = "tensor",
    allow_batch: bool = True
) -> None:
    """
    Validate that a tensor has the expected shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (excluding batch dimension if allow_batch=True)
        name: Name of tensor for error messages
        allow_batch: Whether to allow additional batch dimension
        
    Raises:
        PWMKValidationError: If tensor shape is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise PWMKValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    actual_shape = tensor.shape
    
    if allow_batch:
        # Allow any batch dimensions at the start
        if len(actual_shape) < len(expected_shape):
            raise PWMKValidationError(
                f"{name} shape {actual_shape} is too short, expected at least {expected_shape}"
            )
        
        # Check the last dimensions match expected shape
        if actual_shape[-len(expected_shape):] != expected_shape:
            raise PWMKValidationError(
                f"{name} shape {actual_shape} doesn't match expected {expected_shape} "
                f"(checking last {len(expected_shape)} dimensions)"
            )
    else:
        if actual_shape != expected_shape:
            raise PWMKValidationError(
                f"{name} shape {actual_shape} doesn't match expected {expected_shape}"
            )


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate that a configuration dictionary has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key names
        
    Raises:
        PWMKValidationError: If required keys are missing
    """
    if not isinstance(config, dict):
        raise PWMKValidationError(f"Config must be a dictionary, got {type(config)}")
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise PWMKValidationError(f"Config missing required keys: {missing_keys}")


def validate_observation_space(
    observations: Union[torch.Tensor, List[torch.Tensor]], 
    expected_dim: int,
    num_agents: Optional[int] = None
) -> None:
    """
    Validate observation tensors from environment.
    
    Args:
        observations: Single tensor or list of tensors
        expected_dim: Expected observation dimension
        num_agents: Expected number of agents (if list input)
        
    Raises:
        PWMKValidationError: If observations are invalid
    """
    if isinstance(observations, list):
        if num_agents is not None and len(observations) != num_agents:
            raise PWMKValidationError(
                f"Expected {num_agents} observations, got {len(observations)}"
            )
        
        for i, obs in enumerate(observations):
            validate_tensor_shape(obs, (expected_dim,), f"observation[{i}]")
    
    elif isinstance(observations, torch.Tensor):
        validate_tensor_shape(observations, (expected_dim,), "observations", allow_batch=True)
    
    else:
        raise PWMKValidationError(
            f"Observations must be tensor or list of tensors, got {type(observations)}"
        )


def validate_action_space(
    actions: Union[torch.Tensor, List[int]], 
    action_dim: int,
    num_agents: Optional[int] = None
) -> None:
    """
    Validate action inputs.
    
    Args:
        actions: Action tensor or list of action indices
        action_dim: Maximum action value (for discrete actions)
        num_agents: Expected number of agents (if list input)
        
    Raises:
        PWMKValidationError: If actions are invalid
    """
    if isinstance(actions, list):
        if num_agents is not None and len(actions) != num_agents:
            raise PWMKValidationError(
                f"Expected {num_agents} actions, got {len(actions)}"
            )
        
        for i, action in enumerate(actions):
            if not isinstance(action, int):
                raise PWMKValidationError(
                    f"Action[{i}] must be integer, got {type(action)}"
                )
            if not (0 <= action < action_dim):
                raise PWMKValidationError(
                    f"Action[{i}] value {action} out of range [0, {action_dim})"
                )
    
    elif isinstance(actions, torch.Tensor):
        if actions.dtype not in [torch.int32, torch.int64, torch.long]:
            raise PWMKValidationError(
                f"Action tensor must have integer dtype, got {actions.dtype}"
            )
        
        if torch.any(actions < 0) or torch.any(actions >= action_dim):
            raise PWMKValidationError(
                f"Action values must be in range [0, {action_dim}), "
                f"got min={actions.min().item()}, max={actions.max().item()}"
            )
    
    else:
        raise PWMKValidationError(
            f"Actions must be tensor or list of ints, got {type(actions)}"
        )


def validate_belief_string(belief: str) -> None:
    """
    Validate belief string format.
    
    Args:
        belief: Belief string in Prolog-like syntax
        
    Raises:
        PWMKValidationError: If belief format is invalid
    """
    if not isinstance(belief, str):
        raise PWMKValidationError(f"Belief must be string, got {type(belief)}")
    
    if not belief.strip():
        raise PWMKValidationError("Belief cannot be empty")
    
    # Basic format validation - should contain predicate with arguments
    if "(" not in belief or ")" not in belief:
        raise PWMKValidationError(
            f"Belief '{belief}' must be in predicate(args) format"
        )
    
    # Check balanced parentheses
    paren_count = 0
    for char in belief:
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1
            if paren_count < 0:
                raise PWMKValidationError(f"Unbalanced parentheses in belief '{belief}'")
    
    if paren_count != 0:
        raise PWMKValidationError(f"Unbalanced parentheses in belief '{belief}'")


def validate_agent_id(agent_id: str) -> None:
    """
    Validate agent ID format.
    
    Args:
        agent_id: Agent identifier string
        
    Raises:
        PWMKValidationError: If agent ID is invalid
    """
    if not isinstance(agent_id, str):
        raise PWMKValidationError(f"Agent ID must be string, got {type(agent_id)}")
    
    if not agent_id.strip():
        raise PWMKValidationError("Agent ID cannot be empty")
    
    # Should be alphanumeric with underscores
    if not agent_id.replace("_", "").replace("-", "").isalnum():
        raise PWMKValidationError(
            f"Agent ID '{agent_id}' must be alphanumeric with underscores/hyphens"
        )


def safe_tensor_operation(operation, *args, **kwargs):
    """
    Safely execute tensor operations with error handling.
    
    Args:
        operation: Function to execute
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Result of operation
        
    Raises:
        PWMKValidationError: If operation fails
    """
    try:
        return operation(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise PWMKValidationError(
                f"GPU out of memory during {operation.__name__}. "
                "Try reducing batch size or using CPU."
            )
        elif "size mismatch" in str(e).lower():
            raise PWMKValidationError(
                f"Tensor size mismatch in {operation.__name__}: {e}"
            )
        else:
            raise PWMKValidationError(f"Tensor operation failed: {e}")
    except Exception as e:
        raise PWMKValidationError(f"Unexpected error in {operation.__name__}: {e}")


def validate_model_config(config: Dict[str, Any]) -> None:
    """
    Validate world model configuration.
    
    Args:
        config: Model configuration dictionary
        
    Raises:
        PWMKValidationError: If configuration is invalid
    """
    required_keys = ["obs_dim", "action_dim", "hidden_dim", "num_agents"]
    validate_config(config, required_keys)
    
    # Check value ranges
    if config["obs_dim"] <= 0:
        raise PWMKValidationError("obs_dim must be positive")
    
    if config["action_dim"] <= 0:
        raise PWMKValidationError("action_dim must be positive")
    
    if config["hidden_dim"] <= 0:
        raise PWMKValidationError("hidden_dim must be positive")
    
    if config["num_agents"] <= 0:
        raise PWMKValidationError("num_agents must be positive")
    
    # Check optional parameters
    if "num_layers" in config and config["num_layers"] <= 0:
        raise PWMKValidationError("num_layers must be positive")


def validate_environment_step_output(
    observations: List[np.ndarray],
    rewards: List[float], 
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    num_agents: int
) -> None:
    """
    Validate environment step output format.
    
    Args:
        observations: List of agent observations
        rewards: List of agent rewards
        terminated: Episode termination flag
        truncated: Episode truncation flag
        info: Additional information dictionary
        num_agents: Expected number of agents
        
    Raises:
        PWMKValidationError: If step output is invalid
    """
    if not isinstance(observations, list):
        raise PWMKValidationError("Observations must be a list")
    
    if len(observations) != num_agents:
        raise PWMKValidationError(
            f"Expected {num_agents} observations, got {len(observations)}"
        )
    
    if not isinstance(rewards, list):
        raise PWMKValidationError("Rewards must be a list")
    
    if len(rewards) != num_agents:
        raise PWMKValidationError(
            f"Expected {num_agents} rewards, got {len(rewards)}"
        )
    
    if not isinstance(terminated, bool):
        raise PWMKValidationError("Terminated must be boolean")
    
    if not isinstance(truncated, bool):
        raise PWMKValidationError("Truncated must be boolean")
    
    if not isinstance(info, dict):
        raise PWMKValidationError("Info must be dictionary")