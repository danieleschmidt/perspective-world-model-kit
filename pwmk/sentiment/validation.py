"""
Validation utilities for sentiment analysis components.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from .exceptions import (
    InvalidSentimentScoreError,
    AgentNotFoundError,
    InsufficientDataError
)


def validate_sentiment_scores(sentiment: Dict[str, float]) -> None:
    """
    Validate sentiment score dictionary.
    
    Args:
        sentiment: Dictionary of sentiment scores
        
    Raises:
        InvalidSentimentScoreError: If sentiment scores are invalid
    """
    required_keys = {"negative", "neutral", "positive"}
    
    if not isinstance(sentiment, dict):
        raise InvalidSentimentScoreError("Sentiment must be a dictionary")
        
    if set(sentiment.keys()) != required_keys:
        raise InvalidSentimentScoreError(
            f"Sentiment must contain exactly {required_keys}, got {set(sentiment.keys())}"
        )
        
    for key, value in sentiment.items():
        if not isinstance(value, (int, float)):
            raise InvalidSentimentScoreError(f"Sentiment score for '{key}' must be numeric, got {type(value)}")
            
        if not (0.0 <= value <= 1.0):
            raise InvalidSentimentScoreError(f"Sentiment score for '{key}' must be between 0 and 1, got {value}")
    
    total = sum(sentiment.values())
    if not (0.95 <= total <= 1.05):  # Allow small floating point errors
        raise InvalidSentimentScoreError(f"Sentiment scores must sum to ~1.0, got {total}")


def validate_agent_id(agent_id: int, num_agents: int) -> None:
    """
    Validate agent ID is within valid range.
    
    Args:
        agent_id: Agent identifier
        num_agents: Total number of agents
        
    Raises:
        AgentNotFoundError: If agent ID is invalid
    """
    if not isinstance(agent_id, int):
        raise AgentNotFoundError(f"Agent ID must be an integer, got {type(agent_id)}")
        
    if agent_id < 0 or agent_id >= num_agents:
        raise AgentNotFoundError(f"Agent ID {agent_id} not in range [0, {num_agents})")


def validate_text_input(text: str, max_length: int = 10000) -> str:
    """
    Validate and sanitize text input.
    
    Args:
        text: Input text to validate
        max_length: Maximum allowed text length
        
    Returns:
        Sanitized text
        
    Raises:
        ValueError: If text is invalid
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be a string, got {type(text)}")
        
    if not text.strip():
        raise ValueError("Text cannot be empty or only whitespace")
        
    if len(text) > max_length:
        raise ValueError(f"Text length {len(text)} exceeds maximum {max_length}")
        
    # Basic sanitization - remove control characters except newlines and tabs
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return sanitized.strip()


def validate_world_state(world_state: np.ndarray, expected_dim: Optional[int] = None) -> None:
    """
    Validate world state representation.
    
    Args:
        world_state: World state array
        expected_dim: Expected dimensionality
        
    Raises:
        ValueError: If world state is invalid
    """
    if not isinstance(world_state, np.ndarray):
        raise ValueError(f"World state must be numpy array, got {type(world_state)}")
        
    if world_state.size == 0:
        raise ValueError("World state cannot be empty")
        
    if expected_dim is not None and world_state.shape[-1] != expected_dim:
        raise ValueError(f"World state dimension {world_state.shape[-1]} != expected {expected_dim}")
        
    if not np.isfinite(world_state).all():
        raise ValueError("World state contains non-finite values")


def validate_model_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int = 512
) -> None:
    """
    Validate transformer model inputs.
    
    Args:
        input_ids: Token IDs tensor
        attention_mask: Attention mask tensor
        max_length: Maximum sequence length
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError(f"input_ids must be torch.Tensor, got {type(input_ids)}")
        
    if not isinstance(attention_mask, torch.Tensor):
        raise ValueError(f"attention_mask must be torch.Tensor, got {type(attention_mask)}")
        
    if input_ids.shape != attention_mask.shape:
        raise ValueError(f"input_ids shape {input_ids.shape} != attention_mask shape {attention_mask.shape}")
        
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D tensor, got {input_ids.dim()}D")
        
    if input_ids.size(1) > max_length:
        raise ValueError(f"Sequence length {input_ids.size(1)} exceeds maximum {max_length}")
        
    if input_ids.min() < 0:
        raise ValueError("input_ids contains negative values")


def validate_history_data(history: List[Dict], min_entries: int = 1) -> None:
    """
    Validate agent history data structure.
    
    Args:
        history: List of history entries
        min_entries: Minimum required entries
        
    Raises:
        InsufficientDataError: If history data is insufficient
        ValueError: If history structure is invalid
    """
    if not isinstance(history, list):
        raise ValueError(f"History must be a list, got {type(history)}")
        
    if len(history) < min_entries:
        raise InsufficientDataError(f"History has {len(history)} entries, need at least {min_entries}")
        
    required_keys = {"timestamp", "text", "sentiment", "context"}
    
    for i, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise ValueError(f"History entry {i} must be a dictionary, got {type(entry)}")
            
        missing_keys = required_keys - set(entry.keys())
        if missing_keys:
            raise ValueError(f"History entry {i} missing keys: {missing_keys}")
            
        # Validate sentiment scores in history
        try:
            validate_sentiment_scores(entry["sentiment"])
        except InvalidSentimentScoreError as e:
            raise ValueError(f"Invalid sentiment in history entry {i}: {e}")


def validate_belief_query(query: str) -> str:
    """
    Validate and sanitize belief query string.
    
    Args:
        query: Prolog-style query string
        
    Returns:
        Sanitized query
        
    Raises:
        ValueError: If query is invalid
    """
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, got {type(query)}")
        
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")
        
    # Basic validation for Prolog syntax
    if not query.endswith('.'):
        query += '.'
        
    # Check for balanced parentheses
    paren_count = query.count('(') - query.count(')')
    if paren_count != 0:
        raise ValueError(f"Unbalanced parentheses in query: {query}")
        
    # Sanitize potentially dangerous characters
    dangerous_chars = [';', '!', ':-']
    for char in dangerous_chars:
        if char in query and not _is_safe_prolog_construct(query, char):
            raise ValueError(f"Potentially unsafe character '{char}' in query")
            
    return query


def _is_safe_prolog_construct(query: str, char: str) -> bool:
    """Check if character usage is safe in Prolog context."""
    if char == ':-':
        # Rules are generally safe in queries
        return True
    elif char == '!':
        # Cuts should be used carefully
        return 'cut' in query.lower()
    elif char == ';':
        # Disjunctions are generally safe
        return True
    return False


def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate sentiment analysis configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated and normalized configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
        
    validated_config = {}
    
    # Model configuration
    if "model_name" in config:
        if not isinstance(config["model_name"], str):
            raise ValueError("model_name must be a string")
        validated_config["model_name"] = config["model_name"]
        
    if "hidden_dim" in config:
        if not isinstance(config["hidden_dim"], int) or config["hidden_dim"] <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        validated_config["hidden_dim"] = config["hidden_dim"]
        
    if "dropout" in config:
        dropout = config["dropout"]
        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be a number between 0.0 and 1.0")
        validated_config["dropout"] = float(dropout)
        
    # Agent configuration
    if "num_agents" in config:
        if not isinstance(config["num_agents"], int) or config["num_agents"] <= 0:
            raise ValueError("num_agents must be a positive integer")
        validated_config["num_agents"] = config["num_agents"]
        
    # Analysis configuration
    if "max_history" in config:
        if not isinstance(config["max_history"], int) or config["max_history"] <= 0:
            raise ValueError("max_history must be a positive integer")
        validated_config["max_history"] = config["max_history"]
        
    return validated_config