"""Perspective-aware world model implementation."""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import time

from ..utils.validation import (
    validate_tensor_shape, validate_model_config, safe_tensor_operation, PWMKValidationError
)
from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector
from ..optimization.caching import get_cache_manager


class PerspectiveWorldModel(nn.Module, LoggingMixin):
    """
    Neural world model that learns dynamics from multiple agent perspectives.
    
    Combines transformer-based dynamics learning with belief extraction
    for Theory of Mind capabilities.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int, 
        hidden_dim: int = 256,
        num_agents: int = 2,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Validate configuration
        config = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "num_agents": num_agents,
            "num_layers": num_layers
        }
        validate_model_config(config)
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        
        # Caching
        self.cache_manager = get_cache_manager()
        self.enable_caching = True
        
        self.logger.info(
            f"Initializing PerspectiveWorldModel: obs_dim={obs_dim}, action_dim={action_dim}, "
            f"hidden_dim={hidden_dim}, num_agents={num_agents}, num_layers={num_layers}"
        )
        
        # Perspective encoder with agent-specific processing
        self.perspective_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Agent identity embeddings
        self.agent_embedding = nn.Linear(num_agents, hidden_dim)
        
        # State-action projection
        self.state_action_proj = nn.Linear(hidden_dim + action_dim, hidden_dim)
        
        # Dynamics model with improved architecture
        self.dynamics_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Belief extractor with interpretable predicates
        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 64)  # 64 belief predicates
        )
        
    def forward(
        self, 
        observations: torch.Tensor,
        actions: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the world model.
        
        Args:
            observations: Agent observations [batch, seq_len, obs_dim]
            actions: Actions taken [batch, seq_len, action_dim] 
            agent_ids: Agent perspective IDs [batch, seq_len]
            
        Returns:
            next_states: Predicted next states
            beliefs: Extracted belief predicates
        """
        start_time = time.time()
        
        try:
            # Check cache first (only in eval mode to avoid caching during training)
            if (self.enable_caching and 
                self.cache_manager.is_enabled() and 
                not self.training):
                
                cached_result = self.cache_manager.model_cache.get_prediction(
                    observations, actions, agent_ids
                )
                if cached_result is not None:
                    duration = time.time() - start_time
                    get_metrics_collector().record_model_forward("PerspectiveWorldModel_cached", batch_size, duration)
                    return cached_result
            
            # Input validation
            validate_tensor_shape(observations, (self.obs_dim,), "observations", allow_batch=True)
            
            if observations.dim() < 3:
                raise PWMKValidationError(
                    f"Observations must have at least 3 dimensions [batch, seq, obs], got {observations.dim()}"
                )
            
            batch_size, seq_len = observations.shape[:2]
            
            # Validate actions
            if isinstance(actions, torch.Tensor):
                if actions.dim() == 2 and actions.shape == (batch_size, seq_len):
                    # Discrete actions, convert to one-hot
                    if torch.any(actions < 0) or torch.any(actions >= self.action_dim):
                        raise PWMKValidationError(
                            f"Action indices must be in range [0, {self.action_dim})"
                        )
                    actions = torch.nn.functional.one_hot(actions.long(), self.action_dim).float()
                else:
                    validate_tensor_shape(actions, (self.action_dim,), "actions", allow_batch=True)
            
            # Validate agent IDs if provided
            if agent_ids is not None:
                if agent_ids.shape != (batch_size, seq_len):
                    raise PWMKValidationError(
                        f"Agent IDs shape {agent_ids.shape} doesn't match batch shape {(batch_size, seq_len)}"
                    )
                if torch.any(agent_ids < 0) or torch.any(agent_ids >= self.num_agents):
                    raise PWMKValidationError(
                        f"Agent IDs must be in range [0, {self.num_agents})"
                    )
            
            self.logger.debug(
                f"Forward pass started: batch_size={batch_size}, seq_len={seq_len}, "
                f"has_agent_ids={agent_ids is not None}"
            )
            
            # Encode perspectives with agent-specific processing
            encoded = safe_tensor_operation(self.perspective_encoder, observations)
            
            # Add agent identity embeddings if provided
            if agent_ids is not None:
                agent_embeddings = torch.eye(self.num_agents, device=observations.device)[agent_ids]
                agent_encoded = safe_tensor_operation(self.agent_embedding, agent_embeddings)
                encoded = encoded + agent_encoded
            
            # Concatenate state and action
            combined = torch.cat([encoded, actions], dim=-1)
            combined_proj = safe_tensor_operation(self.state_action_proj, combined)
            
            # Apply dynamics model
            next_states = safe_tensor_operation(self.dynamics_model, combined_proj)
            
            # Extract beliefs from latent states
            beliefs = torch.sigmoid(safe_tensor_operation(self.belief_head, next_states))
            
            # Cache result if enabled (only in eval mode)
            if (self.enable_caching and 
                self.cache_manager.is_enabled() and 
                not self.training):
                
                self.cache_manager.model_cache.cache_prediction(
                    observations, actions, next_states, beliefs, agent_ids
                )
            
            # Record metrics
            duration = time.time() - start_time
            get_metrics_collector().record_model_forward("PerspectiveWorldModel", batch_size, duration)
            
            self.logger.debug(
                f"Forward pass completed: duration={duration:.4f}s, "
                f"output_shapes=({next_states.shape}, {beliefs.shape})"
            )
            
            return next_states, beliefs
            
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e} (type: {type(e).__name__})")
            raise
        
    def predict_trajectory(
        self,
        initial_obs: torch.Tensor,
        action_sequence: torch.Tensor,
        horizon: int = 10
    ) -> List[torch.Tensor]:
        """Predict future trajectory given initial observation and actions."""
        trajectory = [initial_obs]
        current_obs = initial_obs
        
        for t in range(horizon):
            if t < len(action_sequence):
                action = action_sequence[t:t+1]
            else:
                action = torch.zeros(1, self.action_dim)
                
            next_state, _ = self.forward(
                current_obs.unsqueeze(0),
                action.unsqueeze(0)
            )
            trajectory.append(next_state.squeeze(0))
            current_obs = next_state.squeeze(0)
            
        return trajectory