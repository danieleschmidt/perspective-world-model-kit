"""Perspective-aware world model implementation."""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


class PerspectiveWorldModel(nn.Module):
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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        
        # Perspective encoder
        self.perspective_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dynamics model
        self.dynamics_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4
            ),
            num_layers=num_layers
        )
        
        # Belief extractor
        self.belief_head = nn.Linear(hidden_dim, 64)  # Belief predicates
        
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
        batch_size, seq_len = observations.shape[:2]
        
        # Encode perspectives
        encoded = self.perspective_encoder(observations)
        
        # Apply dynamics
        next_states = self.dynamics_model(encoded.transpose(0, 1)).transpose(0, 1)
        
        # Extract beliefs
        beliefs = self.belief_head(next_states)
        
        return next_states, beliefs
        
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