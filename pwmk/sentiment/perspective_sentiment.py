"""
Perspective-aware sentiment analysis that considers each agent's viewpoint.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..core.world_model import PerspectiveWorldModel
from ..core.beliefs import BeliefStore
from .sentiment_analyzer import SentimentAnalyzer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class PerspectiveSentimentModel(nn.Module):
    """
    Sentiment model that considers different agent perspectives.
    Integrates with PWMK's perspective-aware world modeling.
    """
    
    def __init__(
        self,
        world_model: PerspectiveWorldModel,
        sentiment_analyzer: SentimentAnalyzer,
        num_agents: int,
        perspective_dim: int = 128
    ):
        super().__init__()
        self.world_model = world_model
        self.sentiment_analyzer = sentiment_analyzer
        self.num_agents = num_agents
        self.perspective_dim = perspective_dim
        
        # Perspective encoding for sentiment
        self.perspective_encoder = nn.Sequential(
            nn.Linear(world_model.hidden_dim, perspective_dim),
            nn.ReLU(),
            nn.Linear(perspective_dim, perspective_dim)
        )
        
        # Sentiment modulation based on perspective
        self.sentiment_modulator = nn.Sequential(
            nn.Linear(perspective_dim + sentiment_analyzer.sentiment_head[-1].out_features, 
                     perspective_dim),
            nn.ReLU(),
            nn.Linear(perspective_dim, sentiment_analyzer.sentiment_head[-1].out_features)
        )
        
    def forward(
        self,
        text_inputs: Dict[str, torch.Tensor],
        world_state: torch.Tensor,
        agent_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with perspective-aware sentiment analysis.
        
        Args:
            text_inputs: Tokenized text inputs
            world_state: Current world state representation
            agent_id: ID of the agent whose perspective to use
            
        Returns:
            Tuple of (sentiment_logits, perspective_embedding)
        """
        # Get base sentiment analysis
        base_sentiment = self.sentiment_analyzer(
            text_inputs["input_ids"],
            text_inputs["attention_mask"]
        )
        
        # Encode agent's perspective from world state
        perspective_embedding = self.perspective_encoder(world_state)
        
        # Modulate sentiment based on perspective
        combined_features = torch.cat([perspective_embedding, base_sentiment], dim=-1)
        perspective_sentiment = self.sentiment_modulator(combined_features)
        
        return perspective_sentiment, perspective_embedding
        
    def analyze_from_perspective(
        self,
        text: str,
        world_state: np.ndarray,
        agent_id: int
    ) -> Dict[str, float]:
        """
        Analyze sentiment from specific agent's perspective.
        
        Args:
            text: Text to analyze
            world_state: Current world state
            agent_id: Agent whose perspective to use
            
        Returns:
            Perspective-adjusted sentiment scores
        """
        # Tokenize text
        text_inputs = self.sentiment_analyzer.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Convert world state to tensor
        world_tensor = torch.from_numpy(world_state).float().unsqueeze(0)
        
        with torch.no_grad():
            sentiment_logits, perspective_emb = self.forward(
                text_inputs, world_tensor, agent_id
            )
            probabilities = torch.softmax(sentiment_logits, dim=-1).squeeze()
            
        sentiment_labels = ["negative", "neutral", "positive"]
        result = {
            label: float(prob)
            for label, prob in zip(sentiment_labels, probabilities)
        }
        
        logger.debug(f"Perspective sentiment for agent {agent_id}: {result}")
        return result
        
    def compare_perspectives(
        self,
        text: str,
        world_state: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Compare sentiment analysis from all agents' perspectives.
        
        Args:
            text: Text to analyze
            world_state: Current world state
            
        Returns:
            Sentiment scores for each agent's perspective
        """
        perspectives = {}
        for agent_id in range(self.num_agents):
            perspectives[agent_id] = self.analyze_from_perspective(
                text, world_state, agent_id
            )
            
        return perspectives
        
    def get_consensus_sentiment(
        self,
        text: str,
        world_state: np.ndarray,
        weight_method: str = "uniform"
    ) -> Dict[str, float]:
        """
        Get consensus sentiment across all perspectives.
        
        Args:
            text: Text to analyze
            world_state: Current world state
            weight_method: How to weight different perspectives
            
        Returns:
            Consensus sentiment scores
        """
        perspectives = self.compare_perspectives(text, world_state)
        
        if weight_method == "uniform":
            # Simple average
            consensus = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
            for agent_sentiment in perspectives.values():
                for key in consensus:
                    consensus[key] += agent_sentiment[key]
                    
            for key in consensus:
                consensus[key] /= len(perspectives)
                
        elif weight_method == "confidence":
            # Weight by confidence (max probability)
            weighted_consensus = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
            total_weight = 0.0
            
            for agent_sentiment in perspectives.values():
                confidence = max(agent_sentiment.values())
                total_weight += confidence
                
                for key in weighted_consensus:
                    weighted_consensus[key] += agent_sentiment[key] * confidence
                    
            consensus = {
                key: value / total_weight if total_weight > 0 else value
                for key, value in weighted_consensus.items()
            }
            
        else:
            raise ValueError(f"Unknown weight_method: {weight_method}")
            
        return consensus
        
    def detect_sentiment_disagreement(
        self,
        text: str,
        world_state: np.ndarray,
        threshold: float = 0.3
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Detect if agents have significantly different sentiment interpretations.
        
        Args:
            text: Text to analyze
            world_state: Current world state
            threshold: Threshold for disagreement detection
            
        Returns:
            Tuple of (has_disagreement, disagreement_metrics)
        """
        perspectives = self.compare_perspectives(text, world_state)
        
        # Calculate variance in sentiment scores
        sentiment_arrays = {
            "negative": [],
            "neutral": [], 
            "positive": []
        }
        
        for agent_sentiment in perspectives.values():
            for key in sentiment_arrays:
                sentiment_arrays[key].append(agent_sentiment[key])
                
        disagreement_metrics = {}
        has_disagreement = False
        
        for sentiment_type, scores in sentiment_arrays.items():
            variance = np.var(scores)
            disagreement_metrics[f"{sentiment_type}_variance"] = float(variance)
            
            if variance > threshold:
                has_disagreement = True
                
        disagreement_metrics["max_variance"] = max(
            disagreement_metrics[f"{s}_variance"] 
            for s in ["negative", "neutral", "positive"]
        )
        
        return has_disagreement, disagreement_metrics