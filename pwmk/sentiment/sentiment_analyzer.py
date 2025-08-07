"""
Core sentiment analysis implementation with multi-agent capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import AutoTokenizer, AutoModel
from ..core.beliefs import BeliefStore
from ..utils.logging import get_logger
from .exceptions import (
    ModelLoadError,
    TokenizationError,
    BeliefUpdateError,
    AgentNotFoundError
)
from .validation import (
    validate_sentiment_scores,
    validate_agent_id,
    validate_text_input,
    validate_model_inputs,
    validate_history_data
)

logger = get_logger(__name__)


class SentimentAnalyzer(nn.Module):
    """
    Base sentiment analyzer using transformer architecture.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained transformer with error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer = AutoModel.from_pretrained(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {model_name}: {e}")
        
        # Sentiment classification head
        transformer_dim = self.transformer.config.hidden_size
        self.sentiment_head = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through sentiment analyzer."""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.sentiment_head(pooled_output)
        return logits
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of input text."""
        try:
            # Validate and sanitize input
            text = validate_text_input(text)
            
            # Tokenize with error handling
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Validate tokenizer outputs
            validate_model_inputs(inputs["input_ids"], inputs["attention_mask"])
            
            with torch.no_grad():
                logits = self.forward(inputs["input_ids"], inputs["attention_mask"])
                
                # Check for valid logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError("Model produced invalid logits")
                    
                probabilities = torch.softmax(logits, dim=-1).squeeze()
                
            sentiment_labels = ["negative", "neutral", "positive"]
            result = {
                label: float(prob)
                for label, prob in zip(sentiment_labels, probabilities)
            }
            
            # Validate output
            validate_sentiment_scores(result)
            
            logger.debug(f"Analyzed text sentiment: {result}")
            return result
            
        except Exception as e:
            if isinstance(e, (TokenizationError, ValueError)):
                raise
            raise TokenizationError(f"Failed to analyze text sentiment: {e}")


class MultiAgentSentimentAnalyzer:
    """
    Multi-agent sentiment analyzer that tracks sentiment for each agent
    and their beliefs about others' sentiments.
    """
    
    def __init__(
        self,
        num_agents: int,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        belief_store: Optional[BeliefStore] = None
    ):
        self.num_agents = num_agents
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.belief_store = belief_store or BeliefStore()
        
        # Track sentiment history for each agent
        self.agent_sentiment_history: Dict[int, List[Dict[str, float]]] = {
            i: [] for i in range(num_agents)
        }
        
    def analyze_agent_communication(
        self,
        agent_id: int,
        text: str,
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Analyze sentiment of agent's communication."""
        try:
            # Validate inputs
            validate_agent_id(agent_id, self.num_agents)
            text = validate_text_input(text)
            
            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_text(text)
            
            # Store in agent's sentiment history
            history_entry = {
                "timestamp": len(self.agent_sentiment_history[agent_id]),
                "text": text,
                "sentiment": sentiment,
                "context": context or {}
            }
            
            self.agent_sentiment_history[agent_id].append(history_entry)
            
            # Limit history size to prevent memory issues
            max_history = 1000
            if len(self.agent_sentiment_history[agent_id]) > max_history:
                self.agent_sentiment_history[agent_id] = self.agent_sentiment_history[agent_id][-max_history:]
                logger.debug(f"Trimmed history for agent {agent_id} to {max_history} entries")
            
            # Update belief store with sentiment information
            try:
                dominant_sentiment = max(sentiment.items(), key=lambda x: x[1])[0]
                self.belief_store.add_belief(
                    f"agent_{agent_id}",
                    f"current_sentiment(agent_{agent_id}, {dominant_sentiment})"
                )
            except Exception as e:
                logger.warning(f"Failed to update belief store for agent {agent_id}: {e}")
                # Don't raise - sentiment analysis succeeded even if belief update failed
            
            logger.info(f"Agent {agent_id} sentiment: {sentiment}")
            return sentiment
            
        except Exception as e:
            if isinstance(e, (AgentNotFoundError, ValueError)):
                raise
            raise BeliefUpdateError(f"Failed to analyze agent {agent_id} communication: {e}")
        
    def get_agent_sentiment_history(self, agent_id: int) -> List[Dict]:
        """Get sentiment history for specific agent."""
        try:
            validate_agent_id(agent_id, self.num_agents)
            history = self.agent_sentiment_history.get(agent_id, [])
            validate_history_data(history, min_entries=0)  # Allow empty history
            return history
        except Exception as e:
            if isinstance(e, AgentNotFoundError):
                raise
            logger.error(f"Failed to get history for agent {agent_id}: {e}")
            return []
        
    def get_current_sentiment_state(self) -> Dict[int, Dict[str, float]]:
        """Get current sentiment state for all agents."""
        current_state = {}
        default_sentiment = {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
        
        try:
            for agent_id in range(self.num_agents):
                history = self.agent_sentiment_history.get(agent_id, [])
                if history:
                    try:
                        current_sentiment = history[-1]["sentiment"]
                        validate_sentiment_scores(current_sentiment)
                        current_state[agent_id] = current_sentiment
                    except Exception as e:
                        logger.warning(f"Invalid sentiment data for agent {agent_id}: {e}")
                        current_state[agent_id] = default_sentiment
                else:
                    current_state[agent_id] = default_sentiment
                    
            return current_state
            
        except Exception as e:
            logger.error(f"Failed to get current sentiment state: {e}")
            return {i: default_sentiment for i in range(self.num_agents)}
        
    def analyze_group_sentiment(self) -> Dict[str, float]:
        """Analyze overall group sentiment."""
        all_sentiments = []
        for agent_id in range(self.num_agents):
            history = self.agent_sentiment_history[agent_id]
            if history:
                all_sentiments.append(history[-1]["sentiment"])
                
        if not all_sentiments:
            return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
            
        # Average sentiments across agents
        group_sentiment = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for sentiment in all_sentiments:
            for key in group_sentiment:
                group_sentiment[key] += sentiment[key]
                
        for key in group_sentiment:
            group_sentiment[key] /= len(all_sentiments)
            
        return group_sentiment
        
    def predict_agent_sentiment_response(
        self,
        target_agent: int,
        stimulus_text: str
    ) -> Dict[str, float]:
        """Predict how target agent might respond sentimentally to stimulus."""
        # Simple baseline: use agent's historical sentiment patterns
        history = self.agent_sentiment_history[target_agent]
        if not history:
            return self.sentiment_analyzer.analyze_text(stimulus_text)
            
        # Weighted average of recent sentiments
        recent_sentiments = history[-5:]  # Last 5 communications
        weights = np.linspace(0.1, 1.0, len(recent_sentiments))
        
        weighted_sentiment = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        total_weight = sum(weights)
        
        for i, record in enumerate(recent_sentiments):
            weight = weights[i] / total_weight
            for key in weighted_sentiment:
                weighted_sentiment[key] += record["sentiment"][key] * weight
                
        return weighted_sentiment