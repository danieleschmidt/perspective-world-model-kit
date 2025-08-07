"""
Belief-aware sentiment tracking that integrates with PWMK's belief system.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from ..core.beliefs import BeliefStore
from .sentiment_analyzer import MultiAgentSentimentAnalyzer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BeliefAwareSentimentTracker:
    """
    Tracks sentiment in the context of agent beliefs and Theory of Mind.
    Integrates sentiment analysis with PWMK's belief reasoning system.
    """
    
    def __init__(
        self,
        belief_store: BeliefStore,
        sentiment_analyzer: MultiAgentSentimentAnalyzer,
        num_agents: int
    ):
        self.belief_store = belief_store
        self.sentiment_analyzer = sentiment_analyzer
        self.num_agents = num_agents
        
        # Track beliefs about sentiments
        self.sentiment_beliefs: Dict[str, Dict[str, Dict]] = {}
        
        # Define sentiment predicates for belief reasoning
        self._initialize_sentiment_predicates()
        
    def _initialize_sentiment_predicates(self) -> None:
        """Initialize sentiment-related predicates in belief store."""
        predicates = [
            ("current_sentiment", 2),  # current_sentiment(agent, sentiment)
            ("past_sentiment", 3),     # past_sentiment(agent, sentiment, timestep)
            ("believes_sentiment", 3), # believes_sentiment(agent_a, agent_b, sentiment)
            ("sentiment_influenced_by", 3), # sentiment_influenced_by(agent, other_agent, degree)
        ]
        
        for pred_name, arity in predicates:
            self.belief_store.add_predicate_definition(pred_name, arity)
            
        # Add sentiment reasoning rules
        rules = [
            """
            positive_interaction(AgentA, AgentB) :-
                current_sentiment(AgentA, positive),
                current_sentiment(AgentB, positive),
                interacting(AgentA, AgentB).
            """,
            """
            sentiment_conflict(AgentA, AgentB) :-
                current_sentiment(AgentA, positive),
                current_sentiment(AgentB, negative),
                interacting(AgentA, AgentB).
            """,
            """
            sentiment_alignment(AgentA, AgentB) :-
                current_sentiment(AgentA, Sentiment),
                current_sentiment(AgentB, Sentiment),
                Sentiment != neutral.
            """,
        ]
        
        for rule in rules:
            self.belief_store.add_rule(rule.strip())
            
    def update_sentiment_beliefs(
        self,
        agent_id: int,
        text: str,
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Update sentiment and corresponding beliefs.
        
        Args:
            agent_id: ID of the agent expressing sentiment
            text: Text expressing sentiment
            context: Additional context information
            
        Returns:
            Analyzed sentiment scores
        """
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_agent_communication(
            agent_id, text, context
        )
        
        # Update belief store with sentiment facts
        dominant_sentiment = max(sentiment.items(), key=lambda x: x[1])[0]
        
        # Remove old sentiment belief
        old_facts = self.belief_store.query(f"current_sentiment(agent_{agent_id}, X)")
        for fact in old_facts:
            self.belief_store.retract_fact(
                f"current_sentiment(agent_{agent_id}, {fact['X']})"
            )
            
        # Add new sentiment belief
        self.belief_store.assert_fact(
            f"current_sentiment(agent_{agent_id}, {dominant_sentiment})"
        )
        
        # Store sentiment belief with metadata
        self.sentiment_beliefs[f"agent_{agent_id}"] = {
            "sentiment": sentiment,
            "text": text,
            "context": context or {},
            "timestamp": len(self.sentiment_analyzer.agent_sentiment_history[agent_id])
        }
        
        logger.info(f"Updated sentiment beliefs for agent {agent_id}: {dominant_sentiment}")
        return sentiment
        
    def infer_sentiment_beliefs(self, observer_agent: int, target_agent: int) -> Dict[str, float]:
        """
        Infer what observer_agent believes about target_agent's sentiment.
        
        Args:
            observer_agent: Agent doing the observing
            target_agent: Agent being observed
            
        Returns:
            Inferred sentiment beliefs
        """
        # Query existing beliefs
        belief_query = f"believes(agent_{observer_agent}, current_sentiment(agent_{target_agent}, X))"
        existing_beliefs = self.belief_store.query(belief_query)
        
        if existing_beliefs:
            # Use explicit beliefs
            belief_sentiment = existing_beliefs[0]['X']
            if belief_sentiment == "positive":
                return {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
            elif belief_sentiment == "negative":
                return {"negative": 0.7, "neutral": 0.2, "positive": 0.1}
            else:  # neutral
                return {"negative": 0.2, "neutral": 0.6, "positive": 0.2}
        
        # Infer based on interaction history and observer's own sentiment patterns
        observer_history = self.sentiment_analyzer.get_agent_sentiment_history(observer_agent)
        target_history = self.sentiment_analyzer.get_agent_sentiment_history(target_agent)
        
        if not observer_history or not target_history:
            # Default to neutral belief
            return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
            
        # Simple inference: observers tend to project their own sentiment patterns
        observer_avg = self._calculate_average_sentiment(observer_history)
        target_actual = target_history[-1]["sentiment"] if target_history else observer_avg
        
        # Blend observer bias with partial observation of target
        blend_factor = 0.7  # How much observer projects vs observes
        
        inferred_sentiment = {}
        for key in observer_avg:
            inferred_sentiment[key] = (
                blend_factor * observer_avg[key] + 
                (1 - blend_factor) * target_actual[key]
            )
            
        return inferred_sentiment
        
    def _calculate_average_sentiment(self, history: List[Dict]) -> Dict[str, float]:
        """Calculate average sentiment from history."""
        if not history:
            return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
            
        avg_sentiment = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for record in history:
            for key in avg_sentiment:
                avg_sentiment[key] += record["sentiment"][key]
                
        for key in avg_sentiment:
            avg_sentiment[key] /= len(history)
            
        return avg_sentiment
        
    def detect_sentiment_misalignment(self) -> List[Dict[str, any]]:
        """
        Detect cases where agents' beliefs about others' sentiments
        don't match reality.
        
        Returns:
            List of misalignment cases
        """
        misalignments = []
        
        for observer in range(self.num_agents):
            for target in range(self.num_agents):
                if observer == target:
                    continue
                    
                # Get what observer believes about target's sentiment
                inferred_belief = self.infer_sentiment_beliefs(observer, target)
                
                # Get target's actual sentiment
                target_history = self.sentiment_analyzer.get_agent_sentiment_history(target)
                if not target_history:
                    continue
                    
                actual_sentiment = target_history[-1]["sentiment"]
                
                # Calculate misalignment (KL divergence)
                misalignment_score = self._calculate_kl_divergence(
                    inferred_belief, actual_sentiment
                )
                
                if misalignment_score > 0.5:  # Threshold for significant misalignment
                    misalignments.append({
                        "observer": observer,
                        "target": target,
                        "misalignment_score": misalignment_score,
                        "believed_sentiment": inferred_belief,
                        "actual_sentiment": actual_sentiment
                    })
                    
        return misalignments
        
    def _calculate_kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """Calculate KL divergence between two sentiment distributions."""
        kl_div = 0.0
        for key in p:
            if key in q and q[key] > 0:
                kl_div += p[key] * np.log(p[key] / q[key])
        return kl_div
        
    def query_sentiment_beliefs(self, query: str) -> List[Dict]:
        """
        Query sentiment-related beliefs using logical reasoning.
        
        Args:
            query: Prolog-style query about sentiment beliefs
            
        Returns:
            Query results
        """
        return self.belief_store.query(query)
        
    def get_group_sentiment_dynamics(self) -> Dict[str, any]:
        """
        Analyze group-level sentiment dynamics and belief patterns.
        
        Returns:
            Group sentiment analysis
        """
        # Get current group sentiment
        group_sentiment = self.sentiment_analyzer.analyze_group_sentiment()
        
        # Find sentiment conflicts
        conflicts = self.belief_store.query("sentiment_conflict(X, Y)")
        
        # Find positive interactions
        positive_interactions = self.belief_store.query("positive_interaction(X, Y)")
        
        # Find sentiment alignments
        alignments = self.belief_store.query("sentiment_alignment(X, Y)")
        
        # Detect belief misalignments
        misalignments = self.detect_sentiment_misalignment()
        
        return {
            "group_sentiment": group_sentiment,
            "conflicts": conflicts,
            "positive_interactions": positive_interactions,
            "alignments": alignments,
            "belief_misalignments": misalignments,
            "num_agents": self.num_agents
        }