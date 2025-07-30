"""Belief store and reasoning system."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class BeliefState:
    """Represents the belief state of an agent."""
    agent_id: str
    beliefs: Dict[str, Any]
    confidence: Dict[str, float]
    timestamp: int


class BeliefStore:
    """
    Manages beliefs and provides reasoning capabilities.
    
    Supports first-order and nested belief reasoning using
    a simplified Prolog-like interface.
    """
    
    def __init__(self, backend: str = "simple"):
        self.backend = backend
        self.facts: Dict[str, List[str]] = {}
        self.rules: List[str] = []
        self.beliefs: Dict[str, BeliefState] = {}
        
    def add_belief(self, agent_id: str, belief: str) -> None:
        """Add a belief for a specific agent."""
        if agent_id not in self.facts:
            self.facts[agent_id] = []
        self.facts[agent_id].append(belief)
        
    def add_rule(self, rule: str) -> None:
        """Add a reasoning rule."""
        self.rules.append(rule)
        
    def query(self, query_str: str) -> List[Dict[str, str]]:
        """
        Query the belief store.
        
        Args:
            query_str: Query in simplified Prolog syntax
            
        Returns:
            List of variable bindings that satisfy the query
        """
        # Simplified query processing
        results = []
        
        # Extract variables (uppercase identifiers)
        variables = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', query_str)
        
        # Simple pattern matching against facts
        for agent_id, facts in self.facts.items():
            for fact in facts:
                if self._matches_pattern(fact, query_str):
                    binding = self._extract_bindings(fact, query_str, variables)
                    if binding:
                        results.append(binding)
                        
        return results
        
    def get_belief_state(self, agent_id: str) -> Optional[BeliefState]:
        """Get the current belief state for an agent."""
        if agent_id in self.beliefs:
            return self.beliefs[agent_id]
        return None
        
    def update_beliefs(
        self, 
        agent_id: str, 
        observations: Dict[str, Any]
    ) -> None:
        """Update agent beliefs based on new observations."""
        # Simple belief update mechanism
        for key, value in observations.items():
            belief = f"{key}({value})"
            self.add_belief(agent_id, belief)
            
    def _matches_pattern(self, fact: str, pattern: str) -> bool:
        """Check if a fact matches a query pattern."""
        # Simplified pattern matching
        fact_clean = re.sub(r'\([^)]*\)', '(X)', fact)
        pattern_clean = re.sub(r'\([^)]*\)', '(X)', pattern)
        return fact_clean == pattern_clean
        
    def _extract_bindings(
        self, 
        fact: str, 
        pattern: str, 
        variables: List[str]
    ) -> Optional[Dict[str, str]]:
        """Extract variable bindings from a matching fact."""
        # Simplified binding extraction
        binding = {}
        
        # Extract terms from fact and pattern
        fact_terms = re.findall(r'\(([^)]*)\)', fact)
        pattern_terms = re.findall(r'\(([^)]*)\)', pattern)
        
        if len(fact_terms) == len(pattern_terms):
            for i, (fact_term, pattern_term) in enumerate(zip(fact_terms, pattern_terms)):
                if pattern_term in variables:
                    binding[pattern_term] = fact_term
                    
        return binding if binding else None