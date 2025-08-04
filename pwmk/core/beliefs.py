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
        Query the belief store with enhanced pattern matching.
        
        Args:
            query_str: Query in simplified Prolog syntax
            
        Returns:
            List of variable bindings that satisfy the query
        """
        results = []
        
        # Handle nested belief queries
        if "believes(" in query_str:
            return self._query_nested_beliefs(query_str)
        
        # Extract variables (uppercase identifiers)
        variables = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', query_str)
        
        # Pattern matching against all facts
        for agent_id, facts in self.facts.items():
            for fact in facts:
                if self._matches_pattern(fact, query_str):
                    binding = self._extract_bindings(fact, query_str, variables)
                    if binding:
                        binding['agent'] = agent_id  # Add agent context
                        results.append(binding)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for result in results:
            key = tuple(sorted(result.items()))
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
                        
        return unique_results
        
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
    
    def _query_nested_beliefs(self, query_str: str) -> List[Dict[str, str]]:
        """Handle nested belief queries like believes(A, believes(B, X))."""
        results = []
        
        # Extract nested structure
        nested_match = re.search(r'believes\(([^,]+),\s*(.+)\)', query_str)
        if not nested_match:
            return results
            
        believer = nested_match.group(1).strip()
        belief_content = nested_match.group(2).strip()
        
        # Search through facts for matching beliefs
        for agent_id, facts in self.facts.items():
            if believer == agent_id or believer.isupper():  # Variable or exact match
                for fact in facts:
                    if self._matches_pattern(fact, belief_content):
                        binding = {'Believer': agent_id} if believer.isupper() else {}
                        # Extract additional variables from the belief content
                        content_vars = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', belief_content)
                        content_binding = self._extract_bindings(fact, belief_content, content_vars)
                        if content_binding:
                            binding.update(content_binding)
                            results.append(binding)
        
        return results
    
    def add_nested_belief(self, believer: str, target_agent: str, belief: str) -> None:
        """Add a nested belief (A believes that B believes X)."""
        nested_belief = f"believes({target_agent}, {belief})"
        self.add_belief(believer, nested_belief)
    
    def get_all_beliefs(self, agent_id: str) -> List[str]:
        """Get all beliefs for a specific agent."""
        return self.facts.get(agent_id, [])
    
    def clear_beliefs(self, agent_id: str) -> None:
        """Clear all beliefs for a specific agent."""
        if agent_id in self.facts:
            self.facts[agent_id] = []
            
    def belief_exists(self, agent_id: str, belief: str) -> bool:
        """Check if a specific belief exists for an agent."""
        return belief in self.facts.get(agent_id, [])