"""Belief content validation and consistency checking."""

import re
from typing import List, Dict, Set, Tuple, Optional
from ..utils.logging import get_logger
from .input_sanitizer import SecurityError


class BeliefValidator:
    """Validates belief content and maintains logical consistency."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.valid_predicates: Set[str] = {
            'has', 'at', 'believes', 'knows', 'sees', 'location', 
            'can_reach', 'can_unlock', 'safe', 'dangerous', 'path',
            'connects', 'implies', 'not', 'position', 'visible_agents'
        }
        self.max_nesting_depth = 5  # Limit belief nesting depth
    
    def validate_belief_syntax(self, belief: str) -> bool:
        """Validate belief syntax using simplified Prolog rules."""
        if not belief.strip():
            return False
        
        # Check for balanced parentheses
        if belief.count('(') != belief.count(')'):
            self.logger.warning(f"Unbalanced parentheses in belief: {belief}")
            return False
        
        # Check predicate format - allow basic predicates, nested beliefs, and negation
        predicate_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)$|^believes\([^,]+,\s*.+\)$|^not\([^)]+\)$'
        if not re.match(predicate_pattern, belief.strip()):
            # Allow compound beliefs with logical operators
            compound_pattern = r'^.+(AND|OR|implies)\s+.+$'
            if not re.match(compound_pattern, belief.strip(), re.IGNORECASE):
                self.logger.debug(f"Belief syntax check failed for: {belief}")
                return False
        
        return True
    
    def validate_predicate_name(self, predicate: str) -> bool:
        """Validate that predicate name is in allowed set."""
        if predicate.lower() in self.valid_predicates:
            return True
        
        # Allow dynamic predicates matching pattern
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', predicate):
            self.logger.info(f"Adding new predicate to allowed set: {predicate}")
            self.valid_predicates.add(predicate.lower())
            return True
        
        return False
    
    def extract_predicates(self, belief: str) -> List[str]:
        """Extract predicate names from a belief."""
        predicates = []
        
        # Find all predicate calls
        predicate_matches = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', belief)
        for pred in predicate_matches:
            if self.validate_predicate_name(pred):
                predicates.append(pred)
        
        return predicates
    
    def check_nesting_depth(self, belief: str) -> bool:
        """Check if belief nesting doesn't exceed maximum depth."""
        depth = 0
        max_depth = 0
        
        for char in belief:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
        
        if max_depth > self.max_nesting_depth:
            self.logger.warning(f"Belief nesting too deep ({max_depth} > {self.max_nesting_depth}): {belief}")
            return False
        
        return True
    
    def detect_contradictions(self, beliefs: List[str]) -> List[Tuple[str, str]]:
        """Detect basic logical contradictions in belief set."""
        contradictions = []
        
        # Simple contradiction detection
        positive_facts = set()
        negative_facts = set()
        
        for belief in beliefs:
            if belief.strip().startswith('not('):
                # Extract negated fact
                negated = re.search(r'not\((.+)\)', belief)
                if negated:
                    negative_facts.add(negated.group(1).strip())
            else:
                positive_facts.add(belief.strip())
        
        # Find contradictions
        for pos_fact in positive_facts:
            if pos_fact in negative_facts:
                contradictions.append((pos_fact, f"not({pos_fact})"))
        
        return contradictions
    
    def validate_agent_reference(self, agent_id: str) -> bool:
        """Validate agent reference in beliefs."""
        if not isinstance(agent_id, str):
            return False
        
        # Agent ID should be alphanumeric with underscores
        if not re.match(r'^[a-zA-Z0-9_]+$', agent_id):
            return False
        
        # Reasonable length limits
        if len(agent_id) > 50:
            return False
        
        return True
    
    def sanitize_and_validate(self, belief: str, agent_id: str) -> str:
        """Comprehensive belief validation and sanitization."""
        # Basic security checks
        if not isinstance(belief, str) or not isinstance(agent_id, str):
            raise SecurityError("Belief and agent_id must be strings")
        
        # Validate agent reference
        if not self.validate_agent_reference(agent_id):
            raise SecurityError(f"Invalid agent reference: {agent_id}")
        
        # Syntax validation
        if not self.validate_belief_syntax(belief):
            raise SecurityError(f"Invalid belief syntax: {belief}")
        
        # Check nesting depth
        if not self.check_nesting_depth(belief):
            raise SecurityError(f"Belief nesting too deep: {belief}")
        
        # Validate predicates
        predicates = self.extract_predicates(belief)
        for pred in predicates:
            if not self.validate_predicate_name(pred):
                raise SecurityError(f"Invalid predicate name: {pred}")
        
        self.logger.debug(f"Validated belief for {agent_id}: {belief}")
        return belief


# Global validator instance
_validator = None

def get_validator() -> BeliefValidator:
    """Get global belief validator instance."""
    global _validator
    if _validator is None:
        _validator = BeliefValidator()
    return _validator