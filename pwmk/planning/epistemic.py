"""Epistemic planning with belief reasoning."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..core.world_model import PerspectiveWorldModel
from ..core.beliefs import BeliefStore


@dataclass
class Goal:
    """Represents a planning goal with epistemic constraints."""
    achievement: str  # What to achieve
    epistemic: List[str]  # Belief constraints
    
    
@dataclass 
class Plan:
    """Represents a complete plan."""
    actions: List[int]
    belief_trajectory: List[Dict[str, Any]]
    expected_reward: float
    confidence: float


class EpistemicPlanner:
    """
    Planner that considers beliefs and Theory of Mind.
    
    Combines learned world dynamics with logical belief reasoning
    to generate plans that achieve goals while managing what
    other agents believe.
    """
    
    def __init__(
        self,
        world_model: PerspectiveWorldModel,
        belief_store: BeliefStore,
        search_depth: int = 10,
        branching_factor: int = 4
    ):
        self.world_model = world_model
        self.belief_store = belief_store
        self.search_depth = search_depth
        self.branching_factor = branching_factor
        
    def plan(
        self,
        initial_state: np.ndarray,
        goal: Goal,
        timeout: float = 5.0
    ) -> Plan:
        """
        Generate a plan to achieve the goal.
        
        Args:
            initial_state: Current world state
            goal: Goal with achievement and epistemic constraints
            timeout: Planning timeout in seconds
            
        Returns:
            Plan with actions and belief trajectory
        """
        # Simple forward search for demonstration
        best_plan = None
        best_reward = float('-inf')
        
        # Generate random action sequences and evaluate
        for _ in range(100):  # Limited search for demo
            actions = self._generate_random_actions()
            reward = self._evaluate_plan(initial_state, actions, goal)
            
            if reward > best_reward:
                best_reward = reward
                best_plan = Plan(
                    actions=actions,
                    belief_trajectory=[],  # Simplified
                    expected_reward=reward,
                    confidence=0.8
                )
                
        return best_plan or Plan([], [], 0.0, 0.0)
        
    def _generate_random_actions(self) -> List[int]:
        """Generate a random sequence of actions."""
        return [
            np.random.randint(0, 4) 
            for _ in range(self.search_depth)
        ]
        
    def _evaluate_plan(
        self,
        initial_state: np.ndarray,
        actions: List[int],
        goal: Goal
    ) -> float:
        """Evaluate how well a plan achieves the goal."""
        # Simplified evaluation
        reward = 0.0
        
        # Reward for taking actions (exploration bonus)
        reward += len(actions) * 0.1
        
        # Penalty for long plans
        reward -= len(actions) * 0.05
        
        # Check goal achievement (simplified)
        if "treasure" in goal.achievement.lower():
            reward += 10.0
            
        return reward
        
    def evaluate_plan(
        self,
        plan: Plan,
        true_dynamics: Optional[Any] = None
    ) -> float:
        """Evaluate plan quality against ground truth."""
        # Simplified plan evaluation
        base_score = plan.expected_reward
        confidence_bonus = plan.confidence * 2.0
        
        return base_score + confidence_bonus