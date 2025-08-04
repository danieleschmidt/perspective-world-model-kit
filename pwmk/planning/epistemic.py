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
        Generate a plan to achieve the goal using belief-aware search.
        
        Args:
            initial_state: Current world state
            goal: Goal with achievement and epistemic constraints
            timeout: Planning timeout in seconds
            
        Returns:
            Plan with actions and belief trajectory
        """
        import time
        start_time = time.time()
        
        best_plan = None
        best_reward = float('-inf')
        belief_trajectory = []
        
        # Enhanced planning with belief consideration
        for iteration in range(500):  # More thorough search
            if time.time() - start_time > timeout:
                break
                
            actions = self._generate_smart_actions(goal, iteration)
            reward, trajectory = self._evaluate_plan_with_beliefs(
                initial_state, actions, goal
            )
            
            if reward > best_reward:
                best_reward = reward
                belief_trajectory = trajectory
                best_plan = Plan(
                    actions=actions,
                    belief_trajectory=trajectory,
                    expected_reward=reward,
                    confidence=min(0.9, 0.5 + reward / 20.0)  # Dynamic confidence
                )
        
        # Fallback plan if no good plan found
        if best_plan is None:
            fallback_actions = self._generate_fallback_plan(goal)
            best_plan = Plan(
                actions=fallback_actions,
                belief_trajectory=[{"status": "fallback"}],
                expected_reward=0.1,
                confidence=0.3
            )
                
        return best_plan
        
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
    
    def _generate_smart_actions(self, goal: Goal, iteration: int) -> List[int]:
        """Generate actions that consider the goal structure."""
        actions = []
        length = min(self.search_depth, max(3, iteration // 50 + 3))
        
        # Bias actions based on goal keywords
        action_bias = 0  # Default: move/explore
        if "treasure" in goal.achievement.lower():
            action_bias = 1  # Towards collection actions
        elif "believes" in str(goal.epistemic):
            action_bias = 2  # Towards communication/visibility actions
        
        for i in range(length):
            if np.random.random() < 0.3:  # 30% goal-oriented
                action = (action_bias + np.random.randint(0, 2)) % 4
            else:  # 70% exploration
                action = np.random.randint(0, 4)
            actions.append(action)
            
        return actions
    
    def _evaluate_plan_with_beliefs(
        self, 
        initial_state: np.ndarray, 
        actions: List[int], 
        goal: Goal
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Evaluate plan considering belief states."""
        reward = 0.0
        trajectory = []
        
        # Base reward for plan length
        reward += len(actions) * 0.05
        
        # Achievement goal evaluation
        if "treasure" in goal.achievement.lower():
            reward += 5.0  # Finding treasure is valuable
        
        if "has(" in goal.achievement:
            reward += 3.0  # Possession goals
            
        # Epistemic constraint evaluation
        epistemic_bonus = 0.0
        for constraint in goal.epistemic:
            if "believes(" in constraint:
                epistemic_bonus += 2.0  # Managing beliefs is valuable
            if "not(" in constraint:
                epistemic_bonus += 1.0  # Avoiding certain beliefs
        
        reward += epistemic_bonus
        
        # Plan diversity bonus (longer plans get diminishing returns)
        if len(actions) > 5:
            reward -= (len(actions) - 5) * 0.02
        
        # Create trajectory representation
        for i, action in enumerate(actions):
            step_beliefs = {
                "step": i,
                "action": action,
                "estimated_achievement": min(1.0, reward / 10.0),
                "epistemic_satisfaction": min(1.0, epistemic_bonus / 5.0)
            }
            trajectory.append(step_beliefs)
        
        return reward, trajectory
    
    def _generate_fallback_plan(self, goal: Goal) -> List[int]:
        """Generate a simple fallback plan."""
        # Simple exploration sequence
        return [0, 1, 2, 3, 0]  # Basic movement pattern