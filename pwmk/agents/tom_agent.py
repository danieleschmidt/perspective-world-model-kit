"""Theory of Mind agent implementation."""

from typing import Dict, List, Any, Optional
import numpy as np

from ..core.world_model import PerspectiveWorldModel
from ..core.beliefs import BeliefStore
from ..planning.epistemic import EpistemicPlanner, Goal


class ToMAgent:
    """
    Agent with Theory of Mind capabilities.
    
    Maintains beliefs about other agents and uses epistemic planning
    to achieve goals while considering what others know and believe.
    """
    
    def __init__(
        self,
        agent_id: str,
        world_model: PerspectiveWorldModel,
        tom_depth: int = 2,
        planning_horizon: int = 10
    ):
        self.agent_id = agent_id
        self.world_model = world_model
        self.tom_depth = tom_depth
        self.planning_horizon = planning_horizon
        
        # Initialize belief store and planner
        self.belief_store = BeliefStore()
        self.planner = EpistemicPlanner(
            world_model=world_model,
            belief_store=self.belief_store,
            search_depth=planning_horizon
        )
        
        # Agent state
        self.current_beliefs = {}
        self.goal_stack = []
        
    def update_beliefs(self, observation: Dict[str, Any]) -> None:
        """Update beliefs based on new observation."""
        # Extract relevant information from observation
        self.belief_store.update_beliefs(self.agent_id, observation)
        
        # Update beliefs about other agents
        self._update_tom_beliefs(observation)
        
    def set_goal(self, achievement: str, epistemic_constraints: List[str] = None) -> None:
        """Set a new goal for the agent."""
        goal = Goal(
            achievement=achievement,
            epistemic=epistemic_constraints or []
        )
        self.goal_stack.append(goal)
        
    def act_with_tom(self, observation: Dict[str, Any] = None) -> int:
        """
        Select action considering Theory of Mind.
        
        Args:
            observation: Current observation (optional)
            
        Returns:
            Selected action
        """
        if observation:
            self.update_beliefs(observation)
            
        # Get current goal
        if not self.goal_stack:
            return self._random_action()
            
        current_goal = self.goal_stack[-1]
        
        # Plan considering beliefs
        current_state = self._get_current_state()
        plan = self.planner.plan(
            initial_state=current_state,
            goal=current_goal
        )
        
        # Return first action from plan
        if plan.actions:
            return plan.actions[0]
        else:
            return self._random_action()
            
    def get_beliefs_about(self, other_agent_id: str) -> Dict[str, Any]:
        """Get beliefs about another agent."""
        belief_state = self.belief_store.get_belief_state(other_agent_id)
        return belief_state.beliefs if belief_state else {}
        
    def reason_about_beliefs(self, query: str) -> List[Dict[str, str]]:
        """Reason about beliefs using the belief store."""
        return self.belief_store.query(query)
        
    def _update_tom_beliefs(self, observation: Dict[str, Any]) -> None:
        """Update Theory of Mind beliefs about other agents."""
        # Simplified ToM belief update
        for key, value in observation.items():
            if "agent_" in str(value):
                # This is information about another agent
                belief = f"observed({key}, {value})"
                self.belief_store.add_belief(self.agent_id, belief)
                
                # Add nested belief (what we think they know)
                nested_belief = f"believes({value}, aware_of({key}))"
                self.belief_store.add_belief(self.agent_id, nested_belief)
                
    def _get_current_state(self) -> np.ndarray:
        """Get current state representation."""
        # Simplified state representation
        return np.random.randn(64)  # Placeholder
        
    def _random_action(self) -> int:
        """Select a random action."""
        return np.random.randint(0, 4)  # 4 possible actions