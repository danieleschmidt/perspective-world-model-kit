"""Simple demo interface for quick getting started with PWMK."""

from typing import Dict, List, Optional, Any
import numpy as np
from ..core.world_model import PerspectiveWorldModel
from ..core.beliefs import BeliefStore, BeliefState
from ..envs.simple_grid import SimpleGridWorld


class QuickStartDemo:
    """
    Simple demo interface that gets users up and running quickly.
    
    Provides one-command demos of core PWMK functionality without
    complex configuration.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._world_model = None
        self._belief_store = None
        self._env = None
        
    def run_basic_demo(self) -> Dict[str, Any]:
        """Run a basic world model demo with default settings."""
        if self.verbose:
            print("🚀 Starting PWMK Basic Demo...")
            
        # Create simple environment
        self._env = SimpleGridWorld(size=5, num_agents=2)
        
        # Initialize world model
        obs_dim = self._env.observation_space.shape[0]
        action_dim = self._env.action_space.n
        
        self._world_model = PerspectiveWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            num_agents=2
        )
        
        # Initialize belief store
        self._belief_store = BeliefStore()
        
        # Run simple episode
        obs = self._env.reset()
        total_reward = 0
        actions_taken = []
        
        for step in range(10):
            # Simple random actions for demo
            actions = [self._env.action_space.sample() for _ in range(2)]
            obs, rewards, done, info = self._env.step(actions)
            
            total_reward += sum(rewards)
            actions_taken.append(actions)
            
            # Add simple belief
            self._belief_store.add_belief(
                f"agent_0", 
                f"step_{step}_action_{actions[0]}"
            )
            
            if done:
                break
                
        if self.verbose:
            print(f"✅ Demo completed! Total reward: {total_reward}")
            print(f"📊 Actions taken: {len(actions_taken)} steps")
            
        return {
            "total_reward": total_reward,
            "steps": len(actions_taken),
            "final_obs": obs,
            "beliefs_count": len(self._belief_store.get_all_beliefs())
        }
    
    def run_belief_demo(self) -> Dict[str, Any]:
        """Demonstrate belief reasoning capabilities."""
        if self.verbose:
            print("🧠 Starting Belief Reasoning Demo...")
            
        belief_store = BeliefStore()
        
        # Add sample beliefs
        belief_store.add_belief("agent_0", "has_key")
        belief_store.add_belief("agent_0", "at_door") 
        belief_store.add_belief("agent_1", "believes_agent_0_has_key")
        
        # Query beliefs
        agent_0_beliefs = belief_store.get_beliefs("agent_0")
        agent_1_beliefs = belief_store.get_beliefs("agent_1")
        
        if self.verbose:
            print(f"🔍 Agent 0 beliefs: {agent_0_beliefs}")
            print(f"🔍 Agent 1 beliefs: {agent_1_beliefs}")
            print("✅ Belief demo completed!")
            
        return {
            "agent_0_beliefs": agent_0_beliefs,
            "agent_1_beliefs": agent_1_beliefs,
            "total_beliefs": len(belief_store.get_all_beliefs())
        }
    
    def run_theory_of_mind_demo(self) -> Dict[str, Any]:
        """Demonstrate Theory of Mind reasoning."""
        if self.verbose:
            print("🎭 Starting Theory of Mind Demo...")
            
        belief_store = BeliefStore()
        
        # Create nested belief scenario
        belief_store.add_belief("agent_0", "has_treasure")
        belief_store.add_belief("agent_1", "believes_agent_0_lost_treasure")
        belief_store.add_belief("agent_0", "knows_agent_1_believes_lost_treasure")
        
        # This demonstrates the classic ToM false belief scenario
        beliefs = belief_store.get_all_beliefs()
        
        # Simple ToM reasoning
        tom_scenario = {
            "reality": "agent_0 has treasure",
            "agent_1_belief": "agent_0 lost treasure", 
            "agent_0_knows": "agent_1 has false belief"
        }
        
        if self.verbose:
            print("🎭 Theory of Mind Scenario:")
            print(f"   Reality: {tom_scenario['reality']}")
            print(f"   Agent 1 believes: {tom_scenario['agent_1_belief']}")
            print(f"   Agent 0 knows: {tom_scenario['agent_0_knows']}")
            print("✅ Theory of Mind demo completed!")
            
        return {
            "scenario": tom_scenario,
            "beliefs": beliefs,
            "tom_levels": 2  # Agent 0 knows what Agent 1 believes
        }


def quick_demo() -> Dict[str, Any]:
    """Run all demos with a single function call."""
    print("🌟 PWMK Quick Demo Suite")
    print("=" * 40)
    
    demo = QuickStartDemo(verbose=True)
    
    results = {}
    
    # Run basic demo
    results["basic"] = demo.run_basic_demo()
    print()
    
    # Run belief demo  
    results["beliefs"] = demo.run_belief_demo()
    print()
    
    # Run ToM demo
    results["theory_of_mind"] = demo.run_theory_of_mind_demo()
    print()
    
    print("🎉 All demos completed successfully!")
    print("Visit the full documentation for advanced features.")
    
    return results


if __name__ == "__main__":
    quick_demo()