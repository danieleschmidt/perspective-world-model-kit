#!/usr/bin/env python3
"""
Perspective World Model Kit (PWMK) Demo

This demo showcases the core functionality of PWMK:
- Multi-agent environment with partial observability
- Neural world model with perspective-aware learning
- Symbolic belief reasoning and Theory of Mind
- Epistemic planning considering agent beliefs
"""

import torch
import numpy as np
from pwmk import PerspectiveWorldModel, BeliefStore, EpistemicPlanner, ToMAgent
from pwmk.envs import SimpleGridWorld
from pwmk.planning.epistemic import Goal


def demo_basic_functionality():
    """Demonstrate basic PWMK functionality."""
    print("🎯 PWMK Demo: Basic Functionality")
    print("=" * 50)
    
    # 1. Environment Setup
    print("\n1. Setting up multi-agent environment")
    env = SimpleGridWorld(grid_size=6, num_agents=2, view_radius=2, num_treasures=1, num_keys=1)
    observations, info = env.reset(seed=42)
    print(f"   ✓ Environment created with {env.num_agents} agents")
    print(f"   ✓ Grid size: {env.grid_size}x{env.grid_size}")
    print(f"   ✓ Observation space: {env.observation_space.shape}")
    
    # Display initial state
    print("\n   Initial environment:")
    env.render(mode="human")
    
    # 2. World Model
    print("\n2. Creating perspective-aware world model")
    obs_dim = observations[0].shape[0]
    world_model = PerspectiveWorldModel(
        obs_dim=obs_dim,
        action_dim=5,  # 4 movement + 1 pickup
        hidden_dim=128,
        num_agents=2
    )
    print(f"   ✓ World model created with {obs_dim}-dim observations")
    
    # Test forward pass
    obs_tensor = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in observations])
    obs_tensor = obs_tensor.unsqueeze(1)  # Add sequence dimension
    actions_tensor = torch.randint(0, 5, (2, 1))  # Random actions
    
    with torch.no_grad():
        next_states, beliefs = world_model(obs_tensor, actions_tensor)
    
    print(f"   ✓ Forward pass successful: states {next_states.shape}, beliefs {beliefs.shape}")
    
    # 3. Belief Store
    print("\n3. Demonstrating belief reasoning")
    belief_store = BeliefStore()
    
    # Add some beliefs
    belief_store.add_belief("agent_0", "has(agent_1, key)")
    belief_store.add_belief("agent_0", "at(treasure, room_3)")
    belief_store.add_belief("agent_1", "believes(agent_0, location(treasure, room_3))")
    
    print("   ✓ Added beliefs to store")
    
    # Query beliefs
    results = belief_store.query("has(X, key)")
    print(f"   ✓ Query 'has(X, key)' found {len(results)} results: {results}")
    
    # Test nested beliefs
    belief_store.add_nested_belief("agent_0", "agent_1", "safe(room_2)")
    nested_results = belief_store.query("believes(agent_0, believes(agent_1, safe(X)))")
    print(f"   ✓ Nested belief query found {len(nested_results)} results")
    
    # 4. Theory of Mind Agents
    print("\n4. Creating Theory of Mind agents")
    agents = []
    for i in range(2):
        agent = ToMAgent(
            agent_id=f"agent_{i}",
            world_model=world_model,
            tom_depth=2
        )
        agents.append(agent)
    
    # Set goals
    agents[0].set_goal("has(agent_0, treasure)", ["believes(agent_1, safe(agent_0))"])
    agents[1].set_goal("has(agent_1, key)", [])
    
    print("   ✓ Created 2 ToM agents with goals")
    
    # 5. Epistemic Planning
    print("\n5. Demonstrating epistemic planning")
    planner = EpistemicPlanner(
        world_model=world_model,
        belief_store=belief_store,
        search_depth=5
    )
    
    goal = Goal(
        achievement="has(agent_0, treasure)",
        epistemic=["believes(agent_1, at(agent_0, safe_location))"]
    )
    
    initial_state = np.random.randn(128)
    plan = planner.plan(initial_state, goal, timeout=2.0)
    
    print(f"   ✓ Generated plan with {len(plan.actions)} actions")
    print(f"   ✓ Expected reward: {plan.expected_reward:.2f}")
    print(f"   ✓ Confidence: {plan.confidence:.2f}")
    print(f"   ✓ Actions: {plan.actions}")
    
    # 6. Multi-Agent Simulation
    print("\n6. Running multi-agent simulation")
    
    for step in range(8):
        print(f"\n   Step {step + 1}:")
        
        # Get actions from ToM agents
        actions = []
        for i, agent in enumerate(agents):
            obs_dict = {
                "position": f"({env.agent_positions[i][0]}, {env.agent_positions[i][1]})",
                "inventory": env.agent_inventories[i],
                "step": step
            }
            action = agent.act_with_tom(obs_dict)
            actions.append(action)
            
        print(f"      Agent actions: {actions}")
        
        # Environment step
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        print(f"      Rewards: {rewards}")
        print(f"      Agent positions: {env.agent_positions}")
        print(f"      Inventories: {env.agent_inventories}")
        
        # Show environment
        if step % 3 == 0:  # Show every 3 steps
            print("      Environment state:")
            env.render(mode="human")
        
        if terminated or truncated:
            print(f"   ✓ Simulation ended at step {step + 1}")
            break
    
    # Get final belief ground truth
    beliefs = env.get_belief_ground_truth()
    print(f"\n   Final belief ground truth:")
    for agent_id, agent_beliefs in beliefs.items():
        print(f"   {agent_id}: {len(agent_beliefs)} beliefs")
    
    print("\n✅ Demo completed successfully!")


def demo_belief_reasoning():
    """Demonstrate advanced belief reasoning capabilities."""
    print("\n🧠 PWMK Demo: Advanced Belief Reasoning")
    print("=" * 50)
    
    store = BeliefStore()
    
    # Scenario: Multi-agent treasure hunt
    print("\nScenario: Multi-agent treasure hunt with hidden information")
    
    # Agent 0 beliefs
    store.add_belief("agent_0", "has(agent_0, map)")
    store.add_belief("agent_0", "at(treasure, room_5)")
    store.add_belief("agent_0", "has(agent_1, key)")
    
    # Agent 1 beliefs
    store.add_belief("agent_1", "at(agent_0, room_2)")
    store.add_belief("agent_1", "has(agent_1, key)")
    
    # Nested beliefs (Theory of Mind)
    store.add_nested_belief("agent_0", "agent_1", "location(treasure, room_3)")  # Agent 0 thinks Agent 1 believes treasure is in room 3
    store.add_nested_belief("agent_1", "agent_0", "has(agent_0, map)")  # Agent 1 thinks Agent 0 has a map
    
    print("\n✓ Populated belief store with complex beliefs")
    
    # Demonstrate queries
    queries = [
        "has(X, map)",
        "at(treasure, X)",
        "believes(agent_0, believes(agent_1, location(treasure, X)))",
        "has(X, key)"
    ]
    
    for query in queries:
        results = store.query(query)
        print(f"\nQuery: {query}")
        print(f"Results: {results}")
        
    print("\n✅ Belief reasoning demo completed!")


if __name__ == "__main__":
    print("🚀 Perspective World Model Kit (PWMK) Demo")
    print("Neuro-symbolic AI with Theory of Mind capabilities")
    print("Version 0.1.0")
    
    try:
        demo_basic_functionality()
        demo_belief_reasoning()
        
        print("\n🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("- Explore the examples/ directory for more advanced usage")
        print("- Check out the documentation at docs/")
        print("- Run the test suite with: pytest tests/")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise