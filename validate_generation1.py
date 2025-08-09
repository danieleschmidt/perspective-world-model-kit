#!/usr/bin/env python3
"""
Generation 1 Validation: MAKE IT WORK
Simple demonstration of core PWMK functionality
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add pwmk to path
sys.path.insert(0, str(Path(__file__).parent))

def test_core_functionality():
    """Test basic functionality of core components."""
    print("🔬 Testing Core PWMK Components...")
    
    try:
        # Import core modules
        from pwmk.core.world_model import PerspectiveWorldModel
        from pwmk.core.beliefs import BeliefStore, BeliefState
        from pwmk.agents.tom_agent import ToMAgent
        from pwmk.planning.epistemic import EpistemicPlanner
        
        print("✅ Successfully imported core modules")
        
        # Test 1: World Model Creation and Forward Pass
        print("\n1️⃣ Testing PerspectiveWorldModel...")
        model = PerspectiveWorldModel(
            obs_dim=32,
            action_dim=4,
            hidden_dim=64,
            num_agents=2,
            num_layers=2
        )
        
        # Create test data
        batch_size, seq_len = 4, 10
        observations = torch.randn(batch_size, seq_len, 32)
        actions = torch.randint(0, 4, (batch_size, seq_len))
        agent_ids = torch.randint(0, 2, (batch_size, seq_len))
        
        # Forward pass
        next_states, beliefs = model(observations, actions, agent_ids)
        
        print(f"   📊 Input shape: {observations.shape}")
        print(f"   📊 Output shapes: states={next_states.shape}, beliefs={beliefs.shape}")
        print("   ✅ World model forward pass successful")
        
        # Test 2: Belief Store Operations
        print("\n2️⃣ Testing BeliefStore...")
        belief_store = BeliefStore(backend="simple")
        
        # Add basic beliefs
        belief_store.add_belief("agent_0", "has(key)")
        belief_store.add_belief("agent_0", "at(room_1)")
        belief_store.add_belief("agent_1", "at(room_2)")
        
        # Add nested belief (Theory of Mind)
        belief_store.add_nested_belief("agent_0", "agent_1", "has(treasure)")
        
        # Query beliefs
        results = belief_store.query("has(X)")
        print(f"   🔍 Query 'has(X)' results: {results}")
        
        # Test nested belief query
        nested_results = belief_store.query("believes(agent_0, has(treasure))")
        print(f"   🧠 Nested belief query results: {nested_results}")
        
        print("   ✅ Belief store operations successful")
        
        # Test 3: ToM Agent Creation
        print("\n3️⃣ Testing ToMAgent...")
        tom_agent = ToMAgent(
            agent_id="agent_0",
            world_model=model,
            tom_depth=2,
            planning_horizon=10
        )
        
        # Test belief update
        tom_agent.update_beliefs({"location": "room_3", "has_key": True})
        
        print(f"   🤖 Created ToM agent with ID: {tom_agent.agent_id}")
        print("   ✅ ToM agent creation successful")
        
        # Test 4: Epistemic Planning
        print("\n4️⃣ Testing EpistemicPlanner...")
        planner = EpistemicPlanner(
            world_model=model,
            belief_store=belief_store,
            search_depth=3
        )
        
        # Import Goal class for proper planning
        from pwmk.planning.epistemic import Goal
        
        # Simple planning test
        current_state = np.random.randn(32)
        goal = Goal(achievement="has(treasure)", epistemic=[])
        
        # Plan with limited search to avoid complexity
        plan = planner.plan(
            initial_state=current_state,
            goal=goal,
            timeout=2.0
        )
        
        print(f"   🎯 Generated plan with {len(plan.actions)} actions")
        print("   ✅ Epistemic planning successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in core functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantum_integration():
    """Test quantum-enhanced components."""
    print("\n🔬 Testing Quantum Integration...")
    
    try:
        from pwmk.quantum.quantum_planner import QuantumInspiredPlanner
        from pwmk.quantum.quantum_circuits import QuantumCircuitOptimizer
        
        # Test quantum planner
        print("\n1️⃣ Testing QuantumInspiredPlanner...")
        qplanner = QuantumInspiredPlanner(
            num_qubits=4,
            max_depth=3,
            num_agents=2
        )
        
        # Test quantum circuit optimizer
        print("\n2️⃣ Testing QuantumCircuitOptimizer...")
        qoptimizer = QuantumCircuitOptimizer(
            max_qubits=4,
            optimization_level=2
        )
        
        print("   ✅ Quantum components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in quantum integration test: {e}")
        return False

def run_demo_scenario():
    """Run a complete demo scenario."""
    print("\n🎮 Running Complete Demo Scenario...")
    
    try:
        # Set up environment-like scenario
        from pwmk.core.world_model import PerspectiveWorldModel
        from pwmk.core.beliefs import BeliefStore
        from pwmk.agents.tom_agent import ToMAgent
        
        # Create multi-agent scenario
        world_model = PerspectiveWorldModel(
            obs_dim=16,
            action_dim=3,  # left, right, pick_up
            hidden_dim=32,
            num_agents=3
        )
        
        belief_store = BeliefStore()
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = ToMAgent(
                agent_id=f"agent_{i}",
                world_model=world_model,
                tom_depth=1,
                planning_horizon=5
            )
            agents.append(agent)
        
        print(f"   🤖 Created {len(agents)} ToM agents")
        
        # Simulate scenario: agents have different knowledge
        belief_store.add_belief("agent_0", "has(key)")
        belief_store.add_belief("agent_1", "at(treasure, room_5)")
        belief_store.add_nested_belief("agent_0", "agent_1", "location(treasure, room_3)")  # False belief
        belief_store.add_nested_belief("agent_2", "agent_0", "has(key)")  # Correct belief
        
        # Multi-step interaction simulation
        for step in range(5):
            print(f"   📅 Step {step + 1}:")
            
            # Generate observations for each agent
            observations = []
            actions = []
            
            for i, agent in enumerate(agents):
                # Simple observation
                obs = torch.randn(16)
                observations.append(obs)
                
                # Agent acts based on beliefs (simplified)
                action = agent._random_action()  # Use simplified action for demo
                actions.append(action)
                
                print(f"     Agent {i}: action={action}")
            
            # Update world model with all agents' actions
            obs_batch = torch.stack(observations).unsqueeze(1)  # [3, 1, 16]
            action_batch = torch.tensor(actions).unsqueeze(1)   # [3, 1]
            agent_ids = torch.arange(3).unsqueeze(1)            # [3, 1]
            
            next_states, beliefs = world_model(obs_batch, action_batch, agent_ids)
            
            print(f"     🧠 Belief predictions shape: {beliefs.shape}")
        
        print("   ✅ Multi-agent scenario completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in demo scenario: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation script."""
    print("🚀 PWMK Generation 1 Validation: MAKE IT WORK")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Core functionality test
    if test_core_functionality():
        success_count += 1
    
    # Quantum integration test (may fail if dependencies missing)
    try:
        if test_quantum_integration():
            success_count += 1
    except ImportError:
        print("⚠️  Quantum components not fully available (dependencies missing)")
        total_tests -= 1
    
    # Demo scenario test
    if run_demo_scenario():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Validation Summary: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 Generation 1: MAKE IT WORK - SUCCESS!")
        print("   Core PWMK functionality is operational")
        return True
    else:
        print("❌ Some tests failed - needs fixing before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)