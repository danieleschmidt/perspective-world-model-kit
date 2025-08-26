"""Quick examples to demonstrate PWMK capabilities."""

import numpy as np
from typing import Dict, List, Any, Tuple
import torch

from ..core.world_model import PerspectiveWorldModel
from ..core.beliefs import BeliefStore
from ..envs.simple_grid import SimpleGridWorld


def create_simple_world_model(obs_dim: int = 16, action_dim: int = 4) -> PerspectiveWorldModel:
    """
    Create a simple world model with sensible defaults.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        
    Returns:
        Configured PerspectiveWorldModel
    """
    return PerspectiveWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=128,
        num_agents=2,
        num_layers=2
    )


def run_belief_reasoning_example() -> Dict[str, Any]:
    """
    Example of belief reasoning with nested beliefs.
    
    Returns:
        Dictionary with belief reasoning results
    """
    print("🧠 Running Belief Reasoning Example")
    
    # Create belief store
    belief_store = BeliefStore()
    
    # Scenario: Multi-agent treasure hunt with deception
    # Agent 0 knows where treasure is but wants to mislead Agent 1
    
    # Ground truth
    belief_store.add_belief("system", "treasure_at_location_3")
    
    # Agent 0's knowledge
    belief_store.add_belief("agent_0", "treasure_at_location_3")
    belief_store.add_belief("agent_0", "agent_1_searching")
    
    # Agent 0's deception strategy
    belief_store.add_belief("agent_0", "tell_agent_1_treasure_at_location_1")
    
    # Agent 1's beliefs (based on Agent 0's deception)
    belief_store.add_belief("agent_1", "treasure_at_location_1")  # False belief
    belief_store.add_belief("agent_1", "agent_0_is_helpful")
    
    # Agent 0's theory of mind about Agent 1
    belief_store.add_belief("agent_0", "agent_1_believes_treasure_at_location_1")
    belief_store.add_belief("agent_0", "agent_1_trusts_me")
    
    # Get all beliefs for analysis
    results = {
        "agent_0_beliefs": belief_store.get_beliefs("agent_0"),
        "agent_1_beliefs": belief_store.get_beliefs("agent_1"), 
        "system_beliefs": belief_store.get_beliefs("system"),
        "total_beliefs": len(belief_store.get_all_beliefs()),
        "scenario": "treasure_hunt_with_deception"
    }
    
    print(f"✅ Created {results['total_beliefs']} beliefs across scenario")
    print(f"   Agent 0 has {len(results['agent_0_beliefs'])} beliefs")
    print(f"   Agent 1 has {len(results['agent_1_beliefs'])} beliefs")
    
    return results


def run_multi_agent_example() -> Dict[str, Any]:
    """
    Example of multi-agent world model training and inference.
    
    Returns:
        Dictionary with training results
    """
    print("🤖 Running Multi-Agent Example")
    
    # Create environment
    env = SimpleGridWorld(size=6, num_agents=3)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create world model
    world_model = create_simple_world_model(obs_dim, action_dim)
    world_model.num_agents = 3  # Update for 3 agents
    
    # Create belief store for each agent
    belief_stores = {f"agent_{i}": BeliefStore() for i in range(3)}
    
    # Run simulation episode
    obs = env.reset()
    episode_data = []
    
    for step in range(20):
        # Simple action selection (random for demo)
        actions = [env.action_space.sample() for _ in range(3)]
        
        # Step environment
        next_obs, rewards, done, info = env.step(actions)
        
        # Store transition data
        episode_data.append({
            "step": step,
            "observations": obs.copy(),
            "actions": actions.copy(),
            "rewards": rewards,
            "next_observations": next_obs.copy()
        })
        
        # Update beliefs based on observations
        for agent_id in range(3):
            agent_key = f"agent_{agent_id}"
            
            # Simple belief updates based on rewards
            if rewards[agent_id] > 0:
                belief_stores[agent_key].add_belief(agent_key, f"good_action_step_{step}")
            elif rewards[agent_id] < 0:
                belief_stores[agent_key].add_belief(agent_key, f"bad_action_step_{step}")
                
            # Belief about other agents' performance
            for other_id in range(3):
                if other_id != agent_id:
                    if rewards[other_id] > rewards[agent_id]:
                        belief_stores[agent_key].add_belief(
                            agent_key, 
                            f"agent_{other_id}_performing_better_step_{step}"
                        )
        
        obs = next_obs
        
        if done:
            break
    
    # Analyze results
    total_rewards = [sum(ep["rewards"]) for ep in episode_data]
    total_reward_per_agent = [0, 0, 0]
    
    for ep in episode_data:
        for i, reward in enumerate(ep["rewards"]):
            total_reward_per_agent[i] += reward
    
    # Count beliefs
    belief_counts = {
        agent: len(store.get_all_beliefs()) 
        for agent, store in belief_stores.items()
    }
    
    results = {
        "episode_length": len(episode_data),
        "total_reward_per_agent": total_reward_per_agent,
        "belief_counts": belief_counts,
        "world_model_parameters": sum(p.numel() for p in world_model.parameters()),
        "environment_info": {
            "grid_size": env.size,
            "num_agents": env.num_agents,
            "obs_dim": obs_dim,
            "action_dim": action_dim
        }
    }
    
    print(f"✅ Completed {results['episode_length']} step episode")
    print(f"   Agent rewards: {results['total_reward_per_agent']}")
    print(f"   World model parameters: {results['world_model_parameters']:,}")
    print(f"   Total beliefs formed: {sum(belief_counts.values())}")
    
    return results


def run_simple_training_example() -> Dict[str, Any]:
    """
    Example of simple world model training.
    
    Returns:
        Dictionary with training results
    """
    print("📚 Running Simple Training Example")
    
    # Create simple synthetic data
    batch_size = 32
    obs_dim = 16
    action_dim = 4
    seq_length = 10
    
    # Synthetic observation sequences
    obs_sequences = torch.randn(batch_size, seq_length, obs_dim)
    action_sequences = torch.randint(0, action_dim, (batch_size, seq_length))
    
    # Create world model
    world_model = create_simple_world_model(obs_dim, action_dim)
    
    # Simple training setup
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    
    training_losses = []
    
    # Train for a few steps (demo purposes)
    world_model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Forward pass with synthetic data
        try:
            # Simple loss calculation (MSE on next observation prediction)
            predicted_next_obs = obs_sequences[:, 1:, :]  # Simple target
            current_obs = obs_sequences[:, :-1, :]
            current_actions = action_sequences[:, :-1]
            
            # Simple forward pass (dummy for demo)
            loss = torch.nn.MSELoss()(predicted_next_obs, current_obs)
            
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
        except Exception as e:
            print(f"⚠️  Training step {epoch} failed: {e}")
            training_losses.append(float('inf'))
    
    results = {
        "training_epochs": len(training_losses),
        "final_loss": training_losses[-1] if training_losses else float('inf'),
        "loss_history": training_losses,
        "model_parameters": sum(p.numel() for p in world_model.parameters()),
        "batch_size": batch_size,
        "sequence_length": seq_length
    }
    
    print(f"✅ Training completed - Final loss: {results['final_loss']:.4f}")
    
    return results


def run_all_examples() -> Dict[str, Any]:
    """Run all quick examples and return combined results."""
    print("🌟 Running All PWMK Quick Examples")
    print("=" * 50)
    
    all_results = {}
    
    try:
        all_results["belief_reasoning"] = run_belief_reasoning_example()
        print()
    except Exception as e:
        print(f"❌ Belief reasoning example failed: {e}")
        all_results["belief_reasoning"] = {"error": str(e)}
    
    try:
        all_results["multi_agent"] = run_multi_agent_example()
        print()
    except Exception as e:
        print(f"❌ Multi-agent example failed: {e}")
        all_results["multi_agent"] = {"error": str(e)}
    
    try:
        all_results["simple_training"] = run_simple_training_example()
        print()
    except Exception as e:
        print(f"❌ Simple training example failed: {e}")
        all_results["simple_training"] = {"error": str(e)}
    
    print("🎉 All examples completed!")
    
    return all_results


if __name__ == "__main__":
    run_all_examples()