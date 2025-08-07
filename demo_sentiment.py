#!/usr/bin/env python3
"""
Sentiment Analysis Demo for PWMK

Demonstrates multi-agent sentiment analysis with Theory of Mind capabilities.
"""

import numpy as np
from pwmk import (
    PerspectiveWorldModel,
    BeliefStore,
    SentimentAnalyzer,
    MultiAgentSentimentAnalyzer,
    PerspectiveSentimentModel,
    BeliefAwareSentimentTracker
)
from pwmk.envs import MultiAgentBoxWorld
from pwmk.utils.logging import setup_logging

setup_logging(level="INFO")


def demo_basic_sentiment_analysis():
    """Demo basic sentiment analysis functionality."""
    print("\n=== Basic Sentiment Analysis Demo ===")
    
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I'm really excited about this collaboration!",
        "This task is quite frustrating and difficult.",
        "The weather is okay today, nothing special.",
        "Great job everyone, we're making excellent progress!",
        "I'm worried about meeting the deadline."
    ]
    
    for text in test_texts:
        sentiment = analyzer.analyze_text(text)
        dominant = max(sentiment.items(), key=lambda x: x[1])
        print(f"Text: '{text}'")
        print(f"Sentiment: {sentiment}")
        print(f"Dominant: {dominant[0]} ({dominant[1]:.3f})")
        print()


def demo_multi_agent_sentiment():
    """Demo multi-agent sentiment analysis."""
    print("\n=== Multi-Agent Sentiment Analysis Demo ===")
    
    num_agents = 3
    belief_store = BeliefStore()
    multi_analyzer = MultiAgentSentimentAnalyzer(
        num_agents=num_agents,
        belief_store=belief_store
    )
    
    # Simulate agent communications
    communications = [
        (0, "I think we should work together on this problem."),
        (1, "That's a terrible idea, we'll never finish in time!"),
        (2, "I'm neutral about the approach, let's try it."),
        (0, "Agent 1 seems stressed, maybe we should help them."),
        (1, "Actually, you're right. Let's collaborate."),
        (2, "Great! I'm happy to see everyone getting along now."),
    ]
    
    for agent_id, text in communications:
        print(f"Agent {agent_id}: '{text}'")
        sentiment = multi_analyzer.analyze_agent_communication(agent_id, text)
        dominant = max(sentiment.items(), key=lambda x: x[1])
        print(f"  Sentiment: {dominant[0]} ({dominant[1]:.3f})")
        print()
    
    # Analyze group dynamics
    print("=== Group Sentiment Analysis ===")
    group_sentiment = multi_analyzer.analyze_group_sentiment()
    print(f"Group sentiment: {group_sentiment}")
    
    current_states = multi_analyzer.get_current_sentiment_state()
    for agent_id, sentiment in current_states.items():
        dominant = max(sentiment.items(), key=lambda x: x[1])
        print(f"Agent {agent_id} current state: {dominant[0]} ({dominant[1]:.3f})")


def demo_perspective_sentiment():
    """Demo perspective-aware sentiment analysis."""
    print("\n=== Perspective-Aware Sentiment Demo ===")
    
    # Create mock environment and world model
    env = MultiAgentBoxWorld(
        grid_size=8,
        num_agents=3,
        num_boxes=2,
        view_radius=3
    )
    
    world_model = PerspectiveWorldModel(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=256,
        num_agents=3
    )
    
    sentiment_analyzer = SentimentAnalyzer()
    perspective_model = PerspectiveSentimentModel(
        world_model=world_model,
        sentiment_analyzer=sentiment_analyzer,
        num_agents=3
    )
    
    # Simulate different world states
    obs = env.reset()
    world_state = np.random.randn(256)  # Mock world state
    
    test_text = "I think we should move the box to the corner."
    
    print(f"Analyzing text: '{test_text}'")
    print()
    
    # Compare perspectives
    perspectives = perspective_model.compare_perspectives(test_text, world_state)
    
    for agent_id, sentiment in perspectives.items():
        dominant = max(sentiment.items(), key=lambda x: x[1])
        print(f"Agent {agent_id} perspective: {dominant[0]} ({dominant[1]:.3f})")
        
    # Get consensus
    consensus = perspective_model.get_consensus_sentiment(test_text, world_state)
    consensus_dominant = max(consensus.items(), key=lambda x: x[1])
    print(f"\nConsensus sentiment: {consensus_dominant[0]} ({consensus_dominant[1]:.3f})")
    
    # Check for disagreement
    has_disagreement, metrics = perspective_model.detect_sentiment_disagreement(
        test_text, world_state
    )
    print(f"Perspective disagreement detected: {has_disagreement}")
    if has_disagreement:
        print(f"Max variance: {metrics['max_variance']:.3f}")


def demo_belief_aware_sentiment():
    """Demo belief-aware sentiment tracking."""
    print("\n=== Belief-Aware Sentiment Demo ===")
    
    num_agents = 3
    belief_store = BeliefStore()
    multi_analyzer = MultiAgentSentimentAnalyzer(num_agents, belief_store=belief_store)
    
    belief_tracker = BeliefAwareSentimentTracker(
        belief_store=belief_store,
        sentiment_analyzer=multi_analyzer,
        num_agents=num_agents
    )
    
    # Simulate scenario with beliefs about sentiments
    scenario_communications = [
        (0, "I'm excited about this new project!"),
        (1, "I'm not sure about this approach..."),
        (2, "Agent 0 seems really enthusiastic about everything."),
        (0, "I notice Agent 1 might be having doubts."),
        (1, "Actually, I'm feeling more optimistic now."),
        (2, "It's interesting how sentiments change during discussion."),
    ]
    
    print("=== Communication Sequence ===")
    for agent_id, text in scenario_communications:
        print(f"Agent {agent_id}: '{text}'")
        sentiment = belief_tracker.update_sentiment_beliefs(agent_id, text)
        dominant = max(sentiment.items(), key=lambda x: x[1])
        print(f"  Detected sentiment: {dominant[0]} ({dominant[1]:.3f})")
        print()
    
    # Analyze belief misalignments
    print("=== Sentiment Belief Analysis ===")
    misalignments = belief_tracker.detect_sentiment_misalignment()
    
    if misalignments:
        print("Detected sentiment belief misalignments:")
        for misalign in misalignments:
            print(f"  Agent {misalign['observer']} believes Agent {misalign['target']} feels different than reality")
            print(f"  Misalignment score: {misalign['misalignment_score']:.3f}")
    else:
        print("No significant sentiment belief misalignments detected.")
        
    # Show group dynamics
    print("\n=== Group Sentiment Dynamics ===")
    dynamics = belief_tracker.get_group_sentiment_dynamics()
    
    group_sentiment = dynamics["group_sentiment"]
    dominant = max(group_sentiment.items(), key=lambda x: x[1])
    print(f"Overall group sentiment: {dominant[0]} ({dominant[1]:.3f})")
    
    print(f"Positive interactions: {len(dynamics['positive_interactions'])}")
    print(f"Sentiment conflicts: {len(dynamics['conflicts'])}")
    print(f"Sentiment alignments: {len(dynamics['alignments'])}")


def main():
    """Run all sentiment analysis demos."""
    print("🎭 PWMK Sentiment Analysis Demonstration")
    print("=" * 50)
    
    try:
        demo_basic_sentiment_analysis()
        demo_multi_agent_sentiment()
        demo_perspective_sentiment()
        demo_belief_aware_sentiment()
        
        print("\n✅ All sentiment analysis demos completed successfully!")
        print("\nSentiment analysis capabilities:")
        print("• Basic text sentiment classification")
        print("• Multi-agent sentiment tracking") 
        print("• Perspective-aware sentiment analysis")
        print("• Belief-integrated sentiment reasoning")
        print("• Theory of Mind sentiment predictions")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()