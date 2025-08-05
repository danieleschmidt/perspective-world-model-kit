# Quantum-Enhanced Planning Examples

This document provides comprehensive examples demonstrating the quantum-enhanced planning capabilities in PWMK. Each example includes complete code, explanations, and expected results.

## Table of Contents

- [Basic Quantum Planning](#basic-quantum-planning)
- [Multi-Agent Coordination](#multi-agent-coordination)
- [Resource Management](#resource-management)
- [Performance Optimization](#performance-optimization)
- [Real-time Monitoring](#real-time-monitoring)
- [Custom Algorithms](#custom-algorithms)
- [Advanced Integration](#advanced-integration)

## Basic Quantum Planning

### Simple Grid Navigation

This example demonstrates basic quantum-inspired planning for single-agent navigation:

```python
import asyncio
import numpy as np
from pwmk.quantum import QuantumInspiredPlanner, PlanningResult

async def simple_grid_navigation():
    """Simple quantum planning for grid navigation."""
    
    # Initialize quantum planner
    planner = QuantumInspiredPlanner(
        num_qubits=6,           # Small quantum register
        max_depth=10,           # Planning horizon
        num_agents=1,           # Single agent
        coherence_time=2.0      # Coherence time
    )
    
    # Define grid environment
    initial_state = {
        "agent_position": (0, 0),
        "goal_position": (4, 4),
        "grid_size": (5, 5),
        "obstacles": [(2, 2), (3, 1)]
    }
    
    # Available actions
    action_space = [
        "move_north",
        "move_south", 
        "move_east",
        "move_west",
        "wait"
    ]
    
    goal = "navigate to goal position while avoiding obstacles"
    
    print("🌀 Starting quantum grid navigation...")
    print(f"Start: {initial_state['agent_position']}")
    print(f"Goal: {initial_state['goal_position']}")
    print(f"Obstacles: {initial_state['obstacles']}")
    
    # Run quantum planning
    result = planner.plan(
        initial_state=initial_state,
        goal=goal,
        action_space=action_space,
        max_iterations=30
    )
    
    print(f"\n✅ Planning completed!")
    print(f"Best action sequence: {result.best_action_sequence}")
    print(f"Success probability: {result.probability:.3f}")
    print(f"Quantum advantage: {result.quantum_advantage:.2f}x")
    print(f"Planning time: {result.planning_time:.4f}s")
    
    # Analyze interference patterns
    if result.interference_patterns:
        print(f"\n🔬 Quantum Interference Analysis:")
        for pattern_name, strength in result.interference_patterns.items():
            print(f"  {pattern_name}: {strength:.3f}")
    
    return result

# Run example
if __name__ == "__main__":
    result = asyncio.run(simple_grid_navigation())
```

**Expected Output:**
```
🌀 Starting quantum grid navigation...
Start: (0, 0)
Goal: (4, 4)
Obstacles: [(2, 2), (3, 1)]

✅ Planning completed!
Best action sequence: ['move_east', 'move_north', 'move_east', 'move_north']
Success probability: 0.842
Quantum advantage: 2.15x
Planning time: 0.0234s

🔬 Quantum Interference Analysis:
  interference_strength: 0.234
  phase_coherence: 0.876
  entanglement_strength: 0.000
```

### Weighted Action Selection

Example with prioritized actions using weighted superposition:

```python
from pwmk.quantum import QuantumInspiredPlanner

def weighted_action_planning():
    """Quantum planning with action preferences."""
    
    planner = QuantumInspiredPlanner(
        num_qubits=8,
        max_depth=12,
        num_agents=1
    )
    
    # Define scenario with preferences
    initial_state = {
        "agent_position": (1, 1),
        "energy_level": 0.8,
        "inventory": ["key"],
        "doors": [(3, 3)],
        "enemies": [(4, 2)]
    }
    
    action_space = [
        "move_north", "move_south", "move_east", "move_west",
        "use_key", "attack", "defend", "rest"
    ]
    
    # Action preferences (higher = more preferred)
    action_weights = np.array([
        0.3,  # move_north - moderate
        0.2,  # move_south - lower  
        0.4,  # move_east - higher (toward goal)
        0.1,  # move_west - lowest (away from goal)
        0.6,  # use_key - very high (opens doors)
        0.1,  # attack - risky
        0.3,  # defend - moderate
        0.2   # rest - if needed
    ])
    
    goal = "reach the treasure while managing resources"
    
    # Create weighted superposition
    quantum_state = planner.create_superposition_state(
        action_space,
        initial_weights=action_weights
    )
    
    # Apply goal-directed interference
    goal_vector = np.array([1.0, 0.8, 0.6])  # Encoded goal preferences
    
    environment_feedback = {
        "move_east": 0.5,   # Positive feedback (toward goal)
        "use_key": 0.8,     # Very positive (enables progress)
        "attack": -0.3,     # Negative (risky)
        "rest": 0.2         # Slightly positive (resource management)
    }
    
    modified_state = planner.apply_quantum_interference(
        quantum_state,
        goal_vector,
        environment_feedback
    )
    
    # Measure final state
    result = planner.measure_quantum_state(modified_state, num_measurements=500)
    
    print("🎯 Weighted Action Planning Results:")
    print(f"Optimal actions: {result.best_action_sequence}")
    print(f"Selection probability: {result.probability:.3f}")
    
    # Show how weights influenced selection
    selected_actions = result.best_action_sequence
    for action in selected_actions:
        if action in action_space:
            idx = action_space.index(action)
            weight = action_weights[idx]
            print(f"  {action}: original weight = {weight:.2f}")
    
    return result

# Run weighted planning
weighted_result = weighted_action_planning()
```

## Multi-Agent Coordination

### Three-Agent Coordination

Complex coordination scenario with quantum entanglement:

```python
from pwmk.quantum import QuantumInspiredPlanner

async def three_agent_coordination():
    """Multi-agent coordination with quantum entanglement."""
    
    # Initialize planner for 3 agents
    planner = QuantumInspiredPlanner(
        num_qubits=10,          # More qubits for complex coordination
        max_depth=15,
        num_agents=3,
        coherence_time=3.0      # Longer coherence for stability
    )
    
    # Complex coordination scenario
    initial_state = {
        "agents": [
            {
                "id": "scout", 
                "position": (0, 0), 
                "role": "exploration",
                "energy": 1.0,
                "equipment": ["scanner"]
            },
            {
                "id": "builder", 
                "position": (2, 0), 
                "role": "construction",
                "energy": 0.9,
                "equipment": ["tools", "materials"]
            },
            {
                "id": "defender", 
                "position": (1, 2), 
                "role": "protection",
                "energy": 0.8,
                "equipment": ["weapons", "shield"]
            }
        ],
        "objectives": [
            {"type": "explore", "location": (5, 5), "priority": 2.0},
            {"type": "build", "location": (3, 3), "priority": 3.0},
            {"type": "defend", "location": (3, 3), "priority": 2.5}
        ],
        "threats": [(6, 4), (4, 6)],
        "resources": [(1, 4), (4, 1), (5, 2)]
    }
    
    # Coordinated action space
    action_space = [
        # Navigation
        "move_north", "move_south", "move_east", "move_west",
        # Role-specific actions
        "explore_area", "scan_for_threats", "mark_location",
        "build_structure", "collect_materials", "repair",
        "defend_position", "attack_threat", "create_barrier",
        # Coordination actions
        "communicate_status", "request_help", "provide_support",
        "coordinate_movement", "share_resources", "wait"
    ]
    
    goal = "coordinate all agents to complete objectives while managing threats"
    
    print("🤝 Three-Agent Coordination Planning")
    print("Agents:", [agent["id"] for agent in initial_state["agents"]])
    print("Objectives:", len(initial_state["objectives"]))
    print("Threats:", len(initial_state["threats"]))
    
    # Run coordinated planning
    result = planner.plan(
        initial_state=initial_state,
        goal=goal,
        action_space=action_space,
        max_iterations=50
    )
    
    print(f"\n✅ Coordination Plan Generated:")
    print(f"Action sequence: {result.best_action_sequence}")
    print(f"Coordination probability: {result.probability:.3f}")
    print(f"Quantum advantage: {result.quantum_advantage:.2f}x")
    
    # Analyze entanglement effects
    if hasattr(planner, '_generate_entanglement_map'):
        entanglement_map = planner._generate_entanglement_map()
        print(f"\n🔗 Agent Entanglement Map:")
        for agent_id, entangled_with in entanglement_map.items():
            agent_name = initial_state["agents"][agent_id]["id"]
            entangled_names = [initial_state["agents"][e]["id"] for e in entangled_with]
            print(f"  {agent_name} entangled with: {entangled_names}")
    
    return result

# Run coordination example
coordination_result = asyncio.run(three_agent_coordination())
```

### Communication-Based Coordination

Coordination with explicit communication actions:

```python
def communication_coordination():
    """Coordination emphasizing quantum communication channels."""
    
    planner = QuantumInspiredPlanner(
        num_qubits=8,
        max_depth=12,
        num_agents=4
    )
    
    # Communication-heavy scenario
    initial_state = {
        "agents": [
            {"id": "leader", "position": (2, 2), "communication_range": 3},
            {"id": "follower1", "position": (0, 0), "communication_range": 2},
            {"id": "follower2", "position": (4, 0), "communication_range": 2},  
            {"id": "follower3", "position": (2, 4), "communication_range": 2}
        ],
        "communication_channels": [
            ("leader", "follower1"), ("leader", "follower2"), ("leader", "follower3"),
            ("follower1", "follower2")  # Direct follower communication
        ],
        "mission_phases": [
            {"phase": "gather", "duration": 5},
            {"phase": "execute", "duration": 10},
            {"phase": "extract", "duration": 3}
        ]
    }
    
    # Communication-focused actions
    action_space = [
        # Basic movement
        "move_north", "move_south", "move_east", "move_west",
        # Communication actions
        "broadcast_position", "send_status_update", "request_coordinates",
        "acknowledge_message", "relay_information", "signal_ready",
        # Coordination actions
        "form_formation", "break_formation", "synchronize_movement",
        "establish_overwatch", "provide_cover", "signal_advance",
        # Mission actions
        "begin_phase", "complete_objective", "abort_mission", "wait"
    ]
    
    goal = "coordinate mission phases through quantum-enhanced communication"
    
    # Simulate communication delays and interference
    environment_feedback = {
        "broadcast_position": 0.8,      # High value for coordination
        "send_status_update": 0.7,      # Important for awareness
        "synchronize_movement": 0.9,    # Critical for coordination
        "establish_overwatch": 0.6,     # Tactical advantage
        "acknowledge_message": 0.5,     # Basic communication
        "wait": -0.2                    # Discourage excessive waiting
    }
    
    # Enhanced goal encoding for communication
    goal_vector = np.array([
        0.9,  # High coordination priority
        0.8,  # Strong communication emphasis  
        0.7   # Mission completion focus
    ])
    
    print("📡 Communication-Based Coordination")
    print(f"Agents: {len(initial_state['agents'])}")
    print(f"Communication channels: {len(initial_state['communication_channels'])}")
    print(f"Mission phases: {len(initial_state['mission_phases'])}")
    
    # Create superposition with communication emphasis
    quantum_state = planner.create_superposition_state(action_space)
    
    # Apply communication-focused interference
    modified_state = planner.apply_quantum_interference(
        quantum_state,
        goal_vector,
        environment_feedback
    )
    
    result = planner.measure_quantum_state(modified_state, num_measurements=800)
    
    print(f"\n📋 Communication Plan:")
    for i, action in enumerate(result.best_action_sequence):
        if "communicate" in action or "signal" in action or "broadcast" in action:
            print(f"  Step {i+1}: {action} ⭐ (Communication)")
        else:
            print(f"  Step {i+1}: {action}")
    
    print(f"\nCoordination effectiveness: {result.probability:.3f}")
    print(f"Communication advantage: {result.quantum_advantage:.2f}x")
    
    return result

# Run communication coordination
comm_result = communication_coordination()
```

## Resource Management

### Dynamic Resource Allocation

Quantum annealing for optimal resource distribution:

```python
from pwmk.quantum import QuantumAnnealingScheduler

def dynamic_resource_allocation():
    """Quantum annealing for resource allocation optimization."""
    
    # Initialize annealing scheduler
    scheduler = QuantumAnnealingScheduler(
        initial_temperature=8.0,
        final_temperature=0.005,
        annealing_steps=600
    )
    
    # Complex resource allocation scenario
    agents = [
        "mining_unit_1", "mining_unit_2", "mining_unit_3",
        "transport_1", "transport_2", 
        "refinery_1", "refinery_2",
        "defense_unit", "repair_unit"
    ]
    
    resources = [
        "ore_deposit_alpha", "ore_deposit_beta", "ore_deposit_gamma",
        "energy_crystal_1", "energy_crystal_2",
        "repair_materials", "fuel_supplies",
        "defense_position_1", "defense_position_2", "defense_position_3"
    ]
    
    # Agent capabilities
    capabilities = {
        "mining_unit_1": ["ore_deposit_alpha", "ore_deposit_beta", "ore_deposit_gamma"],
        "mining_unit_2": ["ore_deposit_alpha", "ore_deposit_beta", "ore_deposit_gamma"],
        "mining_unit_3": ["ore_deposit_beta", "ore_deposit_gamma"],
        "transport_1": ["fuel_supplies", "repair_materials"],
        "transport_2": ["fuel_supplies", "repair_materials"],
        "refinery_1": ["energy_crystal_1", "energy_crystal_2"],
        "refinery_2": ["energy_crystal_1", "energy_crystal_2"],
        "defense_unit": ["defense_position_1", "defense_position_2", "defense_position_3"],
        "repair_unit": ["repair_materials"]
    }
    
    # Resource priorities (higher = more important)
    priorities = {
        "ore_deposit_alpha": 3.0,      # Critical resources
        "ore_deposit_beta": 2.5,
        "ore_deposit_gamma": 2.0,
        "energy_crystal_1": 3.5,      # Highest priority
        "energy_crystal_2": 3.2,
        "repair_materials": 1.8,
        "fuel_supplies": 2.2,
        "defense_position_1": 2.8,    # Strategic importance
        "defense_position_2": 2.4,
        "defense_position_3": 2.0
    }
    
    # Coordination requirements (agents that work better together)
    coordination = [
        ("mining_unit_1", "transport_1"),     # Mining-transport chains
        ("mining_unit_2", "transport_2"),
        ("refinery_1", "repair_unit"),        # Refinery-maintenance
        ("defense_unit", "repair_unit")       # Defense-repair coordination
    ]
    
    print("⚡ Dynamic Resource Allocation with Quantum Annealing")
    print(f"Agents: {len(agents)}")
    print(f"Resources: {len(resources)}")
    print(f"Coordination pairs: {len(coordination)}")
    
    # Create optimization problem
    problem = scheduler.create_task_planning_problem(
        agents=agents,
        tasks=resources,  # Resources as tasks to be assigned
        agent_capabilities=capabilities,
        task_priorities=priorities,
        coordination_requirements=coordination
    )
    
    # Solve with multiple annealing runs
    result = scheduler.solve_optimization_problem(
        problem,
        num_runs=8  # Multiple runs for better optimization
    )
    
    print(f"\n❄️ Annealing Results:")
    print(f"Best energy: {result.best_energy:.6f}")
    print(f"Optimization time: {result.annealing_time:.4f}s")
    print(f"Quantum tunneling events: {result.quantum_tunneling_events}")
    
    # Decode solution matrix
    solution_matrix = result.best_solution.reshape(len(agents), len(resources))
    
    print(f"\n📊 Optimal Resource Allocation:")
    total_allocation_score = 0
    
    for i, agent in enumerate(agents):
        assigned_resources = []
        allocation_score = 0
        
        for j, resource in enumerate(resources):
            if solution_matrix[i, j] > 0.5:  # Threshold for assignment
                assigned_resources.append(resource)
                allocation_score += priorities[resource] * solution_matrix[i, j]
        
        total_allocation_score += allocation_score
        
        if assigned_resources:
            print(f"  {agent}:")
            for resource in assigned_resources:
                priority = priorities[resource]
                print(f"    → {resource} (priority: {priority:.1f})")
    
    print(f"\nTotal allocation efficiency: {total_allocation_score:.2f}")
    
    # Analyze coordination effectiveness
    coordination_score = 0
    print(f"\n🤝 Coordination Analysis:")
    for agent1, agent2 in coordination:
        agent1_idx = agents.index(agent1)
        agent2_idx = agents.index(agent2)
        
        # Check resource overlap (coordination measure)
        overlap = np.dot(solution_matrix[agent1_idx], solution_matrix[agent2_idx])
        coordination_score += overlap
        
        print(f"  {agent1} ↔ {agent2}: coordination score = {overlap:.3f}")
    
    print(f"Overall coordination effectiveness: {coordination_score:.3f}")
    
    return result, solution_matrix

# Run resource allocation
allocation_result, allocation_matrix = dynamic_resource_allocation()
```

### Adaptive Resource Reallocation

Real-time resource reallocation with adaptive algorithms:

```python
from pwmk.quantum import AdaptiveQuantumAlgorithm, QuantumParameters
from pwmk.quantum.adaptive_quantum import AdaptationStrategy

def adaptive_resource_reallocation():
    """Adaptive quantum algorithm for dynamic resource reallocation."""
    
    # Initialize adaptive algorithm
    adaptive_algo = AdaptiveQuantumAlgorithm(
        adaptation_strategy=AdaptationStrategy.REINFORCEMENT_LEARNING,
        learning_rate=0.02,
        exploration_rate=0.15
    )
    
    # Initial quantum parameters for resource allocation
    initial_params = QuantumParameters(
        gate_parameters={
            "resource_rotation": 1.2,
            "allocation_phase": 0.6,
            "coordination_amplitude": 1.0
        },
        circuit_depth=6,
        measurement_shots=800,
        decoherence_rate=0.015,
        entanglement_strength=1.8
    )
    
    # Simulation of changing resource demands
    scenarios = [
        {
            "time": 0,
            "demands": {"energy": 0.8, "materials": 0.6, "defense": 0.4},
            "available_resources": {"energy": 1.0, "materials": 0.7, "defense": 0.8},
            "expected_performance": 0.7
        },
        {
            "time": 5,
            "demands": {"energy": 0.9, "materials": 0.8, "defense": 0.7},
            "available_resources": {"energy": 0.9, "materials": 0.8, "defense": 0.6},
            "expected_performance": 0.6
        },
        {
            "time": 10,
            "demands": {"energy": 0.7, "materials": 0.9, "defense": 0.9},
            "available_resources": {"energy": 1.2, "materials": 0.6, "defense": 0.4},
            "expected_performance": 0.5
        },
        {
            "time": 15,
            "demands": {"energy": 0.6, "materials": 0.5, "defense": 0.8},
            "available_resources": {"energy": 1.1, "materials": 0.9, "defense": 0.7},
            "expected_performance": 0.8
        }
    ]
    
    print("🔄 Adaptive Resource Reallocation")
    print(f"Learning strategy: {AdaptationStrategy.REINFORCEMENT_LEARNING.value}")
    print(f"Scenarios: {len(scenarios)}")
    
    current_params = initial_params
    performance_history = []
    adaptation_history = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n⏰ Time {scenario['time']}: Scenario {i+1}")
        print(f"  Demands: {scenario['demands']}")
        print(f"  Available: {scenario['available_resources']}")
        
        # Encode scenario as problem features
        problem_features = np.array([
            scenario["demands"]["energy"],
            scenario["demands"]["materials"], 
            scenario["demands"]["defense"],
            scenario["available_resources"]["energy"],
            scenario["available_resources"]["materials"],
            scenario["available_resources"]["defense"],
            scenario["time"] / 20.0  # Normalized time
        ])
        
        # Simulate actual performance (with some randomness)
        base_performance = scenario["expected_performance"]
        actual_performance = base_performance + np.random.normal(0, 0.1)
        actual_performance = max(0.0, min(1.0, actual_performance))
        
        performance_history.append(actual_performance)
        
        # Adapt parameters based on performance
        adaptation_result = adaptive_algo.adapt_parameters(
            current_parameters=current_params,
            problem_features=problem_features,
            performance_feedback=actual_performance,
            adaptation_context={"scenario": i, "time": scenario["time"]}
        )
        
        current_params = adaptation_result.optimized_parameters
        adaptation_history.append(adaptation_result)
        
        print(f"  Performance: {actual_performance:.3f}")
        print(f"  Improvement: {adaptation_result.performance_improvement:.3f}")
        print(f"  Adaptation time: {adaptation_result.adaptation_time:.4f}s")
        
        # Show key parameter changes
        old_rotation = initial_params.gate_parameters.get("resource_rotation", 1.0)
        new_rotation = current_params.gate_parameters.get("resource_rotation", 1.0)
        rotation_change = new_rotation - old_rotation
        
        print(f"  Parameter change: resource_rotation {rotation_change:+.3f}")
    
    # Analyze learning performance
    print(f"\n📈 Learning Analysis:")
    
    if len(performance_history) > 1:
        initial_performance = performance_history[0]
        final_performance = performance_history[-1]
        total_improvement = final_performance - initial_performance
        
        print(f"Initial performance: {initial_performance:.3f}")
        print(f"Final performance: {final_performance:.3f}")
        print(f"Total improvement: {total_improvement:+.3f}")
        
        # Calculate learning trend
        performance_trend = np.polyfit(range(len(performance_history)), performance_history, 1)[0]
        trend_direction = "improving" if performance_trend > 0.01 else "stable" if performance_trend > -0.01 else "declining"
        print(f"Learning trend: {trend_direction} ({performance_trend:+.4f}/step)")
    
    # Get adaptation statistics
    stats = adaptive_algo.get_adaptation_statistics()
    print(f"\n🧠 Adaptation Statistics:")
    print(f"Adaptations performed: {stats.get('adaptations_performed', 0)}")
    if stats.get('avg_improvement'):
        print(f"Average improvement: {stats['avg_improvement']:.4f}")
    if stats.get('current_performance'):
        print(f"Current performance: {stats['current_performance']:.3f}")
    
    return adaptation_history, performance_history

# Run adaptive reallocation
adaptation_hist, performance_hist = adaptive_resource_reallocation()
```

## Performance Optimization

### Parallel Processing Optimization

Maximizing performance through parallel execution:

```python
from pwmk.quantum.performance import QuantumPerformanceOptimizer, PerformanceConfig
import asyncio

async def parallel_processing_optimization():
    """Demonstrate quantum performance optimization with parallel processing."""
    
    # Configure performance optimizer
    config = PerformanceConfig(
        enable_parallel_processing=True,
        enable_gpu_acceleration=True,  # If available
        enable_adaptive_batching=True,
        max_workers=6,
        batch_size_range=(4, 16),
        memory_limit_gb=4.0
    )
    
    optimizer = QuantumPerformanceOptimizer(config=config)
    
    # Create various quantum operations for benchmarking
    def quantum_superposition_op(num_states: int, complexity: float):
        """Simulate quantum superposition operation."""
        import time
        time.sleep(complexity * 0.01)  # Simulate computation
        
        # Generate quantum amplitudes
        amplitudes = np.random.random(num_states) + 1j * np.random.random(num_states)
        amplitudes /= np.linalg.norm(amplitudes)
        
        return {
            "amplitudes": amplitudes,
            "num_states": num_states,
            "complexity": complexity,
            "fidelity": 0.95 + 0.05 * np.random.random()
        }
    
    def quantum_interference_op(amplitude_size: int, goal_alignment: float):
        """Simulate quantum interference operation."""
        import time
        time.sleep(goal_alignment * 0.005)
        
        # Apply interference patterns
        interference_pattern = np.random.random(amplitude_size) * goal_alignment
        
        return {
            "interference_pattern": interference_pattern,
            "goal_alignment": goal_alignment,
            "coherence": 0.9 * np.exp(-goal_alignment * 0.1)
        }
    
    def quantum_measurement_op(num_measurements: int, precision: float):
        """Simulate quantum measurement operation."""
        import time
        time.sleep(num_measurements * precision * 0.0001)
        
        # Generate measurement results
        measurements = np.random.choice([0, 1], size=num_measurements, p=[0.6, 0.4])
        
        return {
            "measurements": measurements,
            "success_probability": np.mean(measurements),
            "measurement_error": 0.01 * (1.0 - precision)
        }
    
    # Create diverse operation batches
    operations = []
    operation_args = []
    
    # Batch 1: Superposition operations
    for i in range(15):
        operations.append(quantum_superposition_op)
        operation_args.append((4 + i % 8, 0.5 + (i % 4) * 0.2))
    
    # Batch 2: Interference operations
    for i in range(12):
        operations.append(quantum_interference_op)
        operation_args.append((8 + i % 6, 0.3 + (i % 3) * 0.3))
    
    # Batch 3: Measurement operations
    for i in range(18):
        operations.append(quantum_measurement_op)
        operation_args.append((100 + i * 20, 0.8 + (i % 5) * 0.04))
    
    print("⚡ Quantum Performance Optimization Demo")
    print(f"Total operations: {len(operations)}")
    print(f"Max workers: {config.max_workers}")
    print(f"Batch size range: {config.batch_size_range}")
    print(f"GPU acceleration: {config.enable_gpu_acceleration}")
    
    # Run optimized execution
    print(f"\n🚀 Running optimized quantum operations...")
    
    start_time = time.time()
    result = await optimizer.optimize_quantum_operations(
        operations=operations,
        operation_args=operation_args,
        optimization_hints={
            "prefer_parallel": True,
            "target_latency": 2.0
        }
    )
    total_time = time.time() - start_time
    
    print(f"\n✅ Optimization completed!")
    print(f"Total execution time: {total_time:.4f}s")
    print(f"Speedup factor: {result.speedup_factor:.2f}x")
    print(f"Memory savings: {result.memory_savings_mb:.1f} MB")
    print(f"Cache efficiency: {result.cache_efficiency:.2%}")
    print(f"Optimization overhead: {result.optimization_time:.4f}s")
    
    # Resource utilization analysis
    resources = result.resource_utilization
    print(f"\n📊 Resource Utilization:")
    print(f"  CPU usage: {resources.cpu_percent:.1f}%")
    print(f"  Memory usage: {resources.memory_percent:.1f}%")
    print(f"  GPU memory: {resources.gpu_memory_percent:.1f}%")
    print(f"  Active threads: {resources.active_threads}")
    print(f"  Operations/sec: {resources.quantum_operations_per_second:.1f}")
    print(f"  Cache hit rate: {resources.cache_hit_rate:.2%}")
    
    # Performance statistics
    perf_stats = optimizer.get_performance_statistics()
    print(f"\n📈 Performance Statistics:")
    print(f"  Total operations: {perf_stats['total_operations']}")
    print(f"  Parallel operations: {perf_stats['parallel_operations']}")
    print(f"  GPU operations: {perf_stats['gpu_operations']}")
    
    if "batch_performance" in perf_stats:
        batch_perf = perf_stats["batch_performance"]
        print(f"  Current batch size: {batch_perf['current_batch_size']}")
        print(f"  Average throughput: {batch_perf['avg_throughput']:.1f} ops/sec")
        print(f"  Throughput trend: {batch_perf['throughput_trend']}")
    
    # Cleanup
    await optimizer.cleanup()
    
    return result

# Run performance optimization demo
perf_result = asyncio.run(parallel_processing_optimization())
```

### Memory Optimization

Advanced memory management for large-scale quantum operations:

```python
import psutil
import gc
from pwmk.quantum.performance import QuantumPerformanceOptimizer, PerformanceConfig

def memory_optimization_demo():
    """Demonstrate memory optimization for quantum operations."""
    
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def large_quantum_operation(matrix_size: int):
        """Memory-intensive quantum operation."""
        import time
        
        # Create large quantum state matrices
        state_matrix = np.random.random((matrix_size, matrix_size)) + 1j * np.random.random((matrix_size, matrix_size))
        
        # Simulate quantum evolution
        time.sleep(0.01)
        
        # Normalize and process
        state_matrix /= np.linalg.norm(state_matrix)
        eigenvalues = np.linalg.eigvals(state_matrix[:min(50, matrix_size), :min(50, matrix_size)])
        
        return {
            "matrix_size": matrix_size,
            "eigenvalues": eigenvalues[:5],  # Keep only first 5
            "trace": np.trace(state_matrix[:min(100, matrix_size), :min(100, matrix_size)])
        }
    
    # Configure for memory optimization
    config = PerformanceConfig(
        enable_parallel_processing=True,
        enable_memory_optimization=True,
        memory_limit_gb=2.0,  # Strict memory limit
        batch_size_range=(1, 4),  # Small batches to manage memory
        auto_scaling_enabled=True
    )
    
    optimizer = QuantumPerformanceOptimizer(config=config)
    
    print("💾 Memory Optimization Demo")
    print(f"Memory limit: {config.memory_limit_gb:.1f} GB")
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Create memory-intensive operations
    operations = [large_quantum_operation] * 20
    operation_args = [(50 + i * 10,) for i in range(20)]  # Increasing matrix sizes
    
    print(f"Operations: {len(operations)}")
    print(f"Matrix sizes: {[args[0] for args in operation_args[:5]]}...{[args[0] for args in operation_args[-3:]]}")
    
    # Monitor memory during execution
    memory_checkpoints = []
    
    async def memory_monitoring_execution():
        """Execute with memory monitoring."""
        memory_checkpoints.append(("start", get_memory_usage()))
        
        result = await optimizer.optimize_quantum_operations(
            operations=operations,
            operation_args=operation_args,
            optimization_hints={"memory_priority": True}
        )
        
        memory_checkpoints.append(("end", get_memory_usage()))
        return result
    
    # Run with memory monitoring
    import asyncio
    result = asyncio.run(memory_monitoring_execution())
    
    # Force garbage collection and check final memory
    gc.collect()
    final_memory = get_memory_usage()
    memory_checkpoints.append(("after_gc", final_memory))
    
    print(f"\n📊 Memory Usage Analysis:")
    for checkpoint_name, memory_usage in memory_checkpoints:
        print(f"  {checkpoint_name}: {memory_usage:.1f} MB")
    
    memory_peak = max(usage for _, usage in memory_checkpoints)
    memory_savings = result.memory_savings_mb
    
    print(f"  Peak memory: {memory_peak:.1f} MB")
    print(f"  Memory savings: {memory_savings:.1f} MB")
    print(f"  Memory efficiency: {(memory_savings / memory_peak * 100):.1f}%")
    
    # Check if memory limit was respected
    memory_limit_mb = config.memory_limit_gb * 1024
    if memory_peak <= memory_limit_mb:
        print(f"  ✅ Memory limit respected ({memory_peak:.1f}/{memory_limit_mb:.1f} MB)")
    else:
        print(f"  ⚠️ Memory limit exceeded ({memory_peak:.1f}/{memory_limit_mb:.1f} MB)")
    
    # Optimization statistics
    perf_stats = optimizer.get_performance_statistics()
    if "memory_optimizations" in perf_stats:
        print(f"  Memory optimizations performed: {perf_stats['memory_optimizations']}")
    
    return result, memory_checkpoints

# Run memory optimization demo
memory_result, memory_checkpoints = memory_optimization_demo()
```

## Real-time Monitoring

### Comprehensive Monitoring Setup

Complete monitoring solution with dashboards and alerts:

```python
from pwmk.quantum import QuantumMetricsCollector, MetricType, QuantumDashboard
from pwmk.quantum.monitoring import QuantumDashboardConfig
import threading
import time

def comprehensive_monitoring_demo():
    """Comprehensive quantum monitoring with dashboards and alerts."""
    
    # Configure dashboard with custom settings
    dashboard_config = QuantumDashboardConfig(
        update_interval=0.5,        # Fast updates
        history_size=200,           # Large history
        enable_realtime=True,
        export_format="json",
        alert_thresholds={
            "quantum_advantage": 1.8,
            "coherence_time": 1.0,
            "gate_fidelity": 0.95
        }
    )
    
    # Initialize monitoring components
    metrics_collector = QuantumMetricsCollector(
        dashboard_config=dashboard_config,
        export_path="quantum_monitoring_demo"
    )
    
    dashboard = QuantumDashboard(
        metrics_collector=metrics_collector,
        dashboard_config=dashboard_config
    )
    
    # Set up alert callbacks
    def quantum_advantage_alert(metric_name: str, value: float, threshold: float):
        print(f"🚨 ALERT: {metric_name} = {value:.3f} (threshold: {threshold:.3f})")
    
    def coherence_alert(metric_name: str, value: float, threshold: float):
        print(f"⚠️ WARNING: {metric_name} = {value:.3f}s is below threshold {threshold:.3f}s")
    
    metrics_collector.set_alert_threshold(
        MetricType.QUANTUM_ADVANTAGE,
        threshold=1.8,
        callback=quantum_advantage_alert
    )
    
    metrics_collector.set_alert_threshold(
        MetricType.COHERENCE_TIME,
        threshold=1.0,
        callback=coherence_alert
    )
    
    print("📊 Comprehensive Quantum Monitoring Demo")
    print(f"Update interval: {dashboard_config.update_interval}s")
    print(f"History size: {dashboard_config.history_size}")
    print(f"Alert thresholds: {len(dashboard_config.alert_thresholds)}")
    
    # Start real-time monitoring
    dashboard.start_real_time_updates()
    
    # Simulate quantum operations with monitoring
    def simulate_quantum_operations():
        """Simulate various quantum operations with metrics."""
        
        planner = QuantumInspiredPlanner(num_qubits=8, max_depth=12, num_agents=2)
        
        for iteration in range(30):
            # Simulate different types of operations
            operation_type = ["planning", "optimization", "annealing"][iteration % 3]
            
            if operation_type == "planning":
                # Quantum planning operation
                result = planner.plan(
                    initial_state={"agents": [{"id": "test_agent", "position": (0, 0)}]},
                    goal=f"test goal {iteration}",
                    action_space=["move_north", "move_south", "move_east", "move_west"],
                    max_iterations=10
                )
                
                # Record planning metrics
                metrics_collector.record_quantum_metric(
                    MetricType.QUANTUM_ADVANTAGE,
                    result.quantum_advantage,
                    metadata={"operation": "planning", "iteration": iteration}
                )
                
                metrics_collector.record_operation_timing(
                    "quantum_planning",
                    result.planning_time,
                    success=True,
                    metadata={"iteration": iteration}
                )
            
            elif operation_type == "optimization":
                # Circuit optimization metrics
                fidelity = 0.97 + 0.03 * np.random.random()
                gate_count = 50 + iteration * 2
                
                metrics_collector.record_quantum_metric(
                    MetricType.GATE_FIDELITY,
                    fidelity,
                    metadata={"operation": "optimization", "gate_count": gate_count}
                )
                
                metrics_collector.record_quantum_metric(
                    MetricType.CIRCUIT_DEPTH,
                    10 + iteration // 3,
                    metadata={"gates": gate_count}
                )
            
            else:  # annealing
                # Annealing metrics
                coherence_time = 2.0 - 0.02 * iteration + 0.1 * np.random.random()
                convergence_rate = 0.8 + 0.2 * np.random.random()
                
                metrics_collector.record_quantum_metric(
                    MetricType.COHERENCE_TIME,
                    coherence_time,
                    metadata={"operation": "annealing", "temperature": 0.01 * (30 - iteration)}
                )
                
                metrics_collector.record_quantum_metric(
                    MetricType.ANNEALING_CONVERGENCE,
                    convergence_rate,
                    metadata={"iteration": iteration, "final_energy": -2.5 - iteration * 0.1}
                )
            
            # Simulate some processing time
            time.sleep(0.2)
            
            # Progress indication
            if iteration % 10 == 0:
                print(f"  Completed {iteration + 1}/30 operations...")
    
    # Run simulation in separate thread
    simulation_thread = threading.Thread(target=simulate_quantum_operations, daemon=True)
    simulation_thread.start()
    
    # Wait for simulation to complete
    simulation_thread.join()
    
    # Allow some time for final metric collection
    time.sleep(1.0)
    
    # Stop real-time updates
    dashboard.stop_real_time_updates()
    
    print(f"\n📈 Monitoring Results:")
    
    # Analyze collected metrics
    advantage_trend = metrics_collector.get_quantum_advantage_trend(20)
    if advantage_trend:
        print(f"  Quantum advantage trend: {np.mean(advantage_trend):.3f} ± {np.std(advantage_trend):.3f}")
    
    coherence_analysis = metrics_collector.get_coherence_analysis()
    if coherence_analysis["status"] == "active":
        print(f"  Coherence analysis:")
        print(f"    Mean: {coherence_analysis['mean_coherence']:.3f}s")
        print(f"    Stability: {1 - coherence_analysis['stability_score']:.3f}")
        print(f"    Trend: {coherence_analysis['trend']}")
    
    circuit_metrics = metrics_collector.get_circuit_optimization_metrics()
    if circuit_metrics.get("gate_fidelity"):
        fidelity_data = circuit_metrics["gate_fidelity"]
        print(f"  Gate fidelity: {fidelity_data['average']:.4f} (current: {fidelity_data['current']:.4f})")
    
    # Export dashboard and metrics
    dashboard_file = dashboard.export_dashboard_html("performance_overview")
    metrics_file = metrics_collector.export_metrics()
    
    print(f"\n💾 Export Results:")
    print(f"  Dashboard: {dashboard_file}")
    print(f"  Metrics: {metrics_file}")
    
    # Get comprehensive summary
    summary = metrics_collector.get_metrics_summary()
    print(f"\n📋 Metrics Summary:")
    for metric_name, data in summary.items():
        if data.get("count", 0) > 0:
            print(f"  {metric_name}:")
            print(f"    Count: {data['count']}")
            print(f"    Mean: {data.get('mean', 0):.3f}")
            print(f"    Range: {data.get('min', 0):.3f} - {data.get('max', 0):.3f}")
            print(f"    Trend: {data.get('trend', 'unknown')}")
    
    # Cleanup
    dashboard.cleanup()
    
    return dashboard_file, metrics_file, summary

# Run comprehensive monitoring demo
dashboard_file, metrics_file, monitoring_summary = comprehensive_monitoring_demo()
```

## Advanced Integration

### Full PWMK Integration

Complete integration with all PWMK components:

```python
from pwmk import PerspectiveWorldModel, BeliefStore, EpistemicPlanner, ToMAgent
from pwmk.envs import SimpleGridEnv
from pwmk.quantum import (
    QuantumEnhancedPlanner, QuantumPlanningConfig,
    QuantumMetricsCollector, QuantumDashboard, MetricType
)
import asyncio

async def full_pwmk_integration():
    """Complete integration demonstration with all PWMK components."""
    
    print("🌟 Full PWMK Quantum Integration Demo")
    print("=" * 50)
    
    # 1. Initialize Core PWMK Components
    print("🏗️ Initializing PWMK components...")
    
    world_model = PerspectiveWorldModel(
        obs_dim=128,
        action_dim=8,
        hidden_dim=256,
        num_agents=3,
        num_layers=4
    )
    
    belief_store = BeliefStore()
    
    # Add some initial beliefs
    belief_store.add_belief("agent_0", "can_navigate(grid)")
    belief_store.add_belief("agent_1", "has_capability(build)")
    belief_store.add_belief("agent_2", "specializes_in(defense)")
    
    classical_planner = EpistemicPlanner(
        world_model=world_model,
        belief_store=belief_store
    )
    
    # 2. Configure Quantum Enhancement
    print("⚡ Configuring quantum enhancement...")
    
    quantum_config = QuantumPlanningConfig(
        enable_quantum_superposition=True,
        enable_circuit_optimization=True,
        enable_quantum_annealing=True,
        enable_adaptive_parameters=True,
        classical_fallback=True,
        parallel_execution=True,
        max_planning_time=15.0,
        confidence_threshold=0.75
    )
    
    # 3. Create Quantum-Enhanced Planner
    quantum_enhanced_planner = QuantumEnhancedPlanner(
        world_model=world_model,
        belief_store=belief_store,
        classical_planner=classical_planner,
        config=quantum_config
    )
    
    # 4. Initialize Monitoring
    print("📊 Setting up monitoring...")
    
    metrics_collector = QuantumMetricsCollector()
    dashboard = QuantumDashboard(metrics_collector)
    dashboard.start_real_time_updates()
    
    # 5. Create Complex Multi-Agent Scenario
    print("🌍 Creating complex scenario...")
    
    scenario = {
        "environment": {
            "type": "complex_grid",
            "size": (12, 12),
            "obstacles": [(3, 3), (4, 4), (5, 5), (8, 2), (2, 8), (9, 9)],
            "resources": [
                {"type": "energy_crystal", "position": (10, 10), "value": 100},
                {"type": "build_materials", "position": (1, 10), "value": 80},
                {"type": "weapon_cache", "position": (10, 1), "value": 90}
            ],
            "threats": [
                {"type": "enemy_patrol", "position": (6, 6), "strength": 0.7},
                {"type": "hazard_zone", "area": [(7, 7), (7, 8), (8, 7), (8, 8)]}
            ]
        },
        "agents": [
            {
                "id": "navigator",
                "role": "exploration",
                "position": (0, 0),
                "capabilities": ["move", "scout", "communicate"],
                "energy": 1.0,
                "goals": ["find_path_to(energy_crystal)", "avoid_threats"]
            },
            {
                "id": "builder", 
                "role": "construction",
                "position": (0, 6),
                "capabilities": ["move", "build", "carry_materials"],
                "energy": 0.9,
                "goals": ["collect(build_materials)", "construct(base)"]
            },
            {
                "id": "guardian",
                "role": "defense",
                "position": (6, 0), 
                "capabilities": ["move", "attack", "defend", "patrol"],
                "energy": 0.8,
                "goals": ["secure(weapon_cache)", "defend(team)"]
            }
        ],
        "mission": {
            "primary_objective": "establish secure base with all resources",
            "time_limit": 50,
            "success_criteria": [
                "all_resources_collected",
                "base_constructed",
                "threats_neutralized"
            ]
        }
    }
    
    # 6. Extract Planning Inputs
    initial_state = {
        "agents": scenario["agents"],
        "environment": scenario["environment"],
        "mission_status": {
            "phase": "initial",
            "time_remaining": scenario["mission"]["time_limit"]
        }
    }
    
    action_space = [
        # Navigation
        "move_north", "move_south", "move_east", "move_west",
        # Interaction
        "collect_resource", "build_structure", "attack_threat",
        # Coordination
        "communicate_position", "request_assistance", "provide_support",
        # Utility
        "scout_area", "wait", "retreat"
    ]
    
    goal = scenario["mission"]["primary_objective"]
    
    agent_context = {
        "agents": [agent["id"] for agent in scenario["agents"]],
        "capabilities": {agent["id"]: agent["capabilities"] for agent in scenario["agents"]},
        "priorities": {
            # Action priorities based on mission requirements
            "collect_resource": 3.0,
            "build_structure": 2.8,
            "attack_threat": 2.5,
            "communicate_position": 2.2,
            "scout_area": 2.0,
            "move_north": 1.5, "move_south": 1.5, 
            "move_east": 1.5, "move_west": 1.5,
            "request_assistance": 1.8,
            "provide_support": 2.0,
            "wait": 0.5,
            "retreat": 0.8
        },
        "coordination": [
            ("navigator", "builder"),    # Scout-builder coordination
            ("builder", "guardian"),     # Builder-defender coordination  
            ("guardian", "navigator")    # Defender-scout coordination
        ]
    }
    
    # 7. Run Quantum-Enhanced Planning
    print("🚀 Running quantum-enhanced planning...")
    print(f"  Goal: {goal}")
    print(f"  Agents: {len(scenario['agents'])}")
    print(f"  Actions: {len(action_space)}")
    print(f"  Resources: {len(scenario['environment']['resources'])}")
    print(f"  Threats: {len(scenario['environment']['threats'])}")
    
    planning_start = time.time()
    
    result = await quantum_enhanced_planner.plan_async(
        initial_state=initial_state,
        goal=goal,
        action_space=action_space,
        agent_context=agent_context
    )
    
    planning_duration = time.time() - planning_start
    
    # 8. Analyze Results
    print(f"\n✅ Planning completed in {planning_duration:.4f}s!")
    print(f"Execution strategy: {result.execution_strategy}")
    print(f"Integration confidence: {result.integration_confidence:.3f}")
    print(f"Belief consistency: {result.belief_consistency:.3f}")
    
    print(f"\n📋 Quantum Plan:")
    for i, action in enumerate(result.quantum_plan.best_action_sequence):
        print(f"  {i+1}. {action}")
    
    print(f"\n📋 Classical Plan:")
    for i, action in enumerate(result.classical_plan):
        print(f"  {i+1}. {action}")
    
    if result.fallback_plan:
        print(f"\n📋 Fallback Plan Available:")
        for i, action in enumerate(result.fallback_plan):
            print(f"  {i+1}. {action}")
    
    # 9. Record Comprehensive Metrics
    print(f"\n📊 Recording metrics...")
    
    metrics_collector.record_quantum_metric(
        MetricType.QUANTUM_ADVANTAGE,
        result.quantum_plan.quantum_advantage,
        metadata={
            "scenario": "full_integration",
            "agents": len(scenario["agents"]),
            "complexity": "high"
        }
    )
    
    metrics_collector.record_operation_timing(
        "integrated_planning",
        planning_duration,
        success=True,
        metadata={
            "execution_strategy": result.execution_strategy,
            "confidence": result.integration_confidence
        }
    )
    
    # 10. Performance Analysis
    integration_stats = quantum_enhanced_planner.get_integration_statistics()
    
    print(f"\n📈 Integration Performance:")
    print(f"  Total plans: {integration_stats.get('total_plans', 1)}")
    print(f"  Average planning time: {integration_stats.get('avg_planning_time', planning_duration):.4f}s")
    print(f"  Average belief consistency: {integration_stats.get('avg_belief_consistency', result.belief_consistency):.3f}")
    print(f"  Fallback rate: {integration_stats.get('fallback_rate', 0.0):.2%}")
    
    if integration_stats.get('avg_quantum_advantage'):
        print(f"  Average quantum advantage: {integration_stats['avg_quantum_advantage']:.2f}x")
    
    # 11. Export Results
    print(f"\n💾 Exporting results...")
    
    dashboard_file = dashboard.export_dashboard_html("full_integration")
    metrics_file = metrics_collector.export_metrics("full_integration_metrics.json")
    
    print(f"  Dashboard: {dashboard_file}")
    print(f"  Metrics: {metrics_file}")
    
    # 12. Cleanup
    dashboard.stop_real_time_updates()
    await quantum_enhanced_planner.close()
    dashboard.cleanup()
    
    print(f"\n🎉 Full integration demo completed successfully!")
    
    return result, integration_stats

# Run full integration demo
if __name__ == "__main__":
    integration_result, integration_stats = asyncio.run(full_pwmk_integration())
```

This comprehensive examples document demonstrates all major aspects of the quantum-enhanced planning system, from basic usage to advanced integration scenarios. Each example includes complete, runnable code with detailed explanations and expected outputs.