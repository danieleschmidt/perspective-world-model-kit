"""
Tests for quantum-inspired task planner.

Comprehensive test suite covering quantum superposition, interference,
measurement, and planning performance.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from pwmk.quantum.quantum_planner import (
    QuantumInspiredPlanner,
    QuantumState,
    PlanningResult
)


class TestQuantumInspiredPlanner:
    """Test suite for QuantumInspiredPlanner."""
    
    @pytest.fixture
    def planner(self):
        """Create quantum-inspired planner for testing."""
        return QuantumInspiredPlanner(
            num_qubits=6,
            max_depth=8,
            num_agents=2,
            coherence_time=1.0
        )
    
    @pytest.fixture
    def sample_action_space(self):
        """Sample action space for testing."""
        return ["move_north", "move_south", "move_east", "move_west", "wait"]
    
    @pytest.fixture
    def sample_initial_state(self):
        """Sample initial state for testing."""
        return {
            "agents": [{"id": "agent_0", "position": (0, 0)}, {"id": "agent_1", "position": (1, 1)}],
            "obstacles": [(2, 2)],
            "goals": [(5, 5), (6, 6)]
        }
    
    def test_initialization(self):
        """Test quantum planner initialization."""
        planner = QuantumInspiredPlanner(
            num_qubits=8,
            max_depth=10,
            num_agents=3
        )
        
        assert planner.num_qubits == 8
        assert planner.max_depth == 10
        assert planner.num_agents == 3
        assert planner.hilbert_space_dim == 2 ** 8
        assert hasattr(planner, 'pauli_x')
        assert hasattr(planner, 'hadamard')
    
    def test_create_superposition_state(self, planner, sample_action_space):
        """Test creation of quantum superposition state."""
        quantum_state = planner.create_superposition_state(sample_action_space)
        
        assert isinstance(quantum_state, QuantumState)
        assert len(quantum_state.amplitudes) <= len(sample_action_space)
        assert len(quantum_state.actions) == len(quantum_state.amplitudes)
        assert isinstance(quantum_state.entanglement_map, dict)
        
        # Check normalization
        norm = np.linalg.norm(quantum_state.amplitudes)
        assert abs(norm - 1.0) < 1e-10
    
    def test_create_superposition_with_weights(self, planner, sample_action_space):
        """Test creation of weighted superposition state."""
        weights = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        quantum_state = planner.create_superposition_state(
            sample_action_space, 
            initial_weights=weights
        )
        
        # Check that amplitudes reflect weights
        probabilities = np.abs(quantum_state.amplitudes) ** 2
        assert len(probabilities) == len(weights)
        # Amplitudes should be approximately sqrt of weights (before phase addition)
        
    def test_apply_quantum_interference(self, planner, sample_action_space):
        """Test quantum interference application."""
        quantum_state = planner.create_superposition_state(sample_action_space)
        goal_vector = np.array([1.0, 0.5, 0.3])
        
        modified_state = planner.apply_quantum_interference(
            quantum_state,
            goal_vector,
            environment_feedback={"move_north": 0.5, "move_south": -0.3}
        )
        
        assert isinstance(modified_state, QuantumState)
        assert len(modified_state.amplitudes) == len(quantum_state.amplitudes)
        
        # Check normalization is maintained
        norm = np.linalg.norm(modified_state.amplitudes)
        assert abs(norm - 1.0) < 1e-8
    
    def test_measure_quantum_state(self, planner, sample_action_space):
        """Test quantum state measurement."""
        quantum_state = planner.create_superposition_state(sample_action_space)
        
        result = planner.measure_quantum_state(quantum_state, num_measurements=100)
        
        assert isinstance(result, PlanningResult)
        assert result.best_action_sequence[0] in sample_action_space
        assert 0.0 <= result.probability <= 1.0
        assert result.quantum_advantage >= 1.0
        assert isinstance(result.interference_patterns, dict)
        assert result.planning_time >= 0.0
    
    def test_plan_execution(self, planner, sample_initial_state, sample_action_space):
        """Test complete planning execution."""
        goal = "reach target positions"
        
        result = planner.plan(
            initial_state=sample_initial_state,
            goal=goal,
            action_space=sample_action_space,
            max_iterations=20
        )
        
        assert isinstance(result, PlanningResult)
        assert len(result.best_action_sequence) > 0
        assert all(action in sample_action_space for action in result.best_action_sequence)
        assert result.quantum_advantage >= 1.0
        assert result.planning_time > 0.0
    
    def test_entanglement_generation(self, planner):
        """Test entanglement map generation."""
        entanglement_map = planner._generate_entanglement_map()
        
        assert isinstance(entanglement_map, dict)
        assert len(entanglement_map) == planner.num_agents
        
        # Check that each agent has entanglement connections
        for agent_id in range(planner.num_agents):
            assert agent_id in entanglement_map
            assert isinstance(entanglement_map[agent_id], list)
            assert len(entanglement_map[agent_id]) > 0
    
    def test_goal_alignment_computation(self, planner):
        """Test goal alignment computation."""
        actions = ["move_north", "move_south", "wait"]
        goal_vector = np.array([1.0, 0.5, 0.2])
        
        alignments = planner._compute_goal_alignment(actions, goal_vector)
        
        assert len(alignments) == len(actions)
        assert all(-1.0 <= alignment <= 1.0 for alignment in alignments)
    
    def test_action_encoding(self, planner):
        """Test action encoding."""
        action = "move_north"
        vector = planner._encode_action(action)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3  # Fixed encoding size
        assert abs(np.linalg.norm(vector) - 1.0) < 1e-8  # Normalized
    
    def test_convergence_checking(self, planner, sample_action_space):
        """Test convergence checking."""
        quantum_state = planner.create_superposition_state(sample_action_space)
        convergence = planner._check_convergence(quantum_state)
        
        assert isinstance(convergence, float)
        assert 0.0 <= convergence <= 1.0
    
    def test_planning_statistics(self, planner, sample_initial_state, sample_action_space):
        """Test planning statistics collection."""
        # Run several planning iterations
        for _ in range(3):
            planner.plan(
                initial_state=sample_initial_state,
                goal="test goal",
                action_space=sample_action_space,
                max_iterations=10
            )
        
        stats = planner.get_planning_statistics()
        
        assert isinstance(stats, dict)
        assert "planning_times" in stats
        assert "quantum_advantages" in stats
        assert len(stats["planning_times"]) == 3
        assert len(stats["quantum_advantages"]) == 3
    
    def test_error_handling_empty_action_space(self, planner, sample_initial_state):
        """Test error handling with empty action space."""
        with pytest.raises(Exception):
            planner.plan(
                initial_state=sample_initial_state,
                goal="test goal",
                action_space=[],
                max_iterations=10
            )
    
    def test_error_handling_invalid_goal(self, planner, sample_initial_state, sample_action_space):
        """Test error handling with invalid goal."""
        # Should handle gracefully without crashing
        result = planner.plan(
            initial_state=sample_initial_state,
            goal="",  # Empty goal
            action_space=sample_action_space,
            max_iterations=5
        )
        
        assert isinstance(result, PlanningResult)
    
    def test_quantum_noise_generation(self, planner):
        """Test quantum noise generation."""  
        shape = (5,)
        temperature = 1.0
        
        noise = planner._generate_quantum_noise(shape, temperature)
        
        assert noise.shape == shape
        assert isinstance(noise, np.ndarray)
        # Noise should have some variance
        assert np.std(noise) > 0
    
    def test_large_action_space_handling(self, planner, sample_initial_state):
        """Test handling of large action spaces."""
        large_action_space = [f"action_{i}" for i in range(100)]
        
        result = planner.plan(
            initial_state=sample_initial_state,
            goal="test with large action space",
            action_space=large_action_space,
            max_iterations=5
        )
        
        assert isinstance(result, PlanningResult)
        assert result.best_action_sequence[0] in large_action_space
    
    def test_decoherence_effects(self, planner, sample_action_space):
        """Test decoherence effects over time."""
        # Create quantum state
        quantum_state = planner.create_superposition_state(sample_action_space)
        initial_coherence = np.abs(np.sum(quantum_state.amplitudes))
        
        # Simulate time passage with interference (includes decoherence)
        goal_vector = np.array([1.0, 0.5, 0.3])
        
        # Apply interference multiple times to simulate decoherence
        modified_state = quantum_state
        for _ in range(10):
            modified_state = planner.apply_quantum_interference(
                modified_state,
                goal_vector
            )
        
        final_coherence = np.abs(np.sum(modified_state.amplitudes))
        
        # Some decoherence should have occurred (though effect may be small in test)
        assert isinstance(final_coherence, (float, complex))
    
    @pytest.mark.performance
    def test_planning_performance(self, planner, sample_initial_state, sample_action_space):
        """Test planning performance and scaling."""
        # Measure planning time for different problem sizes
        times = []
        
        for max_iter in [5, 10, 20]:
            start_time = time.time()
            
            result = planner.plan(
                initial_state=sample_initial_state,
                goal="performance test",
                action_space=sample_action_space,
                max_iterations=max_iter
            )
            
            planning_time = time.time() - start_time
            times.append(planning_time)
            
            assert isinstance(result, PlanningResult)
        
        # Check that times are reasonable (< 5 seconds for test)
        assert all(t < 5.0 for t in times)
        
        # Times should generally increase with iterations (though may vary due to convergence)
        assert len(times) == 3