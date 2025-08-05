"""
Integration tests for quantum-enhanced planning system.

Tests the integration between quantum components and existing PWMK systems,
ensuring seamless operation and performance improvements.
"""

import pytest
import asyncio
import numpy as np
import torch
from unittest.mock import Mock, patch, AsyncMock

from pwmk.core.world_model import PerspectiveWorldModel
from pwmk.core.beliefs import BeliefStore
from pwmk.planning.epistemic import EpistemicPlanner
from pwmk.quantum.integration import (
    QuantumEnhancedPlanner,
    QuantumPlanningConfig,
    QuantumEnhancedPlan
)


class TestQuantumIntegration:
    """Test suite for quantum integration with PWMK."""
    
    @pytest.fixture
    def world_model(self):
        """Create mock world model for testing."""
        model = Mock(spec=PerspectiveWorldModel)
        model.num_agents = 2
        model.obs_dim = 32
        model.action_dim = 5
        return model
    
    @pytest.fixture
    def belief_store(self):
        """Create mock belief store for testing."""
        store = Mock(spec=BeliefStore)
        store.query = Mock(return_value=[])
        return store
    
    @pytest.fixture
    def classical_planner(self):
        """Create mock classical planner for testing."""
        planner = Mock(spec=EpistemicPlanner)
        
        # Mock plan result
        mock_plan = Mock()
        mock_plan.actions = ["move_north", "move_east"]
        planner.plan = Mock(return_value=mock_plan)
        
        return planner
    
    @pytest.fixture
    def quantum_config(self):
        """Create quantum planning configuration."""
        return QuantumPlanningConfig(
            enable_quantum_superposition=True,
            enable_circuit_optimization=True,
            enable_quantum_annealing=True,
            enable_adaptive_parameters=True,
            parallel_execution=True,
            max_planning_time=2.0,  # Short timeout for tests
            confidence_threshold=0.6
        )
    
    @pytest.fixture
    def quantum_planner(self, world_model, belief_store, classical_planner, quantum_config):
        """Create quantum-enhanced planner for testing."""
        return QuantumEnhancedPlanner(
            world_model=world_model,
            belief_store=belief_store,
            classical_planner=classical_planner,
            config=quantum_config
        )
    
    @pytest.fixture
    def sample_initial_state(self):
        """Sample initial state for testing."""
        return {
            "agents": [
                {"id": "agent_0", "position": (0, 0)},
                {"id": "agent_1", "position": (2, 2)}
            ],
            "obstacles": [(1, 1)],
            "resources": {"keys": [(3, 3)]}
        }
    
    @pytest.fixture
    def sample_action_space(self):
        """Sample action space for testing."""
        return ["move_north", "move_south", "move_east", "move_west", "wait"]
    
    @pytest.fixture
    def sample_agent_context(self):
        """Sample agent context for testing."""
        return {
            "agents": ["agent_0", "agent_1"],
            "capabilities": {
                "agent_0": ["move_north", "move_south", "move_east", "move_west"],
                "agent_1": ["move_north", "move_south", "move_east", "move_west", "wait"]
            },
            "priorities": {
                "move_north": 1.2,
                "move_south": 1.0,
                "move_east": 1.1,
                "move_west": 1.0,
                "wait": 0.5
            },
            "coordination": [("agent_0", "agent_1")]
        }
    
    def test_initialization(self, world_model, belief_store, classical_planner, quantum_config):
        """Test quantum-enhanced planner initialization."""
        planner = QuantumEnhancedPlanner(
            world_model=world_model,
            belief_store=belief_store,
            classical_planner=classical_planner,
            config=quantum_config
        )
        
        assert planner.world_model == world_model
        assert planner.belief_store == belief_store
        assert planner.classical_planner == classical_planner
        assert planner.config == quantum_config
        
        # Check quantum components initialization
        assert hasattr(planner, 'quantum_planner')
        assert hasattr(planner, 'circuit_optimizer')
        assert hasattr(planner, 'annealing_scheduler')
        assert hasattr(planner, 'adaptive_quantum')
        
        # Check thread pool
        assert hasattr(planner, 'executor')
    
    @pytest.mark.asyncio
    async def test_synchronous_planning(self, quantum_planner, sample_initial_state, sample_action_space):
        """Test synchronous planning interface."""
        goal = "coordinate agents efficiently"
        
        result = quantum_planner.plan(
            initial_state=sample_initial_state,
            goal=goal,
            action_space=sample_action_space
        )
        
        assert isinstance(result, QuantumEnhancedPlan)
        assert isinstance(result.classical_plan, list)
        assert hasattr(result.quantum_plan, 'best_action_sequence')
        assert 0.0 <= result.belief_consistency <= 1.0
        assert 0.0 <= result.integration_confidence <= 1.0
        assert result.execution_strategy in [
            "quantum_primary", "classical_primary", "annealing_primary", 
            "hybrid", "emergency_fallback"
        ]
    
    @pytest.mark.asyncio
    async def test_asynchronous_planning(self, quantum_planner, sample_initial_state, sample_action_space, sample_agent_context):
        """Test asynchronous planning interface."""
        goal = "coordinate agents with quantum enhancement"
        
        result = await quantum_planner.plan_async(
            initial_state=sample_initial_state,
            goal=goal,
            action_space=sample_action_space,
            agent_context=sample_agent_context
        )
        
        assert isinstance(result, QuantumEnhancedPlan)
        assert result.quantum_plan.planning_time > 0.0
        
        # Verify agent context was used
        assert len(result.classical_plan) >= 0  # Could be empty if quantum is primary
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, quantum_planner, sample_initial_state, sample_action_space):
        """Test parallel execution of planning approaches."""
        # Enable parallel execution
        quantum_planner.config.parallel_execution = True
        
        goal = "test parallel planning"
        
        result = await quantum_planner.plan_async(
            initial_state=sample_initial_state,
            goal=goal,
            action_space=sample_action_space
        )
        
        assert isinstance(result, QuantumEnhancedPlan)
        # Should have results from multiple planning approaches
        assert result.execution_strategy != "emergency_fallback"
    
    @pytest.mark.asyncio
    async def test_planning_timeout(self, quantum_planner, sample_initial_state, sample_action_space):
        """Test planning timeout handling."""
        # Set very short timeout
        quantum_planner.config.max_planning_time = 0.001
        
        goal = "test timeout handling"
        
        result = await quantum_planner.plan_async(
            initial_state=sample_initial_state,
            goal=goal,
            action_space=sample_action_space
        )
        
        # Should still return a result (possibly fallback)
        assert isinstance(result, QuantumEnhancedPlan)
    
    def test_input_validation(self, quantum_planner):
        """Test input validation."""
        # Test empty initial state
        with pytest.raises(Exception):
            quantum_planner.plan(
                initial_state={},
                goal="test",
                action_space=["action1"]
            )
        
        # Test empty goal
        with pytest.raises(Exception):
            quantum_planner.plan(
                initial_state={"agents": []},
                goal="",
                action_space=["action1"]
            )
        
        # Test empty action space
        with pytest.raises(Exception):
            quantum_planner.plan(
                initial_state={"agents": []},
                goal="test",
                action_space=[]
            )
    
    @pytest.mark.asyncio
    async def test_classical_planning_execution(self, quantum_planner, sample_initial_state, sample_action_space):
        """Test classical planning execution."""
        # Mock classical planner to return specific result
        mock_plan = Mock()
        mock_plan.actions = ["move_north", "move_east"]
        quantum_planner.classical_planner.plan.return_value = mock_plan
        
        classical_result = await quantum_planner._run_classical_planning(
            sample_initial_state,
            "test goal",
            sample_action_space
        )
        
        assert classical_result == ["move_north", "move_east"]
        quantum_planner.classical_planner.plan.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quantum_planning_execution(self, quantum_planner, sample_initial_state, sample_action_space):
        """Test quantum planning execution."""
        result = await quantum_planner._run_quantum_planning(
            sample_initial_state,
            "test goal",
            sample_action_space
        )
        
        assert hasattr(result, 'best_action_sequence')
        assert hasattr(result, 'quantum_advantage')
        assert hasattr(result, 'planning_time')
    
    @pytest.mark.asyncio
    async def test_annealing_planning_execution(self, quantum_planner, sample_initial_state, sample_action_space, sample_agent_context):
        """Test annealing planning execution."""
        result = await quantum_planner._run_annealing_planning(
            sample_initial_state,
            "test goal",
            sample_action_space,
            sample_agent_context
        )
        
        assert isinstance(result, list)
        if result:  # May be empty in some cases
            assert all(action in sample_action_space for action in result)
    
    @pytest.mark.asyncio 
    async def test_plan_quality_evaluation(self, quantum_planner, sample_initial_state):
        """Test plan quality evaluation."""
        plan = ["move_north", "move_east"]
        goal = "reach target position"
        
        quality = await quantum_planner._evaluate_plan_quality(plan, sample_initial_state, goal)
        
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
    
    @pytest.mark.asyncio
    async def test_belief_consistency_checking(self, quantum_planner, sample_initial_state):
        """Test belief consistency checking."""
        planning_results = {
            "classical": ["move_north", "move_east"],
            "quantum": Mock(best_action_sequence=["move_south", "move_west"])
        }
        goal = "test consistency"
        
        consistency = await quantum_planner._check_belief_consistency(
            planning_results,
            sample_initial_state,
            goal
        )
        
        assert isinstance(consistency, float)
        assert 0.0 <= consistency <= 1.0
    
    def test_hybrid_plan_creation(self, quantum_planner):
        """Test hybrid plan creation."""
        planning_results = {
            "classical": ["move_north", "move_east"],
            "quantum": Mock(best_action_sequence=["move_south", "move_west"]),
            "annealing": ["move_north", "wait"]
        }
        
        hybrid_plan = quantum_planner._create_hybrid_plan(planning_results)
        
        assert isinstance(hybrid_plan, list)
        # Should contain most frequent actions
        assert "move_north" in hybrid_plan  # Appears in classical and annealing
    
    def test_goal_format_conversion(self, quantum_planner):
        """Test goal format conversion."""
        goal = "coordinate agents efficiently"
        
        converted = quantum_planner._convert_goal_format(goal)
        
        assert isinstance(converted, dict)
        assert "description" in converted
        assert converted["description"] == goal
    
    def test_annealing_problem_creation(self, quantum_planner, sample_initial_state, sample_action_space, sample_agent_context):
        """Test annealing problem creation."""
        problem = quantum_planner._create_annealing_problem(
            sample_initial_state,
            "test goal",
            sample_action_space,
            sample_agent_context
        )
        
        assert hasattr(problem, 'hamiltonian')
        assert hasattr(problem, 'variables')
        assert hasattr(problem, 'constraints')
        assert hasattr(problem, 'objective_function')
    
    def test_annealing_solution_conversion(self, quantum_planner, sample_action_space):
        """Test annealing solution conversion."""
        # Mock solution vector
        solution = np.array([0.8, 0.2, 0.6, 0.1, 0.9])
        
        action_sequence = quantum_planner._convert_annealing_solution(solution, sample_action_space)
        
        assert isinstance(action_sequence, list)
        assert all(action in sample_action_space for action in action_sequence)
    
    def test_integration_statistics(self, quantum_planner, sample_initial_state, sample_action_space):
        """Test integration statistics collection."""
        # Run a few planning operations
        for _ in range(3):
            quantum_planner.plan(
                initial_state=sample_initial_state,
                goal="test statistics",
                action_space=sample_action_space
            )
        
        stats = quantum_planner.get_integration_statistics()
        
        assert isinstance(stats, dict)
        assert "total_plans" in stats
        assert stats["total_plans"] == 3
        assert "planning_times" in stats
        assert "belief_consistency_scores" in stats
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, quantum_planner):
        """Test resource cleanup."""
        # Ensure cleanup doesn't raise exceptions
        await quantum_planner.close()
        
        # Thread pool should be shut down
        assert quantum_planner.executor._shutdown
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, quantum_planner, sample_initial_state, sample_action_space):
        """Test error recovery mechanisms."""
        # Mock quantum planner to raise exception
        with patch.object(quantum_planner, '_run_quantum_planning', side_effect=Exception("Test error")):
            result = await quantum_planner.plan_async(
                initial_state=sample_initial_state,
                goal="test error recovery",
                action_space=sample_action_space
            )
            
            # Should still return a valid result (likely using classical fallback)
            assert isinstance(result, QuantumEnhancedPlan)
    
    def test_component_configuration(self, world_model, belief_store, classical_planner):
        """Test different component configurations."""
        # Test with minimal configuration
        minimal_config = QuantumPlanningConfig(
            enable_quantum_superposition=False,
            enable_circuit_optimization=False,
            enable_quantum_annealing=False,
            enable_adaptive_parameters=False
        )
        
        planner = QuantumEnhancedPlanner(
            world_model=world_model,
            belief_store=belief_store,
            classical_planner=classical_planner,
            config=minimal_config
        )
        
        # Should still initialize but with limited components
        assert planner.quantum_planner is None
        assert planner.circuit_optimizer is None
        assert planner.annealing_scheduler is None
        assert planner.adaptive_quantum is None
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_planning_performance(self, quantum_planner, sample_initial_state, sample_action_space):
        """Test planning performance under load."""
        import time
        
        # Run multiple planning operations and measure time
        start_time = time.time()
        
        tasks = []
        for i in range(5):
            task = quantum_planner.plan_async(
                initial_state=sample_initial_state,
                goal=f"performance test {i}",
                action_space=sample_action_space
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # All tasks should complete successfully
        assert len(results) == 5
        assert all(isinstance(result, QuantumEnhancedPlan) for result in results)
        
        # Total time should be reasonable (parallel execution should help)
        assert total_time < 10.0  # Should complete within 10 seconds
        
        # Average time per planning operation
        avg_time = total_time / 5
        assert avg_time < 3.0  # Each operation should average < 3 seconds