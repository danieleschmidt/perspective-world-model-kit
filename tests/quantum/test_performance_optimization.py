"""
Tests for quantum performance optimization.

Comprehensive test suite for performance optimization, resource management,
and auto-scaling capabilities.
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch
import psutil

from pwmk.quantum.performance import (
    QuantumPerformanceOptimizer,
    PerformanceConfig,
    ResourceMetrics,
    OptimizationResult
)


class TestQuantumPerformanceOptimizer:
    """Test suite for QuantumPerformanceOptimizer."""
    
    @pytest.fixture
    def perf_config(self):
        """Create performance configuration for testing."""
        return PerformanceConfig(
            enable_parallel_processing=True,
            enable_gpu_acceleration=False,  # Disable GPU for tests
            enable_adaptive_batching=True,
            max_workers=2,  # Small number for tests
            batch_size_range=(1, 4),
            memory_limit_gb=1.0,  # Low limit for testing
            auto_scaling_enabled=True
        )
    
    @pytest.fixture
    def optimizer(self, perf_config):
        """Create performance optimizer for testing."""
        return QuantumPerformanceOptimizer(config=perf_config)
    
    @pytest.fixture
    def mock_operations(self):
        """Create mock quantum operations for testing."""
        def mock_op1(data_size, multiplier=1.0):
            time.sleep(0.01)  # Simulate computation
            return {"result": np.random.random(data_size) * multiplier}
        
        def mock_op2(complexity, iterations=1):
            time.sleep(complexity * 0.005 * iterations)
            return {"computation": complexity * iterations}
        
        def mock_op3(array_size):
            return {"array": np.ones(array_size)}
        
        return [mock_op1, mock_op2, mock_op3]
    
    @pytest.fixture
    def mock_operation_args(self):
        """Create mock operation arguments."""
        return [
            (5, 2.0),      # For mock_op1
            (0.5, 3),      # For mock_op2
            (10,)          # For mock_op3
        ]
    
    def test_initialization(self, perf_config):
        """Test optimizer initialization."""
        optimizer = QuantumPerformanceOptimizer(config=perf_config)
        
        assert optimizer.config == perf_config
        assert optimizer.max_workers == 2
        assert hasattr(optimizer, 'thread_pool')
        assert hasattr(optimizer, 'process_pool')
        assert optimizer.current_batch_size == 1  # Start of range
        assert optimizer.memory_monitor_active
    
    def test_initialization_default_config(self):
        """Test optimizer initialization with default config."""
        optimizer = QuantumPerformanceOptimizer()
        
        assert isinstance(optimizer.config, PerformanceConfig)
        assert optimizer.max_workers > 0
        assert hasattr(optimizer, 'operation_stats')
    
    @pytest.mark.asyncio
    async def test_optimize_quantum_operations_sequential(self, optimizer, mock_operations, mock_operation_args):
        """Test sequential optimization strategy."""
        # Force sequential strategy
        result = await optimizer.optimize_quantum_operations(
            operations=mock_operations,
            operation_args=mock_operation_args,
            optimization_hints={"force_sequential": True}
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.speedup_factor >= 1.0
        assert result.optimization_time > 0.0
        assert isinstance(result.resource_utilization, ResourceMetrics)
    
    @pytest.mark.asyncio
    async def test_optimize_quantum_operations_parallel_cpu(self, optimizer, mock_operations, mock_operation_args):
        """Test parallel CPU optimization strategy."""
        result = await optimizer.optimize_quantum_operations(
            operations=mock_operations,
            operation_args=mock_operation_args,
            optimization_hints={"prefer_parallel": True}
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.speedup_factor >= 1.0
        assert optimizer.operation_stats["parallel_operations"] >= len(mock_operations)
    
    @pytest.mark.asyncio
    async def test_optimize_quantum_operations_batched(self, optimizer):
        """Test batched optimization strategy."""
        # Create larger set of operations to trigger batching
        def simple_op(x):
            return x * 2
        
        operations = [simple_op] * 10
        operation_args = [(i,) for i in range(10)]
        
        result = await optimizer.optimize_quantum_operations(
            operations=operations,
            operation_args=operation_args,
            optimization_hints={"prefer_batching": True}
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.speedup_factor >= 1.0
    
    @pytest.mark.asyncio
    async def test_optimize_quantum_operations_adaptive(self, optimizer):
        """Test adaptive optimization strategy."""
        def variable_complexity_op(complexity):
            time.sleep(complexity * 0.001)
            return {"result": complexity}
        
        operations = [variable_complexity_op] * 8
        operation_args = [(0.1 + i * 0.1,) for i in range(8)]
        
        result = await optimizer.optimize_quantum_operations(
            operations=operations,
            operation_args=operation_args
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.speedup_factor >= 1.0
    
    def test_determine_optimization_strategy(self, optimizer):
        """Test optimization strategy determination."""
        # Test with different operation counts and system states
        
        # Single operation
        strategy = optimizer._determine_optimization_strategy([lambda: None], None)
        assert strategy == "sequential"
        
        # Small number of operations
        operations = [lambda: None] * 3
        strategy = optimizer._determine_optimization_strategy(operations, None)
        assert strategy in ["parallel_cpu", "parallel_gpu"]
        
        # Large number of operations
        operations = [lambda: None] * 25
        strategy = optimizer._determine_optimization_strategy(operations, None)
        assert strategy in ["adaptive", "batched"]
        
        # With hints
        strategy = optimizer._determine_optimization_strategy(
            [lambda: None] * 5,
            {"force_sequential": True}
        )
        assert strategy == "sequential"
    
    def test_get_optimal_batch_size(self, optimizer):
        """Test optimal batch size determination."""
        # Initially should return current batch size
        initial_batch_size = optimizer._get_optimal_batch_size(10)
        assert initial_batch_size == optimizer.current_batch_size
        
        # Add some performance history
        optimizer.batch_performance_history.extend([
            {"batch_size": 2, "execution_time": 0.1, "throughput": 20.0},
            {"batch_size": 4, "execution_time": 0.15, "throughput": 26.7},
            {"batch_size": 2, "execution_time": 0.12, "throughput": 16.7}
        ])
        
        optimal_size = optimizer._get_optimal_batch_size(10)
        assert 1 <= optimal_size <= 4  # Within configured range
    
    def test_get_current_system_load(self, optimizer):
        """Test system load calculation."""
        load = optimizer._get_current_system_load()
        
        assert isinstance(load, float)
        assert 0.0 <= load <= 1.0
        assert len(optimizer.load_history) > 0
    
    def test_get_resource_metrics(self, optimizer):
        """Test resource metrics collection."""
        metrics = optimizer._get_resource_metrics()
        
        assert isinstance(metrics, ResourceMetrics)
        assert 0.0 <= metrics.cpu_percent <= 100.0
        assert 0.0 <= metrics.memory_percent <= 100.0
        assert 0.0 <= metrics.gpu_memory_percent <= 100.0
        assert metrics.active_threads >= 0
        assert metrics.quantum_operations_per_second >= 0.0
        assert 0.0 <= metrics.cache_hit_rate <= 1.0
    
    def test_calculate_ops_per_second(self, optimizer):
        """Test operations per second calculation."""
        # Initially should be 0
        ops_per_sec = optimizer._calculate_ops_per_second()
        assert ops_per_sec == 0.0
        
        # Add some timestamps
        if not hasattr(optimizer, '_ops_timestamps'):
            optimizer._ops_timestamps = []
        
        current_time = time.time()
        for i in range(5):
            optimizer._ops_timestamps.append(current_time - (5 - i))
        
        ops_per_sec = optimizer._calculate_ops_per_second()
        assert ops_per_sec > 0.0
    
    def test_memory_optimization(self, optimizer):
        """Test memory optimization functionality."""
        # Trigger memory optimization
        optimizer._optimize_memory_usage()
        
        # Should increment memory optimization counter
        assert optimizer.operation_stats["memory_optimizations"] > 0
    
    def test_emergency_memory_cleanup(self, optimizer):
        """Test emergency memory cleanup."""
        original_batch_size = optimizer.current_batch_size
        
        # Trigger emergency cleanup
        optimizer._emergency_memory_cleanup()
        
        # Batch size should be reduced to minimum
        min_batch, _ = optimizer.config.batch_size_range
        assert optimizer.current_batch_size == min_batch
        assert optimizer.operation_stats["memory_optimizations"] > 0
    
    @pytest.mark.asyncio
    async def test_execute_sequential(self, optimizer):
        """Test sequential execution."""
        def simple_op(x):
            return x * 2
        
        operations = [simple_op, simple_op, simple_op]
        operation_args = [(1,), (2,), (3,)]
        
        results = await optimizer._execute_sequential(operations, operation_args)
        
        assert len(results) == 3
        assert results == [2, 4, 6]
    
    @pytest.mark.asyncio
    async def test_execute_parallel_cpu(self, optimizer):
        """Test parallel CPU execution."""
        def simple_op(x):
            time.sleep(0.01)  # Small delay
            return x * 2
        
        operations = [simple_op, simple_op, simple_op]
        operation_args = [(1,), (2,), (3,)]
        
        start_time = time.time()
        results = await optimizer._execute_parallel_cpu(operations, operation_args)
        execution_time = time.time() - start_time
        
        assert len(results) == 3
        assert results == [2, 4, 6]
        # Should be faster than sequential (though timing can vary)
        assert execution_time < 0.1  # Should complete quickly with parallelization
    
    @pytest.mark.asyncio
    async def test_execute_batched(self, optimizer):
        """Test batched execution."""
        def simple_op(x):
            return x * 2
        
        operations = [simple_op] * 8
        operation_args = [(i,) for i in range(8)]
        
        results = await optimizer._execute_batched(operations, operation_args)
        
        assert len(results) == 8
        assert results == [i * 2 for i in range(8)]
        
        # Should have batch performance history
        assert len(optimizer.batch_performance_history) > 0
    
    @pytest.mark.asyncio
    async def test_execute_adaptive(self, optimizer):
        """Test adaptive execution."""
        def variable_op(x, delay=0.001):
            time.sleep(delay)
            return x * 2
        
        operations = [variable_op] * 6
        operation_args = [(i, 0.001) for i in range(6)]
        
        results = await optimizer._execute_adaptive(operations, operation_args)
        
        assert len(results) == 6
        assert results == [i * 2 for i in range(6)]
    
    def test_error_handling_in_operations(self, optimizer):
        """Test error handling in quantum operations."""
        def failing_op():
            raise ValueError("Test error")
        
        def working_op():
            return "success"
        
        # Test with asyncio.run to handle the async method
        async def run_test():
            operations = [failing_op, working_op]
            operation_args = [(), ()]
            
            results = await optimizer._execute_sequential(operations, operation_args)
            
            # First operation should return None due to error
            # Second operation should succeed
            assert results[0] is None
            assert results[1] == "success"
        
        asyncio.run(run_test())
    
    def test_performance_statistics(self, optimizer):
        """Test performance statistics collection."""
        # Add some mock statistics
        optimizer.operation_stats["total_operations"] = 100
        optimizer.operation_stats["parallel_operations"] = 80
        optimizer.operation_stats["gpu_operations"] = 20
        
        # Add batch performance history
        optimizer.batch_performance_history.extend([
            {"batch_size": 4, "execution_time": 0.1, "throughput": 40.0},
            {"batch_size": 6, "execution_time": 0.12, "throughput": 50.0}
        ])
        
        # Add load history
        optimizer.load_history.extend([0.3, 0.4, 0.5])
        
        stats = optimizer.get_performance_statistics()
        
        assert isinstance(stats, dict)
        assert stats["total_operations"] == 100
        assert stats["parallel_operations"] == 80
        assert stats["gpu_operations"] == 20
        assert "resource_utilization" in stats
        assert "batch_performance" in stats
        assert "system_load" in stats
        assert "configuration" in stats
    
    @pytest.mark.asyncio
    async def test_cleanup(self, optimizer):
        """Test resource cleanup."""
        # Add some operations to create activity
        optimizer.operation_stats["total_operations"] = 5
        
        # Cleanup should not raise exceptions
        await optimizer.cleanup()
        
        # Memory monitor should be stopped
        assert not optimizer.memory_monitor_active
    
    def test_memory_monitoring_thread(self, optimizer):
        """Test memory monitoring thread."""
        # Memory monitor should be running
        assert optimizer.memory_monitor_active
        assert optimizer.memory_monitor.is_alive()
        
        # Stop monitoring
        optimizer.memory_monitor_active = False
        
        # Give thread time to stop
        time.sleep(0.1)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_scale_optimization(self, optimizer):
        """Test optimization with large number of operations."""
        def simple_op(x):
            return x ** 2
        
        # Large number of operations
        num_ops = 50
        operations = [simple_op] * num_ops
        operation_args = [(i,) for i in range(num_ops)]
        
        start_time = time.time()
        result = await optimizer.optimize_quantum_operations(
            operations=operations,
            operation_args=operation_args
        )
        total_time = time.time() - start_time
        
        assert isinstance(result, OptimizationResult)
        assert result.speedup_factor >= 1.0
        assert total_time < 5.0  # Should complete within reasonable time
        
        # Check that operations were executed correctly
        assert optimizer.operation_stats["total_operations"] >= num_ops
    
    @pytest.mark.asyncio
    async def test_concurrent_optimization_calls(self, optimizer):
        """Test concurrent optimization calls."""
        def simple_op(x):
            time.sleep(0.01)
            return x + 1
        
        # Create multiple concurrent optimization tasks
        async def single_optimization(offset):
            operations = [simple_op] * 3
            operation_args = [(i + offset,) for i in range(3)]
            return await optimizer.optimize_quantum_operations(operations, operation_args)
        
        # Run multiple optimizations concurrently
        tasks = [single_optimization(i * 10) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        assert all(isinstance(result, OptimizationResult) for result in results)
        assert all(result.speedup_factor >= 1.0 for result in results)