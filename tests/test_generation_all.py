"""
Comprehensive test suite for all three generations of PWMK implementation.
Tests Generation 1 (Basic), Generation 2 (Robust), and Generation 3 (Scalable).
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pwmk.core.beliefs import BeliefStore
from pwmk.security.input_sanitizer import SecurityError, get_sanitizer
from pwmk.security.belief_validator import get_validator
from pwmk.optimization.parallel_processing import ParallelBeliefProcessor


class TestGeneration1Basic:
    """Test Generation 1: MAKE IT WORK - Basic functionality."""
    
    def test_basic_belief_creation(self):
        """Test basic belief store creation and initialization."""
        belief_store = BeliefStore()
        
        assert belief_store.backend == "simple"
        assert len(belief_store.facts) == 0
        assert len(belief_store.rules) == 0
        assert belief_store.query_count == 0
        assert belief_store.belief_count == 0
    
    def test_add_single_belief(self):
        """Test adding a single belief."""
        belief_store = BeliefStore()
        
        belief_store.add_belief("agent_0", "has(key)")
        
        assert "agent_0" in belief_store.facts
        assert "has(key)" in belief_store.facts["agent_0"]
        assert belief_store.belief_count == 1
    
    def test_basic_query(self):
        """Test basic query functionality."""
        belief_store = BeliefStore()
        
        belief_store.add_belief("agent_0", "has(key)")
        belief_store.add_belief("agent_1", "has(treasure)")
        
        results = belief_store.query("has(X)")
        
        assert len(results) == 2
        assert belief_store.query_count == 1
        
        # Check that results contain expected bindings
        values = [r.get("X") for r in results if "X" in r]
        assert "key" in values
        assert "treasure" in values
    
    def test_empty_query(self):
        """Test empty query handling."""
        belief_store = BeliefStore()
        
        results = belief_store.query("")
        
        assert results == []
    
    def test_nested_beliefs(self):
        """Test nested belief functionality."""
        belief_store = BeliefStore()
        
        belief_store.add_nested_belief("agent_0", "agent_1", "location(treasure)")
        
        assert "agent_0" in belief_store.facts
        nested_beliefs = belief_store.get_all_beliefs("agent_0")
        assert any("believes(agent_1, location(treasure))" in b for b in nested_beliefs)


class TestGeneration2Robust:
    """Test Generation 2: MAKE IT ROBUST - Security and reliability."""
    
    def test_input_sanitization(self):
        """Test input sanitization and security."""
        belief_store = BeliefStore()
        
        # Test dangerous content blocking
        with pytest.raises(SecurityError):
            belief_store.add_belief("agent_0", "__import__('os')")
        
        with pytest.raises(SecurityError):
            belief_store.query("DROP TABLE users")
    
    def test_empty_agent_id_handling(self):
        """Test empty agent ID sanitization."""
        belief_store = BeliefStore()
        
        with pytest.raises(SecurityError):
            belief_store.add_belief("", "valid_belief")
    
    def test_malicious_query_blocking(self):
        """Test malicious query pattern blocking."""
        belief_store = BeliefStore()
        
        malicious_queries = [
            "eval()",
            "subprocess.run(['rm', '-rf', '/'])",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --"
        ]
        
        for query in malicious_queries:
            with pytest.raises(SecurityError):
                belief_store.query(query)
    
    def test_belief_syntax_validation(self):
        """Test belief syntax validation."""
        belief_store = BeliefStore()
        validator = get_validator()
        
        # Valid beliefs should pass
        valid_beliefs = [
            "has(key)",
            "at(room_1)",
            "believes(agent_1, location(treasure))"
        ]
        
        for belief in valid_beliefs:
            assert validator.validate_belief_syntax(belief)
    
    def test_error_handling_and_logging(self):
        """Test comprehensive error handling."""
        belief_store = BeliefStore()
        
        # Test that system continues to work after errors
        try:
            belief_store.add_belief("", "invalid")
        except SecurityError:
            pass
        
        # Should still work for valid operations
        belief_store.add_belief("agent_0", "has(key)")
        results = belief_store.query("has(X)")
        assert len(results) == 1
    
    def test_metrics_collection(self):
        """Test metrics collection and monitoring."""
        belief_store = BeliefStore()
        
        initial_stats = belief_store.get_performance_stats()
        assert "query_count" in initial_stats
        assert "belief_count" in initial_stats
        
        # Perform operations
        belief_store.add_belief("agent_0", "has(key)")
        belief_store.query("has(X)")
        
        final_stats = belief_store.get_performance_stats()
        assert final_stats["belief_count"] > initial_stats["belief_count"]
        assert final_stats["query_count"] > initial_stats["query_count"]
    
    def test_concurrent_safety(self):
        """Test thread safety with concurrent operations."""
        belief_store = BeliefStore()
        
        import threading
        import time
        
        def add_beliefs(agent_prefix: str, count: int):
            for i in range(count):
                try:
                    belief_store.add_belief(f"{agent_prefix}_{i}", f"fact_{i}(value)")
                    time.sleep(0.001)  # Small delay to encourage race conditions
                except Exception:
                    pass  # Ignore errors for this test
        
        # Start concurrent threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_beliefs, args=[f"agent{i}", 10])
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify system is still functional
        results = belief_store.query("fact_1(X)")
        assert isinstance(results, list)


class TestGeneration3Scalable:
    """Test Generation 3: MAKE IT SCALE - Performance and scalability."""
    
    def test_batch_add_beliefs(self):
        """Test batch belief addition."""
        belief_store = BeliefStore()
        
        batch_beliefs = [
            ("agent_0", "has(key)"),
            ("agent_1", "at(room_1)"),
            ("agent_2", "sees(treasure)"),
            ("agent_0", "knows(location)")
        ]
        
        results = belief_store.batch_add_beliefs(batch_beliefs, use_parallel=False)
        
        assert len(results) == len(batch_beliefs)
        assert all(results)  # All should succeed
        assert belief_store.belief_count == len(batch_beliefs)
    
    def test_batch_query(self):
        """Test batch query functionality."""
        belief_store = BeliefStore()
        
        # Setup data
        belief_store.add_belief("agent_0", "has(key)")
        belief_store.add_belief("agent_1", "at(room_1)")
        belief_store.add_belief("agent_2", "sees(treasure)")
        
        queries = ["has(X)", "at(Y)", "sees(Z)", "missing(W)"]
        results = belief_store.batch_query(queries, use_parallel=False)
        
        assert len(results) == len(queries)
        assert len(results[0]) == 1  # has(X) should find 1 result
        assert len(results[1]) == 1  # at(Y) should find 1 result  
        assert len(results[2]) == 1  # sees(Z) should find 1 result
        assert len(results[3]) == 0  # missing(W) should find 0 results
    
    def test_parallel_processing_setup(self):
        """Test parallel processing infrastructure."""
        from pwmk.optimization.parallel_processing import get_parallel_processor
        
        processor = get_parallel_processor()
        
        assert processor is not None
        assert processor.max_workers > 0
        assert hasattr(processor, 'submit_belief_query_batch')
        assert hasattr(processor, 'submit_belief_update_batch')
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets."""
        belief_store = BeliefStore()
        
        # Create larger dataset
        large_batch = [(f"agent_{i}", f"fact_{i}(value_{i})") for i in range(100)]
        
        start_time = time.time()
        results = belief_store.batch_add_beliefs(large_batch, use_parallel=False)
        duration = time.time() - start_time
        
        assert all(results)
        assert duration < 10.0  # Should complete within reasonable time
        assert belief_store.belief_count == len(large_batch)
        
        # Test batch queries on large dataset
        queries = [f"fact_{i}(X)" for i in range(10)]
        start_time = time.time()
        query_results = belief_store.batch_query(queries, use_parallel=False)
        query_duration = time.time() - start_time
        
        assert len(query_results) == len(queries)
        assert query_duration < 5.0  # Should complete quickly
    
    def test_cache_integration(self):
        """Test caching integration."""
        belief_store = BeliefStore()
        
        stats = belief_store.get_performance_stats()
        assert "cache_stats" in stats
        assert "caching_enabled" in stats
    
    def test_auto_scaling_integration(self):
        """Test auto-scaling integration."""
        from pwmk.optimization.auto_scaling import get_auto_scaler
        
        auto_scaler = get_auto_scaler()
        
        assert auto_scaler is not None
        assert hasattr(auto_scaler, 'start_monitoring')
        assert hasattr(auto_scaler, 'stop_monitoring')
        assert hasattr(auto_scaler, 'get_stats')
    
    def test_load_balancing(self):
        """Test load balancing capabilities."""
        from pwmk.optimization.parallel_processing import LoadBalancer
        
        # Create multiple belief stores
        stores = [BeliefStore() for _ in range(3)]
        balancer = LoadBalancer(stores, strategy="round_robin")
        
        # Test distribution
        selected_stores = []
        for _ in range(6):
            store, index = balancer.get_next_store()
            selected_stores.append(index)
        
        # Should distribute evenly with round-robin
        assert len(set(selected_stores)) == len(stores)
    
    def test_performance_stats_comprehensive(self):
        """Test comprehensive performance statistics."""
        belief_store = BeliefStore()
        
        # Perform various operations
        belief_store.add_belief("agent_0", "has(key)")
        belief_store.query("has(X)")
        belief_store.batch_add_beliefs([("agent_1", "at(room)")], use_parallel=False)
        belief_store.batch_query(["at(Y)"], use_parallel=False)
        
        stats = belief_store.get_performance_stats()
        
        # Verify all expected stats are present
        required_stats = [
            "query_count", "belief_count", "total_agents", "total_facts",
            "caching_enabled", "parallel_enabled", "parallel_stats", "cache_stats"
        ]
        
        for stat in required_stats:
            assert stat in stats
        
        assert stats["total_agents"] > 0
        assert stats["total_facts"] > 0


class TestQualityGates:
    """Test overall quality gates and system integration."""
    
    def test_system_integration(self):
        """Test overall system integration."""
        belief_store = BeliefStore()
        
        # Test complete workflow
        # 1. Add beliefs
        belief_store.add_belief("agent_0", "has(key)")
        belief_store.add_belief("agent_1", "at(room_1)")
        
        # 2. Query beliefs
        results = belief_store.query("has(X)")
        assert len(results) == 1
        
        # 3. Batch operations
        batch_beliefs = [("agent_2", "sees(treasure)"), ("agent_3", "knows(secret)")]
        batch_results = belief_store.batch_add_beliefs(batch_beliefs, use_parallel=False)
        assert all(batch_results)
        
        # 4. Complex queries
        all_results = belief_store.batch_query(["has(X)", "at(Y)", "sees(Z)"], use_parallel=False)
        assert len(all_results) == 3
        
        # 5. Verify final state
        final_stats = belief_store.get_performance_stats()
        assert final_stats["total_agents"] == 4
        assert final_stats["total_facts"] == 4
    
    def test_error_recovery(self):
        """Test system recovery from errors."""
        belief_store = BeliefStore()
        
        # Add valid belief
        belief_store.add_belief("agent_0", "has(key)")
        
        # Try invalid operations
        try:
            belief_store.add_belief("", "invalid")
        except SecurityError:
            pass
        
        try:
            belief_store.query("__import__('os')")
        except SecurityError:
            pass
        
        # System should still work
        results = belief_store.query("has(X)")
        assert len(results) == 1
        
        belief_store.add_belief("agent_1", "at(room)")
        results = belief_store.query("at(Y)")
        assert len(results) == 1
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        belief_store = BeliefStore()
        
        # Add substantial amount of data
        for i in range(1000):
            belief_store.add_belief(f"agent_{i % 10}", f"fact_{i}(value)")
        
        # Query the data
        for i in range(100):
            belief_store.query(f"fact_{i}(X)")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_security_comprehensive(self):
        """Comprehensive security testing."""
        belief_store = BeliefStore()
        sanitizer = get_sanitizer()
        
        # Test all dangerous patterns
        dangerous_inputs = [
            "__import__('os')",
            "eval('malicious_code')",
            "exec('rm -rf /')",
            "subprocess.run(['cat', '/etc/passwd'])",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd"
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(SecurityError):
                belief_store.add_belief("agent_0", dangerous_input)
            
            with pytest.raises(SecurityError):
                belief_store.query(dangerous_input)


@pytest.mark.asyncio
async def test_async_processing():
    """Test asynchronous processing capabilities."""
    from pwmk.optimization.parallel_processing import AsyncBeliefProcessor
    
    processor = AsyncBeliefProcessor()
    
    # Create multiple belief stores
    stores = [BeliefStore() for _ in range(3)]
    for i, store in enumerate(stores):
        store.add_belief(f"agent_{i}", f"fact_{i}(value)")
    
    queries = [f"fact_{i}(X)" for i in range(3)]
    
    # Test async processing
    results = await processor.process_belief_queries_async(stores, queries)
    
    assert len(results) == 3
    assert all(isinstance(r, list) for r in results if r is not None)


def test_performance_benchmarks():
    """Test performance benchmarks meet requirements."""
    belief_store = BeliefStore()
    
    # Benchmark single operations
    start_time = time.time()
    for i in range(1000):
        belief_store.add_belief(f"agent_{i % 10}", f"fact_{i}(value)")
    single_add_duration = time.time() - start_time
    
    # Should handle 1000 adds in reasonable time (< 5 seconds)
    assert single_add_duration < 5.0
    
    # Benchmark batch operations
    batch_beliefs = [(f"batch_agent_{i}", f"batch_fact_{i}(value)") for i in range(1000)]
    
    start_time = time.time()
    results = belief_store.batch_add_beliefs(batch_beliefs, use_parallel=False)
    batch_add_duration = time.time() - start_time
    
    # Batch operations should be faster or comparable
    assert batch_add_duration < single_add_duration + 2.0
    assert all(results)
    
    # Benchmark queries
    start_time = time.time()
    for i in range(100):
        belief_store.query(f"fact_{i}(X)")
    query_duration = time.time() - start_time
    
    # Should handle 100 queries quickly (< 2 seconds)
    assert query_duration < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])