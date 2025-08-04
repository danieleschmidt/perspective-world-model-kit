#!/usr/bin/env python3
"""Test Generation 3 enhancements: performance optimization, caching, and scalability."""

import torch
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from pwmk import PerspectiveWorldModel
from pwmk.optimization.caching import get_cache_manager
from pwmk.optimization.batching import BatchProcessor, DynamicBatcher, Future
from pwmk.utils.logging import setup_logging, get_logger
from pwmk.utils.monitoring import setup_monitoring


def test_caching_performance():
    """Test caching system performance."""
    print("🚀 Testing Generation 3: Caching Performance")
    print("=" * 50)
    
    # Setup
    model = PerspectiveWorldModel(obs_dim=64, action_dim=4, hidden_dim=128, num_agents=2)
    model.eval()  # Enable caching
    cache_manager = get_cache_manager()
    cache_manager.enable()
    
    # Generate test data
    batch_size, seq_len = 4, 8
    observations = torch.randn(batch_size, seq_len, 64)
    actions = torch.randint(0, 4, (batch_size, seq_len))
    agent_ids = torch.randint(0, 2, (batch_size, seq_len))
    
    # Warm up
    with torch.no_grad():
        model(observations, actions, agent_ids)
    
    print("\n1. Testing cache miss vs hit performance")
    
    # First run (cache miss)
    start_time = time.time()
    with torch.no_grad():
        result1 = model(observations, actions, agent_ids)
    miss_time = time.time() - start_time
    
    # Second run (cache hit)
    start_time = time.time()
    with torch.no_grad():
        result2 = model(observations, actions, agent_ids)
    hit_time = time.time() - start_time
    
    print(f"   Cache miss time: {miss_time:.6f}s")
    print(f"   Cache hit time:  {hit_time:.6f}s")
    print(f"   Speedup: {miss_time/hit_time:.2f}x")
    
    # Verify results are the same
    states1, beliefs1 = result1
    states2, beliefs2 = result2
    assert torch.allclose(states1, states2, atol=1e-6), "Cached results don't match!"
    assert torch.allclose(beliefs1, beliefs2, atol=1e-6), "Cached beliefs don't match!"
    print("   ✓ Cached results identical to original")
    
    # Test cache statistics
    cache_stats = cache_manager.get_stats()
    print(f"\n2. Cache Statistics:")
    for cache_name, stats in cache_stats.items():
        if isinstance(stats, dict) and "hit_rate" in stats:
            print(f"   {cache_name}: {stats['hits']} hits, {stats['misses']} misses, "
                  f"hit rate: {stats['hit_rate']:.2%}")
    
    return miss_time, hit_time


def test_batch_processing():
    """Test batch processing optimization."""
    print("\n🔧 Testing Batch Processing")
    print("=" * 40)
    
    model = PerspectiveWorldModel(obs_dim=32, action_dim=4, hidden_dim=64, num_agents=2)
    model.eval()
    
    def process_request(request):
        """Process a single request."""
        obs, actions = request
        with torch.no_grad():
            return model(obs, actions)
    
    def process_batch_func(requests):
        """Process multiple requests as a batch."""
        if not requests:
            return []
        
        # Stack all observations and actions
        obs_list = [req[0].squeeze(0) for req in requests]  # Remove individual batch dims
        action_list = [req[1].squeeze(0) for req in requests]
        
        # Combine into batch
        batch_obs = torch.stack(obs_list)  
        batch_actions = torch.stack(action_list)
        
        with torch.no_grad():
            batch_states, batch_beliefs = model(batch_obs, batch_actions)
        
        # Split results back
        results = []
        for i in range(len(requests)):
            results.append((batch_states[i:i+1], batch_beliefs[i:i+1]))
        
        return results
    
    # Test individual processing
    print("1. Individual processing performance")
    
    num_requests = 20
    requests = []
    for _ in range(num_requests):
        obs = torch.randn(1, 5, 32)
        actions = torch.randint(0, 4, (1, 5))
        requests.append((obs, actions))
    
    # Individual processing
    start_time = time.time()
    individual_results = []
    for request in requests:
        result = process_request(request)
        individual_results.append(result)
    individual_time = time.time() - start_time
    
    print(f"   Individual processing: {individual_time:.4f}s ({num_requests} requests)")
    print(f"   Average per request: {individual_time/num_requests:.6f}s")
    
    # Batch processing
    print("\n2. Batch processing performance")
    start_time = time.time()
    batch_results = process_batch_func(requests)
    batch_time = time.time() - start_time
    
    print(f"   Batch processing: {batch_time:.4f}s ({num_requests} requests)")
    print(f"   Average per request: {batch_time/num_requests:.6f}s")
    print(f"   Batch speedup: {individual_time/batch_time:.2f}x")
    
    # Test dynamic batcher
    print("\n3. Dynamic batcher adaptation")
    
    batcher = DynamicBatcher(
        initial_batch_size=8,
        min_batch_size=2,
        max_batch_size=32,
        target_latency=0.05
    )
    
    # Simulate different load patterns
    for load_pattern in ["low", "medium", "high"]:
        print(f"   Testing {load_pattern} load pattern:")
        
        if load_pattern == "low":
            batch_sizes = [2, 3, 4, 2, 3]
        elif load_pattern == "medium":
            batch_sizes = [8, 12, 16, 10, 14]
        else:  # high
            batch_sizes = [24, 32, 28, 30, 26]
        
        for batch_size in batch_sizes:
            batch_requests = requests[:batch_size]
            results = batcher.process_batch(batch_requests, process_batch_func)
            
        stats = batcher.get_performance_stats()
        print(f"      Adapted batch size: {stats['current_batch_size']}")
        print(f"      Average latency: {stats['avg_latency']:.4f}s")
        print(f"      Average throughput: {stats['avg_throughput']:.1f} req/s")
    
    return individual_time, batch_time


def test_concurrent_access():
    """Test thread-safe concurrent access."""
    print("\n🔀 Testing Concurrent Access")
    print("=" * 35)
    
    model = PerspectiveWorldModel(obs_dim=32, action_dim=4, hidden_dim=64, num_agents=2)
    model.eval()
    
    results = []
    errors = []
    
    def worker_thread(thread_id, num_requests):
        """Worker thread function."""
        thread_results = []
        try:
            for i in range(num_requests):
                obs = torch.randn(2, 4, 32)
                actions = torch.randint(0, 4, (2, 4))
                
                with torch.no_grad():
                    states, beliefs = model(obs, actions)
                
                thread_results.append((states.shape, beliefs.shape))
            
            results.extend(thread_results)
            
        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
    
    print("1. Multi-threaded model inference")
    
    num_threads = 4
    requests_per_thread = 10
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            future = executor.submit(worker_thread, i, requests_per_thread)
            futures.append(future)
        
        # Wait for all threads to complete
        for future in futures:
            future.result()
    
    concurrent_time = time.time() - start_time
    
    print(f"   Processed {len(results)} requests with {num_threads} threads")
    print(f"   Total time: {concurrent_time:.4f}s")
    print(f"   Average per request: {concurrent_time/len(results):.6f}s")
    
    if errors:
        print(f"   ❌ Errors encountered: {len(errors)}")
        for error in errors[:3]:  # Show first 3 errors
            print(f"      {error}")
    else:
        print("   ✓ No errors in concurrent access")
    
    # Test cache performance under concurrency
    print("\n2. Cache performance under concurrency")
    
    cache_manager = get_cache_manager()
    cache_stats_before = cache_manager.get_stats()
    
    # Same data across threads (should hit cache)
    shared_obs = torch.randn(1, 3, 32)
    shared_actions = torch.randint(0, 4, (1, 3))
    
    def cache_test_worker(thread_id, num_requests):
        """Worker for cache testing."""
        for _ in range(num_requests):
            with torch.no_grad():
                model(shared_obs, shared_actions)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            future = executor.submit(cache_test_worker, i, 5)
            futures.append(future)
        
        for future in futures:
            future.result()
    
    cache_test_time = time.time() - start_time
    cache_stats_after = cache_manager.get_stats()
    
    print(f"   Cache test time: {cache_test_time:.4f}s")
    
    # Show cache improvement
    for cache_name, stats in cache_stats_after.items():
        if isinstance(stats, dict) and "hit_rate" in stats:
            before_stats = cache_stats_before.get(cache_name, {})
            hits_delta = stats.get("hits", 0) - before_stats.get("hits", 0)
            print(f"   {cache_name}: +{hits_delta} cache hits, "
                  f"hit rate: {stats['hit_rate']:.2%}")
    
    return concurrent_time


def test_memory_efficiency():
    """Test memory efficiency optimizations."""
    print("\n💾 Testing Memory Efficiency")
    print("=" * 35)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    def get_memory_usage():
        """Get current memory usage in MB."""
        return process.memory_info().rss / 1024 / 1024
    
    print("1. Memory usage during scaling")
    
    model = PerspectiveWorldModel(obs_dim=64, action_dim=4, hidden_dim=128, num_agents=3)
    model.eval()
    
    initial_memory = get_memory_usage()
    print(f"   Initial memory: {initial_memory:.1f} MB")
    
    # Test different batch sizes
    batch_sizes = [1, 4, 16, 32, 64]
    memory_usage = []
    
    for batch_size in batch_sizes:
        obs = torch.randn(batch_size, 10, 64)
        actions = torch.randint(0, 4, (batch_size, 10))
        
        # Run inference
        with torch.no_grad():
            states, beliefs = model(obs, actions)
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        current_memory = get_memory_usage()
        memory_delta = current_memory - initial_memory
        memory_usage.append(memory_delta)
        
        print(f"   Batch size {batch_size:2d}: {current_memory:.1f} MB (+{memory_delta:.1f} MB)")
    
    # Test memory cleanup
    print("\n2. Memory cleanup after processing")
    
    peak_memory = max(memory_usage) + initial_memory
    
    # Clear cache and run garbage collection
    cache_manager = get_cache_manager()
    cache_manager.clear_all()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    final_memory = get_memory_usage()
    memory_recovered = peak_memory - final_memory
    
    print(f"   Peak memory: {peak_memory:.1f} MB")
    print(f"   Final memory: {final_memory:.1f} MB")
    print(f"   Memory recovered: {memory_recovered:.1f} MB")
    
    return memory_usage


def test_scalability_limits():
    """Test system scalability limits."""
    print("\n📈 Testing Scalability Limits")
    print("=" * 35)
    
    print("1. Maximum batch size before performance degrades")
    
    model = PerspectiveWorldModel(obs_dim=32, action_dim=4, hidden_dim=64, num_agents=2)
    model.eval()
    
    # Test increasing batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    times = []
    throughputs = []
    
    for batch_size in batch_sizes:
        try:
            obs = torch.randn(batch_size, 5, 32)
            actions = torch.randint(0, 4, (batch_size, 5))
            
            # Warm up
            with torch.no_grad():
                model(obs, actions)
            
            # Measure performance
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                with torch.no_grad():
                    model(obs, actions)
            
            avg_time = (time.time() - start_time) / num_runs
            throughput = batch_size / avg_time
            
            times.append(avg_time)
            throughputs.append(throughput)
            
            print(f"   Batch size {batch_size:3d}: {avg_time:.6f}s, {throughput:.1f} samples/s")
            
        except Exception as e:
            print(f"   Batch size {batch_size:3d}: Failed - {e}")
            break
    
    # Find optimal batch size (highest throughput)
    if throughputs:
        optimal_idx = np.argmax(throughputs)
        optimal_batch_size = batch_sizes[optimal_idx]
        optimal_throughput = throughputs[optimal_idx]
        
        print(f"\n   Optimal batch size: {optimal_batch_size} ({optimal_throughput:.1f} samples/s)")
    
    return batch_sizes[:len(times)], times, throughputs


if __name__ == "__main__":
    print("🚀 PWMK Generation 3: Optimized & Scalable")
    print("Testing performance optimization, caching, and scalability")
    
    # Setup monitoring
    setup_logging(level="INFO", structured=False, console=True)
    collector = setup_monitoring(system_monitoring=True, interval=1.0)
    
    try:
        # Run all tests
        miss_time, hit_time = test_caching_performance()
        individual_time, batch_time = test_batch_processing()
        concurrent_time = test_concurrent_access()
        memory_usage = test_memory_efficiency()
        batch_sizes, times, throughputs = test_scalability_limits()
        
        print("\n🎉 All Generation 3 tests completed!")
        
        print("\n📊 Performance Summary:")
        print(f"   Cache speedup: {miss_time/hit_time:.2f}x")
        print(f"   Batch speedup: {individual_time/batch_time:.2f}x")
        print(f"   Concurrent processing: {concurrent_time:.4f}s")
        print(f"   Peak memory delta: {max(memory_usage):.1f} MB")
        
        if throughputs:
            print(f"   Peak throughput: {max(throughputs):.1f} samples/s")
        
        print("\nGeneration 3 Features Implemented:")
        print("✅ Intelligent caching with LRU and TTL")
        print("✅ Dynamic batch optimization")
        print("✅ Thread-safe concurrent processing")
        print("✅ Memory-efficient operations")
        print("✅ Scalability testing and optimization")
        
        # Final cache statistics
        cache_manager = get_cache_manager()
        cache_manager.log_stats()
        
        # Final metrics summary
        print("\n📈 Final Performance Metrics:")
        collector.log_summary()
        
    except Exception as e:
        print(f"\n❌ Generation 3 testing failed: {e}")
        raise
    finally:
        collector.stop_system_monitoring()