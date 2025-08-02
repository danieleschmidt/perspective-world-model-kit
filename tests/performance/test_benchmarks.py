"""Performance benchmarks for PWMK components."""
import pytest
import time
import torch
import psutil
import gc
from typing import Dict, List, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    duration: float
    memory_peak: float
    memory_delta: float
    iterations: int
    throughput: float
    metadata: Dict[str, Any]


@contextmanager
def benchmark_context():
    """Context manager for benchmarking with memory tracking."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Store results in pytest metadata
        duration = end_time - start_time
        memory_delta = memory_after - memory_before
        
        # Make results available to test functions
        if hasattr(pytest, '_benchmark_results'):
            pytest._benchmark_results.append({
                'duration': duration,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_delta
            })


@pytest.fixture
def benchmark_runner():
    """Fixture for running performance benchmarks."""
    results = []
    
    def run_benchmark(
        name: str,
        func: Callable,
        iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """Run a performance benchmark."""
        # Warmup
        for _ in range(warmup_iterations):
            func(**kwargs)
        
        # Benchmark
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        memory_peak = memory_before
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            func(**kwargs)
            
            # Track peak memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_peak = max(memory_peak, current_memory)
        
        end_time = time.perf_counter()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        throughput = iterations / duration
        memory_delta = memory_after - memory_before
        
        result = BenchmarkResult(
            name=name,
            duration=duration,
            memory_peak=memory_peak,
            memory_delta=memory_delta,
            iterations=iterations,
            throughput=throughput,
            metadata=kwargs
        )
        
        results.append(result)
        return result
    
    # Expose results for inspection
    run_benchmark.results = results
    return run_benchmark


@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for model components."""
    
    def test_world_model_forward_pass_performance(self, benchmark_runner, simple_test_model):
        """Benchmark world model forward pass."""
        model = simple_test_model
        batch_size = 32
        input_data = torch.randn(batch_size, 64)
        
        def forward_pass():
            with torch.no_grad():
                return model(input_data)
        
        result = benchmark_runner(
            "world_model_forward",
            forward_pass,
            iterations=1000,
            batch_size=batch_size
        )
        
        # Performance assertions
        assert result.throughput > 100, f"Throughput too low: {result.throughput} it/s"
        assert result.memory_delta < 100, f"Memory usage too high: {result.memory_delta} MB"
    
    def test_belief_reasoning_performance(self, benchmark_runner, mock_belief_store):
        """Benchmark belief reasoning operations."""
        store = mock_belief_store
        
        def belief_operations():
            store.add_belief("agent_0", "has(agent_1, key)")
            store.query("has(X, key)")
            store.get_beliefs("agent_0")
        
        result = benchmark_runner(
            "belief_reasoning",
            belief_operations,
            iterations=1000
        )
        
        assert result.throughput > 500, f"Belief reasoning too slow: {result.throughput} ops/s"
    
    @pytest.mark.slow
    def test_large_batch_processing(self, benchmark_runner, simple_test_model):
        """Benchmark processing of large batches."""
        model = simple_test_model
        batch_sizes = [64, 128, 256, 512]
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, 64)
            
            def process_batch():
                with torch.no_grad():
                    return model(input_data)
            
            result = benchmark_runner(
                f"large_batch_{batch_size}",
                process_batch,
                iterations=100,
                batch_size=batch_size
            )
            
            # Memory usage should scale reasonably with batch size
            memory_per_sample = result.memory_delta / batch_size
            assert memory_per_sample < 1.0, f"Memory per sample too high: {memory_per_sample} MB"
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_performance(self, benchmark_runner, device_aware_model):
        """Benchmark GPU performance vs CPU."""
        cpu_model = device_aware_model(torch.device("cpu"))
        gpu_model = device_aware_model(torch.device("cuda"))
        
        batch_size = 128
        input_data_cpu = torch.randn(batch_size, 64)
        input_data_gpu = input_data_cpu.cuda()
        
        def cpu_forward():
            with torch.no_grad():
                return cpu_model(input_data_cpu)
        
        def gpu_forward():
            with torch.no_grad():
                return gpu_model(input_data_gpu)
        
        cpu_result = benchmark_runner("cpu_forward", cpu_forward, iterations=100)
        gpu_result = benchmark_runner("gpu_forward", gpu_forward, iterations=100)
        
        # GPU should be faster for larger batches
        speedup = cpu_result.duration / gpu_result.duration
        print(f"GPU speedup: {speedup:.2f}x")
        
        # Don't assert speedup as it depends on hardware
        assert gpu_result.throughput > 0, "GPU processing failed"


@pytest.mark.performance
class TestEnvironmentPerformance:
    """Performance tests for environment components."""
    
    def test_environment_step_performance(self, benchmark_runner, mock_multi_agent_env):
        """Benchmark environment step operations."""
        env = mock_multi_agent_env
        actions = [0, 1, 2]  # Actions for 3 agents
        
        def env_step():
            return env.step(actions)
        
        result = benchmark_runner(
            "env_step",
            env_step,
            iterations=10000
        )
        
        assert result.throughput > 1000, f"Environment steps too slow: {result.throughput} steps/s"
    
    def test_trajectory_collection_performance(self, benchmark_runner, mock_multi_agent_env):
        """Benchmark trajectory collection."""
        env = mock_multi_agent_env
        
        def collect_trajectory():
            env.reset()
            for _ in range(100):  # 100 step episode
                actions = [0, 1, 2]
                env.step(actions)
        
        result = benchmark_runner(
            "trajectory_collection",
            collect_trajectory,
            iterations=10
        )
        
        steps_per_second = (result.iterations * 100) / result.duration
        assert steps_per_second > 500, f"Trajectory collection too slow: {steps_per_second} steps/s"


@pytest.mark.performance
class TestMemoryUsage:
    """Memory usage tests."""
    
    def test_model_memory_scaling(self, model_configs):
        """Test memory usage scaling with model size."""
        for config_name, config in model_configs.items():
            model = torch.nn.Linear(config["obs_dim"], config["hidden_dim"])
            
            # Measure memory before and after model creation
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Create multiple instances
            models = [torch.nn.Linear(config["obs_dim"], config["hidden_dim"]) for _ in range(10)]
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_per_model = (memory_after - memory_before) / 10
            
            # Memory usage should be reasonable
            assert memory_per_model < 50, f"Model {config_name} uses too much memory: {memory_per_model} MB"
            
            # Cleanup
            del models
            del model
            gc.collect()
    
    def test_memory_leak_detection(self, simple_test_model):
        """Test for memory leaks in repeated operations."""
        model = simple_test_model
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Run many iterations
        for _ in range(1000):
            input_data = torch.randn(32, 64)
            with torch.no_grad():
                output = model(input_data)
            del input_data, output
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal
        assert memory_growth < 50, f"Potential memory leak: {memory_growth} MB growth"


@pytest.fixture(scope="session")
def performance_report():
    """Generate a performance report at the end of the session."""
    yield
    
    # This would run after all performance tests
    if hasattr(pytest, '_benchmark_results'):
        results = pytest._benchmark_results
        
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        
        for result in results:
            print(f"Duration: {result['duration']:.4f}s")
            print(f"Memory Delta: {result['memory_delta']:.2f} MB")
            print("-" * 30)


# Utility functions for performance testing
def measure_time(func: Callable, *args, **kwargs) -> float:
    """Measure execution time of a function."""
    start = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - start


def measure_memory(func: Callable, *args, **kwargs) -> Dict[str, float]:
    """Measure memory usage of a function."""
    process = psutil.Process()
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    memory_before = process.memory_info().rss / 1024 / 1024
    func(*args, **kwargs)
    memory_after = process.memory_info().rss / 1024 / 1024
    
    return {
        "before": memory_before,
        "after": memory_after,
        "delta": memory_after - memory_before
    }