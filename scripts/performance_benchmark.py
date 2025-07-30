#!/usr/bin/env python3
"""
Performance Benchmarking and Regression Testing

Automated performance testing suite for PWMK components with
trend analysis and regression detection.
"""

import json
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import matplotlib.pyplot as plt


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    duration_ms: float
    memory_peak_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float]
    throughput: Optional[float]
    accuracy: Optional[float]
    timestamp: str
    environment: Dict[str, Any]


class PerformanceBenchmark:
    """Performance benchmarking suite for PWMK components."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks."""
        print("PWMK Performance Benchmarking")
        print("==============================")
        
        benchmarks = [
            ("world_model_inference", self._benchmark_world_model),
            ("belief_reasoning", self._benchmark_belief_reasoning),
            ("epistemic_planning", self._benchmark_epistemic_planning),
            ("memory_usage", self._benchmark_memory_usage),
            ("batch_processing", self._benchmark_batch_processing),
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\nRunning benchmark: {name}")
            try:
                result = self._run_benchmark(name, benchmark_func)
                self.results.append(result)
                print(f"✓ {name}: {result.duration_ms:.2f}ms")
            except Exception as e:
                print(f"✗ {name}: Failed - {e}")
                
        return self.results
        
    def _run_benchmark(self, name: str, benchmark_func) -> BenchmarkResult:
        """Run a single benchmark with monitoring."""
        # Setup monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU monitoring if available
        gpu_memory_before = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024
            
        # Run benchmark
        start_time = time.perf_counter()
        start_cpu = process.cpu_percent()
        
        benchmark_data = benchmark_func()
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Collect metrics
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_peak = max(initial_memory, final_memory)
        cpu_percent = process.cpu_percent()
        
        gpu_memory_peak = None
        if torch.cuda.is_available():
            gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
            
        return BenchmarkResult(
            test_name=name,
            duration_ms=duration_ms,
            memory_peak_mb=memory_peak,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory_peak,
            throughput=benchmark_data.get("throughput"),
            accuracy=benchmark_data.get("accuracy"),
            timestamp=datetime.now().isoformat(),
            environment=self._get_environment_info()
        )
        
    def _benchmark_world_model(self) -> Dict[str, Any]:
        """Benchmark world model inference performance."""
        from pwmk import PerspectiveWorldModel
        
        # Create model
        model = PerspectiveWorldModel(
            obs_dim=64,
            action_dim=4,
            hidden_dim=256,
            num_agents=3
        )
        model.eval()
        
        # Generate test data
        batch_size = 32
        seq_len = 10
        obs = torch.randn(batch_size, seq_len, 64)
        actions = torch.randint(0, 4, (batch_size, seq_len, 4))
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(obs, actions)
                
        # Benchmark inference
        num_iterations = 100
        total_samples = num_iterations * batch_size
        
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            with torch.no_grad():
                states, beliefs = model(obs, actions)
        end_time = time.perf_counter()
        
        throughput = total_samples / (end_time - start_time)
        
        return {
            "throughput": throughput,
            "batch_size": batch_size,
            "sequence_length": seq_len
        }
        
    def _benchmark_belief_reasoning(self) -> Dict[str, Any]:
        """Benchmark belief store query performance."""
        from pwmk import BeliefStore
        
        store = BeliefStore()
        
        # Populate with test data
        num_agents = 10
        num_facts_per_agent = 100
        
        for agent_id in range(num_agents):
            for fact_id in range(num_facts_per_agent):
                belief = f"has(agent_{agent_id}, item_{fact_id})"
                store.add_belief(f"agent_{agent_id}", belief)
                
        # Benchmark queries
        num_queries = 1000
        query = "has(X, Y)"
        
        start_time = time.perf_counter()
        for _ in range(num_queries):
            results = store.query(query)
        end_time = time.perf_counter()
        
        query_rate = num_queries / (end_time - start_time)
        
        return {
            "throughput": query_rate,
            "num_facts": num_agents * num_facts_per_agent,
            "query_complexity": "simple_pattern"
        }
        
    def _benchmark_epistemic_planning(self) -> Dict[str, Any]:
        """Benchmark epistemic planner performance."""
        from pwmk import PerspectiveWorldModel, BeliefStore, EpistemicPlanner
        from pwmk.planning.epistemic import Goal
        
        # Setup planner
        world_model = PerspectiveWorldModel(64, 4, 256, 3)
        belief_store = BeliefStore()
        planner = EpistemicPlanner(world_model, belief_store, search_depth=5)
        
        # Create planning problem
        initial_state = np.random.randn(64)
        goal = Goal(
            achievement="has(agent_0, treasure)",
            epistemic=["believes(agent_1, at(agent_0, room_2))"]
        )
        
        # Benchmark planning
        num_plans = 50
        
        start_time = time.perf_counter()
        for _ in range(num_plans):
            plan = planner.plan(initial_state, goal, timeout=1.0)
        end_time = time.perf_counter()
        
        plans_per_second = num_plans / (end_time - start_time)
        
        return {
            "throughput": plans_per_second,
            "search_depth": 5,
            "planning_horizon": 10
        }
        
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        from pwmk import PerspectiveWorldModel, ToMAgent
        
        # Create multiple agents with models
        agents = []
        for i in range(5):
            model = PerspectiveWorldModel(64, 4, 256, 3)
            agent = ToMAgent(f"agent_{i}", model, tom_depth=2)
            agents.append(agent)
            
        # Simulate agent interactions
        for step in range(100):
            obs = {"step": step, "position": np.random.randn(2)}
            for agent in agents:
                agent.update_beliefs(obs)
                action = agent.act_with_tom(obs)
                
        memory_per_agent = psutil.Process().memory_info().rss / (1024 * 1024 * len(agents))
        
        return {
            "memory_per_agent_mb": memory_per_agent,
            "num_agents": len(agents),
            "simulation_steps": 100
        }
        
    def _benchmark_batch_processing(self) -> Dict[str, Any]:
        """Benchmark batch processing capabilities."""
        from pwmk import PerspectiveWorldModel
        
        model = PerspectiveWorldModel(64, 4, 256, 3)
        model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 4, 16, 32, 64, 128]
        throughputs = []
        
        for batch_size in batch_sizes:
            obs = torch.randn(batch_size, 10, 64)
            actions = torch.randint(0, 4, (batch_size, 10, 4))
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(obs, actions)
                    
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(20):
                with torch.no_grad():
                    _ = model(obs, actions)
            end_time = time.perf_counter()
            
            throughput = (20 * batch_size) / (end_time - start_time)
            throughputs.append(throughput)
            
        optimal_batch = batch_sizes[np.argmax(throughputs)]
        max_throughput = max(throughputs)
        
        return {
            "throughput": max_throughput,
            "optimal_batch_size": optimal_batch,
            "batch_scaling": list(zip(batch_sizes, throughputs))
        }
        
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for benchmark context."""
        env_info = {
            "python_version": psutil.__version__,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "platform": psutil.LINUX if hasattr(psutil, 'LINUX') else "unknown"
        }
        
        if torch.cuda.is_available():
            env_info.update({
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0)
            })
        else:
            env_info["cuda_available"] = False
            
        return env_info
        
    def save_results(self, filename: str = None) -> Path:
        """Save benchmark results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
        output_file = self.output_dir / filename
        
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_benchmarks": len(self.results),
                "pwmk_version": "0.1.0"  # Should be imported
            },
            "results": [asdict(result) for result in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"Results saved to: {output_file}")
        return output_file
        
    def generate_report(self) -> None:
        """Generate performance report with visualizations."""
        if not self.results:
            print("No results to report")
            return
            
        # Generate summary statistics
        total_duration = sum(r.duration_ms for r in self.results)
        avg_memory = np.mean([r.memory_peak_mb for r in self.results])
        
        print(f"\nPerformance Summary")
        print(f"==================")
        print(f"Total benchmarks: {len(self.results)}")
        print(f"Total duration: {total_duration:.2f}ms")
        print(f"Average memory usage: {avg_memory:.2f}MB")
        
        # Create performance chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        test_names = [r.test_name for r in self.results]
        durations = [r.duration_ms for r in self.results]
        memory_usage = [r.memory_peak_mb for r in self.results]
        
        ax1.bar(test_names, durations)
        ax1.set_title("Execution Time by Test")
        ax1.set_ylabel("Duration (ms)")
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(test_names, memory_usage)
        ax2.set_title("Memory Usage by Test")
        ax2.set_ylabel("Memory (MB)")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        chart_path = self.output_dir / "performance_report.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"Performance chart saved to: {chart_path}")


def main():
    """Main benchmarking entry point."""
    benchmark = PerformanceBenchmark()
    
    # Run all benchmarks
    results = benchmark.run_all_benchmarks()
    
    # Save results and generate report
    benchmark.save_results()
    benchmark.generate_report()
    
    print(f"\nBenchmarking complete. {len(results)} tests executed.")


if __name__ == "__main__":
    main()