"""
Performance optimization and scaling for quantum-enhanced planning.

Implements advanced performance optimizations including parallel processing,
intelligent caching, memory management, and auto-scaling capabilities.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import numpy as np
import torch
import asyncio
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import psutil
from collections import deque
import gc
import threading

from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector
from ..optimization.caching import get_cache_manager


@dataclass
class PerformanceConfig:
    """Configuration for quantum performance optimization."""
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    enable_adaptive_batching: bool = True
    enable_memory_optimization: bool = True
    max_workers: Optional[int] = None
    batch_size_range: Tuple[int, int] = (1, 32)
    memory_limit_gb: float = 8.0
    cache_ttl_seconds: float = 300.0
    auto_scaling_enabled: bool = True


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: float
    active_threads: int
    quantum_operations_per_second: float
    cache_hit_rate: float


@dataclass
class OptimizationResult:
    """Result from performance optimization."""
    speedup_factor: float
    memory_savings_mb: float
    cache_efficiency: float
    resource_utilization: ResourceMetrics
    optimization_time: float


class QuantumPerformanceOptimizer(LoggingMixin):
    """
    Advanced performance optimizer for quantum planning algorithms.
    
    Provides intelligent resource management, parallel processing,
    and adaptive optimization strategies for quantum computations.
    """
    
    def __init__(
        self,
        config: Optional[PerformanceConfig] = None
    ):
        super().__init__()
        
        self.config = config or PerformanceConfig()
        
        # Resource management
        self.max_workers = self.config.max_workers or min(cpu_count(), 8)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(cpu_count(), 4))
        
        # GPU availability
        self.gpu_available = torch.cuda.is_available() and self.config.enable_gpu_acceleration
        if self.gpu_available:
            self.device = torch.device("cuda")
            self.gpu_memory_pool = torch.cuda.memory_pool()
        else:
            self.device = torch.device("cpu")
        
        # Adaptive batching
        self.current_batch_size = self.config.batch_size_range[0]
        self.batch_performance_history = deque(maxlen=100)
        
        # Memory management
        self.memory_monitor = threading.Thread(target=self._monitor_memory, daemon=True)
        self.memory_monitor_active = True
        self.memory_monitor.start()
        
        # Performance tracking
        self.operation_stats = {
            "total_operations": 0,
            "parallel_operations": 0,
            "gpu_operations": 0,
            "cache_hits": 0,
            "memory_optimizations": 0
        }
        
        # Auto-scaling metrics
        self.load_history = deque(maxlen=50)
        self.scaling_lock = threading.Lock()
        
        self.logger.info(
            f"Initialized QuantumPerformanceOptimizer: max_workers={self.max_workers}, "
            f"gpu_available={self.gpu_available}, batch_size={self.current_batch_size}"
        )
    
    async def optimize_quantum_operations(
        self,
        operations: List[Callable],
        operation_args: List[Tuple],
        optimization_hints: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize a batch of quantum operations for maximum performance.
        
        Args:
            operations: List of quantum operations to execute
            operation_args: Arguments for each operation
            optimization_hints: Hints for optimization strategy
            
        Returns:
            OptimizationResult with performance metrics
        """
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            self.logger.info(f"Optimizing {len(operations)} quantum operations")
            
            # Determine optimization strategy
            strategy = self._determine_optimization_strategy(operations, optimization_hints)
            
            # Execute optimized operations
            if strategy == "parallel_cpu":
                results = await self._execute_parallel_cpu(operations, operation_args)
            elif strategy == "parallel_gpu":
                results = await self._execute_parallel_gpu(operations, operation_args)
            elif strategy == "batched":
                results = await self._execute_batched(operations, operation_args)
            elif strategy == "adaptive":
                results = await self._execute_adaptive(operations, operation_args)
            else:
                results = await self._execute_sequential(operations, operation_args)
            
            # Calculate optimization metrics
            optimization_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_savings = initial_memory - final_memory
            
            # Get resource metrics
            resource_metrics = self._get_resource_metrics()
            
            # Calculate speedup (heuristic based on parallelization)
            sequential_time_estimate = len(operations) * 0.1  # Rough estimate
            speedup_factor = max(1.0, sequential_time_estimate / optimization_time)
            
            # Cache efficiency
            cache_efficiency = self._calculate_cache_efficiency()
            
            result = OptimizationResult(
                speedup_factor=speedup_factor,
                memory_savings_mb=memory_savings,
                cache_efficiency=cache_efficiency,
                resource_utilization=resource_metrics,
                optimization_time=optimization_time
            )
            
            # Update statistics
            self.operation_stats["total_operations"] += len(operations)
            if "parallel" in strategy:
                self.operation_stats["parallel_operations"] += len(operations)
            if "gpu" in strategy:
                self.operation_stats["gpu_operations"] += len(operations)
            
            # Record metrics
            get_metrics_collector().record_metric("quantum_speedup", speedup_factor)
            get_metrics_collector().record_metric("memory_savings", memory_savings)
            get_metrics_collector().record_quantum_operation("batch_optimization", optimization_time)
            
            self.logger.info(
                f"Quantum optimization complete: strategy={strategy}, "
                f"speedup={speedup_factor:.2f}x, memory_saved={memory_savings:.1f}MB, "
                f"time={optimization_time:.4f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quantum optimization failed: {e}")
            raise
    
    def _determine_optimization_strategy(
        self,
        operations: List[Callable],
        hints: Optional[Dict[str, Any]]
    ) -> str:
        """Determine optimal execution strategy for operations."""
        
        num_operations = len(operations)
        current_load = self._get_current_system_load()
        
        # Check hints
        if hints:
            if hints.get("force_gpu", False) and self.gpu_available:
                return "parallel_gpu"
            if hints.get("force_sequential", False):
                return "sequential"
            if hints.get("prefer_batching", False):
                return "batched"
        
        # Adaptive strategy based on system state and operation count
        if num_operations == 1:
            return "sequential"
        
        if num_operations < 5:
            if current_load < 0.7 and self.gpu_available:
                return "parallel_gpu"
            else:
                return "parallel_cpu"
        
        if num_operations < 20:
            if self.gpu_available and current_load < 0.6:
                return "parallel_gpu"
            else:
                return "batched"
        
        # Large number of operations
        if current_load < 0.5:
            return "adaptive"
        else:
            return "batched"
    
    async def _execute_parallel_cpu(
        self,
        operations: List[Callable],
        operation_args: List[Tuple]
    ) -> List[Any]:
        """Execute operations in parallel on CPU."""
        
        loop = asyncio.get_event_loop()
        
        # Create tasks for thread pool execution
        tasks = []
        for op, args in zip(operations, operation_args):
            task = loop.run_in_executor(self.thread_pool, op, *args)
            tasks.append(task)
        
        # Execute with timeout and error handling
        results = []
        for i, task in enumerate(tasks):
            try:
                result = await asyncio.wait_for(task, timeout=30.0)
                results.append(result)
            except asyncio.TimeoutError:
                self.logger.warning(f"Operation {i} timed out")
                results.append(None)
            except Exception as e:
                self.logger.error(f"Operation {i} failed: {e}")
                results.append(None)
        
        return results
    
    async def _execute_parallel_gpu(
        self,
        operations: List[Callable],
        operation_args: List[Tuple]
    ) -> List[Any]:
        """Execute operations in parallel on GPU."""
        
        if not self.gpu_available:
            return await self._execute_parallel_cpu(operations, operation_args)
        
        try:
            # Batch operations for GPU efficiency
            gpu_batches = self._create_gpu_batches(operations, operation_args)
            
            results = []
            for batch_ops, batch_args in gpu_batches:
                # Move data to GPU
                gpu_args = []
                for args in batch_args:
                    gpu_batch_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            gpu_batch_args.append(arg.to(self.device))
                        else:
                            gpu_batch_args.append(arg)
                    gpu_args.append(tuple(gpu_batch_args))
                
                # Execute batch on GPU
                batch_results = []
                for op, args in zip(batch_ops, gpu_args):
                    try:
                        with torch.cuda.device(self.device):
                            result = op(*args)
                            # Move result back to CPU if it's a tensor
                            if isinstance(result, torch.Tensor):
                                result = result.cpu()
                            batch_results.append(result)
                    except Exception as e:
                        self.logger.error(f"GPU operation failed: {e}")
                        batch_results.append(None)
                
                results.extend(batch_results)
            
            # Clean up GPU memory
            if self.gpu_available:
                torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            self.logger.error(f"GPU execution failed: {e}")
            # Fallback to CPU
            return await self._execute_parallel_cpu(operations, operation_args)
    
    async def _execute_batched(
        self,
        operations: List[Callable],
        operation_args: List[Tuple]
    ) -> List[Any]:
        """Execute operations in optimized batches."""
        
        # Determine optimal batch size
        optimal_batch_size = self._get_optimal_batch_size(len(operations))
        
        results = []
        for i in range(0, len(operations), optimal_batch_size):
            batch_ops = operations[i:i + optimal_batch_size]
            batch_args = operation_args[i:i + optimal_batch_size]
            
            # Execute batch
            batch_start_time = time.time()
            
            if self.gpu_available and len(batch_ops) > 2:
                batch_results = await self._execute_parallel_gpu(batch_ops, batch_args)
            else:
                batch_results = await self._execute_parallel_cpu(batch_ops, batch_args)
            
            batch_time = time.time() - batch_start_time
            
            # Update batch performance tracking
            self.batch_performance_history.append({
                "batch_size": len(batch_ops),
                "execution_time": batch_time,
                "throughput": len(batch_ops) / batch_time if batch_time > 0 else 0
            })
            
            results.extend(batch_results)
            
            # Adaptive delay to prevent resource exhaustion
            if i + optimal_batch_size < len(operations):
                await asyncio.sleep(0.01)  # Small delay between batches
        
        return results
    
    async def _execute_adaptive(
        self,
        operations: List[Callable],
        operation_args: List[Tuple]
    ) -> List[Any]:
        """Execute operations with adaptive strategy based on real-time performance."""
        
        results = []
        remaining_ops = list(zip(operations, operation_args))
        
        while remaining_ops:
            # Determine current optimal strategy
            current_load = self._get_current_system_load()
            memory_usage = psutil.Process().memory_percent()
            
            if memory_usage > 80:
                # High memory usage: process smaller batches
                batch_size = min(2, len(remaining_ops))
                strategy = "sequential"
            elif current_load < 0.3:
                # Low load: maximize parallelization
                batch_size = min(self.max_workers * 2, len(remaining_ops))
                strategy = "parallel_gpu" if self.gpu_available else "parallel_cpu"
            else:
                # Medium load: balanced approach
                batch_size = min(self.max_workers, len(remaining_ops))
                strategy = "batched"
            
            # Execute current batch
            current_batch = remaining_ops[:batch_size]
            batch_ops, batch_args = zip(*current_batch)
            
            if strategy == "sequential":
                batch_results = []
                for op, args in current_batch:
                    try:
                        result = op(*args)
                        batch_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Sequential operation failed: {e}")
                        batch_results.append(None)
            elif strategy == "parallel_gpu":
                batch_results = await self._execute_parallel_gpu(list(batch_ops), list(batch_args))
            elif strategy == "parallel_cpu":
                batch_results = await self._execute_parallel_cpu(list(batch_ops), list(batch_args))
            else:
                batch_results = await self._execute_batched(list(batch_ops), list(batch_args))
            
            results.extend(batch_results)
            remaining_ops = remaining_ops[batch_size:]
            
            # Brief pause for system recovery if needed
            if memory_usage > 70:
                await asyncio.sleep(0.05)
                gc.collect()  # Force garbage collection
        
        return results
    
    async def _execute_sequential(
        self,
        operations: List[Callable],
        operation_args: List[Tuple]
    ) -> List[Any]:
        """Execute operations sequentially."""
        
        results = []
        for op, args in zip(operations, operation_args):
            try:
                result = op(*args)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Sequential operation failed: {e}")
                results.append(None)
        
        return results
    
    def _create_gpu_batches(
        self,
        operations: List[Callable],
        operation_args: List[Tuple]
    ) -> List[Tuple[List[Callable], List[Tuple]]]:
        """Create optimized batches for GPU execution."""
        
        if not self.gpu_available:
            return [(operations, operation_args)]
        
        # Get GPU memory info
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_free = gpu_memory_total - torch.cuda.memory_allocated()
        
        # Estimate memory per operation (heuristic)
        estimated_memory_per_op = 50 * 1024 * 1024  # 50MB per operation
        max_ops_per_batch = max(1, int(gpu_memory_free * 0.8 / estimated_memory_per_op))
        
        # Create batches
        batches = []
        for i in range(0, len(operations), max_ops_per_batch):
            batch_ops = operations[i:i + max_ops_per_batch]
            batch_args = operation_args[i:i + max_ops_per_batch]
            batches.append((batch_ops, batch_args))
        
        return batches
    
    def _get_optimal_batch_size(self, total_operations: int) -> int:
        """Determine optimal batch size based on performance history."""
        
        if not self.batch_performance_history:
            return min(self.current_batch_size, total_operations)
        
        # Analyze recent performance
        recent_performance = list(self.batch_performance_history)[-20:]  # Last 20 batches
        
        if len(recent_performance) < 5:
            return min(self.current_batch_size, total_operations)
        
        # Find batch size with best throughput
        batch_size_performance = {}
        for perf in recent_performance:
            batch_size = perf["batch_size"]
            throughput = perf["throughput"]
            
            if batch_size not in batch_size_performance:
                batch_size_performance[batch_size] = []
            batch_size_performance[batch_size].append(throughput)
        
        # Calculate average throughput for each batch size
        avg_throughput = {}
        for batch_size, throughputs in batch_size_performance.items():
            avg_throughput[batch_size] = np.mean(throughputs)
        
        # Select batch size with highest average throughput
        if avg_throughput:
            optimal_batch_size = max(avg_throughput.keys(), key=lambda k: avg_throughput[k])
            
            # Ensure within valid range
            min_batch, max_batch = self.config.batch_size_range
            optimal_batch_size = max(min_batch, min(max_batch, optimal_batch_size))
            
            self.current_batch_size = optimal_batch_size
        
        return min(self.current_batch_size, total_operations)
    
    def _get_current_system_load(self) -> float:
        """Get current system load as a fraction (0.0 to 1.0)."""
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Combined load metric
        system_load = (cpu_percent + memory_percent) / 200.0  # Average of CPU and memory
        
        # Update load history for auto-scaling
        self.load_history.append(system_load)
        
        return min(1.0, system_load)
    
    def _get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource utilization metrics."""
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # GPU memory
        gpu_memory_percent = 0.0
        if self.gpu_available:
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        
        # Thread count
        active_threads = threading.active_count()
        
        # Operations per second (estimate from recent activity)
        ops_per_second = self._calculate_ops_per_second()
        
        # Cache hit rate
        cache_hit_rate = self._calculate_cache_hit_rate()
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_memory_percent=gpu_memory_percent,
            active_threads=active_threads,
            quantum_operations_per_second=ops_per_second,
            cache_hit_rate=cache_hit_rate
        )
    
    def _calculate_ops_per_second(self) -> float:
        """Calculate operations per second based on recent activity."""
        
        if not hasattr(self, '_ops_timestamps'):
            self._ops_timestamps = deque(maxlen=100)
        
        current_time = time.time()
        self._ops_timestamps.append(current_time)
        
        if len(self._ops_timestamps) < 2:
            return 0.0
        
        # Calculate ops/sec over last 60 seconds
        recent_ops = [t for t in self._ops_timestamps if current_time - t <= 60.0]
        
        if len(recent_ops) < 2:
            return 0.0
        
        time_span = recent_ops[-1] - recent_ops[0]
        ops_per_second = len(recent_ops) / time_span if time_span > 0 else 0.0
        
        return ops_per_second
    
    def _calculate_cache_efficiency(self) -> float:
        """Calculate overall cache efficiency."""
        
        cache_manager = get_cache_manager()
        
        if cache_manager and hasattr(cache_manager, 'get_cache_stats'):
            stats = cache_manager.get_cache_stats()
            if 'hit_rate' in stats:
                return stats['hit_rate']
        
        # Fallback calculation
        total_ops = self.operation_stats["total_operations"]
        cache_hits = self.operation_stats["cache_hits"]
        
        if total_ops > 0:
            return cache_hits / total_ops
        
        return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate current cache hit rate."""
        return self._calculate_cache_efficiency()
    
    def _monitor_memory(self) -> None:
        """Background memory monitoring and optimization."""
        
        while self.memory_monitor_active:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # Check if memory usage is too high
                if memory_percent > 80:
                    self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
                    
                    # Trigger memory optimization
                    self._optimize_memory_usage()
                
                # Check against configured limit
                memory_gb = memory_info.rss / (1024 ** 3)
                if memory_gb > self.config.memory_limit_gb:
                    self.logger.warning(
                        f"Memory limit exceeded: {memory_gb:.2f}GB > {self.config.memory_limit_gb}GB"
                    )
                    self._emergency_memory_cleanup()
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(10.0)  # Longer delay on error
    
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage when high usage detected."""
        
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if self.gpu_available:
                torch.cuda.empty_cache()
            
            # Clear old batch performance history
            if len(self.batch_performance_history) > 50:
                # Keep only most recent entries
                recent_entries = list(self.batch_performance_history)[-30:]
                self.batch_performance_history.clear()
                self.batch_performance_history.extend(recent_entries)
            
            self.operation_stats["memory_optimizations"] += 1
            
            self.logger.debug("Memory optimization completed")
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
    
    def _emergency_memory_cleanup(self) -> None:
        """Emergency memory cleanup when limit is exceeded."""
        
        try:
            # Aggressive cleanup
            self._optimize_memory_usage()
            
            # Clear all caches
            cache_manager = get_cache_manager()
            if cache_manager:
                cache_manager.clear_all_caches()
            
            # Reduce batch size temporarily
            min_batch, _ = self.config.batch_size_range
            self.current_batch_size = min_batch
            
            self.logger.warning("Emergency memory cleanup performed")
            
        except Exception as e:
            self.logger.error(f"Emergency memory cleanup failed: {e}")
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        stats = self.operation_stats.copy()
        
        # Resource metrics
        resource_metrics = self._get_resource_metrics()
        stats["resource_utilization"] = {
            "cpu_percent": resource_metrics.cpu_percent,
            "memory_percent": resource_metrics.memory_percent,
            "gpu_memory_percent": resource_metrics.gpu_memory_percent,
            "active_threads": resource_metrics.active_threads,
            "ops_per_second": resource_metrics.quantum_operations_per_second
        }
        
        # Performance metrics
        if self.batch_performance_history:
            recent_throughputs = [p["throughput"] for p in self.batch_performance_history]
            stats["batch_performance"] = {
                "current_batch_size": self.current_batch_size,
                "avg_throughput": float(np.mean(recent_throughputs)),
                "max_throughput": float(np.max(recent_throughputs)),
                "throughput_trend": "increasing" if len(recent_throughputs) > 5 and recent_throughputs[-1] > recent_throughputs[0] else "stable"
            }
        
        # Load history
        if self.load_history:
            stats["system_load"] = {
                "current_load": float(self.load_history[-1]) if self.load_history else 0.0,
                "avg_load": float(np.mean(self.load_history)),
                "load_trend": "increasing" if len(self.load_history) > 10 and self.load_history[-1] > self.load_history[0] else "stable"
            }
        
        # Configuration
        stats["configuration"] = {
            "max_workers": self.max_workers,
            "gpu_available": self.gpu_available,
            "memory_limit_gb": self.config.memory_limit_gb,
            "auto_scaling_enabled": self.config.auto_scaling_enabled
        }
        
        return stats
    
    async def cleanup(self) -> None:
        """Clean up resources and stop background processes."""
        
        try:
            self.memory_monitor_active = False
            
            # Shutdown thread pools
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            if self.process_pool:
                self.process_pool.shutdown(wait=True)
            
            # GPU cleanup
            if self.gpu_available:
                torch.cuda.empty_cache()
            
            self.logger.info("QuantumPerformanceOptimizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.memory_monitor_active = False
        except:
            pass