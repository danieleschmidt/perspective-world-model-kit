"""
Advanced Performance Optimization Module for PWMK

Provides comprehensive optimization including caching, parallel processing,
auto-scaling, memory optimization, and performance profiling.
"""

from .caching import ModelCache, BeliefCache, PredictionCache, get_cache_manager
from .batching import BatchProcessor, DynamicBatcher
from .parallel_processing import get_parallel_processor, ParallelBeliefProcessor
from .auto_scaling import get_auto_scaler, AutoScaler
from .advanced_optimization import (
    AdvancedOptimizationSystem,
    IntelligentCache,
    TensorCache,
    ConcurrencyManager,
    ResourceMonitor,
    AutoScaler as AdvancedAutoScaler,
    MemoryOptimizer,
    PerformanceProfiler,
    OptimizationConfig,
    PerformanceMetrics,
    create_advanced_optimization
)
from .quantum_accelerated_inference import get_quantum_inference_engine

__all__ = [
    "ModelCache",
    "BeliefCache", 
    "PredictionCache",
    "get_cache_manager",
    "BatchProcessor",
    "DynamicBatcher",
    "get_parallel_processor", 
    "ParallelBeliefProcessor",
    "get_auto_scaler", 
    "AutoScaler",
    "AdvancedOptimizationSystem",
    "IntelligentCache",
    "TensorCache",
    "ConcurrencyManager",
    "ResourceMonitor",
    "AdvancedAutoScaler",
    "MemoryOptimizer",
    "PerformanceProfiler",
    "OptimizationConfig",
    "PerformanceMetrics",
    "create_advanced_optimization",
    "get_quantum_inference_engine"
]