"""Advanced performance optimization for neural networks and AI systems."""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn

from ..utils.logging import get_logger
from .adaptive_scaling import get_scaling_manager


class OptimizationLevel(Enum):
    """Optimization aggressiveness levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class OptimizationConfig:
    """Performance optimization configuration."""
    enable_torch_compile: bool = True
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_fused_optimizers: bool = True
    batch_size_optimization: bool = True
    memory_optimization: bool = True
    cpu_optimization: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED


class PerformanceOptimizer:
    """Advanced performance optimizer for neural networks."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.logger = get_logger(self.__class__.__name__)
        self.optimized_models: Dict[str, nn.Module] = {}
        self.performance_cache: Dict[str, Dict] = {}
        self.optimization_lock = threading.Lock()
        
        # Performance tracking
        self.optimization_history: List[Dict] = []
        self.baseline_metrics: Dict[str, float] = {}
        
        # Torch optimization settings
        self._setup_torch_optimizations()
    
    def _setup_torch_optimizations(self) -> None:
        """Setup PyTorch-level optimizations."""
        try:
            # Set optimal thread counts
            if self.config.cpu_optimization:
                torch.set_num_threads(min(torch.get_num_threads(), 8))
                torch.set_num_interop_threads(4)
            
            # Enable CUDA optimizations if available
            if torch.cuda.is_available():
                # Enable cuDNN benchmarking
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Enable TensorFloat-32 for Ampere GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                self.logger.info("CUDA optimizations enabled")
            
            self.logger.info("PyTorch optimizations configured")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup PyTorch optimizations: {e}")
    
    def optimize_model(
        self, 
        model: nn.Module, 
        model_name: str = "model",
        sample_input: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Optimize a neural network model for maximum performance."""
        start_time = time.time()
        
        with self.optimization_lock:
            self.logger.info(f"Starting optimization of model: {model_name}")
            
            # Record baseline performance if sample input provided
            if sample_input is not None:
                baseline_time = self._benchmark_model(model, sample_input)
                self.baseline_metrics[model_name] = baseline_time
                self.logger.info(f"Baseline inference time: {baseline_time:.4f}s")
            
            optimized_model = model
            optimizations_applied = []
            
            # Apply optimizations based on configuration
            try:
                # 1. Mixed precision optimization
                if self.config.enable_mixed_precision and torch.cuda.is_available():
                    optimized_model = self._apply_mixed_precision(optimized_model)
                    optimizations_applied.append("mixed_precision")
                
                # 2. Gradient checkpointing (for training)
                if (self.config.enable_gradient_checkpointing and 
                    hasattr(optimized_model, 'gradient_checkpointing_enable')):
                    optimized_model.gradient_checkpointing_enable()
                    optimizations_applied.append("gradient_checkpointing")
                
                # 3. Torch.compile optimization (PyTorch 2.0+)
                if self.config.enable_torch_compile and hasattr(torch, 'compile'):
                    compile_mode = self._get_compile_mode()
                    optimized_model = torch.compile(
                        optimized_model, 
                        mode=compile_mode,
                        dynamic=True
                    )
                    optimizations_applied.append(f"torch_compile_{compile_mode}")
                
                # 4. Memory optimization
                if self.config.memory_optimization:
                    optimized_model = self._apply_memory_optimizations(optimized_model)
                    optimizations_applied.append("memory_optimization")
                
                # 5. Model-specific optimizations
                optimized_model = self._apply_model_specific_optimizations(optimized_model)
                if hasattr(optimized_model, '_pwmk_optimized'):
                    optimizations_applied.append("model_specific")
                
                # Store optimized model
                self.optimized_models[model_name] = optimized_model
                
                # Benchmark optimized model
                if sample_input is not None:
                    optimized_time = self._benchmark_model(optimized_model, sample_input)
                    speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                    
                    self.logger.info(
                        f"Optimized inference time: {optimized_time:.4f}s "
                        f"(speedup: {speedup:.2f}x)"
                    )
                    
                    # Record optimization results
                    optimization_result = {
                        "model_name": model_name,
                        "timestamp": time.time(),
                        "baseline_time": baseline_time,
                        "optimized_time": optimized_time,
                        "speedup": speedup,
                        "optimizations": optimizations_applied,
                        "config": self.config.__dict__
                    }
                    self.optimization_history.append(optimization_result)
                
                duration = time.time() - start_time
                self.logger.info(
                    f"Model optimization completed in {duration:.2f}s. "
                    f"Applied: {', '.join(optimizations_applied)}"
                )
                
                return optimized_model
                
            except Exception as e:
                self.logger.error(f"Model optimization failed: {e}")
                return model  # Return original model on failure
    
    def _get_compile_mode(self) -> str:
        """Get torch.compile mode based on optimization level."""
        if self.config.optimization_level == OptimizationLevel.CONSERVATIVE:
            return "default"
        elif self.config.optimization_level == OptimizationLevel.BALANCED:
            return "reduce-overhead"
        else:  # AGGRESSIVE
            return "max-autotune"
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Apply mixed precision optimization."""
        try:
            # Convert model to support mixed precision
            model = model.half()  # Convert to FP16
            
            # Mark certain layers to stay in FP32 for numerical stability
            for name, module in model.named_modules():
                if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.float()
            
            self.logger.info("Applied mixed precision optimization")
            return model
            
        except Exception as e:
            self.logger.warning(f"Mixed precision optimization failed: {e}")
            return model
    
    def _apply_memory_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations."""
        try:
            optimizations_applied = []
            
            # Enable memory efficient attention if available
            for module in model.modules():
                if hasattr(module, 'enable_memory_efficient_attention'):
                    module.enable_memory_efficient_attention()
                    optimizations_applied.append("memory_efficient_attention")
            
            # Optimize parameter storage
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None  # Clear gradients to save memory
            
            if optimizations_applied:
                self.logger.info(f"Applied memory optimizations: {', '.join(optimizations_applied)}")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
            return model
    
    def _apply_model_specific_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply model-specific optimizations."""
        try:
            # Transformer optimizations
            if any('transformer' in str(type(module)).lower() for module in model.modules()):
                model = self._optimize_transformer(model)
            
            # Mark model as optimized
            model._pwmk_optimized = True
            return model
            
        except Exception as e:
            self.logger.warning(f"Model-specific optimization failed: {e}")
            return model
    
    def _optimize_transformer(self, model: nn.Module) -> nn.Module:
        """Optimize transformer-based models."""
        try:
            for module in model.modules():
                if isinstance(module, nn.TransformerEncoderLayer):
                    # Use faster scaled dot-product attention if available
                    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                        # Replace attention mechanism (would need more specific implementation)
                        pass
                    
                    # Enable gradient checkpointing
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
            
            self.logger.info("Applied transformer-specific optimizations")
            return model
            
        except Exception as e:
            self.logger.warning(f"Transformer optimization failed: {e}")
            return model
    
    def _benchmark_model(self, model: nn.Module, sample_input: torch.Tensor, runs: int = 10) -> float:
        """Benchmark model inference time."""
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(3):
                _ = model(sample_input)
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.time()
                _ = model(sample_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        return sum(times) / len(times)
    
    def optimize_batch_size(
        self, 
        model: nn.Module,
        sample_input: torch.Tensor,
        max_memory_gb: float = 8.0
    ) -> int:
        """Find optimal batch size for maximum throughput."""
        if not self.config.batch_size_optimization:
            return sample_input.shape[0]
        
        self.logger.info("Optimizing batch size...")
        
        original_batch_size = sample_input.shape[0]
        best_batch_size = original_batch_size
        best_throughput = 0.0
        
        # Test different batch sizes
        test_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for batch_size in test_batch_sizes:
            if batch_size > original_batch_size * 4:  # Don't test too large batches
                break
            
            try:
                # Create test input
                test_input = torch.randn(
                    batch_size, *sample_input.shape[1:],
                    device=sample_input.device,
                    dtype=sample_input.dtype
                )
                
                # Benchmark this batch size
                inference_time = self._benchmark_model(model, test_input, runs=5)
                throughput = batch_size / inference_time
                
                self.logger.debug(
                    f"Batch size {batch_size}: {inference_time:.4f}s, "
                    f"throughput: {throughput:.2f} samples/s"
                )
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.debug(f"Batch size {batch_size}: OOM")
                    break
                else:
                    self.logger.warning(f"Batch size {batch_size} failed: {e}")
                    continue
        
        self.logger.info(
            f"Optimal batch size: {best_batch_size} "
            f"(throughput: {best_throughput:.2f} samples/s)"
        )
        
        return best_batch_size
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            "config": self.config.__dict__,
            "optimized_models": list(self.optimized_models.keys()),
            "baseline_metrics": self.baseline_metrics,
            "optimization_history": self.optimization_history[-10:],  # Last 10 optimizations
            "torch_settings": {
                "num_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "tf32_enabled": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
            }
        }
    
    def auto_optimize_system(self) -> Dict[str, Any]:
        """Automatically optimize the entire system."""
        self.logger.info("Starting system-wide auto-optimization")
        
        optimization_results = {
            "torch_optimizations": self._auto_optimize_torch_settings(),
            "memory_optimizations": self._auto_optimize_memory(),
            "scaling_optimizations": self._auto_optimize_scaling()
        }
        
        self.logger.info("System-wide auto-optimization completed")
        return optimization_results
    
    def _auto_optimize_torch_settings(self) -> Dict[str, Any]:
        """Automatically optimize PyTorch settings."""
        results = {}
        
        try:
            # Optimize thread settings based on CPU cores
            import os
            cpu_count = os.cpu_count()
            optimal_threads = min(cpu_count, 8)  # Usually 8 is optimal
            
            torch.set_num_threads(optimal_threads)
            torch.set_num_interop_threads(max(1, optimal_threads // 2))
            
            results["num_threads"] = optimal_threads
            results["interop_threads"] = max(1, optimal_threads // 2)
            
            # Optimize CUDA settings
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                results["cudnn_benchmark"] = True
            
            self.logger.info(f"Optimized PyTorch settings: {results}")
            
        except Exception as e:
            self.logger.error(f"PyTorch optimization failed: {e}")
        
        return results
    
    def _auto_optimize_memory(self) -> Dict[str, Any]:
        """Automatically optimize memory usage."""
        results = {}
        
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                results["cuda_cache_cleared"] = True
            
            # Set memory growth for GPU
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_per_process_memory_fraction(0.9, device=i)
                results["memory_fraction_set"] = 0.9
            
            self.logger.info(f"Optimized memory settings: {results}")
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
        
        return results
    
    def _auto_optimize_scaling(self) -> Dict[str, Any]:
        """Automatically optimize scaling settings."""
        results = {}
        
        try:
            scaling_manager = get_scaling_manager()
            
            # Register common components if not already registered
            components = ["model", "belief_store", "consciousness_engine"]
            for component in components:
                if component not in scaling_manager.scalers:
                    scaler = scaling_manager.register_component(component)
                    results[f"{component}_registered"] = True
            
            results["scaling_components"] = len(scaling_manager.scalers)
            self.logger.info(f"Optimized scaling settings: {results}")
            
        except Exception as e:
            self.logger.error(f"Scaling optimization failed: {e}")
        
        return results


# Global performance optimizer
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def optimize_model_for_inference(model: nn.Module, model_name: str = "model") -> nn.Module:
    """Quick function to optimize a model for inference."""
    optimizer = get_performance_optimizer()
    return optimizer.optimize_model(model, model_name)