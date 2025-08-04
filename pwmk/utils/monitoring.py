"""Performance monitoring and metrics collection for PWMK."""

import time
import psutil
import torch
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import json
from pathlib import Path

from .logging import get_logger, LoggingMixin


@dataclass
class MetricValue:
    """Container for a metric value with metadata."""
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class PerformanceMonitor(LoggingMixin):
    """Monitor and track performance metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self._lock:
            metric = MetricValue(
                value=value,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
            self.logger.debug(f"Recorded metric {name}: {value} (name={name}, value={value})")
    
    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += amount
            self.logger.debug(f"Incremented counter {name} by {amount} (counter={name}, increment={amount})")
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(f"{name}_duration", duration, tags)
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self.metrics:
                return {}
            
            values = [m.value for m in self.metrics[name]]
            if not values:
                return {}
            
            return {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else 0.0
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all metrics."""
        stats = {}
        
        # Metric statistics
        for name in self.metrics:
            stats[name] = self.get_metric_stats(name)
        
        # Counter values
        with self._lock:
            stats["counters"] = dict(self.counters)
        
        return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.timers.clear()
        self.logger.info("Reset all metrics")


class SystemMonitor(LoggingMixin):
    """Monitor system resources."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start monitoring system resources."""
        if self._monitoring:
            self.logger.warning("System monitoring already started")
            return
        
        self._monitoring = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._thread.start()
        self.logger.info(f"Started system monitoring with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring system resources."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.logger.info("Stopped system monitoring")
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.monitor.record_metric("system.cpu_percent", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.monitor.record_metric("system.memory_percent", memory.percent)
                self.monitor.record_metric("system.memory_available_gb", memory.available / (1024**3))
                
                # GPU usage (if available)
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        # GPU memory
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        cached = torch.cuda.memory_reserved(i) / (1024**3)
                        
                        self.monitor.record_metric(
                            "system.gpu_memory_allocated_gb", 
                            allocated,
                            tags={"device": str(i)}
                        )
                        self.monitor.record_metric(
                            "system.gpu_memory_cached_gb", 
                            cached,
                            tags={"device": str(i)}
                        )
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(interval)


class MetricsCollector(LoggingMixin):
    """Collect and export metrics from PWMK components."""
    
    def __init__(self, export_path: Optional[str] = None):
        self.monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor(self.monitor)
        self.export_path = Path(export_path) if export_path else None
        
        # Component-specific metrics
        self.model_metrics = defaultdict(lambda: defaultdict(float))
        self.training_metrics = defaultdict(list)
        self.belief_metrics = defaultdict(int)
        
    def start_system_monitoring(self, interval: float = 1.0) -> None:
        """Start monitoring system resources."""
        self.system_monitor.start_monitoring(interval)
    
    def stop_system_monitoring(self) -> None:
        """Stop monitoring system resources."""
        self.system_monitor.stop_monitoring()
    
    def record_model_forward(self, model_name: str, batch_size: int, duration: float) -> None:
        """Record model forward pass metrics."""
        self.monitor.record_metric(
            "model.forward_duration", 
            duration,
            tags={"model": model_name, "batch_size": str(batch_size)}
        )
        self.monitor.record_metric(
            "model.throughput_samples_per_sec",
            batch_size / duration if duration > 0 else 0,
            tags={"model": model_name}
        )
        self.monitor.increment_counter(f"model.{model_name}.forward_calls")
    
    def record_belief_operation(self, operation: str, agent_id: str, duration: float) -> None:
        """Record belief store operation metrics."""
        self.monitor.record_metric(
            "belief.operation_duration",
            duration,
            tags={"operation": operation, "agent": agent_id}
        )
        self.monitor.increment_counter(f"belief.{operation}")
        self.belief_metrics[operation] += 1
    
    def record_planning_step(self, planner_type: str, duration: float, plan_length: int) -> None:
        """Record planning metrics."""
        self.monitor.record_metric(
            "planning.step_duration",
            duration,
            tags={"planner": planner_type}
        )
        self.monitor.record_metric(
            "planning.plan_length",
            plan_length,
            tags={"planner": planner_type}
        )
        self.monitor.increment_counter(f"planning.{planner_type}.steps")
    
    def record_environment_step(self, env_name: str, num_agents: int, duration: float) -> None:
        """Record environment step metrics."""
        self.monitor.record_metric(
            "environment.step_duration",
            duration,
            tags={"environment": env_name, "agents": str(num_agents)}
        )
        self.monitor.increment_counter(f"environment.{env_name}.steps")
    
    def record_training_loss(self, loss_name: str, value: float, epoch: int, step: int) -> None:
        """Record training loss metrics."""
        self.monitor.record_metric(
            f"training.{loss_name}",
            value,
            tags={"epoch": str(epoch), "step": str(step)}
        )
        self.training_metrics[loss_name].append({
            "value": value,
            "epoch": epoch,
            "step": step,
            "timestamp": time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        summary = {
            "timestamp": time.time(),
            "metrics": self.monitor.get_all_stats(),
            "belief_operations": dict(self.belief_metrics),
            "model_metrics": dict(self.model_metrics)
        }
        
        # Add training summaries
        if self.training_metrics:
            training_summary = {}
            for loss_name, values in self.training_metrics.items():
                if values:
                    latest_values = values[-10:]  # Last 10 values
                    training_summary[loss_name] = {
                        "latest": values[-1]["value"],
                        "recent_mean": sum(v["value"] for v in latest_values) / len(latest_values),
                        "total_steps": len(values)
                    }
            summary["training"] = training_summary
        
        return summary
    
    def export_metrics(self, filepath: Optional[str] = None) -> None:
        """Export metrics to file."""
        export_file = Path(filepath) if filepath else self.export_path
        if not export_file:
            self.logger.warning("No export path specified")
            return
        
        summary = self.get_summary()
        
        export_file.parent.mkdir(parents=True, exist_ok=True)
        with open(export_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Exported metrics to {export_file}")
    
    def log_summary(self) -> None:
        """Log current metrics summary."""
        summary = self.get_summary()
        
        self.logger.info("=== PWMK Metrics Summary ===")
        
        # System metrics
        if "system.cpu_percent" in summary["metrics"]:
            cpu_stats = summary["metrics"]["system.cpu_percent"]
            self.logger.info(f"CPU: {cpu_stats.get('latest', 0):.1f}% (avg: {cpu_stats.get('mean', 0):.1f}%)")
        
        if "system.memory_percent" in summary["metrics"]:
            mem_stats = summary["metrics"]["system.memory_percent"]
            self.logger.info(f"Memory: {mem_stats.get('latest', 0):.1f}% (avg: {mem_stats.get('mean', 0):.1f}%)")
        
        # Counters
        if summary["metrics"].get("counters"):
            self.logger.info("Counters:")
            for name, count in summary["metrics"]["counters"].items():
                self.logger.info(f"  {name}: {count}")
        
        # Belief operations
        if summary["belief_operations"]:
            self.logger.info("Belief Operations:")
            for op, count in summary["belief_operations"].items():
                self.logger.info(f"  {op}: {count}")


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def setup_monitoring(
    export_path: Optional[str] = None,
    system_monitoring: bool = True,
    interval: float = 1.0
) -> MetricsCollector:
    """
    Setup global monitoring system.
    
    Args:
        export_path: Path to export metrics file
        system_monitoring: Whether to start system resource monitoring
        interval: System monitoring interval in seconds
        
    Returns:
        Configured metrics collector
    """
    collector = get_metrics_collector()
    
    if export_path:
        collector.export_path = Path(export_path)
    
    if system_monitoring:
        collector.start_system_monitoring(interval)
    
    logger = get_logger(__name__)
    logger.info("PWMK monitoring system initialized")
    
    return collector