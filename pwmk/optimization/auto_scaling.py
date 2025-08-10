"""Auto-scaling and adaptive resource management for PWMK."""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
import psutil
import torch

from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector


@dataclass 
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    queue_size: int
    error_rate: float
    timestamp: float


class AutoScaler(LoggingMixin):
    """Automatic scaling based on system metrics and load."""
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 16,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_period: float = 60.0,
        monitoring_interval: float = 10.0
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.monitoring_interval = monitoring_interval
        
        # Current state
        self.current_workers = min_workers
        self.last_scale_time = 0.0
        self.metrics_history: deque = deque(maxlen=100)
        
        # Monitoring
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics = get_metrics_collector()
        
        # Callbacks for scaling actions
        self.scale_up_callbacks: List[Callable[[int], None]] = []
        self.scale_down_callbacks: List[Callable[[int], None]] = []
        
        self.logger.info(
            f"AutoScaler initialized: min={min_workers}, max={max_workers}, "
            f"up_threshold={scale_up_threshold}, down_threshold={scale_down_threshold}"
        )
    
    def start_monitoring(self) -> None:
        """Start the auto-scaling monitor."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Auto-scaling monitor started")
    
    def stop_monitoring(self) -> None:
        """Stop the auto-scaling monitor."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Auto-scaling monitor stopped")
    
    def add_scale_up_callback(self, callback: Callable[[int], None]) -> None:
        """Add callback for scale-up events."""
        self.scale_up_callbacks.append(callback)
    
    def add_scale_down_callback(self, callback: Callable[[int], None]) -> None:
        """Add callback for scale-down events."""
        self.scale_down_callbacks.append(callback)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Make scaling decision
                self._evaluate_scaling_decision(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system and application metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent / 100.0
        
        # Application metrics from metrics collector
        stats = self.metrics.monitor.get_all_stats()
        
        # Calculate request rate (requests per second)
        request_rate = 0.0
        if "belief_query_duration" in stats and "count" in stats["belief_query_duration"]:
            recent_queries = stats["belief_query_duration"]["count"]
            if len(self.metrics_history) > 0:
                time_diff = time.time() - self.metrics_history[-1].timestamp
                if time_diff > 0:
                    prev_queries = getattr(self.metrics_history[-1], 'total_queries', 0)
                    request_rate = max(0, (recent_queries - prev_queries) / time_diff)
        
        # Calculate average response time
        response_time = 0.0
        if "belief_query_duration" in stats and "mean" in stats["belief_query_duration"]:
            response_time = stats["belief_query_duration"]["mean"]
        
        # Queue size (approximation)
        queue_size = len(getattr(self.metrics, 'active_tasks', {}))
        
        # Error rate
        error_rate = 0.0
        counters = stats.get("counters", {})
        total_errors = (
            counters.get("belief_query_security_errors", 0) +
            counters.get("belief_query_general_errors", 0) +
            counters.get("belief_add_security_errors", 0) +
            counters.get("belief_add_general_errors", 0)
        )
        total_operations = (
            counters.get("belief_queries", 0) + 
            counters.get("belief_adds", 0)
        )
        if total_operations > 0:
            error_rate = total_errors / total_operations
        
        return ScalingMetrics(
            cpu_usage=cpu_usage / 100.0,
            memory_usage=memory_usage,
            request_rate=request_rate,
            response_time=response_time,
            queue_size=queue_size,
            error_rate=error_rate,
            timestamp=time.time()
        )
    
    def _evaluate_scaling_decision(self, current_metrics: ScalingMetrics) -> None:
        """Evaluate whether to scale up or down."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_time < self.cooldown_period:
            return
        
        # Calculate load score (weighted combination of metrics)
        load_score = self._calculate_load_score(current_metrics)
        
        # Make scaling decision
        if load_score > self.scale_up_threshold and self.current_workers < self.max_workers:
            self._scale_up(current_metrics, load_score)
        elif load_score < self.scale_down_threshold and self.current_workers > self.min_workers:
            self._scale_down(current_metrics, load_score)
    
    def _calculate_load_score(self, metrics: ScalingMetrics) -> float:
        """Calculate overall load score from metrics."""
        # Weighted combination of different metrics
        weights = {
            "cpu": 0.3,
            "memory": 0.2,
            "response_time": 0.3,
            "queue_size": 0.1,
            "error_rate": 0.1
        }
        
        # Normalize response time (assume 1 second is high load)
        normalized_response_time = min(metrics.response_time, 1.0)
        
        # Normalize queue size (assume 100 items is high load)
        normalized_queue_size = min(metrics.queue_size / 100.0, 1.0)
        
        load_score = (
            weights["cpu"] * metrics.cpu_usage +
            weights["memory"] * metrics.memory_usage +
            weights["response_time"] * normalized_response_time +
            weights["queue_size"] * normalized_queue_size +
            weights["error_rate"] * metrics.error_rate
        )
        
        return load_score
    
    def _scale_up(self, metrics: ScalingMetrics, load_score: float) -> None:
        """Scale up the number of workers."""
        new_worker_count = min(self.current_workers + 1, self.max_workers)
        
        self.logger.info(
            f"Scaling UP: {self.current_workers} -> {new_worker_count} "
            f"(load_score={load_score:.3f}, cpu={metrics.cpu_usage:.3f}, "
            f"mem={metrics.memory_usage:.3f}, resp_time={metrics.response_time:.3f})"
        )
        
        # Execute scale-up callbacks
        for callback in self.scale_up_callbacks:
            try:
                callback(new_worker_count)
            except Exception as e:
                self.logger.error(f"Scale-up callback failed: {e}")
        
        self.current_workers = new_worker_count
        self.last_scale_time = time.time()
        
        # Record metrics
        self.metrics.monitor.increment_counter("autoscaler_scale_ups")
        self.metrics.monitor.record_metric("autoscaler_worker_count", new_worker_count)
    
    def _scale_down(self, metrics: ScalingMetrics, load_score: float) -> None:
        """Scale down the number of workers."""
        new_worker_count = max(self.current_workers - 1, self.min_workers)
        
        self.logger.info(
            f"Scaling DOWN: {self.current_workers} -> {new_worker_count} "
            f"(load_score={load_score:.3f}, cpu={metrics.cpu_usage:.3f}, "
            f"mem={metrics.memory_usage:.3f})"
        )
        
        # Execute scale-down callbacks
        for callback in self.scale_down_callbacks:
            try:
                callback(new_worker_count)
            except Exception as e:
                self.logger.error(f"Scale-down callback failed: {e}")
        
        self.current_workers = new_worker_count
        self.last_scale_time = time.time()
        
        # Record metrics
        self.metrics.monitor.increment_counter("autoscaler_scale_downs")
        self.metrics.monitor.record_metric("autoscaler_worker_count", new_worker_count)
    
    def get_current_metrics(self) -> Optional[ScalingMetrics]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_scaling_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        recent_metrics = list(self.metrics_history)[-limit:]
        return [
            {
                "timestamp": m.timestamp,
                "cpu_usage": m.cpu_usage,
                "memory_usage": m.memory_usage,
                "request_rate": m.request_rate,
                "response_time": m.response_time,
                "queue_size": m.queue_size,
                "error_rate": m.error_rate,
                "load_score": self._calculate_load_score(m)
            }
            for m in recent_metrics
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        current_metrics = self.get_current_metrics()
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "running": self.running,
            "last_scale_time": self.last_scale_time,
            "current_metrics": {
                "cpu_usage": current_metrics.cpu_usage if current_metrics else 0,
                "memory_usage": current_metrics.memory_usage if current_metrics else 0,
                "load_score": (
                    self._calculate_load_score(current_metrics) 
                    if current_metrics else 0
                )
            }
        }


# Global auto-scaler instance
_auto_scaler = None

def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler