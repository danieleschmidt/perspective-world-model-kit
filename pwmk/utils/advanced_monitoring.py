"""Advanced monitoring and observability system for PWMK."""

import time
import threading
import psutil
import queue
import json
from typing import Dict, Any, Optional, Callable, List, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from contextlib import contextmanager

from .logging import LoggingMixin


class MetricType:
    """Metric types for monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class Metric:
    """Individual metric data structure."""
    name: str
    value: float
    metric_type: str
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


class MetricBuffer:
    """Thread-safe metric buffer with automatic aggregation."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.metrics: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.aggregations: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def add_metric(self, metric: Metric) -> None:
        """Add a metric to the buffer."""
        with self.lock:
            self.metrics.append(metric)
            self._update_aggregations(metric)
    
    def _update_aggregations(self, metric: Metric) -> None:
        """Update running aggregations."""
        key = f"{metric.name}:{metric.metric_type}"
        
        if key not in self.aggregations:
            self.aggregations[key] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0
            }
        
        agg = self.aggregations[key]
        agg["count"] += 1
        agg["sum"] += metric.value
        agg["min"] = min(agg["min"], metric.value)
        agg["max"] = max(agg["max"], metric.value)
        agg["avg"] = agg["sum"] / agg["count"]
    
    def get_recent_metrics(self, minutes: int = 5) -> List[Metric]:
        """Get metrics from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.lock:
            return [m for m in self.metrics if m.timestamp >= cutoff_time]
    
    def get_aggregations(self) -> Dict[str, Dict[str, float]]:
        """Get current aggregations."""
        with self.lock:
            return dict(self.aggregations)
    
    def clear_old_metrics(self, hours: int = 24) -> None:
        """Clear metrics older than specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            while self.metrics and self.metrics[0].timestamp < cutoff_time:
                self.metrics.popleft()


class SystemMonitor(LoggingMixin):
    """System resource monitoring."""
    
    def __init__(self, interval: float = 1.0):
        super().__init__()
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.metric_buffer = MetricBuffer()
    
    def start(self) -> None:
        """Start system monitoring."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        self.log_info("System monitoring started")
    
    def stop(self) -> None:
        """Stop system monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
        self.log_info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.interval)
            except Exception as e:
                self.log_error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metric_buffer.add_metric(Metric(
            name="system.cpu.usage",
            value=cpu_percent,
            metric_type=MetricType.GAUGE,
            unit="%"
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metric_buffer.add_metric(Metric(
            name="system.memory.usage",
            value=memory.percent,
            metric_type=MetricType.GAUGE,
            unit="%"
        ))
        
        self.metric_buffer.add_metric(Metric(
            name="system.memory.available",
            value=memory.available,
            metric_type=MetricType.GAUGE,
            unit="bytes"
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metric_buffer.add_metric(Metric(
            name="system.disk.usage",
            value=(disk.used / disk.total) * 100,
            metric_type=MetricType.GAUGE,
            unit="%"
        ))
        
        # Network metrics (if available)
        try:
            net_io = psutil.net_io_counters()
            self.metric_buffer.add_metric(Metric(
                name="system.network.bytes_sent",
                value=net_io.bytes_sent,
                metric_type=MetricType.COUNTER,
                unit="bytes"
            ))
            
            self.metric_buffer.add_metric(Metric(
                name="system.network.bytes_recv",
                value=net_io.bytes_recv,
                metric_type=MetricType.COUNTER,
                unit="bytes"
            ))
        except Exception:
            pass  # Network stats might not be available
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current system statistics."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": (lambda d: (d.used / d.total) * 100)(psutil.disk_usage('/')),
            "load_avg": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        }


class PerformanceProfiler(LoggingMixin):
    """Advanced performance profiling for PWMK components."""
    
    def __init__(self):
        super().__init__()
        self.active_timers: Dict[str, float] = {}
        self.completed_timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
    
    @contextmanager
    def profile_operation(self, operation_name: str, **metadata):
        """Context manager for profiling operations."""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            with self.lock:
                self.completed_timings[operation_name].append(duration)
                self.call_counts[operation_name] += 1
            
            self.log_metric(f"performance.{operation_name}.duration", duration, **metadata)
            self.log_metric(f"performance.{operation_name}.calls", self.call_counts[operation_name])
    
    def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        with self.lock:
            if operation:
                if operation not in self.completed_timings:
                    return {}
                
                timings = self.completed_timings[operation]
                return {
                    "operation": operation,
                    "total_calls": self.call_counts[operation],
                    "avg_duration": sum(timings) / len(timings),
                    "min_duration": min(timings),
                    "max_duration": max(timings),
                    "total_time": sum(timings)
                }
            
            # Summary for all operations
            summary = {}
            for op_name, timings in self.completed_timings.items():
                summary[op_name] = {
                    "total_calls": self.call_counts[op_name],
                    "avg_duration": sum(timings) / len(timings),
                    "min_duration": min(timings),
                    "max_duration": max(timings),
                    "total_time": sum(timings)
                }
            
            return summary


class AlertManager(LoggingMixin):
    """Alert management system for monitoring thresholds."""
    
    def __init__(self):
        super().__init__()
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
    
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[float], bool],
        threshold: float,
        metric_name: str,
        severity: str = "warning",
        cooldown: int = 300  # 5 minutes
    ) -> None:
        """Add an alert rule."""
        self.alert_rules[name] = {
            "condition": condition,
            "threshold": threshold,
            "metric_name": metric_name,
            "severity": severity,
            "cooldown": cooldown,
            "last_triggered": 0
        }
        
        self.log_info(f"Added alert rule: {name}")
    
    def check_alerts(self, metrics: List[Metric]) -> List[Dict[str, Any]]:
        """Check metrics against alert rules."""
        triggered_alerts = []
        current_time = time.time()
        
        # Group metrics by name for easy lookup
        metric_values = {}
        for metric in metrics:
            if metric.name not in metric_values:
                metric_values[metric.name] = []
            metric_values[metric.name].append(metric.value)
        
        for alert_name, rule in self.alert_rules.items():
            metric_name = rule["metric_name"]
            
            if metric_name not in metric_values:
                continue
            
            # Use the latest value
            latest_value = metric_values[metric_name][-1]
            
            # Check condition
            if rule["condition"](latest_value):
                # Check cooldown
                if (current_time - rule["last_triggered"]) < rule["cooldown"]:
                    continue
                
                alert_data = {
                    "name": alert_name,
                    "metric_name": metric_name,
                    "value": latest_value,
                    "threshold": rule["threshold"],
                    "severity": rule["severity"],
                    "timestamp": current_time
                }
                
                triggered_alerts.append(alert_data)
                self.active_alerts[alert_name] = alert_data
                self.alert_history.append(alert_data)
                rule["last_triggered"] = current_time
                
                self.log_warning(
                    f"Alert triggered: {alert_name}",
                    extra={
                        "alert_name": alert_name,
                        "metric_value": latest_value,
                        "threshold": rule["threshold"],
                        "severity": rule["severity"]
                    }
                )
        
        return triggered_alerts
    
    def resolve_alert(self, alert_name: str) -> bool:
        """Manually resolve an alert."""
        if alert_name in self.active_alerts:
            del self.active_alerts[alert_name]
            self.log_info(f"Alert resolved: {alert_name}")
            return True
        return False
    
    def get_active_alerts(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active alerts."""
        return dict(self.active_alerts)


class ComprehensiveMonitor(LoggingMixin):
    """Comprehensive monitoring system combining all monitoring capabilities."""
    
    def __init__(self, system_monitor_interval: float = 1.0):
        super().__init__()
        self.system_monitor = SystemMonitor(system_monitor_interval)
        self.profiler = PerformanceProfiler()
        self.alert_manager = AlertManager()
        self.metric_buffer = MetricBuffer()
        self.running = False
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules for common issues."""
        # High CPU usage
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            lambda x: x > 80,
            80.0,
            "system.cpu.usage",
            "warning"
        )
        
        # High memory usage
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            lambda x: x > 85,
            85.0,
            "system.memory.usage",
            "warning"
        )
        
        # Disk space warning
        self.alert_manager.add_alert_rule(
            "low_disk_space",
            lambda x: x > 90,
            90.0,
            "system.disk.usage",
            "critical"
        )
    
    def start(self) -> None:
        """Start comprehensive monitoring."""
        if self.running:
            return
        
        self.running = True
        self.system_monitor.start()
        self.log_info("Comprehensive monitoring started")
    
    def stop(self) -> None:
        """Stop comprehensive monitoring."""
        if not self.running:
            return
        
        self.running = False
        self.system_monitor.stop()
        self.log_info("Comprehensive monitoring stopped")
    
    def add_metric(self, name: str, value: float, metric_type: str = MetricType.GAUGE, **labels) -> None:
        """Add a custom metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels
        )
        
        self.metric_buffer.add_metric(metric)
        self.system_monitor.metric_buffer.add_metric(metric)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        # Get system stats
        system_stats = self.system_monitor.get_current_stats()
        
        # Get performance summaries
        performance_summary = self.profiler.get_performance_summary()
        
        # Get recent metrics
        recent_metrics = self.metric_buffer.get_recent_metrics(minutes=5)
        
        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Check for new alerts
        triggered_alerts = self.alert_manager.check_alerts(recent_metrics)
        
        return {
            "system_stats": system_stats,
            "performance_summary": performance_summary,
            "recent_metrics_count": len(recent_metrics),
            "active_alerts": active_alerts,
            "new_alerts": triggered_alerts,
            "timestamp": time.time()
        }
    
    @contextmanager
    def monitor_operation(self, operation_name: str, **metadata):
        """Context manager for monitoring operations with metrics and profiling."""
        with self.profiler.profile_operation(operation_name, **metadata):
            start_time = time.time()
            
            try:
                yield
                # Record success metric
                self.add_metric(f"{operation_name}.success", 1, MetricType.COUNTER)
                
            except Exception as e:
                # Record failure metric
                self.add_metric(f"{operation_name}.failure", 1, MetricType.COUNTER)
                self.log_error(f"Operation {operation_name} failed: {str(e)}")
                raise
            
            finally:
                # Record duration
                duration = time.time() - start_time
                self.add_metric(f"{operation_name}.duration", duration, MetricType.TIMING)


# Global monitoring instance
global_monitor = ComprehensiveMonitor()