"""
Quantum algorithm monitoring and metrics collection.

Provides comprehensive monitoring capabilities for quantum-enhanced planning
algorithms with real-time performance tracking and visualization.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from enum import Enum
from pathlib import Path

from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector


class MetricType(Enum):
    """Types of quantum metrics to track."""
    QUANTUM_ADVANTAGE = "quantum_advantage"
    COHERENCE_TIME = "coherence_time"
    ENTANGLEMENT_STRENGTH = "entanglement_strength"
    INTERFERENCE_PATTERN = "interference_pattern"
    GATE_FIDELITY = "gate_fidelity"
    CIRCUIT_DEPTH = "circuit_depth"
    ANNEALING_CONVERGENCE = "annealing_convergence"
    PARAMETER_ADAPTATION = "parameter_adaptation"


@dataclass
class QuantumMetric:
    """Individual quantum metric measurement."""
    name: str
    value: Any
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class QuantumDashboardConfig:
    """Configuration for quantum monitoring dashboard."""
    update_interval: float = 1.0
    history_size: int = 1000
    enable_realtime: bool = True
    export_format: str = "json"
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class QuantumMetricsCollector(LoggingMixin):
    """
    Collector for quantum algorithm metrics and performance data.
    
    Tracks quantum-specific metrics like quantum advantage, coherence,
    and algorithm performance for monitoring and optimization.
    """
    
    def __init__(
        self,
        dashboard_config: Optional[QuantumDashboardConfig] = None,
        export_path: Optional[str] = None
    ):
        super().__init__()
        
        self.config = dashboard_config or QuantumDashboardConfig()
        self.export_path = Path(export_path) if export_path else Path("quantum_metrics")
        self.export_path.mkdir(exist_ok=True)
        
        # Metric storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=self.config.history_size))
        self.real_time_metrics = {}
        self.aggregated_metrics = {}
        
        # Alert system
        self.alert_callbacks = {}
        self.alert_states = {}
        
        # Performance tracking
        self.operation_timings = defaultdict(list)
        self.algorithm_stats = defaultdict(dict)
        
        self.logger.info(
            f"Initialized QuantumMetricsCollector: history_size={self.config.history_size}, "
            f"export_path={self.export_path}"
        )
    
    def record_quantum_metric(
        self,
        metric_type: MetricType,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Record a quantum-specific metric.
        
        Args:
            metric_type: Type of quantum metric
            value: Metric value
            metadata: Additional metadata
            tags: Metric tags for categorization
        """
        try:
            timestamp = time.time()
            
            metric = QuantumMetric(
                name=metric_type.value,
                value=value,
                timestamp=timestamp,
                metadata=metadata or {},
                tags=tags or []
            )
            
            # Store in history
            self.metrics_history[metric_type.value].append(metric)
            
            # Update real-time metrics
            self.real_time_metrics[metric_type.value] = metric
            
            # Check alerts
            self._check_metric_alerts(metric_type, value)
            
            # Log significant metrics
            if metric_type in [MetricType.QUANTUM_ADVANTAGE, MetricType.ANNEALING_CONVERGENCE]:
                self.logger.debug(f"Recorded {metric_type.value}: {value}")
            
            # Update core metrics collector
            get_metrics_collector().record_metric(metric_type.value, float(value) if isinstance(value, (int, float)) else 0.0)
            
        except Exception as e:
            self.logger.error(f"Failed to record quantum metric {metric_type.value}: {e}")
    
    def record_operation_timing(
        self,
        operation_name: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record timing for quantum operations."""
        
        try:
            timing_data = {
                "duration": duration,
                "success": success,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            self.operation_timings[operation_name].append(timing_data)
            
            # Keep only recent timings
            max_timings = 1000
            if len(self.operation_timings[operation_name]) > max_timings:
                self.operation_timings[operation_name] = self.operation_timings[operation_name][-max_timings:]
            
            # Record in core metrics
            get_metrics_collector().record_quantum_operation(operation_name, duration)
            
        except Exception as e:
            self.logger.error(f"Failed to record operation timing for {operation_name}: {e}")
    
    def update_algorithm_stats(
        self,
        algorithm_name: str,
        stats: Dict[str, Any]
    ) -> None:
        """Update algorithm-specific statistics."""
        
        try:
            self.algorithm_stats[algorithm_name].update(stats)
            self.algorithm_stats[algorithm_name]["last_updated"] = time.time()
            
            # Record key stats as metrics
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    self.record_quantum_metric(
                        MetricType.PARAMETER_ADAPTATION,
                        value,
                        metadata={"algorithm": algorithm_name, "stat_name": key}
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to update algorithm stats for {algorithm_name}: {e}")
    
    def get_quantum_advantage_trend(self, window_size: int = 100) -> List[float]:
        """Get trend of quantum advantage over time."""
        
        try:
            advantage_metrics = list(self.metrics_history[MetricType.QUANTUM_ADVANTAGE.value])
            
            if len(advantage_metrics) < 2:
                return []
            
            # Get recent measurements
            recent_metrics = advantage_metrics[-window_size:]
            values = [m.value for m in recent_metrics if isinstance(m.value, (int, float))]
            
            return values
            
        except Exception as e:
            self.logger.error(f"Failed to get quantum advantage trend: {e}")
            return []
    
    def get_coherence_analysis(self) -> Dict[str, Any]:
        """Analyze quantum coherence patterns."""
        
        try:
            coherence_metrics = list(self.metrics_history[MetricType.COHERENCE_TIME.value])
            
            if not coherence_metrics:
                return {"status": "no_data"}
            
            values = [m.value for m in coherence_metrics if isinstance(m.value, (int, float))]
            
            if not values:
                return {"status": "no_valid_data"}
            
            analysis = {
                "status": "active",
                "mean_coherence": float(np.mean(values)),
                "std_coherence": float(np.std(values)),
                "min_coherence": float(np.min(values)),
                "max_coherence": float(np.max(values)),
                "trend": self._calculate_trend(values),
                "stability_score": self._calculate_stability_score(values)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze coherence: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_circuit_optimization_metrics(self) -> Dict[str, Any]:
        """Get circuit optimization performance metrics."""
        
        try:
            gate_metrics = list(self.metrics_history[MetricType.GATE_FIDELITY.value])
            depth_metrics = list(self.metrics_history[MetricType.CIRCUIT_DEPTH.value])
            
            optimization_data = {
                "gate_fidelity": {
                    "current": gate_metrics[-1].value if gate_metrics else None,
                    "average": float(np.mean([m.value for m in gate_metrics if isinstance(m.value, (int, float))])) if gate_metrics else None,
                    "count": len(gate_metrics)
                },
                "circuit_depth": {
                    "current": depth_metrics[-1].value if depth_metrics else None,
                    "average": float(np.mean([m.value for m in depth_metrics if isinstance(m.value, (int, float))])) if depth_metrics else None,
                    "reduction_trend": self._calculate_reduction_trend(depth_metrics)
                }
            }
            
            return optimization_data
            
        except Exception as e:
            self.logger.error(f"Failed to get circuit optimization metrics: {e}")
            return {"error": str(e)}
    
    def get_annealing_performance(self) -> Dict[str, Any]:
        """Get quantum annealing performance analysis."""
        
        try:
            convergence_metrics = list(self.metrics_history[MetricType.ANNEALING_CONVERGENCE.value])
            
            if not convergence_metrics:
                return {"status": "no_data"}
            
            # Analyze convergence patterns
            convergence_times = []
            final_energies = []
            
            for metric in convergence_metrics:
                if isinstance(metric.value, dict):
                    if "convergence_time" in metric.value:
                        convergence_times.append(metric.value["convergence_time"])
                    if "final_energy" in metric.value:
                        final_energies.append(metric.value["final_energy"])
                elif isinstance(metric.value, (int, float)):
                    convergence_times.append(metric.value)
            
            performance = {
                "status": "active",
                "convergence_analysis": {
                    "mean_time": float(np.mean(convergence_times)) if convergence_times else None,
                    "std_time": float(np.std(convergence_times)) if convergence_times else None,
                    "success_rate": len([t for t in convergence_times if t > 0]) / len(convergence_times) if convergence_times else 0
                },
                "energy_analysis": {
                    "mean_energy": float(np.mean(final_energies)) if final_energies else None,
                    "best_energy": float(np.min(final_energies)) if final_energies else None,
                    "energy_trend": self._calculate_trend(final_energies) if final_energies else None
                }
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Failed to analyze annealing performance: {e}")
            return {"status": "error", "error": str(e)}
    
    def set_alert_threshold(
        self,
        metric_type: MetricType,
        threshold: float,
        callback: Optional[Callable[[str, float, float], None]] = None
    ) -> None:
        """Set alert threshold for a metric type."""
        
        self.config.alert_thresholds[metric_type.value] = threshold
        
        if callback:
            self.alert_callbacks[metric_type.value] = callback
        
        self.logger.info(f"Set alert threshold for {metric_type.value}: {threshold}")
    
    def _check_metric_alerts(self, metric_type: MetricType, value: Any) -> None:
        """Check if metric value triggers any alerts."""
        
        try:
            metric_name = metric_type.value
            
            if metric_name not in self.config.alert_thresholds:
                return
            
            threshold = self.config.alert_thresholds[metric_name]
            
            if not isinstance(value, (int, float)):
                return
            
            # Check threshold violation
            alert_triggered = False
            
            if metric_type == MetricType.QUANTUM_ADVANTAGE:
                # Alert if quantum advantage drops below threshold
                alert_triggered = value < threshold
            elif metric_type == MetricType.COHERENCE_TIME:
                # Alert if coherence time drops below threshold
                alert_triggered = value < threshold
            elif metric_type == MetricType.GATE_FIDELITY:
                # Alert if fidelity drops below threshold
                alert_triggered = value < threshold
            else:
                # Generic threshold check
                alert_triggered = abs(value) > threshold
            
            if alert_triggered:
                self._trigger_alert(metric_name, value, threshold)
            
        except Exception as e:
            self.logger.error(f"Failed to check alerts for {metric_type.value}: {e}")
    
    def _trigger_alert(self, metric_name: str, value: float, threshold: float) -> None:
        """Trigger alert for metric threshold violation."""
        
        # Avoid duplicate alerts
        alert_key = f"{metric_name}_{threshold}"
        current_time = time.time()
        
        if alert_key in self.alert_states:
            last_alert_time = self.alert_states[alert_key]
            if current_time - last_alert_time < 60.0:  # 1 minute cooldown
                return
        
        self.alert_states[alert_key] = current_time
        
        # Log alert
        self.logger.warning(
            f"QUANTUM ALERT: {metric_name} = {value} (threshold: {threshold})"
        )
        
        # Call custom callback if registered
        if metric_name in self.alert_callbacks:
            try:
                self.alert_callbacks[metric_name](metric_name, value, threshold)
            except Exception as e:
                self.logger.error(f"Alert callback failed for {metric_name}: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:  
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_stability_score(self, values: List[float]) -> float:
        """Calculate stability score (lower is more stable)."""
        
        if len(values) < 2:
            return 1.0
        
        # Coefficient of variation as stability measure
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 1.0
        
        cv = std_val / abs(mean_val)
        return min(1.0, cv)  # Cap at 1.0
    
    def _calculate_reduction_trend(self, metrics: List[QuantumMetric]) -> Optional[float]:
        """Calculate reduction trend for optimization metrics."""
        
        if len(metrics) < 2:
            return None
        
        values = [m.value for m in metrics if isinstance(m.value, (int, float))]
        
        if len(values) < 2:
            return None
        
        # Calculate percentage reduction from first to last
        first_val = values[0]
        last_val = values[-1]
        
        if first_val == 0:
            return 0.0
        
        reduction = (first_val - last_val) / first_val
        return float(reduction)
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """Export metrics to file."""
        
        try:
            if filename is None:
                timestamp = int(time.time())
                filename = f"quantum_metrics_{timestamp}.{self.config.export_format}"
            
            export_file = self.export_path / filename
            
            # Prepare export data
            export_data = {
                "export_timestamp": time.time(),
                "config": {
                    "history_size": self.config.history_size,
                    "update_interval": self.config.update_interval
                },
                "metrics_summary": self.get_metrics_summary(),
                "algorithm_stats": dict(self.algorithm_stats),
                "operation_timings": self._summarize_operation_timings()
            }
            
            # Export based on format
            if self.config.export_format == "json":
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {self.config.export_format}")
            
            self.logger.info(f"Exported quantum metrics to {export_file}")
            return str(export_file)
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            raise
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all quantum metrics."""
        
        summary = {}
        
        for metric_type in MetricType:
            metric_name = metric_type.value
            metrics = list(self.metrics_history[metric_name])
            
            if metrics:
                values = [m.value for m in metrics if isinstance(m.value, (int, float))]
                
                if values:
                    summary[metric_name] = {
                        "count": len(values),
                        "latest": values[-1],
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "trend": self._calculate_trend(values)
                    }
                else:
                    summary[metric_name] = {"count": 0, "status": "no_numeric_data"}
            else:
                summary[metric_name] = {"count": 0, "status": "no_data"}
        
        return summary
    
    def _summarize_operation_timings(self) -> Dict[str, Any]:
        """Summarize operation timing statistics."""
        
        timing_summary = {}
        
        for operation_name, timings in self.operation_timings.items():
            if timings:
                durations = [t["duration"] for t in timings]
                success_count = sum(1 for t in timings if t["success"])
                
                timing_summary[operation_name] = {
                    "total_operations": len(timings),
                    "success_rate": success_count / len(timings),
                    "mean_duration": float(np.mean(durations)),
                    "std_duration": float(np.std(durations)),
                    "min_duration": float(np.min(durations)),
                    "max_duration": float(np.max(durations))
                }
        
        return timing_summary
    
    def clear_metrics(self, metric_type: Optional[MetricType] = None) -> None:
        """Clear metrics history."""
        
        if metric_type:
            self.metrics_history[metric_type.value].clear()
            self.logger.info(f"Cleared metrics for {metric_type.value}")
        else:
            self.metrics_history.clear()
            self.real_time_metrics.clear()
            self.logger.info("Cleared all metrics")
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time dashboard display."""
        
        try:
            dashboard_data = {
                "timestamp": time.time(),
                "quantum_advantage": self.get_quantum_advantage_trend(50),
                "coherence_analysis": self.get_coherence_analysis(),
                "circuit_metrics": self.get_circuit_optimization_metrics(),
                "annealing_performance": self.get_annealing_performance(),
                "real_time_metrics": {
                    name: {
                        "value": metric.value,
                        "timestamp": metric.timestamp,
                        "tags": metric.tags
                    }
                    for name, metric in self.real_time_metrics.items()
                },
                "alert_status": {
                    "active_alerts": len(self.alert_states),
                    "thresholds": self.config.alert_thresholds
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e), "timestamp": time.time()}