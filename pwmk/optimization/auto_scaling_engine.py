"""Advanced auto-scaling engine with predictive scaling and resource optimization."""

import time
import threading
import math
import statistics
from typing import Dict, Any, List, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

from ..utils.logging import LoggingMixin
from ..utils.advanced_monitoring import MetricType, Metric
from ..utils.resilience import resilient


class ScalingAction:
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ResourceMetrics:
    """Resource metrics for scaling decisions."""
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    queue_length: int
    error_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: float
    action: str
    reason: str
    from_instances: int
    to_instances: int
    metrics: ResourceMetrics
    success: bool = True


class ScalingStrategy(ABC):
    """Abstract base class for scaling strategies."""
    
    @abstractmethod
    def decide_scaling_action(
        self, 
        current_instances: int,
        metrics: ResourceMetrics,
        historical_data: List[ResourceMetrics]
    ) -> tuple[str, int, str]:
        """
        Decide scaling action.
        
        Returns:
            (action, target_instances, reason)
        """
        pass


class ThresholdScalingStrategy(ScalingStrategy, LoggingMixin):
    """Traditional threshold-based scaling."""
    
    def __init__(
        self,
        cpu_scale_up_threshold: float = 70.0,
        cpu_scale_down_threshold: float = 30.0,
        memory_scale_up_threshold: float = 80.0,
        memory_scale_down_threshold: float = 40.0,
        response_time_threshold: float = 1.0,
        min_instances: int = 1,
        max_instances: int = 10
    ):
        super().__init__()
        self.cpu_scale_up = cpu_scale_up_threshold
        self.cpu_scale_down = cpu_scale_down_threshold
        self.memory_scale_up = memory_scale_up_threshold
        self.memory_scale_down = memory_scale_down_threshold
        self.response_time_threshold = response_time_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
    
    def decide_scaling_action(
        self,
        current_instances: int,
        metrics: ResourceMetrics,
        historical_data: List[ResourceMetrics]
    ) -> tuple[str, int, str]:
        """Decide based on threshold rules."""
        
        # Scale up conditions
        if (metrics.cpu_usage > self.cpu_scale_up or 
            metrics.memory_usage > self.memory_scale_up or
            metrics.response_time > self.response_time_threshold):
            
            if current_instances < self.max_instances:
                target = min(current_instances + 1, self.max_instances)
                reason = f"High resource usage - CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%, RT: {metrics.response_time:.2f}s"
                return ScalingAction.SCALE_UP, target, reason
        
        # Scale down conditions
        elif (metrics.cpu_usage < self.cpu_scale_down and
              metrics.memory_usage < self.memory_scale_down and
              metrics.response_time < self.response_time_threshold * 0.5):
            
            if current_instances > self.min_instances:
                target = max(current_instances - 1, self.min_instances)
                reason = f"Low resource usage - CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%"
                return ScalingAction.SCALE_DOWN, target, reason
        
        return ScalingAction.NO_ACTION, current_instances, "Within thresholds"


class PredictiveScalingStrategy(ScalingStrategy, LoggingMixin):
    """Machine learning-based predictive scaling."""
    
    def __init__(
        self,
        prediction_window: int = 300,  # 5 minutes
        min_instances: int = 1,
        max_instances: int = 20,
        confidence_threshold: float = 0.7
    ):
        super().__init__()
        self.prediction_window = prediction_window
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.confidence_threshold = confidence_threshold
        
        # Time series data for prediction
        self.cpu_history: deque = deque(maxlen=1000)
        self.memory_history: deque = deque(maxlen=1000)
        self.request_history: deque = deque(maxlen=1000)
    
    def decide_scaling_action(
        self,
        current_instances: int,
        metrics: ResourceMetrics,
        historical_data: List[ResourceMetrics]
    ) -> tuple[str, int, str]:
        """Decide based on predictive analysis."""
        
        # Update history
        self.cpu_history.append(metrics.cpu_usage)
        self.memory_history.append(metrics.memory_usage)
        self.request_history.append(metrics.request_rate)
        
        # Need sufficient data for prediction
        if len(self.cpu_history) < 10:
            return ScalingAction.NO_ACTION, current_instances, "Insufficient data for prediction"
        
        # Predict future resource usage
        predicted_cpu = self._predict_next_value(list(self.cpu_history))
        predicted_memory = self._predict_next_value(list(self.memory_history))
        predicted_requests = self._predict_next_value(list(self.request_history))
        
        # Calculate required instances based on predictions
        required_instances_cpu = self._calculate_required_instances(predicted_cpu, "cpu")
        required_instances_memory = self._calculate_required_instances(predicted_memory, "memory")
        required_instances_requests = self._calculate_required_instances(predicted_requests, "requests")
        
        # Take the maximum requirement
        predicted_required = max(
            required_instances_cpu,
            required_instances_memory,
            required_instances_requests
        )
        
        # Add safety margin
        safety_margin = max(1, int(predicted_required * 0.2))
        target_instances = min(
            max(predicted_required + safety_margin, self.min_instances),
            self.max_instances
        )
        
        # Decide action
        if target_instances > current_instances:
            reason = f"Predicted high load - CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%, Requests: {predicted_requests:.1f}/s"
            return ScalingAction.SCALE_UP, target_instances, reason
        elif target_instances < current_instances:
            reason = f"Predicted low load - CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%"
            return ScalingAction.SCALE_DOWN, target_instances, reason
        else:
            return ScalingAction.NO_ACTION, current_instances, "Predicted load matches current capacity"
    
    def _predict_next_value(self, values: List[float]) -> float:
        """Simple linear regression prediction."""
        if len(values) < 3:
            return values[-1] if values else 0.0
        
        # Use last 20 points for prediction
        recent_values = values[-20:]
        n = len(recent_values)
        
        # Calculate linear trend
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(recent_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, recent_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return recent_values[-1]
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predict next value
        predicted = slope * n + intercept
        
        # Apply smoothing and bounds
        current_value = recent_values[-1]
        smoothed_prediction = 0.7 * predicted + 0.3 * current_value
        
        return max(0, min(smoothed_prediction, 100))  # Bound between 0-100 for percentages
    
    def _calculate_required_instances(self, predicted_value: float, metric_type: str) -> int:
        """Calculate required instances based on predicted metric value."""
        if metric_type == "cpu":
            # Assume 70% CPU per instance is optimal
            return max(1, math.ceil(predicted_value / 70.0))
        elif metric_type == "memory":
            # Assume 80% memory per instance is optimal
            return max(1, math.ceil(predicted_value / 80.0))
        elif metric_type == "requests":
            # Assume 100 requests per second per instance
            return max(1, math.ceil(predicted_value / 100.0))
        
        return 1


class QuantumInspiredScalingStrategy(ScalingStrategy, LoggingMixin):
    """Quantum-inspired scaling using superposition and entanglement principles."""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 50,
        quantum_coherence_time: float = 60.0  # seconds
    ):
        super().__init__()
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.coherence_time = quantum_coherence_time
        
        # Quantum state components
        self.resource_states: Dict[str, List[float]] = defaultdict(list)
        self.entanglement_matrix: Dict[str, Dict[str, float]] = {}
        
        # Initialize entanglement relationships
        self._initialize_entanglement()
    
    def _initialize_entanglement(self) -> None:
        """Initialize quantum entanglement between metrics."""
        metrics = ["cpu", "memory", "requests", "response_time", "errors"]
        
        for metric1 in metrics:
            self.entanglement_matrix[metric1] = {}
            for metric2 in metrics:
                if metric1 != metric2:
                    # CPU and memory are strongly entangled
                    if (metric1, metric2) in [("cpu", "memory"), ("memory", "cpu")]:
                        self.entanglement_matrix[metric1][metric2] = 0.8
                    # Requests entangled with CPU and response time
                    elif (metric1, metric2) in [("requests", "cpu"), ("cpu", "requests"),
                                                ("requests", "response_time"), ("response_time", "requests")]:
                        self.entanglement_matrix[metric1][metric2] = 0.7
                    # Errors entangled with response time
                    elif (metric1, metric2) in [("errors", "response_time"), ("response_time", "errors")]:
                        self.entanglement_matrix[metric1][metric2] = 0.6
                    else:
                        self.entanglement_matrix[metric1][metric2] = 0.3
                else:
                    self.entanglement_matrix[metric1][metric2] = 1.0
    
    def decide_scaling_action(
        self,
        current_instances: int,
        metrics: ResourceMetrics,
        historical_data: List[ResourceMetrics]
    ) -> tuple[str, int, str]:
        """Quantum-inspired scaling decision."""
        
        # Update quantum states
        self._update_quantum_states(metrics)
        
        # Calculate quantum superposition of scaling needs
        scaling_amplitudes = self._calculate_scaling_amplitudes(metrics, historical_data)
        
        # Apply quantum interference
        interfered_amplitudes = self._apply_quantum_interference(scaling_amplitudes)
        
        # Collapse to classical scaling decision
        target_instances = self._collapse_to_classical_decision(
            interfered_amplitudes, current_instances
        )
        
        # Determine action
        if target_instances > current_instances:
            reason = f"Quantum superposition indicates scale up (amplitude: {interfered_amplitudes['scale_up']:.3f})"
            return ScalingAction.SCALE_UP, target_instances, reason
        elif target_instances < current_instances:
            reason = f"Quantum superposition indicates scale down (amplitude: {interfered_amplitudes['scale_down']:.3f})"
            return ScalingAction.SCALE_DOWN, target_instances, reason
        else:
            return ScalingAction.NO_ACTION, current_instances, "Quantum equilibrium maintained"
    
    def _update_quantum_states(self, metrics: ResourceMetrics) -> None:
        """Update quantum states with new measurements."""
        current_time = time.time()
        
        # Add new measurements
        self.resource_states["cpu"].append((metrics.cpu_usage, current_time))
        self.resource_states["memory"].append((metrics.memory_usage, current_time))
        self.resource_states["requests"].append((metrics.request_rate, current_time))
        self.resource_states["response_time"].append((metrics.response_time, current_time))
        self.resource_states["errors"].append((metrics.error_rate, current_time))
        
        # Maintain coherence time (remove old measurements)
        cutoff_time = current_time - self.coherence_time
        
        for metric_name in self.resource_states:
            self.resource_states[metric_name] = [
                (value, timestamp) for value, timestamp in self.resource_states[metric_name]
                if timestamp >= cutoff_time
            ]
    
    def _calculate_scaling_amplitudes(
        self,
        current_metrics: ResourceMetrics,
        historical_data: List[ResourceMetrics]
    ) -> Dict[str, float]:
        """Calculate quantum amplitudes for scaling actions."""
        
        # Normalize metrics to 0-1 range for quantum calculations
        normalized_cpu = current_metrics.cpu_usage / 100.0
        normalized_memory = current_metrics.memory_usage / 100.0
        normalized_response = min(current_metrics.response_time / 2.0, 1.0)  # Cap at 2 seconds
        normalized_errors = min(current_metrics.error_rate / 0.1, 1.0)  # Cap at 10% error rate
        
        # Calculate quantum wave functions
        cpu_wave = math.sin(normalized_cpu * math.pi / 2)
        memory_wave = math.sin(normalized_memory * math.pi / 2)
        response_wave = math.sin(normalized_response * math.pi / 2)
        error_wave = math.sin(normalized_errors * math.pi / 2)
        
        # Scale up amplitude (high when resources are stressed)
        scale_up_amplitude = (
            cpu_wave * 0.4 +
            memory_wave * 0.3 +
            response_wave * 0.2 +
            error_wave * 0.1
        )
        
        # Scale down amplitude (high when resources are underutilized)
        scale_down_amplitude = (
            (1 - cpu_wave) * 0.4 +
            (1 - memory_wave) * 0.3 +
            (1 - response_wave) * 0.2 +
            (1 - error_wave) * 0.1
        )
        
        # No action amplitude (high when balanced)
        balance_factor = 1 - abs(scale_up_amplitude - scale_down_amplitude)
        no_action_amplitude = balance_factor * 0.8
        
        return {
            "scale_up": scale_up_amplitude,
            "scale_down": scale_down_amplitude,
            "no_action": no_action_amplitude
        }
    
    def _apply_quantum_interference(self, amplitudes: Dict[str, float]) -> Dict[str, float]:
        """Apply quantum interference between entangled metrics."""
        
        # Apply entanglement correlations
        for metric1 in self.resource_states:
            if not self.resource_states[metric1]:
                continue
            
            recent_values1 = [v for v, t in self.resource_states[metric1][-5:]]
            if not recent_values1:
                continue
            
            trend1 = self._calculate_trend(recent_values1)
            
            for metric2, entanglement in self.entanglement_matrix[metric1].items():
                if metric2 not in self.resource_states or not self.resource_states[metric2]:
                    continue
                
                recent_values2 = [v for v, t in self.resource_states[metric2][-5:]]
                if not recent_values2:
                    continue
                
                trend2 = self._calculate_trend(recent_values2)
                
                # Apply interference based on entanglement and trend correlation
                correlation = self._calculate_correlation(recent_values1, recent_values2)
                interference_factor = entanglement * correlation
                
                # Modify amplitudes based on interference
                if trend1 > 0 and trend2 > 0:  # Both increasing
                    amplitudes["scale_up"] += interference_factor * 0.1
                elif trend1 < 0 and trend2 < 0:  # Both decreasing
                    amplitudes["scale_down"] += interference_factor * 0.1
                else:  # Conflicting trends
                    amplitudes["no_action"] += interference_factor * 0.05
        
        # Normalize amplitudes
        total_amplitude = sum(amplitudes.values())
        if total_amplitude > 0:
            for key in amplitudes:
                amplitudes[key] /= total_amplitude
        
        return amplitudes
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum(i * values[i] for i in range(n)) - n * x_mean * y_mean
        denominator = sum(i * i for i in range(n)) - n * x_mean * x_mean
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return max(-1, min(1, slope / max(abs(y_mean), 1)))
    
    def _calculate_correlation(self, values1: List[float], values2: List[float]) -> float:
        """Calculate correlation between two value series."""
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0
        
        mean1 = sum(values1) / len(values1)
        mean2 = sum(values2) / len(values2)
        
        numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
        denom1 = sum((v1 - mean1) ** 2 for v1 in values1)
        denom2 = sum((v2 - mean2) ** 2 for v2 in values2)
        
        if denom1 == 0 or denom2 == 0:
            return 0.0
        
        return numerator / math.sqrt(denom1 * denom2)
    
    def _collapse_to_classical_decision(
        self,
        amplitudes: Dict[str, float],
        current_instances: int
    ) -> int:
        """Collapse quantum superposition to classical scaling decision."""
        
        # Quantum measurement (probabilistic collapse)
        max_amplitude_action = max(amplitudes, key=amplitudes.get)
        max_amplitude_value = amplitudes[max_amplitude_action]
        
        # Only act if amplitude is significantly higher
        if max_amplitude_value < 0.6:
            return current_instances  # No significant quantum state
        
        if max_amplitude_action == "scale_up":
            # Scale up by amount proportional to amplitude
            scale_factor = max_amplitude_value
            additional_instances = max(1, int(current_instances * scale_factor * 0.5))
            return min(current_instances + additional_instances, self.max_instances)
        
        elif max_amplitude_action == "scale_down":
            # Scale down by amount proportional to amplitude
            scale_factor = max_amplitude_value
            remove_instances = max(1, int(current_instances * scale_factor * 0.3))
            return max(current_instances - remove_instances, self.min_instances)
        
        return current_instances


class AutoScalingEngine(LoggingMixin):
    """Advanced auto-scaling engine with multiple strategies and predictive capabilities."""
    
    def __init__(
        self,
        scaling_strategy: ScalingStrategy,
        cooldown_period: float = 300.0,  # 5 minutes
        evaluation_interval: float = 30.0,  # 30 seconds
        metrics_window: int = 20
    ):
        super().__init__()
        self.scaling_strategy = scaling_strategy
        self.cooldown_period = cooldown_period
        self.evaluation_interval = evaluation_interval
        self.metrics_window = metrics_window
        
        # State
        self.current_instances = 1
        self.target_instances = 1
        self.last_scaling_time = 0.0
        self.running = False
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=metrics_window * 10)  # 10x window for history
        self.scaling_events: List[ScalingEvent] = []
        
        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        self.metrics_collector: Optional[Callable[[], ResourceMetrics]] = None
        
        # Threading
        self.engine_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def set_scaling_callbacks(
        self,
        scale_up_callback: Callable[[int], bool],
        scale_down_callback: Callable[[int], bool]
    ) -> None:
        """Set callbacks for actual scaling operations."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
    
    def set_metrics_collector(self, collector: Callable[[], ResourceMetrics]) -> None:
        """Set callback for collecting current metrics."""
        self.metrics_collector = collector
    
    def start(self, initial_instances: int = 1) -> None:
        """Start the auto-scaling engine."""
        if self.running:
            self.log_warning("Auto-scaling engine already running")
            return
        
        self.current_instances = initial_instances
        self.target_instances = initial_instances
        self.running = True
        
        self.engine_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.engine_thread.start()
        
        self.log_info(f"Auto-scaling engine started with {initial_instances} instances")
    
    def stop(self) -> None:
        """Stop the auto-scaling engine."""
        self.running = False
        
        if self.engine_thread:
            self.engine_thread.join()
        
        self.executor.shutdown(wait=True)
        self.log_info("Auto-scaling engine stopped")
    
    def _scaling_loop(self) -> None:
        """Main scaling evaluation loop."""
        while self.running:
            try:
                self._evaluate_scaling()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                self.log_error(f"Error in scaling loop: {str(e)}")
                time.sleep(self.evaluation_interval)
    
    @resilient(retry_attempts=3, timeout=10.0)
    def _evaluate_scaling(self) -> None:
        """Evaluate and execute scaling decisions."""
        
        # Collect current metrics
        if not self.metrics_collector:
            self.log_warning("No metrics collector configured")
            return
        
        current_metrics = self.metrics_collector()
        self.metrics_history.append(current_metrics)
        
        # Check cooldown period
        current_time = time.time()
        if (current_time - self.last_scaling_time) < self.cooldown_period:
            return
        
        # Get historical data for decision making
        historical_data = list(self.metrics_history)
        
        # Make scaling decision
        action, target_instances, reason = self.scaling_strategy.decide_scaling_action(
            self.current_instances, current_metrics, historical_data
        )
        
        # Execute scaling action
        if action != ScalingAction.NO_ACTION:
            success = self._execute_scaling_action(action, target_instances)
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=current_time,
                action=action,
                reason=reason,
                from_instances=self.current_instances,
                to_instances=target_instances,
                metrics=current_metrics,
                success=success
            )
            
            self.scaling_events.append(event)
            
            if success:
                self.current_instances = target_instances
                self.last_scaling_time = current_time
                
                self.log_info(
                    f"Scaling {action}: {event.from_instances} → {event.to_instances}",
                    reason=reason,
                    cpu=current_metrics.cpu_usage,
                    memory=current_metrics.memory_usage,
                    response_time=current_metrics.response_time
                )
    
    def _execute_scaling_action(self, action: str, target_instances: int) -> bool:
        """Execute the scaling action."""
        try:
            if action == ScalingAction.SCALE_UP and self.scale_up_callback:
                return self.scale_up_callback(target_instances)
            elif action == ScalingAction.SCALE_DOWN and self.scale_down_callback:
                return self.scale_down_callback(target_instances)
            
            return False
        except Exception as e:
            self.log_error(f"Failed to execute scaling action {action}: {str(e)}")
            return False
    
    def force_scale_to(self, instances: int, reason: str = "Manual override") -> bool:
        """Force scaling to specific number of instances."""
        current_time = time.time()
        
        if instances > self.current_instances:
            action = ScalingAction.SCALE_UP
            callback = self.scale_up_callback
        elif instances < self.current_instances:
            action = ScalingAction.SCALE_DOWN
            callback = self.scale_down_callback
        else:
            return True  # Already at target
        
        if callback:
            success = callback(instances)
            
            if success:
                # Record event
                event = ScalingEvent(
                    timestamp=current_time,
                    action=action,
                    reason=reason,
                    from_instances=self.current_instances,
                    to_instances=instances,
                    metrics=self.metrics_history[-1] if self.metrics_history else None,
                    success=True
                )
                
                self.scaling_events.append(event)
                self.current_instances = instances
                self.last_scaling_time = current_time
                
                self.log_info(f"Forced scaling to {instances} instances: {reason}")
            
            return success
        
        return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics and current state."""
        recent_events = [e for e in self.scaling_events if e.timestamp > (time.time() - 3600)]  # Last hour
        
        scale_ups = len([e for e in recent_events if e.action == ScalingAction.SCALE_UP])
        scale_downs = len([e for e in recent_events if e.action == ScalingAction.SCALE_DOWN])
        
        avg_response_time = 0.0
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        
        return {
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "strategy": self.scaling_strategy.__class__.__name__,
            "last_scaling_time": self.last_scaling_time,
            "time_since_last_scaling": time.time() - self.last_scaling_time,
            "recent_scale_ups": scale_ups,
            "recent_scale_downs": scale_downs,
            "total_scaling_events": len(self.scaling_events),
            "average_response_time": avg_response_time,
            "metrics_history_size": len(self.metrics_history),
            "is_running": self.running
        }
    
    def get_recent_events(self, hours: int = 24) -> List[ScalingEvent]:
        """Get recent scaling events."""
        cutoff_time = time.time() - (hours * 3600)
        return [e for e in self.scaling_events if e.timestamp > cutoff_time]