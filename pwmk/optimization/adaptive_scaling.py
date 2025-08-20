"""Adaptive scaling and auto-optimization for PWMK components."""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

from ..utils.logging import get_logger
from ..utils.health_monitor import get_health_monitor
from ..utils.fallback_manager import get_fallback_manager


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions."""
    timestamp: float
    latency_p95: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    error_rate: float
    queue_depth: int


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    name: str
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_seconds: float = 300.0
    min_instances: int = 1
    max_instances: int = 10
    enabled: bool = True
    last_action_time: float = 0.0


class AdaptiveScaler:
    """Adaptive auto-scaling for PWMK components."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = get_logger(f"AdaptiveScaler-{component_name}")
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=100)
        self.current_instances = 1
        self.target_instances = 1
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        self._setup_default_rules()
        
        # State management
        self.scaling_enabled = True
        self.last_scale_time = 0.0
        self.scale_lock = threading.Lock()
        
        # Callbacks for scaling actions
        self.scale_up_callback: Optional[Callable[[int], None]] = None
        self.scale_down_callback: Optional[Callable[[int], None]] = None
    
    def _setup_default_rules(self) -> None:
        """Setup default scaling rules for the component."""
        base_rules = [
            ScalingRule(
                name="latency_scaling",
                metric_name="latency_p95",
                scale_up_threshold=1000.0,  # 1 second
                scale_down_threshold=200.0,  # 200ms
                cooldown_seconds=120.0
            ),
            ScalingRule(
                name="throughput_scaling",
                metric_name="throughput", 
                scale_up_threshold=80.0,     # 80% of max throughput
                scale_down_threshold=30.0,   # 30% of max throughput
                cooldown_seconds=180.0
            ),
            ScalingRule(
                name="error_rate_scaling",
                metric_name="error_rate",
                scale_up_threshold=0.05,     # 5% error rate
                scale_down_threshold=0.01,   # 1% error rate
                cooldown_seconds=60.0
            )
        ]
        
        # Component-specific rules
        if self.component_name == "model":
            base_rules.extend([
                ScalingRule(
                    name="memory_scaling",
                    metric_name="memory_usage",
                    scale_up_threshold=85.0,
                    scale_down_threshold=50.0,
                    cooldown_seconds=300.0,
                    max_instances=4  # GPU memory constraints
                )
            ])
        elif self.component_name == "belief_store":
            base_rules.extend([
                ScalingRule(
                    name="queue_depth_scaling",
                    metric_name="queue_depth",
                    scale_up_threshold=100.0,
                    scale_down_threshold=10.0,
                    cooldown_seconds=90.0
                )
            ])
        
        self.scaling_rules = base_rules
    
    def record_metrics(
        self,
        latency_p95: float,
        throughput: float,
        cpu_usage: float,
        memory_usage: float,
        error_rate: float,
        queue_depth: int = 0
    ) -> None:
        """Record performance metrics for scaling decisions."""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            latency_p95=latency_p95,
            throughput=throughput,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            error_rate=error_rate,
            queue_depth=queue_depth
        )
        
        self.metrics_history.append(metrics)
        self.logger.debug(
            f"Recorded metrics: latency={latency_p95:.2f}ms, "
            f"throughput={throughput:.2f}, error_rate={error_rate:.3f}"
        )
        
        # Trigger scaling evaluation
        if self.scaling_enabled:
            self._evaluate_scaling()
    
    def _evaluate_scaling(self) -> None:
        """Evaluate if scaling action is needed."""
        if len(self.metrics_history) < 3:  # Need some history
            return
        
        current_time = time.time()
        
        # Get recent metrics for evaluation
        recent_metrics = list(self.metrics_history)[-5:]  # Last 5 data points
        
        # Evaluate each scaling rule
        scaling_decisions = []
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if current_time - rule.last_action_time < rule.cooldown_seconds:
                continue
            
            decision = self._evaluate_rule(rule, recent_metrics)
            if decision != ScalingDirection.MAINTAIN:
                scaling_decisions.append((rule, decision))
        
        # Make scaling decision based on all rules
        if scaling_decisions:
            self._make_scaling_decision(scaling_decisions, current_time)
    
    def _evaluate_rule(
        self, 
        rule: ScalingRule, 
        recent_metrics: List[PerformanceMetrics]
    ) -> ScalingDirection:
        """Evaluate a single scaling rule."""
        if not recent_metrics:
            return ScalingDirection.MAINTAIN
        
        # Extract metric values
        metric_values = []
        for metrics in recent_metrics:
            if rule.metric_name == "latency_p95":
                metric_values.append(metrics.latency_p95)
            elif rule.metric_name == "throughput":
                metric_values.append(metrics.throughput)
            elif rule.metric_name == "cpu_usage":
                metric_values.append(metrics.cpu_usage)
            elif rule.metric_name == "memory_usage":
                metric_values.append(metrics.memory_usage)
            elif rule.metric_name == "error_rate":
                metric_values.append(metrics.error_rate)
            elif rule.metric_name == "queue_depth":
                metric_values.append(metrics.queue_depth)
        
        if not metric_values:
            return ScalingDirection.MAINTAIN
        
        # Use median to avoid outliers
        current_value = statistics.median(metric_values)
        
        # Make scaling decision
        if current_value > rule.scale_up_threshold and self.current_instances < rule.max_instances:
            self.logger.info(
                f"Rule {rule.name}: {rule.metric_name}={current_value:.2f} > "
                f"threshold={rule.scale_up_threshold} -> SCALE UP"
            )
            return ScalingDirection.UP
        elif current_value < rule.scale_down_threshold and self.current_instances > rule.min_instances:
            self.logger.info(
                f"Rule {rule.name}: {rule.metric_name}={current_value:.2f} < "
                f"threshold={rule.scale_down_threshold} -> SCALE DOWN"
            )
            return ScalingDirection.DOWN
        
        return ScalingDirection.MAINTAIN
    
    def _make_scaling_decision(
        self, 
        scaling_decisions: List[tuple], 
        current_time: float
    ) -> None:
        """Make final scaling decision based on all rule evaluations."""
        with self.scale_lock:
            # Count votes for each direction
            scale_up_votes = sum(1 for _, direction in scaling_decisions if direction == ScalingDirection.UP)
            scale_down_votes = sum(1 for _, direction in scaling_decisions if direction == ScalingDirection.DOWN)
            
            # Require majority for scaling action
            if scale_up_votes > scale_down_votes and scale_up_votes >= 2:
                self._scale_up(current_time)
                # Update rule action times
                for rule, direction in scaling_decisions:
                    if direction == ScalingDirection.UP:
                        rule.last_action_time = current_time
                        
            elif scale_down_votes > scale_up_votes and scale_down_votes >= 2:
                self._scale_down(current_time)
                # Update rule action times
                for rule, direction in scaling_decisions:
                    if direction == ScalingDirection.DOWN:
                        rule.last_action_time = current_time
    
    def _scale_up(self, current_time: float) -> None:
        """Scale up the component."""
        if self.current_instances >= max(rule.max_instances for rule in self.scaling_rules):
            self.logger.warning("Cannot scale up: already at maximum instances")
            return
        
        new_instances = min(
            self.current_instances + 1,
            max(rule.max_instances for rule in self.scaling_rules)
        )
        
        self.logger.info(
            f"Scaling {self.component_name} UP: {self.current_instances} -> {new_instances}"
        )
        
        if self.scale_up_callback:
            try:
                self.scale_up_callback(new_instances)
                self.current_instances = new_instances
                self.target_instances = new_instances
                self.last_scale_time = current_time
            except Exception as e:
                self.logger.error(f"Scale up callback failed: {e}")
        else:
            # Default scaling action
            self.current_instances = new_instances
            self.target_instances = new_instances
            self.last_scale_time = current_time
    
    def _scale_down(self, current_time: float) -> None:
        """Scale down the component."""
        if self.current_instances <= max(rule.min_instances for rule in self.scaling_rules):
            self.logger.warning("Cannot scale down: already at minimum instances")
            return
        
        new_instances = max(
            self.current_instances - 1,
            max(rule.min_instances for rule in self.scaling_rules)
        )
        
        self.logger.info(
            f"Scaling {self.component_name} DOWN: {self.current_instances} -> {new_instances}"
        )
        
        if self.scale_down_callback:
            try:
                self.scale_down_callback(new_instances)
                self.current_instances = new_instances
                self.target_instances = new_instances
                self.last_scale_time = current_time
            except Exception as e:
                self.logger.error(f"Scale down callback failed: {e}")
        else:
            # Default scaling action
            self.current_instances = new_instances
            self.target_instances = new_instances
            self.last_scale_time = current_time
    
    def set_scaling_callbacks(
        self, 
        scale_up_callback: Callable[[int], None],
        scale_down_callback: Callable[[int], None]
    ) -> None:
        """Set callbacks for scaling actions."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
    
    def add_rule(self, rule: ScalingRule) -> None:
        """Add custom scaling rule."""
        self.scaling_rules.append(rule)
        self.logger.info(f"Added scaling rule: {rule.name}")
    
    def enable_scaling(self, enabled: bool = True) -> None:
        """Enable or disable auto-scaling."""
        self.scaling_enabled = enabled
        self.logger.info(f"Auto-scaling {'enabled' if enabled else 'disabled'}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "component": self.component_name,
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "scaling_enabled": self.scaling_enabled,
            "last_scale_time": self.last_scale_time,
            "metrics_history_size": len(self.metrics_history),
            "active_rules": [r.name for r in self.scaling_rules if r.enabled],
            "recent_metrics": list(self.metrics_history)[-3:] if self.metrics_history else []
        }
    
    def force_scale(self, target_instances: int) -> None:
        """Force scaling to specific number of instances."""
        with self.scale_lock:
            if target_instances < 1:
                raise ValueError("Target instances must be >= 1")
            
            max_instances = max(rule.max_instances for rule in self.scaling_rules)
            if target_instances > max_instances:
                raise ValueError(f"Target instances {target_instances} exceeds maximum {max_instances}")
            
            self.logger.info(
                f"Force scaling {self.component_name}: {self.current_instances} -> {target_instances}"
            )
            
            if target_instances > self.current_instances and self.scale_up_callback:
                self.scale_up_callback(target_instances)
            elif target_instances < self.current_instances and self.scale_down_callback:
                self.scale_down_callback(target_instances)
            
            self.current_instances = target_instances
            self.target_instances = target_instances
            self.last_scale_time = time.time()


class GlobalScalingManager:
    """Manages scaling across all PWMK components."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.scalers: Dict[str, AdaptiveScaler] = {}
        self.resource_limits = {
            "max_total_gpu_instances": 8,
            "max_total_cpu_instances": 20,
            "memory_limit_gb": 32
        }
    
    def register_component(self, component_name: str) -> AdaptiveScaler:
        """Register a component for auto-scaling."""
        if component_name in self.scalers:
            return self.scalers[component_name]
        
        scaler = AdaptiveScaler(component_name)
        self.scalers[component_name] = scaler
        
        self.logger.info(f"Registered component for scaling: {component_name}")
        return scaler
    
    def get_scaler(self, component_name: str) -> Optional[AdaptiveScaler]:
        """Get scaler for a component."""
        return self.scalers.get(component_name)
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global scaling status."""
        total_instances = sum(scaler.current_instances for scaler in self.scalers.values())
        
        return {
            "total_instances": total_instances,
            "resource_limits": self.resource_limits,
            "components": {
                name: scaler.get_status() 
                for name, scaler in self.scalers.items()
            }
        }
    
    def emergency_scale_down(self) -> None:
        """Emergency scale down all components."""
        self.logger.warning("Emergency scale down initiated")
        
        for name, scaler in self.scalers.items():
            try:
                min_instances = max(rule.min_instances for rule in scaler.scaling_rules)
                if scaler.current_instances > min_instances:
                    scaler.force_scale(min_instances)
            except Exception as e:
                self.logger.error(f"Emergency scale down failed for {name}: {e}")


# Global scaling manager
_scaling_manager = None


def get_scaling_manager() -> GlobalScalingManager:
    """Get global scaling manager."""
    global _scaling_manager
    if _scaling_manager is None:
        _scaling_manager = GlobalScalingManager()
    return _scaling_manager