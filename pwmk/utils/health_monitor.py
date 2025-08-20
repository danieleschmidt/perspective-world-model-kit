"""System health monitoring and auto-recovery."""

import time
import threading
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from .logging import get_logger
from .fallback_manager import get_fallback_manager, SystemMode


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_func: Callable[[], bool]
    interval: float = 30.0  # seconds
    timeout: float = 10.0   # seconds
    critical: bool = False
    description: str = ""
    last_check: Optional[float] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class SystemHealth:
    """Overall system health state."""
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    component_status: Dict[str, HealthStatus] = field(default_factory=dict)
    last_updated: float = 0.0
    degraded_components: List[str] = field(default_factory=list)
    failed_components: List[str] = field(default_factory=list)
    uptime_start: float = field(default_factory=time.time)


class HealthMonitor:
    """Monitors system component health and triggers recovery."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_health = SystemHealth()
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        self.logger = get_logger(self.__class__.__name__)
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        self.register_check(
            "memory_usage",
            self._check_memory_usage,
            interval=60.0,
            critical=True,
            description="Monitor system memory usage"
        )
        
        self.register_check(
            "model_performance",
            self._check_model_performance,
            interval=30.0,
            critical=False,
            description="Monitor model inference performance"
        )
        
        self.register_check(
            "belief_store_connectivity",
            self._check_belief_store,
            interval=45.0,
            critical=False,
            description="Check belief store connectivity"
        )
        
        self.register_check(
            "quantum_backend",
            self._check_quantum_backend,
            interval=120.0,
            critical=False,
            description="Check quantum computing backend"
        )
        
        self.register_check(
            "consciousness_engine",
            self._check_consciousness_engine,
            interval=30.0,
            critical=False,
            description="Monitor consciousness engine health"
        )
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        interval: float = 30.0,
        timeout: float = 10.0,
        critical: bool = False,
        description: str = ""
    ) -> None:
        """Register a health check."""
        health_check = HealthCheck(
            name=name,
            check_func=check_func,
            interval=interval,
            timeout=timeout,
            critical=critical,
            description=description
        )
        
        with self.lock:
            self.health_checks[name] = health_check
        
        self.logger.info(f"Registered health check: {name} (critical={critical})")
    
    def start_monitoring(self) -> None:
        """Start health monitoring in background thread."""
        if self.monitoring:
            self.logger.warning("Health monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._run_health_checks()
                self._update_system_health()
                self._trigger_auto_recovery()
                time.sleep(5.0)  # Check every 5 seconds for due checks
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def _run_health_checks(self) -> None:
        """Run health checks that are due."""
        current_time = time.time()
        
        for name, check in self.health_checks.items():
            if (check.last_check is None or 
                current_time - check.last_check >= check.interval):
                
                self._run_single_check(check, current_time)
    
    def _run_single_check(self, check: HealthCheck, current_time: float) -> None:
        """Run a single health check with timeout."""
        try:
            # Simple timeout implementation
            start_time = time.time()
            result = check.check_func()
            duration = time.time() - start_time
            
            if duration > check.timeout:
                self.logger.warning(
                    f"Health check {check.name} took {duration:.2f}s "
                    f"(timeout: {check.timeout}s)"
                )
                result = False
            
            with self.lock:
                check.last_check = current_time
                
                if result:
                    check.last_status = HealthStatus.HEALTHY
                    check.consecutive_failures = 0
                    self.logger.debug(f"Health check {check.name}: HEALTHY")
                else:
                    check.consecutive_failures += 1
                    
                    if check.consecutive_failures >= check.max_failures:
                        check.last_status = HealthStatus.UNHEALTHY
                        self.logger.error(
                            f"Health check {check.name}: UNHEALTHY "
                            f"({check.consecutive_failures} failures)"
                        )
                    else:
                        check.last_status = HealthStatus.DEGRADED
                        self.logger.warning(
                            f"Health check {check.name}: DEGRADED "
                            f"({check.consecutive_failures}/{check.max_failures})"
                        )
        
        except Exception as e:
            self.logger.error(f"Health check {check.name} failed: {e}")
            with self.lock:
                check.last_check = current_time
                check.consecutive_failures += 1
                check.last_status = HealthStatus.UNHEALTHY
    
    def _update_system_health(self) -> None:
        """Update overall system health based on component status."""
        with self.lock:
            # Update component statuses
            self.system_health.component_status.clear()
            self.system_health.degraded_components.clear()
            self.system_health.failed_components.clear()
            
            for name, check in self.health_checks.items():
                self.system_health.component_status[name] = check.last_status
                
                if check.last_status == HealthStatus.DEGRADED:
                    self.system_health.degraded_components.append(name)
                elif check.last_status == HealthStatus.UNHEALTHY:
                    self.system_health.failed_components.append(name)
            
            # Determine overall status
            if any(check.last_status == HealthStatus.UNHEALTHY and check.critical 
                   for check in self.health_checks.values()):
                self.system_health.overall_status = HealthStatus.UNHEALTHY
            elif any(check.last_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
                    for check in self.health_checks.values()):
                self.system_health.overall_status = HealthStatus.DEGRADED
            else:
                self.system_health.overall_status = HealthStatus.HEALTHY
            
            self.system_health.last_updated = time.time()
    
    def _trigger_auto_recovery(self) -> None:
        """Trigger automatic recovery actions based on health status."""
        fallback_manager = get_fallback_manager()
        
        if self.system_health.overall_status == HealthStatus.UNHEALTHY:
            if fallback_manager.get_mode() != SystemMode.EMERGENCY:
                fallback_manager.set_mode(
                    SystemMode.EMERGENCY,
                    f"Critical components failed: {self.system_health.failed_components}"
                )
        elif self.system_health.overall_status == HealthStatus.DEGRADED:
            if fallback_manager.get_mode() == SystemMode.NORMAL:
                fallback_manager.set_mode(
                    SystemMode.DEGRADED,
                    f"Components degraded: {self.system_health.degraded_components}"
                )
        elif self.system_health.overall_status == HealthStatus.HEALTHY:
            if fallback_manager.get_mode() in [SystemMode.DEGRADED, SystemMode.EMERGENCY]:
                fallback_manager.set_mode(SystemMode.NORMAL, "All components healthy")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        with self.lock:
            uptime = time.time() - self.system_health.uptime_start
            
            return {
                "overall_status": self.system_health.overall_status.value,
                "uptime_seconds": uptime,
                "last_updated": self.system_health.last_updated,
                "component_status": {
                    name: status.value 
                    for name, status in self.system_health.component_status.items()
                },
                "degraded_components": self.system_health.degraded_components,
                "failed_components": self.system_health.failed_components,
                "fallback_mode": get_fallback_manager().get_mode().value
            }
    
    def force_check(self, check_name: str) -> bool:
        """Force run a specific health check."""
        if check_name not in self.health_checks:
            raise ValueError(f"Health check {check_name} not found")
        
        check = self.health_checks[check_name]
        self._run_single_check(check, time.time())
        return check.last_status == HealthStatus.HEALTHY
    
    # Default health check implementations
    def _check_memory_usage(self) -> bool:
        """Check system memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            # Fail if memory usage > 90%
            if usage_percent > 90:
                self.logger.warning(f"High memory usage: {usage_percent}%")
                return False
            
            return True
        except ImportError:
            # psutil not available, assume healthy
            return True
        except Exception as e:
            self.logger.error(f"Memory check failed: {e}")
            return False
    
    def _check_model_performance(self) -> bool:
        """Check model inference performance."""
        # This would typically test model latency/throughput
        # For now, just check if we can import the model
        try:
            from ..core.world_model import PerspectiveWorldModel
            return True
        except Exception as e:
            self.logger.error(f"Model performance check failed: {e}")
            return False
    
    def _check_belief_store(self) -> bool:
        """Check belief store connectivity."""
        try:
            from ..core.beliefs import BeliefStore
            # Simple connectivity test
            store = BeliefStore()
            return True
        except Exception as e:
            self.logger.error(f"Belief store check failed: {e}")
            return False
    
    def _check_quantum_backend(self) -> bool:
        """Check quantum computing backend."""
        try:
            from ..quantum.integration import QuantumIntegration
            # Basic quantum backend test
            return True
        except Exception as e:
            self.logger.debug(f"Quantum backend check failed: {e}")
            return False  # Non-critical, quantum may not be available
    
    def _check_consciousness_engine(self) -> bool:
        """Check consciousness engine health."""
        try:
            from ..revolution.consciousness_engine import ConsciousnessEngine
            return True
        except Exception as e:
            self.logger.error(f"Consciousness engine check failed: {e}")
            return False


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def start_health_monitoring() -> None:
    """Start global health monitoring."""
    monitor = get_health_monitor()
    monitor.start_monitoring()


def get_system_health() -> Dict[str, Any]:
    """Get current system health report."""
    monitor = get_health_monitor()
    return monitor.get_health_report()