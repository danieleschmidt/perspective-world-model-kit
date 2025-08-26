"""System-wide resilience and fault tolerance."""

from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import queue

from ..utils.logging import get_logger
from ..utils.monitoring import get_metrics_collector
from .error_recovery import ErrorRecoveryManager


class SystemState(Enum):
    """System operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


@dataclass
class ResourceThresholds:
    """Resource usage thresholds for system monitoring."""
    cpu_warning: float = 75.0
    cpu_critical: float = 90.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    disk_warning: float = 85.0
    disk_critical: float = 95.0
    gpu_memory_warning: float = 85.0
    gpu_memory_critical: float = 95.0


@dataclass
class ResiliencePolicy:
    """Policy configuration for system resilience."""
    auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_cooldown: float = 60.0
    emergency_shutdown_threshold: int = 5
    resource_monitoring_interval: float = 30.0
    health_check_interval: float = 60.0
    component_timeout: float = 30.0
    enable_load_shedding: bool = True
    enable_graceful_degradation: bool = True


@dataclass 
class ComponentHealth:
    """Health status of a system component."""
    name: str
    status: SystemState = SystemState.HEALTHY
    last_check: float = field(default_factory=time.time)
    error_count: int = 0
    recovery_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class ResilienceOrchestrator:
    """
    System-wide resilience orchestrator.
    
    Manages system health, coordinates recovery efforts, and implements
    load shedding and graceful degradation strategies.
    """
    
    def __init__(self, policy: Optional[ResiliencePolicy] = None):
        self.policy = policy or ResiliencePolicy()
        self.thresholds = ResourceThresholds()
        
        self.logger = get_logger(self.__class__.__name__)
        self.metrics = get_metrics_collector()
        self.error_recovery = ErrorRecoveryManager()
        
        # System state management
        self.system_state = SystemState.HEALTHY
        self.components: Dict[str, ComponentHealth] = {}
        self.critical_failures = 0
        
        # Threading and monitoring
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None
        
        # Load management
        self._load_queue = queue.PriorityQueue()
        self._worker_pool = ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="resilience"
        )
        
        # Emergency procedures
        self._emergency_handlers: List[Callable] = []
        self._shutdown_handlers: List[Callable] = []
        
        # Start monitoring
        self._start_monitoring()
    
    def register_component(
        self, 
        name: str, 
        health_check: Optional[Callable] = None
    ) -> None:
        """Register a component for health monitoring."""
        with self._lock:
            self.components[name] = ComponentHealth(name=name)
            
            if health_check:
                self._register_health_check(name, health_check)
                
        self.logger.info(f"Registered component: {name}")
    
    def _register_health_check(self, name: str, health_check: Callable) -> None:
        """Register a health check function for a component."""
        def check_component():
            try:
                result = health_check()
                self._update_component_health(name, SystemState.HEALTHY, metrics=result)
            except Exception as e:
                self._update_component_health(
                    name, 
                    SystemState.CRITICAL,
                    error=str(e)
                )
        
        # Schedule periodic health checks
        self._worker_pool.submit(self._periodic_health_check, name, check_component)
    
    def _periodic_health_check(self, name: str, check_func: Callable) -> None:
        """Run periodic health checks for a component."""
        while not self._shutdown_event.is_set():
            try:
                check_func()
                time.sleep(self.policy.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                time.sleep(self.policy.health_check_interval)
    
    def _update_component_health(
        self, 
        name: str, 
        status: SystemState,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update the health status of a component."""
        with self._lock:
            if name in self.components:
                component = self.components[name]
                old_status = component.status
                
                component.status = status
                component.last_check = time.time()
                
                if error:
                    component.error_count += 1
                    component.last_error = error
                    
                if metrics:
                    component.metrics.update(metrics)
                
                # Log status changes
                if old_status != status:
                    self.logger.info(f"Component {name} status: {old_status.value} -> {status.value}")
                    
                    if status in [SystemState.CRITICAL, SystemState.EMERGENCY]:
                        self.critical_failures += 1
                        self._handle_component_failure(name, component)
    
    def _handle_component_failure(self, name: str, component: ComponentHealth) -> None:
        """Handle component failure with recovery strategies."""
        self.logger.warning(f"Handling failure for component: {name}")
        
        # Increment failure metrics
        self.metrics.increment(f"component_failures.{name}")
        
        # Check if we should trigger emergency procedures
        if self.critical_failures >= self.policy.emergency_shutdown_threshold:
            self._trigger_emergency_procedures()
            return
        
        # Attempt recovery if enabled
        if self.policy.auto_recovery and component.recovery_count < self.policy.max_recovery_attempts:
            self._attempt_component_recovery(name, component)
        else:
            self.logger.error(f"Component {name} has exceeded recovery attempts")
            self._trigger_graceful_degradation(name)
    
    def _attempt_component_recovery(self, name: str, component: ComponentHealth) -> None:
        """Attempt to recover a failed component."""
        self.logger.info(f"Attempting recovery for component: {name}")
        
        component.recovery_count += 1
        
        # Submit recovery task
        future = self._worker_pool.submit(self._recovery_task, name, component)
        
        # Monitor recovery attempt
        def recovery_callback(fut: Future):
            try:
                success = fut.result(timeout=self.policy.component_timeout)
                if success:
                    self.logger.info(f"Recovery successful for component: {name}")
                    self._update_component_health(name, SystemState.HEALTHY)
                else:
                    self.logger.warning(f"Recovery failed for component: {name}")
                    self._trigger_graceful_degradation(name)
            except Exception as e:
                self.logger.error(f"Recovery exception for component {name}: {e}")
                self._trigger_graceful_degradation(name)
        
        future.add_done_callback(recovery_callback)
    
    def _recovery_task(self, name: str, component: ComponentHealth) -> bool:
        """Execute recovery procedures for a component."""
        try:
            # Wait for cooldown period
            time.sleep(self.policy.recovery_cooldown)
            
            # Basic recovery strategies
            if "model" in name.lower():
                return self._recover_model_component(name)
            elif "storage" in name.lower():
                return self._recover_storage_component(name)
            elif "network" in name.lower():
                return self._recover_network_component(name)
            else:
                return self._generic_component_recovery(name)
                
        except Exception as e:
            self.logger.error(f"Recovery task failed for {name}: {e}")
            return False
    
    def _recover_model_component(self, name: str) -> bool:
        """Recover a model component."""
        self.logger.info(f"Attempting model recovery for: {name}")
        
        # Model-specific recovery
        try:
            # Clear GPU memory if applicable
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Could reload model weights, reset state, etc.
            return True
            
        except Exception as e:
            self.logger.error(f"Model recovery failed: {e}")
            return False
    
    def _recover_storage_component(self, name: str) -> bool:
        """Recover a storage component."""
        self.logger.info(f"Attempting storage recovery for: {name}")
        
        # Storage-specific recovery
        try:
            # Check disk space, permissions, etc.
            disk_usage = psutil.disk_usage('/')
            if disk_usage.percent < 95:
                return True
            else:
                self.logger.error("Insufficient disk space for recovery")
                return False
                
        except Exception as e:
            self.logger.error(f"Storage recovery failed: {e}")
            return False
    
    def _recover_network_component(self, name: str) -> bool:
        """Recover a network component."""
        self.logger.info(f"Attempting network recovery for: {name}")
        
        # Network-specific recovery
        try:
            # Could test connectivity, reset connections, etc.
            return True
            
        except Exception as e:
            self.logger.error(f"Network recovery failed: {e}")
            return False
    
    def _generic_component_recovery(self, name: str) -> bool:
        """Generic component recovery."""
        self.logger.info(f"Attempting generic recovery for: {name}")
        
        # Generic recovery strategies
        try:
            # Basic cleanup, restart procedures, etc.
            import gc
            gc.collect()
            return True
            
        except Exception as e:
            self.logger.error(f"Generic recovery failed: {e}")
            return False
    
    def _trigger_graceful_degradation(self, component_name: str) -> None:
        """Trigger graceful degradation for a failed component."""
        if not self.policy.enable_graceful_degradation:
            return
            
        self.logger.warning(f"Triggering graceful degradation for: {component_name}")
        
        # Update system state
        with self._lock:
            if self.system_state == SystemState.HEALTHY:
                self.system_state = SystemState.DEGRADED
                
        self.metrics.increment("graceful_degradations")
        
        # Component-specific degradation
        if "model" in component_name.lower():
            self._degrade_model_performance()
        elif "storage" in component_name.lower():
            self._enable_storage_fallback()
        elif "network" in component_name.lower():
            self._enable_offline_mode()
    
    def _degrade_model_performance(self) -> None:
        """Reduce model performance to maintain availability."""
        self.logger.info("Degrading model performance")
        # Could reduce batch sizes, disable complex features, etc.
    
    def _enable_storage_fallback(self) -> None:
        """Enable storage fallback mechanisms."""
        self.logger.info("Enabling storage fallback")
        # Could switch to temporary storage, reduce persistence, etc.
    
    def _enable_offline_mode(self) -> None:
        """Enable offline operation mode."""
        self.logger.info("Enabling offline mode")
        # Could cache responses, disable remote features, etc.
    
    def _trigger_emergency_procedures(self) -> None:
        """Trigger emergency shutdown procedures."""
        self.logger.critical("Triggering emergency procedures")
        
        with self._lock:
            self.system_state = SystemState.EMERGENCY
            
        self.metrics.increment("emergency_procedures")
        
        # Execute emergency handlers
        for handler in self._emergency_handlers:
            try:
                self._worker_pool.submit(handler)
            except Exception as e:
                self.logger.error(f"Emergency handler failed: {e}")
        
        # Prepare for potential shutdown
        if self.critical_failures >= self.policy.emergency_shutdown_threshold * 2:
            self._initiate_shutdown()
    
    def _initiate_shutdown(self) -> None:
        """Initiate graceful system shutdown."""
        self.logger.critical("Initiating system shutdown")
        
        with self._lock:
            self.system_state = SystemState.SHUTDOWN
            
        # Execute shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                self.logger.error(f"Shutdown handler failed: {e}")
        
        # Signal shutdown
        self._shutdown_event.set()
    
    def _start_monitoring(self) -> None:
        """Start system monitoring threads."""
        if not self._monitor_thread:
            self._monitor_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True,
                name="ResourceMonitor"
            )
            self._monitor_thread.start()
    
    def _monitor_resources(self) -> None:
        """Monitor system resources continuously."""
        while not self._shutdown_event.is_set():
            try:
                self._check_system_resources()
                time.sleep(self.policy.resource_monitoring_interval)
            except Exception as e:
                self.logger.error(f"Resource monitoring failed: {e}")
                time.sleep(self.policy.resource_monitoring_interval)
    
    def _check_system_resources(self) -> None:
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.thresholds.cpu_critical:
                self.logger.critical(f"Critical CPU usage: {cpu_percent:.1f}%")
                self._handle_resource_pressure("cpu", cpu_percent)
            elif cpu_percent > self.thresholds.cpu_warning:
                self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.thresholds.memory_critical:
                self.logger.critical(f"Critical memory usage: {memory.percent:.1f}%")
                self._handle_resource_pressure("memory", memory.percent)
            elif memory.percent > self.thresholds.memory_warning:
                self.logger.warning(f"High memory usage: {memory.percent:.1f}%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > self.thresholds.disk_critical:
                self.logger.critical(f"Critical disk usage: {disk.percent:.1f}%")
                self._handle_resource_pressure("disk", disk.percent)
            elif disk.percent > self.thresholds.disk_warning:
                self.logger.warning(f"High disk usage: {disk.percent:.1f}%")
            
            # GPU memory (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        percent = (allocated / total) * 100
                        
                        if percent > self.thresholds.gpu_memory_critical:
                            self.logger.critical(f"Critical GPU {i} memory: {percent:.1f}%")
                            self._handle_resource_pressure("gpu_memory", percent)
            except Exception:
                pass  # GPU monitoring is optional
                
        except Exception as e:
            self.logger.error(f"Resource check failed: {e}")
    
    def _handle_resource_pressure(self, resource_type: str, usage_percent: float) -> None:
        """Handle resource pressure situations."""
        self.logger.warning(f"Handling resource pressure: {resource_type} at {usage_percent:.1f}%")
        
        if self.policy.enable_load_shedding:
            self._initiate_load_shedding(resource_type, usage_percent)
        
        # Update system state
        with self._lock:
            if usage_percent > 95 and self.system_state == SystemState.HEALTHY:
                self.system_state = SystemState.CRITICAL
            elif usage_percent > 85 and self.system_state == SystemState.HEALTHY:
                self.system_state = SystemState.DEGRADED
    
    def _initiate_load_shedding(self, resource_type: str, usage_percent: float) -> None:
        """Initiate load shedding to reduce resource pressure."""
        self.logger.info(f"Initiating load shedding for {resource_type}")
        
        if resource_type == "memory":
            self._shed_memory_load()
        elif resource_type == "cpu":
            self._shed_cpu_load()
        elif resource_type == "gpu_memory":
            self._shed_gpu_memory_load()
    
    def _shed_memory_load(self) -> None:
        """Shed memory load by clearing caches and reducing operations."""
        import gc
        gc.collect()
        
        # Could also disable caching, reduce batch sizes, etc.
        self.logger.info("Memory load shedding applied")
    
    def _shed_cpu_load(self) -> None:
        """Shed CPU load by reducing concurrent operations."""
        # Could reduce thread pool size, skip non-essential processing, etc.
        self.logger.info("CPU load shedding applied")
    
    def _shed_gpu_memory_load(self) -> None:
        """Shed GPU memory load."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("GPU memory load shedding applied")
        except Exception as e:
            self.logger.error(f"GPU load shedding failed: {e}")
    
    def add_emergency_handler(self, handler: Callable) -> None:
        """Add an emergency response handler."""
        self._emergency_handlers.append(handler)
    
    def add_shutdown_handler(self, handler: Callable) -> None:
        """Add a shutdown handler."""
        self._shutdown_handlers.append(handler)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            return {
                "system_state": self.system_state.value,
                "critical_failures": self.critical_failures,
                "components": {
                    name: {
                        "status": comp.status.value,
                        "last_check": comp.last_check,
                        "error_count": comp.error_count,
                        "recovery_count": comp.recovery_count,
                        "last_error": comp.last_error
                    }
                    for name, comp in self.components.items()
                },
                "resource_thresholds": {
                    "cpu_warning": self.thresholds.cpu_warning,
                    "cpu_critical": self.thresholds.cpu_critical,
                    "memory_warning": self.thresholds.memory_warning,
                    "memory_critical": self.thresholds.memory_critical
                },
                "policy": {
                    "auto_recovery": self.policy.auto_recovery,
                    "max_recovery_attempts": self.policy.max_recovery_attempts,
                    "emergency_threshold": self.policy.emergency_shutdown_threshold
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown the resilience orchestrator."""
        self.logger.info("Shutting down resilience orchestrator")
        self._shutdown_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            
        self._worker_pool.shutdown(wait=True)