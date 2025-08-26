"""Advanced health monitoring and alerting system."""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import json
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil
import statistics

from ..utils.logging import get_logger
from ..utils.monitoring import get_metrics_collector


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    timestamp: float
    unit: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass 
class Alert:
    """Health alert."""
    id: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_name: str
    metric_value: float
    threshold: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[float] = None


class AdvancedHealthMonitor:
    """
    Advanced health monitoring system with alerting and persistence.
    
    Provides comprehensive health metrics collection, trend analysis,
    and configurable alerting for PWMK systems.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.metrics_collector = get_metrics_collector()
        
        # Database setup
        self.db_path = db_path or "monitoring_health.db"
        self._init_database()
        
        # Monitoring state
        self.collectors: Dict[str, Callable] = {}
        self.metric_history: Dict[str, List[HealthMetric]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Configuration
        self.collection_interval = 30.0  # seconds
        self.history_retention = 86400  # 24 hours in seconds
        self.max_metrics_per_type = 1000
        
        # Threading
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._collector_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="health")
        
        # Register default collectors
        self._register_default_collectors()
        
        # Start monitoring
        self.start_monitoring()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metric storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS health_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp REAL NOT NULL,
                        unit TEXT DEFAULT '',
                        tags TEXT DEFAULT '{}',
                        INDEX(name, timestamp)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        threshold REAL,
                        resolved INTEGER DEFAULT 0,
                        resolved_at REAL,
                        INDEX(timestamp, resolved)
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _register_default_collectors(self) -> None:
        """Register default health metric collectors."""
        self.register_collector("system.cpu_percent", self._collect_cpu_usage)
        self.register_collector("system.memory_percent", self._collect_memory_usage)
        self.register_collector("system.disk_percent", self._collect_disk_usage)
        self.register_collector("system.load_average", self._collect_load_average)
        self.register_collector("process.memory_rss", self._collect_process_memory)
        self.register_collector("process.cpu_percent", self._collect_process_cpu)
        self.register_collector("pytorch.gpu_memory", self._collect_gpu_memory)
    
    def register_collector(self, metric_name: str, collector_func: Callable) -> None:
        """Register a metric collector function."""
        with self._lock:
            self.collectors[metric_name] = collector_func
            self.metric_history[metric_name] = []
            
        self.logger.info(f"Registered health collector: {metric_name}")
    
    def register_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler function."""
        self.alert_handlers.append(handler)
        self.logger.info(f"Registered alert handler: {handler.__name__}")
    
    def start_monitoring(self) -> None:
        """Start health monitoring threads."""
        if self._collector_thread is None or not self._collector_thread.is_alive():
            self._collector_thread = threading.Thread(
                target=self._collection_loop,
                daemon=True,
                name="HealthCollector"
            )
            self._collector_thread.start()
            
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="HealthCleanup"
            )
            self._cleanup_thread.start()
            
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._shutdown_event.set()
        
        if self._collector_thread:
            self._collector_thread.join(timeout=5.0)
            
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
            
        self._executor.shutdown(wait=True)
        self.logger.info("Health monitoring stopped")
    
    def _collection_loop(self) -> None:
        """Main collection loop."""
        while not self._shutdown_event.is_set():
            try:
                self._collect_all_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Metric collection failed: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_all_metrics(self) -> None:
        """Collect all registered metrics."""
        current_time = time.time()
        
        for metric_name, collector in self.collectors.items():
            try:
                # Run collector in thread pool to prevent blocking
                future = self._executor.submit(collector)
                metric = future.result(timeout=10.0)  # 10 second timeout
                
                if metric:
                    metric.timestamp = current_time
                    metric.name = metric_name
                    
                    self._store_metric(metric)
                    self._check_thresholds(metric)
                    
            except Exception as e:
                self.logger.warning(f"Collection failed for {metric_name}: {e}")
    
    def _store_metric(self, metric: HealthMetric) -> None:
        """Store metric in memory and database."""
        with self._lock:
            # Store in memory
            if metric.name not in self.metric_history:
                self.metric_history[metric.name] = []
                
            history = self.metric_history[metric.name]
            history.append(metric)
            
            # Limit memory usage
            if len(history) > self.max_metrics_per_type:
                history.pop(0)
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO health_metrics 
                    (name, value, timestamp, unit, tags)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.value,
                    metric.timestamp,
                    metric.unit,
                    json.dumps(metric.tags)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.warning(f"Database storage failed for {metric.name}: {e}")
    
    def _check_thresholds(self, metric: HealthMetric) -> None:
        """Check metric against thresholds and generate alerts."""
        alert_generated = False
        
        # Check critical threshold
        if metric.threshold_critical is not None and metric.value >= metric.threshold_critical:
            alert = Alert(
                id=f"{metric.name}_critical_{int(metric.timestamp)}",
                severity=AlertSeverity.CRITICAL,
                message=f"Critical threshold exceeded for {metric.name}: {metric.value} {metric.unit}",
                timestamp=metric.timestamp,
                metric_name=metric.name,
                metric_value=metric.value,
                threshold=metric.threshold_critical
            )
            self._handle_alert(alert)
            alert_generated = True
            
        # Check warning threshold (only if no critical alert)
        elif metric.threshold_warning is not None and metric.value >= metric.threshold_warning:
            alert = Alert(
                id=f"{metric.name}_warning_{int(metric.timestamp)}",
                severity=AlertSeverity.WARNING,
                message=f"Warning threshold exceeded for {metric.name}: {metric.value} {metric.unit}",
                timestamp=metric.timestamp,
                metric_name=metric.name,
                metric_value=metric.value,
                threshold=metric.threshold_warning
            )
            self._handle_alert(alert)
            alert_generated = True
        
        # Check for recovery from previous alerts
        if not alert_generated:
            self._check_alert_recovery(metric)
    
    def _handle_alert(self, alert: Alert) -> None:
        """Handle a new alert."""
        with self._lock:
            # Check if similar alert already exists
            existing_alert_id = None
            for aid, existing in self.active_alerts.items():
                if (existing.metric_name == alert.metric_name and 
                    existing.severity == alert.severity and
                    not existing.resolved):
                    existing_alert_id = aid
                    break
            
            if existing_alert_id:
                # Update existing alert
                self.active_alerts[existing_alert_id].timestamp = alert.timestamp
                self.active_alerts[existing_alert_id].metric_value = alert.metric_value
            else:
                # New alert
                self.active_alerts[alert.id] = alert
                self.logger.warning(f"Health alert: {alert.message}")
                
                # Store in database
                self._store_alert(alert)
                
                # Notify handlers
                for handler in self.alert_handlers:
                    try:
                        self._executor.submit(handler, alert)
                    except Exception as e:
                        self.logger.error(f"Alert handler failed: {e}")
    
    def _check_alert_recovery(self, metric: HealthMetric) -> None:
        """Check if any active alerts can be resolved."""
        with self._lock:
            to_resolve = []
            
            for alert_id, alert in self.active_alerts.items():
                if (alert.metric_name == metric.name and 
                    not alert.resolved and
                    alert.threshold is not None and
                    metric.value < alert.threshold * 0.9):  # 10% buffer to prevent flapping
                    to_resolve.append(alert_id)
            
            for alert_id in to_resolve:
                self._resolve_alert(alert_id, metric.timestamp)
    
    def _resolve_alert(self, alert_id: str, timestamp: float) -> None:
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = timestamp
                
                self.logger.info(f"Alert resolved: {alert.message}")
                
                # Update in database
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            UPDATE alerts 
                            SET resolved = 1, resolved_at = ?
                            WHERE id = ?
                        """, (timestamp, alert_id))
                        conn.commit()
                except Exception as e:
                    self.logger.warning(f"Database update failed for alert {alert_id}: {e}")
    
    def _store_alert(self, alert: Alert) -> None:
        """Store alert in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts
                    (id, severity, message, timestamp, metric_name, metric_value, threshold)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id,
                    alert.severity.value,
                    alert.message,
                    alert.timestamp,
                    alert.metric_name,
                    alert.metric_value,
                    alert.threshold
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.warning(f"Alert storage failed: {e}")
    
    def _cleanup_loop(self) -> None:
        """Cleanup old data periodically."""
        while not self._shutdown_event.is_set():
            try:
                self._cleanup_old_data()
                # Cleanup every hour
                self._shutdown_event.wait(3600)
            except Exception as e:
                self.logger.error(f"Cleanup failed: {e}")
                self._shutdown_event.wait(3600)
    
    def _cleanup_old_data(self) -> None:
        """Remove old metrics and alerts from storage."""
        cutoff_time = time.time() - self.history_retention
        
        # Cleanup database
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Remove old metrics
                result = conn.execute("""
                    DELETE FROM health_metrics 
                    WHERE timestamp < ?
                """, (cutoff_time,))
                
                metrics_deleted = result.rowcount
                
                # Remove old resolved alerts
                result = conn.execute("""
                    DELETE FROM alerts 
                    WHERE resolved = 1 AND resolved_at < ?
                """, (cutoff_time,))
                
                alerts_deleted = result.rowcount
                conn.commit()
                
                if metrics_deleted > 0 or alerts_deleted > 0:
                    self.logger.info(f"Cleaned up {metrics_deleted} metrics and {alerts_deleted} alerts")
                    
        except Exception as e:
            self.logger.warning(f"Database cleanup failed: {e}")
        
        # Cleanup memory
        with self._lock:
            for metric_name, history in self.metric_history.items():
                self.metric_history[metric_name] = [
                    m for m in history if m.timestamp >= cutoff_time
                ]
    
    # Default metric collectors
    def _collect_cpu_usage(self) -> Optional[HealthMetric]:
        """Collect CPU usage percentage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return HealthMetric(
                name="system.cpu_percent",
                value=cpu_percent,
                timestamp=time.time(),
                unit="%",
                threshold_warning=75.0,
                threshold_critical=90.0
            )
        except Exception as e:
            self.logger.warning(f"CPU collection failed: {e}")
            return None
    
    def _collect_memory_usage(self) -> Optional[HealthMetric]:
        """Collect memory usage percentage."""
        try:
            memory = psutil.virtual_memory()
            return HealthMetric(
                name="system.memory_percent",
                value=memory.percent,
                timestamp=time.time(),
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0
            )
        except Exception as e:
            self.logger.warning(f"Memory collection failed: {e}")
            return None
    
    def _collect_disk_usage(self) -> Optional[HealthMetric]:
        """Collect disk usage percentage."""
        try:
            disk = psutil.disk_usage('/')
            return HealthMetric(
                name="system.disk_percent",
                value=disk.percent,
                timestamp=time.time(),
                unit="%",
                threshold_warning=85.0,
                threshold_critical=95.0
            )
        except Exception as e:
            self.logger.warning(f"Disk collection failed: {e}")
            return None
    
    def _collect_load_average(self) -> Optional[HealthMetric]:
        """Collect system load average."""
        try:
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            cpu_count = psutil.cpu_count()
            load_percent = (load_avg / cpu_count) * 100
            
            return HealthMetric(
                name="system.load_average",
                value=load_percent,
                timestamp=time.time(),
                unit="%",
                threshold_warning=80.0,
                threshold_critical=100.0
            )
        except Exception as e:
            self.logger.warning(f"Load average collection failed: {e}")
            return None
    
    def _collect_process_memory(self) -> Optional[HealthMetric]:
        """Collect current process memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            return HealthMetric(
                name="process.memory_rss",
                value=memory_mb,
                timestamp=time.time(),
                unit="MB",
                threshold_warning=1000.0,  # 1GB
                threshold_critical=2000.0  # 2GB
            )
        except Exception as e:
            self.logger.warning(f"Process memory collection failed: {e}")
            return None
    
    def _collect_process_cpu(self) -> Optional[HealthMetric]:
        """Collect current process CPU usage."""
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            
            return HealthMetric(
                name="process.cpu_percent",
                value=cpu_percent,
                timestamp=time.time(),
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0
            )
        except Exception as e:
            self.logger.warning(f"Process CPU collection failed: {e}")
            return None
    
    def _collect_gpu_memory(self) -> Optional[HealthMetric]:
        """Collect GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    percent = (allocated / total) * 100
                    
                    return HealthMetric(
                        name="pytorch.gpu_memory",
                        value=percent,
                        timestamp=time.time(),
                        unit="%",
                        threshold_warning=85.0,
                        threshold_critical=95.0,
                        tags={"gpu_id": "0", "allocated_gb": f"{allocated:.2f}", "total_gb": f"{total:.2f}"}
                    )
        except Exception as e:
            self.logger.warning(f"GPU memory collection failed: {e}")
            return None
    
    def get_current_metrics(self) -> Dict[str, HealthMetric]:
        """Get the most recent metrics for all collectors."""
        current_metrics = {}
        
        with self._lock:
            for metric_name, history in self.metric_history.items():
                if history:
                    current_metrics[metric_name] = history[-1]
                    
        return current_metrics
    
    def get_metric_history(
        self, 
        metric_name: str, 
        hours: int = 1
    ) -> List[HealthMetric]:
        """Get metric history for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            if metric_name in self.metric_history:
                return [
                    m for m in self.metric_history[metric_name]
                    if m.timestamp >= cutoff_time
                ]
        return []
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        current_metrics = self.get_current_metrics()
        active_alerts = self.get_active_alerts()
        
        # Calculate overall health score
        health_score = 100.0
        critical_alerts = sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL)
        warning_alerts = sum(1 for a in active_alerts if a.severity == AlertSeverity.WARNING)
        
        health_score -= (critical_alerts * 30)  # -30 points per critical alert
        health_score -= (warning_alerts * 10)   # -10 points per warning alert
        health_score = max(0.0, health_score)
        
        return {
            "overall_health_score": health_score,
            "status": "critical" if health_score < 30 else "warning" if health_score < 70 else "healthy",
            "active_alerts": len(active_alerts),
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "monitored_metrics": len(current_metrics),
            "last_collection": max([m.timestamp for m in current_metrics.values()] or [0]),
            "current_metrics": {
                name: {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp
                }
                for name, metric in current_metrics.items()
            }
        }