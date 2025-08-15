"""
Comprehensive Monitoring System - Advanced AI System Monitoring

Provides comprehensive monitoring, alerting, and observability for
consciousness, quantum, emergent intelligence, and research systems.
"""

import time
import logging
import json
import threading
import queue
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import torch
from pathlib import Path
import sqlite3
import pickle
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MonitoringMetric:
    """Individual monitoring metric."""
    name: str
    value: Union[float, int, str, bool]
    timestamp: float
    source: str
    unit: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: str  # info, warning, error, critical
    title: str
    description: str
    source: str
    timestamp: float
    triggered_by: str
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    resolved: bool = False
    resolution_time: Optional[float] = None
    actions_taken: List[str] = field(default_factory=list)


class MetricsCollector:
    """Collect and aggregate metrics from various system components."""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.metric_aggregates = defaultdict(list)
        self.collection_active = False
        self.collection_thread = None
        self.collection_interval = 1.0  # seconds
        
        # Metric sources
        self.registered_sources = {}
        self.source_callbacks = {}
        
        # Database for persistence
        self.db_path = Path("monitoring_metrics.db")
        self._init_database()
        
    def _init_database(self):
        """Initialize metrics database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        source TEXT NOT NULL,
                        unit TEXT,
                        tags TEXT,
                        metadata TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                    ON metrics(timestamp)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_source 
                    ON metrics(source)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_name 
                    ON metrics(name)
                """)
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
    
    def register_source(self, source_name: str, callback: Callable[[], Dict[str, Any]]):
        """Register a metrics source with callback function."""
        self.registered_sources[source_name] = callback
        logger.info(f"Registered metrics source: {source_name}")
    
    def collect_metric(self, metric: MonitoringMetric):
        """Collect a single metric."""
        try:
            self.metrics_buffer.append(metric)
            self.metric_aggregates[metric.name].append(metric.value)
            
            # Keep only recent values for aggregation
            if len(self.metric_aggregates[metric.name]) > 1000:
                self.metric_aggregates[metric.name] = self.metric_aggregates[metric.name][-1000:]
            
            # Persist to database periodically
            if len(self.metrics_buffer) % 100 == 0:
                self._persist_metrics()
                
        except Exception as e:
            logger.error(f"Failed to collect metric {metric.name}: {e}")
    
    def start_collection(self):
        """Start automatic metrics collection."""
        if not self.collection_active:
            self.collection_active = True
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                name="MetricsCollector",
                daemon=True
            )
            self.collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self.collection_active = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        # Persist remaining metrics
        self._persist_metrics()
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.collection_active:
            try:
                current_time = time.time()
                
                # Collect from registered sources
                for source_name, callback in self.registered_sources.items():
                    try:
                        metrics_data = callback()
                        
                        for metric_name, value in metrics_data.items():
                            metric = MonitoringMetric(
                                name=metric_name,
                                value=value,
                                timestamp=current_time,
                                source=source_name
                            )
                            self.collect_metric(metric)
                            
                    except Exception as e:
                        logger.error(f"Failed to collect from source {source_name}: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                time.sleep(self.collection_interval)
    
    def _persist_metrics(self):
        """Persist metrics to database."""
        if not self.metrics_buffer:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                metrics_to_persist = list(self.metrics_buffer)
                
                for metric in metrics_to_persist:
                    conn.execute("""
                        INSERT INTO metrics (name, value, timestamp, source, unit, tags, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metric.name,
                        json.dumps(metric.value, default=str),
                        metric.timestamp,
                        metric.source,
                        metric.unit,
                        json.dumps(metric.tags),
                        json.dumps(metric.metadata, default=str)
                    ))
                
                logger.debug(f"Persisted {len(metrics_to_persist)} metrics to database")
                
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    def get_metrics(self, name: str = None, source: str = None, 
                   start_time: float = None, end_time: float = None,
                   limit: int = 1000) -> List[MonitoringMetric]:
        """Retrieve metrics from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM metrics WHERE 1=1"
                params = []
                
                if name:
                    query += " AND name = ?"
                    params.append(name)
                
                if source:
                    query += " AND source = ?"
                    params.append(source)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                metrics = []
                for row in rows:
                    metric = MonitoringMetric(
                        name=row[1],
                        value=json.loads(row[2]),
                        timestamp=row[3],
                        source=row[4],
                        unit=row[5],
                        tags=json.loads(row[6]) if row[6] else {},
                        metadata=json.loads(row[7]) if row[7] else {}
                    )
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {e}")
            return []
    
    def get_metric_statistics(self, name: str, hours: int = 24) -> Dict[str, float]:
        """Get statistical summary of metric over time period."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        metrics = self.get_metrics(name=name, start_time=start_time, end_time=end_time)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics if isinstance(m.value, (int, float))]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }


class AlertManager:
    """Manage system alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_rules = {}
        self.notification_channels = []
        
        # Alert escalation
        self.escalation_rules = {
            'critical': {'escalate_after': 300, 'notification_interval': 60},  # 5 min, notify every 1 min
            'error': {'escalate_after': 600, 'notification_interval': 300},   # 10 min, notify every 5 min
            'warning': {'escalate_after': 1800, 'notification_interval': 900}, # 30 min, notify every 15 min
            'info': {'escalate_after': 3600, 'notification_interval': 1800}    # 1 hour, notify every 30 min
        }
        
        # Database for persistence
        self.db_path = Path("monitoring_alerts.db")
        self._init_database()
        
        # Start alert processing
        self.processing_active = False
        self.processing_thread = None
        
    def _init_database(self):
        """Initialize alerts database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        severity TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        source TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        triggered_by TEXT NOT NULL,
                        threshold_value REAL,
                        current_value REAL,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolution_time REAL,
                        actions_taken TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                    ON alerts(timestamp)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_severity 
                    ON alerts(severity)
                """)
        except Exception as e:
            logger.error(f"Failed to initialize alerts database: {e}")
    
    def add_alert_rule(self, rule_name: str, metric_name: str, 
                      condition: str, threshold: float, severity: str = 'warning'):
        """Add alert rule for metric monitoring."""
        self.alert_rules[rule_name] = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq', 'ne'
            'threshold': threshold,
            'severity': severity,
            'enabled': True
        }
        logger.info(f"Added alert rule: {rule_name}")
    
    def trigger_alert(self, alert: Alert):
        """Trigger a new alert."""
        try:
            alert_id = alert.alert_id
            
            # Check if similar alert is already active
            existing_alert = self.active_alerts.get(alert_id)
            if existing_alert and not existing_alert.resolved:
                # Update existing alert
                existing_alert.current_value = alert.current_value
                existing_alert.timestamp = alert.timestamp
                logger.debug(f"Updated existing alert: {alert_id}")
                return
            
            # Add new alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Persist to database
            self._persist_alert(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            logger.warning(f"Alert triggered: {alert.severity.upper()} - {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = time.time()
            alert.actions_taken.append(f"Resolved: {resolution_notes}")
            
            # Update database
            self._update_alert(alert)
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    def start_processing(self):
        """Start alert processing."""
        if not self.processing_active:
            self.processing_active = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="AlertManager",
                daemon=True
            )
            self.processing_thread.start()
            logger.info("Alert processing started")
    
    def stop_processing(self):
        """Stop alert processing."""
        self.processing_active = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        logger.info("Alert processing stopped")
    
    def _processing_loop(self):
        """Main alert processing loop."""
        while self.processing_active:
            try:
                # Evaluate alert rules
                self._evaluate_alert_rules()
                
                # Process escalations
                self._process_escalations()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Alert processing loop error: {e}")
                time.sleep(30)
    
    def _evaluate_alert_rules(self):
        """Evaluate all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            if not rule['enabled']:
                continue
                
            try:
                # Get recent metric values
                recent_metrics = self.metrics_collector.get_metrics(
                    name=rule['metric_name'],
                    start_time=time.time() - 300,  # Last 5 minutes
                    limit=10
                )
                
                if not recent_metrics:
                    continue
                
                # Get latest value
                latest_metric = recent_metrics[0]
                current_value = latest_metric.value
                
                if not isinstance(current_value, (int, float)):
                    continue
                
                # Check condition
                threshold = rule['threshold']
                condition_met = False
                
                if rule['condition'] == 'gt' and current_value > threshold:
                    condition_met = True
                elif rule['condition'] == 'lt' and current_value < threshold:
                    condition_met = True
                elif rule['condition'] == 'eq' and current_value == threshold:
                    condition_met = True
                elif rule['condition'] == 'ne' and current_value != threshold:
                    condition_met = True
                
                if condition_met:
                    # Create alert
                    alert_id = f"{rule_name}_{latest_metric.source}"
                    
                    alert = Alert(
                        alert_id=alert_id,
                        severity=rule['severity'],
                        title=f"Metric Alert: {rule['metric_name']}",
                        description=f"Metric {rule['metric_name']} value {current_value} {rule['condition']} {threshold}",
                        source=latest_metric.source,
                        timestamp=latest_metric.timestamp,
                        triggered_by=rule_name,
                        threshold_value=threshold,
                        current_value=current_value
                    )
                    
                    self.trigger_alert(alert)
                
            except Exception as e:
                logger.error(f"Failed to evaluate alert rule {rule_name}: {e}")
    
    def _process_escalations(self):
        """Process alert escalations."""
        current_time = time.time()
        
        for alert_id, alert in self.active_alerts.items():
            if alert.resolved:
                continue
                
            severity = alert.severity
            if severity not in self.escalation_rules:
                continue
                
            escalation = self.escalation_rules[severity]
            time_since_trigger = current_time - alert.timestamp
            
            # Check if alert should be escalated
            if time_since_trigger > escalation['escalate_after']:
                self._escalate_alert(alert)
    
    def _escalate_alert(self, alert: Alert):
        """Escalate an alert to higher severity."""
        severity_levels = ['info', 'warning', 'error', 'critical']
        current_index = severity_levels.index(alert.severity)
        
        if current_index < len(severity_levels) - 1:
            new_severity = severity_levels[current_index + 1]
            alert.severity = new_severity
            alert.actions_taken.append(f"Escalated to {new_severity}")
            
            self._update_alert(alert)
            self._send_notifications(alert)
            
            logger.warning(f"Alert escalated: {alert.alert_id} -> {new_severity}")
    
    def _persist_alert(self, alert: Alert):
        """Persist alert to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, severity, title, description, source, timestamp, 
                     triggered_by, threshold_value, current_value, resolved, 
                     resolution_time, actions_taken)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id, alert.severity, alert.title, alert.description,
                    alert.source, alert.timestamp, alert.triggered_by,
                    alert.threshold_value, alert.current_value, alert.resolved,
                    alert.resolution_time, json.dumps(alert.actions_taken)
                ))
        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")
    
    def _update_alert(self, alert: Alert):
        """Update existing alert in database."""
        self._persist_alert(alert)
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels."""
        for channel in self.notification_channels:
            try:
                channel.send_notification(alert)
            except Exception as e:
                logger.error(f"Failed to send notification through {channel}: {e}")
    
    def get_active_alerts(self, severity: str = None) -> List[Alert]:
        """Get currently active alerts."""
        alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for time period."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.timestamp >= start_time
        ]
        
        if not recent_alerts:
            return {'total': 0}
        
        by_severity = defaultdict(int)
        by_source = defaultdict(int)
        resolved_count = 0
        
        for alert in recent_alerts:
            by_severity[alert.severity] += 1
            by_source[alert.source] += 1
            if alert.resolved:
                resolved_count += 1
        
        return {
            'total': len(recent_alerts),
            'by_severity': dict(by_severity),
            'by_source': dict(by_source),
            'resolved': resolved_count,
            'resolution_rate': resolved_count / len(recent_alerts),
            'active': len(recent_alerts) - resolved_count
        }


class SystemHealthMonitor:
    """Monitor overall system health and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        # Health check components
        self.health_checks = {}
        self.health_status = {}
        self.last_health_check = 0
        self.health_check_interval = 60  # seconds
        
        # System performance tracking
        self.performance_baselines = {}
        self.performance_trends = defaultdict(list)
        
        # Register built-in health checks
        self._register_builtin_health_checks()
        
    def _register_builtin_health_checks(self):
        """Register built-in health checks."""
        self.register_health_check(
            'consciousness_engine',
            self._check_consciousness_health,
            critical=True
        )
        
        self.register_health_check(
            'quantum_processor',
            self._check_quantum_health,
            critical=True
        )
        
        self.register_health_check(
            'emergent_system',
            self._check_emergent_health,
            critical=True
        )
        
        self.register_health_check(
            'metrics_collection',
            self._check_metrics_health,
            critical=False
        )
        
        self.register_health_check(
            'alert_processing',
            self._check_alert_health,
            critical=False
        )
    
    def register_health_check(self, name: str, check_function: Callable[[], bool], 
                            critical: bool = False):
        """Register a health check function."""
        self.health_checks[name] = {
            'function': check_function,
            'critical': critical,
            'last_check': 0,
            'last_result': None,
            'failure_count': 0
        }
        logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        current_time = time.time()
        results = {}
        overall_healthy = True
        critical_failures = []
        
        for name, check_config in self.health_checks.items():
            try:
                # Run health check
                check_result = check_config['function']()
                
                # Update check status
                check_config['last_check'] = current_time
                check_config['last_result'] = check_result
                
                if check_result:
                    check_config['failure_count'] = 0
                    status = 'healthy'
                else:
                    check_config['failure_count'] += 1
                    status = 'unhealthy'
                    
                    if check_config['critical']:
                        critical_failures.append(name)
                        overall_healthy = False
                
                results[name] = {
                    'status': status,
                    'critical': check_config['critical'],
                    'failure_count': check_config['failure_count'],
                    'last_check': current_time
                }
                
                # Create alert for unhealthy critical components
                if not check_result and check_config['critical']:
                    alert = Alert(
                        alert_id=f"health_check_{name}",
                        severity='critical',
                        title=f"Critical Health Check Failed: {name}",
                        description=f"Critical component {name} failed health check",
                        source='health_monitor',
                        timestamp=current_time,
                        triggered_by='health_check'
                    )
                    self.alert_manager.trigger_alert(alert)
                
            except Exception as e:
                logger.error(f"Health check {name} failed with exception: {e}")
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'critical': check_config['critical'],
                    'last_check': current_time
                }
                
                if check_config['critical']:
                    overall_healthy = False
                    critical_failures.append(name)
        
        self.health_status = {
            'overall_healthy': overall_healthy,
            'critical_failures': critical_failures,
            'last_check': current_time,
            'checks': results
        }
        
        self.last_health_check = current_time
        return self.health_status
    
    def _check_consciousness_health(self) -> bool:
        """Check consciousness engine health."""
        try:
            # Check if consciousness metrics are being collected
            recent_metrics = self.metrics_collector.get_metrics(
                source='consciousness_engine',
                start_time=time.time() - 300,  # Last 5 minutes
                limit=5
            )
            
            if not recent_metrics:
                return False
            
            # Check for healthy consciousness levels
            consciousness_metrics = [
                m for m in recent_metrics 
                if m.name in ['consciousness_level', 'overall_score']
            ]
            
            return len(consciousness_metrics) > 0
            
        except Exception as e:
            logger.error(f"Consciousness health check failed: {e}")
            return False
    
    def _check_quantum_health(self) -> bool:
        """Check quantum processor health."""
        try:
            recent_metrics = self.metrics_collector.get_metrics(
                source='quantum_processor',
                start_time=time.time() - 300,
                limit=5
            )
            
            if not recent_metrics:
                return False
            
            # Check for quantum advantage metrics
            quantum_metrics = [
                m for m in recent_metrics 
                if m.name in ['quantum_advantage', 'circuit_depth']
            ]
            
            return len(quantum_metrics) > 0
            
        except Exception as e:
            logger.error(f"Quantum health check failed: {e}")
            return False
    
    def _check_emergent_health(self) -> bool:
        """Check emergent intelligence system health."""
        try:
            recent_metrics = self.metrics_collector.get_metrics(
                source='emergent_system',
                start_time=time.time() - 300,
                limit=5
            )
            
            if not recent_metrics:
                return False
            
            # Check for emergence metrics
            emergence_metrics = [
                m for m in recent_metrics 
                if m.name in ['intelligence_score', 'emergence_level']
            ]
            
            return len(emergence_metrics) > 0
            
        except Exception as e:
            logger.error(f"Emergent health check failed: {e}")
            return False
    
    def _check_metrics_health(self) -> bool:
        """Check metrics collection health."""
        try:
            return (self.metrics_collector.collection_active and 
                   len(self.metrics_collector.metrics_buffer) > 0)
        except Exception:
            return False
    
    def _check_alert_health(self) -> bool:
        """Check alert processing health."""
        try:
            return self.alert_manager.processing_active
        except Exception:
            return False
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview."""
        current_time = time.time()
        
        # Run health checks if needed
        if current_time - self.last_health_check > self.health_check_interval:
            self.run_health_checks()
        
        # Get metrics summary
        metrics_summary = self._get_metrics_summary()
        
        # Get alerts summary
        alerts_summary = self.alert_manager.get_alert_statistics()
        
        # Get performance summary
        performance_summary = self._get_performance_summary()
        
        return {
            'timestamp': current_time,
            'health_status': self.health_status,
            'metrics_summary': metrics_summary,
            'alerts_summary': alerts_summary,
            'performance_summary': performance_summary,
            'system_uptime': self._calculate_uptime()
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics collection summary."""
        total_metrics = len(self.metrics_collector.metrics_buffer)
        
        sources = defaultdict(int)
        for metric in self.metrics_collector.metrics_buffer:
            sources[metric.source] += 1
        
        return {
            'total_metrics_collected': total_metrics,
            'metrics_by_source': dict(sources),
            'collection_active': self.metrics_collector.collection_active,
            'buffer_utilization': total_metrics / self.metrics_collector.buffer_size
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        # Simplified performance metrics
        return {
            'consciousness_performance': self._get_component_performance('consciousness_engine'),
            'quantum_performance': self._get_component_performance('quantum_processor'),
            'emergent_performance': self._get_component_performance('emergent_system'),
            'overall_efficiency': 0.85  # Simulated overall efficiency
        }
    
    def _get_component_performance(self, component: str) -> Dict[str, float]:
        """Get performance metrics for specific component."""
        # Get recent performance metrics
        recent_metrics = self.metrics_collector.get_metrics(
            source=component,
            start_time=time.time() - 3600,  # Last hour
            limit=100
        )
        
        if not recent_metrics:
            return {'status': 'no_data'}
        
        # Calculate performance indicators
        performance_metrics = [
            m for m in recent_metrics 
            if isinstance(m.value, (int, float)) and m.name.endswith(('_score', '_level', '_advantage'))
        ]
        
        if not performance_metrics:
            return {'status': 'no_performance_data'}
        
        values = [m.value for m in performance_metrics]
        
        return {
            'average_performance': np.mean(values),
            'peak_performance': np.max(values),
            'performance_stability': 1.0 - np.std(values),  # Higher stability = lower variance
            'data_points': len(values)
        }
    
    def _calculate_uptime(self) -> float:
        """Calculate system uptime in hours."""
        # Simplified uptime calculation
        if hasattr(self, 'start_time'):
            return (time.time() - self.start_time) / 3600
        return 0.0


class ComprehensiveMonitoringSystem:
    """Complete monitoring system integrating all components."""
    
    def __init__(self):
        # Core components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_monitor = SystemHealthMonitor(self.metrics_collector, self.alert_manager)
        
        # System state
        self.monitoring_active = False
        self.start_time = time.time()
        
        # Register default alert rules
        self._setup_default_alert_rules()
        
        logger.info("Comprehensive monitoring system initialized")
    
    def _setup_default_alert_rules(self):
        """Setup default monitoring alert rules."""
        # Consciousness alerts
        self.alert_manager.add_alert_rule(
            'consciousness_level_high',
            'consciousness_level',
            'gt', 5.0, 'warning'
        )
        
        self.alert_manager.add_alert_rule(
            'consciousness_score_low',
            'overall_score',
            'lt', 0.3, 'warning'
        )
        
        # Quantum alerts
        self.alert_manager.add_alert_rule(
            'quantum_advantage_low',
            'quantum_advantage',
            'lt', 0.1, 'warning'
        )
        
        # Emergent intelligence alerts
        self.alert_manager.add_alert_rule(
            'intelligence_score_low',
            'intelligence_score',
            'lt', 0.5, 'warning'
        )
        
        # System performance alerts
        self.alert_manager.add_alert_rule(
            'system_efficiency_low',
            'overall_efficiency',
            'lt', 0.7, 'error'
        )
    
    def start_monitoring(self):
        """Start comprehensive monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.start_time = time.time()
            
            # Start all components
            self.metrics_collector.start_collection()
            self.alert_manager.start_processing()
            
            logger.info("🔍 Comprehensive monitoring started")
    
    def stop_monitoring(self):
        """Stop comprehensive monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            
            # Stop all components
            self.metrics_collector.stop_collection()
            self.alert_manager.stop_processing()
            
            logger.info("Comprehensive monitoring stopped")
    
    def register_component_source(self, component_name: str, metrics_callback: Callable[[], Dict[str, Any]]):
        """Register a component as metrics source."""
        self.metrics_collector.register_source(component_name, metrics_callback)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            'system_overview': self.health_monitor.get_system_overview(),
            'active_alerts': self.alert_manager.get_active_alerts(),
            'critical_alerts': self.alert_manager.get_active_alerts('critical'),
            'alert_statistics': self.alert_manager.get_alert_statistics(),
            'monitoring_status': {
                'active': self.monitoring_active,
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'metrics_collected': len(self.metrics_collector.metrics_buffer),
                'alert_rules': len(self.alert_manager.alert_rules)
            }
        }
    
    def create_monitoring_report(self, hours: int = 24) -> Dict[str, Any]:
        """Create comprehensive monitoring report."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        # Get metrics for all major components
        consciousness_stats = self.metrics_collector.get_metric_statistics('consciousness_level', hours)
        quantum_stats = self.metrics_collector.get_metric_statistics('quantum_advantage', hours)
        intelligence_stats = self.metrics_collector.get_metric_statistics('intelligence_score', hours)
        
        # Get alert statistics
        alert_stats = self.alert_manager.get_alert_statistics(hours)
        
        # Get system health
        health_status = self.health_monitor.get_system_overview()
        
        report = {
            'report_period': {
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': hours
            },
            'executive_summary': {
                'overall_health': health_status['health_status']['overall_healthy'],
                'total_alerts': alert_stats.get('total', 0),
                'critical_alerts': alert_stats.get('by_severity', {}).get('critical', 0),
                'system_efficiency': health_status['performance_summary']['overall_efficiency']
            },
            'component_performance': {
                'consciousness_engine': consciousness_stats,
                'quantum_processor': quantum_stats,
                'emergent_intelligence': intelligence_stats
            },
            'alert_analysis': alert_stats,
            'health_analysis': health_status['health_status'],
            'recommendations': self._generate_recommendations(health_status, alert_stats)
        }
        
        return report
    
    def _generate_recommendations(self, health_status: Dict[str, Any], 
                                alert_stats: Dict[str, Any]) -> List[str]:
        """Generate operational recommendations based on monitoring data."""
        recommendations = []
        
        # Health-based recommendations
        if not health_status['health_status']['overall_healthy']:
            recommendations.append("Investigate critical component failures immediately")
        
        if health_status['health_status'].get('critical_failures'):
            for failure in health_status['health_status']['critical_failures']:
                recommendations.append(f"Restore {failure} component functionality")
        
        # Alert-based recommendations
        if alert_stats.get('total', 0) > 50:
            recommendations.append("High alert volume detected - review alert thresholds")
        
        if alert_stats.get('resolution_rate', 1.0) < 0.8:
            recommendations.append("Improve alert resolution processes - many alerts unresolved")
        
        # Performance-based recommendations
        if health_status['performance_summary']['overall_efficiency'] < 0.8:
            recommendations.append("System efficiency below optimal - consider performance tuning")
        
        # Default recommendations if none generated
        if not recommendations:
            recommendations.append("System operating normally - continue monitoring")
        
        return recommendations


# Factory function
def create_comprehensive_monitoring() -> ComprehensiveMonitoringSystem:
    """Create configured comprehensive monitoring system."""
    return ComprehensiveMonitoringSystem()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create monitoring system
    monitoring = create_comprehensive_monitoring()
    
    # Start monitoring
    monitoring.start_monitoring()
    
    # Simulate some metrics
    def consciousness_metrics():
        return {
            'consciousness_level': np.random.uniform(2.0, 4.0),
            'overall_score': np.random.uniform(0.6, 0.9),
            'subjective_richness': np.random.uniform(0.5, 0.8)
        }
    
    def quantum_metrics():
        return {
            'quantum_advantage': np.random.uniform(0.1, 0.5),
            'circuit_depth': np.random.randint(5, 15),
            'speedup_factor': np.random.uniform(1.2, 2.5)
        }
    
    # Register component sources
    monitoring.register_component_source('consciousness_engine', consciousness_metrics)
    monitoring.register_component_source('quantum_processor', quantum_metrics)
    
    # Let it run for a short time
    print("Monitoring for 10 seconds...")
    time.sleep(10)
    
    # Get dashboard
    dashboard = monitoring.get_monitoring_dashboard()
    print(f"Dashboard: {json.dumps(dashboard, indent=2, default=str)}")
    
    # Generate report
    report = monitoring.create_monitoring_report(hours=1)
    print(f"Report: {json.dumps(report, indent=2, default=str)}")
    
    # Stop monitoring
    monitoring.stop_monitoring()