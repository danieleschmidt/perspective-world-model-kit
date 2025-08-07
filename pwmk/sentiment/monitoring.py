"""
Monitoring and observability for sentiment analysis operations.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import json
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentEvent:
    """Individual sentiment analysis event."""
    timestamp: float
    agent_id: int
    event_type: str  # 'analysis', 'belief_update', 'error'
    text: str
    sentiment_scores: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SentimentMonitor:
    """
    Real-time monitoring for sentiment analysis operations.
    Tracks performance, errors, and system health.
    """
    
    def __init__(
        self,
        max_events: int = 10000,
        metric_window_minutes: int = 15
    ):
        self.max_events = max_events
        self.metric_window = timedelta(minutes=metric_window_minutes)
        
        # Thread-safe event storage
        self._lock = threading.Lock()
        self.events: deque = deque(maxlen=max_events)
        
        # Real-time metrics
        self.metrics = {
            "total_analyses": 0,
            "total_errors": 0,
            "avg_processing_time": 0.0,
            "analyses_per_minute": 0.0,
            "error_rate": 0.0
        }
        
        # Agent-specific metrics
        self.agent_metrics = defaultdict(lambda: {
            "total_analyses": 0,
            "total_errors": 0,
            "avg_sentiment_scores": {"negative": 0.0, "neutral": 0.0, "positive": 0.0},
            "sentiment_variance": 0.0
        })
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.recent_errors = deque(maxlen=100)
        
        logger.info(f"Sentiment monitoring initialized with {max_events} max events, {metric_window_minutes}min window")
        
    def record_analysis(
        self,
        agent_id: int,
        text: str,
        sentiment_scores: Dict[str, float],
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a successful sentiment analysis event."""
        event = SentimentEvent(
            timestamp=time.time(),
            agent_id=agent_id,
            event_type="analysis",
            text=text,
            sentiment_scores=sentiment_scores,
            processing_time=processing_time,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.events.append(event)
            self._update_metrics(event)
            
        logger.debug(f"Recorded analysis for agent {agent_id}, processing time: {processing_time:.3f}s")
        
    def record_error(
        self,
        agent_id: int,
        text: str,
        error: str,
        processing_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a sentiment analysis error event."""
        event = SentimentEvent(
            timestamp=time.time(),
            agent_id=agent_id,
            event_type="error",
            text=text,
            error=error,
            processing_time=processing_time,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.events.append(event)
            self._update_error_metrics(event)
            
        logger.warning(f"Recorded error for agent {agent_id}: {error}")
        
    def record_belief_update(
        self,
        agent_id: int,
        belief_query: str,
        processing_time: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a belief store update event."""
        event = SentimentEvent(
            timestamp=time.time(),
            agent_id=agent_id,
            event_type="belief_update",
            text=belief_query,
            processing_time=processing_time,
            error=None if success else "Belief update failed",
            metadata=metadata or {}
        )
        
        with self._lock:
            self.events.append(event)
            
        logger.debug(f"Recorded belief update for agent {agent_id}, success: {success}")
        
    def _update_metrics(self, event: SentimentEvent) -> None:
        """Update real-time metrics based on new event."""
        # Global metrics
        self.metrics["total_analyses"] += 1
        
        if event.processing_time:
            # Update running average for processing time
            total_analyses = self.metrics["total_analyses"]
            current_avg = self.metrics["avg_processing_time"]
            self.metrics["avg_processing_time"] = (
                (current_avg * (total_analyses - 1) + event.processing_time) / total_analyses
            )
            
        # Agent-specific metrics
        agent_metrics = self.agent_metrics[event.agent_id]
        agent_metrics["total_analyses"] += 1
        
        if event.sentiment_scores:
            # Update running average sentiment scores
            for sentiment, score in event.sentiment_scores.items():
                current_avg = agent_metrics["avg_sentiment_scores"][sentiment]
                count = agent_metrics["total_analyses"]
                agent_metrics["avg_sentiment_scores"][sentiment] = (
                    (current_avg * (count - 1) + score) / count
                )
                
        # Calculate recent analysis rate
        self._update_analysis_rate()
        
    def _update_error_metrics(self, event: SentimentEvent) -> None:
        """Update error-related metrics."""
        self.metrics["total_errors"] += 1
        self.agent_metrics[event.agent_id]["total_errors"] += 1
        
        # Track error types
        if event.error:
            self.error_counts[event.error] += 1
            self.recent_errors.append({
                "timestamp": event.timestamp,
                "agent_id": event.agent_id,
                "error": event.error,
                "text_preview": event.text[:100] + "..." if len(event.text) > 100 else event.text
            })
            
        # Update error rate
        total_operations = self.metrics["total_analyses"] + self.metrics["total_errors"]
        if total_operations > 0:
            self.metrics["error_rate"] = self.metrics["total_errors"] / total_operations
            
    def _update_analysis_rate(self) -> None:
        """Update analyses per minute metric."""
        current_time = time.time()
        cutoff_time = current_time - self.metric_window.total_seconds()
        
        recent_analyses = sum(
            1 for event in self.events
            if event.timestamp >= cutoff_time and event.event_type == "analysis"
        )
        
        window_minutes = self.metric_window.total_seconds() / 60
        self.metrics["analyses_per_minute"] = recent_analyses / window_minutes
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        with self._lock:
            self._update_analysis_rate()
            return {
                "global_metrics": dict(self.metrics),
                "agent_metrics": dict(self.agent_metrics),
                "error_counts": dict(self.error_counts),
                "total_events": len(self.events),
                "last_update": time.time()
            }
            
    def get_agent_performance(self, agent_id: int) -> Dict[str, Any]:
        """Get performance metrics for specific agent."""
        agent_metrics = self.agent_metrics.get(agent_id, {})
        
        # Calculate agent-specific error rate
        total_ops = agent_metrics.get("total_analyses", 0) + agent_metrics.get("total_errors", 0)
        agent_error_rate = agent_metrics.get("total_errors", 0) / total_ops if total_ops > 0 else 0.0
        
        return {
            "agent_id": agent_id,
            "total_analyses": agent_metrics.get("total_analyses", 0),
            "total_errors": agent_metrics.get("total_errors", 0),
            "error_rate": agent_error_rate,
            "avg_sentiment_scores": agent_metrics.get("avg_sentiment_scores", {}),
            "relative_activity": self._calculate_relative_activity(agent_id)
        }
        
    def _calculate_relative_activity(self, agent_id: int) -> float:
        """Calculate agent's activity relative to group average."""
        agent_analyses = self.agent_metrics[agent_id].get("total_analyses", 0)
        
        if not self.agent_metrics:
            return 0.0
            
        total_analyses = sum(
            metrics.get("total_analyses", 0)
            for metrics in self.agent_metrics.values()
        )
        
        avg_analyses = total_analyses / len(self.agent_metrics)
        return agent_analyses / avg_analyses if avg_analyses > 0 else 0.0
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        current_metrics = self.get_current_metrics()
        
        # Determine health status based on error rate and processing time
        error_rate = current_metrics["global_metrics"]["error_rate"]
        avg_processing_time = current_metrics["global_metrics"]["avg_processing_time"]
        
        if error_rate > 0.1:  # More than 10% error rate
            health_status = "unhealthy"
            health_score = max(0, 1.0 - error_rate)
        elif error_rate > 0.05:  # More than 5% error rate
            health_status = "degraded"
            health_score = 0.8
        elif avg_processing_time > 2.0:  # Slow processing
            health_status = "slow"
            health_score = 0.7
        else:
            health_status = "healthy"
            health_score = 1.0
            
        return {
            "status": health_status,
            "score": health_score,
            "error_rate": error_rate,
            "avg_processing_time": avg_processing_time,
            "analyses_per_minute": current_metrics["global_metrics"]["analyses_per_minute"],
            "total_events": current_metrics["total_events"],
            "recent_errors": list(self.recent_errors)[-5:] if self.recent_errors else []
        }
        
    def get_trend_analysis(self, hours: int = 1) -> Dict[str, Any]:
        """Analyze trends over specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            recent_events = [
                event for event in self.events
                if event.timestamp >= cutoff_time
            ]
            
        if not recent_events:
            return {"message": "No events in specified time period"}
            
        # Analyze trends
        analysis_events = [e for e in recent_events if e.event_type == "analysis"]
        error_events = [e for e in recent_events if e.event_type == "error"]
        
        # Processing time trend
        if analysis_events:
            processing_times = [e.processing_time for e in analysis_events if e.processing_time]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        else:
            avg_processing_time = 0.0
            
        # Sentiment distribution
        sentiment_totals = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        sentiment_count = 0
        
        for event in analysis_events:
            if event.sentiment_scores:
                sentiment_count += 1
                for sentiment, score in event.sentiment_scores.items():
                    sentiment_totals[sentiment] += score
                    
        if sentiment_count > 0:
            avg_sentiment_distribution = {
                sentiment: total / sentiment_count
                for sentiment, total in sentiment_totals.items()
            }
        else:
            avg_sentiment_distribution = {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
            
        return {
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "analysis_events": len(analysis_events),
            "error_events": len(error_events),
            "error_rate": len(error_events) / len(recent_events) if recent_events else 0.0,
            "avg_processing_time": avg_processing_time,
            "avg_sentiment_distribution": avg_sentiment_distribution,
            "events_per_hour": len(recent_events) / hours
        }
        
    def export_metrics(self, filepath: str) -> None:
        """Export current metrics to JSON file."""
        try:
            metrics_data = {
                "export_timestamp": time.time(),
                "export_datetime": datetime.now().isoformat(),
                "current_metrics": self.get_current_metrics(),
                "health_status": self.get_health_status(),
                "trend_analysis": self.get_trend_analysis()
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            
    def reset_metrics(self) -> None:
        """Reset all metrics and event history."""
        with self._lock:
            self.events.clear()
            self.metrics = {
                "total_analyses": 0,
                "total_errors": 0,
                "avg_processing_time": 0.0,
                "analyses_per_minute": 0.0,
                "error_rate": 0.0
            }
            self.agent_metrics.clear()
            self.error_counts.clear()
            self.recent_errors.clear()
            
        logger.info("All metrics and event history reset")