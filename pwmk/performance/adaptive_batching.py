"""Adaptive batching system for optimal throughput and latency."""

from typing import Dict, List, Any, Optional, Callable, Tuple, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import statistics
import heapq

from ..utils.logging import get_logger
from ..utils.monitoring import get_metrics_collector

T = TypeVar('T')


class BatchingStrategy(Enum):
    """Batching strategies for different workload patterns."""
    FIXED_SIZE = "fixed_size"
    ADAPTIVE_SIZE = "adaptive_size"  
    LATENCY_AWARE = "latency_aware"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    RESOURCE_AWARE = "resource_aware"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE_SIZE
    min_batch_size: int = 1
    max_batch_size: int = 64
    max_wait_time: float = 100.0  # milliseconds
    target_latency: float = 200.0  # milliseconds
    target_throughput: float = 100.0  # items per second
    adaptive_adjustment_rate: float = 0.1
    resource_threshold: float = 0.8


@dataclass
class BatchItem(Generic[T]):
    """Individual item in a batch queue."""
    data: T
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    future: Optional[Future] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance."""
    batch_size: int
    processing_time: float
    wait_time: float
    throughput: float
    latency: float
    cpu_usage: float
    memory_usage: float
    success_rate: float


class AdaptiveBatchProcessor(Generic[T]):
    """
    Adaptive batch processor that optimizes batch sizes dynamically.
    
    Automatically adjusts batch sizes based on latency, throughput,
    resource usage, and workload characteristics.
    """
    
    def __init__(
        self,
        processor_func: Callable[[List[T]], List[Any]],
        config: Optional[BatchConfig] = None
    ):
        self.processor_func = processor_func
        self.config = config or BatchConfig()
        
        self.logger = get_logger(self.__class__.__name__)
        self.metrics = get_metrics_collector()
        
        # Queue management
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._shutdown_event = threading.Event()
        
        # Performance tracking
        self._metrics_history: List[BatchMetrics] = []
        self._current_batch_size = self.config.min_batch_size
        self._last_adjustment = time.time()
        self._adjustment_cooldown = 5.0  # seconds
        
        # Threading
        self._lock = threading.RLock()
        self._processor_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="adaptive_batch")
        
        # Adaptive parameters
        self._latency_samples: List[float] = []
        self._throughput_samples: List[float] = []
        self._resource_samples: List[float] = []
        self._success_rate_window = []
        
        # Start processing
        self.start()
    
    def submit(self, item: T, priority: int = 0, metadata: Optional[Dict[str, Any]] = None) -> Future:
        """
        Submit an item for batch processing.
        
        Args:
            item: Item to process
            priority: Processing priority (lower = higher priority)
            metadata: Additional metadata for the item
            
        Returns:
            Future that will contain the processing result
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("Batch processor is shutdown")
        
        future = self._executor.submit(lambda: None)  # Placeholder
        
        batch_item = BatchItem(
            data=item,
            priority=priority,
            timestamp=time.time(),
            future=future,
            metadata=metadata or {}
        )
        
        # Use negative priority for heapq (min-heap)
        self._queue.put((-priority, time.time(), batch_item))
        
        return future
    
    def start(self) -> None:
        """Start the batch processing thread."""
        if self._processor_thread is None or not self._processor_thread.is_alive():
            self._processor_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="AdaptiveBatchProcessor"
            )
            self._processor_thread.start()
            self.logger.info("Adaptive batch processor started")
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop the batch processor."""
        self._shutdown_event.set()
        
        if self._processor_thread:
            self._processor_thread.join(timeout=timeout)
            
        self._executor.shutdown(wait=True)
        self.logger.info("Adaptive batch processor stopped")
    
    def _processing_loop(self) -> None:
        """Main processing loop."""
        while not self._shutdown_event.is_set():
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
                else:
                    # No items to process, wait briefly
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def _collect_batch(self) -> List[BatchItem[T]]:
        """Collect items for next batch based on current strategy."""
        batch = []
        start_time = time.time()
        max_wait = self.config.max_wait_time / 1000.0  # Convert to seconds
        
        # Collect items until batch size or timeout
        while (len(batch) < self._current_batch_size and 
               (time.time() - start_time) < max_wait and
               not self._shutdown_event.is_set()):
            
            try:
                # Wait for item with timeout
                remaining_wait = max_wait - (time.time() - start_time)
                if remaining_wait <= 0:
                    break
                    
                priority, timestamp, item = self._queue.get(timeout=min(remaining_wait, 0.1))
                batch.append(item)
                
            except queue.Empty:
                # Timeout or no items
                if batch:
                    break  # Process what we have
                continue
        
        return batch
    
    def _process_batch(self, batch: List[BatchItem[T]]) -> None:
        """Process a batch of items."""
        if not batch:
            return
            
        batch_start_time = time.time()
        
        try:
            # Extract data from batch items
            batch_data = [item.data for item in batch]
            
            # Calculate wait times
            wait_times = [batch_start_time - item.timestamp for item in batch]
            avg_wait_time = statistics.mean(wait_times)
            
            # Process the batch
            processing_start = time.time()
            results = self.processor_func(batch_data)
            processing_time = time.time() - processing_start
            
            # Handle results
            if len(results) != len(batch):
                self.logger.warning(f"Result count mismatch: {len(results)} vs {len(batch)}")
                # Pad or truncate results
                while len(results) < len(batch):
                    results.append(None)
                results = results[:len(batch)]
            
            success_count = 0
            for item, result in zip(batch, results):
                try:
                    if item.future and not item.future.cancelled():
                        item.future.set_result(result)
                        success_count += 1
                except Exception as e:
                    if item.future and not item.future.cancelled():
                        item.future.set_exception(e)
                    self.logger.warning(f"Failed to set future result: {e}")
            
            # Calculate metrics
            total_time = time.time() - batch_start_time
            throughput = len(batch) / total_time if total_time > 0 else 0
            latency = (avg_wait_time + processing_time) * 1000  # Convert to ms
            success_rate = success_count / len(batch) if batch else 0
            
            # Collect resource metrics
            cpu_usage, memory_usage = self._get_resource_usage()
            
            batch_metrics = BatchMetrics(
                batch_size=len(batch),
                processing_time=processing_time * 1000,  # ms
                wait_time=avg_wait_time * 1000,  # ms
                throughput=throughput,
                latency=latency,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                success_rate=success_rate
            )
            
            # Update performance tracking
            self._update_performance_tracking(batch_metrics)
            
            # Adaptive adjustment
            self._adaptive_adjustment(batch_metrics)
            
            # Log metrics
            self.metrics.histogram("batch_size", len(batch))
            self.metrics.histogram("batch_processing_time_ms", processing_time * 1000)
            self.metrics.histogram("batch_latency_ms", latency)
            self.metrics.histogram("batch_throughput", throughput)
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            
            # Set exceptions for all futures
            for item in batch:
                if item.future and not item.future.cancelled():
                    item.future.set_exception(e)
    
    def _get_resource_usage(self) -> Tuple[float, float]:
        """Get current resource usage."""
        try:
            import psutil
            process = psutil.Process()
            cpu_usage = process.cpu_percent()
            memory_usage = process.memory_percent()
            return cpu_usage, memory_usage
        except Exception:
            return 0.0, 0.0
    
    def _update_performance_tracking(self, metrics: BatchMetrics) -> None:
        """Update performance tracking with new metrics."""
        with self._lock:
            self._metrics_history.append(metrics)
            
            # Keep limited history
            if len(self._metrics_history) > 1000:
                self._metrics_history = self._metrics_history[-500:]
            
            # Update sample arrays
            self._latency_samples.append(metrics.latency)
            self._throughput_samples.append(metrics.throughput)
            self._resource_samples.append(max(metrics.cpu_usage, metrics.memory_usage))
            self._success_rate_window.append(metrics.success_rate)
            
            # Keep sample windows limited
            for samples in [self._latency_samples, self._throughput_samples, 
                          self._resource_samples, self._success_rate_window]:
                if len(samples) > 100:
                    samples.pop(0)
    
    def _adaptive_adjustment(self, metrics: BatchMetrics) -> None:
        """Adjust batch size based on performance metrics."""
        if self.config.strategy == BatchingStrategy.FIXED_SIZE:
            return
            
        current_time = time.time()
        if current_time - self._last_adjustment < self._adjustment_cooldown:
            return
        
        with self._lock:
            old_batch_size = self._current_batch_size
            
            if self.config.strategy == BatchingStrategy.ADAPTIVE_SIZE:
                self._adaptive_size_adjustment(metrics)
            elif self.config.strategy == BatchingStrategy.LATENCY_AWARE:
                self._latency_aware_adjustment(metrics)
            elif self.config.strategy == BatchingStrategy.THROUGHPUT_OPTIMIZED:
                self._throughput_optimized_adjustment(metrics)
            elif self.config.strategy == BatchingStrategy.RESOURCE_AWARE:
                self._resource_aware_adjustment(metrics)
            
            # Ensure batch size is within bounds
            self._current_batch_size = max(
                self.config.min_batch_size,
                min(self.config.max_batch_size, self._current_batch_size)
            )
            
            if self._current_batch_size != old_batch_size:
                self.logger.info(f"Adjusted batch size: {old_batch_size} -> {self._current_batch_size}")
                self._last_adjustment = current_time
    
    def _adaptive_size_adjustment(self, metrics: BatchMetrics) -> None:
        """General adaptive batch size adjustment."""
        if not self._metrics_history or len(self._metrics_history) < 3:
            return
        
        # Calculate trends
        recent_metrics = self._metrics_history[-3:]
        latency_trend = recent_metrics[-1].latency - recent_metrics[0].latency
        throughput_trend = recent_metrics[-1].throughput - recent_metrics[0].throughput
        
        # Adjust based on trends
        if latency_trend > self.config.target_latency * 0.1:  # Latency increasing
            self._current_batch_size = int(self._current_batch_size * 0.9)
        elif throughput_trend > 0 and latency_trend < self.config.target_latency * 0.05:
            self._current_batch_size = int(self._current_batch_size * 1.1)
    
    def _latency_aware_adjustment(self, metrics: BatchMetrics) -> None:
        """Adjust batch size to meet latency targets."""
        if metrics.latency > self.config.target_latency * 1.2:  # 20% over target
            self._current_batch_size = max(
                self.config.min_batch_size,
                int(self._current_batch_size * 0.8)
            )
        elif metrics.latency < self.config.target_latency * 0.8:  # 20% under target
            self._current_batch_size = min(
                self.config.max_batch_size,
                int(self._current_batch_size * 1.2)
            )
    
    def _throughput_optimized_adjustment(self, metrics: BatchMetrics) -> None:
        """Adjust batch size to maximize throughput."""
        if len(self._throughput_samples) < 3:
            return
        
        recent_throughput = statistics.mean(self._throughput_samples[-3:])
        
        if recent_throughput < self.config.target_throughput:
            # Try to increase batch size for better throughput
            self._current_batch_size = min(
                self.config.max_batch_size,
                int(self._current_batch_size * 1.1)
            )
        elif recent_throughput > self.config.target_throughput * 1.2:
            # Can afford to reduce batch size if latency is high
            if metrics.latency > self.config.target_latency:
                self._current_batch_size = max(
                    self.config.min_batch_size,
                    int(self._current_batch_size * 0.9)
                )
    
    def _resource_aware_adjustment(self, metrics: BatchMetrics) -> None:
        """Adjust batch size based on resource usage."""
        resource_usage = max(metrics.cpu_usage, metrics.memory_usage) / 100.0
        
        if resource_usage > self.config.resource_threshold:
            # High resource usage, reduce batch size
            self._current_batch_size = max(
                self.config.min_batch_size,
                int(self._current_batch_size * 0.8)
            )
        elif resource_usage < self.config.resource_threshold * 0.6:
            # Low resource usage, can increase batch size
            self._current_batch_size = min(
                self.config.max_batch_size,
                int(self._current_batch_size * 1.1)
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if not self._metrics_history:
                return {"status": "no_data"}
            
            recent_metrics = self._metrics_history[-10:] if len(self._metrics_history) >= 10 else self._metrics_history
            
            return {
                "current_batch_size": self._current_batch_size,
                "strategy": self.config.strategy.value,
                "queue_size": self._queue.qsize(),
                "total_batches_processed": len(self._metrics_history),
                "recent_performance": {
                    "avg_latency_ms": statistics.mean([m.latency for m in recent_metrics]),
                    "avg_throughput": statistics.mean([m.throughput for m in recent_metrics]),
                    "avg_batch_size": statistics.mean([m.batch_size for m in recent_metrics]),
                    "avg_success_rate": statistics.mean([m.success_rate for m in recent_metrics])
                } if recent_metrics else {},
                "performance_trends": {
                    "latency_samples": len(self._latency_samples),
                    "throughput_samples": len(self._throughput_samples),
                    "resource_samples": len(self._resource_samples)
                }
            }
    
    def update_config(self, config: BatchConfig) -> None:
        """Update batch processing configuration."""
        with self._lock:
            self.config = config
            
            # Reset batch size within new bounds
            self._current_batch_size = max(
                config.min_batch_size,
                min(config.max_batch_size, self._current_batch_size)
            )
            
        self.logger.info(f"Updated batch config: strategy={config.strategy.value}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()