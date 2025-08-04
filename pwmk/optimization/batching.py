"""Intelligent batching and dynamic processing for PWMK."""

import time
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from collections import deque
import threading
import torch
import numpy as np
from queue import Queue, Empty

from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector


class BatchProcessor(LoggingMixin):
    """Efficient batch processing for model inference."""
    
    def __init__(
        self,
        batch_size: int = 32,
        timeout: float = 0.1,
        max_queue_size: int = 1000
    ):
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_queue_size = max_queue_size
        
        self.request_queue: Queue = Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.collector = get_metrics_collector()
        
    def start(self, processor_func: Callable[[List[Any]], List[Any]]) -> None:
        """Start the batch processor with a processing function."""
        if self.running:
            self.logger.warning("Batch processor already running")
            return
        
        self.running = True
        self.processor_func = processor_func
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.logger.info(f"Started batch processor: batch_size={self.batch_size}, timeout={self.timeout}")
    
    def stop(self) -> None:
        """Stop the batch processor."""
        if not self.running:
            return
        
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        self.logger.info("Stopped batch processor")
    
    def process_async(self, request: Any) -> 'Future':
        """Process request asynchronously."""
        if not self.running:
            raise RuntimeError("Batch processor not running")
        
        future = Future()
        
        try:
            self.request_queue.put((request, future), timeout=1.0)
        except:
            future.set_exception(Exception("Request queue full"))
        
        return future
    
    def _worker_loop(self) -> None:
        """Main worker loop for batch processing."""
        batch_requests = []
        batch_futures = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Try to get a request with timeout
                try:
                    request, future = self.request_queue.get(timeout=0.01)
                    batch_requests.append(request)
                    batch_futures.append(future)
                except Empty:
                    pass
                
                current_time = time.time()
                
                # Process batch if conditions are met
                should_process = (
                    len(batch_requests) >= self.batch_size or
                    (batch_requests and (current_time - last_batch_time) >= self.timeout)
                )
                
                if should_process and batch_requests:
                    self._process_batch(batch_requests, batch_futures)
                    batch_requests.clear()
                    batch_futures.clear()
                    last_batch_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error in batch processor: {e}")
                # Set exceptions on pending futures
                for future in batch_futures:
                    future.set_exception(e)
                batch_requests.clear()
                batch_futures.clear()
        
        # Process remaining requests
        if batch_requests:
            self._process_batch(batch_requests, batch_futures)
    
    def _process_batch(self, requests: List[Any], futures: List['Future']) -> None:
        """Process a batch of requests."""
        start_time = time.time()
        
        try:
            # Process the batch
            results = self.processor_func(requests)
            
            # Set results on futures
            for result, future in zip(results, futures):
                future.set_result(result)
            
            # Record metrics
            duration = time.time() - start_time
            self.collector.record_model_forward("BatchProcessor", len(requests), duration)
            
            self.logger.debug(f"Processed batch of {len(requests)} requests in {duration:.4f}s")
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            for future in futures:
                future.set_exception(e)


class Future:
    """Simple future implementation for async results."""
    
    def __init__(self):
        self._result = None
        self._exception = None
        self._done = False
        self._event = threading.Event()
    
    def set_result(self, result: Any) -> None:
        """Set the result."""
        self._result = result
        self._done = True
        self._event.set()
    
    def set_exception(self, exception: Exception) -> None:
        """Set an exception."""
        self._exception = exception
        self._done = True
        self._event.set()
    
    def result(self, timeout: Optional[float] = None) -> Any:
        """Get the result, blocking if necessary."""
        if not self._event.wait(timeout):
            raise TimeoutError("Future timed out")
        
        if self._exception:
            raise self._exception
        
        return self._result
    
    def done(self) -> bool:
        """Check if the future is done."""
        return self._done


class DynamicBatcher(LoggingMixin):
    """Dynamic batching that adapts to system load and performance."""
    
    def __init__(
        self,
        initial_batch_size: int = 16,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        target_latency: float = 0.1,
        adaptation_rate: float = 0.1
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        
        self.collector = get_metrics_collector()
    
    def process_batch(
        self, 
        requests: List[Any], 
        processor_func: Callable[[List[Any]], List[Any]]
    ) -> List[Any]:
        """Process a batch and adapt batch size based on performance."""
        start_time = time.time()
        
        # Process the batch
        results = processor_func(requests)
        
        # Record performance metrics
        latency = time.time() - start_time
        throughput = len(requests) / latency if latency > 0 else 0
        
        self.latency_history.append(latency)
        self.throughput_history.append(throughput)
        
        # Adapt batch size
        self._adapt_batch_size(latency, throughput)
        
        # Record metrics
        self.collector.record_model_forward("DynamicBatcher", len(requests), latency)
        
        self.logger.debug(
            f"Processed batch: size={len(requests)}, latency={latency:.4f}s, "
            f"throughput={throughput:.1f} req/s, next_batch_size={self.current_batch_size}"
        )
        
        return results
    
    def _adapt_batch_size(self, latency: float, throughput: float) -> None:
        """Adapt batch size based on performance metrics."""
        if len(self.latency_history) < 10:
            return  # Need some history
        
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
        
        # Adaptation logic
        if avg_latency > self.target_latency * 1.2:
            # Latency too high, decrease batch size
            new_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adaptation_rate))
            )
        elif avg_latency < self.target_latency * 0.8 and throughput > avg_throughput * 0.9:
            # Latency acceptable and good throughput, try increasing batch size
            new_size = min(
                self.max_batch_size,
                int(self.current_batch_size * (1 + self.adaptation_rate))
            )
        else:
            new_size = self.current_batch_size
        
        if new_size != self.current_batch_size:
            self.logger.debug(
                f"Adapting batch size: {self.current_batch_size} -> {new_size} "
                f"(latency: {avg_latency:.4f}s, target: {self.target_latency:.4f}s)"
            )
            self.current_batch_size = new_size
    
    def get_optimal_batch_size(self) -> int:
        """Get the current optimal batch size."""
        return self.current_batch_size
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.latency_history:
            return {}
        
        return {
            "avg_latency": sum(self.latency_history) / len(self.latency_history),
            "avg_throughput": sum(self.throughput_history) / len(self.throughput_history),
            "current_batch_size": self.current_batch_size,
            "target_latency": self.target_latency
        }


class StreamingBatcher(LoggingMixin):
    """Streaming batch processor for real-time scenarios."""
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.05,
        buffer_size: int = 1000
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.buffer_size = buffer_size
        
        self.buffer: deque = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.collector = get_metrics_collector()
    
    def add_request(self, request: Any) -> None:
        """Add a request to the streaming buffer."""
        with self.lock:
            self.buffer.append((request, time.time()))
    
    def get_batch(self) -> Tuple[List[Any], List[float]]:
        """Get a batch of requests ready for processing."""
        with self.lock:
            if not self.buffer:
                return [], []
            
            current_time = time.time()
            batch_requests = []
            batch_timestamps = []
            
            # Collect requests for batch
            while self.buffer and len(batch_requests) < self.max_batch_size:
                request, timestamp = self.buffer[0]
                
                # Check if request is ready (either buffer full or max wait time exceeded)
                if (len(batch_requests) > 0 and 
                    current_time - timestamp < self.max_wait_time and 
                    len(batch_requests) < self.max_batch_size):
                    break
                
                request, timestamp = self.buffer.popleft()
                batch_requests.append(request)
                batch_timestamps.append(timestamp)
            
            return batch_requests, batch_timestamps
    
    def process_stream(
        self, 
        processor_func: Callable[[List[Any]], List[Any]],
        result_handler: Callable[[Any, float], None]
    ) -> None:
        """Process streaming requests continuously."""
        self.logger.info("Started streaming batch processor")
        
        while True:
            batch_requests, batch_timestamps = self.get_batch()
            
            if not batch_requests:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            start_time = time.time()
            
            try:
                # Process batch
                results = processor_func(batch_requests)
                processing_time = time.time() - start_time
                
                # Handle results
                for result, timestamp in zip(results, batch_timestamps):
                    latency = time.time() - timestamp
                    result_handler(result, latency)
                
                # Record metrics
                self.collector.record_model_forward("StreamingBatcher", len(batch_requests), processing_time)
                
                self.logger.debug(
                    f"Processed streaming batch: size={len(batch_requests)}, "
                    f"processing_time={processing_time:.4f}s"
                )
                
            except Exception as e:
                self.logger.error(f"Streaming batch processing failed: {e}")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                "buffer_size": len(self.buffer),
                "max_buffer_size": self.buffer_size,
                "max_batch_size": self.max_batch_size,
                "max_wait_time": self.max_wait_time
            }