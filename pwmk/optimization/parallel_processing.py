"""Parallel processing and concurrent execution for PWMK."""

import asyncio
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import time
import queue
from dataclasses import dataclass
import torch

from ..utils.logging import LoggingMixin, get_logger
from ..utils.monitoring import get_metrics_collector


@dataclass
class ParallelTask:
    """Represents a task for parallel execution."""
    task_id: str
    function: Callable
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None


class ParallelBeliefProcessor(LoggingMixin):
    """Parallel processing for belief operations."""
    
    def __init__(
        self, 
        max_workers: int = None,
        use_processes: bool = False
    ):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.use_processes = use_processes
        self.metrics = get_metrics_collector()
        
        # Initialize executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Task tracking
        self.active_tasks: Dict[str, Future] = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        self.logger.info(
            f"Initialized parallel processor: {self.max_workers} workers, "
            f"use_processes={use_processes}"
        )
    
    def submit_belief_query_batch(
        self, 
        belief_stores: List[Any], 
        queries: List[str]
    ) -> List[Future]:
        """Submit multiple belief queries for parallel processing."""
        start_time = time.time()
        
        if len(belief_stores) != len(queries):
            raise ValueError("Number of belief stores must match number of queries")
        
        futures = []
        for i, (store, query) in enumerate(zip(belief_stores, queries)):
            task_id = f"query_batch_{int(time.time() * 1000000)}_{i}"
            
            future = self.executor.submit(
                self._safe_belief_query,
                store,
                query,
                task_id
            )
            
            self.active_tasks[task_id] = future
            futures.append(future)
        
        # Record metrics
        duration = time.time() - start_time
        self.metrics.monitor.record_metric("parallel_query_batch_submit", duration)
        self.metrics.monitor.increment_counter("parallel_query_batches", len(queries))
        
        self.logger.debug(f"Submitted {len(queries)} parallel belief queries")
        return futures
    
    def submit_belief_update_batch(
        self,
        belief_stores: List[Any],
        agent_ids: List[str],
        beliefs: List[str]
    ) -> List[Future]:
        """Submit multiple belief updates for parallel processing."""
        start_time = time.time()
        
        if not (len(belief_stores) == len(agent_ids) == len(beliefs)):
            raise ValueError("All lists must have the same length")
        
        futures = []
        for i, (store, agent_id, belief) in enumerate(zip(belief_stores, agent_ids, beliefs)):
            task_id = f"update_batch_{int(time.time() * 1000000)}_{i}"
            
            future = self.executor.submit(
                self._safe_belief_update,
                store,
                agent_id,
                belief,
                task_id
            )
            
            self.active_tasks[task_id] = future
            futures.append(future)
        
        # Record metrics
        duration = time.time() - start_time
        self.metrics.monitor.record_metric("parallel_update_batch_submit", duration)
        self.metrics.monitor.increment_counter("parallel_update_batches", len(beliefs))
        
        self.logger.debug(f"Submitted {len(beliefs)} parallel belief updates")
        return futures
    
    def _safe_belief_query(self, belief_store: Any, query: str, task_id: str) -> Tuple[str, Any]:
        """Safely execute belief query with error handling."""
        try:
            start_time = time.time()
            result = belief_store.query(query)
            duration = time.time() - start_time
            
            self.metrics.monitor.record_metric("parallel_query_duration", duration)
            self.completed_tasks += 1
            
            return task_id, result
            
        except Exception as e:
            self.logger.error(f"Parallel query failed for task {task_id}: {e}")
            self.metrics.monitor.increment_counter("parallel_query_failures")
            self.failed_tasks += 1
            return task_id, None
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    def _safe_belief_update(
        self, 
        belief_store: Any, 
        agent_id: str, 
        belief: str, 
        task_id: str
    ) -> Tuple[str, bool]:
        """Safely execute belief update with error handling."""
        try:
            start_time = time.time()
            belief_store.add_belief(agent_id, belief)
            duration = time.time() - start_time
            
            self.metrics.monitor.record_metric("parallel_update_duration", duration)
            self.completed_tasks += 1
            
            return task_id, True
            
        except Exception as e:
            self.logger.error(f"Parallel update failed for task {task_id}: {e}")
            self.metrics.monitor.increment_counter("parallel_update_failures")
            self.failed_tasks += 1
            return task_id, False
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    def wait_for_completion(self, futures: List[Future], timeout: Optional[float] = None) -> List[Any]:
        """Wait for all futures to complete and return results."""
        results = []
        start_time = time.time()
        
        try:
            for future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Future failed: {e}")
                    results.append(None)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.monitor.record_metric("parallel_wait_duration", duration)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error waiting for completion: {e}")
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "max_workers": self.max_workers,
            "use_processes": self.use_processes,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": (
                self.completed_tasks / max(self.completed_tasks + self.failed_tasks, 1)
            )
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the parallel processor."""
        self.logger.info("Shutting down parallel processor...")
        self.executor.shutdown(wait=wait)
        self.active_tasks.clear()


class AsyncBeliefProcessor:
    """Asynchronous belief processing using asyncio."""
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = get_logger(self.__class__.__name__)
        
    async def process_belief_queries_async(
        self, 
        belief_stores: List[Any], 
        queries: List[str]
    ) -> List[Any]:
        """Process multiple belief queries asynchronously."""
        if len(belief_stores) != len(queries):
            raise ValueError("Number of belief stores must match number of queries")
        
        tasks = []
        for store, query in zip(belief_stores, queries):
            task = asyncio.create_task(self._async_belief_query(store, query))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _async_belief_query(self, belief_store: Any, query: str) -> Any:
        """Execute single belief query asynchronously."""
        async with self.semaphore:
            try:
                # Run in thread pool since belief operations aren't naturally async
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, belief_store.query, query)
                return result
            except Exception as e:
                self.logger.error(f"Async belief query failed: {e}")
                return None


class LoadBalancer(LoggingMixin):
    """Load balancer for distributing belief operations across multiple stores."""
    
    def __init__(self, belief_stores: List[Any], strategy: str = "round_robin"):
        self.belief_stores = belief_stores
        self.strategy = strategy
        self.current_index = 0
        self.request_counts = [0] * len(belief_stores)
        self.lock = threading.Lock()
        
        self.logger.info(
            f"Initialized load balancer with {len(belief_stores)} stores, "
            f"strategy={strategy}"
        )
    
    def get_next_store(self) -> Tuple[Any, int]:
        """Get next belief store based on load balancing strategy."""
        with self.lock:
            if self.strategy == "round_robin":
                store_index = self.current_index
                self.current_index = (self.current_index + 1) % len(self.belief_stores)
                
            elif self.strategy == "least_used":
                store_index = min(
                    range(len(self.request_counts)),
                    key=lambda i: self.request_counts[i]
                )
                
            else:
                # Default to round_robin
                store_index = self.current_index
                self.current_index = (self.current_index + 1) % len(self.belief_stores)
            
            self.request_counts[store_index] += 1
            return self.belief_stores[store_index], store_index
    
    def distributed_query(self, query: str) -> List[Any]:
        """Execute query across all belief stores."""
        results = []
        for i, store in enumerate(self.belief_stores):
            try:
                result = store.query(query)
                results.append({"store_id": i, "result": result, "error": None})
            except Exception as e:
                results.append({"store_id": i, "result": None, "error": str(e)})
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            "strategy": self.strategy,
            "num_stores": len(self.belief_stores),
            "request_counts": self.request_counts,
            "total_requests": sum(self.request_counts)
        }


# Global instances
_parallel_processor = None
_async_processor = None

def get_parallel_processor() -> ParallelBeliefProcessor:
    """Get global parallel processor instance."""
    global _parallel_processor
    if _parallel_processor is None:
        _parallel_processor = ParallelBeliefProcessor()
    return _parallel_processor

def get_async_processor() -> AsyncBeliefProcessor:
    """Get global async processor instance."""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncBeliefProcessor()
    return _async_processor