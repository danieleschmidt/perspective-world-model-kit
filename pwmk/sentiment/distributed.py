"""
Distributed sentiment analysis for large-scale multi-agent systems.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
from abc import ABC, abstractmethod
import uuid
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentTask:
    """Individual sentiment analysis task."""
    task_id: str
    agent_id: int
    text: str
    timestamp: float
    priority: int = 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SentimentResult:
    """Result of sentiment analysis task."""
    task_id: str
    agent_id: int
    sentiment_scores: Dict[str, float]
    processing_time: float
    timestamp: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TaskQueue:
    """
    Thread-safe priority queue for sentiment analysis tasks.
    """
    
    def __init__(self, maxsize: int = 10000):
        self._queue = queue.PriorityQueue(maxsize=maxsize)
        self._task_map = {}
        self._lock = threading.RLock()
        
    def put_task(self, task: SentimentTask, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add task to queue."""
        try:
            priority_item = (-task.priority, task.timestamp, task)
            self._queue.put(priority_item, block=block, timeout=timeout)
            
            with self._lock:
                self._task_map[task.task_id] = task
                
            return True
        except queue.Full:
            logger.warning(f"Task queue full, dropping task {task.task_id}")
            return False
            
    def get_task(self, block: bool = True, timeout: Optional[float] = None) -> Optional[SentimentTask]:
        """Get next task from queue."""
        try:
            _, _, task = self._queue.get(block=block, timeout=timeout)
            
            with self._lock:
                self._task_map.pop(task.task_id, None)
                
            return task
        except queue.Empty:
            return None
            
    def task_done(self) -> None:
        """Mark task as done."""
        self._queue.task_done()
        
    def get_pending_count(self) -> int:
        """Get number of pending tasks."""
        return self._queue.qsize()
        
    def has_task(self, task_id: str) -> bool:
        """Check if task is in queue."""
        with self._lock:
            return task_id in self._task_map
            
    def clear(self) -> None:
        """Clear all tasks."""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._task_map.clear()


class WorkerNode(ABC):
    """Abstract base class for sentiment analysis worker nodes."""
    
    def __init__(self, node_id: str, max_concurrent: int = 4):
        self.node_id = node_id
        self.max_concurrent = max_concurrent
        self.is_running = False
        self._stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0
        }
        
    @abstractmethod
    async def process_task(self, task: SentimentTask) -> SentimentResult:
        """Process a sentiment analysis task."""
        pass
        
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "node_id": self.node_id,
            "is_running": self.is_running,
            "max_concurrent": self.max_concurrent,
            **self._stats
        }
        
    def _update_stats(self, processing_time: float, success: bool) -> None:
        """Update worker statistics."""
        if success:
            self._stats["tasks_processed"] += 1
        else:
            self._stats["tasks_failed"] += 1
            
        self._stats["total_processing_time"] += processing_time
        
        total_tasks = self._stats["tasks_processed"] + self._stats["tasks_failed"]
        if total_tasks > 0:
            self._stats["avg_processing_time"] = self._stats["total_processing_time"] / total_tasks


class LocalWorkerNode(WorkerNode):
    """Local worker node for sentiment analysis."""
    
    def __init__(
        self,
        node_id: str,
        sentiment_analyzer,
        max_concurrent: int = 4
    ):
        super().__init__(node_id, max_concurrent)
        self.analyzer = sentiment_analyzer
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
    async def process_task(self, task: SentimentTask) -> SentimentResult:
        """Process sentiment analysis task locally."""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            sentiment_scores = await loop.run_in_executor(
                self.executor,
                self.analyzer.analyze_text,
                task.text
            )
            
            processing_time = time.time() - start_time
            self._update_stats(processing_time, True)
            
            return SentimentResult(
                task_id=task.task_id,
                agent_id=task.agent_id,
                sentiment_scores=sentiment_scores,
                processing_time=processing_time,
                timestamp=time.time(),
                metadata={"node_id": self.node_id}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)
            
            logger.error(f"Task {task.task_id} failed on node {self.node_id}: {e}")
            
            return SentimentResult(
                task_id=task.task_id,
                agent_id=task.agent_id,
                sentiment_scores={"negative": 0.33, "neutral": 0.34, "positive": 0.33},
                processing_time=processing_time,
                timestamp=time.time(),
                error=str(e),
                metadata={"node_id": self.node_id}
            )


class DistributedSentimentCoordinator:
    """
    Coordinates distributed sentiment analysis across multiple worker nodes.
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        result_timeout: float = 30.0
    ):
        self.task_queue = TaskQueue(maxsize=max_queue_size)
        self.result_timeout = result_timeout
        
        self.workers: Dict[str, WorkerNode] = {}
        self.pending_results: Dict[str, asyncio.Future] = {}
        
        self._coordinator_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Statistics
        self._stats = {
            "total_tasks_submitted": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "avg_queue_time": 0.0
        }
        
        logger.info("Distributed sentiment coordinator initialized")
        
    def add_worker(self, worker: WorkerNode) -> None:
        """Add a worker node to the coordinator."""
        self.workers[worker.node_id] = worker
        logger.info(f"Added worker node: {worker.node_id}")
        
    def remove_worker(self, node_id: str) -> bool:
        """Remove a worker node from the coordinator."""
        if node_id in self.workers:
            del self.workers[node_id]
            logger.info(f"Removed worker node: {node_id}")
            return True
        return False
        
    async def start(self) -> None:
        """Start the distributed coordinator."""
        if self._is_running:
            return
            
        self._is_running = True
        self._coordinator_task = asyncio.create_task(self._coordinate_tasks())
        logger.info("Distributed coordinator started")
        
    async def stop(self) -> None:
        """Stop the distributed coordinator."""
        if not self._is_running:
            return
            
        self._is_running = False
        
        if self._coordinator_task:
            self._coordinator_task.cancel()
            try:
                await self._coordinator_task
            except asyncio.CancelledError:
                pass
                
        # Cancel pending results
        for future in self.pending_results.values():
            future.cancel()
        self.pending_results.clear()
        
        logger.info("Distributed coordinator stopped")
        
    async def submit_task(
        self,
        agent_id: int,
        text: str,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SentimentResult:
        """
        Submit a sentiment analysis task and wait for result.
        
        Args:
            agent_id: Agent identifier
            text: Text to analyze
            priority: Task priority (higher = more urgent)
            metadata: Additional task metadata
            
        Returns:
            Sentiment analysis result
        """
        task_id = str(uuid.uuid4())
        
        task = SentimentTask(
            task_id=task_id,
            agent_id=agent_id,
            text=text,
            timestamp=time.time(),
            priority=priority,
            metadata=metadata or {}
        )
        
        # Create future for result
        result_future = asyncio.Future()
        self.pending_results[task_id] = result_future
        
        # Submit task to queue
        if not self.task_queue.put_task(task, block=False):
            del self.pending_results[task_id]
            raise RuntimeError("Task queue is full")
            
        self._stats["total_tasks_submitted"] += 1
        
        try:
            # Wait for result with timeout
            result = await asyncio.wait_for(result_future, timeout=self.result_timeout)
            return result
        except asyncio.TimeoutError:
            # Clean up
            self.pending_results.pop(task_id, None)
            self._stats["total_tasks_failed"] += 1
            
            logger.warning(f"Task {task_id} timed out")
            
            # Return default result
            return SentimentResult(
                task_id=task_id,
                agent_id=agent_id,
                sentiment_scores={"negative": 0.33, "neutral": 0.34, "positive": 0.33},
                processing_time=self.result_timeout,
                timestamp=time.time(),
                error="Task timeout",
                metadata={"timeout": True}
            )
            
    async def submit_batch(
        self,
        tasks: List[Tuple[int, str]],  # (agent_id, text) pairs
        priority: int = 1
    ) -> List[SentimentResult]:
        """Submit multiple tasks as a batch."""
        batch_tasks = []
        
        for agent_id, text in tasks:
            batch_tasks.append(
                self.submit_task(agent_id, text, priority)
            )
            
        return await asyncio.gather(*batch_tasks, return_exceptions=True)
        
    async def _coordinate_tasks(self) -> None:
        """Main coordination loop."""
        worker_tasks = {}
        
        while self._is_running:
            try:
                # Get available workers
                available_workers = [
                    worker for worker in self.workers.values()
                    if len([t for t in worker_tasks.get(worker.node_id, []) if not t.done()]) < worker.max_concurrent
                ]
                
                if not available_workers:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Get task from queue
                task = self.task_queue.get_task(block=False)
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Select worker (simple round-robin for now)
                worker = available_workers[0]
                
                # Process task
                worker_task = asyncio.create_task(self._process_with_worker(worker, task))
                
                if worker.node_id not in worker_tasks:
                    worker_tasks[worker.node_id] = []
                worker_tasks[worker.node_id].append(worker_task)
                
                # Clean up completed tasks
                for node_id, tasks in worker_tasks.items():
                    worker_tasks[node_id] = [t for t in tasks if not t.done()]
                    
            except Exception as e:
                logger.error(f"Coordination error: {e}")
                await asyncio.sleep(1.0)
                
    async def _process_with_worker(self, worker: WorkerNode, task: SentimentTask) -> None:
        """Process task with specific worker."""
        try:
            result = await worker.process_task(task)
            
            # Send result back to waiting future
            if task.task_id in self.pending_results:
                future = self.pending_results.pop(task.task_id)
                if not future.done():
                    future.set_result(result)
                    
            self._stats["total_tasks_completed"] += 1
            self.task_queue.task_done()
            
        except Exception as e:
            logger.error(f"Worker processing error: {e}")
            
            # Send error result
            if task.task_id in self.pending_results:
                future = self.pending_results.pop(task.task_id)
                if not future.done():
                    error_result = SentimentResult(
                        task_id=task.task_id,
                        agent_id=task.agent_id,
                        sentiment_scores={"negative": 0.33, "neutral": 0.34, "positive": 0.33},
                        processing_time=0.0,
                        timestamp=time.time(),
                        error=str(e),
                        metadata={"worker_error": True}
                    )
                    future.set_result(error_result)
                    
            self._stats["total_tasks_failed"] += 1
            self.task_queue.task_done()
            
    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status and statistics."""
        worker_stats = {
            node_id: worker.get_stats()
            for node_id, worker in self.workers.items()
        }
        
        return {
            "is_running": self._is_running,
            "num_workers": len(self.workers),
            "pending_tasks": self.task_queue.get_pending_count(),
            "pending_results": len(self.pending_results),
            "stats": self._stats,
            "worker_stats": worker_stats
        }


class LoadBalancer:
    """
    Load balancer for distributing sentiment analysis tasks.
    """
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.worker_loads: Dict[str, int] = {}
        self.last_worker_index = 0
        
    def select_worker(self, workers: List[WorkerNode], task: SentimentTask) -> Optional[WorkerNode]:
        """Select best worker for task based on load balancing strategy."""
        if not workers:
            return None
            
        if self.strategy == "round_robin":
            return self._round_robin_selection(workers)
        elif self.strategy == "least_loaded":
            return self._least_loaded_selection(workers)
        elif self.strategy == "priority_aware":
            return self._priority_aware_selection(workers, task)
        else:
            return workers[0]  # Default to first worker
            
    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin worker selection."""
        worker = workers[self.last_worker_index % len(workers)]
        self.last_worker_index += 1
        return worker
        
    def _least_loaded_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least current load."""
        return min(workers, key=lambda w: self.worker_loads.get(w.node_id, 0))
        
    def _priority_aware_selection(self, workers: List[WorkerNode], task: SentimentTask) -> WorkerNode:
        """Priority-aware worker selection."""
        if task.priority > 5:  # High priority task
            # Select worker with best performance
            return min(workers, key=lambda w: w.get_stats()["avg_processing_time"])
        else:
            # Use least loaded for normal priority
            return self._least_loaded_selection(workers)
            
    def update_worker_load(self, node_id: str, load_delta: int) -> None:
        """Update worker load count."""
        current_load = self.worker_loads.get(node_id, 0)
        self.worker_loads[node_id] = max(0, current_load + load_delta)