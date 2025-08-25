"""Intelligent caching system with adaptive algorithms and quantum-enhanced optimization."""

import time
import threading
import hashlib
import pickle
from typing import Any, Dict, Optional, Callable, List, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
from abc import ABC, abstractmethod

from ..utils.logging import LoggingMixin
from ..utils.resilience import resilient


class CacheStrategy:
    """Cache strategies for different usage patterns."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # Adaptive Replacement Cache
    QUANTUM = "quantum"  # Quantum-inspired optimization


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent eviction."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    hit_rate: float = 0.0
    computation_cost: float = 0.0
    size: int = 0
    priority_score: float = 0.0
    
    def __post_init__(self):
        self.size = len(pickle.dumps(self.value))
    
    def update_access(self) -> None:
        """Update access statistics."""
        current_time = time.time()
        self.last_accessed = current_time
        self.access_count += 1
        
        # Update hit rate (exponentially weighted moving average)
        time_since_creation = current_time - self.created_at
        if time_since_creation > 0:
            self.hit_rate = 0.9 * self.hit_rate + 0.1 * (self.access_count / time_since_creation)
    
    def calculate_priority(self, strategy: str = CacheStrategy.ADAPTIVE) -> float:
        """Calculate priority score for eviction decisions."""
        current_time = time.time()
        age = current_time - self.created_at
        recency = current_time - self.last_accessed
        
        if strategy == CacheStrategy.LRU:
            return -recency  # Most recent first
        elif strategy == CacheStrategy.LFU:
            return self.access_count  # Most frequent first
        elif strategy == CacheStrategy.ADAPTIVE:
            # Adaptive scoring considering multiple factors
            frequency_score = self.access_count / max(age, 1.0)
            recency_score = 1.0 / (1.0 + recency)
            cost_score = self.computation_cost
            size_penalty = 1.0 / (1.0 + self.size / 1024)  # Penalize large entries
            
            return frequency_score * recency_score * cost_score * size_penalty
        elif strategy == CacheStrategy.QUANTUM:
            # Quantum-inspired scoring using superposition of factors
            return self._quantum_priority_score()
        
        return 0.0
    
    def _quantum_priority_score(self) -> float:
        """Quantum-inspired priority calculation using entangled factors."""
        current_time = time.time()
        
        # Quantum state components (normalized)
        frequency_amplitude = min(self.access_count / 100.0, 1.0)
        recency_amplitude = 1.0 / (1.0 + (current_time - self.last_accessed) / 3600)
        cost_amplitude = min(self.computation_cost / 10.0, 1.0)
        
        # Quantum interference pattern
        phase1 = (self.access_count % 10) / 10.0
        phase2 = ((current_time - self.created_at) % 100) / 100.0
        
        # Superposition with quantum interference
        quantum_score = (
            frequency_amplitude * recency_amplitude * 
            (1 + cost_amplitude * (0.5 + 0.5 * (phase1 * phase2)))
        )
        
        return quantum_score


class IntelligentCache(LoggingMixin):
    """Advanced cache with multiple eviction strategies and adaptive optimization."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        strategy: str = CacheStrategy.ADAPTIVE,
        auto_optimize: bool = True,
        optimization_interval: int = 300  # 5 minutes
    ):
        super().__init__()
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.auto_optimize = auto_optimize
        self.optimization_interval = optimization_interval
        
        self.cache: Dict[str, CacheEntry] = OrderedDict()
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_memory_usage = 0
        
        # Background optimization
        self.optimizer_thread: Optional[threading.Thread] = None
        self.optimization_running = False
        
        if auto_optimize:
            self._start_optimization_thread()
    
    def _start_optimization_thread(self) -> None:
        """Start background optimization thread."""
        if self.optimizer_thread and self.optimizer_thread.is_alive():
            return
        
        self.optimization_running = True
        self.optimizer_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self.optimizer_thread.start()
        self.log_info("Started cache optimization thread")
    
    def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while self.optimization_running:
            try:
                time.sleep(self.optimization_interval)
                self._optimize_cache()
                self._analyze_access_patterns()
            except Exception as e:
                self.log_error(f"Cache optimization error: {str(e)}")
    
    def _calculate_key(self, key: Union[str, tuple, Any]) -> str:
        """Calculate cache key from input."""
        if isinstance(key, str):
            return key
        elif isinstance(key, (tuple, list)):
            return hashlib.md5(str(key).encode()).hexdigest()
        else:
            return hashlib.md5(pickle.dumps(key)).hexdigest()
    
    @resilient(retry_attempts=3, timeout=5.0)
    def get(self, key: Union[str, tuple, Any]) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._calculate_key(key)
        
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.update_access()
                
                # Move to end for LRU
                self.cache.move_to_end(cache_key)
                
                # Track access pattern
                self.access_patterns[cache_key].append(time.time())
                
                self.hits += 1
                self.log_debug(f"Cache hit for key: {cache_key[:16]}...")
                return entry.value
            else:
                self.misses += 1
                self.log_debug(f"Cache miss for key: {cache_key[:16]}...")
                return None
    
    @resilient(retry_attempts=3, timeout=10.0)
    def put(
        self, 
        key: Union[str, tuple, Any], 
        value: Any, 
        computation_cost: float = 1.0
    ) -> None:
        """Put value in cache."""
        cache_key = self._calculate_key(key)
        current_time = time.time()
        
        with self.lock:
            # Check if we need to evict entries
            self._ensure_capacity()
            
            # Create new cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                computation_cost=computation_cost
            )
            
            # Add to cache
            self.cache[cache_key] = entry
            self.current_memory_usage += entry.size
            
            self.log_debug(f"Cache put for key: {cache_key[:16]}..., size: {entry.size} bytes")
    
    def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed size or memory limits."""
        # Size-based eviction
        while len(self.cache) >= self.max_size:
            self._evict_one()
        
        # Memory-based eviction
        while self.current_memory_usage > self.max_memory_bytes:
            self._evict_one()
    
    def _evict_one(self) -> None:
        """Evict one cache entry based on strategy."""
        if not self.cache:
            return
        
        # Calculate priorities for all entries
        priorities = {}
        for cache_key, entry in self.cache.items():
            priorities[cache_key] = entry.calculate_priority(self.strategy)
        
        # Find entry with lowest priority
        evict_key = min(priorities, key=priorities.get)
        evicted_entry = self.cache.pop(evict_key)
        
        self.current_memory_usage -= evicted_entry.size
        self.evictions += 1
        
        self.log_debug(f"Evicted cache entry: {evict_key[:16]}...")
    
    def _optimize_cache(self) -> None:
        """Optimize cache performance based on access patterns."""
        with self.lock:
            if not self.cache:
                return
            
            # Analyze cache efficiency
            efficiency_metrics = self._calculate_efficiency_metrics()
            
            # Adjust strategy if needed
            if efficiency_metrics["hit_rate"] < 0.5:
                self._suggest_strategy_change()
            
            # Preemptive eviction of stale entries
            self._evict_stale_entries()
            
            self.log_info(
                "Cache optimization completed",
                hit_rate=efficiency_metrics["hit_rate"],
                memory_usage=self.current_memory_usage,
                entry_count=len(self.cache)
            )
    
    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate cache efficiency metrics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        memory_efficiency = self.current_memory_usage / self.max_memory_bytes
        size_efficiency = len(self.cache) / self.max_size
        
        return {
            "hit_rate": hit_rate,
            "memory_efficiency": memory_efficiency,
            "size_efficiency": size_efficiency,
            "average_entry_size": (
                self.current_memory_usage / len(self.cache) 
                if self.cache else 0
            )
        }
    
    def _suggest_strategy_change(self) -> None:
        """Suggest optimal cache strategy based on access patterns."""
        if self.strategy == CacheStrategy.ADAPTIVE:
            return  # Already using best strategy
        
        # Analyze patterns to suggest better strategy
        temporal_locality = self._analyze_temporal_locality()
        frequency_variance = self._analyze_frequency_variance()
        
        if temporal_locality > 0.8:
            suggested_strategy = CacheStrategy.LRU
        elif frequency_variance < 0.3:
            suggested_strategy = CacheStrategy.LFU
        else:
            suggested_strategy = CacheStrategy.ADAPTIVE
        
        if suggested_strategy != self.strategy:
            self.log_info(
                f"Suggesting strategy change from {self.strategy} to {suggested_strategy}",
                temporal_locality=temporal_locality,
                frequency_variance=frequency_variance
            )
    
    def _analyze_temporal_locality(self) -> float:
        """Analyze temporal locality in access patterns."""
        if not self.access_patterns:
            return 0.0
        
        locality_scores = []
        current_time = time.time()
        
        for access_times in self.access_patterns.values():
            if len(access_times) < 2:
                continue
            
            # Calculate time gaps between accesses
            gaps = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
            
            # Temporal locality score (smaller gaps = higher locality)
            avg_gap = sum(gaps) / len(gaps) if gaps else float('inf')
            locality_score = 1.0 / (1.0 + avg_gap / 3600)  # Normalize to hours
            locality_scores.append(locality_score)
        
        return sum(locality_scores) / len(locality_scores) if locality_scores else 0.0
    
    def _analyze_frequency_variance(self) -> float:
        """Analyze variance in access frequencies."""
        if not self.cache:
            return 0.0
        
        frequencies = [entry.access_count for entry in self.cache.values()]
        if len(frequencies) < 2:
            return 0.0
        
        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)
        
        return variance / (mean_freq ** 2) if mean_freq > 0 else 0.0
    
    def _evict_stale_entries(self) -> None:
        """Evict entries that haven't been accessed recently."""
        current_time = time.time()
        stale_threshold = 3600  # 1 hour
        
        stale_keys = []
        for cache_key, entry in self.cache.items():
            if (current_time - entry.last_accessed) > stale_threshold:
                stale_keys.append(cache_key)
        
        for key in stale_keys:
            entry = self.cache.pop(key)
            self.current_memory_usage -= entry.size
            self.evictions += 1
    
    def _analyze_access_patterns(self) -> None:
        """Analyze access patterns for insights."""
        current_time = time.time()
        
        # Clean old access pattern data (keep last 24 hours)
        cutoff_time = current_time - 86400
        
        for key in list(self.access_patterns.keys()):
            self.access_patterns[key] = [
                t for t in self.access_patterns[key] if t > cutoff_time
            ]
            
            if not self.access_patterns[key]:
                del self.access_patterns[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            efficiency_metrics = self._calculate_efficiency_metrics()
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": efficiency_metrics["hit_rate"],
                "entry_count": len(self.cache),
                "memory_usage_mb": self.current_memory_usage / (1024 * 1024),
                "memory_efficiency": efficiency_metrics["memory_efficiency"],
                "strategy": self.strategy,
                "average_entry_size": efficiency_metrics["average_entry_size"],
                "top_accessed_keys": self._get_top_accessed_keys(10)
            }
    
    def _get_top_accessed_keys(self, limit: int) -> List[Dict[str, Any]]:
        """Get top accessed cache keys."""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        
        return [
            {
                "key": key[:32] + "..." if len(key) > 32 else key,
                "access_count": entry.access_count,
                "hit_rate": entry.hit_rate,
                "size": entry.size
            }
            for key, entry in sorted_entries[:limit]
        ]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_patterns.clear()
            self.current_memory_usage = 0
            self.log_info("Cache cleared")
    
    def stop_optimization(self) -> None:
        """Stop background optimization."""
        self.optimization_running = False
        if self.optimizer_thread:
            self.optimizer_thread.join()
        self.log_info("Cache optimization stopped")


class DistributedCache(LoggingMixin):
    """Distributed cache with consistent hashing and replication."""
    
    def __init__(
        self,
        nodes: List[str],
        replication_factor: int = 2,
        local_cache_size: int = 500
    ):
        super().__init__()
        self.nodes = nodes
        self.replication_factor = min(replication_factor, len(nodes))
        self.local_cache = IntelligentCache(max_size=local_cache_size)
        
        # Consistent hashing ring
        self.hash_ring: Dict[int, str] = {}
        self.virtual_nodes_per_node = 100
        self._build_hash_ring()
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _build_hash_ring(self) -> None:
        """Build consistent hashing ring."""
        for node in self.nodes:
            for i in range(self.virtual_nodes_per_node):
                virtual_node = f"{node}:{i}"
                hash_value = int(hashlib.md5(virtual_node.encode()).hexdigest(), 16)
                self.hash_ring[hash_value] = node
        
        self.log_info(f"Built hash ring with {len(self.hash_ring)} virtual nodes")
    
    def _get_nodes_for_key(self, key: str) -> List[str]:
        """Get nodes responsible for a key."""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        # Find nodes clockwise from hash position
        sorted_hashes = sorted(self.hash_ring.keys())
        
        # Find first node at or after hash value
        start_idx = 0
        for i, h in enumerate(sorted_hashes):
            if h >= hash_value:
                start_idx = i
                break
        
        # Select nodes for replication
        selected_nodes = set()
        for i in range(len(sorted_hashes)):
            idx = (start_idx + i) % len(sorted_hashes)
            node = self.hash_ring[sorted_hashes[idx]]
            selected_nodes.add(node)
            
            if len(selected_nodes) >= self.replication_factor:
                break
        
        return list(selected_nodes)
    
    async def get_distributed(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        # Try local cache first
        local_value = self.local_cache.get(key)
        if local_value is not None:
            return local_value
        
        # Try distributed nodes
        nodes = self._get_nodes_for_key(key)
        
        # Query nodes in parallel
        futures = []
        for node in nodes:
            future = self.executor.submit(self._query_node, node, key)
            futures.append(future)
        
        for future in as_completed(futures, timeout=5.0):
            try:
                result = future.result()
                if result is not None:
                    # Cache locally
                    self.local_cache.put(key, result)
                    return result
            except Exception as e:
                self.log_warning(f"Node query failed: {str(e)}")
        
        return None
    
    def _query_node(self, node: str, key: str) -> Optional[Any]:
        """Query a specific node for a key."""
        # This would implement actual network communication
        # For now, return None (placeholder)
        return None
    
    def put_distributed(self, key: str, value: Any) -> None:
        """Put value in distributed cache."""
        # Update local cache
        self.local_cache.put(key, value)
        
        # Update distributed nodes
        nodes = self._get_nodes_for_key(key)
        
        futures = []
        for node in nodes:
            future = self.executor.submit(self._update_node, node, key, value)
            futures.append(future)
        
        # Wait for at least one successful update
        success_count = 0
        for future in as_completed(futures, timeout=10.0):
            try:
                future.result()
                success_count += 1
            except Exception as e:
                self.log_warning(f"Node update failed: {str(e)}")
        
        if success_count == 0:
            self.log_error(f"Failed to update any nodes for key: {key}")
    
    def _update_node(self, node: str, key: str, value: Any) -> None:
        """Update a specific node with key-value pair."""
        # This would implement actual network communication
        # For now, do nothing (placeholder)
        pass


# Global cache instance
global_cache = IntelligentCache(max_size=2000, max_memory_mb=200)