"""Intelligent caching systems for PWMK components."""

import time
import hashlib
from typing import Dict, Any, Optional, Tuple, Union, List
from collections import OrderedDict
import threading
import torch
import numpy as np

from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector


class LRUCache(LoggingMixin):
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Check TTL
                if current_time - self._timestamps[key] < self.ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return self._cache[key]
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._timestamps[key]
            
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Update existing
                self._cache[key] = value
                self._timestamps[key] = current_time
                self._cache.move_to_end(key)
            else:
                # Add new
                if len(self._cache) >= self.max_size:
                    # Remove oldest
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    del self._timestamps[oldest_key]
                
                self._cache[key] = value
                self._timestamps[key] = current_time
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl": self.ttl
            }


class ModelCache(LoggingMixin):
    """Cache for model predictions and intermediate results."""
    
    def __init__(self, max_size: int = 500, ttl: float = 1800.0):
        self.cache = LRUCache(max_size, ttl)
        self.collector = get_metrics_collector()
    
    def _hash_inputs(self, observations: torch.Tensor, actions: torch.Tensor, agent_ids: Optional[torch.Tensor] = None) -> str:
        """Create hash key for inputs."""
        # Create a deterministic hash of the inputs
        obs_hash = hashlib.md5(observations.detach().cpu().numpy().tobytes()).hexdigest()[:8]
        action_hash = hashlib.md5(actions.detach().cpu().numpy().tobytes()).hexdigest()[:8]
        
        if agent_ids is not None:
            agent_hash = hashlib.md5(agent_ids.detach().cpu().numpy().tobytes()).hexdigest()[:8]
            return f"model_{obs_hash}_{action_hash}_{agent_hash}"
        else:
            return f"model_{obs_hash}_{action_hash}"
    
    def get_prediction(
        self, 
        observations: torch.Tensor, 
        actions: torch.Tensor, 
        agent_ids: Optional[torch.Tensor] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached model prediction."""
        key = self._hash_inputs(observations, actions, agent_ids)
        
        start_time = time.time()
        result = self.cache.get(key)
        lookup_time = time.time() - start_time
        
        self.collector.record_model_forward("ModelCache_lookup", 1, lookup_time)
        
        if result is not None:
            states, beliefs = result
            self.logger.debug(f"Cache hit for model prediction: {key[:16]}...")
            return states.clone(), beliefs.clone()  # Return clones to prevent modification
        
        return None
    
    def cache_prediction(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        beliefs: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None
    ) -> None:
        """Cache model prediction."""
        key = self._hash_inputs(observations, actions, agent_ids)
        
        # Store clones to prevent issues with gradient tracking
        self.cache.put(key, (states.clone().detach(), beliefs.clone().detach()))
        self.logger.debug(f"Cached model prediction: {key[:16]}...")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()


class BeliefCache(LoggingMixin):
    """Cache for belief store queries and results."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 600.0):
        self.cache = LRUCache(max_size, ttl)
        self.collector = get_metrics_collector()
    
    def _hash_query(self, query: str, agent_context: Optional[str] = None) -> str:
        """Create hash key for belief query."""
        query_context = f"{query}_{agent_context}" if agent_context else query
        return hashlib.md5(query_context.encode()).hexdigest()
    
    def get_query_result(self, query: str, agent_context: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
        """Get cached query result."""
        key = self._hash_query(query, agent_context)
        
        start_time = time.time()
        result = self.cache.get(key)
        lookup_time = time.time() - start_time
        
        self.collector.record_belief_operation("cache_lookup", agent_context or "unknown", lookup_time)
        
        if result is not None:
            self.logger.debug(f"Cache hit for belief query: {query[:50]}...")
            return result.copy()  # Return copy to prevent modification
        
        return None
    
    def cache_query_result(
        self,
        query: str,
        result: List[Dict[str, str]],
        agent_context: Optional[str] = None
    ) -> None:
        """Cache query result."""
        key = self._hash_query(query, agent_context)
        self.cache.put(key, result.copy())
        self.logger.debug(f"Cached belief query result: {query[:50]}...")
    
    def invalidate_agent_queries(self, agent_id: str) -> None:
        """Invalidate cached queries for a specific agent."""
        # Clear entire cache for simplicity - could be optimized to only clear agent-specific entries
        self.cache.clear()
        self.logger.info(f"Invalidated belief cache for agent: {agent_id}")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()


class PredictionCache(LoggingMixin):
    """Cache for trajectory predictions and planning results."""
    
    def __init__(self, max_size: int = 200, ttl: float = 300.0):
        self.cache = LRUCache(max_size, ttl)
        self.collector = get_metrics_collector()
    
    def _hash_plan_request(
        self, 
        initial_state: np.ndarray, 
        goal_str: str, 
        planner_config: Dict[str, Any]
    ) -> str:
        """Create hash key for planning request."""
        state_hash = hashlib.md5(initial_state.tobytes()).hexdigest()[:8]
        goal_hash = hashlib.md5(goal_str.encode()).hexdigest()[:8]
        config_str = str(sorted(planner_config.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"plan_{state_hash}_{goal_hash}_{config_hash}"
    
    def get_plan(
        self,
        initial_state: np.ndarray,
        goal_str: str,
        planner_config: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached plan."""
        key = self._hash_plan_request(initial_state, goal_str, planner_config)
        
        start_time = time.time()
        result = self.cache.get(key)
        lookup_time = time.time() - start_time
        
        self.collector.record_planning_step("cache_lookup", lookup_time, 0)
        
        if result is not None:
            self.logger.debug(f"Cache hit for plan: {key[:16]}...")
            return result
        
        return None
    
    def cache_plan(
        self,
        initial_state: np.ndarray,
        goal_str: str,
        planner_config: Dict[str, Any],
        plan: Any
    ) -> None:
        """Cache planning result."""
        key = self._hash_plan_request(initial_state, goal_str, planner_config)
        self.cache.put(key, plan)
        self.logger.debug(f"Cached plan: {key[:16]}...")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()


class CacheManager(LoggingMixin):
    """Central manager for all PWMK caches."""
    
    def __init__(self):
        self.model_cache = ModelCache()
        self.belief_cache = BeliefCache()
        self.prediction_cache = PredictionCache()
        self._enabled = True
    
    def enable(self) -> None:
        """Enable all caching."""
        self._enabled = True
        self.logger.info("Caching enabled")
    
    def disable(self) -> None:
        """Disable all caching."""
        self._enabled = False
        self.logger.info("Caching disabled")
    
    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.model_cache.cache.clear()
        self.belief_cache.cache.clear()
        self.prediction_cache.cache.clear()
        self.logger.info("Cleared all caches")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            "model_cache": self.model_cache.stats(),
            "belief_cache": self.belief_cache.stats(),
            "prediction_cache": self.prediction_cache.stats(),
            "enabled": self._enabled
        }
    
    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        
        self.logger.info("=== Cache Statistics ===")
        for cache_name, cache_stats in stats.items():
            if isinstance(cache_stats, dict) and "hit_rate" in cache_stats:
                self.logger.info(
                    f"{cache_name}: {cache_stats['size']}/{cache_stats['max_size']} items, "
                    f"hit rate: {cache_stats['hit_rate']:.2%}"
                )


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager