"""
Advanced caching system for sentiment analysis operations.
"""

import hashlib
import pickle
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600,
        cleanup_interval: float = 300
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        return time.time() - entry.timestamp > self.ttl_seconds
        
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size estimate
            
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
                
            entry = self._cache[key]
            
            if self._is_expired(entry):
                del self._cache[key]
                self._stats["expired"] += 1
                self._stats["misses"] += 1
                return None
                
            # Update access info and move to end (most recently used)
            entry.access_count += 1
            entry.last_access = time.time()
            self._cache.move_to_end(key)
            
            self._stats["hits"] += 1
            return entry.value
            
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self._lock:
            current_time = time.time()
            size_bytes = self._calculate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
                
            # Ensure we don't exceed max size
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
                
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=current_time,
                access_count=1,
                last_access=current_time,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            
    def cached_call(self, func, *args, **kwargs) -> Any:
        """Execute function with caching."""
        key = self._generate_key(func.__name__, *args, **kwargs)
        
        # Try to get from cache
        cached_result = self.get(key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {func.__name__}")
            return cached_result
            
        # Execute function and cache result
        result = func(*args, **kwargs)
        self.put(key, result)
        logger.debug(f"Cache miss for {func.__name__}, result cached")
        
        return result
        
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired entries."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time - entry.timestamp > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._stats["expired"] += 1
                
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "stats": dict(self._stats),
                "total_size_bytes": total_size,
                "avg_entry_size": total_size / len(self._cache) if self._cache else 0
            }
            
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {key: 0 for key in self._stats}


class SentimentCache:
    """
    Specialized cache for sentiment analysis results.
    """
    
    def __init__(
        self,
        max_text_entries: int = 5000,
        max_model_entries: int = 100,
        text_ttl: float = 1800,  # 30 minutes
        model_ttl: float = 7200   # 2 hours
    ):
        # Cache for text sentiment analysis results
        self.text_cache = LRUCache(
            max_size=max_text_entries,
            ttl_seconds=text_ttl
        )
        
        # Cache for model intermediate results
        self.model_cache = LRUCache(
            max_size=max_model_entries,
            ttl_seconds=model_ttl
        )
        
        # Cache for belief queries
        self.belief_cache = LRUCache(
            max_size=1000,
            ttl_seconds=600  # 10 minutes
        )
        
        logger.info("Sentiment cache system initialized")
        
    def get_text_sentiment(self, text: str, model_name: str) -> Optional[Dict[str, float]]:
        """Get cached sentiment analysis for text."""
        key = self._text_cache_key(text, model_name)
        return self.text_cache.get(key)
        
    def cache_text_sentiment(
        self,
        text: str,
        model_name: str,
        sentiment: Dict[str, float]
    ) -> None:
        """Cache sentiment analysis result."""
        key = self._text_cache_key(text, model_name)
        self.text_cache.put(key, sentiment)
        
    def _text_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text sentiment."""
        # Normalize text for consistent caching
        normalized_text = text.strip().lower()
        key_data = f"{model_name}:{normalized_text}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def get_model_embeddings(
        self,
        input_ids_hash: str,
        model_name: str
    ) -> Optional[Any]:
        """Get cached model embeddings."""
        key = f"{model_name}:{input_ids_hash}"
        return self.model_cache.get(key)
        
    def cache_model_embeddings(
        self,
        input_ids_hash: str,
        model_name: str,
        embeddings: Any
    ) -> None:
        """Cache model embeddings."""
        key = f"{model_name}:{input_ids_hash}"
        self.model_cache.put(key, embeddings)
        
    def get_belief_query_result(self, query: str) -> Optional[List[Dict]]:
        """Get cached belief query result."""
        key = hashlib.md5(query.encode()).hexdigest()
        return self.belief_cache.get(key)
        
    def cache_belief_query_result(self, query: str, result: List[Dict]) -> None:
        """Cache belief query result."""
        key = hashlib.md5(query.encode()).hexdigest()
        self.belief_cache.put(key, result)
        
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "text_cache": self.text_cache.get_stats(),
            "model_cache": self.model_cache.get_stats(),
            "belief_cache": self.belief_cache.get_stats()
        }
        
    def clear_all(self) -> None:
        """Clear all caches."""
        self.text_cache.clear()
        self.model_cache.clear()
        self.belief_cache.clear()
        logger.info("All sentiment caches cleared")


class AsyncSentimentCache:
    """
    Asynchronous version of sentiment cache for high-performance scenarios.
    """
    
    def __init__(self, cache: SentimentCache, executor: Optional[ThreadPoolExecutor] = None):
        self.cache = cache
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        
    async def get_text_sentiment_async(
        self,
        text: str,
        model_name: str
    ) -> Optional[Dict[str, float]]:
        """Asynchronously get cached sentiment."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.cache.get_text_sentiment,
            text,
            model_name
        )
        
    async def cache_text_sentiment_async(
        self,
        text: str,
        model_name: str,
        sentiment: Dict[str, float]
    ) -> None:
        """Asynchronously cache sentiment."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.cache.cache_text_sentiment,
            text,
            model_name,
            sentiment
        )
        
    async def batch_get_sentiments(
        self,
        texts: List[str],
        model_name: str
    ) -> List[Optional[Dict[str, float]]]:
        """Get multiple cached sentiments in batch."""
        tasks = [
            self.get_text_sentiment_async(text, model_name)
            for text in texts
        ]
        return await asyncio.gather(*tasks)
        
    async def batch_cache_sentiments(
        self,
        texts_and_sentiments: List[Tuple[str, Dict[str, float]]],
        model_name: str
    ) -> None:
        """Cache multiple sentiments in batch."""
        tasks = [
            self.cache_text_sentiment_async(text, model_name, sentiment)
            for text, sentiment in texts_and_sentiments
        ]
        await asyncio.gather(*tasks)


# Global cache instances
_sentiment_cache = None
_async_cache = None


def get_sentiment_cache() -> SentimentCache:
    """Get global sentiment cache instance."""
    global _sentiment_cache
    if _sentiment_cache is None:
        _sentiment_cache = SentimentCache()
    return _sentiment_cache


def get_async_sentiment_cache() -> AsyncSentimentCache:
    """Get global async sentiment cache instance."""
    global _async_cache
    if _async_cache is None:
        _async_cache = AsyncSentimentCache(get_sentiment_cache())
    return _async_cache


def clear_global_caches() -> None:
    """Clear all global cache instances."""
    global _sentiment_cache, _async_cache
    if _sentiment_cache:
        _sentiment_cache.clear_all()
    _sentiment_cache = None
    _async_cache = None