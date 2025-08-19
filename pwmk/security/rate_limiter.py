"""Rate limiting and throttling for security and performance."""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from threading import Lock
from ..utils.logging import get_logger


class RateLimiter:
    """Token bucket rate limiter for API and belief operations."""
    
    def __init__(
        self,
        requests_per_second: float = 100.0,
        burst_capacity: int = 200,
        window_size: int = 60
    ):
        self.requests_per_second = requests_per_second
        self.burst_capacity = burst_capacity
        self.window_size = window_size
        
        # Per-client rate limiting
        self.client_buckets: Dict[str, Tuple[float, float]] = {}  # (tokens, last_update)
        self.client_windows: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Global rate limiting
        self.global_bucket = [burst_capacity, time.time()]
        self.global_window = deque()
        
        self.lock = Lock()
        self.logger = get_logger(self.__class__.__name__)
    
    def _refill_bucket(self, tokens: float, last_update: float) -> Tuple[float, float]:
        """Refill token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - last_update
        
        # Add tokens based on elapsed time
        new_tokens = min(
            self.burst_capacity,
            tokens + (elapsed * self.requests_per_second)
        )
        
        return new_tokens, now
    
    def check_rate_limit(self, client_id: str, tokens_needed: int = 1) -> bool:
        """Check if request is within rate limits."""
        with self.lock:
            now = time.time()
            
            # Check global rate limit first
            if not self._check_global_limit(now, tokens_needed):
                self.logger.warning(f"Global rate limit exceeded for client {client_id}")
                return False
            
            # Check per-client rate limit
            if not self._check_client_limit(client_id, now, tokens_needed):
                self.logger.warning(f"Client rate limit exceeded for {client_id}")
                return False
            
            # Consume tokens if allowed
            self._consume_tokens(client_id, now, tokens_needed)
            return True
    
    def _check_global_limit(self, now: float, tokens_needed: int) -> bool:
        """Check global rate limit."""
        # Clean old entries from sliding window
        cutoff = now - self.window_size
        while self.global_window and self.global_window[0] < cutoff:
            self.global_window.popleft()
        
        # Check if we're within window limits
        max_requests_in_window = self.requests_per_second * self.window_size
        if len(self.global_window) + tokens_needed > max_requests_in_window:
            return False
        
        # Check token bucket
        self.global_bucket[0], self.global_bucket[1] = self._refill_bucket(
            self.global_bucket[0], self.global_bucket[1]
        )
        
        return self.global_bucket[0] >= tokens_needed
    
    def _check_client_limit(self, client_id: str, now: float, tokens_needed: int) -> bool:
        """Check per-client rate limit."""
        # Clean old entries from client window
        client_window = self.client_windows[client_id]
        cutoff = now - self.window_size
        while client_window and client_window[0] < cutoff:
            client_window.popleft()
        
        # Check window limits (per-client limits are typically lower)
        max_client_requests = (self.requests_per_second * self.window_size) / 10  # 10% of global
        if len(client_window) + tokens_needed > max_client_requests:
            return False
        
        # Check/update client token bucket
        if client_id not in self.client_buckets:
            self.client_buckets[client_id] = (self.burst_capacity / 10, now)  # Smaller bucket
        
        tokens, last_update = self.client_buckets[client_id]
        tokens, _ = self._refill_bucket(tokens, last_update)
        
        return tokens >= tokens_needed
    
    def _consume_tokens(self, client_id: str, now: float, tokens_needed: int):
        """Consume tokens from buckets and update windows."""
        # Consume from global bucket
        self.global_bucket[0] -= tokens_needed
        for _ in range(tokens_needed):
            self.global_window.append(now)
        
        # Consume from client bucket
        tokens, _ = self.client_buckets[client_id]
        tokens, _ = self._refill_bucket(tokens, self.client_buckets[client_id][1])
        self.client_buckets[client_id] = (tokens - tokens_needed, now)
        
        # Update client window
        client_window = self.client_windows[client_id]
        for _ in range(tokens_needed):
            client_window.append(now)
    
    def get_stats(self, client_id: Optional[str] = None) -> Dict:
        """Get rate limiting statistics."""
        with self.lock:
            now = time.time()
            
            stats = {
                "global_tokens": self.global_bucket[0],
                "global_requests_in_window": len(self.global_window),
                "requests_per_second": self.requests_per_second,
                "burst_capacity": self.burst_capacity
            }
            
            if client_id and client_id in self.client_buckets:
                tokens, last_update = self.client_buckets[client_id]
                tokens, _ = self._refill_bucket(tokens, last_update)
                
                stats["client_tokens"] = tokens
                stats["client_requests_in_window"] = len(self.client_windows[client_id])
            
            return stats
    
    def reset_client_limits(self, client_id: str):
        """Reset rate limits for a specific client."""
        with self.lock:
            if client_id in self.client_buckets:
                del self.client_buckets[client_id]
            if client_id in self.client_windows:
                self.client_windows[client_id].clear()
            
            self.logger.info(f"Reset rate limits for client {client_id}")


class SecurityThrottler:
    """Advanced throttling for suspicious patterns and security events."""
    
    def __init__(self):
        self.failed_attempts: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_clients: Dict[str, float] = {}  # client_id -> unblock_time
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        
        self.lock = Lock()
        self.logger = get_logger(self.__class__.__name__)
        
        # Thresholds
        self.max_failed_attempts = 5
        self.failure_window = 300  # 5 minutes
        self.block_duration = 900  # 15 minutes
        self.escalation_multiplier = 2
    
    def record_failed_attempt(self, client_id: str, attempt_type: str = "general"):
        """Record a failed attempt for progressive blocking."""
        with self.lock:
            now = time.time()
            
            # Clean old failures
            failures = self.failed_attempts[client_id]
            cutoff = now - self.failure_window
            while failures and failures[0] < cutoff:
                failures.popleft()
            
            # Add new failure
            failures.append(now)
            
            # Check if client should be blocked
            if len(failures) >= self.max_failed_attempts:
                # Progressive blocking - longer each time
                previous_blocks = sum(1 for t in self.blocked_clients.values() if t > now - 3600)
                block_duration = self.block_duration * (self.escalation_multiplier ** previous_blocks)
                
                self.blocked_clients[client_id] = now + block_duration
                self.logger.warning(
                    f"Blocked client {client_id} for {block_duration}s due to {len(failures)} "
                    f"failed attempts of type '{attempt_type}'"
                )
                
                # Clear failures since client is now blocked
                failures.clear()
    
    def is_client_blocked(self, client_id: str) -> bool:
        """Check if client is currently blocked."""
        with self.lock:
            if client_id not in self.blocked_clients:
                return False
            
            unblock_time = self.blocked_clients[client_id]
            if time.time() >= unblock_time:
                # Unblock client
                del self.blocked_clients[client_id]
                self.logger.info(f"Unblocked client {client_id}")
                return False
            
            return True
    
    def record_suspicious_pattern(self, pattern: str, client_id: str):
        """Record suspicious patterns for monitoring."""
        with self.lock:
            key = f"{client_id}:{pattern}"
            self.suspicious_patterns[key] += 1
            
            # Auto-block for repeated suspicious patterns
            if self.suspicious_patterns[key] >= 3:
                self.blocked_clients[client_id] = time.time() + (self.block_duration * 2)
                self.logger.error(
                    f"Auto-blocked client {client_id} for repeated suspicious pattern: {pattern}"
                )
    
    def get_security_stats(self) -> Dict:
        """Get security statistics."""
        with self.lock:
            now = time.time()
            active_blocks = sum(1 for t in self.blocked_clients.values() if t > now)
            
            return {
                "active_blocked_clients": active_blocks,
                "total_blocked_clients": len(self.blocked_clients),
                "suspicious_patterns": len(self.suspicious_patterns),
                "recent_failures": sum(len(failures) for failures in self.failed_attempts.values())
            }


# Global instances
_rate_limiter = None
_security_throttler = None

def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

def get_security_throttler() -> SecurityThrottler:
    """Get global security throttler instance."""
    global _security_throttler
    if _security_throttler is None:
        _security_throttler = SecurityThrottler()
    return _security_throttler