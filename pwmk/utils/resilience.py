"""Advanced resilience and error recovery system for PWMK."""

import functools
import time
import asyncio
import random
from typing import Callable, Any, Dict, Optional, Type, Union, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from .logging import LoggingMixin, get_logger


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retriable_exceptions: tuple = (Exception,)


class CircuitBreakerState:
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    name: str = "default"


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.logger = get_logger(f"circuit_breaker.{config.name}")
    
    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.logger.info("Circuit breaker reset to CLOSED state")
    
    def _open_circuit(self):
        """Open the circuit breaker."""
        self.state = CircuitBreakerState.OPEN
        self.last_failure_time = time.time()
        self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _can_attempt(self) -> bool:
        """Check if we can attempt the operation."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if (time.time() - self.last_failure_time) > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker moved to HALF_OPEN state")
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def _record_success(self):
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._reset()
    
    def _record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._open_circuit()
        elif self.failure_count >= self.config.failure_threshold:
            self._open_circuit()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._can_attempt():
                raise Exception(f"Circuit breaker is OPEN for {self.config.name}")
            
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except self.config.expected_exception as e:
                self._record_failure()
                raise
        
        return wrapper


class RetryHandler:
    """Advanced retry handler with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = get_logger("retry_handler")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = min(
            self.config.base_delay * (self.config.backoff_factor ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with retry logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except self.config.retriable_exceptions as e:
                    last_exception = e
                    
                    if attempt == self.config.max_attempts - 1:
                        self.logger.error(
                            f"All {self.config.max_attempts} attempts failed for {func.__name__}",
                            extra={"exception": str(e)}
                        )
                        break
                    
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s",
                        extra={"exception": str(e), "delay": delay}
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper


class TimeoutHandler:
    """Timeout handler for operations."""
    
    def __init__(self, timeout: float, executor: Optional[ThreadPoolExecutor] = None):
        self.timeout = timeout
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self.logger = get_logger("timeout_handler")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with timeout."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            future = self.executor.submit(func, *args, **kwargs)
            
            try:
                return future.result(timeout=self.timeout)
            except FutureTimeoutError:
                self.logger.error(
                    f"Operation {func.__name__} timed out after {self.timeout}s"
                )
                future.cancel()
                raise TimeoutError(f"Operation {func.__name__} timed out")
        
        return wrapper


class FallbackManager:
    """Manager for fallback strategies."""
    
    def __init__(self, fallbacks: List[Callable]):
        self.fallbacks = fallbacks
        self.logger = get_logger("fallback_manager")
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute with fallback strategies."""
        last_exception = None
        
        for i, fallback in enumerate(self.fallbacks):
            try:
                result = fallback(*args, **kwargs)
                if i > 0:
                    self.logger.info(f"Fallback {i} succeeded after primary failed")
                return result
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Fallback {i} failed: {str(e)}",
                    extra={"fallback_index": i, "exception": str(e)}
                )
        
        self.logger.error("All fallback strategies failed")
        raise last_exception


class ResilienceManager(LoggingMixin):
    """Comprehensive resilience manager combining multiple strategies."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        timeout: Optional[float] = None,
        fallbacks: Optional[List[Callable]] = None
    ):
        super().__init__()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config
        self.timeout = timeout
        self.fallbacks = fallbacks or []
        
        # Initialize components
        self.retry_handler = RetryHandler(self.retry_config)
        self.circuit_breaker = (
            CircuitBreaker(self.circuit_breaker_config) 
            if self.circuit_breaker_config else None
        )
        self.timeout_handler = TimeoutHandler(self.timeout) if self.timeout else None
        self.fallback_manager = FallbackManager(self.fallbacks) if self.fallbacks else None
    
    def wrap_function(self, func: Callable) -> Callable:
        """Wrap function with all enabled resilience strategies."""
        wrapped_func = func
        
        # Apply wrappers in order: timeout -> circuit breaker -> retry
        if self.timeout_handler:
            wrapped_func = self.timeout_handler(wrapped_func)
        
        if self.circuit_breaker:
            wrapped_func = self.circuit_breaker(wrapped_func)
        
        wrapped_func = self.retry_handler(wrapped_func)
        
        # Handle fallbacks at the highest level
        if self.fallback_manager:
            @functools.wraps(func)
            def final_wrapper(*args, **kwargs):
                try:
                    return wrapped_func(*args, **kwargs)
                except Exception as e:
                    self.log_warning(f"Primary execution failed, trying fallbacks: {str(e)}")
                    return self.fallback_manager.execute(*args, **kwargs)
            
            return final_wrapper
        
        return wrapped_func
    
    def execute_with_resilience(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with all resilience strategies."""
        resilient_func = self.wrap_function(func)
        return resilient_func(*args, **kwargs)


def resilient(
    retry_attempts: int = 3,
    timeout: Optional[float] = None,
    circuit_breaker_failures: Optional[int] = None,
    fallback_func: Optional[Callable] = None
):
    """Decorator for making functions resilient."""
    def decorator(func: Callable) -> Callable:
        # Build configurations
        retry_config = RetryConfig(max_attempts=retry_attempts)
        
        circuit_breaker_config = None
        if circuit_breaker_failures:
            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=circuit_breaker_failures,
                name=func.__name__
            )
        
        fallbacks = [fallback_func] if fallback_func else []
        
        # Create resilience manager
        manager = ResilienceManager(
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout=timeout,
            fallbacks=fallbacks
        )
        
        return manager.wrap_function(func)
    
    return decorator


class HealthChecker(LoggingMixin):
    """Health checking system for PWMK components."""
    
    def __init__(self):
        super().__init__()
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, bool] = {}
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.log_info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> bool:
        """Run a specific health check."""
        if name not in self.health_checks:
            self.log_error(f"Unknown health check: {name}")
            return False
        
        try:
            result = self.health_checks[name]()
            self.health_status[name] = bool(result)
            return self.health_status[name]
        except Exception as e:
            self.log_error(f"Health check {name} failed: {str(e)}")
            self.health_status[name] = False
            return False
    
    def run_all_checks(self) -> Dict[str, bool]:
        """Run all registered health checks."""
        results = {}
        for name in self.health_checks:
            results[name] = self.run_check(name)
        return results
    
    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        if not self.health_checks:
            return True
        
        results = self.run_all_checks()
        return all(results.values())


# Global health checker instance
global_health_checker = HealthChecker()