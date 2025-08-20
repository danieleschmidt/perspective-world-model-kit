"""Circuit breaker pattern for fault tolerance and resilience."""

import time
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass
from threading import Lock
from .logging import get_logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: type = Exception
    call_timeout: float = 30.0  # seconds


class CircuitBreakerError(Exception):
    """Circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.lock = Lock()
        self.logger = get_logger(self.__class__.__name__)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        def wrapper(*args, **kwargs) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    self.logger.warning("Circuit breaker is OPEN, failing fast")
                    raise CircuitBreakerError("Circuit breaker is open")
        
        try:
            # Execute with timeout
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Check for timeout
            if duration > self.config.call_timeout:
                self.logger.warning(f"Function call took {duration:.2f}s, exceeded timeout")
                raise TimeoutError(f"Function call exceeded timeout of {self.config.call_timeout}s")
            
            # Success - reset failure count
            with self.lock:
                self._on_success()
            
            return result
            
        except self.config.expected_exception as e:
            with self.lock:
                self._on_failure()
            self.logger.error(f"Circuit breaker recorded failure: {e}")
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.success_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.error(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.logger.info("Circuit breaker manually reset")


class ModelCircuitBreaker(CircuitBreaker):
    """Specialized circuit breaker for ML models."""
    
    def __init__(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=(RuntimeError, ValueError, TypeError),
            call_timeout=10.0
        )
        super().__init__(config)
    
    def _on_failure(self) -> None:
        """Handle model failure with additional logging."""
        super()._on_failure()
        self.logger.error(
            f"Model failure {self.failure_count}/{self.config.failure_threshold}. "
            f"State: {self.state}"
        )


class BeliefStoreCircuitBreaker(CircuitBreaker):
    """Specialized circuit breaker for belief store operations."""
    
    def __init__(self):
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=(RuntimeError, ValueError, ConnectionError),
            call_timeout=5.0
        )
        super().__init__(config)


class NetworkCircuitBreaker(CircuitBreaker):
    """Specialized circuit breaker for network operations."""
    
    def __init__(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=120.0,
            expected_exception=(ConnectionError, TimeoutError, OSError),
            call_timeout=30.0
        )
        super().__init__(config)


# Global circuit breaker instances
_model_circuit_breaker = None
_belief_store_circuit_breaker = None
_network_circuit_breaker = None


def get_model_circuit_breaker() -> ModelCircuitBreaker:
    """Get global model circuit breaker."""
    global _model_circuit_breaker
    if _model_circuit_breaker is None:
        _model_circuit_breaker = ModelCircuitBreaker()
    return _model_circuit_breaker


def get_belief_store_circuit_breaker() -> BeliefStoreCircuitBreaker:
    """Get global belief store circuit breaker."""
    global _belief_store_circuit_breaker
    if _belief_store_circuit_breaker is None:
        _belief_store_circuit_breaker = BeliefStoreCircuitBreaker()
    return _belief_store_circuit_breaker


def get_network_circuit_breaker() -> NetworkCircuitBreaker:
    """Get global network circuit breaker."""
    global _network_circuit_breaker
    if _network_circuit_breaker is None:
        _network_circuit_breaker = NetworkCircuitBreaker()
    return _network_circuit_breaker