"""Advanced error recovery and failure handling."""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import time
import traceback
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..utils.logging import get_logger
from ..utils.monitoring import get_metrics_collector


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback" 
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class RecoveryAction:
    """Represents a recovery action for a specific error."""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    backoff_factor: float = 1.5
    timeout_seconds: float = 30.0
    fallback_function: Optional[Callable] = None
    custom_handler: Optional[Callable] = None


class ErrorRecoveryManager:
    """
    Advanced error recovery system with multiple strategies.
    
    Provides automatic error handling, recovery strategies, and
    system resilience for PWMK components.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.metrics = get_metrics_collector()
        
        # Recovery configuration
        self.recovery_strategies: Dict[str, RecoveryAction] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="recovery")
        
        # Default recovery strategies
        self._setup_default_strategies()
        
    def _setup_default_strategies(self) -> None:
        """Setup default recovery strategies for common errors."""
        self.recovery_strategies.update({
            "torch.cuda.OutOfMemoryError": RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                max_attempts=1,
                custom_handler=self._handle_cuda_oom
            ),
            "ConnectionError": RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=5,
                backoff_factor=2.0
            ),
            "FileNotFoundError": RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                max_attempts=1,
                fallback_function=self._create_default_file
            ),
            "PermissionError": RecoveryAction(
                strategy=RecoveryStrategy.EMERGENCY_SHUTDOWN,
                max_attempts=1
            ),
            "MemoryError": RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                max_attempts=2,
                custom_handler=self._handle_memory_error
            ),
            "TimeoutError": RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=3,
                backoff_factor=1.2,
                timeout_seconds=60.0
            )
        })
    
    def register_recovery_strategy(
        self, 
        error_type: str, 
        action: RecoveryAction
    ) -> None:
        """Register a custom recovery strategy for an error type."""
        with self._lock:
            self.recovery_strategies[error_type] = action
            self.logger.info(f"Registered recovery strategy for {error_type}: {action.strategy.value}")
    
    def recover_from_error(
        self, 
        error: Exception, 
        function: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Attempt to recover from an error using registered strategies.
        
        Args:
            error: The exception that occurred
            function: The function that failed
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Result of successful recovery or re-raises exception
        """
        error_type = error.__class__.__name__
        error_key = f"{error.__class__.__module__}.{error_type}"
        
        # Log error occurrence
        error_info = {
            "timestamp": time.time(),
            "error_type": error_type,
            "error_key": error_key,
            "error_message": str(error),
            "function": function.__name__ if hasattr(function, '__name__') else str(function),
            "traceback": traceback.format_exc()
        }
        
        with self._lock:
            self.error_history.append(error_info)
            
        self.logger.error(f"Error occurred in {function.__name__ if hasattr(function, '__name__') else 'function'}: {error}")
        
        # Find recovery strategy
        recovery_action = None
        for strategy_key in [error_key, error_type, "default"]:
            if strategy_key in self.recovery_strategies:
                recovery_action = self.recovery_strategies[strategy_key]
                break
                
        if not recovery_action:
            self.logger.error(f"No recovery strategy found for {error_type}")
            raise error
            
        return self._execute_recovery(recovery_action, error, function, *args, **kwargs)
    
    def _execute_recovery(
        self, 
        action: RecoveryAction, 
        error: Exception, 
        function: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """Execute the recovery strategy."""
        strategy = action.strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_with_backoff(action, function, *args, **kwargs)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback(action, error, function, *args, **kwargs)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(action, error, function, *args, **kwargs)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._circuit_breaker_recovery(action, error, function, *args, **kwargs)
        elif strategy == RecoveryStrategy.EMERGENCY_SHUTDOWN:
            self._emergency_shutdown(error)
            raise error
        else:
            self.logger.error(f"Unknown recovery strategy: {strategy}")
            raise error
    
    def _retry_with_backoff(
        self, 
        action: RecoveryAction, 
        function: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Retry function with exponential backoff."""
        last_error = None
        
        for attempt in range(action.max_attempts):
            try:
                if attempt > 0:
                    delay = action.backoff_factor ** (attempt - 1)
                    self.logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{action.max_attempts})")
                    time.sleep(delay)
                
                result = function(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Recovery successful after {attempt + 1} attempts")
                    
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                
        self.logger.error(f"All {action.max_attempts} retry attempts failed")
        raise last_error
    
    def _execute_fallback(
        self, 
        action: RecoveryAction, 
        error: Exception, 
        function: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """Execute fallback function."""
        if action.fallback_function:
            self.logger.info("Executing fallback function")
            try:
                return action.fallback_function(*args, **kwargs)
            except Exception as fallback_error:
                self.logger.error(f"Fallback function failed: {fallback_error}")
                raise fallback_error
        else:
            self.logger.error("No fallback function configured")
            raise error
    
    def _graceful_degradation(
        self, 
        action: RecoveryAction, 
        error: Exception, 
        function: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """Handle graceful degradation."""
        if action.custom_handler:
            self.logger.info("Executing custom degradation handler")
            return action.custom_handler(error, function, *args, **kwargs)
        else:
            # Default graceful degradation - return safe default value
            self.logger.warning("Using default graceful degradation")
            return self._get_safe_default(function)
    
    def _circuit_breaker_recovery(
        self, 
        action: RecoveryAction, 
        error: Exception, 
        function: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """Implement circuit breaker pattern."""
        func_name = function.__name__ if hasattr(function, '__name__') else str(function)
        
        with self._lock:
            if func_name not in self.circuit_breakers:
                self.circuit_breakers[func_name] = {
                    "failure_count": 0,
                    "last_failure": 0,
                    "state": "closed"  # closed, open, half_open
                }
            
            breaker = self.circuit_breakers[func_name]
            current_time = time.time()
            
            # Check circuit breaker state
            if breaker["state"] == "open":
                if current_time - breaker["last_failure"] > action.timeout_seconds:
                    breaker["state"] = "half_open"
                    self.logger.info(f"Circuit breaker for {func_name} transitioning to half-open")
                else:
                    raise Exception(f"Circuit breaker open for {func_name}")
            
            try:
                result = function(*args, **kwargs)
                
                # Success - reset circuit breaker
                breaker["failure_count"] = 0
                breaker["state"] = "closed"
                
                return result
                
            except Exception as e:
                breaker["failure_count"] += 1
                breaker["last_failure"] = current_time
                
                if breaker["failure_count"] >= action.max_attempts:
                    breaker["state"] = "open"
                    self.logger.error(f"Circuit breaker opened for {func_name}")
                
                raise e
    
    def _emergency_shutdown(self, error: Exception) -> None:
        """Perform emergency shutdown procedures."""
        self.logger.critical(f"Emergency shutdown triggered by: {error}")
        
        # Notify monitoring systems
        self.metrics.increment("emergency_shutdowns")
        
        # Could trigger additional cleanup here
        # For now, just log the critical error
        
    def _handle_cuda_oom(self, error: Exception, function: Callable, *args, **kwargs) -> Any:
        """Handle CUDA out of memory errors."""
        self.logger.warning("CUDA OOM detected - attempting memory cleanup")
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("CUDA cache cleared")
                
            # Try with reduced batch size or other modifications
            return self._get_safe_default(function)
            
        except Exception as e:
            self.logger.error(f"CUDA OOM recovery failed: {e}")
            raise error
    
    def _handle_memory_error(self, error: Exception, function: Callable, *args, **kwargs) -> Any:
        """Handle general memory errors."""
        self.logger.warning("Memory error detected - attempting cleanup")
        
        # Basic cleanup attempt
        import gc
        gc.collect()
        
        return self._get_safe_default(function)
    
    def _create_default_file(self, *args, **kwargs) -> str:
        """Create a default file as fallback."""
        default_path = "/tmp/pwmk_default_file.txt"
        with open(default_path, 'w') as f:
            f.write("# Default PWMK file created by error recovery\n")
        return default_path
    
    def _get_safe_default(self, function: Callable) -> Any:
        """Return a safe default value based on function type."""
        func_name = function.__name__ if hasattr(function, '__name__') else str(function)
        
        # Simple heuristics for default values
        if "predict" in func_name.lower():
            return {"prediction": None, "confidence": 0.0}
        elif "train" in func_name.lower():
            return {"loss": float('inf'), "status": "failed"}
        elif "process" in func_name.lower():
            return []
        else:
            return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and recovery success rates."""
        with self._lock:
            if not self.error_history:
                return {"total_errors": 0}
                
            error_types = {}
            recent_errors = [e for e in self.error_history if time.time() - e["timestamp"] < 3600]  # Last hour
            
            for error in self.error_history:
                error_type = error["error_type"]
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1
            
            return {
                "total_errors": len(self.error_history),
                "recent_errors": len(recent_errors),
                "error_types": error_types,
                "circuit_breakers": self.circuit_breakers.copy(),
                "registered_strategies": len(self.recovery_strategies)
            }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            try:
                # Attempt recovery if an exception occurred
                self.logger.info(f"Context manager attempting recovery for {exc_type.__name__}")
            except Exception as recovery_error:
                self.logger.error(f"Recovery attempt failed: {recovery_error}")
        
        self._executor.shutdown(wait=False)


def with_error_recovery(recovery_manager: Optional[ErrorRecoveryManager] = None):
    """Decorator to add error recovery to functions."""
    if recovery_manager is None:
        recovery_manager = ErrorRecoveryManager()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return recovery_manager.recover_from_error(e, func, *args, **kwargs)
        return wrapper
    return decorator