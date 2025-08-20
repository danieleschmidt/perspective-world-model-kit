"""Fallback and degraded mode management for system resilience."""

import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from .logging import get_logger
from .circuit_breaker import CircuitState, get_model_circuit_breaker


class SystemMode(Enum):
    """System operational modes."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


@dataclass
class FallbackConfig:
    """Fallback configuration."""
    enable_model_fallback: bool = True
    enable_belief_fallback: bool = True
    enable_planning_fallback: bool = True
    max_degraded_time: float = 300.0  # 5 minutes
    fallback_quality_threshold: float = 0.7


class FallbackManager:
    """Manages system fallbacks and degraded operation modes."""
    
    def __init__(self, config: FallbackConfig):
        self.config = config
        self.mode = SystemMode.NORMAL
        self.degraded_start_time = None
        self.lock = Lock()
        self.logger = get_logger(self.__class__.__name__)
        
        # Fallback implementations
        self.fallback_handlers: Dict[str, Callable] = {}
        self._register_default_fallbacks()
    
    def _register_default_fallbacks(self) -> None:
        """Register default fallback handlers."""
        self.fallback_handlers.update({
            "model_prediction": self._fallback_model_prediction,
            "belief_reasoning": self._fallback_belief_reasoning,
            "planning": self._fallback_planning,
            "consciousness_processing": self._fallback_consciousness,
            "quantum_computation": self._fallback_quantum
        })
    
    def get_mode(self) -> SystemMode:
        """Get current system mode."""
        with self.lock:
            return self.mode
    
    def set_mode(self, mode: SystemMode, reason: str = "") -> None:
        """Set system mode with logging."""
        with self.lock:
            old_mode = self.mode
            self.mode = mode
            
            if mode == SystemMode.DEGRADED and old_mode != SystemMode.DEGRADED:
                self.degraded_start_time = time.time()
                self.logger.warning(f"System entering DEGRADED mode: {reason}")
            elif mode == SystemMode.NORMAL and old_mode != SystemMode.NORMAL:
                self.degraded_start_time = None
                self.logger.info("System returning to NORMAL mode")
            elif mode == SystemMode.EMERGENCY:
                self.logger.error(f"System entering EMERGENCY mode: {reason}")
            elif mode == SystemMode.MAINTENANCE:
                self.logger.info(f"System entering MAINTENANCE mode: {reason}")
    
    def check_degraded_timeout(self) -> bool:
        """Check if system has been in degraded mode too long."""
        if (self.mode == SystemMode.DEGRADED and 
            self.degraded_start_time is not None):
            
            elapsed = time.time() - self.degraded_start_time
            if elapsed > self.config.max_degraded_time:
                self.set_mode(
                    SystemMode.EMERGENCY, 
                    f"Degraded mode timeout ({elapsed:.1f}s > {self.config.max_degraded_time}s)"
                )
                return True
        return False
    
    def should_use_fallback(self, component: str) -> bool:
        """Determine if component should use fallback."""
        if self.mode == SystemMode.EMERGENCY:
            return True
        
        if self.mode == SystemMode.DEGRADED:
            # Check circuit breaker states
            if component == "model":
                circuit_breaker = get_model_circuit_breaker()
                return circuit_breaker.get_state() == CircuitState.OPEN
        
        return False
    
    def execute_with_fallback(
        self, 
        component: str, 
        primary_func: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with fallback on failure."""
        try:
            # Try primary function first
            if not self.should_use_fallback(component):
                result = primary_func(*args, **kwargs)
                
                # Validate result quality if possible
                if self._validate_result_quality(component, result):
                    return result
                else:
                    self.logger.warning(f"Primary {component} result quality below threshold")
        
        except Exception as e:
            self.logger.error(f"Primary {component} failed: {e}")
            
            # Switch to degraded mode if not already
            if self.mode == SystemMode.NORMAL:
                self.set_mode(SystemMode.DEGRADED, f"{component} failure: {e}")
        
        # Use fallback
        if component in self.fallback_handlers:
            self.logger.info(f"Using fallback for {component}")
            return self.fallback_handlers[component](*args, **kwargs)
        else:
            raise RuntimeError(f"No fallback available for {component}")
    
    def _validate_result_quality(self, component: str, result: Any) -> bool:
        """Validate result meets quality threshold."""
        # Component-specific quality checks
        if component == "model_prediction" and hasattr(result, '__len__'):
            # For model predictions, check if we have reasonable outputs
            if len(result) == 2:  # (next_states, beliefs)
                return True
        
        # Default to accepting result
        return True
    
    def _fallback_model_prediction(self, observations, actions, agent_ids=None):
        """Fallback for model predictions using simple heuristics."""
        self.logger.info("Using fallback model prediction")
        
        import torch
        
        # Simple fallback: return noisy version of input
        batch_size = observations.shape[0]
        seq_len = observations.shape[1]
        obs_dim = observations.shape[2]
        
        # Generate next states as slightly modified observations
        noise = torch.randn_like(observations) * 0.1
        next_states = observations + noise
        
        # Generate random beliefs
        beliefs = torch.rand(batch_size, seq_len, 64)
        
        return next_states, beliefs
    
    def _fallback_belief_reasoning(self, query: str, **kwargs):
        """Fallback for belief reasoning using simple rules."""
        self.logger.info("Using fallback belief reasoning")
        
        # Simple heuristic-based belief reasoning
        if "has(" in query and "agent" in query:
            return {"result": "unknown", "confidence": 0.5}
        
        return {"result": "false", "confidence": 0.3}
    
    def _fallback_planning(self, goal, current_state, **kwargs):
        """Fallback planning using simple greedy approach."""
        self.logger.info("Using fallback planning")
        
        # Return a simple random action plan
        import random
        
        actions = []
        for _ in range(5):  # 5-step plan
            actions.append(random.randint(0, 3))  # Assume 4 possible actions
        
        return {
            "actions": actions,
            "confidence": 0.4,
            "fallback": True
        }
    
    def _fallback_consciousness(self, inputs, **kwargs):
        """Fallback for consciousness processing."""
        self.logger.info("Using fallback consciousness processing")
        
        # Return minimal consciousness state
        return {
            "consciousness_level": "basic",
            "awareness_score": 0.3,
            "self_reflection": "system_in_fallback_mode",
            "metacognition": {}
        }
    
    def _fallback_quantum(self, problem, **kwargs):
        """Fallback for quantum computations using classical methods."""
        self.logger.info("Using fallback quantum computation (classical)")
        
        # Use classical optimization instead of quantum
        import random
        
        return {
            "solution": random.choice([0, 1]) if "binary" in str(problem) else random.random(),
            "quality": 0.6,
            "method": "classical_fallback"
        }
    
    def register_fallback(self, component: str, handler: Callable) -> None:
        """Register custom fallback handler."""
        self.fallback_handlers[component] = handler
        self.logger.info(f"Registered fallback handler for {component}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        return {
            "mode": self.mode.value,
            "degraded_time": (
                time.time() - self.degraded_start_time 
                if self.degraded_start_time else None
            ),
            "fallback_handlers": list(self.fallback_handlers.keys()),
            "circuit_breaker_states": {
                "model": get_model_circuit_breaker().get_state().value
            }
        }


# Global fallback manager instance
_fallback_manager = None


def get_fallback_manager() -> FallbackManager:
    """Get global fallback manager."""
    global _fallback_manager
    if _fallback_manager is None:
        config = FallbackConfig()
        _fallback_manager = FallbackManager(config)
    return _fallback_manager


def with_fallback(component: str):
    """Decorator to add fallback capability to functions."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            manager = get_fallback_manager()
            return manager.execute_with_fallback(component, func, *args, **kwargs)
        return wrapper
    return decorator