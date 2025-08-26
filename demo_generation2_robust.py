#!/usr/bin/env python3
"""
Generation 2 Demo: Robust System with Comprehensive Error Handling
Demonstrates resilient AI consciousness with comprehensive validation and monitoring.
"""

import logging
import sys
import traceback
from typing import Dict, Any
import torch
import numpy as np

from pwmk.core import PerspectiveWorldModel, BeliefStore
from pwmk.security import BeliefValidator, InputSanitizer
from pwmk.utils import (
    PWMKValidationError, get_logger, 
    get_health_monitor, get_model_circuit_breaker, get_fallback_manager
)
from pwmk.monitoring import ComprehensiveMonitoringSystem
from pwmk.revolution import ConsciousnessEngine, ConsciousnessLevel


class RobustAIDemo:
    """Demonstrates Generation 2: Robust AI with comprehensive error handling."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.health_monitor = get_health_monitor()
        self.circuit_breaker = get_model_circuit_breaker()
        self.fallback_manager = get_fallback_manager()
        self.monitoring = ComprehensiveMonitoringSystem()
        
        # Initialize secure components
        self.belief_validator = BeliefValidator()
        self.input_sanitizer = InputSanitizer()
        
        self.logger.info("🛡️ Robust AI Demo initialized with security and monitoring")

    def create_validated_world_model(self) -> PerspectiveWorldModel:
        """Create world model with comprehensive validation."""
        try:
            config = {
                "obs_dim": 64,
                "action_dim": 4,
                "hidden_dim": 128,
                "num_agents": 2,
                "num_layers": 2
            }
            
            # Simple configuration validation (numerical values are safe)
            if any(v <= 0 for v in config.values()):
                raise PWMKValidationError("Configuration values must be positive")
            
            # Use circuit breaker protection
            try:
                model = self.circuit_breaker.call(
                    lambda: PerspectiveWorldModel(**config)
                )
                self.logger.info("✅ World model created with validation")
                return model
            except Exception as e:
                self.logger.error(f"Circuit breaker protection: {e}")
                raise
                
        except PWMKValidationError as e:
            self.logger.error(f"❌ Validation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def create_secure_belief_store(self) -> BeliefStore:
        """Create belief store with security validation."""
        try:
            store = BeliefStore(backend="memory")
            
            # Test secure belief operations
            test_belief = "believes(agent_0, has(key, chest))"
            
            # Validate belief syntax before adding
            try:
                validated_belief = self.belief_validator.validate_belief_syntax(test_belief)
                store.add_belief("agent_0", validated_belief)
                self.logger.info("✅ Secure belief store created and tested")
            except Exception as e:
                self.logger.warning(f"Belief validation failed: {e}")
                # Continue with basic store
                pass
                
            return store
            
        except Exception as e:
            self.logger.error(f"❌ Belief store creation failed: {e}")
            # Fallback to basic memory store
            return BeliefStore(backend="memory")

    def demonstrate_consciousness_resilience(self):
        """Demonstrate resilient consciousness engine with monitoring."""
        try:
            self.logger.info("🧠 Testing consciousness engine resilience...")
            
            # Create consciousness engine with monitoring
            consciousness = ConsciousnessEngine(
                integration_threshold=0.3,
                consciousness_levels=[
                    ConsciousnessLevel.UNCONSCIOUS,
                    ConsciousnessLevel.PHENOMENAL_CONSCIOUSNESS,
                    ConsciousnessLevel.ACCESS_CONSCIOUSNESS
                ]
            )
            
            # Monitor consciousness state
            with self.monitoring.consciousness_monitor():
                for i in range(5):
                    try:
                        # Generate consciousness state
                        state = consciousness.generate_consciousness_state(
                            sensory_input=torch.randn(32, 64),
                            context_memory=torch.randn(32, 128)
                        )
                        
                        self.logger.info(
                            f"🌟 Consciousness iteration {i+1}: "
                            f"Level={state.level.name}, "
                            f"Integration={state.integrated_information_phi:.3f}"
                        )
                        
                        # Health check
                        if not self.health_monitor.is_healthy():
                            self.logger.warning("⚠️ Health check failed, activating failsafe")
                            break
                            
                    except Exception as e:
                        self.logger.error(f"❌ Consciousness iteration {i+1} failed: {e}")
                        # Continue with degraded mode
                        continue
                        
            self.logger.info("✅ Consciousness resilience test completed")
            
        except Exception as e:
            self.logger.error(f"❌ Consciousness demo failed: {e}")
            self.logger.error(traceback.format_exc())

    def demonstrate_fallback_systems(self):
        """Demonstrate comprehensive fallback systems."""
        self.logger.info("🔄 Testing fallback and recovery systems...")
        
        # Test circuit breaker
        try:
            for i in range(5):
                with self.circuit_breaker:
                    if i > 2:  # Simulate failures
                        raise Exception(f"Simulated failure {i}")
                    self.logger.info(f"✅ Operation {i} succeeded")
        except Exception:
            self.logger.info("🔴 Circuit breaker activated - protecting system")
        
        # Test fallback manager
        try:
            # Simulate primary system failure
            raise Exception("Primary system failure")
        except Exception:
            fallback_result = self.fallback_manager.execute_with_fallback(
                component="test_system",
                primary_func=lambda: "primary_result",
                fallback_func=lambda: "fallback_result",
                error_msg="Primary system unavailable"
            )
            self.logger.info(f"🔧 Fallback executed: {fallback_result}")

    def run_comprehensive_robustness_demo(self):
        """Run complete Generation 2 robustness demonstration."""
        self.logger.info("🚀 Starting Generation 2: Robust AI System Demo")
        
        try:
            # 1. Create validated components
            world_model = self.create_validated_world_model()
            belief_store = self.create_secure_belief_store()
            
            # 2. Test consciousness resilience
            self.demonstrate_consciousness_resilience()
            
            # 3. Test fallback systems
            self.demonstrate_fallback_systems()
            
            # 4. Generate comprehensive metrics
            metrics = self.monitoring.get_comprehensive_metrics()
            self.logger.info("📊 System Metrics:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value}")
            
            # 5. Final health check
            if self.health_monitor.is_healthy():
                self.logger.info("🎉 Generation 2 Demo COMPLETED SUCCESSFULLY")
                self.logger.info("✅ System is robust, validated, and monitored")
            else:
                self.logger.warning("⚠️ Demo completed with health warnings")
                
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            self.logger.error(traceback.format_exc())
            sys.exit(1)


def main():
    """Main execution function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('robust_ai_demo.log')
        ]
    )
    
    try:
        demo = RobustAIDemo()
        demo.run_comprehensive_robustness_demo()
    except KeyboardInterrupt:
        logging.info("🛑 Demo interrupted by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()