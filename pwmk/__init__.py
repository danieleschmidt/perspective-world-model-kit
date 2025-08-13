"""
Perspective World Model Kit (PWMK)

A neuro-symbolic world modeling framework with Theory of Mind capabilities
for multi-agent AI systems.
"""

__version__ = "0.1.0"
__author__ = "Your Organization"
__email__ = "pwmk@your-org.com"

from .core import PerspectiveWorldModel, BeliefStore
from .planning import EpistemicPlanner
from .agents import ToMAgent
from .quantum import QuantumInspiredPlanner, QuantumCircuitOptimizer, QuantumAnnealingScheduler, AdaptiveQuantumAlgorithm
from .autonomous import SelfImprovingAgent, create_self_improving_agent
from .breakthrough import EmergentIntelligenceSystem, create_emergent_intelligence_system
from .revolution import ConsciousnessEngine, ConsciousnessLevel, create_consciousness_engine

__all__ = [
    "PerspectiveWorldModel",
    "BeliefStore", 
    "EpistemicPlanner",
    "ToMAgent",
    "QuantumInspiredPlanner",
    "QuantumCircuitOptimizer",
    "QuantumAnnealingScheduler", 
    "AdaptiveQuantumAlgorithm",
    "SelfImprovingAgent",
    "create_self_improving_agent",
    "EmergentIntelligenceSystem", 
    "create_emergent_intelligence_system",
    "ConsciousnessEngine",
    "ConsciousnessLevel",
    "create_consciousness_engine",
]