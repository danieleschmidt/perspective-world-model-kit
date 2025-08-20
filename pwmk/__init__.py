"""
Perspective World Model Kit (PWMK)

A neuro-symbolic world modeling framework with Theory of Mind capabilities
for multi-agent AI systems.
"""

__version__ = "0.1.0"
__author__ = "Your Organization"
__email__ = "pwmk@your-org.com"

# Conditional imports to handle missing dependencies
try:
    from .core import PerspectiveWorldModel, BeliefStore
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False
    PerspectiveWorldModel = None
    BeliefStore = None

try:
    from .planning import EpistemicPlanner
    _PLANNING_AVAILABLE = True
except ImportError:
    _PLANNING_AVAILABLE = False
    EpistemicPlanner = None

try:
    from .agents import ToMAgent
    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False
    ToMAgent = None

try:
    from .quantum import QuantumInspiredPlanner, QuantumCircuitOptimizer, QuantumAnnealingScheduler, AdaptiveQuantumAlgorithm
    _QUANTUM_AVAILABLE = True
except ImportError:
    _QUANTUM_AVAILABLE = False
    QuantumInspiredPlanner = None
    QuantumCircuitOptimizer = None
    QuantumAnnealingScheduler = None
    AdaptiveQuantumAlgorithm = None

try:
    from .autonomous import SelfImprovingAgent, create_self_improving_agent
    _AUTONOMOUS_AVAILABLE = True
except ImportError:
    _AUTONOMOUS_AVAILABLE = False
    SelfImprovingAgent = None
    create_self_improving_agent = None

try:
    from .breakthrough import EmergentIntelligenceSystem, create_emergent_intelligence_system
    _BREAKTHROUGH_AVAILABLE = True
except ImportError:
    _BREAKTHROUGH_AVAILABLE = False
    EmergentIntelligenceSystem = None
    create_emergent_intelligence_system = None

try:
    from .revolution import ConsciousnessEngine, ConsciousnessLevel, create_consciousness_engine
    _REVOLUTION_AVAILABLE = True
except ImportError:
    _REVOLUTION_AVAILABLE = False
    ConsciousnessEngine = None
    ConsciousnessLevel = None
    create_consciousness_engine = None

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