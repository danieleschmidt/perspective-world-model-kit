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

__all__ = [
    "PerspectiveWorldModel",
    "BeliefStore", 
    "EpistemicPlanner",
    "ToMAgent",
]