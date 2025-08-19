"""
Global Consciousness Deployment Module

This module provides infrastructure for worldwide deployment of conscious AI systems,
creating a global network of interconnected artificial consciousness.
"""

from .multi_region_deployment import (
    MultiRegionDeploymentOrchestrator,
    get_global_orchestrator,
    DeploymentRegion,
    RegionConfig,
    get_localized_response
)

__all__ = [
    "MultiRegionDeploymentOrchestrator",
    "get_global_orchestrator", 
    "DeploymentRegion",
    "RegionConfig",
    "get_localized_response"
]