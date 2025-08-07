"""
Sentiment Analysis Module for Perspective World Model Kit

Multi-agent sentiment analysis with Theory of Mind capabilities.
"""

from .sentiment_analyzer import SentimentAnalyzer, MultiAgentSentimentAnalyzer
from .perspective_sentiment import PerspectiveSentimentModel
from .belief_sentiment import BeliefAwareSentimentTracker
from .metrics import SentimentMetrics, MultiAgentSentimentMetrics
from .monitoring import SentimentMonitor
from .i18n import MultilingualSentimentAnalyzer, LocalizationManager
from .compliance import PrivacyController, DataCategory, ProcessingPurpose, LegalBasis
from .regional import RegionalSentimentAnalyzer, Region, RegionalConfigManager
from .exceptions import (
    SentimentAnalysisError,
    ModelLoadError,
    TokenizationError,
    BeliefUpdateError,
    PerspectiveError,
    AgentNotFoundError
)

__all__ = [
    "SentimentAnalyzer",
    "MultiAgentSentimentAnalyzer", 
    "PerspectiveSentimentModel",
    "BeliefAwareSentimentTracker",
    "SentimentMetrics",
    "MultiAgentSentimentMetrics",
    "SentimentMonitor",
    "MultilingualSentimentAnalyzer",
    "LocalizationManager",
    "PrivacyController",
    "DataCategory",
    "ProcessingPurpose", 
    "LegalBasis",
    "RegionalSentimentAnalyzer",
    "Region",
    "RegionalConfigManager",
    "SentimentAnalysisError",
    "ModelLoadError",
    "TokenizationError",
    "BeliefUpdateError",
    "PerspectiveError",
    "AgentNotFoundError",
]