"""
Custom exceptions for sentiment analysis module.
"""


class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis operations."""
    pass


class ModelLoadError(SentimentAnalysisError):
    """Raised when sentiment model fails to load."""
    pass


class TokenizationError(SentimentAnalysisError):
    """Raised when text tokenization fails."""
    pass


class BeliefUpdateError(SentimentAnalysisError):
    """Raised when belief store update fails."""
    pass


class PerspectiveError(SentimentAnalysisError):
    """Raised when perspective analysis fails."""
    pass


class AgentNotFoundError(SentimentAnalysisError):
    """Raised when specified agent ID is not found."""
    pass


class InvalidSentimentScoreError(SentimentAnalysisError):
    """Raised when sentiment scores are invalid."""
    pass


class InsufficientDataError(SentimentAnalysisError):
    """Raised when insufficient data for analysis."""
    pass