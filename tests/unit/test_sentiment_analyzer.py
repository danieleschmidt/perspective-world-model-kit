"""
Unit tests for sentiment analysis components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pwmk.sentiment import (
    SentimentAnalyzer,
    MultiAgentSentimentAnalyzer,
    SentimentAnalysisError,
    ModelLoadError,
    TokenizationError,
    AgentNotFoundError
)
from pwmk.sentiment.validation import validate_sentiment_scores
from pwmk.core.beliefs import BeliefStore


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        return tokenizer
        
    @pytest.fixture
    def mock_transformer(self):
        """Mock transformer model for testing."""
        transformer = Mock()
        transformer.config.hidden_size = 768
        transformer.return_value.pooler_output = torch.randn(1, 768)
        return transformer
        
    @pytest.fixture
    def sentiment_analyzer(self, mock_tokenizer, mock_transformer):
        """Create SentimentAnalyzer with mocked dependencies."""
        with patch('pwmk.sentiment.sentiment_analyzer.AutoTokenizer.from_pretrained') as mock_tok, \
             patch('pwmk.sentiment.sentiment_analyzer.AutoModel.from_pretrained') as mock_model:
            
            mock_tok.return_value = mock_tokenizer
            mock_model.return_value = mock_transformer
            
            return SentimentAnalyzer()
            
    def test_initialization_success(self, sentiment_analyzer):
        """Test successful initialization."""
        assert sentiment_analyzer.model_name == "bert-base-uncased"
        assert sentiment_analyzer.num_classes == 3
        
    def test_initialization_failure(self):
        """Test initialization failure handling."""
        with patch('pwmk.sentiment.sentiment_analyzer.AutoTokenizer.from_pretrained') as mock_tok:
            mock_tok.side_effect = Exception("Model not found")
            
            with pytest.raises(ModelLoadError):
                SentimentAnalyzer()
                
    def test_analyze_text_success(self, sentiment_analyzer):
        """Test successful text analysis."""
        # Mock forward pass
        with patch.object(sentiment_analyzer, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([[0.1, 0.2, 0.7]])
            
            result = sentiment_analyzer.analyze_text("I love this!")
            
            assert isinstance(result, dict)
            assert set(result.keys()) == {"negative", "neutral", "positive"}
            assert abs(sum(result.values()) - 1.0) < 1e-6
            assert result["positive"] > result["negative"]
            
    def test_analyze_text_invalid_input(self, sentiment_analyzer):
        """Test analysis with invalid input."""
        with pytest.raises(ValueError):
            sentiment_analyzer.analyze_text("")
            
        with pytest.raises(ValueError):
            sentiment_analyzer.analyze_text("   ")
            
    def test_analyze_text_model_error(self, sentiment_analyzer):
        """Test analysis with model error."""
        with patch.object(sentiment_analyzer, 'forward') as mock_forward:
            mock_forward.side_effect = Exception("Model error")
            
            with pytest.raises(TokenizationError):
                sentiment_analyzer.analyze_text("test text")
                
    def test_forward_pass(self, sentiment_analyzer):
        """Test forward pass through model."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        
        with patch.object(sentiment_analyzer, 'transformer') as mock_transformer, \
             patch.object(sentiment_analyzer, 'sentiment_head') as mock_head:
            
            mock_transformer.return_value.pooler_output = torch.randn(2, 768)
            mock_head.return_value = torch.randn(2, 3)
            
            result = sentiment_analyzer.forward(input_ids, attention_mask)
            
            assert result.shape == (2, 3)


class TestMultiAgentSentimentAnalyzer:
    """Test cases for MultiAgentSentimentAnalyzer."""
    
    @pytest.fixture
    def mock_sentiment_analyzer(self):
        """Mock sentiment analyzer."""
        analyzer = Mock()
        analyzer.analyze_text.return_value = {
            "negative": 0.1,
            "neutral": 0.3,
            "positive": 0.6
        }
        return analyzer
        
    @pytest.fixture
    def mock_belief_store(self):
        """Mock belief store."""
        store = Mock(spec=BeliefStore)
        store.add_belief = Mock()
        return store
        
    @pytest.fixture
    def multi_agent_analyzer(self, mock_sentiment_analyzer, mock_belief_store):
        """Create MultiAgentSentimentAnalyzer with mocked dependencies."""
        return MultiAgentSentimentAnalyzer(
            num_agents=3,
            sentiment_analyzer=mock_sentiment_analyzer,
            belief_store=mock_belief_store
        )
        
    def test_initialization(self, multi_agent_analyzer):
        """Test proper initialization."""
        assert multi_agent_analyzer.num_agents == 3
        assert len(multi_agent_analyzer.agent_sentiment_history) == 3
        
    def test_analyze_agent_communication_success(self, multi_agent_analyzer):
        """Test successful agent communication analysis."""
        result = multi_agent_analyzer.analyze_agent_communication(
            agent_id=0,
            text="I'm happy about this!"
        )
        
        assert isinstance(result, dict)
        validate_sentiment_scores(result)
        
        # Check history was updated
        history = multi_agent_analyzer.get_agent_sentiment_history(0)
        assert len(history) == 1
        assert history[0]["text"] == "I'm happy about this!"
        assert history[0]["sentiment"] == result
        
    def test_analyze_agent_communication_invalid_agent(self, multi_agent_analyzer):
        """Test analysis with invalid agent ID."""
        with pytest.raises(AgentNotFoundError):
            multi_agent_analyzer.analyze_agent_communication(
                agent_id=10,  # Invalid agent ID
                text="test"
            )
            
    def test_analyze_agent_communication_invalid_text(self, multi_agent_analyzer):
        """Test analysis with invalid text."""
        with pytest.raises(ValueError):
            multi_agent_analyzer.analyze_agent_communication(
                agent_id=0,
                text=""  # Empty text
            )
            
    def test_get_current_sentiment_state(self, multi_agent_analyzer):
        """Test getting current sentiment state."""
        # Add some history
        multi_agent_analyzer.analyze_agent_communication(0, "I'm happy!")
        multi_agent_analyzer.analyze_agent_communication(1, "This is okay.")
        
        state = multi_agent_analyzer.get_current_sentiment_state()
        
        assert isinstance(state, dict)
        assert len(state) == 3  # All agents should have state
        
        for agent_id, sentiment in state.items():
            validate_sentiment_scores(sentiment)
            
    def test_analyze_group_sentiment(self, multi_agent_analyzer):
        """Test group sentiment analysis."""
        # Add communications for multiple agents
        multi_agent_analyzer.analyze_agent_communication(0, "I love this!")
        multi_agent_analyzer.analyze_agent_communication(1, "This is terrible!")
        multi_agent_analyzer.analyze_agent_communication(2, "It's okay I guess.")
        
        group_sentiment = multi_agent_analyzer.analyze_group_sentiment()
        
        validate_sentiment_scores(group_sentiment)
        
    def test_predict_agent_sentiment_response(self, multi_agent_analyzer):
        """Test predicting agent sentiment response."""
        # Build some history
        multi_agent_analyzer.analyze_agent_communication(0, "I'm excited!")
        multi_agent_analyzer.analyze_agent_communication(0, "This is great!")
        
        prediction = multi_agent_analyzer.predict_agent_sentiment_response(
            target_agent=0,
            stimulus_text="How do you feel about the new project?"
        )
        
        validate_sentiment_scores(prediction)
        
    def test_history_limit(self, multi_agent_analyzer):
        """Test that history is limited to prevent memory issues."""
        # Add many communications
        for i in range(1100):  # More than the limit
            multi_agent_analyzer.analyze_agent_communication(
                0, f"Message {i}"
            )
            
        history = multi_agent_analyzer.get_agent_sentiment_history(0)
        assert len(history) <= 1000  # Should be limited
        
    def test_belief_store_error_handling(self, multi_agent_analyzer, mock_belief_store):
        """Test handling of belief store errors."""
        mock_belief_store.add_belief.side_effect = Exception("Belief store error")
        
        # Should still work even if belief store fails
        result = multi_agent_analyzer.analyze_agent_communication(
            agent_id=0,
            text="Test message"
        )
        
        validate_sentiment_scores(result)
        
    def test_empty_history_handling(self, multi_agent_analyzer):
        """Test handling of empty history."""
        # No communications yet
        state = multi_agent_analyzer.get_current_sentiment_state()
        
        # Should return default sentiment for all agents
        for agent_sentiment in state.values():
            assert abs(sum(agent_sentiment.values()) - 1.0) < 1e-6
            
        group_sentiment = multi_agent_analyzer.analyze_group_sentiment()
        assert abs(sum(group_sentiment.values()) - 1.0) < 1e-6


class TestSentimentValidation:
    """Test sentiment validation functions."""
    
    def test_valid_sentiment_scores(self):
        """Test validation of valid sentiment scores."""
        valid_scores = {
            "negative": 0.2,
            "neutral": 0.3,
            "positive": 0.5
        }
        
        # Should not raise exception
        validate_sentiment_scores(valid_scores)
        
    def test_invalid_sentiment_scores(self):
        """Test validation of invalid sentiment scores."""
        from pwmk.sentiment.exceptions import InvalidSentimentScoreError
        
        # Missing key
        with pytest.raises(InvalidSentimentScoreError):
            validate_sentiment_scores({"negative": 0.5, "positive": 0.5})
            
        # Extra key
        with pytest.raises(InvalidSentimentScoreError):
            validate_sentiment_scores({
                "negative": 0.2,
                "neutral": 0.3,
                "positive": 0.5,
                "extra": 0.0
            })
            
        # Values don't sum to 1
        with pytest.raises(InvalidSentimentScoreError):
            validate_sentiment_scores({
                "negative": 0.2,
                "neutral": 0.3,
                "positive": 0.8  # Sum = 1.3
            })
            
        # Negative value
        with pytest.raises(InvalidSentimentScoreError):
            validate_sentiment_scores({
                "negative": -0.1,
                "neutral": 0.3,
                "positive": 0.8
            })
            
        # Value > 1
        with pytest.raises(InvalidSentimentScoreError):
            validate_sentiment_scores({
                "negative": 0.2,
                "neutral": 0.3,
                "positive": 1.5
            })
            
    def test_sentiment_scores_tolerance(self):
        """Test tolerance for floating point errors."""
        # Slightly off due to floating point arithmetic
        slightly_off_scores = {
            "negative": 0.33333333333333333,
            "neutral": 0.33333333333333333,
            "positive": 0.33333333333333334
        }
        
        # Should pass validation (within tolerance)
        validate_sentiment_scores(slightly_off_scores)


@pytest.mark.integration
class TestSentimentIntegration:
    """Integration tests for sentiment analysis components."""
    
    def test_full_pipeline(self):
        """Test complete sentiment analysis pipeline."""
        # This would test with actual models if available
        pytest.skip("Requires actual model files - skipping in unit tests")
        
    def test_belief_store_integration(self):
        """Test integration with belief store."""
        # Mock belief store for integration testing
        belief_store = Mock(spec=BeliefStore)
        belief_store.add_belief = Mock()
        belief_store.query = Mock(return_value=[])
        
        analyzer = Mock()
        analyzer.analyze_text.return_value = {
            "negative": 0.1,
            "neutral": 0.2,
            "positive": 0.7
        }
        
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=2,
            sentiment_analyzer=analyzer,
            belief_store=belief_store
        )
        
        # Analyze some communications
        multi_analyzer.analyze_agent_communication(0, "I'm happy!")
        multi_analyzer.analyze_agent_communication(1, "This is great!")
        
        # Verify belief store was called
        assert belief_store.add_belief.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])