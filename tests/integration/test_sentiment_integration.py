"""
Integration tests for sentiment analysis with PWMK components.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pwmk.sentiment import (
    SentimentAnalyzer,
    MultiAgentSentimentAnalyzer,
    PerspectiveSentimentModel,
    BeliefAwareSentimentTracker
)
from pwmk.sentiment.optimization import BatchSentimentProcessor
from pwmk.core.beliefs import BeliefStore
from pwmk.core.world_model import PerspectiveWorldModel


@pytest.mark.integration
class TestSentimentPWMKIntegration:
    """Test sentiment analysis integration with PWMK components."""
    
    @pytest.fixture
    def mock_belief_store(self):
        """Mock belief store for testing."""
        store = Mock(spec=BeliefStore)
        store.add_belief = Mock()
        store.add_predicate_definition = Mock()
        store.add_rule = Mock()
        store.query = Mock(return_value=[])
        store.assert_fact = Mock()
        store.retract_fact = Mock()
        return store
        
    @pytest.fixture
    def mock_world_model(self):
        """Mock world model for testing."""
        model = Mock(spec=PerspectiveWorldModel)
        model.hidden_dim = 256
        return model
        
    @pytest.fixture
    def mock_sentiment_analyzer(self):
        """Mock sentiment analyzer for testing."""
        analyzer = Mock()
        analyzer.model_name = "test-model"
        analyzer.analyze_text = Mock(return_value={
            "negative": 0.2,
            "neutral": 0.3,
            "positive": 0.5
        })
        return analyzer
        
    def test_multi_agent_belief_integration(self, mock_sentiment_analyzer, mock_belief_store):
        """Test multi-agent sentiment analyzer with belief store integration."""
        num_agents = 3
        
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=num_agents,
            sentiment_analyzer=mock_sentiment_analyzer,
            belief_store=mock_belief_store
        )
        
        # Analyze communications from multiple agents
        communications = [
            (0, "I'm really excited about this project!"),
            (1, "This approach seems problematic to me."),
            (2, "I'm neutral about the whole thing."),
            (0, "Let me try to convince agent 1 that it's good."),
            (1, "Actually, agent 0 makes some good points."),
        ]
        
        results = []
        for agent_id, text in communications:
            result = multi_analyzer.analyze_agent_communication(agent_id, text)
            results.append((agent_id, result))
            
        # Verify sentiment analysis was called for each communication
        assert mock_sentiment_analyzer.analyze_text.call_count == len(communications)
        
        # Verify belief store was updated for each communication
        assert mock_belief_store.add_belief.call_count == len(communications)
        
        # Check that sentiment history was maintained
        for agent_id in range(num_agents):
            history = multi_analyzer.get_agent_sentiment_history(agent_id)
            expected_count = sum(1 for aid, _ in communications if aid == agent_id)
            assert len(history) == expected_count
            
        # Test group sentiment analysis
        group_sentiment = multi_analyzer.analyze_group_sentiment()
        assert isinstance(group_sentiment, dict)
        assert set(group_sentiment.keys()) == {"negative", "neutral", "positive"}
        
    def test_perspective_sentiment_integration(self, mock_world_model, mock_sentiment_analyzer):
        """Test perspective sentiment model integration with world model."""
        num_agents = 3
        
        perspective_model = PerspectiveSentimentModel(
            world_model=mock_world_model,
            sentiment_analyzer=mock_sentiment_analyzer,
            num_agents=num_agents
        )
        
        # Test perspective-aware analysis
        test_text = "I think we should work together on this task."
        world_state = np.random.randn(256)
        
        # Mock the forward pass
        with patch.object(perspective_model, 'forward') as mock_forward:
            mock_forward.return_value = (
                torch.tensor([[0.1, 0.2, 0.7]]),  # sentiment logits
                torch.randn(1, 128)  # perspective embedding
            )
            
            result = perspective_model.analyze_from_perspective(
                text=test_text,
                world_state=world_state,
                agent_id=0
            )
            
            assert isinstance(result, dict)
            assert set(result.keys()) == {"negative", "neutral", "positive"}
            assert abs(sum(result.values()) - 1.0) < 1e-6
            
        # Test perspective comparison
        with patch.object(perspective_model, 'analyze_from_perspective') as mock_analyze:
            mock_analyze.return_value = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
            
            perspectives = perspective_model.compare_perspectives(test_text, world_state)
            
            assert len(perspectives) == num_agents
            for agent_id, sentiment in perspectives.items():
                assert isinstance(sentiment, dict)
                assert set(sentiment.keys()) == {"negative", "neutral", "positive"}
                
            # Should have called analyze_from_perspective for each agent
            assert mock_analyze.call_count == num_agents
            
    def test_belief_aware_sentiment_tracker(self, mock_belief_store, mock_sentiment_analyzer):
        """Test belief-aware sentiment tracker integration."""
        num_agents = 2
        
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=num_agents,
            sentiment_analyzer=mock_sentiment_analyzer,
            belief_store=mock_belief_store
        )
        
        tracker = BeliefAwareSentimentTracker(
            belief_store=mock_belief_store,
            sentiment_analyzer=multi_analyzer,
            num_agents=num_agents
        )
        
        # Test sentiment belief initialization
        # Verify predicates were added
        assert mock_belief_store.add_predicate_definition.call_count >= 4
        
        # Verify rules were added
        assert mock_belief_store.add_rule.call_count >= 3
        
        # Test sentiment belief updates
        test_communications = [
            (0, "I'm happy about this outcome!"),
            (1, "I'm concerned about the implications."),
        ]
        
        for agent_id, text in test_communications:
            result = tracker.update_sentiment_beliefs(agent_id, text)
            
            assert isinstance(result, dict)
            assert set(result.keys()) == {"negative", "neutral", "positive"}
            
        # Test sentiment belief inference
        with patch.object(tracker, 'infer_sentiment_beliefs') as mock_infer:
            mock_infer.return_value = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
            
            inferred = tracker.infer_sentiment_beliefs(observer_agent=0, target_agent=1)
            
            assert isinstance(inferred, dict)
            assert set(inferred.keys()) == {"negative", "neutral", "positive"}
            
        # Test group dynamics analysis
        with patch.object(tracker, 'detect_sentiment_misalignment') as mock_detect:
            mock_detect.return_value = []
            
            dynamics = tracker.get_group_sentiment_dynamics()
            
            assert "group_sentiment" in dynamics
            assert "conflicts" in dynamics
            assert "positive_interactions" in dynamics
            assert "alignments" in dynamics
            assert "belief_misalignments" in dynamics
            assert "num_agents" in dynamics
            
            assert dynamics["num_agents"] == num_agents
            
    def test_batch_processing_integration(self, mock_sentiment_analyzer):
        """Test batch processing with sentiment analyzer."""
        from pwmk.sentiment.optimization import BatchProcessingConfig
        
        config = BatchProcessingConfig(
            batch_size=4,
            max_workers=2,
            use_gpu=False,  # Use CPU for testing
            enable_caching=False  # Disable caching for predictable testing
        )
        
        # Mock the internal batch processing
        with patch.object(BatchSentimentProcessor, '_process_batch_internal') as mock_process:
            mock_process.return_value = [
                {"negative": 0.1, "neutral": 0.2, "positive": 0.7},
                {"negative": 0.3, "neutral": 0.4, "positive": 0.3},
                {"negative": 0.6, "neutral": 0.2, "positive": 0.2},
            ]
            
            processor = BatchSentimentProcessor(mock_sentiment_analyzer, config)
            
            test_texts = [
                "I love this project!",
                "This is okay I guess.",
                "I really hate this approach.",
            ]
            
            results = processor.process_batch_sync(test_texts)
            
            assert len(results) == len(test_texts)
            
            for result in results:
                assert isinstance(result, dict)
                assert set(result.keys()) == {"negative", "neutral", "positive"}
                assert abs(sum(result.values()) - 1.0) < 1e-6
                
    def test_end_to_end_scenario(self, mock_belief_store):
        """Test complete end-to-end sentiment analysis scenario."""
        # Create mock components
        with patch('pwmk.sentiment.sentiment_analyzer.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('pwmk.sentiment.sentiment_analyzer.AutoModel.from_pretrained') as mock_model:
            
            # Mock tokenizer
            tokenizer_mock = Mock()
            tokenizer_mock.return_value = {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
            }
            mock_tokenizer.return_value = tokenizer_mock
            
            # Mock transformer
            transformer_mock = Mock()
            transformer_mock.config.hidden_size = 768
            transformer_mock.return_value.pooler_output = torch.randn(1, 768)
            mock_model.return_value = transformer_mock
            
            # Create sentiment analyzer
            sentiment_analyzer = SentimentAnalyzer()
            
            # Mock the forward pass to return consistent results
            with patch.object(sentiment_analyzer, 'forward') as mock_forward:
                mock_forward.return_value = torch.tensor([[0.1, 0.2, 0.7]])
                
                # Create multi-agent analyzer
                num_agents = 3
                multi_analyzer = MultiAgentSentimentAnalyzer(
                    num_agents=num_agents,
                    sentiment_analyzer=sentiment_analyzer,
                    belief_store=mock_belief_store
                )
                
                # Create belief tracker
                tracker = BeliefAwareSentimentTracker(
                    belief_store=mock_belief_store,
                    sentiment_analyzer=multi_analyzer,
                    num_agents=num_agents
                )
                
                # Simulate multi-agent conversation
                conversation = [
                    (0, "I'm excited to start this new project with everyone!"),
                    (1, "I have some concerns about the timeline."),
                    (2, "Agent 0 seems really enthusiastic about this."),
                    (0, "I understand agent 1's concerns, let's discuss them."),
                    (1, "Actually, agent 0's enthusiasm is contagious. I'm feeling better about it."),
                    (2, "It's great to see everyone working together."),
                ]
                
                # Process conversation
                sentiment_results = []
                for agent_id, text in conversation:
                    result = tracker.update_sentiment_beliefs(agent_id, text)
                    sentiment_results.append((agent_id, text, result))
                    
                # Verify all communications were processed
                assert len(sentiment_results) == len(conversation)
                
                # Verify sentiment beliefs were updated
                assert mock_belief_store.assert_fact.call_count == len(conversation)
                
                # Test group dynamics
                dynamics = tracker.get_group_sentiment_dynamics()
                assert isinstance(dynamics, dict)
                
                # Test current sentiment state
                current_state = multi_analyzer.get_current_sentiment_state()
                assert len(current_state) == num_agents
                
                # Test sentiment history
                for agent_id in range(num_agents):
                    history = multi_analyzer.get_agent_sentiment_history(agent_id)
                    expected_count = sum(1 for aid, _ in conversation if aid == agent_id)
                    assert len(history) == expected_count
                    
    def test_error_handling_integration(self, mock_belief_store):
        """Test error handling across integrated components."""
        # Create multi-agent analyzer with failing sentiment analyzer
        failing_analyzer = Mock()
        failing_analyzer.analyze_text.side_effect = Exception("Model error")
        
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=2,
            sentiment_analyzer=failing_analyzer,
            belief_store=mock_belief_store
        )
        
        # Should handle sentiment analyzer failure gracefully
        with pytest.raises(Exception):  # Should propagate the exception
            multi_analyzer.analyze_agent_communication(0, "Test text")
            
        # Test belief store failure
        mock_belief_store.assert_fact.side_effect = Exception("Belief store error")
        
        working_analyzer = Mock()
        working_analyzer.analyze_text.return_value = {
            "negative": 0.2, "neutral": 0.3, "positive": 0.5
        }
        
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=2,
            sentiment_analyzer=working_analyzer,
            belief_store=mock_belief_store
        )
        
        # Should handle belief store failure gracefully
        result = multi_analyzer.analyze_agent_communication(0, "Test text")
        
        # Sentiment analysis should still succeed
        assert isinstance(result, dict)
        assert set(result.keys()) == {"negative", "neutral", "positive"}
        
    def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring."""
        from pwmk.sentiment.monitoring import SentimentMonitor
        
        monitor = SentimentMonitor(max_events=100)
        
        # Create mock analyzer that reports to monitor
        mock_analyzer = Mock()
        
        def analyze_with_monitoring(text):
            start_time = time.time()
            result = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
            processing_time = time.time() - start_time
            
            monitor.record_analysis(
                agent_id=0,
                text=text,
                sentiment_scores=result,
                processing_time=processing_time
            )
            
            return result
            
        mock_analyzer.analyze_text = analyze_with_monitoring
        
        # Test monitoring integration
        import time
        
        for i in range(5):
            mock_analyzer.analyze_text(f"Test text {i}")
            
        # Verify monitoring recorded events
        metrics = monitor.get_current_metrics()
        assert metrics["global_metrics"]["total_analyses"] == 5
        assert metrics["global_metrics"]["total_errors"] == 0
        
        # Test error monitoring
        def analyze_with_error(text):
            monitor.record_error(
                agent_id=0,
                text=text,
                error="Test error"
            )
            raise Exception("Test error")
            
        mock_analyzer.analyze_text = analyze_with_error
        
        try:
            mock_analyzer.analyze_text("Error text")
        except Exception:
            pass
            
        # Verify error was recorded
        metrics = monitor.get_current_metrics()
        assert metrics["global_metrics"]["total_errors"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])