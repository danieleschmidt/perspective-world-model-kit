"""
Unit tests for sentiment analysis metrics and monitoring.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch
from pwmk.sentiment.metrics import SentimentMetrics, MultiAgentSentimentMetrics
from pwmk.sentiment.monitoring import SentimentMonitor, SentimentEvent


class TestSentimentMetrics:
    """Test cases for SentimentMetrics."""
    
    @pytest.fixture
    def metrics(self):
        """Create SentimentMetrics instance."""
        return SentimentMetrics()
        
    def test_initialization(self, metrics):
        """Test proper initialization."""
        assert metrics.sentiment_labels == ["negative", "neutral", "positive"]
        assert len(metrics.predictions) == 0
        assert len(metrics.ground_truth) == 0
        
    def test_add_valid_prediction(self, metrics):
        """Test adding valid predictions."""
        predicted = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        true_sentiment = "positive"
        
        metrics.add_prediction(predicted, true_sentiment, confidence=0.8)
        
        assert len(metrics.predictions) == 1
        assert len(metrics.ground_truth) == 1
        assert len(metrics.confidence_scores) == 1
        assert metrics.predictions[0] == "positive"
        assert metrics.ground_truth[0] == "positive"
        assert metrics.confidence_scores[0] == 0.8
        
    def test_add_prediction_auto_confidence(self, metrics):
        """Test adding prediction with automatic confidence calculation."""
        predicted = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        
        metrics.add_prediction(predicted, "positive")
        
        assert metrics.confidence_scores[0] == 0.7  # Max probability
        
    def test_add_invalid_prediction(self, metrics):
        """Test handling of invalid predictions."""
        # Invalid sentiment scores
        invalid_predicted = {"negative": 0.1, "positive": 0.5}  # Missing neutral
        
        # Should handle gracefully (logged as warning)
        metrics.add_prediction(invalid_predicted, "positive")
        
        # Should not add invalid prediction
        assert len(metrics.predictions) == 0
        
    def test_calculate_accuracy_empty(self, metrics):
        """Test accuracy calculation with no predictions."""
        assert metrics.calculate_accuracy() == 0.0
        
    def test_calculate_accuracy_with_predictions(self, metrics):
        """Test accuracy calculation with predictions."""
        # Add some predictions
        metrics.add_prediction({"negative": 0.1, "neutral": 0.2, "positive": 0.7}, "positive")
        metrics.add_prediction({"negative": 0.8, "neutral": 0.1, "positive": 0.1}, "negative")
        metrics.add_prediction({"negative": 0.2, "neutral": 0.7, "positive": 0.1}, "neutral")
        metrics.add_prediction({"negative": 0.1, "neutral": 0.2, "positive": 0.7}, "negative")  # Wrong
        
        accuracy = metrics.calculate_accuracy()
        assert accuracy == 0.75  # 3 out of 4 correct
        
    def test_precision_recall_f1(self, metrics):
        """Test precision, recall, and F1 calculation."""
        # Add predictions with known outcomes
        predictions = [
            ({"negative": 0.8, "neutral": 0.1, "positive": 0.1}, "negative"),
            ({"negative": 0.8, "neutral": 0.1, "positive": 0.1}, "negative"),
            ({"negative": 0.1, "neutral": 0.2, "positive": 0.7}, "positive"),
            ({"negative": 0.1, "neutral": 0.7, "positive": 0.2}, "neutral"),
            ({"negative": 0.8, "neutral": 0.1, "positive": 0.1}, "positive"),  # FN for positive
        ]
        
        for pred, true in predictions:
            metrics.add_prediction(pred, true)
            
        results = metrics.calculate_precision_recall_f1()
        
        assert "negative" in results
        assert "neutral" in results
        assert "positive" in results
        
        for sentiment_data in results.values():
            assert "precision" in sentiment_data
            assert "recall" in sentiment_data
            assert "f1" in sentiment_data
            assert "support" in sentiment_data
            
    def test_macro_metrics(self, metrics):
        """Test macro-averaged metrics."""
        # Add some predictions
        for i in range(10):
            sentiment_idx = i % 3
            sentiment = ["negative", "neutral", "positive"][sentiment_idx]
            predicted = {"negative": 0.33, "neutral": 0.33, "positive": 0.34}
            predicted[sentiment] = 0.7
            predicted[list(predicted.keys())[0]] = 0.15
            predicted[list(predicted.keys())[1]] = 0.15
            
            metrics.add_prediction(predicted, sentiment)
            
        macro_metrics = metrics.calculate_macro_metrics()
        
        assert "macro_precision" in macro_metrics
        assert "macro_recall" in macro_metrics
        assert "macro_f1" in macro_metrics
        assert 0.0 <= macro_metrics["macro_precision"] <= 1.0
        
    def test_confusion_matrix(self, metrics):
        """Test confusion matrix calculation."""
        predictions = [
            ("negative", "negative"),
            ("negative", "negative"),
            ("positive", "positive"),
            ("neutral", "neutral"),
            ("negative", "positive"),  # Misclassification
        ]
        
        for pred_label, true_label in predictions:
            pred_scores = {"negative": 0.33, "neutral": 0.33, "positive": 0.34}
            pred_scores[pred_label] = 0.7
            metrics.add_prediction(pred_scores, true_label)
            
        cm = metrics.calculate_confusion_matrix()
        
        assert cm.shape == (3, 3)  # 3x3 matrix for 3 classes
        assert cm.sum() == 5  # Total predictions
        
    def test_confidence_metrics(self, metrics):
        """Test confidence-related metrics."""
        confidences = [0.9, 0.7, 0.6, 0.8, 0.5]
        
        for i, conf in enumerate(confidences):
            sentiment = ["negative", "neutral", "positive"][i % 3]
            pred = {"negative": 0.33, "neutral": 0.33, "positive": 0.34}
            metrics.add_prediction(pred, sentiment, confidence=conf)
            
        conf_metrics = metrics.calculate_confidence_metrics()
        
        assert "avg_confidence" in conf_metrics
        assert "confidence_std" in conf_metrics
        assert "min_confidence" in conf_metrics
        assert "max_confidence" in conf_metrics
        
        assert conf_metrics["avg_confidence"] == np.mean(confidences)
        assert conf_metrics["min_confidence"] == min(confidences)
        assert conf_metrics["max_confidence"] == max(confidences)
        
    def test_comprehensive_report(self, metrics):
        """Test comprehensive report generation."""
        # Add some data
        for i in range(5):
            sentiment = ["negative", "neutral", "positive"][i % 3]
            pred = {"negative": 0.33, "neutral": 0.33, "positive": 0.34}
            pred[sentiment] = 0.8
            metrics.add_prediction(pred, sentiment)
            
        report = metrics.get_comprehensive_report()
        
        assert "total_samples" in report
        assert "accuracy" in report
        assert "per_class_metrics" in report
        assert "macro_metrics" in report
        assert "confidence_metrics" in report
        assert "confusion_matrix" in report
        
        assert report["total_samples"] == 5
        
    def test_reset(self, metrics):
        """Test metrics reset functionality."""
        # Add some data
        metrics.add_prediction({"negative": 0.1, "neutral": 0.2, "positive": 0.7}, "positive")
        
        assert len(metrics.predictions) > 0
        
        metrics.reset()
        
        assert len(metrics.predictions) == 0
        assert len(metrics.ground_truth) == 0
        assert len(metrics.confidence_scores) == 0


class TestMultiAgentSentimentMetrics:
    """Test cases for MultiAgentSentimentMetrics."""
    
    @pytest.fixture
    def multi_metrics(self):
        """Create MultiAgentSentimentMetrics instance."""
        return MultiAgentSentimentMetrics(num_agents=3)
        
    def test_initialization(self, multi_metrics):
        """Test proper initialization."""
        assert multi_metrics.num_agents == 3
        assert len(multi_metrics.agent_metrics) == 3
        assert len(multi_metrics.interaction_data) == 0
        
    def test_add_agent_prediction(self, multi_metrics):
        """Test adding predictions for specific agents."""
        predicted = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        
        multi_metrics.add_agent_prediction(0, predicted, "positive", confidence=0.8)
        multi_metrics.add_agent_prediction(1, predicted, "positive", confidence=0.7)
        
        # Check that predictions were added to correct agents
        agent0_metrics = multi_metrics.agent_metrics[0]
        agent1_metrics = multi_metrics.agent_metrics[1]
        agent2_metrics = multi_metrics.agent_metrics[2]
        
        assert len(agent0_metrics.predictions) == 1
        assert len(agent1_metrics.predictions) == 1
        assert len(agent2_metrics.predictions) == 0  # No predictions for agent 2
        
    def test_add_invalid_agent_prediction(self, multi_metrics):
        """Test handling of invalid agent IDs."""
        predicted = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        
        # Should handle gracefully
        multi_metrics.add_agent_prediction(10, predicted, "positive")  # Invalid agent ID
        
        # No metrics should be affected
        for metrics in multi_metrics.agent_metrics.values():
            assert len(metrics.predictions) == 0
            
    def test_add_interaction_data(self, multi_metrics):
        """Test adding interaction data."""
        agent_pairs = [(0, 1), (1, 2), (0, 2)]
        alignments = [True, False, True]
        interaction_types = ["cooperation", "conflict", "cooperation"]
        
        multi_metrics.add_interaction_data(agent_pairs, alignments, interaction_types)
        
        assert len(multi_metrics.interaction_data) == 3
        
        # Check first interaction
        first_interaction = multi_metrics.interaction_data[0]
        assert first_interaction["agent1"] == 0
        assert first_interaction["agent2"] == 1
        assert first_interaction["sentiment_aligned"] == True
        assert first_interaction["interaction_type"] == "cooperation"
        
    def test_calculate_agent_performance(self, multi_metrics):
        """Test calculating performance for specific agent."""
        # Add some predictions for agent 0
        for i in range(5):
            predicted = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
            multi_metrics.add_agent_prediction(0, predicted, "positive")
            
        performance = multi_metrics.calculate_agent_performance(0)
        
        assert "total_samples" in performance
        assert "accuracy" in performance
        assert performance["total_samples"] == 5
        
        # Agent with no data
        empty_performance = multi_metrics.calculate_agent_performance(1)
        assert empty_performance == {}
        
    def test_calculate_group_metrics(self, multi_metrics):
        """Test calculating group-level metrics."""
        # Add predictions for multiple agents
        for agent_id in range(3):
            for _ in range(3):
                predicted = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
                multi_metrics.add_agent_prediction(agent_id, predicted, "positive")
                
        # Add interaction data
        multi_metrics.add_interaction_data(
            [(0, 1), (1, 2)],
            [True, False],
            ["cooperation", "conflict"]
        )
        
        group_metrics = multi_metrics.calculate_group_metrics()
        
        assert "group_accuracy" in group_metrics
        assert "sentiment_alignment_rate" in group_metrics
        assert "total_interactions" in group_metrics
        assert "interaction_type_distribution" in group_metrics
        assert "num_agents" in group_metrics
        
        assert group_metrics["num_agents"] == 3
        assert group_metrics["total_interactions"] == 2
        assert group_metrics["sentiment_alignment_rate"] == 0.5  # 1 out of 2 aligned
        
    def test_get_agent_comparison(self, multi_metrics):
        """Test agent performance comparison."""
        # Add different performance data for agents
        for agent_id in range(3):
            accuracy = 0.8 + (agent_id * 0.1)  # Different accuracies
            
            for i in range(5):
                if i < int(5 * accuracy):
                    # Correct prediction
                    predicted = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
                    true_label = "positive"
                else:
                    # Incorrect prediction
                    predicted = {"negative": 0.7, "neutral": 0.2, "positive": 0.1}
                    true_label = "positive"
                    
                multi_metrics.add_agent_prediction(agent_id, predicted, true_label)
                
        comparison = multi_metrics.get_agent_comparison()
        
        assert "accuracies" in comparison
        assert "avg_confidences" in comparison
        assert "macro_f1_scores" in comparison
        
        assert len(comparison["accuracies"]) == 3
        
        # Check that accuracies are different (reflecting our setup)
        accuracies = comparison["accuracies"]
        assert accuracies[0] != accuracies[1] or accuracies[1] != accuracies[2]
        
    def test_generate_detailed_report(self, multi_metrics):
        """Test detailed report generation."""
        # Add some data
        for agent_id in range(2):  # Only 2 agents for this test
            predicted = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
            multi_metrics.add_agent_prediction(agent_id, predicted, "positive")
            
        multi_metrics.add_interaction_data([(0, 1)], [True], ["cooperation"])
        
        report = multi_metrics.generate_detailed_report()
        
        assert "group_metrics" in report
        assert "agent_comparison" in report
        assert "individual_agent_reports" in report
        
        # Check individual reports exist for all agents
        individual_reports = report["individual_agent_reports"]
        assert "agent_0" in individual_reports
        assert "agent_1" in individual_reports
        assert "agent_2" in individual_reports


class TestSentimentMonitor:
    """Test cases for SentimentMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create SentimentMonitor instance."""
        return SentimentMonitor(max_events=100, metric_window_minutes=1)
        
    def test_initialization(self, monitor):
        """Test proper initialization."""
        assert len(monitor.events) == 0
        assert monitor.metrics["total_analyses"] == 0
        assert monitor.metrics["total_errors"] == 0
        
    def test_record_analysis(self, monitor):
        """Test recording analysis events."""
        sentiment_scores = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        
        monitor.record_analysis(
            agent_id=0,
            text="Test text",
            sentiment_scores=sentiment_scores,
            processing_time=0.5
        )
        
        assert len(monitor.events) == 1
        assert monitor.metrics["total_analyses"] == 1
        assert monitor.metrics["avg_processing_time"] == 0.5
        
        # Check agent metrics
        agent_metrics = monitor.agent_metrics[0]
        assert agent_metrics["total_analyses"] == 1
        assert agent_metrics["avg_sentiment_scores"]["positive"] == 0.7
        
    def test_record_error(self, monitor):
        """Test recording error events."""
        monitor.record_error(
            agent_id=1,
            text="Test text",
            error="Model failed",
            processing_time=0.3
        )
        
        assert len(monitor.events) == 1
        assert monitor.metrics["total_errors"] == 1
        assert monitor.error_counts["Model failed"] == 1
        assert len(monitor.recent_errors) == 1
        
    def test_record_belief_update(self, monitor):
        """Test recording belief update events."""
        monitor.record_belief_update(
            agent_id=0,
            belief_query="believes(agent_0, positive)",
            processing_time=0.1,
            success=True
        )
        
        assert len(monitor.events) == 1
        
        event = monitor.events[0]
        assert event.event_type == "belief_update"
        assert event.agent_id == 0
        assert event.error is None
        
    def test_get_current_metrics(self, monitor):
        """Test getting current metrics."""
        # Add some events
        sentiment_scores = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        monitor.record_analysis(0, "text1", sentiment_scores, 0.5)
        monitor.record_analysis(1, "text2", sentiment_scores, 0.3)
        monitor.record_error(0, "text3", "Error occurred", 0.1)
        
        metrics = monitor.get_current_metrics()
        
        assert "global_metrics" in metrics
        assert "agent_metrics" in metrics
        assert "error_counts" in metrics
        assert "total_events" in metrics
        assert "last_update" in metrics
        
        global_metrics = metrics["global_metrics"]
        assert global_metrics["total_analyses"] == 2
        assert global_metrics["total_errors"] == 1
        assert global_metrics["error_rate"] == 1/3  # 1 error out of 3 total operations
        
    def test_get_agent_performance(self, monitor):
        """Test getting agent-specific performance."""
        sentiment_scores = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        
        # Add events for agent 0
        monitor.record_analysis(0, "text1", sentiment_scores, 0.5)
        monitor.record_analysis(0, "text2", sentiment_scores, 0.3)
        monitor.record_error(0, "text3", "Error", 0.1)
        
        # Add events for agent 1
        monitor.record_analysis(1, "text4", sentiment_scores, 0.4)
        
        performance_0 = monitor.get_agent_performance(0)
        performance_1 = monitor.get_agent_performance(1)
        
        assert performance_0["agent_id"] == 0
        assert performance_0["total_analyses"] == 2
        assert performance_0["total_errors"] == 1
        assert performance_0["error_rate"] == 1/3
        
        assert performance_1["agent_id"] == 1
        assert performance_1["total_analyses"] == 1
        assert performance_1["total_errors"] == 0
        assert performance_1["error_rate"] == 0.0
        
    def test_get_health_status(self, monitor):
        """Test health status assessment."""
        # Test healthy status
        sentiment_scores = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        for i in range(10):
            monitor.record_analysis(0, f"text{i}", sentiment_scores, 0.1)
            
        health = monitor.get_health_status()
        assert health["status"] == "healthy"
        assert health["score"] == 1.0
        
        # Test unhealthy status (high error rate)
        for i in range(5):
            monitor.record_error(0, f"error_text{i}", "Error", 0.1)
            
        health = monitor.get_health_status()
        assert health["status"] in ["unhealthy", "degraded"]
        assert health["score"] < 1.0
        
    def test_get_trend_analysis(self, monitor):
        """Test trend analysis."""
        sentiment_scores = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        
        # Add events over time
        for i in range(5):
            monitor.record_analysis(i % 2, f"text{i}", sentiment_scores, 0.1 * (i + 1))
            
        monitor.record_error(0, "error_text", "Error", 0.05)
        
        trends = monitor.get_trend_analysis(hours=1)
        
        assert "time_period_hours" in trends
        assert "total_events" in trends
        assert "analysis_events" in trends
        assert "error_events" in trends
        assert "error_rate" in trends
        assert "avg_processing_time" in trends
        assert "avg_sentiment_distribution" in trends
        assert "events_per_hour" in trends
        
        assert trends["analysis_events"] == 5
        assert trends["error_events"] == 1
        assert trends["error_rate"] == 1/6
        
    def test_export_metrics(self, monitor, tmp_path):
        """Test metrics export functionality."""
        # Add some data
        sentiment_scores = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        monitor.record_analysis(0, "text", sentiment_scores, 0.5)
        
        # Export to temp file
        export_path = tmp_path / "metrics.json"
        monitor.export_metrics(str(export_path))
        
        assert export_path.exists()
        
        # Verify content
        import json
        with open(export_path, 'r') as f:
            data = json.load(f)
            
        assert "export_timestamp" in data
        assert "current_metrics" in data
        assert "health_status" in data
        assert "trend_analysis" in data
        
    def test_reset_metrics(self, monitor):
        """Test metrics reset."""
        # Add some data
        sentiment_scores = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        monitor.record_analysis(0, "text", sentiment_scores, 0.5)
        monitor.record_error(0, "error_text", "Error", 0.1)
        
        assert len(monitor.events) > 0
        assert monitor.metrics["total_analyses"] > 0
        
        monitor.reset_metrics()
        
        assert len(monitor.events) == 0
        assert monitor.metrics["total_analyses"] == 0
        assert monitor.metrics["total_errors"] == 0
        assert len(monitor.agent_metrics) == 0
        assert len(monitor.error_counts) == 0
        assert len(monitor.recent_errors) == 0
        
    def test_event_limit(self, monitor):
        """Test that events are limited to max_events."""
        sentiment_scores = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        
        # Add more events than the limit (100)
        for i in range(150):
            monitor.record_analysis(0, f"text{i}", sentiment_scores, 0.1)
            
        # Should be limited to max_events
        assert len(monitor.events) <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])