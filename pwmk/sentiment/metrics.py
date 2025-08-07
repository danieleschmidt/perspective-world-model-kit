"""
Metrics and evaluation utilities for sentiment analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from ..utils.logging import get_logger
from .validation import validate_sentiment_scores

logger = get_logger(__name__)


class SentimentMetrics:
    """
    Comprehensive metrics calculation for sentiment analysis performance.
    """
    
    def __init__(self):
        self.sentiment_labels = ["negative", "neutral", "positive"]
        self.reset()
        
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.predictions = []
        self.ground_truth = []
        self.confidence_scores = []
        
    def add_prediction(
        self,
        predicted_sentiment: Dict[str, float],
        true_sentiment: str,
        confidence: Optional[float] = None
    ) -> None:
        """
        Add a prediction for metric calculation.
        
        Args:
            predicted_sentiment: Predicted sentiment scores
            true_sentiment: True sentiment label
            confidence: Prediction confidence score
        """
        try:
            validate_sentiment_scores(predicted_sentiment)
            
            if true_sentiment not in self.sentiment_labels:
                raise ValueError(f"True sentiment must be one of {self.sentiment_labels}")
                
            # Get predicted label
            predicted_label = max(predicted_sentiment.items(), key=lambda x: x[1])[0]
            
            self.predictions.append(predicted_label)
            self.ground_truth.append(true_sentiment)
            
            # Use max probability as confidence if not provided
            if confidence is None:
                confidence = max(predicted_sentiment.values())
            self.confidence_scores.append(confidence)
            
        except Exception as e:
            logger.warning(f"Failed to add prediction: {e}")
            
    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy."""
        if not self.predictions:
            return 0.0
        return accuracy_score(self.ground_truth, self.predictions)
        
    def calculate_precision_recall_f1(self) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, and F1 scores per class."""
        if not self.predictions:
            return {label: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for label in self.sentiment_labels}
            
        precision, recall, f1, support = precision_recall_fscore_support(
            self.ground_truth,
            self.predictions,
            labels=self.sentiment_labels,
            average=None,
            zero_division=0
        )
        
        results = {}
        for i, label in enumerate(self.sentiment_labels):
            results[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
            
        return results
        
    def calculate_macro_metrics(self) -> Dict[str, float]:
        """Calculate macro-averaged metrics."""
        if not self.predictions:
            return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.ground_truth,
            self.predictions,
            labels=self.sentiment_labels,
            average="macro",
            zero_division=0
        )
        
        return {
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1)
        }
        
    def calculate_confusion_matrix(self) -> np.ndarray:
        """Calculate confusion matrix."""
        if not self.predictions:
            return np.zeros((len(self.sentiment_labels), len(self.sentiment_labels)))
            
        return confusion_matrix(
            self.ground_truth,
            self.predictions,
            labels=self.sentiment_labels
        )
        
    def calculate_confidence_metrics(self) -> Dict[str, float]:
        """Calculate confidence-related metrics."""
        if not self.confidence_scores:
            return {"avg_confidence": 0.0, "confidence_std": 0.0}
            
        confidence_array = np.array(self.confidence_scores)
        
        return {
            "avg_confidence": float(np.mean(confidence_array)),
            "confidence_std": float(np.std(confidence_array)),
            "min_confidence": float(np.min(confidence_array)),
            "max_confidence": float(np.max(confidence_array))
        }
        
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive evaluation report."""
        if not self.predictions:
            logger.warning("No predictions available for metrics calculation")
            return {}
            
        report = {
            "total_samples": len(self.predictions),
            "accuracy": self.calculate_accuracy(),
            "per_class_metrics": self.calculate_precision_recall_f1(),
            "macro_metrics": self.calculate_macro_metrics(),
            "confidence_metrics": self.calculate_confidence_metrics(),
            "confusion_matrix": self.calculate_confusion_matrix().tolist()
        }
        
        return report
        
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix visualization."""
        cm = self.calculate_confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Sentiment Analysis Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(self.sentiment_labels))
        plt.xticks(tick_marks, self.sentiment_labels, rotation=45)
        plt.yticks(tick_marks, self.sentiment_labels)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()


class MultiAgentSentimentMetrics:
    """
    Metrics for multi-agent sentiment analysis scenarios.
    """
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agent_metrics = {i: SentimentMetrics() for i in range(num_agents)}
        self.interaction_data = []
        
    def add_agent_prediction(
        self,
        agent_id: int,
        predicted_sentiment: Dict[str, float],
        true_sentiment: str,
        confidence: Optional[float] = None
    ) -> None:
        """Add prediction for specific agent."""
        if 0 <= agent_id < self.num_agents:
            self.agent_metrics[agent_id].add_prediction(
                predicted_sentiment, true_sentiment, confidence
            )
        else:
            logger.warning(f"Invalid agent_id {agent_id}, must be in range [0, {self.num_agents})")
            
    def add_interaction_data(
        self,
        agent_pairs: List[Tuple[int, int]],
        sentiment_alignments: List[bool],
        interaction_types: List[str]
    ) -> None:
        """
        Add interaction data for analysis.
        
        Args:
            agent_pairs: List of (agent1, agent2) pairs
            sentiment_alignments: Whether agents had aligned sentiments
            interaction_types: Type of interaction (e.g., 'cooperation', 'conflict')
        """
        for (agent1, agent2), aligned, interaction_type in zip(agent_pairs, sentiment_alignments, interaction_types):
            self.interaction_data.append({
                "agent1": agent1,
                "agent2": agent2,
                "sentiment_aligned": aligned,
                "interaction_type": interaction_type
            })
            
    def calculate_agent_performance(self, agent_id: int) -> Dict[str, Any]:
        """Calculate performance metrics for specific agent."""
        if agent_id not in self.agent_metrics:
            return {}
        return self.agent_metrics[agent_id].get_comprehensive_report()
        
    def calculate_group_metrics(self) -> Dict[str, Any]:
        """Calculate group-level metrics."""
        group_accuracy = np.mean([
            metrics.calculate_accuracy()
            for metrics in self.agent_metrics.values()
            if metrics.predictions
        ])
        
        # Calculate sentiment alignment rate
        if self.interaction_data:
            alignment_rate = np.mean([
                interaction["sentiment_aligned"]
                for interaction in self.interaction_data
            ])
        else:
            alignment_rate = 0.0
            
        # Calculate interaction type distribution
        interaction_types = {}
        for interaction in self.interaction_data:
            interaction_type = interaction["interaction_type"]
            interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
            
        return {
            "group_accuracy": float(group_accuracy),
            "sentiment_alignment_rate": float(alignment_rate),
            "total_interactions": len(self.interaction_data),
            "interaction_type_distribution": interaction_types,
            "num_agents": self.num_agents
        }
        
    def get_agent_comparison(self) -> Dict[str, List[float]]:
        """Compare performance across agents."""
        comparison = {
            "accuracies": [],
            "avg_confidences": [],
            "macro_f1_scores": []
        }
        
        for agent_id in range(self.num_agents):
            metrics = self.agent_metrics[agent_id]
            comparison["accuracies"].append(metrics.calculate_accuracy())
            
            conf_metrics = metrics.calculate_confidence_metrics()
            comparison["avg_confidences"].append(conf_metrics.get("avg_confidence", 0.0))
            
            macro_metrics = metrics.calculate_macro_metrics()
            comparison["macro_f1_scores"].append(macro_metrics.get("macro_f1", 0.0))
            
        return comparison
        
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive multi-agent sentiment analysis report."""
        report = {
            "group_metrics": self.calculate_group_metrics(),
            "agent_comparison": self.get_agent_comparison(),
            "individual_agent_reports": {}
        }
        
        # Add individual agent reports
        for agent_id in range(self.num_agents):
            report["individual_agent_reports"][f"agent_{agent_id}"] = \
                self.calculate_agent_performance(agent_id)
                
        return report