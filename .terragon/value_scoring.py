#!/usr/bin/env python3
"""
Comprehensive Value Scoring Engine for AI Research Repository
Combines WSJF, ICE, and Technical Debt scoring methodologies
"""

import json
import math
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(Enum):
    TECHNICAL_DEBT = "technical_debt"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    RESEARCH = "research"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class ValueItem:
    """Represents a work item with comprehensive scoring"""
    id: str
    title: str
    description: str
    category: Category
    
    # WSJF Components (1-10 scale)
    business_value: float  # Research impact, user value
    time_criticality: float  # Urgency, deadline pressure
    risk_reduction: float  # Risk mitigation value
    effort_estimate: float  # Development effort (story points)
    
    # ICE Components (1-10 scale)
    impact: float  # Expected outcome magnitude
    confidence: float  # Certainty in estimates
    ease: float  # Implementation difficulty (inverse)
    
    # Technical Debt Metrics
    complexity_score: float  # Code complexity impact
    maintainability_score: float  # Long-term maintenance cost
    test_coverage_impact: float  # Testing improvement potential
    documentation_gap: float  # Documentation completeness gap
    security_risk: float  # Security vulnerability level
    
    # AI Research Specific Metrics
    research_novelty: float = 0.0  # Scientific contribution potential
    benchmark_impact: float = 0.0  # Performance improvement potential
    reproducibility_impact: float = 0.0  # Research reproducibility improvement
    community_value: float = 0.0  # Open source community benefit
    
    # Computed scores
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    research_value_score: float = 0.0
    final_score: float = 0.0
    priority: Priority = Priority.MEDIUM
    
    # Metadata
    source: str = ""  # How this item was discovered
    file_paths: List[str] = None  # Associated files
    estimated_hours: float = 0.0
    
    def __post_init__(self):
        if self.file_paths is None:
            self.file_paths = []


class ValueScoringEngine:
    """Advanced value scoring engine for AI research repositories"""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config = self._load_config(config_path)
        self.scoring_config = self.config.get("scoring", {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if yaml is None:
                return {"scoring": {"weights": {"advanced": {"wsjf": 0.5, "ice": 0.1, "technicalDebt": 0.3, "security": 0.1}}}}
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if file loading fails"""
        return {
            "scoring": {
                "wsjf": {
                    "business_value_weight": 0.3,
                    "time_criticality_weight": 0.2,
                    "risk_reduction_weight": 0.25,
                    "effort_weight": 0.25
                },
                "ice": {
                    "impact_weight": 0.4,
                    "confidence_weight": 0.3,
                    "ease_weight": 0.3
                },
                "technical_debt": {
                    "complexity_weight": 0.25,
                    "maintainability_weight": 0.25,
                    "test_coverage_weight": 0.2,
                    "documentation_weight": 0.15,
                    "security_weight": 0.15
                }
            }
        }
    
    def calculate_wsjf_score(self, item: ValueItem) -> float:
        """Calculate Weighted Shortest Job First score"""
        weights = self.scoring_config.get("wsjf", {})
        
        # Cost of Delay = Business Value + Time Criticality + Risk Reduction
        cost_of_delay = (
            item.business_value * weights.get("business_value_weight", 0.3) +
            item.time_criticality * weights.get("time_criticality_weight", 0.2) +
            item.risk_reduction * weights.get("risk_reduction_weight", 0.25)
        )
        
        # WSJF = Cost of Delay / Job Size (effort)
        # Add small epsilon to avoid division by zero
        effort_weighted = item.effort_estimate * weights.get("effort_weight", 0.25)
        wsjf_score = cost_of_delay / max(effort_weighted, 0.1)
        
        return min(wsjf_score, 100.0)  # Cap at 100
    
    def calculate_ice_score(self, item: ValueItem) -> float:
        """Calculate ICE (Impact, Confidence, Ease) score"""
        weights = self.scoring_config.get("ice", {})
        
        ice_score = (
            item.impact * weights.get("impact_weight", 0.4) +
            item.confidence * weights.get("confidence_weight", 0.3) +
            item.ease * weights.get("ease_weight", 0.3)
        )
        
        return ice_score
    
    def calculate_technical_debt_score(self, item: ValueItem) -> float:
        """Calculate technical debt severity score"""
        weights = self.scoring_config.get("technical_debt", {})
        
        debt_score = (
            item.complexity_score * weights.get("complexity_weight", 0.25) +
            item.maintainability_score * weights.get("maintainability_weight", 0.25) +
            item.test_coverage_impact * weights.get("test_coverage_weight", 0.2) +
            item.documentation_gap * weights.get("documentation_weight", 0.15) +
            item.security_risk * weights.get("security_weight", 0.15)
        )
        
        return debt_score
    
    def calculate_research_value_score(self, item: ValueItem) -> float:
        """Calculate AI research specific value score"""
        # Equal weighting for research metrics
        research_score = (
            item.research_novelty * 0.3 +
            item.benchmark_impact * 0.25 +
            item.reproducibility_impact * 0.25 +
            item.community_value * 0.2
        )
        
        return research_score
    
    def calculate_final_score(self, item: ValueItem) -> float:
        """Calculate comprehensive final score combining all methodologies"""
        # Normalize WSJF score to 0-10 scale
        normalized_wsjf = min(item.wsjf_score / 10.0, 10.0)
        
        # Weighted combination based on item category
        if item.category == Category.RESEARCH:
            final_score = (
                normalized_wsjf * 0.25 +
                item.ice_score * 0.25 +
                item.research_value_score * 0.35 +
                item.technical_debt_score * 0.15
            )
        elif item.category == Category.TECHNICAL_DEBT:
            final_score = (
                normalized_wsjf * 0.2 +
                item.ice_score * 0.3 +
                item.technical_debt_score * 0.4 +
                item.research_value_score * 0.1
            )
        elif item.category == Category.SECURITY:
            final_score = (
                normalized_wsjf * 0.35 +
                item.ice_score * 0.25 +
                item.technical_debt_score * 0.35 +
                item.research_value_score * 0.05
            )
        else:
            # Balanced scoring for other categories
            final_score = (
                normalized_wsjf * 0.3 +
                item.ice_score * 0.3 +
                item.technical_debt_score * 0.25 +
                item.research_value_score * 0.15
            )
        
        return final_score
    
    def determine_priority(self, final_score: float) -> Priority:
        """Determine priority level based on final score"""
        if final_score >= 8.5:
            return Priority.CRITICAL
        elif final_score >= 7.0:
            return Priority.HIGH
        elif final_score >= 5.0:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def estimate_effort_hours(self, item: ValueItem) -> float:
        """Estimate development hours based on effort points and category"""
        # Base hours per effort point by category
        base_hours = {
            Category.TECHNICAL_DEBT: 4.0,
            Category.SECURITY: 6.0,
            Category.PERFORMANCE: 8.0,
            Category.DOCUMENTATION: 2.0,
            Category.TESTING: 3.0,
            Category.RESEARCH: 12.0,
            Category.INFRASTRUCTURE: 10.0
        }
        
        category_hours = base_hours.get(item.category, 6.0)
        return item.effort_estimate * category_hours
    
    def score_item(self, item: ValueItem) -> ValueItem:
        """Calculate all scores for a value item"""
        # Calculate individual scores
        item.wsjf_score = self.calculate_wsjf_score(item)
        item.ice_score = self.calculate_ice_score(item)
        item.technical_debt_score = self.calculate_technical_debt_score(item)
        item.research_value_score = self.calculate_research_value_score(item)
        
        # Calculate final score and priority
        item.final_score = self.calculate_final_score(item)
        item.priority = self.determine_priority(item.final_score)
        
        # Estimate effort in hours
        item.estimated_hours = self.estimate_effort_hours(item)
        
        return item
    
    def score_items(self, items: List[ValueItem]) -> List[ValueItem]:
        """Score multiple items and sort by final score"""
        scored_items = [self.score_item(item) for item in items]
        return sorted(scored_items, key=lambda x: x.final_score, reverse=True)
    
    def export_scored_items(self, items: List[ValueItem], output_path: str):
        """Export scored items to JSON file"""
        items_dict = [asdict(item) for item in items]
        
        # Convert enum values to strings
        for item_dict in items_dict:
            item_dict['category'] = item_dict['category'].value
            item_dict['priority'] = item_dict['priority'].value
        
        with open(output_path, 'w') as f:
            json.dump(items_dict, f, indent=2, default=str)
    
    def get_scoring_summary(self, items: List[ValueItem]) -> Dict[str, Any]:
        """Generate summary statistics for scored items"""
        if not items:
            return {}
        
        scores = [item.final_score for item in items]
        priority_counts = {}
        category_counts = {}
        
        for item in items:
            priority = item.priority.value
            category = item.category.value
            
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_items": len(items),
            "score_statistics": {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores)
            },
            "priority_distribution": priority_counts,
            "category_distribution": category_counts,
            "total_estimated_hours": sum(item.estimated_hours for item in items)
        }


def create_sample_item() -> ValueItem:
    """Create a sample value item for testing"""
    return ValueItem(
        id="sample-001",
        title="Optimize belief reasoning performance",
        description="Improve the performance of nested belief queries in the Prolog backend",
        category=Category.PERFORMANCE,
        business_value=8.0,
        time_criticality=6.0,
        risk_reduction=7.0,
        effort_estimate=5.0,
        impact=8.5,
        confidence=7.0,
        ease=6.0,
        complexity_score=7.0,
        maintainability_score=6.0,
        test_coverage_impact=5.0,
        documentation_gap=4.0,
        security_risk=2.0,
        research_novelty=6.0,
        benchmark_impact=8.0,
        reproducibility_impact=7.0,
        community_value=7.5,
        source="performance_analyzer",
        file_paths=["pwmk/core/beliefs.py", "pwmk/planning/epistemic.py"]
    )


if __name__ == "__main__":
    # Test the scoring engine
    engine = ValueScoringEngine()
    sample_item = create_sample_item()
    scored_item = engine.score_item(sample_item)
    
    print("Sample Value Item Scoring:")
    print(f"Title: {scored_item.title}")
    print(f"WSJF Score: {scored_item.wsjf_score:.2f}")
    print(f"ICE Score: {scored_item.ice_score:.2f}")
    print(f"Technical Debt Score: {scored_item.technical_debt_score:.2f}")
    print(f"Research Value Score: {scored_item.research_value_score:.2f}")
    print(f"Final Score: {scored_item.final_score:.2f}")
    print(f"Priority: {scored_item.priority.value}")
    print(f"Estimated Hours: {scored_item.estimated_hours:.1f}")