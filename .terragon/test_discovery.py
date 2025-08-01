#!/usr/bin/env python3
"""
Test script for the value discovery system that works without external dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock yaml module for testing
class MockYAML:
    @staticmethod
    def safe_load(content):
        # Simple YAML parser for our config
        config = {
            "repository": {
                "name": "perspective-world-model-kit",
                "type": "ai-research",
                "maturity_level": "advanced",
                "domain": "neuro-symbolic-ai",
                "research_areas": [
                    "theory-of-mind",
                    "multi-agent-systems", 
                    "neuro-symbolic-learning",
                    "belief-reasoning",
                    "epistemic-planning",
                    "world-models"
                ]
            },
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
            },
            "discovery": {
                "code_analysis": {"enabled": True, "max_complexity": 10},
                "security": {"enabled": True},
                "performance": {"enabled": True},
                "documentation": {"enabled": True}
            },
            "output": {
                "display": {
                    "max_items": 50,
                    "group_by_category": True
                }
            }
        }
        return config

# Patch yaml import
sys.modules['yaml'] = MockYAML()

# Now import our modules
from value_scoring import ValueScoringEngine, ValueItem, Category, Priority, create_sample_item
from analyzers import TechnicalDebtAnalyzer

def test_value_scoring():
    """Test the value scoring engine"""
    print("🧪 Testing Value Scoring Engine...")
    
    engine = ValueScoringEngine()  # Will use mock config
    sample_item = create_sample_item()
    
    scored_item = engine.score_item(sample_item)
    
    print(f"✅ Sample Item Scored:")
    print(f"   Title: {scored_item.title}")
    print(f"   WSJF Score: {scored_item.wsjf_score:.2f}")
    print(f"   ICE Score: {scored_item.ice_score:.2f}")
    print(f"   Technical Debt Score: {scored_item.technical_debt_score:.2f}")
    print(f"   Research Value Score: {scored_item.research_value_score:.2f}")
    print(f"   Final Score: {scored_item.final_score:.2f}")
    print(f"   Priority: {scored_item.priority.value}")
    print(f"   Estimated Hours: {scored_item.estimated_hours:.1f}")
    print()

def test_technical_debt_analyzer():
    """Test the technical debt analyzer"""
    print("🔧 Testing Technical Debt Analyzer...")
    
    config = MockYAML.safe_load("")
    analyzer = TechnicalDebtAnalyzer(config, ".")
    
    # This will analyze actual files in the repo
    result = analyzer.analyze()
    
    print(f"✅ Analysis Complete:")
    print(f"   Items Found: {len(result.items_discovered)}")
    print(f"   Execution Time: {result.execution_time:.2f}s")
    print(f"   Files Analyzed: {result.metadata.get('files_analyzed', 0)}")
    print(f"   Complexity Violations: {result.metadata.get('complexity_violations', 0)}")
    print(f"   Code Smells: {result.metadata.get('code_smells', 0)}")
    print()
    
    if result.items_discovered:
        print("📋 Top 3 Issues Found:")
        for i, item in enumerate(result.items_discovered[:3], 1):
            print(f"   {i}. {item.title}")
            print(f"      Category: {item.category.value}")
            print(f"      Description: {item.description[:100]}...")
            print()

def create_sample_backlog():
    """Create a sample backlog to demonstrate the output"""
    print("📝 Creating Sample Backlog...")
    
    # Create some sample items
    items = [
        ValueItem(
            id="perf-001",
            title="Optimize belief reasoning performance",
            description="Improve the performance of nested belief queries in the Prolog backend for faster Theory of Mind inference",
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
        ),
        ValueItem(
            id="doc-001", 
            title="Add comprehensive API documentation",
            description="Core world model classes lack docstrings and API documentation for research reproducibility",
            category=Category.DOCUMENTATION,
            business_value=6.0,
            time_criticality=4.0,
            risk_reduction=5.0,
            effort_estimate=3.0,
            impact=7.0,
            confidence=9.0,
            ease=8.0,
            complexity_score=2.0,
            maintainability_score=8.0,
            test_coverage_impact=3.0,
            documentation_gap=9.0,
            security_risk=1.0,
            research_novelty=2.0,
            benchmark_impact=2.0,
            reproducibility_impact=9.0,
            community_value=8.0,
            source="documentation_analyzer",
            file_paths=["pwmk/core/world_model.py"]
        ),
        ValueItem(
            id="sec-001",
            title="Fix unsafe model loading",
            description="torch.load calls without weights_only=True pose security risks for model deserialization",
            category=Category.SECURITY,
            business_value=8.0,
            time_criticality=9.0,
            risk_reduction=9.0,
            effort_estimate=1.0,
            impact=8.0,
            confidence=9.0,
            ease=9.0,
            complexity_score=1.0,
            maintainability_score=3.0,
            test_coverage_impact=3.0,
            documentation_gap=2.0,
            security_risk=9.0,
            research_novelty=0.0,
            benchmark_impact=1.0,
            reproducibility_impact=2.0,
            community_value=3.0,
            source="security_analyzer",
            file_paths=["pwmk/core/world_model.py"]
        )
    ]
    
    # Score the items
    engine = ValueScoringEngine()
    scored_items = engine.score_items(items)
    
    print(f"✅ Sample items scored and prioritized:")
    for i, item in enumerate(scored_items, 1):
        print(f"   {i}. {item.title} (Score: {item.final_score:.1f}, {item.priority.value})")
    
    return scored_items

def main():
    """Run all tests"""
    print("🚀 Testing Terragon Value Discovery System")
    print("==========================================")
    print()
    
    try:
        # Test value scoring
        test_value_scoring()
        
        # Test technical debt analyzer
        test_technical_debt_analyzer()
        
        # Create sample backlog
        sample_items = create_sample_backlog()
        
        print("✅ All tests completed successfully!")
        print(f"📊 System is ready for autonomous value discovery")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())