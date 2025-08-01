#!/usr/bin/env python3
"""
Continuous Value Discovery Orchestrator
Runs all analyzers and generates prioritized backlog for AI research repository
"""

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not available. Using fallback JSON configuration.")
    yaml = None

from value_scoring import ValueScoringEngine, ValueItem, Category, Priority
from analyzers import (
    TechnicalDebtAnalyzer,
    SecurityAnalyzer, 
    PerformanceAnalyzer,
    DocumentationAnalyzer,
    TestCoverageAnalyzer,
    AnalysisResult
)


class ValueDiscoveryOrchestrator:
    """Orchestrates the continuous value discovery process"""
    
    def __init__(self, config_path: str = ".terragon/config.yaml", repo_path: str = "."):
        self.config_path = config_path
        self.repo_path = Path(repo_path)
        self.config = self._load_config()
        self.scoring_engine = ValueScoringEngine(config_path)
        
        # Initialize analyzers
        self.analyzers = {
            'technical_debt': TechnicalDebtAnalyzer(self.config, repo_path),
            'security': SecurityAnalyzer(self.config, repo_path),
            'performance': PerformanceAnalyzer(self.config, repo_path),
            'documentation': DocumentationAnalyzer(self.config, repo_path),
            'test_coverage': TestCoverageAnalyzer(self.config, repo_path)
        }
        
        # Setup output directories
        self.reports_dir = self.repo_path / ".terragon" / "reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if yaml is None:
                # Fallback to basic config if PyYAML not available
                return {
                    "scoring": {
                        "weights": {"advanced": {"wsjf": 0.5, "ice": 0.1, "technicalDebt": 0.3, "security": 0.1}},
                        "thresholds": {"minScore": 10, "maxRisk": 0.8}
                    },
                    "discovery": {"sources": ["gitHistory", "staticAnalysis"]},
                    "execution": {"maxConcurrentTasks": 1}
                }
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def discover_value_items(self, analyzers_to_run: Optional[List[str]] = None) -> List[ValueItem]:
        """Run analyzers and discover value items"""
        if analyzers_to_run is None:
            analyzers_to_run = list(self.analyzers.keys())
        
        all_items = []
        analysis_results = {}
        
        print("🔍 Starting value discovery analysis...")
        start_time = time.time()
        
        for analyzer_name in analyzers_to_run:
            if analyzer_name not in self.analyzers:
                print(f"⚠️  Unknown analyzer: {analyzer_name}")
                continue
            
            analyzer = self.analyzers[analyzer_name]
            print(f"📊 Running {analyzer_name} analyzer...")
            
            try:
                result = analyzer.analyze()
                analysis_results[analyzer_name] = result
                all_items.extend(result.items_discovered)
                
                print(f"✅ {analyzer_name}: Found {len(result.items_discovered)} items "
                      f"({result.execution_time:.2f}s)")
                
            except Exception as e:
                print(f"❌ {analyzer_name} failed: {e}")
                traceback.print_exc()
        
        total_time = time.time() - start_time
        print(f"🎯 Discovery complete: {len(all_items)} items found in {total_time:.2f}s")
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        return all_items
    
    def prioritize_items(self, items: List[ValueItem]) -> List[ValueItem]:
        """Score and prioritize value items"""
        print("📈 Scoring and prioritizing items...")
        
        scored_items = self.scoring_engine.score_items(items)
        
        # Apply additional AI research context
        for item in scored_items:
            item = self._apply_research_context(item)
        
        # Re-sort after context adjustment
        scored_items.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored_items
    
    def _apply_research_context(self, item: ValueItem) -> ValueItem:
        """Apply AI research specific context to item scoring"""
        research_config = self.config.get("repository", {})
        research_areas = research_config.get("research_areas", [])
        
        # Boost items related to core research areas
        if any(area.replace('-', '_') in item.description.lower() or 
               area.replace('-', '_') in item.title.lower() 
               for area in research_areas):
            item.research_novelty = min(item.research_novelty + 2.0, 10.0)
            item.community_value = min(item.community_value + 1.5, 10.0)
        
        # Boost items that improve reproducibility
        reproducibility_keywords = ['config', 'seed', 'deterministic', 'random', 'reproduce']
        if any(keyword in item.description.lower() for keyword in reproducibility_keywords):
            item.reproducibility_impact = min(item.reproducibility_impact + 3.0, 10.0)
        
        # Boost performance items for AI workloads
        if item.category == Category.PERFORMANCE and any(keyword in item.description.lower() 
                                                        for keyword in ['neural', 'model', 'training', 'inference']):
            item.benchmark_impact = min(item.benchmark_impact + 2.0, 10.0)
        
        # Re-calculate final score with updated values
        item.final_score = self.scoring_engine.calculate_final_score(item)
        item.priority = self.scoring_engine.determine_priority(item.final_score)
        
        return item
    
    def generate_backlog(self, items: List[ValueItem]) -> str:
        """Generate markdown backlog from prioritized items"""
        max_items = self.config.get("output", {}).get("display", {}).get("max_items", 50)
        display_items = items[:max_items]
        
        # Group by category if configured
        group_by_category = self.config.get("output", {}).get("display", {}).get("group_by_category", True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Start building markdown
        markdown = [
            "# 🎯 Autonomous Value Discovery Backlog",
            f"*Generated on {timestamp} by Terragon*",
            "",
            f"**Repository**: {self.config.get('repository', {}).get('name', 'Unknown')}",
            f"**Domain**: {self.config.get('repository', {}).get('domain', 'Unknown')}",
            f"**Maturity**: {self.config.get('repository', {}).get('maturity_level', 'Unknown')}",
            "",
            "## 📊 Summary",
            "",
            f"- **Total Items Discovered**: {len(items)}",
            f"- **Showing Top**: {len(display_items)}",
            f"- **Estimated Total Effort**: {sum(item.estimated_hours for item in display_items):.1f} hours",
            "",
        ]
        
        # Priority distribution
        priority_counts = {}
        for item in display_items:
            priority_counts[item.priority.value] = priority_counts.get(item.priority.value, 0) + 1
        
        markdown.extend([
            "### Priority Distribution",
            "",
            f"- 🔴 **Critical**: {priority_counts.get('critical', 0)}",
            f"- 🟠 **High**: {priority_counts.get('high', 0)}",
            f"- 🟡 **Medium**: {priority_counts.get('medium', 0)}",
            f"- 🟢 **Low**: {priority_counts.get('low', 0)}",
            "",
        ])
        
        # Category distribution
        category_counts = {}
        for item in display_items:
            category_counts[item.category.value] = category_counts.get(item.category.value, 0) + 1
        
        markdown.extend([
            "### Category Distribution",
            "",
        ])
        
        category_emojis = {
            'technical_debt': '🔧',
            'security': '🔒',
            'performance': '⚡',
            'documentation': '📚',
            'testing': '🧪',
            'research': '🔬',
            'infrastructure': '🏗️'
        }
        
        for category, count in category_counts.items():
            emoji = category_emojis.get(category, '📋')
            markdown.append(f"- {emoji} **{category.replace('_', ' ').title()}**: {count}")
        
        markdown.extend(["", "---", ""])
        
        if group_by_category:
            # Group items by category
            categories = {}
            for item in display_items:
                if item.category not in categories:
                    categories[item.category] = []
                categories[item.category].append(item)
            
            for category, category_items in categories.items():
                emoji = category_emojis.get(category.value, '📋')
                markdown.extend([
                    f"## {emoji} {category.value.replace('_', ' ').title()} ({len(category_items)} items)",
                    ""
                ])
                
                for item in category_items:
                    markdown.extend(self._format_item(item))
                
                markdown.append("")
        else:
            # Flat list ordered by priority
            markdown.extend([
                "## 📋 Prioritized Items",
                ""
            ])
            
            for item in display_items:
                markdown.extend(self._format_item(item))
        
        # Add research insights section
        if any(item.category == Category.RESEARCH for item in display_items):
            research_items = [item for item in display_items if item.category == Category.RESEARCH]
            markdown.extend([
                "---",
                "",
                "## 🔬 Research Insights",
                "",
                f"Found {len(research_items)} research-related opportunities that could enhance the scientific contribution of this work:",
                ""
            ])
            
            for item in research_items[:5]:  # Top 5 research items
                markdown.extend([
                    f"### {item.title}",
                    f"{item.description}",
                    f"**Research Value**: {item.research_value_score:.1f}/10 | **Novelty**: {item.research_novelty:.1f}/10",
                    ""
                ])
        
        # Footer
        markdown.extend([
            "---",
            "",
            "## 🤖 About This Backlog",
            "",
            "This backlog was automatically generated using Terragon's autonomous value discovery system. It combines:",
            "",
            "- **WSJF Scoring**: Weighted Shortest Job First prioritization",
            "- **ICE Framework**: Impact, Confidence, Ease evaluation", 
            "- **Technical Debt Analysis**: Code quality and maintainability metrics",
            "- **AI Research Context**: Domain-specific value assessment",
            "",
            "### Scoring Methodology",
            "",
            "Each item receives scores across multiple dimensions:",
            "",
            "- **Business Value**: Research impact and user benefit",
            "- **Time Criticality**: Urgency and deadline pressure",
            "- **Risk Reduction**: Risk mitigation value",
            "- **Effort Estimate**: Development complexity",
            "- **Research Value**: Scientific contribution potential",
            "",
            f"Items are automatically re-prioritized based on the {self.config.get('repository', {}).get('domain', 'AI research')} domain context.",
            "",
            "### Next Steps",
            "",
            "1. Review high-priority items with the research team",
            "2. Validate effort estimates and business value",
            "3. Create GitHub issues for approved items",  
            "4. Schedule work based on research timeline and conference deadlines",
            "",
            "*For questions about this backlog, see `.terragon/` configuration and scripts.*"
        ])
        
        return "\n".join(markdown)
    
    def _format_item(self, item: ValueItem) -> List[str]:
        """Format a single item for markdown display"""
        priority_emoji = {
            Priority.CRITICAL: "🔴",
            Priority.HIGH: "🟠", 
            Priority.MEDIUM: "🟡",
            Priority.LOW: "🟢"
        }
        
        lines = [
            f"### {priority_emoji[item.priority]} {item.title}",
            "",
            f"**Score**: {item.final_score:.1f}/10 | **Priority**: {item.priority.value.title()} | **Effort**: {item.estimated_hours:.1f}h",
            "",
            f"{item.description}",
            ""
        ]
        
        # Add technical details if available
        if item.file_paths:
            lines.extend([
                f"**Files**: {', '.join(item.file_paths[:3])}{'...' if len(item.file_paths) > 3 else ''}",
                ""
            ])
        
        # Add scoring breakdown for high-priority items
        if item.priority in [Priority.CRITICAL, Priority.HIGH]:
            lines.extend([
                "**Scoring Breakdown**:",
                f"- WSJF: {item.wsjf_score:.1f} | ICE: {item.ice_score:.1f} | Tech Debt: {item.technical_debt_score:.1f}",
                ""
            ])
            
            if item.research_value_score > 0:
                lines.extend([
                    f"- Research Value: {item.research_value_score:.1f} | Reproducibility: {item.reproducibility_impact:.1f}",
                    ""
                ])
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _save_analysis_results(self, results: Dict[str, AnalysisResult]):
        """Save detailed analysis results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.reports_dir / f"analysis_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for analyzer_name, result in results.items():
            serializable_results[analyzer_name] = {
                'analyzer_name': result.analyzer_name,
                'items_count': len(result.items_discovered),
                'metadata': result.metadata,
                'execution_time': result.execution_time
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Analysis results saved to {results_file}")
    
    def save_metrics(self, items: List[ValueItem]):
        """Save metrics for monitoring and trending"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_items': len(items),
            'priority_distribution': {},
            'category_distribution': {},
            'average_scores': {},
            'total_estimated_hours': sum(item.estimated_hours for item in items)
        }
        
        # Priority distribution
        for item in items:
            priority = item.priority.value
            metrics['priority_distribution'][priority] = metrics['priority_distribution'].get(priority, 0) + 1
        
        # Category distribution
        for item in items:
            category = item.category.value
            metrics['category_distribution'][category] = metrics['category_distribution'].get(category, 0) + 1
        
        # Average scores
        if items:
            metrics['average_scores'] = {
                'final_score': sum(item.final_score for item in items) / len(items),
                'wsjf_score': sum(item.wsjf_score for item in items) / len(items),
                'ice_score': sum(item.ice_score for item in items) / len(items),
                'technical_debt_score': sum(item.technical_debt_score for item in items) / len(items),
                'research_value_score': sum(item.research_value_score for item in items) / len(items)
            }
        
        # Save metrics
        metrics_file = self.repo_path / ".terragon" / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 Metrics saved to {metrics_file}")
    
    def run_full_discovery(self, output_backlog: bool = True) -> List[ValueItem]:
        """Run complete value discovery process"""
        print("🚀 Starting autonomous value discovery...")
        start_time = time.time()
        
        # Discover items
        items = self.discover_value_items()
        
        if not items:
            print("ℹ️  No value items discovered")
            return []
        
        # Prioritize items
        prioritized_items = self.prioritize_items(items)
        
        # Save metrics
        self.save_metrics(prioritized_items)
        
        # Generate and save backlog
        if output_backlog:
            backlog_content = self.generate_backlog(prioritized_items)
            backlog_file = self.repo_path / "BACKLOG.md"
            
            with open(backlog_file, 'w') as f:
                f.write(backlog_content)
            
            print(f"📝 Backlog saved to {backlog_file}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        items_file = self.reports_dir / f"value_items_{timestamp}.json"
        self.scoring_engine.export_scored_items(prioritized_items, str(items_file))
        
        total_time = time.time() - start_time
        
        print(f"✅ Value discovery complete in {total_time:.2f}s")
        print(f"📊 {len(prioritized_items)} items prioritized")
        print(f"⏱️  Total estimated effort: {sum(item.estimated_hours for item in prioritized_items):.1f} hours")
        
        # Print top 5 items
        print("\n🏆 Top 5 Value Items:")
        for i, item in enumerate(prioritized_items[:5], 1):
            print(f"{i}. {item.title} (Score: {item.final_score:.1f}, {item.priority.value.title()})")
        
        return prioritized_items


def main():
    """Command-line interface for value discovery"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Value Discovery for AI Research Repository")
    parser.add_argument('--config', default='.terragon/config.yaml', help='Configuration file path')
    parser.add_argument('--analyzers', nargs='+', help='Specific analyzers to run')
    parser.add_argument('--no-backlog', action='store_true', help='Skip backlog generation')
    parser.add_argument('--repo-path', default='.', help='Repository path')
    
    args = parser.parse_args()
    
    try:
        orchestrator = ValueDiscoveryOrchestrator(args.config, args.repo_path)
        
        if args.analyzers:
            items = orchestrator.discover_value_items(args.analyzers)
            prioritized_items = orchestrator.prioritize_items(items)
        else:
            prioritized_items = orchestrator.run_full_discovery(not args.no_backlog)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during value discovery: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())