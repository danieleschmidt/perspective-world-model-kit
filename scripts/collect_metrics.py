#!/usr/bin/env python3
"""
Automated metrics collection script for PWMK repository.
Collects various project health and performance metrics.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests


class MetricsCollector:
    """Collects various project metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = os.getenv("GITHUB_REPOSITORY", "").split("/")[0] if "/" in os.getenv("GITHUB_REPOSITORY", "") else ""
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "").split("/")[1] if "/" in os.getenv("GITHUB_REPOSITORY", "") else ""
        
    def run_command(self, cmd: List[str]) -> Optional[str]:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception as e:
            print(f"Error running command {' '.join(cmd)}: {e}")
            return None
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git-related metrics."""
        metrics = {}
        
        # Commits in last 30 days
        since_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        commit_count = self.run_command(["git", "rev-list", "--count", f"--since={since_date}", "HEAD"])
        metrics["commits_last_30_days"] = int(commit_count) if commit_count else 0
        
        # Contributors
        contributors = self.run_command(["git", "shortlog", "-sn", "HEAD"])
        metrics["total_contributors"] = len(contributors.split("\n")) if contributors else 0
        
        # Active contributors (last 30 days)
        active_contributors = self.run_command(["git", "shortlog", "-sn", f"--since={since_date}", "HEAD"])
        metrics["active_contributors_30_days"] = len(active_contributors.split("\n")) if active_contributors else 0
        
        # Lines of code
        loc_output = self.run_command(["find", ".", "-name", "*.py", "-type", "f", "-exec", "wc", "-l", "{}", "+"])
        if loc_output:
            lines = [int(line.split()[0]) for line in loc_output.split("\n") if line and line.split()[0].isdigit()]
            metrics["lines_of_code"] = sum(lines)
        else:
            metrics["lines_of_code"] = 0
            
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub API metrics."""
        if not self.github_token or not self.repo_owner or not self.repo_name:
            return {}
        
        headers = {"Authorization": f"token {self.github_token}"}
        base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        
        metrics = {}
        
        try:
            # Repository info
            repo_response = requests.get(base_url, headers=headers)
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                metrics.update({
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "watchers": repo_data.get("watchers_count", 0),
                    "open_issues": repo_data.get("open_issues_count", 0),
                    "size_kb": repo_data.get("size", 0)
                })
            
            # Pull requests
            prs_response = requests.get(f"{base_url}/pulls?state=all&per_page=100", headers=headers)
            if prs_response.status_code == 200:
                prs_data = prs_response.json()
                open_prs = len([pr for pr in prs_data if pr["state"] == "open"])
                closed_prs = len([pr for pr in prs_data if pr["state"] == "closed"])
                metrics.update({
                    "open_pull_requests": open_prs,
                    "closed_pull_requests": closed_prs
                })
            
            # Issues
            issues_response = requests.get(f"{base_url}/issues?state=all&per_page=100", headers=headers)
            if issues_response.status_code == 200:
                issues_data = issues_response.json()
                # Filter out pull requests (GitHub API includes PRs in issues)
                actual_issues = [issue for issue in issues_data if not issue.get("pull_request")]
                open_issues = len([issue for issue in actual_issues if issue["state"] == "open"])
                closed_issues = len([issue for issue in actual_issues if issue["state"] == "closed"])
                metrics.update({
                    "open_issues_actual": open_issues,
                    "closed_issues": closed_issues
                })
            
            # Releases
            releases_response = requests.get(f"{base_url}/releases", headers=headers)
            if releases_response.status_code == 200:
                releases_data = releases_response.json()
                metrics["total_releases"] = len(releases_data)
                if releases_data:
                    latest_release = releases_data[0]
                    metrics["latest_release"] = {
                        "name": latest_release.get("name"),
                        "tag": latest_release.get("tag_name"),
                        "published_at": latest_release.get("published_at"),
                        "downloads": sum(asset.get("download_count", 0) for asset in latest_release.get("assets", []))
                    }
        
        except requests.RequestException as e:
            print(f"Error fetching GitHub metrics: {e}")
        
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Test coverage (if coverage report exists)
        coverage_file = self.repo_path / "coverage.xml"
        if coverage_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                coverage_elem = root.find(".//coverage")
                if coverage_elem is not None:
                    metrics["test_coverage"] = float(coverage_elem.get("line-rate", 0)) * 100
            except Exception as e:
                print(f"Error parsing coverage file: {e}")
        
        # Count test files
        test_files = list(self.repo_path.glob("tests/**/*.py"))
        metrics["test_files_count"] = len(test_files)
        
        # Count Python files
        python_files = list(self.repo_path.glob("**/*.py"))
        metrics["python_files_count"] = len([f for f in python_files if not any(part.startswith('.') for part in f.parts)])
        
        # Check for common quality files
        quality_indicators = {
            "has_readme": (self.repo_path / "README.md").exists(),
            "has_license": (self.repo_path / "LICENSE").exists(),
            "has_contributing": (self.repo_path / "CONTRIBUTING.md").exists(),
            "has_changelog": (self.repo_path / "CHANGELOG.md").exists(),
            "has_pyproject": (self.repo_path / "pyproject.toml").exists(),
            "has_requirements": (self.repo_path / "requirements.txt").exists() or (self.repo_path / "requirements").exists(),
            "has_dockerfile": (self.repo_path / "Dockerfile").exists(),
            "has_github_workflows": (self.repo_path / ".github" / "workflows").exists(),
            "has_tests": (self.repo_path / "tests").exists(),
            "has_docs": (self.repo_path / "docs").exists()
        }
        metrics.update(quality_indicators)
        
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        metrics = {}
        
        # Parse pyproject.toml
        pyproject_file = self.repo_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import tomli
                with open(pyproject_file, "rb") as f:
                    pyproject_data = tomli.load(f)
                
                dependencies = pyproject_data.get("project", {}).get("dependencies", [])
                optional_deps = pyproject_data.get("project", {}).get("optional-dependencies", {})
                
                metrics["main_dependencies"] = len(dependencies)
                metrics["optional_dependency_groups"] = len(optional_deps)
                metrics["total_optional_dependencies"] = sum(len(deps) for deps in optional_deps.values())
                
            except ImportError:
                print("tomli not available, skipping pyproject.toml parsing")
            except Exception as e:
                print(f"Error parsing pyproject.toml: {e}")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        metrics = {}
        
        # Check if benchmark results exist
        benchmark_file = self.repo_path / "benchmark-results.json"
        if benchmark_file.exists():
            try:
                with open(benchmark_file, "r") as f:
                    benchmark_data = json.load(f)
                
                if "benchmarks" in benchmark_data:
                    benchmarks = benchmark_data["benchmarks"]
                    metrics["benchmark_count"] = len(benchmarks)
                    if benchmarks:
                        avg_time = sum(b.get("stats", {}).get("mean", 0) for b in benchmarks) / len(benchmarks)
                        metrics["average_benchmark_time"] = avg_time
                        
            except Exception as e:
                print(f"Error parsing benchmark results: {e}")
        
        return metrics
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("Collecting project metrics...")
        
        all_metrics = {
            "collection_timestamp": datetime.now().isoformat(),
            "git": self.collect_git_metrics(),
            "github": self.collect_github_metrics(),
            "code_quality": self.collect_code_quality_metrics(),
            "dependencies": self.collect_dependency_metrics(),
            "performance": self.collect_performance_metrics()
        }
        
        return all_metrics
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: str = "collected-metrics.json"):
        """Save metrics to a JSON file."""
        output_path = self.repo_path / output_file
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {output_path}")
    
    def compare_with_targets(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare collected metrics with targets from project-metrics.json."""
        targets_file = self.repo_path / ".github" / "project-metrics.json"
        if not targets_file.exists():
            return {}
        
        try:
            with open(targets_file, "r") as f:
                project_config = json.load(f)
            
            targets = project_config.get("metrics", {})
            comparison = {}
            
            # Compare code quality metrics
            if "code_quality" in targets and "test_coverage" in targets["code_quality"]:
                target_coverage = targets["code_quality"]["test_coverage"]["target"]
                actual_coverage = metrics.get("code_quality", {}).get("test_coverage", 0)
                comparison["test_coverage"] = {
                    "target": target_coverage,
                    "actual": actual_coverage,
                    "status": "pass" if actual_coverage >= target_coverage else "fail",
                    "difference": actual_coverage - target_coverage
                }
            
            # Compare community metrics
            if "community" in targets:
                github_metrics = metrics.get("github", {})
                if "github_stars" in targets["community"]:
                    target_stars = targets["community"]["github_stars"]["target"]
                    actual_stars = github_metrics.get("stars", 0)
                    comparison["github_stars"] = {
                        "target": target_stars,
                        "actual": actual_stars,
                        "status": "pass" if actual_stars >= target_stars else "fail",
                        "difference": actual_stars - target_stars
                    }
                
                if "contributors" in targets["community"]:
                    target_contributors = targets["community"]["contributors"]["target"]
                    actual_contributors = metrics.get("git", {}).get("total_contributors", 0)
                    comparison["contributors"] = {
                        "target": target_contributors,
                        "actual": actual_contributors,
                        "status": "pass" if actual_contributors >= target_contributors else "fail",
                        "difference": actual_contributors - target_contributors
                    }
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing with targets: {e}")
            return {}
    
    def generate_report(self, metrics: Dict[str, Any], comparison: Dict[str, Any]) -> str:
        """Generate a human-readable metrics report."""
        report = []
        report.append("# PWMK Project Metrics Report")
        report.append(f"Generated on: {metrics['collection_timestamp']}")
        report.append("")
        
        # Git metrics
        git_metrics = metrics.get("git", {})
        report.append("## Git Activity")
        report.append(f"- Commits (last 30 days): {git_metrics.get('commits_last_30_days', 0)}")
        report.append(f"- Total contributors: {git_metrics.get('total_contributors', 0)}")
        report.append(f"- Active contributors (30 days): {git_metrics.get('active_contributors_30_days', 0)}")
        report.append(f"- Lines of code: {git_metrics.get('lines_of_code', 0):,}")
        report.append("")
        
        # GitHub metrics
        github_metrics = metrics.get("github", {})
        if github_metrics:
            report.append("## GitHub Statistics")
            report.append(f"- Stars: {github_metrics.get('stars', 0)}")
            report.append(f"- Forks: {github_metrics.get('forks', 0)}")
            report.append(f"- Open issues: {github_metrics.get('open_issues_actual', 0)}")
            report.append(f"- Open pull requests: {github_metrics.get('open_pull_requests', 0)}")
            report.append(f"- Total releases: {github_metrics.get('total_releases', 0)}")
            report.append("")
        
        # Code quality
        quality_metrics = metrics.get("code_quality", {})
        report.append("## Code Quality")
        if "test_coverage" in quality_metrics:
            coverage = quality_metrics["test_coverage"]
            status = "✅" if coverage >= 80 else "⚠️" if coverage >= 60 else "❌"
            report.append(f"- Test coverage: {coverage:.1f}% {status}")
        report.append(f"- Python files: {quality_metrics.get('python_files_count', 0)}")
        report.append(f"- Test files: {quality_metrics.get('test_files_count', 0)}")
        report.append("")
        
        # Quality indicators
        report.append("## Quality Indicators")
        indicators = [
            ("README", quality_metrics.get("has_readme", False)),
            ("License", quality_metrics.get("has_license", False)),
            ("Contributing guide", quality_metrics.get("has_contributing", False)),
            ("Changelog", quality_metrics.get("has_changelog", False)),
            ("Tests", quality_metrics.get("has_tests", False)),
            ("Documentation", quality_metrics.get("has_docs", False)),
            ("Docker", quality_metrics.get("has_dockerfile", False)),
            ("CI/CD", quality_metrics.get("has_github_workflows", False))
        ]
        
        for name, has_indicator in indicators:
            status = "✅" if has_indicator else "❌"
            report.append(f"- {name}: {status}")
        report.append("")
        
        # Target comparison
        if comparison:
            report.append("## Target Comparison")
            for metric_name, comp_data in comparison.items():
                status_icon = "✅" if comp_data["status"] == "pass" else "❌"
                report.append(f"- {metric_name}: {comp_data['actual']}/{comp_data['target']} {status_icon}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function."""
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    collector = MetricsCollector(repo_path)
    
    # Collect metrics
    metrics = collector.collect_all_metrics()
    
    # Save raw metrics
    collector.save_metrics(metrics)
    
    # Compare with targets
    comparison = collector.compare_with_targets(metrics)
    
    # Generate and save report
    report = collector.generate_report(metrics, comparison)
    report_path = Path(repo_path) / "metrics-report.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Report saved to {report_path}")
    
    # Print summary
    print("\n=== Metrics Summary ===")
    git_metrics = metrics.get("git", {})
    github_metrics = metrics.get("github", {})
    quality_metrics = metrics.get("code_quality", {})
    
    print(f"Commits (30 days): {git_metrics.get('commits_last_30_days', 0)}")
    print(f"Contributors: {git_metrics.get('total_contributors', 0)}")
    if github_metrics:
        print(f"GitHub stars: {github_metrics.get('stars', 0)}")
        print(f"Open issues: {github_metrics.get('open_issues_actual', 0)}")
    if "test_coverage" in quality_metrics:
        print(f"Test coverage: {quality_metrics['test_coverage']:.1f}%")
    
    # Exit with error code if any targets are not met
    if comparison:
        failed_targets = [name for name, data in comparison.items() if data["status"] == "fail"]
        if failed_targets:
            print(f"\n❌ Failed targets: {', '.join(failed_targets)}")
            return 1
        else:
            print("\n✅ All targets met!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())