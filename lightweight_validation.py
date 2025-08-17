#!/usr/bin/env python3
"""
Lightweight Validation Framework
Validates PWMK architecture without heavy dependencies
"""

import os
import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class LightweightValidator:
    """Validates PWMK system without requiring torch/heavy deps."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.results = {
            "architecture": {},
            "code_quality": {},
            "documentation": {},
            "deployment": {},
            "overall_score": 0
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("🔍 LIGHTWEIGHT VALIDATION FRAMEWORK")
        print("=" * 60)
        
        self.validate_architecture()
        self.validate_code_quality()
        self.validate_documentation()
        self.validate_deployment()
        self.calculate_overall_score()
        
        return self.results
    
    def validate_architecture(self):
        """Validate system architecture and module structure."""
        print("\n🏗️ ARCHITECTURE VALIDATION")
        print("-" * 40)
        
        # Check core modules exist
        core_modules = [
            "pwmk/core/world_model.py",
            "pwmk/core/beliefs.py", 
            "pwmk/agents/tom_agent.py",
            "pwmk/planning/epistemic.py"
        ]
        
        module_scores = []
        for module in core_modules:
            path = self.repo_root / module
            if path.exists():
                # Check if it's a valid Python file
                try:
                    with open(path, 'r') as f:
                        ast.parse(f.read())
                    print(f"✅ {module}: Valid")
                    module_scores.append(1)
                except SyntaxError:
                    print(f"❌ {module}: Syntax Error")
                    module_scores.append(0)
            else:
                print(f"❌ {module}: Missing")
                module_scores.append(0)
        
        # Check advanced modules
        advanced_modules = [
            "pwmk/quantum/",
            "pwmk/optimization/",
            "pwmk/security/",
            "pwmk/monitoring/"
        ]
        
        advanced_scores = []
        for module in advanced_modules:
            path = self.repo_root / module
            if path.exists() and path.is_dir():
                python_files = list(path.glob("*.py"))
                if python_files:
                    print(f"✅ {module}: {len(python_files)} files")
                    advanced_scores.append(1)
                else:
                    print(f"❌ {module}: No Python files")
                    advanced_scores.append(0)
            else:
                print(f"❌ {module}: Missing")
                advanced_scores.append(0)
        
        self.results["architecture"] = {
            "core_modules": sum(module_scores) / len(module_scores),
            "advanced_modules": sum(advanced_scores) / len(advanced_scores),
            "total_score": (sum(module_scores) + sum(advanced_scores)) / (len(module_scores) + len(advanced_scores))
        }
    
    def validate_code_quality(self):
        """Validate code quality metrics."""
        print("\n📝 CODE QUALITY VALIDATION")
        print("-" * 40)
        
        # Count total lines of code
        python_files = list(self.repo_root.glob("**/*.py"))
        total_lines = 0
        total_files = 0
        
        for file_path in python_files:
            if "/.venv/" in str(file_path) or "__pycache__" in str(file_path):
                continue
            try:
                with open(file_path, 'r') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    total_files += 1
            except Exception:
                continue
        
        print(f"✅ Total Python files: {total_files}")
        print(f"✅ Total lines of code: {total_lines}")
        
        # Check for docstrings in key modules
        docstring_score = 0
        key_files = [
            "pwmk/core/world_model.py",
            "pwmk/agents/tom_agent.py"
        ]
        
        for file_path in key_files:
            path = self.repo_root / file_path
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                        tree = ast.parse(content)
                        
                    # Check for module docstring
                    if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                        isinstance(tree.body[0].value, ast.Constant) and 
                        isinstance(tree.body[0].value.value, str)):
                        print(f"✅ {file_path}: Has module docstring")
                        docstring_score += 1
                    else:
                        print(f"❌ {file_path}: Missing module docstring")
                except Exception:
                    print(f"❌ {file_path}: Parse error")
        
        self.results["code_quality"] = {
            "total_files": total_files,
            "total_lines": total_lines,
            "docstring_score": docstring_score / len(key_files),
            "complexity_score": 1.0 if total_lines > 10000 else total_lines / 10000
        }
    
    def validate_documentation(self):
        """Validate documentation quality."""
        print("\n📚 DOCUMENTATION VALIDATION")
        print("-" * 40)
        
        docs_to_check = [
            "README.md",
            "CONTRIBUTING.md", 
            "docs/DEVELOPMENT.md",
            "ARCHITECTURE.md"
        ]
        
        doc_scores = []
        for doc in docs_to_check:
            path = self.repo_root / doc
            if path.exists():
                with open(path, 'r') as f:
                    content = f.read()
                    word_count = len(content.split())
                    if word_count > 100:
                        print(f"✅ {doc}: {word_count} words")
                        doc_scores.append(1)
                    else:
                        print(f"⚠️ {doc}: Too short ({word_count} words)")
                        doc_scores.append(0.5)
            else:
                print(f"❌ {doc}: Missing")
                doc_scores.append(0)
        
        self.results["documentation"] = {
            "doc_completeness": sum(doc_scores) / len(doc_scores),
            "api_docs": 1.0 if (self.repo_root / "docs").exists() else 0.0
        }
    
    def validate_deployment(self):
        """Validate deployment readiness."""
        print("\n🚀 DEPLOYMENT VALIDATION")
        print("-" * 40)
        
        deployment_files = [
            "Dockerfile",
            "docker-compose.yml", 
            "pyproject.toml",
            "requirements",
            "scripts/"
        ]
        
        deploy_scores = []
        for item in deployment_files:
            if item.endswith("/"):
                path = self.repo_root / item[:-1]
                if path.exists() and path.is_dir():
                    print(f"✅ {item}: Directory exists")
                    deploy_scores.append(1)
                else:
                    print(f"❌ {item}: Missing directory")
                    deploy_scores.append(0)
            else:
                path = self.repo_root / item
                if path.exists():
                    print(f"✅ {item}: File exists")
                    deploy_scores.append(1)
                elif item == "requirements" and (self.repo_root / "pyproject.toml").exists():
                    print(f"✅ {item}: Covered by pyproject.toml")
                    deploy_scores.append(1)
                else:
                    print(f"❌ {item}: Missing")
                    deploy_scores.append(0)
        
        # Check for monitoring setup
        monitoring_score = 0
        if (self.repo_root / "monitoring").exists():
            monitoring_files = list((self.repo_root / "monitoring").glob("**/*"))
            if monitoring_files:
                print(f"✅ Monitoring: {len(monitoring_files)} files")
                monitoring_score = 1
        
        self.results["deployment"] = {
            "container_ready": 1.0 if (self.repo_root / "Dockerfile").exists() else 0.0,
            "orchestration": 1.0 if (self.repo_root / "docker-compose.yml").exists() else 0.0,
            "monitoring": monitoring_score,
            "overall": (sum(deploy_scores) + monitoring_score) / (len(deploy_scores) + 1)
        }
    
    def calculate_overall_score(self):
        """Calculate overall validation score."""
        arch_score = self.results["architecture"]["total_score"]
        quality_score = (self.results["code_quality"]["docstring_score"] + 
                        self.results["code_quality"]["complexity_score"]) / 2
        doc_score = (self.results["documentation"]["doc_completeness"] + 
                    self.results["documentation"]["api_docs"]) / 2
        deploy_score = self.results["deployment"]["overall"]
        
        overall = (arch_score + quality_score + doc_score + deploy_score) / 4
        self.results["overall_score"] = overall
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"🏗️ Architecture Score: {arch_score:.2%}")
        print(f"📝 Code Quality Score: {quality_score:.2%}")
        print(f"📚 Documentation Score: {doc_score:.2%}")
        print(f"🚀 Deployment Score: {deploy_score:.2%}")
        print("-" * 60)
        print(f"🎯 OVERALL SCORE: {overall:.2%}")
        
        if overall >= 0.9:
            print("🏆 EXCELLENT - Production ready!")
        elif overall >= 0.8:
            print("✅ GOOD - Minor improvements needed")
        elif overall >= 0.7:
            print("⚠️ FAIR - Some work required")
        else:
            print("❌ NEEDS WORK - Significant improvements required")


def main():
    """Run lightweight validation."""
    validator = LightweightValidator()
    results = validator.validate_all()
    
    # Save results
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to validation_results.json")
    return results


if __name__ == "__main__":
    main()