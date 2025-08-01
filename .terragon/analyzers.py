#!/usr/bin/env python3
"""
Comprehensive analyzers for discovering value opportunities in AI research repository
Includes technical debt, security, performance, documentation, and testing analyzers
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import yaml
except ImportError:
    yaml = None
from value_scoring import ValueItem, Category


@dataclass
class AnalysisResult:
    """Result from running an analyzer"""
    analyzer_name: str
    items_discovered: List[ValueItem]
    metadata: Dict[str, Any]
    execution_time: float


class BaseAnalyzer(ABC):
    """Base class for all value discovery analyzers"""
    
    def __init__(self, config: Dict[str, Any], repo_path: str = "."):
        self.config = config
        self.repo_path = Path(repo_path)
        self.analyzer_config = config.get("discovery", {}).get(self.__class__.__name__.lower().replace("analyzer", ""), {})
    
    @abstractmethod
    def analyze(self) -> AnalysisResult:
        """Run the analyzer and return discovered value items"""
        pass
    
    def _is_enabled(self) -> bool:
        """Check if this analyzer is enabled in configuration"""
        return self.analyzer_config.get("enabled", True)
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the repository"""
        python_files = []
        for pattern in ["**/*.py"]:
            python_files.extend(self.repo_path.glob(pattern))
        return [f for f in python_files if not any(part.startswith('.') for part in f.parts)]


class TechnicalDebtAnalyzer(BaseAnalyzer):
    """Analyzes technical debt in Python AI research code"""
    
    def analyze(self) -> AnalysisResult:
        """Analyze technical debt across the codebase"""
        import time
        start_time = time.time()
        
        if not self._is_enabled():
            return AnalysisResult("TechnicalDebtAnalyzer", [], {}, 0.0)
        
        items = []
        metadata = {
            "files_analyzed": 0,
            "complexity_violations": 0,
            "code_smells": 0,
            "ai_specific_issues": 0
        }
        
        python_files = self._get_python_files()
        
        for file_path in python_files:
            try:
                file_items, file_metadata = self._analyze_file(file_path)
                items.extend(file_items)
                
                # Update metadata
                metadata["files_analyzed"] += 1
                metadata["complexity_violations"] += file_metadata.get("complexity_violations", 0)
                metadata["code_smells"] += file_metadata.get("code_smells", 0)
                metadata["ai_specific_issues"] += file_metadata.get("ai_specific_issues", 0)
                
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
        
        execution_time = time.time() - start_time
        return AnalysisResult("TechnicalDebtAnalyzer", items, metadata, execution_time)
    
    def _analyze_file(self, file_path: Path) -> Tuple[List[ValueItem], Dict[str, Any]]:
        """Analyze a single Python file for technical debt"""
        items = []
        metadata = {"complexity_violations": 0, "code_smells": 0, "ai_specific_issues": 0}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
        except:
            return items, metadata
        
        # Analyze complexity
        complexity_items = self._analyze_complexity(tree, file_path, content)
        items.extend(complexity_items)
        metadata["complexity_violations"] = len(complexity_items)
        
        # Analyze code smells
        smell_items = self._analyze_code_smells(tree, file_path, content)
        items.extend(smell_items)
        metadata["code_smells"] = len(smell_items)
        
        # AI/ML specific analysis
        ai_items = self._analyze_ai_specific_issues(tree, file_path, content)
        items.extend(ai_items)
        metadata["ai_specific_issues"] = len(ai_items)
        
        return items, metadata
    
    def _analyze_complexity(self, tree: ast.AST, file_path: Path, content: str) -> List[ValueItem]:
        """Analyze cyclomatic complexity"""
        items = []
        max_complexity = self.analyzer_config.get("max_complexity", 10)
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity_violations = []
            
            def visit_FunctionDef(self, node):
                complexity = self._calculate_complexity(node)
                if complexity > max_complexity:
                    self.complexity_violations.append((node, complexity))
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
            
            def _calculate_complexity(self, node):
                """Simple complexity calculation"""
                complexity = 1  # Base complexity
                
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                        complexity += 1
                    elif isinstance(child, ast.ExceptHandler):
                        complexity += 1
                    elif isinstance(child, (ast.And, ast.Or)):
                        complexity += 1
                
                return complexity
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        for node, complexity in visitor.complexity_violations:
            items.append(ValueItem(
                id=f"complexity-{file_path.stem}-{node.name}-{node.lineno}",
                title=f"High complexity function: {node.name}",
                description=f"Function '{node.name}' has complexity {complexity} (max: {max_complexity})",
                category=Category.TECHNICAL_DEBT,
                business_value=4.0,
                time_criticality=3.0,
                risk_reduction=6.0,
                effort_estimate=complexity / 3.0,  # Rough estimate
                impact=5.0,
                confidence=8.0,
                ease=4.0,
                complexity_score=min(complexity / 2.0, 10.0),
                maintainability_score=8.0,
                test_coverage_impact=6.0,
                documentation_gap=3.0,
                security_risk=2.0,
                source="technical_debt_analyzer",
                file_paths=[str(file_path)]
            ))
        
        return items
    
    def _analyze_code_smells(self, tree: ast.AST, file_path: Path, content: str) -> List[ValueItem]:
        """Analyze common code smells"""
        items = []
        
        # Long parameter lists
        class LongParameterVisitor(ast.NodeVisitor):
            def __init__(self):
                self.long_param_functions = []
            
            def visit_FunctionDef(self, node):
                if len(node.args.args) > 6:  # Configurable threshold
                    self.long_param_functions.append(node)
                self.generic_visit(node)
        
        visitor = LongParameterVisitor()
        visitor.visit(tree)
        
        for node in visitor.long_param_functions:
            items.append(ValueItem(
                id=f"long-params-{file_path.stem}-{node.name}-{node.lineno}",
                title=f"Long parameter list: {node.name}",
                description=f"Function '{node.name}' has {len(node.args.args)} parameters",
                category=Category.TECHNICAL_DEBT,
                business_value=3.0,
                time_criticality=2.0,
                risk_reduction=4.0,
                effort_estimate=2.0,
                impact=4.0,
                confidence=7.0,
                ease=6.0,
                complexity_score=5.0,
                maintainability_score=7.0,
                test_coverage_impact=3.0,
                documentation_gap=4.0,
                security_risk=1.0,
                source="technical_debt_analyzer",
                file_paths=[str(file_path)]
            ))
        
        # TODO/FIXME comments
        todo_pattern = re.compile(r'#\s*(TODO|FIXME|BUG|HACK)[\s:]*(.+)', re.IGNORECASE)
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            match = todo_pattern.search(line)
            if match:
                todo_type = match.group(1).upper()
                todo_text = match.group(2).strip()
                
                priority_map = {"BUG": 8.0, "FIXME": 7.0, "TODO": 5.0, "HACK": 6.0}
                priority = priority_map.get(todo_type, 5.0)
                
                items.append(ValueItem(
                    id=f"todo-{file_path.stem}-{i}",
                    title=f"{todo_type}: {todo_text[:50]}...",
                    description=f"{todo_type} comment in {file_path}: {todo_text}",
                    category=Category.TECHNICAL_DEBT,
                    business_value=priority,
                    time_criticality=4.0,
                    risk_reduction=priority,
                    effort_estimate=2.0,
                    impact=priority,
                    confidence=6.0,
                    ease=7.0,
                    complexity_score=3.0,
                    maintainability_score=priority,
                    test_coverage_impact=2.0,
                    documentation_gap=5.0,
                    security_risk=1.0 if todo_type != "BUG" else 4.0,
                    source="technical_debt_analyzer",
                    file_paths=[str(file_path)]
                ))
        
        return items
    
    def _analyze_ai_specific_issues(self, tree: ast.AST, file_path: Path, content: str) -> List[ValueItem]:
        """Analyze AI/ML specific technical debt"""
        items = []
        
        # Check for hardcoded hyperparameters
        class HyperparameterVisitor(ast.NodeVisitor):
            def __init__(self):
                self.hardcoded_values = []
            
            def visit_Assign(self, node):
                # Look for common ML hyperparameter patterns
                ml_keywords = ['lr', 'learning_rate', 'batch_size', 'epochs', 'hidden_dim', 'dropout']
                
                for target in node.targets:
                    if isinstance(target, ast.Name) and any(keyword in target.id.lower() for keyword in ml_keywords):
                        if isinstance(node.value, ast.Constant):
                            self.hardcoded_values.append((target.id, node.lineno))
                
                self.generic_visit(node)
        
        visitor = HyperparameterVisitor()
        visitor.visit(tree)
        
        for param_name, line_no in visitor.hardcoded_values:
            items.append(ValueItem(
                id=f"hardcoded-hyperparam-{file_path.stem}-{param_name}-{line_no}",
                title=f"Hardcoded hyperparameter: {param_name}",
                description=f"Hyperparameter '{param_name}' is hardcoded, should be configurable",
                category=Category.TECHNICAL_DEBT,
                business_value=6.0,
                time_criticality=3.0,
                risk_reduction=5.0,
                effort_estimate=1.5,
                impact=6.0,
                confidence=8.0,
                ease=8.0,
                complexity_score=2.0,
                maintainability_score=7.0,
                test_coverage_impact=4.0,
                documentation_gap=3.0,
                security_risk=1.0,
                research_novelty=0.0,
                benchmark_impact=4.0,
                reproducibility_impact=8.0,
                community_value=5.0,
                source="technical_debt_analyzer",
                file_paths=[str(file_path)]
            ))
        
        # Check for missing error handling in model operations
        model_operations = ['forward', 'train', 'eval', 'predict', 'fit']
        
        class ModelErrorHandlingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.missing_error_handling = []
            
            def visit_FunctionDef(self, node):
                if any(op in node.name.lower() for op in model_operations):
                    # Check if function has try-except blocks
                    has_try_except = any(isinstance(child, ast.Try) for child in ast.walk(node))
                    if not has_try_except:
                        self.missing_error_handling.append(node)
                
                self.generic_visit(node)
        
        error_visitor = ModelErrorHandlingVisitor()
        error_visitor.visit(tree)
        
        for node in error_visitor.missing_error_handling:
            items.append(ValueItem(
                id=f"missing-error-handling-{file_path.stem}-{node.name}-{node.lineno}",
                title=f"Missing error handling: {node.name}",
                description=f"Model operation '{node.name}' lacks proper error handling",
                category=Category.TECHNICAL_DEBT,
                business_value=7.0,
                time_criticality=5.0,
                risk_reduction=8.0,
                effort_estimate=2.0,
                impact=7.0,
                confidence=7.0,
                ease=6.0,
                complexity_score=4.0,
                maintainability_score=8.0,
                test_coverage_impact=6.0,
                documentation_gap=2.0,
                security_risk=3.0,
                research_novelty=0.0,
                benchmark_impact=2.0,
                reproducibility_impact=6.0,
                community_value=4.0,
                source="technical_debt_analyzer",
                file_paths=[str(file_path)]
            ))
        
        return items


class SecurityAnalyzer(BaseAnalyzer):
    """Analyzes security vulnerabilities in AI/ML dependencies and code"""
    
    def analyze(self) -> AnalysisResult:
        """Analyze security vulnerabilities"""
        import time
        start_time = time.time()
        
        if not self._is_enabled():
            return AnalysisResult("SecurityAnalyzer", [], {}, 0.0)
        
        items = []
        metadata = {
            "dependency_vulnerabilities": 0,
            "code_security_issues": 0,
            "ai_security_concerns": 0
        }
        
        # Analyze dependencies
        dep_items = self._analyze_dependencies()
        items.extend(dep_items)
        metadata["dependency_vulnerabilities"] = len(dep_items)
        
        # Analyze code for security issues
        code_items = self._analyze_code_security()
        items.extend(code_items)
        metadata["code_security_issues"] = len(code_items)
        
        # AI/ML specific security analysis
        ai_items = self._analyze_ai_security()
        items.extend(ai_items)
        metadata["ai_security_concerns"] = len(ai_items)
        
        execution_time = time.time() - start_time
        return AnalysisResult("SecurityAnalyzer", items, metadata, execution_time)
    
    def _analyze_dependencies(self) -> List[ValueItem]:
        """Analyze dependencies for known vulnerabilities"""
        items = []
        
        # Look for requirements files
        req_files = list(self.repo_path.glob("**/requirements*.txt")) + \
                   list(self.repo_path.glob("**/requirements*.in")) + \
                   [self.repo_path / "pyproject.toml"]
        
        vulnerable_packages = {
            # Common vulnerable packages (this would be populated from CVE databases)
            "tensorflow": ["<2.9.0", "Potential model extraction vulnerability"],
            "torch": ["<1.13.0", "Arbitrary code execution in pickle loading"],
            "pillow": ["<8.3.2", "Buffer overflow in image processing"],
            "numpy": ["<1.21.0", "Buffer overflow in array operations"],
        }
        
        for req_file in req_files:
            if req_file.exists():
                try:
                    content = req_file.read_text()
                    for package, (vulnerable_version, description) in vulnerable_packages.items():
                        if package in content.lower():
                            items.append(ValueItem(
                                id=f"vuln-{package}-{req_file.name}",
                                title=f"Potential vulnerability: {package}",
                                description=f"Package '{package}' may be vulnerable: {description}",
                                category=Category.SECURITY,
                                business_value=8.0,
                                time_criticality=9.0,
                                risk_reduction=9.0,
                                effort_estimate=1.0,
                                impact=8.0,
                                confidence=6.0,  # Would be higher with actual CVE checking
                                ease=9.0,
                                complexity_score=2.0,
                                maintainability_score=3.0,
                                test_coverage_impact=2.0,
                                documentation_gap=1.0,
                                security_risk=9.0,
                                source="security_analyzer",
                                file_paths=[str(req_file)]
                            ))
                except Exception as e:
                    continue
        
        return items
    
    def _analyze_code_security(self) -> List[ValueItem]:
        """Analyze code for security issues"""
        items = []
        
        python_files = self._get_python_files()
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                
                # Check for pickle usage (dangerous for ML models)
                if 'pickle.load' in content or 'pickle.loads' in content:
                    items.append(ValueItem(
                        id=f"pickle-security-{file_path.stem}",
                        title=f"Unsafe pickle usage in {file_path.name}",
                        description="Using pickle.load/loads can execute arbitrary code",
                        category=Category.SECURITY,
                        business_value=7.0,
                        time_criticality=8.0,
                        risk_reduction=9.0,
                        effort_estimate=3.0,
                        impact=8.0,
                        confidence=9.0,
                        ease=6.0,
                        complexity_score=4.0,
                        maintainability_score=5.0,
                        test_coverage_impact=3.0,
                        documentation_gap=2.0,
                        security_risk=9.0,
                        source="security_analyzer",
                        file_paths=[str(file_path)]
                    ))
                
                # Check for hardcoded secrets/keys
                secret_patterns = [
                    r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
                    r'secret[_-]?key\s*=\s*["\'][^"\']+["\']',
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'token\s*=\s*["\'][^"\']+["\']'
                ]
                
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        items.append(ValueItem(
                            id=f"hardcoded-secret-{file_path.stem}",
                            title=f"Hardcoded secret in {file_path.name}",
                            description="Found potential hardcoded secret or API key",
                            category=Category.SECURITY,
                            business_value=8.0,
                            time_criticality=9.0,
                            risk_reduction=9.0,
                            effort_estimate=2.0,
                            impact=9.0,
                            confidence=7.0,
                            ease=8.0,
                            complexity_score=2.0,
                            maintainability_score=4.0,
                            test_coverage_impact=2.0,
                            documentation_gap=1.0,
                            security_risk=10.0,
                            source="security_analyzer",
                            file_paths=[str(file_path)]
                        ))
                        break  # Only report once per file
                
            except Exception:
                continue
        
        return items
    
    def _analyze_ai_security(self) -> List[ValueItem]:
        """Analyze AI/ML specific security concerns"""
        items = []
        
        # Check for model serialization without validation
        python_files = self._get_python_files()
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                # Look for torch.load without map_location or weights_only
                class TorchLoadVisitor(ast.NodeVisitor):
                    def __init__(self):
                        self.unsafe_loads = []
                    
                    def visit_Call(self, node):
                        if (isinstance(node.func, ast.Attribute) and 
                            isinstance(node.func.value, ast.Name) and
                            node.func.value.id == 'torch' and
                            node.func.attr == 'load'):
                            
                            # Check if weights_only=True is used
                            has_weights_only = any(
                                isinstance(kw, ast.keyword) and 
                                kw.arg == 'weights_only' and
                                isinstance(kw.value, ast.Constant) and
                                kw.value.value is True
                                for kw in node.keywords
                            )
                            
                            if not has_weights_only:
                                self.unsafe_loads.append(node.lineno)
                        
                        self.generic_visit(node)
                
                visitor = TorchLoadVisitor()
                visitor.visit(tree)
                
                for line_no in visitor.unsafe_loads:
                    items.append(ValueItem(
                        id=f"unsafe-torch-load-{file_path.stem}-{line_no}",
                        title=f"Unsafe torch.load in {file_path.name}",
                        description="torch.load without weights_only=True can execute arbitrary code",
                        category=Category.SECURITY,
                        business_value=8.0,
                        time_criticality=7.0,
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
                        file_paths=[str(file_path)]
                    ))
                
            except Exception:
                continue
        
        return items


class PerformanceAnalyzer(BaseAnalyzer):
    """Analyzes performance optimization opportunities for neural networks"""
    
    def analyze(self) -> AnalysisResult:
        """Analyze performance optimization opportunities"""
        import time
        start_time = time.time()
        
        if not self._is_enabled():
            return AnalysisResult("PerformanceAnalyzer", [], {}, 0.0)
        
        items = []
        metadata = {
            "neural_network_optimizations": 0,
            "memory_optimizations": 0,
            "compute_optimizations": 0
        }
        
        # Analyze neural network performance opportunities
        nn_items = self._analyze_neural_networks()
        items.extend(nn_items)
        metadata["neural_network_optimizations"] = len(nn_items)
        
        # Memory optimization opportunities
        mem_items = self._analyze_memory_usage()
        items.extend(mem_items)
        metadata["memory_optimizations"] = len(mem_items)
        
        # Compute optimization opportunities
        compute_items = self._analyze_compute_efficiency()
        items.extend(compute_items)
        metadata["compute_optimizations"] = len(compute_items)
        
        execution_time = time.time() - start_time
        return AnalysisResult("PerformanceAnalyzer", items, metadata, execution_time)
    
    def _analyze_neural_networks(self) -> List[ValueItem]:
        """Analyze neural network implementation for optimization opportunities"""
        items = []
        python_files = self._get_python_files()
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                
                # Check for missing torch.compile usage (PyTorch 2.0+)
                if 'torch.nn' in content and 'torch.compile' not in content:
                    if any(pattern in content for pattern in ['class', 'def forward']):
                        items.append(ValueItem(
                            id=f"missing-torch-compile-{file_path.stem}",
                            title=f"Missing torch.compile optimization in {file_path.name}",
                            description="Consider using torch.compile for 2x performance improvement",
                            category=Category.PERFORMANCE,
                            business_value=7.0,
                            time_criticality=4.0,
                            risk_reduction=3.0,
                            effort_estimate=2.0,
                            impact=8.0,
                            confidence=8.0,
                            ease=7.0,
                            complexity_score=3.0,
                            maintainability_score=4.0,
                            test_coverage_impact=4.0,
                            documentation_gap=3.0,
                            security_risk=1.0,
                            research_novelty=2.0,
                            benchmark_impact=9.0,
                            reproducibility_impact=3.0,
                            community_value=6.0,
                            source="performance_analyzer",
                            file_paths=[str(file_path)]
                        ))
                
                # Check for inefficient attention implementations
                if 'attention' in content.lower() and 'scaled_dot_product_attention' not in content:
                    items.append(ValueItem(
                        id=f"inefficient-attention-{file_path.stem}",
                        title=f"Potentially inefficient attention in {file_path.name}",
                        description="Consider using torch.nn.functional.scaled_dot_product_attention for better performance",
                        category=Category.PERFORMANCE,
                        business_value=6.0,
                        time_criticality=3.0,
                        risk_reduction=2.0,
                        effort_estimate=4.0,
                        impact=7.0,
                        confidence=6.0,
                        ease=5.0,
                        complexity_score=5.0,
                        maintainability_score=5.0,
                        test_coverage_impact=5.0,
                        documentation_gap=4.0,
                        security_risk=1.0,
                        research_novelty=3.0,
                        benchmark_impact=8.0,
                        reproducibility_impact=4.0,
                        community_value=5.0,
                        source="performance_analyzer",
                        file_paths=[str(file_path)]
                    ))
                
            except Exception:
                continue
        
        return items
    
    def _analyze_memory_usage(self) -> List[ValueItem]:
        """Analyze memory usage patterns"""
        items = []
        python_files = self._get_python_files()
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                
                # Check for potential memory leaks in training loops
                if 'for epoch' in content and '.backward()' in content:
                    if 'torch.cuda.empty_cache()' not in content and 'del ' not in content:
                        items.append(ValueItem(
                            id=f"memory-leak-training-{file_path.stem}",
                            title=f"Potential memory leak in training loop in {file_path.name}",
                            description="Training loop may accumulate memory, consider adding torch.cuda.empty_cache()",
                            category=Category.PERFORMANCE,
                            business_value=6.0,
                            time_criticality=5.0,
                            risk_reduction=7.0,
                            effort_estimate=1.0,
                            impact=7.0,
                            confidence=7.0,
                            ease=9.0,
                            complexity_score=2.0,
                            maintainability_score=6.0,
                            test_coverage_impact=3.0,
                            documentation_gap=2.0,
                            security_risk=1.0,
                            research_novelty=1.0,
                            benchmark_impact=5.0,
                            reproducibility_impact=6.0,
                            community_value=4.0,
                            source="performance_analyzer",
                            file_paths=[str(file_path)]
                        ))
                
            except Exception:
                continue
        
        return items
    
    def _analyze_compute_efficiency(self) -> List[ValueItem]:
        """Analyze compute efficiency opportunities"""
        items = []
        python_files = self._get_python_files()
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                
                # Check for CPU-bound operations that could be GPU-accelerated
                cpu_patterns = [
                    'numpy.',
                    'scipy.',
                    'pandas.',
                    'sklearn.'
                ]
                
                has_torch = 'torch' in content
                has_cpu_ops = any(pattern in content for pattern in cpu_patterns)
                
                if has_torch and has_cpu_ops and '.cuda' not in content and '.to(device)' not in content:
                    items.append(ValueItem(
                        id=f"cpu-bound-ops-{file_path.stem}",
                        title=f"CPU-bound operations in {file_path.name}",
                        description="Consider moving computations to GPU for better performance",
                        category=Category.PERFORMANCE,
                        business_value=5.0,
                        time_criticality=3.0,
                        risk_reduction=2.0,
                        effort_estimate=3.0,
                        impact=6.0,
                        confidence=5.0,
                        ease=4.0,
                        complexity_score=4.0,
                        maintainability_score=4.0,
                        test_coverage_impact=4.0,
                        documentation_gap=3.0,
                        security_risk=1.0,
                        research_novelty=2.0,
                        benchmark_impact=7.0,
                        reproducibility_impact=3.0,
                        community_value=4.0,
                        source="performance_analyzer",
                        file_paths=[str(file_path)]
                    ))
                
            except Exception:
                continue
        
        return items


class DocumentationAnalyzer(BaseAnalyzer):
    """Analyzes documentation gaps in AI research projects"""
    
    def analyze(self) -> AnalysisResult:
        """Analyze documentation completeness"""
        import time
        start_time = time.time()
        
        if not self._is_enabled():
            return AnalysisResult("DocumentationAnalyzer", [], {}, 0.0)
        
        items = []
        metadata = {
            "missing_docstrings": 0,
            "api_coverage_gaps": 0,
            "research_doc_gaps": 0
        }
        
        # Analyze docstring coverage
        docstring_items = self._analyze_docstrings()
        items.extend(docstring_items)
        metadata["missing_docstrings"] = len(docstring_items)
        
        # Analyze API documentation
        api_items = self._analyze_api_documentation()
        items.extend(api_items)
        metadata["api_coverage_gaps"] = len(api_items)
        
        # Research-specific documentation
        research_items = self._analyze_research_documentation()
        items.extend(research_items)
        metadata["research_doc_gaps"] = len(research_items)
        
        execution_time = time.time() - start_time
        return AnalysisResult("DocumentationAnalyzer", items, metadata, execution_time)
    
    def _analyze_docstrings(self) -> List[ValueItem]:
        """Analyze missing docstrings"""
        items = []
        python_files = self._get_python_files()
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                class DocstringVisitor(ast.NodeVisitor):
                    def __init__(self):
                        self.missing_docstrings = []
                    
                    def visit_FunctionDef(self, node):
                        if not self._has_docstring(node) and not node.name.startswith('_'):
                            self.missing_docstrings.append(('function', node.name, node.lineno))
                        self.generic_visit(node)
                    
                    def visit_ClassDef(self, node):
                        if not self._has_docstring(node):
                            self.missing_docstrings.append(('class', node.name, node.lineno))
                        self.generic_visit(node)
                    
                    def _has_docstring(self, node):
                        return (node.body and 
                                isinstance(node.body[0], ast.Expr) and
                                isinstance(node.body[0].value, ast.Constant) and
                                isinstance(node.body[0].value.value, str))
                
                visitor = DocstringVisitor()
                visitor.visit(tree)
                
                for item_type, name, line_no in visitor.missing_docstrings:
                    items.append(ValueItem(
                        id=f"missing-docstring-{file_path.stem}-{name}-{line_no}",
                        title=f"Missing docstring: {item_type} {name}",
                        description=f"{item_type.title()} '{name}' in {file_path.name} lacks documentation",
                        category=Category.DOCUMENTATION,
                        business_value=4.0,
                        time_criticality=2.0,
                        risk_reduction=3.0,
                        effort_estimate=0.5,
                        impact=5.0,
                        confidence=9.0,
                        ease=9.0,
                        complexity_score=1.0,
                        maintainability_score=6.0,
                        test_coverage_impact=2.0,
                        documentation_gap=9.0,
                        security_risk=0.0,
                        research_novelty=1.0,
                        benchmark_impact=1.0,
                        reproducibility_impact=5.0,
                        community_value=7.0,
                        source="documentation_analyzer",
                        file_paths=[str(file_path)]
                    ))
                
            except Exception:
                continue
        
        return items
    
    def _analyze_api_documentation(self) -> List[ValueItem]:
        """Analyze API documentation coverage"""
        items = []
        
        # Check if main modules have proper API documentation
        main_modules = ['pwmk/core/', 'pwmk/agents/', 'pwmk/planning/']
        
        for module_path in main_modules:
            full_path = self.repo_path / module_path
            if full_path.exists():
                # Check for __init__.py with proper exports
                init_file = full_path / '__init__.py'
                if init_file.exists():
                    try:
                        content = init_file.read_text()
                        if '__all__' not in content:
                            items.append(ValueItem(
                                id=f"missing-api-exports-{module_path.replace('/', '-')}",
                                title=f"Missing API exports in {module_path}__init__.py",
                                description=f"Module {module_path} lacks __all__ definition for clear API",
                                category=Category.DOCUMENTATION,
                                business_value=5.0,
                                time_criticality=2.0,
                                risk_reduction=4.0,
                                effort_estimate=1.0,
                                impact=6.0,
                                confidence=8.0,
                                ease=8.0,
                                complexity_score=2.0,
                                maintainability_score=7.0,
                                test_coverage_impact=2.0,
                                documentation_gap=8.0,
                                security_risk=0.0,
                                research_novelty=1.0,
                                benchmark_impact=1.0,
                                reproducibility_impact=4.0,
                                community_value=8.0,
                                source="documentation_analyzer",
                                file_paths=[str(init_file)]
                            ))
                    except Exception:
                        continue
        
        return items
    
    def _analyze_research_documentation(self) -> List[ValueItem]:
        """Analyze research-specific documentation needs"""
        items = []
        
        # Check for examples and tutorials
        examples_dir = self.repo_path / 'examples'
        if not examples_dir.exists():
            items.append(ValueItem(
                id="missing-examples-directory",
                title="Missing examples directory",
                description="AI research project should have examples/ directory with usage demonstrations",
                category=Category.DOCUMENTATION,
                business_value=7.0,
                time_criticality=3.0,
                risk_reduction=2.0,
                effort_estimate=8.0,
                impact=8.0,
                confidence=9.0,
                ease=6.0,
                complexity_score=3.0,
                maintainability_score=5.0,
                test_coverage_impact=3.0,
                documentation_gap=10.0,
                security_risk=0.0,
                research_novelty=3.0,
                benchmark_impact=2.0,
                reproducibility_impact=8.0,
                community_value=9.0,
                source="documentation_analyzer",
                file_paths=[]
            ))
        
        # Check for benchmarks documentation
        benchmarks_doc = self.repo_path / 'docs' / 'BENCHMARKS.md'
        if not benchmarks_doc.exists():
            items.append(ValueItem(
                id="missing-benchmarks-documentation",
                title="Missing benchmarks documentation",
                description="AI research project should document performance benchmarks and comparisons",
                category=Category.DOCUMENTATION,
                business_value=6.0,
                time_criticality=4.0,
                risk_reduction=3.0,
                effort_estimate=4.0,
                impact=7.0,
                confidence=8.0,
                ease=7.0,
                complexity_score=2.0,
                maintainability_score=4.0,
                test_coverage_impact=2.0,
                documentation_gap=8.0,
                security_risk=0.0,
                research_novelty=5.0,
                benchmark_impact=8.0,
                reproducibility_impact=7.0,
                community_value=8.0,
                source="documentation_analyzer",
                file_paths=[]
            ))
        
        return items


class TestCoverageAnalyzer(BaseAnalyzer):
    """Analyzes test coverage and suggests improvements"""
    
    def analyze(self) -> AnalysisResult:
        """Analyze test coverage and identify gaps"""
        import time
        start_time = time.time()
        
        if not self._is_enabled():
            return AnalysisResult("TestCoverageAnalyzer", [], {}, 0.0)
        
        items = []
        metadata = {
            "untested_modules": 0,
            "missing_integration_tests": 0,
            "ai_specific_tests": 0
        }
        
        # Analyze module coverage
        module_items = self._analyze_module_coverage()
        items.extend(module_items)
        metadata["untested_modules"] = len(module_items)
        
        # Check for integration tests
        integration_items = self._analyze_integration_tests()
        items.extend(integration_items)
        metadata["missing_integration_tests"] = len(integration_items)
        
        # AI-specific testing needs
        ai_test_items = self._analyze_ai_testing_needs()
        items.extend(ai_test_items)
        metadata["ai_specific_tests"] = len(ai_test_items)
        
        execution_time = time.time() - start_time
        return AnalysisResult("TestCoverageAnalyzer", items, metadata, execution_time)
    
    def _analyze_module_coverage(self) -> List[ValueItem]:
        """Check which modules lack corresponding test files"""
        items = []
        
        # Get all Python modules
        python_files = self._get_python_files()
        test_files = set()
        
        # Find existing test files
        tests_dir = self.repo_path / 'tests'
        if tests_dir.exists():
            for test_file in tests_dir.rglob('test_*.py'):
                test_files.add(test_file.stem)
        
        # Check which modules lack tests
        for py_file in python_files:
            if py_file.parts[0] == 'pwmk':  # Only check main package files
                expected_test_name = f"test_{py_file.stem}"
                if expected_test_name not in test_files and py_file.stem != '__init__':
                    items.append(ValueItem(
                        id=f"missing-test-{py_file.stem}",
                        title=f"Missing tests for {py_file.name}",
                        description=f"Module {py_file} lacks corresponding test file",
                        category=Category.TESTING,
                        business_value=5.0,
                        time_criticality=3.0,
                        risk_reduction=7.0,
                        effort_estimate=4.0,
                        impact=6.0,
                        confidence=8.0,
                        ease=6.0,
                        complexity_score=3.0,
                        maintainability_score=7.0,
                        test_coverage_impact=10.0,
                        documentation_gap=2.0,
                        security_risk=1.0,
                        research_novelty=1.0,
                        benchmark_impact=2.0,
                        reproducibility_impact=8.0,
                        community_value=6.0,
                        source="test_coverage_analyzer",
                        file_paths=[str(py_file)]
                    ))
        
        return items
    
    def _analyze_integration_tests(self) -> List[ValueItem]:
        """Check for comprehensive integration tests"""
        items = []
        
        # Check if integration tests exist
        integration_dir = self.repo_path / 'tests' / 'integration'
        if not integration_dir.exists() or not list(integration_dir.glob('*.py')):
            items.append(ValueItem(
                id="missing-integration-tests",
                title="Missing integration tests",
                description="AI research project needs integration tests for end-to-end workflows",
                category=Category.TESTING,
                business_value=7.0,
                time_criticality=4.0,
                risk_reduction=8.0,
                effort_estimate=8.0,
                impact=8.0,
                confidence=9.0,
                ease=5.0,
                complexity_score=5.0,
                maintainability_score=8.0,
                test_coverage_impact=9.0,
                documentation_gap=3.0,
                security_risk=2.0,
                research_novelty=2.0,
                benchmark_impact=4.0,
                reproducibility_impact=9.0,
                community_value=7.0,
                source="test_coverage_analyzer",
                file_paths=[]
            ))
        
        return items
    
    def _analyze_ai_testing_needs(self) -> List[ValueItem]:
        """Analyze AI/ML specific testing requirements"""
        items = []
        
        # Check for model testing
        model_files = list(self.repo_path.glob('**/models/**/*.py')) + \
                     list(self.repo_path.glob('**/*model*.py'))
        
        if model_files:
            # Check for model validation tests
            has_model_tests = any('model' in test_file.name 
                                for test_file in self.repo_path.glob('**/test_*model*.py'))
            
            if not has_model_tests:
                items.append(ValueItem(
                    id="missing-model-validation-tests",
                    title="Missing model validation tests",
                    description="Neural network models need validation tests for architecture and output shapes",
                    category=Category.TESTING,
                    business_value=8.0,
                    time_criticality=5.0,
                    risk_reduction=8.0,
                    effort_estimate=6.0,
                    impact=8.0,
                    confidence=9.0,
                    ease=6.0,
                    complexity_score=4.0,
                    maintainability_score=7.0,
                    test_coverage_impact=9.0,
                    documentation_gap=2.0,
                    security_risk=2.0,
                    research_novelty=3.0,
                    benchmark_impact=6.0,
                    reproducibility_impact=9.0,
                    community_value=7.0,
                    source="test_coverage_analyzer",
                    file_paths=[str(f) for f in model_files[:3]]  # First 3 files
                ))
        
        # Check for belief reasoning tests (specific to this project)
        belief_files = list(self.repo_path.glob('**/beliefs.py')) + \
                      list(self.repo_path.glob('**/belief*.py'))
        
        if belief_files:
            has_belief_tests = any('belief' in test_file.name 
                                 for test_file in self.repo_path.glob('**/test_*belief*.py'))
            
            if not has_belief_tests:
                items.append(ValueItem(
                    id="missing-belief-reasoning-tests",
                    title="Missing belief reasoning tests",
                    description="Theory of Mind belief reasoning needs comprehensive test coverage",
                    category=Category.TESTING,
                    business_value=9.0,
                    time_criticality=6.0,
                    risk_reduction=9.0,
                    effort_estimate=7.0,
                    impact=9.0,
                    confidence=9.0,
                    ease=5.0,
                    complexity_score=6.0,
                    maintainability_score=8.0,
                    test_coverage_impact=10.0,
                    documentation_gap=3.0,
                    security_risk=1.0,
                    research_novelty=8.0,
                    benchmark_impact=7.0,
                    reproducibility_impact=10.0,
                    community_value=8.0,
                    source="test_coverage_analyzer",
                    file_paths=[str(f) for f in belief_files]
                ))
        
        return items


if __name__ == "__main__":
    # Test the analyzers
    try:
        import yaml
    except ImportError:
        yaml = None
    
    # Load config
    with open('.terragon/config.yaml', 'r') as f:
        if yaml:
            config = yaml.safe_load(f)
        else:
            config = {"scoring": {"weights": {"advanced": {"wsjf": 0.5}}}}
    
    # Run technical debt analyzer
    debt_analyzer = TechnicalDebtAnalyzer(config)
    result = debt_analyzer.analyze()
    
    print(f"Technical Debt Analysis Results:")
    print(f"- Items discovered: {len(result.items_discovered)}")
    print(f"- Execution time: {result.execution_time:.2f}s")
    print(f"- Metadata: {result.metadata}")
    
    if result.items_discovered:
        print(f"\nTop 3 issues:")
        for item in result.items_discovered[:3]:
            print(f"- {item.title} (Score: {item.final_score:.2f})")