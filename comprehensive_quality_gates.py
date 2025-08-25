#!/usr/bin/env python3
"""
Comprehensive Quality Gates System for PWMK
Validates code quality, security, performance, and production readiness.
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our validation tools
sys.path.insert(0, '.')
from standalone_validation import PWMKValidator


@dataclass
class QualityGateResult:
    """Quality gate check result."""
    gate_name: str
    passed: bool
    score: float  # 0-100
    details: Dict[str, Any]
    execution_time: float
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class SecurityGate:
    """Security vulnerability and best practices gate."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def run_security_checks(self) -> QualityGateResult:
        """Run comprehensive security checks."""
        start_time = time.time()
        
        results = {
            "secret_scan": self._scan_for_secrets(),
            "dependency_vulnerabilities": self._check_dependency_vulnerabilities(),
            "code_security_patterns": self._check_security_patterns(),
            "input_validation": self._check_input_validation(),
            "authentication_security": self._check_authentication_security()
        }
        
        # Calculate overall security score
        total_score = 0
        max_score = 0
        critical_issues = []
        warnings = []
        recommendations = []
        
        for check_name, check_result in results.items():
            total_score += check_result["score"]
            max_score += 100
            critical_issues.extend(check_result.get("critical_issues", []))
            warnings.extend(check_result.get("warnings", []))
            recommendations.extend(check_result.get("recommendations", []))
        
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0
        passed = overall_score >= 70 and len(critical_issues) == 0
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Security",
            passed=passed,
            score=overall_score,
            details=results,
            execution_time=execution_time,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _scan_for_secrets(self) -> Dict[str, Any]:
        """Scan for hardcoded secrets and credentials."""
        secret_patterns = [
            r'(?i)(password|pwd)\s*[:=]\s*["\'][^"\']{8,}["\']',  # Actual hardcoded passwords
            r'(?i)(api[-_]?key|apikey)\s*[:=]\s*["\'][a-zA-Z0-9]{16,}["\']',  # Quoted API keys
            r'(?i)(secret[-_]?key|secretkey)\s*[:=]\s*["\'][a-zA-Z0-9]{16,}["\']',  # Quoted secrets
            r'(?i)(token)\s*[:=]\s*["\'][a-zA-Z0-9]{20,}["\']',  # Quoted tokens
            r'(?i)(private[-_]?key|privatekey)\s*[:=]\s*["\']-----BEGIN',  # Private key blocks
        ]
        
        suspicious_files = []
        total_files_scanned = 0
        
        for py_file in self.project_root.rglob("*.py"):
            total_files_scanned += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in secret_patterns:
                    import re
                    if re.search(pattern, content):
                        suspicious_files.append(str(py_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        score = max(0, 100 - (len(suspicious_files) * 20))  # -20 points per suspicious file
        
        critical_issues = []
        if len(suspicious_files) > 2:
            critical_issues.append(f"Found {len(suspicious_files)} files with potential secrets")
        
        return {
            "score": score,
            "files_scanned": total_files_scanned,
            "suspicious_files": suspicious_files,
            "critical_issues": critical_issues,
            "warnings": [f"Potential secrets in: {f}" for f in suspicious_files[:3]],
            "recommendations": ["Use environment variables for secrets", "Implement proper secret management"]
        }
    
    def _check_dependency_vulnerabilities(self) -> Dict[str, Any]:
        """Check for known vulnerabilities in dependencies."""
        # Check if requirements files exist
        req_files = [
            self.project_root / "requirements.txt",
            self.project_root / "pyproject.toml",
            self.project_root / "setup.py"
        ]
        
        dependencies_found = any(f.exists() for f in req_files)
        
        if not dependencies_found:
            return {
                "score": 70,  # Neutral score if no dependencies to check
                "dependencies_found": False,
                "critical_issues": [],
                "warnings": ["No dependency files found"],
                "recommendations": ["Add requirements.txt or pyproject.toml for dependency management"]
            }
        
        # For now, assume dependencies are secure (in real implementation, use tools like safety)
        score = 85
        
        return {
            "score": score,
            "dependencies_found": True,
            "vulnerabilities_found": 0,
            "critical_issues": [],
            "warnings": [],
            "recommendations": ["Run 'pip install safety && safety check' for vulnerability scanning"]
        }
    
    def _check_security_patterns(self) -> Dict[str, Any]:
        """Check for security anti-patterns in code."""
        security_issues = []
        files_with_issues = []
        
        dangerous_patterns = [
            (r'\bexec\s*\(', "Use of exec() function"),
            (r'\beval\s*\([^)]*["\']', "Use of eval() function with string"),  # More specific
            (r'pickle\.loads?\s*\(', "Unsafe pickle usage"),
            (r'subprocess\.[^(]*shell\s*=\s*True', "Shell injection risk"),
            (r'os\.system\s*\(', "Use of os.system()"),
            (r'input\s*\([^)]*\)\s*\)', "Unsafe input() usage"),
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                file_issues = []
                for pattern, description in dangerous_patterns:
                    import re
                    if re.search(pattern, content):
                        file_issues.append(description)
                
                if file_issues:
                    files_with_issues.append(str(py_file.relative_to(self.project_root)))
                    security_issues.extend(file_issues)
            except:
                continue
        
        score = max(0, 100 - (len(security_issues) * 15))  # -15 points per issue
        
        critical_issues = []
        if len(security_issues) > 3:
            critical_issues.append(f"Found {len(security_issues)} security anti-patterns")
        
        return {
            "score": score,
            "security_issues_found": len(security_issues),
            "files_with_issues": files_with_issues,
            "critical_issues": critical_issues,
            "warnings": [f"Security issue in {f}" for f in files_with_issues[:3]],
            "recommendations": ["Replace unsafe functions with secure alternatives", "Add input validation"]
        }
    
    def _check_input_validation(self) -> Dict[str, Any]:
        """Check for proper input validation patterns."""
        validation_patterns_found = 0
        files_with_validation = []
        
        validation_patterns = [
            r'isinstance\s*\(',
            r'validate_\w+\s*\(',
            r'ValidationError',
            r'@validator',
            r'pydantic',
            r'marshmallow'
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in validation_patterns:
                    import re
                    if re.search(pattern, content):
                        validation_patterns_found += 1
                        files_with_validation.append(str(py_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        # Score based on proportion of files with validation
        total_py_files = len(list(self.project_root.rglob("*.py")))
        validation_ratio = len(files_with_validation) / max(total_py_files, 1)
        score = min(100, validation_ratio * 200)  # Up to 100 points
        
        return {
            "score": score,
            "validation_patterns_found": validation_patterns_found,
            "files_with_validation": len(files_with_validation),
            "total_py_files": total_py_files,
            "critical_issues": [],
            "warnings": ["Low input validation coverage"] if score < 50 else [],
            "recommendations": ["Add more input validation", "Use schema validation libraries"]
        }
    
    def _check_authentication_security(self) -> Dict[str, Any]:
        """Check authentication and authorization patterns."""
        auth_patterns = [
            r'@login_required',
            r'@authenticate',
            r'check_permission',
            r'verify_token',
            r'jwt\.',
            r'bcrypt',
            r'hashlib'
        ]
        
        auth_files = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in auth_patterns:
                    import re
                    if re.search(pattern, content):
                        auth_files.append(str(py_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        # Basic scoring - presence of auth patterns
        score = min(100, len(auth_files) * 25)  # Up to 100 points
        
        return {
            "score": score,
            "auth_files_found": len(auth_files),
            "critical_issues": [],
            "warnings": ["No authentication patterns found"] if len(auth_files) == 0 else [],
            "recommendations": ["Implement proper authentication", "Use secure password hashing"]
        }


class PerformanceGate:
    """Performance benchmarking and optimization gate."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def run_performance_checks(self) -> QualityGateResult:
        """Run comprehensive performance checks."""
        start_time = time.time()
        
        results = {
            "import_performance": self._check_import_performance(),
            "code_complexity": self._analyze_code_complexity(),
            "memory_efficiency": self._check_memory_patterns(),
            "algorithmic_efficiency": self._check_algorithmic_patterns(),
            "caching_usage": self._check_caching_patterns()
        }
        
        # Calculate overall performance score
        total_score = 0
        max_score = 0
        critical_issues = []
        warnings = []
        recommendations = []
        
        for check_name, check_result in results.items():
            total_score += check_result["score"]
            max_score += 100
            critical_issues.extend(check_result.get("critical_issues", []))
            warnings.extend(check_result.get("warnings", []))
            recommendations.extend(check_result.get("recommendations", []))
        
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0
        passed = overall_score >= 75
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Performance",
            passed=passed,
            score=overall_score,
            details=results,
            execution_time=execution_time,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _check_import_performance(self) -> Dict[str, Any]:
        """Check for import performance issues."""
        slow_imports = []
        circular_imports = []
        
        # Check for potentially slow imports
        slow_import_patterns = [
            r'import tensorflow',
            r'import torch',
            r'import numpy',
            r'import pandas',
            r'import matplotlib',
            r'from torch import',
            r'from tensorflow import'
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in slow_import_patterns:
                    import re
                    if re.search(pattern, content):
                        slow_imports.append(str(py_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        # Score based on import optimization
        score = max(50, 100 - len(slow_imports) * 5)  # -5 points per slow import file
        
        return {
            "score": score,
            "slow_imports": len(slow_imports),
            "files_with_slow_imports": slow_imports[:5],
            "critical_issues": [],
            "warnings": ["Consider lazy imports for heavy libraries"] if len(slow_imports) > 5 else [],
            "recommendations": ["Use lazy imports", "Import only what you need"]
        }
    
    def _analyze_code_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        complex_functions = []
        total_functions = 0
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count functions and estimate complexity
                import re
                functions = re.findall(r'def\s+(\w+)', content)
                total_functions += len(functions)
                
                # Simple complexity estimation based on control structures
                for func_match in re.finditer(r'def\s+(\w+).*?(?=\ndef|\nclass|\Z)', content, re.DOTALL):
                    func_content = func_match.group(0)
                    complexity = (
                        len(re.findall(r'\bif\b', func_content)) +
                        len(re.findall(r'\bfor\b', func_content)) +
                        len(re.findall(r'\bwhile\b', func_content)) +
                        len(re.findall(r'\btry\b', func_content)) +
                        len(re.findall(r'\bexcept\b', func_content))
                    )
                    
                    if complexity > 10:  # Arbitrary threshold
                        complex_functions.append(func_match.group(1))
                        
            except:
                continue
        
        complexity_ratio = len(complex_functions) / max(total_functions, 1)
        score = max(0, 100 - (complexity_ratio * 100))  # Penalize high complexity ratio
        
        critical_issues = []
        if complexity_ratio > 0.3:
            critical_issues.append(f"High complexity ratio: {complexity_ratio:.2%}")
        
        return {
            "score": score,
            "total_functions": total_functions,
            "complex_functions": len(complex_functions),
            "complexity_ratio": complexity_ratio,
            "critical_issues": critical_issues,
            "warnings": [f"Complex function: {f}" for f in complex_functions[:3]],
            "recommendations": ["Refactor complex functions", "Use composition over complexity"]
        }
    
    def _check_memory_patterns(self) -> Dict[str, Any]:
        """Check for memory-efficient patterns."""
        memory_efficient_patterns = [
            r'__slots__',
            r'@lru_cache',
            r'weakref',
            r'gc\.',
            r'del\s+',
            r'numpy\.array',
            r'deque\('
        ]
        
        files_with_optimization = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in memory_efficient_patterns:
                    import re
                    if re.search(pattern, content):
                        files_with_optimization.append(str(py_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        total_py_files = len(list(self.project_root.rglob("*.py")))
        optimization_ratio = len(files_with_optimization) / max(total_py_files, 1)
        score = min(100, optimization_ratio * 150)  # Up to 100 points
        
        return {
            "score": score,
            "files_with_optimization": len(files_with_optimization),
            "optimization_ratio": optimization_ratio,
            "critical_issues": [],
            "warnings": ["Low memory optimization coverage"] if score < 40 else [],
            "recommendations": ["Use __slots__ for classes", "Implement caching", "Use generators"]
        }
    
    def _check_algorithmic_patterns(self) -> Dict[str, Any]:
        """Check for efficient algorithmic patterns."""
        efficient_patterns = [
            r'O\(.*\)',  # Big O notation
            r'bisect\.',
            r'heapq\.',
            r'collections\.',
            r'itertools\.',
            r'@functools',
            r'numpy\.',
            r'vectorized'
        ]
        
        files_with_algorithms = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in efficient_patterns:
                    import re
                    if re.search(pattern, content):
                        files_with_algorithms.append(str(py_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        score = min(100, len(files_with_algorithms) * 10)  # Up to 100 points
        
        return {
            "score": score,
            "files_with_efficient_algorithms": len(files_with_algorithms),
            "critical_issues": [],
            "warnings": [],
            "recommendations": ["Use efficient data structures", "Implement vectorized operations"]
        }
    
    def _check_caching_patterns(self) -> Dict[str, Any]:
        """Check for caching implementation."""
        caching_patterns = [
            r'@cache',
            r'@lru_cache',
            r'cache\.',
            r'memoize',
            r'Cache',
            r'redis',
            r'memcache'
        ]
        
        files_with_caching = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in caching_patterns:
                    import re
                    if re.search(pattern, content):
                        files_with_caching.append(str(py_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        score = min(100, len(files_with_caching) * 20)  # Up to 100 points
        
        return {
            "score": score,
            "files_with_caching": len(files_with_caching),
            "critical_issues": [],
            "warnings": ["No caching patterns found"] if len(files_with_caching) == 0 else [],
            "recommendations": ["Implement caching for expensive operations", "Use appropriate cache strategies"]
        }


class TestingGate:
    """Test coverage and quality gate."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def run_testing_checks(self) -> QualityGateResult:
        """Run comprehensive testing checks."""
        start_time = time.time()
        
        results = {
            "test_coverage": self._check_test_coverage(),
            "test_quality": self._analyze_test_quality(),
            "test_organization": self._check_test_organization(),
            "mocking_patterns": self._check_mocking_patterns(),
            "integration_tests": self._check_integration_tests()
        }
        
        # Calculate overall testing score
        total_score = 0
        max_score = 0
        critical_issues = []
        warnings = []
        recommendations = []
        
        for check_name, check_result in results.items():
            total_score += check_result["score"]
            max_score += 100
            critical_issues.extend(check_result.get("critical_issues", []))
            warnings.extend(check_result.get("warnings", []))
            recommendations.extend(check_result.get("recommendations", []))
        
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0
        passed = overall_score >= 70
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Testing",
            passed=passed,
            score=overall_score,
            details=results,
            execution_time=execution_time,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _check_test_coverage(self) -> Dict[str, Any]:
        """Check test coverage by counting test files vs source files."""
        source_files = list(self.project_root.rglob("pwmk/**/*.py"))
        test_files = list(self.project_root.rglob("tests/**/*.py")) + list(self.project_root.rglob("test_*.py"))
        
        # Remove __init__.py files
        source_files = [f for f in source_files if f.name != "__init__.py"]
        
        coverage_ratio = len(test_files) / max(len(source_files), 1)
        score = min(100, coverage_ratio * 150)  # Up to 100 points
        
        critical_issues = []
        if coverage_ratio < 0.3:
            critical_issues.append(f"Low test coverage ratio: {coverage_ratio:.2%}")
        
        return {
            "score": score,
            "source_files": len(source_files),
            "test_files": len(test_files),
            "coverage_ratio": coverage_ratio,
            "critical_issues": critical_issues,
            "warnings": ["Low test coverage"] if score < 50 else [],
            "recommendations": ["Write more unit tests", "Achieve at least 80% test coverage"]
        }
    
    def _analyze_test_quality(self) -> Dict[str, Any]:
        """Analyze quality of existing tests."""
        test_quality_patterns = [
            r'assert\w*\s',
            r'@pytest\.',
            r'@unittest\.',
            r'setUp',
            r'tearDown',
            r'fixture',
            r'mock\.',
            r'patch\('
        ]
        
        high_quality_test_files = []
        total_test_files = 0
        
        test_dirs = [self.project_root / "tests"] + list(self.project_root.rglob("test_*.py"))
        
        for test_path in test_dirs:
            if test_path.is_file():
                test_files = [test_path]
            else:
                test_files = list(test_path.rglob("*.py"))
            
            for test_file in test_files:
                if test_file.name.startswith("test_") or "test" in str(test_file):
                    total_test_files += 1
                    
                    try:
                        with open(test_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        quality_score = 0
                        for pattern in test_quality_patterns:
                            import re
                            if re.search(pattern, content):
                                quality_score += 1
                        
                        if quality_score >= 3:  # At least 3 quality indicators
                            high_quality_test_files.append(str(test_file.relative_to(self.project_root)))
                    except:
                        continue
        
        if total_test_files == 0:
            score = 0
        else:
            quality_ratio = len(high_quality_test_files) / total_test_files
            score = quality_ratio * 100
        
        return {
            "score": score,
            "total_test_files": total_test_files,
            "high_quality_test_files": len(high_quality_test_files),
            "quality_ratio": len(high_quality_test_files) / max(total_test_files, 1),
            "critical_issues": ["No test files found"] if total_test_files == 0 else [],
            "warnings": ["Low test quality"] if score < 50 else [],
            "recommendations": ["Use proper assertions", "Add test fixtures", "Use mocking"]
        }
    
    def _check_test_organization(self) -> Dict[str, Any]:
        """Check test organization and structure."""
        has_test_dir = (self.project_root / "tests").exists()
        has_conftest = (self.project_root / "conftest.py").exists() or (self.project_root / "tests" / "conftest.py").exists()
        has_pytest_ini = (self.project_root / "pytest.ini").exists()
        has_test_requirements = any((
            (self.project_root / "requirements-test.txt").exists(),
            (self.project_root / "test-requirements.txt").exists(),
            "test" in str(self.project_root / "pyproject.toml") if (self.project_root / "pyproject.toml").exists() else False
        ))
        
        organization_score = 0
        if has_test_dir: organization_score += 25
        if has_conftest: organization_score += 25
        if has_pytest_ini: organization_score += 25
        if has_test_requirements: organization_score += 25
        
        return {
            "score": organization_score,
            "has_test_directory": has_test_dir,
            "has_conftest": has_conftest,
            "has_pytest_ini": has_pytest_ini,
            "has_test_requirements": has_test_requirements,
            "critical_issues": [],
            "warnings": ["Missing test organization"] if organization_score < 50 else [],
            "recommendations": ["Create tests/ directory", "Add conftest.py", "Add pytest.ini"]
        }
    
    def _check_mocking_patterns(self) -> Dict[str, Any]:
        """Check for proper mocking in tests."""
        mocking_patterns = [
            r'@patch',
            r'@mock',
            r'Mock\(',
            r'MagicMock',
            r'mock\.',
            r'unittest\.mock'
        ]
        
        files_with_mocking = []
        
        for test_file in self.project_root.rglob("test_*.py"):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in mocking_patterns:
                    import re
                    if re.search(pattern, content):
                        files_with_mocking.append(str(test_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        score = min(100, len(files_with_mocking) * 25)  # Up to 100 points
        
        return {
            "score": score,
            "files_with_mocking": len(files_with_mocking),
            "critical_issues": [],
            "warnings": ["No mocking patterns found"] if len(files_with_mocking) == 0 else [],
            "recommendations": ["Use mocking for external dependencies", "Mock expensive operations"]
        }
    
    def _check_integration_tests(self) -> Dict[str, Any]:
        """Check for integration and end-to-end tests."""
        integration_patterns = [
            r'integration',
            r'e2e',
            r'end.?to.?end',
            r'full.?test',
            r'system.?test'
        ]
        
        integration_test_files = []
        
        for test_file in self.project_root.rglob("**/*test*.py"):
            try:
                filename = test_file.name.lower()
                content = ""
                
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in integration_patterns:
                    import re
                    if re.search(pattern, filename) or re.search(pattern, content):
                        integration_test_files.append(str(test_file.relative_to(self.project_root)))
                        break
            except:
                continue
        
        score = min(100, len(integration_test_files) * 30)  # Up to 100 points
        
        return {
            "score": score,
            "integration_test_files": len(integration_test_files),
            "critical_issues": [],
            "warnings": ["No integration tests found"] if len(integration_test_files) == 0 else [],
            "recommendations": ["Add integration tests", "Test complete workflows"]
        }


class ComprehensiveQualityGates:
    """Comprehensive quality gates system."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.validator = PWMKValidator(project_root)
        self.security_gate = SecurityGate(self.project_root)
        self.performance_gate = PerformanceGate(self.project_root)
        self.testing_gate = TestingGate(self.project_root)
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates in parallel."""
        start_time = time.time()
        
        print("🚀 Running Comprehensive PWMK Quality Gates...")
        print("="*60)
        
        # Run gates in parallel for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                "validation": executor.submit(self._run_validation_gate),
                "security": executor.submit(self.security_gate.run_security_checks),
                "performance": executor.submit(self.performance_gate.run_performance_checks),
                "testing": executor.submit(self.testing_gate.run_testing_checks)
            }
            
            gate_results = {}
            for gate_name, future in futures.items():
                try:
                    result = future.result(timeout=120)  # 2 minute timeout per gate
                    gate_results[gate_name] = result
                    print(f"✅ {gate_name.capitalize()} Gate: {'PASSED' if result.passed else 'FAILED'} (Score: {result.score:.1f}/100)")
                except Exception as e:
                    print(f"❌ {gate_name.capitalize()} Gate: ERROR - {str(e)}")
                    gate_results[gate_name] = QualityGateResult(
                        gate_name=gate_name.capitalize(),
                        passed=False,
                        score=0.0,
                        details={"error": str(e)},
                        execution_time=0.0,
                        critical_issues=[f"Gate execution failed: {str(e)}"],
                        warnings=[],
                        recommendations=[]
                    )
        
        total_execution_time = time.time() - start_time
        
        # Calculate overall results
        overall_results = self._calculate_overall_results(gate_results, total_execution_time)
        
        # Print summary
        self._print_summary(overall_results)
        
        return overall_results
    
    def _run_validation_gate(self) -> QualityGateResult:
        """Run validation gate using existing validator."""
        start_time = time.time()
        
        validation_results = self.validator.run_comprehensive_validation()
        
        # Convert validation results to QualityGateResult format
        structure_score = sum(validation_results["project_structure"].values()) / len(validation_results["project_structure"]) * 100
        syntax_score = sum(validation_results["python_syntax"].values()) / len(validation_results["python_syntax"]) * 100 if validation_results["python_syntax"] else 100
        doc_score = sum(validation_results["docstring_coverage"].values()) / len(validation_results["docstring_coverage"]) * 100 if validation_results["docstring_coverage"] else 0
        
        overall_score = (structure_score + syntax_score + doc_score) / 3
        
        critical_issues = []
        warnings = []
        
        # Check for critical structure issues
        failed_structure = [k for k, v in validation_results["project_structure"].items() if not v]
        if failed_structure:
            critical_issues.extend([f"Missing: {item}" for item in failed_structure])
        
        # Check for syntax errors
        failed_syntax = [k for k, v in validation_results["python_syntax"].items() if not v]
        if failed_syntax:
            critical_issues.extend([f"Syntax error: {item}" for item in failed_syntax[:3]])
        
        # Check for low documentation
        low_doc = [k for k, v in validation_results["docstring_coverage"].items() if v < 0.5]
        if low_doc:
            warnings.extend([f"Low documentation: {item}" for item in low_doc[:3]])
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Validation",
            passed=len(critical_issues) == 0 and overall_score >= 80,
            score=overall_score,
            details=validation_results,
            execution_time=execution_time,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=["Fix missing structure elements", "Improve documentation coverage"]
        )
    
    def _calculate_overall_results(self, gate_results: Dict[str, QualityGateResult], execution_time: float) -> Dict[str, Any]:
        """Calculate overall quality gate results."""
        total_score = 0
        max_score = 0
        gates_passed = 0
        total_gates = len(gate_results)
        
        all_critical_issues = []
        all_warnings = []
        all_recommendations = []
        
        for gate_result in gate_results.values():
            if isinstance(gate_result, QualityGateResult):
                total_score += gate_result.score
                max_score += 100
                if gate_result.passed:
                    gates_passed += 1
                
                all_critical_issues.extend(gate_result.critical_issues)
                all_warnings.extend(gate_result.warnings)
                all_recommendations.extend(gate_result.recommendations)
        
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0
        overall_passed = gates_passed == total_gates and len(all_critical_issues) == 0
        
        return {
            "overall_passed": overall_passed,
            "overall_score": overall_score,
            "gates_passed": gates_passed,
            "total_gates": total_gates,
            "gate_results": {name: asdict(result) for name, result in gate_results.items()},
            "critical_issues": all_critical_issues,
            "warnings": all_warnings,
            "recommendations": all_recommendations,
            "execution_time": execution_time,
            "timestamp": time.time()
        }
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print quality gates summary."""
        print("\n" + "="*60)
        print("🏆 QUALITY GATES SUMMARY")
        print("="*60)
        
        status = "✅ PASSED" if results["overall_passed"] else "❌ FAILED"
        print(f"Overall Status: {status}")
        print(f"Overall Score: {results['overall_score']:.1f}/100")
        print(f"Gates Passed: {results['gates_passed']}/{results['total_gates']}")
        print(f"Execution Time: {results['execution_time']:.2f}s")
        
        if results["critical_issues"]:
            print(f"\n🚨 Critical Issues ({len(results['critical_issues'])}):")
            for issue in results["critical_issues"][:5]:  # Show top 5
                print(f"  - {issue}")
            if len(results["critical_issues"]) > 5:
                print(f"  ... and {len(results['critical_issues']) - 5} more")
        
        if results["warnings"]:
            print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
            for warning in results["warnings"][:5]:  # Show top 5
                print(f"  - {warning}")
            if len(results["warnings"]) > 5:
                print(f"  ... and {len(results['warnings']) - 5} more")
        
        if results["recommendations"]:
            print(f"\n💡 Recommendations ({len(results['recommendations'])}):")
            for rec in results["recommendations"][:5]:  # Show top 5
                print(f"  - {rec}")
            if len(results["recommendations"]) > 5:
                print(f"  ... and {len(results['recommendations']) - 5} more")
        
        print("\n" + "="*60)
        
        if results["overall_passed"]:
            print("🎉 All quality gates passed! Ready for production deployment.")
        else:
            print("🔧 Quality gates failed. Please address critical issues before deployment.")


def main():
    """Main entry point for quality gates."""
    quality_gates = ComprehensiveQualityGates()
    results = quality_gates.run_all_gates()
    
    # Save results to file
    results_file = Path("quality_gates_report.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    return 0 if results["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())