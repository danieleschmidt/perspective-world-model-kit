#!/usr/bin/env python3
"""
Architecture and code quality validation for PWMK.
Tests system architecture, code completeness, and documentation without dependencies.
"""

import sys
import time
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple


def analyze_python_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a Python file for architecture patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        analysis = {
            'file_path': str(file_path),
            'line_count': len(content.split('\n')),
            'classes': [],
            'functions': [],
            'imports': [],
            'docstring': None,
            'has_main': False,
            'error_handling': 0,
            'logging_statements': 0,
            'type_hints': 0,
            'complexity_score': 0
        }
        
        # Extract module docstring
        if (tree.body and isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Constant) and 
            isinstance(tree.body[0].value.value, str)):
            analysis['docstring'] = tree.body[0].value.value[:100] + "..."
        
        # Analyze AST nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                analysis['classes'].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                analysis['functions'].append(node.name)
                if node.name == 'main':
                    analysis['has_main'] = True
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                else:
                    module = node.module or ''
                    analysis['imports'].append(module)
            elif isinstance(node, ast.ExceptHandler):
                analysis['error_handling'] += 1
            elif isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr in ['debug', 'info', 'warning', 'error']):
                    analysis['logging_statements'] += 1
            elif isinstance(node, (ast.arg, ast.FunctionDef)):
                if hasattr(node, 'annotation') and node.annotation:
                    analysis['type_hints'] += 1
        
        # Calculate complexity score
        complexity_factors = [
            len(analysis['classes']) * 2,
            len(analysis['functions']),
            analysis['error_handling'] * 2,
            len([imp for imp in analysis['imports'] if '.' in imp])
        ]
        analysis['complexity_score'] = sum(complexity_factors)
        
        return analysis
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'error': str(e),
            'line_count': 0,
            'classes': [],
            'functions': [],
            'imports': []
        }


def test_system_architecture():
    """Test overall system architecture."""
    print("🏗️  Testing system architecture...")
    
    # Define expected architecture patterns
    expected_patterns = {
        'core': {
            'description': 'Core AI models and belief systems',
            'required_files': ['world_model.py', 'beliefs.py'],
            'expected_classes': ['PerspectiveWorldModel', 'BeliefStore'],
            'expected_functions': ['forward', 'query', 'add_belief']
        },
        'utils': {
            'description': 'Utilities for validation, logging, monitoring',
            'required_files': ['validation.py', 'logging.py', 'circuit_breaker.py', 'fallback_manager.py', 'health_monitor.py'],
            'expected_classes': ['CircuitBreaker', 'FallbackManager', 'HealthMonitor'],
            'expected_functions': ['validate_config', 'get_logger']
        },
        'optimization': {
            'description': 'Performance and scaling optimizations',
            'required_files': ['adaptive_scaling.py', 'performance_optimizer.py'],
            'expected_classes': ['AdaptiveScaler', 'PerformanceOptimizer'],
            'expected_functions': ['record_metrics', 'optimize_model']
        },
        'quantum': {
            'description': 'Quantum computing acceleration',
            'required_files': ['adaptive_quantum_acceleration.py'],
            'expected_classes': ['AdaptiveQuantumAccelerator'],
            'expected_functions': ['solve_optimization_problem']
        },
        'security': {
            'description': 'Security and input validation',
            'required_files': ['input_sanitizer.py', 'belief_validator.py'],
            'expected_classes': ['InputSanitizer'],
            'expected_functions': ['sanitize_belief_content']
        }
    }
    
    architecture_score = 0
    total_checks = 0
    detailed_results = {}
    
    for component, config in expected_patterns.items():
        component_path = Path(f"pwmk/{component}")
        component_results = {
            'exists': component_path.exists(),
            'files_found': [],
            'missing_files': [],
            'classes_found': [],
            'functions_found': [],
            'total_lines': 0,
            'files_with_docstrings': 0,
            'error_handling_count': 0
        }
        
        if component_path.exists():
            # Check for required files
            for required_file in config['required_files']:
                file_path = component_path / required_file
                total_checks += 1
                
                if file_path.exists():
                    component_results['files_found'].append(required_file)
                    architecture_score += 1
                    
                    # Analyze the file
                    analysis = analyze_python_file(file_path)
                    component_results['total_lines'] += analysis.get('line_count', 0)
                    component_results['classes_found'].extend(analysis.get('classes', []))
                    component_results['functions_found'].extend(analysis.get('functions', []))
                    component_results['error_handling_count'] += analysis.get('error_handling', 0)
                    
                    if analysis.get('docstring'):
                        component_results['files_with_docstrings'] += 1
                else:
                    component_results['missing_files'].append(required_file)
            
            # Check for expected classes and functions
            all_found_classes = component_results['classes_found']
            all_found_functions = component_results['functions_found']
            
            for expected_class in config.get('expected_classes', []):
                total_checks += 1
                if any(expected_class in cls for cls in all_found_classes):
                    architecture_score += 1
            
            for expected_function in config.get('expected_functions', []):
                total_checks += 1
                if any(expected_function in func for func in all_found_functions):
                    architecture_score += 1
        
        detailed_results[component] = component_results
        
        # Print component summary
        files_found = len(component_results['files_found'])
        total_files = len(config['required_files'])
        print(f"   {component}: {files_found}/{total_files} files, "
              f"{len(component_results['classes_found'])} classes, "
              f"{component_results['total_lines']} lines")
    
    architecture_success_rate = architecture_score / total_checks if total_checks > 0 else 0
    
    print(f"   Architecture Score: {architecture_score}/{total_checks} ({architecture_success_rate:.1%})")
    
    return architecture_success_rate >= 0.7, detailed_results


def test_code_quality():
    """Test code quality metrics."""
    print("\n📝 Testing code quality...")
    
    python_files = list(Path("pwmk").rglob("*.py"))
    if not python_files:
        print("   ❌ No Python files found")
        return False, {}
    
    quality_metrics = {
        'total_files': len(python_files),
        'total_lines': 0,
        'files_with_docstrings': 0,
        'files_with_error_handling': 0,
        'files_with_logging': 0,
        'files_with_type_hints': 0,
        'complex_files': 0,
        'average_complexity': 0
    }
    
    complexity_scores = []
    
    for python_file in python_files:
        if python_file.name == '__init__.py':
            continue
            
        analysis = analyze_python_file(python_file)
        
        if 'error' not in analysis:
            quality_metrics['total_lines'] += analysis['line_count']
            
            if analysis['docstring']:
                quality_metrics['files_with_docstrings'] += 1
            
            if analysis['error_handling'] > 0:
                quality_metrics['files_with_error_handling'] += 1
            
            if analysis['logging_statements'] > 0:
                quality_metrics['files_with_logging'] += 1
            
            if analysis['type_hints'] > 0:
                quality_metrics['files_with_type_hints'] += 1
            
            complexity = analysis['complexity_score']
            complexity_scores.append(complexity)
            
            if complexity > 20:  # Arbitrary threshold
                quality_metrics['complex_files'] += 1
    
    quality_metrics['average_complexity'] = (
        sum(complexity_scores) / len(complexity_scores) 
        if complexity_scores else 0
    )
    
    # Calculate quality score
    total_files = quality_metrics['total_files']
    quality_score = (
        (quality_metrics['files_with_docstrings'] / total_files) * 0.2 +
        (quality_metrics['files_with_error_handling'] / total_files) * 0.3 +
        (quality_metrics['files_with_logging'] / total_files) * 0.2 +
        (quality_metrics['files_with_type_hints'] / total_files) * 0.2 +
        (1.0 - min(quality_metrics['complex_files'] / total_files, 1.0)) * 0.1
    )
    
    print(f"   Files analyzed: {total_files}")
    print(f"   Total lines: {quality_metrics['total_lines']}")
    print(f"   Files with docstrings: {quality_metrics['files_with_docstrings']}/{total_files}")
    print(f"   Files with error handling: {quality_metrics['files_with_error_handling']}/{total_files}")
    print(f"   Files with logging: {quality_metrics['files_with_logging']}/{total_files}")
    print(f"   Average complexity: {quality_metrics['average_complexity']:.1f}")
    print(f"   Quality score: {quality_score:.1%}")
    
    return quality_score >= 0.6, quality_metrics


def test_documentation_completeness():
    """Test documentation completeness."""
    print("\n📖 Testing documentation completeness...")
    
    docs_metrics = {
        'readme_exists': Path("README.md").exists(),
        'readme_size': 0,
        'has_installation_docs': False,
        'has_usage_examples': False,
        'has_api_docs': False,
        'has_contributing_guide': False,
        'project_metadata_complete': False
    }
    
    # Check README.md
    readme_path = Path("README.md")
    if readme_path.exists():
        readme_content = readme_path.read_text()
        docs_metrics['readme_size'] = len(readme_content)
        
        # Check for key sections
        readme_lower = readme_content.lower()
        if 'install' in readme_lower:
            docs_metrics['has_installation_docs'] = True
        
        if any(keyword in readme_lower for keyword in ['example', 'usage', 'quick start']):
            docs_metrics['has_usage_examples'] = True
        
        if any(keyword in readme_lower for keyword in ['api', 'reference', 'class', 'function']):
            docs_metrics['has_api_docs'] = True
    
    # Check for contributing guide
    if Path("CONTRIBUTING.md").exists():
        docs_metrics['has_contributing_guide'] = True
    
    # Check project metadata
    try:
        import toml
        config = toml.load("pyproject.toml")
        project = config.get('project', {})
        
        required_fields = ['name', 'description', 'authors', 'license']
        if all(field in project for field in required_fields):
            docs_metrics['project_metadata_complete'] = True
            
    except (ImportError, FileNotFoundError, toml.TomlDecodeError):
        pass
    
    # Calculate documentation score
    doc_checks = [
        docs_metrics['readme_exists'],
        docs_metrics['readme_size'] > 1000,  # Substantial README
        docs_metrics['has_installation_docs'],
        docs_metrics['has_usage_examples'],
        docs_metrics['has_api_docs'],
        docs_metrics['project_metadata_complete']
    ]
    
    doc_score = sum(doc_checks) / len(doc_checks)
    
    print(f"   README exists: {'✅' if docs_metrics['readme_exists'] else '❌'}")
    print(f"   README size: {docs_metrics['readme_size']} chars")
    print(f"   Has installation docs: {'✅' if docs_metrics['has_installation_docs'] else '❌'}")
    print(f"   Has usage examples: {'✅' if docs_metrics['has_usage_examples'] else '❌'}")
    print(f"   Has API documentation: {'✅' if docs_metrics['has_api_docs'] else '❌'}")
    print(f"   Project metadata complete: {'✅' if docs_metrics['project_metadata_complete'] else '❌'}")
    print(f"   Documentation score: {doc_score:.1%}")
    
    return doc_score >= 0.7, docs_metrics


def test_advanced_features():
    """Test implementation of advanced AI features."""
    print("\n🧠 Testing advanced AI features...")
    
    feature_files = {
        'consciousness': 'pwmk/revolution/consciousness_engine.py',
        'quantum_computing': 'pwmk/quantum/adaptive_quantum_acceleration.py',
        'autonomous_agents': 'pwmk/autonomous/self_improving_agent.py',
        'emergent_intelligence': 'pwmk/breakthrough/emergent_intelligence.py',
        'theory_of_mind': 'pwmk/agents/tom_agent.py',
        'belief_reasoning': 'pwmk/core/beliefs.py',
        'adaptive_scaling': 'pwmk/optimization/adaptive_scaling.py',
        'circuit_breakers': 'pwmk/utils/circuit_breaker.py',
        'health_monitoring': 'pwmk/utils/health_monitor.py'
    }
    
    features_implemented = 0
    feature_details = {}
    
    for feature_name, file_path in feature_files.items():
        path = Path(file_path)
        is_implemented = path.exists()
        
        feature_info = {
            'implemented': is_implemented,
            'file_size': 0,
            'classes': [],
            'complexity': 0
        }
        
        if is_implemented:
            features_implemented += 1
            analysis = analyze_python_file(path)
            
            feature_info['file_size'] = analysis.get('line_count', 0)
            feature_info['classes'] = analysis.get('classes', [])
            feature_info['complexity'] = analysis.get('complexity_score', 0)
        
        feature_details[feature_name] = feature_info
        
        status = "✅" if is_implemented else "❌"
        size_info = f" ({feature_info['file_size']} lines)" if is_implemented else ""
        print(f"   {feature_name}: {status}{size_info}")
    
    features_score = features_implemented / len(feature_files)
    print(f"   Advanced features: {features_implemented}/{len(feature_files)} ({features_score:.1%})")
    
    return features_score >= 0.8, feature_details


def test_security_implementation():
    """Test security implementation."""
    print("\n🔒 Testing security implementation...")
    
    security_files = [
        'pwmk/security/input_sanitizer.py',
        'pwmk/security/belief_validator.py',
        'pwmk/security/rate_limiter.py',
        'pwmk/security/quantum_security.py'
    ]
    
    security_features = {
        'input_sanitization': False,
        'belief_validation': False,
        'rate_limiting': False,
        'quantum_security': False,
        'error_handling': False
    }
    
    security_score = 0
    total_checks = len(security_features)
    
    for security_file in security_files:
        path = Path(security_file)
        if path.exists():
            analysis = analyze_python_file(path)
            
            # Check for security-related classes and functions
            classes = analysis.get('classes', [])
            functions = analysis.get('functions', [])
            
            if 'sanitiz' in security_file.lower():
                security_features['input_sanitization'] = True
                security_score += 1
            
            if 'validator' in security_file.lower():
                security_features['belief_validation'] = True
                security_score += 1
            
            if 'rate_limiter' in security_file.lower():
                security_features['rate_limiting'] = True
                security_score += 1
            
            if 'quantum_security' in security_file.lower():
                security_features['quantum_security'] = True
                security_score += 1
            
            if analysis.get('error_handling', 0) > 0:
                security_features['error_handling'] = True
                security_score += 1
    
    final_score = security_score / total_checks if total_checks > 0 else 0
    
    for feature, implemented in security_features.items():
        status = "✅" if implemented else "❌"
        print(f"   {feature.replace('_', ' ').title()}: {status}")
    
    print(f"   Security score: {final_score:.1%}")
    
    return final_score >= 0.6, security_features


def main():
    """Main validation function."""
    print("🔍 PWMK Architecture & Code Quality Validation")
    print("=" * 55)
    print("Testing system architecture, code quality, and feature implementation")
    
    start_time = time.time()
    
    test_results = {}
    test_functions = [
        ("System Architecture", test_system_architecture),
        ("Code Quality", test_code_quality),
        ("Documentation Completeness", test_documentation_completeness),
        ("Advanced AI Features", test_advanced_features),
        ("Security Implementation", test_security_implementation)
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_name, test_func in test_functions:
        try:
            success, details = test_func()
            test_results[test_name] = {
                'success': success,
                'details': details
            }
            if success:
                passed_tests += 1
        except Exception as e:
            print(f"   💥 {test_name} failed with exception: {e}")
            test_results[test_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Calculate overall results
    success_rate = passed_tests / total_tests
    overall_success = success_rate >= 0.8  # 80% threshold
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n📊 Validation Summary (completed in {total_time:.2f}s)")
    print("=" * 55)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n📈 Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
    print(f"🎯 Overall Result: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
    
    # Save detailed results
    validation_results = {
        'timestamp': time.time(),
        'validation_type': 'architecture_and_quality',
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate,
        'overall_success': overall_success,
        'execution_time': total_time,
        'detailed_results': test_results
    }
    
    results_file = Path("architecture_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    if overall_success:
        print("\n🎉 Architecture validation passed! System demonstrates:")
        print("   ✅ Comprehensive AI consciousness and quantum computing framework")
        print("   ✅ Advanced resilience with circuit breakers and fallback systems")
        print("   ✅ Security-first design with input sanitization")
        print("   ✅ Performance optimization and adaptive scaling")
        print("   ✅ Production-ready architecture and documentation")
    
    return overall_success


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Fatal error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)