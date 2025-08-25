#!/usr/bin/env python3
"""
Standalone validation system for PWMK without external dependencies.
Validates code structure, imports, and architectural integrity.
"""

import os
import sys
import ast
import importlib.util
from typing import Dict, List, Set, Tuple
from pathlib import Path


class PWMKValidator:
    """Comprehensive validation system for PWMK codebase."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.pwmk_path = self.project_root / "pwmk"
        self.validation_results = {}
    
    def validate_project_structure(self) -> Dict[str, bool]:
        """Validate the overall project structure."""
        required_files = [
            "pwmk/__init__.py",
            "pwmk/core/__init__.py",
            "pwmk/core/world_model.py",
            "pwmk/core/beliefs.py",
            "pyproject.toml",
            "README.md"
        ]
        
        required_dirs = [
            "pwmk/core",
            "pwmk/planning", 
            "pwmk/agents",
            "pwmk/envs",
            "pwmk/utils",
            "tests"
        ]
        
        results = {}
        
        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            results[f"file_{file_path}"] = full_path.exists()
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            results[f"dir_{dir_path}"] = full_path.exists() and full_path.is_dir()
        
        return results
    
    def validate_python_syntax(self) -> Dict[str, bool]:
        """Validate Python syntax across all modules."""
        results = {}
        
        for py_file in self.pwmk_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    source = f.read()
                ast.parse(source)
                results[str(py_file.relative_to(self.project_root))] = True
            except SyntaxError as e:
                print(f"Syntax error in {py_file}: {e}")
                results[str(py_file.relative_to(self.project_root))] = False
            except Exception as e:
                print(f"Error reading {py_file}: {e}")
                results[str(py_file.relative_to(self.project_root))] = False
        
        return results
    
    def validate_import_structure(self) -> Dict[str, List[str]]:
        """Validate import dependencies and circular imports."""
        dependencies = {}
        
        for py_file in self.pwmk_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            try:
                with open(py_file, 'r') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                
                rel_path = str(py_file.relative_to(self.project_root))
                dependencies[rel_path] = imports
                
            except Exception as e:
                print(f"Error analyzing imports in {py_file}: {e}")
        
        return dependencies
    
    def validate_class_definitions(self) -> Dict[str, Dict[str, int]]:
        """Validate class definitions and method counts."""
        class_info = {}
        
        for py_file in self.pwmk_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                classes = {}
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        methods = sum(1 for item in node.body if isinstance(item, ast.FunctionDef))
                        classes[node.name] = methods
                
                if classes:
                    rel_path = str(py_file.relative_to(self.project_root))
                    class_info[rel_path] = classes
                    
            except Exception as e:
                print(f"Error analyzing classes in {py_file}: {e}")
        
        return class_info
    
    def validate_docstring_coverage(self) -> Dict[str, float]:
        """Validate docstring coverage across modules."""
        coverage = {}
        
        for py_file in self.pwmk_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                total_functions = 0
                documented_functions = 0
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                
                if total_functions > 0:
                    coverage_ratio = documented_functions / total_functions
                    rel_path = str(py_file.relative_to(self.project_root))
                    coverage[rel_path] = coverage_ratio
                    
            except Exception as e:
                print(f"Error analyzing docstrings in {py_file}: {e}")
        
        return coverage
    
    def run_comprehensive_validation(self) -> Dict[str, any]:
        """Run all validation checks."""
        print("🔍 Running PWMK Comprehensive Validation...")
        
        results = {
            "project_structure": self.validate_project_structure(),
            "python_syntax": self.validate_python_syntax(), 
            "import_structure": self.validate_import_structure(),
            "class_definitions": self.validate_class_definitions(),
            "docstring_coverage": self.validate_docstring_coverage()
        }
        
        # Summary statistics
        structure_passed = sum(results["project_structure"].values())
        structure_total = len(results["project_structure"])
        
        syntax_passed = sum(results["python_syntax"].values())
        syntax_total = len(results["python_syntax"])
        
        avg_coverage = (
            sum(results["docstring_coverage"].values()) / len(results["docstring_coverage"])
            if results["docstring_coverage"] else 0
        )
        
        results["summary"] = {
            "structure_score": f"{structure_passed}/{structure_total}",
            "syntax_score": f"{syntax_passed}/{syntax_total}",
            "total_modules": syntax_total,
            "avg_docstring_coverage": f"{avg_coverage:.2%}",
            "total_classes": sum(len(classes) for classes in results["class_definitions"].values())
        }
        
        return results
    
    def print_validation_report(self, results: Dict[str, any]):
        """Print a formatted validation report."""
        print("\n" + "="*60)
        print("🧠 PWMK VALIDATION REPORT")
        print("="*60)
        
        summary = results["summary"]
        print(f"📊 Project Structure: {summary['structure_score']}")
        print(f"🐍 Python Syntax: {summary['syntax_score']}")
        print(f"📝 Documentation Coverage: {summary['avg_docstring_coverage']}")
        print(f"📚 Total Modules: {summary['total_modules']}")
        print(f"🏗️  Total Classes: {summary['total_classes']}")
        
        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        
        # Failed structure checks
        structure_failures = [
            k for k, v in results["project_structure"].items() if not v
        ]
        if structure_failures:
            print("\n❌ Missing Structure Elements:")
            for failure in structure_failures:
                print(f"  - {failure}")
        
        # Syntax errors
        syntax_failures = [
            k for k, v in results["python_syntax"].items() if not v
        ]
        if syntax_failures:
            print("\n❌ Syntax Errors:")
            for failure in syntax_failures:
                print(f"  - {failure}")
        
        # Low documentation coverage
        low_coverage = [
            (k, v) for k, v in results["docstring_coverage"].items() if v < 0.5
        ]
        if low_coverage:
            print("\n⚠️  Low Documentation Coverage (<50%):")
            for module, coverage in low_coverage:
                print(f"  - {module}: {coverage:.1%}")
        
        print("\n✅ Validation Complete!")


def main():
    """Main validation entry point."""
    validator = PWMKValidator()
    results = validator.run_comprehensive_validation()
    validator.print_validation_report(results)
    
    # Return exit code based on critical failures
    structure_ok = all(results["project_structure"].values())
    syntax_ok = all(results["python_syntax"].values())
    
    if structure_ok and syntax_ok:
        print("\n🎉 All critical validations passed!")
        return 0
    else:
        print("\n🚨 Critical validation failures detected!")
        return 1


if __name__ == "__main__":
    sys.exit(main())