#!/usr/bin/env python3
"""
Performance Analysis Framework
Analyzes system performance characteristics without heavy dependencies
"""

import os
import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor


class PerformanceAnalyzer:
    """Analyzes performance characteristics of PWMK system."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.results = {}
    
    def analyze_all(self) -> Dict[str, Any]:
        """Run all performance analyses."""
        print("⚡ PERFORMANCE ANALYSIS FRAMEWORK")
        print("=" * 60)
        
        self.analyze_codebase_metrics()
        self.analyze_module_complexity()
        self.analyze_file_structure()
        self.benchmark_file_operations()
        
        return self.results
    
    def analyze_codebase_metrics(self):
        """Analyze overall codebase performance metrics."""
        print("\n📊 CODEBASE METRICS")
        print("-" * 40)
        
        python_files = list(self.repo_root.glob("**/*.py"))
        total_lines = 0
        total_functions = 0
        total_classes = 0
        
        import ast
        
        for file_path in python_files:
            if "/.venv/" in str(file_path) or "__pycache__" in str(file_path):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    lines = len(content.splitlines())
                    total_lines += lines
                    
                    # Parse AST to count functions and classes
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            
            except Exception:
                continue
        
        print(f"✅ Total Python files: {len(python_files)}")
        print(f"✅ Total lines of code: {total_lines}")
        print(f"✅ Total functions: {total_functions}")
        print(f"✅ Total classes: {total_classes}")
        print(f"✅ Avg lines per file: {total_lines / len(python_files):.1f}")
        
        # Calculate complexity score
        complexity_score = min(1.0, total_lines / 50000)  # Scale to 50k lines
        print(f"✅ Complexity score: {complexity_score:.2%}")
        
        self.results["codebase"] = {
            "total_files": len(python_files),
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "avg_lines_per_file": total_lines / len(python_files),
            "complexity_score": complexity_score
        }
    
    def analyze_module_complexity(self):
        """Analyze module-level complexity."""
        print("\n🧩 MODULE COMPLEXITY")
        print("-" * 40)
        
        modules = [
            "pwmk/core/",
            "pwmk/agents/", 
            "pwmk/planning/",
            "pwmk/quantum/",
            "pwmk/optimization/",
            "pwmk/security/"
        ]
        
        module_metrics = {}
        for module in modules:
            module_path = self.repo_root / module
            if module_path.exists():
                py_files = list(module_path.glob("*.py"))
                total_lines = 0
                
                for file_path in py_files:
                    try:
                        with open(file_path, 'r') as f:
                            total_lines += len(f.readlines())
                    except Exception:
                        continue
                
                module_metrics[module] = {
                    "files": len(py_files),
                    "lines": total_lines
                }
                print(f"✅ {module}: {len(py_files)} files, {total_lines} lines")
            else:
                module_metrics[module] = {"files": 0, "lines": 0}
                print(f"❌ {module}: Missing")
        
        self.results["modules"] = module_metrics
    
    def analyze_file_structure(self):
        """Analyze file structure performance characteristics."""
        print("\n📁 FILE STRUCTURE")
        print("-" * 40)
        
        # Count different file types
        file_types = {}
        total_size = 0
        
        for file_path in self.repo_root.glob("**/*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                size = file_path.stat().st_size
                total_size += size
                
                if suffix in file_types:
                    file_types[suffix]["count"] += 1
                    file_types[suffix]["size"] += size
                else:
                    file_types[suffix] = {"count": 1, "size": size}
        
        # Show top file types
        sorted_types = sorted(file_types.items(), 
                             key=lambda x: x[1]["size"], reverse=True)[:10]
        
        for suffix, metrics in sorted_types:
            size_mb = metrics["size"] / (1024 * 1024)
            print(f"✅ {suffix or 'no ext'}: {metrics['count']} files, {size_mb:.1f}MB")
        
        print(f"✅ Total repository size: {total_size / (1024 * 1024):.1f}MB")
        
        self.results["file_structure"] = {
            "total_files": sum(ft["count"] for ft in file_types.values()),
            "total_size_mb": total_size / (1024 * 1024),
            "file_types": dict(sorted_types[:5])  # Top 5 only
        }
    
    def benchmark_file_operations(self):
        """Benchmark basic file operations."""
        print("\n⚡ FILE OPERATION BENCHMARKS")
        print("-" * 40)
        
        # Test file reading performance
        test_files = list(self.repo_root.glob("**/*.py"))[:20]  # Test 20 files
        
        # Sequential read test
        start_time = time.time()
        total_chars = 0
        for file_path in test_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    total_chars += len(content)
            except Exception:
                continue
        sequential_time = time.time() - start_time
        
        print(f"✅ Sequential read: {len(test_files)} files in {sequential_time:.3f}s")
        print(f"✅ Read throughput: {total_chars / sequential_time / 1000:.1f}K chars/sec")
        
        # Concurrent read test
        def read_file(file_path):
            try:
                with open(file_path, 'r') as f:
                    return len(f.read())
            except Exception:
                return 0
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(read_file, test_files))
        concurrent_time = time.time() - start_time
        concurrent_chars = sum(results)
        
        print(f"✅ Concurrent read: {len(test_files)} files in {concurrent_time:.3f}s")
        print(f"✅ Concurrent throughput: {concurrent_chars / concurrent_time / 1000:.1f}K chars/sec")
        
        # Calculate performance metrics
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
        print(f"✅ Concurrency speedup: {speedup:.2f}x")
        
        self.results["benchmarks"] = {
            "sequential_time": sequential_time,
            "concurrent_time": concurrent_time,
            "sequential_throughput": total_chars / sequential_time,
            "concurrent_throughput": concurrent_chars / concurrent_time,
            "concurrency_speedup": speedup,
            "files_tested": len(test_files)
        }
    
    def generate_summary(self):
        """Generate performance analysis summary."""
        codebase_score = min(1.0, self.results["codebase"]["complexity_score"])
        structure_score = min(1.0, self.results["file_structure"]["total_size_mb"] / 100)  # Scale to 100MB
        benchmark_score = min(1.0, self.results["benchmarks"]["concurrency_speedup"] / 4)  # Scale to 4x speedup
        
        overall_score = (codebase_score + structure_score + benchmark_score) / 3
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"📊 Codebase Complexity: {codebase_score:.2%}")
        print(f"📁 File Structure: {structure_score:.2%}")
        print(f"⚡ Benchmark Performance: {benchmark_score:.2%}")
        print("-" * 60)
        print(f"🎯 OVERALL PERFORMANCE SCORE: {overall_score:.2%}")
        
        if overall_score >= 0.9:
            status = "🟢 EXCELLENT PERFORMANCE"
        elif overall_score >= 0.8:
            status = "🟡 GOOD PERFORMANCE"
        elif overall_score >= 0.7:
            status = "🟠 ADEQUATE PERFORMANCE"
        else:
            status = "🔴 PERFORMANCE ISSUES"
        
        print(f"{status}")
        
        self.results["overall"] = {
            "score": overall_score,
            "status": status
        }
        
        return overall_score


def main():
    """Run performance analysis."""
    analyzer = PerformanceAnalyzer()
    results = analyzer.analyze_all()
    overall_score = analyzer.generate_summary()
    
    # Save results
    with open("performance_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to performance_analysis_results.json")
    return results


if __name__ == "__main__":
    main()