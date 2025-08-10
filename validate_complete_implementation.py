#!/usr/bin/env python3
"""
Complete implementation validation for PWMK autonomous SDLC execution.
Validates all three generations and quality gates.
"""

import sys
import time
import warnings
from pathlib import Path

# Add pwmk to path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

def print_header(title: str) -> None:
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_success(message: str) -> None:
    """Print success message."""
    print(f"✅ {message}")

def print_error(message: str) -> None:
    """Print error message."""
    print(f"❌ {message}")

def print_info(message: str) -> None:
    """Print info message."""
    print(f"ℹ️  {message}")

def validate_generation_1() -> bool:
    """Validate Generation 1: MAKE IT WORK (Simple)."""
    print_header("GENERATION 1: MAKE IT WORK (Simple)")
    
    try:
        from pwmk.core.beliefs import BeliefStore
        
        # Test basic functionality
        belief_store = BeliefStore()
        belief_store.add_belief("agent_0", "has(key)")
        belief_store.add_belief("agent_1", "at(room_1)")
        
        results = belief_store.query("has(X)")
        if len(results) == 1 and "key" in str(results):
            print_success("Basic belief addition and query working")
        else:
            print_error("Basic functionality failed")
            return False
        
        # Test nested beliefs
        belief_store.add_nested_belief("agent_0", "agent_1", "location(treasure)")
        nested_beliefs = belief_store.get_all_beliefs("agent_0")
        if any("believes" in belief for belief in nested_beliefs):
            print_success("Nested belief functionality working")
        else:
            print_error("Nested beliefs failed")
            return False
        
        print_success("Generation 1 validation PASSED")
        return True
        
    except Exception as e:
        print_error(f"Generation 1 validation FAILED: {e}")
        return False

def validate_generation_2() -> bool:
    """Validate Generation 2: MAKE IT ROBUST (Reliable)."""
    print_header("GENERATION 2: MAKE IT ROBUST (Reliable)")
    
    try:
        from pwmk.core.beliefs import BeliefStore
        from pwmk.security.input_sanitizer import SecurityError
        
        belief_store = BeliefStore()
        
        # Test security features
        security_tests_passed = 0
        
        # Test dangerous content blocking
        try:
            belief_store.add_belief("agent_0", "__import__('os')")
            print_error("Security failed - dangerous content allowed")
            return False
        except SecurityError:
            security_tests_passed += 1
        
        # Test SQL injection blocking
        try:
            belief_store.query("DROP TABLE users")
            print_error("Security failed - SQL injection not blocked")
            return False
        except SecurityError:
            security_tests_passed += 1
        
        # Test empty agent ID blocking
        try:
            belief_store.add_belief("", "valid_belief")
            print_error("Security failed - empty agent ID allowed")
            return False
        except SecurityError:
            security_tests_passed += 1
        
        if security_tests_passed == 3:
            print_success("Security validation passed (3/3 threats blocked)")
        
        # Test error recovery
        belief_store.add_belief("agent_0", "has(key)")
        results = belief_store.query("has(X)")
        if len(results) == 1:
            print_success("Error recovery working - system functional after security errors")
        else:
            print_error("Error recovery failed")
            return False
        
        # Test metrics collection
        stats = belief_store.get_performance_stats()
        required_stats = ["query_count", "belief_count", "total_agents", "total_facts"]
        if all(stat in stats for stat in required_stats):
            print_success(f"Metrics collection working: {stats['query_count']} queries, {stats['belief_count']} beliefs")
        else:
            print_error("Metrics collection failed")
            return False
        
        print_success("Generation 2 validation PASSED")
        return True
        
    except Exception as e:
        print_error(f"Generation 2 validation FAILED: {e}")
        return False

def validate_generation_3() -> bool:
    """Validate Generation 3: MAKE IT SCALE (Optimized)."""
    print_header("GENERATION 3: MAKE IT SCALE (Optimized)")
    
    try:
        from pwmk.core.beliefs import BeliefStore
        from pwmk.optimization.parallel_processing import get_parallel_processor
        from pwmk.optimization.auto_scaling import get_auto_scaler
        
        belief_store = BeliefStore()
        
        # Test batch operations
        batch_beliefs = [
            ("agent_0", "has(key)"),
            ("agent_1", "at(room_1)"), 
            ("agent_2", "sees(treasure)"),
            ("agent_3", "knows(secret)")
        ]
        
        start_time = time.time()
        results = belief_store.batch_add_beliefs(batch_beliefs, use_parallel=False)
        batch_duration = time.time() - start_time
        
        if all(results) and len(results) == 4:
            print_success(f"Batch add beliefs working: 4/4 successful in {batch_duration:.4f}s")
        else:
            print_error("Batch add beliefs failed")
            return False
        
        # Test batch queries
        queries = ["has(X)", "at(Y)", "sees(Z)", "knows(W)"]
        start_time = time.time()
        query_results = belief_store.batch_query(queries, use_parallel=False)
        query_duration = time.time() - start_time
        
        total_results = sum(len(r) for r in query_results)
        if len(query_results) == 4 and total_results == 4:
            print_success(f"Batch queries working: {total_results} results in {query_duration:.4f}s")
        else:
            print_error("Batch queries failed")
            return False
        
        # Test parallel processing setup
        parallel_processor = get_parallel_processor()
        if parallel_processor and parallel_processor.max_workers > 0:
            print_success(f"Parallel processing ready: {parallel_processor.max_workers} workers")
        else:
            print_error("Parallel processing setup failed")
            return False
        
        # Test auto-scaling setup
        auto_scaler = get_auto_scaler()
        if auto_scaler and hasattr(auto_scaler, 'start_monitoring'):
            print_success("Auto-scaling infrastructure ready")
        else:
            print_error("Auto-scaling setup failed")
            return False
        
        # Test performance stats
        stats = belief_store.get_performance_stats()
        if (stats["parallel_enabled"] and stats["caching_enabled"] and 
            "parallel_stats" in stats and "cache_stats" in stats):
            print_success("Advanced performance monitoring working")
        else:
            print_error("Performance monitoring incomplete")
            return False
        
        print_success("Generation 3 validation PASSED")
        return True
        
    except Exception as e:
        print_error(f"Generation 3 validation FAILED: {e}")
        return False

def validate_quality_gates() -> bool:
    """Validate mandatory quality gates."""
    print_header("QUALITY GATES VALIDATION")
    
    try:
        from pwmk.core.beliefs import BeliefStore
        from pwmk.security.input_sanitizer import SecurityError
        
        # Test system integration
        belief_store = BeliefStore()
        
        # Complete workflow test
        belief_store.add_belief("agent_0", "has(key)")
        belief_store.add_belief("agent_1", "at(room_1)")
        
        results = belief_store.query("has(X)")
        if len(results) != 1:
            print_error("System integration test failed")
            return False
        
        batch_beliefs = [("agent_2", "sees(treasure)"), ("agent_3", "knows(secret)")]
        batch_results = belief_store.batch_add_beliefs(batch_beliefs, use_parallel=False)
        if not all(batch_results):
            print_error("System integration test failed")
            return False
        
        all_results = belief_store.batch_query(["has(X)", "at(Y)", "sees(Z)"], use_parallel=False)
        if len(all_results) != 3:
            print_error("System integration test failed")
            return False
        
        print_success("System integration test passed")
        
        # Error recovery test
        try:
            belief_store.add_belief("", "invalid")
        except SecurityError:
            pass
        
        recovery_results = belief_store.query("has(X)")
        if len(recovery_results) == 1:
            print_success("Error recovery test passed")
        else:
            print_error("Error recovery test failed")
            return False
        
        # Performance test
        large_batch = [(f"perf_agent_{i}", f"perf_fact_{i}(value)") for i in range(50)]
        start_time = time.time()
        perf_results = belief_store.batch_add_beliefs(large_batch, use_parallel=False)
        perf_duration = time.time() - start_time
        
        if all(perf_results) and perf_duration < 5.0:
            print_success(f"Performance test passed: 50 beliefs in {perf_duration:.4f}s")
        else:
            print_error("Performance test failed")
            return False
        
        print_success("Quality gates validation PASSED")
        return True
        
    except Exception as e:
        print_error(f"Quality gates validation FAILED: {e}")
        return False

def validate_production_readiness() -> bool:
    """Validate production readiness."""
    print_header("PRODUCTION READINESS")
    
    try:
        # Check for required files and structure
        required_files = [
            "pyproject.toml",
            "README.md", 
            "LICENSE",
            "pwmk/__init__.py",
            "pwmk/core/__init__.py",
            "pwmk/security/__init__.py",
            "pwmk/optimization/__init__.py",
            "tests/test_generation_all.py",
            "docker-compose.yml",
            "Dockerfile"
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            print_error(f"Missing production files: {missing_files}")
            return False
        
        print_success("Production file structure complete")
        
        # Test import structure
        import pwmk
        from pwmk.core import BeliefStore, PerspectiveWorldModel
        from pwmk.security import SecurityError
        from pwmk.optimization import get_parallel_processor, get_auto_scaler
        
        print_success("Import structure verified")
        
        # Test package version
        if hasattr(pwmk, '__version__'):
            print_success(f"Package version: {pwmk.__version__}")
        else:
            print_error("Package version not defined")
            return False
        
        print_success("Production readiness PASSED")
        return True
        
    except Exception as e:
        print_error(f"Production readiness FAILED: {e}")
        return False

def main():
    """Run complete validation."""
    print_header("PWMK AUTONOMOUS SDLC VALIDATION")
    print_info("Validating complete three-generation implementation...")
    
    start_time = time.time()
    
    # Run all validations
    validations = [
        ("Generation 1 (Basic)", validate_generation_1),
        ("Generation 2 (Robust)", validate_generation_2), 
        ("Generation 3 (Scalable)", validate_generation_3),
        ("Quality Gates", validate_quality_gates),
        ("Production Readiness", validate_production_readiness)
    ]
    
    results = {}
    for name, validator in validations:
        try:
            results[name] = validator()
        except Exception as e:
            print_error(f"Validation '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    duration = time.time() - start_time
    print_header("VALIDATION SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        icon = "✅" if result else "❌"
        print(f"{icon} {name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} validations passed in {duration:.2f}s")
    
    if passed == total:
        print_success("🎉 AUTONOMOUS SDLC IMPLEMENTATION COMPLETE!")
        print_info("✓ Generation 1: Basic functionality implemented")
        print_info("✓ Generation 2: Robust security and reliability added")
        print_info("✓ Generation 3: Scalable optimization implemented")
        print_info("✓ Quality gates: All mandatory requirements met")
        print_info("✓ Production ready: Complete deployment package")
        return True
    else:
        print_error(f"❌ {total - passed} validation(s) failed - implementation incomplete")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)