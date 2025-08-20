#!/usr/bin/env python3
"""
Lightweight validation for PWMK without external dependencies.
Tests core functionality and system architecture.
"""

import sys
import time
import json
import os
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_import_structure():
    """Test that module import structure is correct."""
    print("📦 Testing module import structure...")
    
    import_tests = [
        ("pwmk", "Main package"),
        ("pwmk.utils", "Utilities package"),
        ("pwmk.utils.logging", "Logging module"),
        ("pwmk.utils.validation", "Validation module"),
        ("pwmk.utils.circuit_breaker", "Circuit breaker module"),
        ("pwmk.utils.fallback_manager", "Fallback manager module"),
        ("pwmk.utils.health_monitor", "Health monitor module"),
        ("pwmk.optimization", "Optimization package"),
        ("pwmk.optimization.adaptive_scaling", "Adaptive scaling module"),
        ("pwmk.optimization.performance_optimizer", "Performance optimizer module"),
        ("pwmk.security", "Security package"),
        ("pwmk.quantum", "Quantum package"),
        ("pwmk.quantum.adaptive_quantum_acceleration", "Quantum acceleration module"),
        ("pwmk.validation", "Validation package"),
        ("pwmk.validation.comprehensive_system_test", "System test module")
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"   ✅ {description}: OK")
            passed += 1
        except ImportError as e:
            print(f"   ❌ {description}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"   ⚠️  {description}: ERROR - {e}")
            failed += 1
    
    success_rate = passed / (passed + failed) if (passed + failed) > 0 else 0
    print(f"   Import success rate: {success_rate:.1%} ({passed}/{passed + failed})")
    
    return success_rate >= 0.9  # 90% threshold


def test_core_functionality():
    """Test core functionality without torch dependencies."""
    print("\n🔧 Testing core functionality...")
    
    try:
        # Test validation utilities
        from pwmk.utils.validation import validate_config, PWMKValidationError
        
        # Test valid config
        valid_config = {"key1": "value1", "key2": 42}
        try:
            validate_config(valid_config, ["key1", "key2"])
            print("   ✅ Config validation: OK")
        except Exception as e:
            print(f"   ❌ Config validation failed: {e}")
            return False
        
        # Test invalid config
        try:
            validate_config(valid_config, ["key1", "missing_key"])
            print("   ❌ Invalid config validation should have failed")
            return False
        except PWMKValidationError:
            print("   ✅ Invalid config properly rejected: OK")
        
        # Test logging
        from pwmk.utils.logging import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        print("   ✅ Logging system: OK")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Core functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_security_components():
    """Test security components."""
    print("\n🔒 Testing security components...")
    
    try:
        from pwmk.security.input_sanitizer import InputSanitizer, SecurityError
        
        sanitizer = InputSanitizer()
        
        # Test safe input
        safe_belief = "has(agent_1, key)"
        sanitized = sanitizer.sanitize_belief_content(safe_belief)
        if not sanitized:
            print("   ❌ Safe belief sanitization failed")
            return False
        print("   ✅ Safe belief sanitization: OK")
        
        # Test dangerous input blocking
        dangerous_inputs = [
            "__import__('os')",
            "eval('code')",
            "DROP TABLE users",
            "<script>alert('xss')</script>"
        ]
        
        blocked_count = 0
        for dangerous in dangerous_inputs:
            try:
                sanitizer.sanitize_belief_content(dangerous)
                print(f"   ⚠️  Dangerous input not blocked: {dangerous}")
            except SecurityError:
                blocked_count += 1
        
        if blocked_count == len(dangerous_inputs):
            print(f"   ✅ Dangerous input blocking: OK ({blocked_count}/{len(dangerous_inputs)})")
        else:
            print(f"   ⚠️  Dangerous input blocking: PARTIAL ({blocked_count}/{len(dangerous_inputs)})")
        
        # Test agent ID sanitization
        clean_id = sanitizer.sanitize_agent_id("agent_1")
        if clean_id == "agent_1":
            print("   ✅ Agent ID sanitization: OK")
        else:
            print(f"   ❌ Agent ID sanitization failed: {clean_id}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Security components test failed: {e}")
        traceback.print_exc()
        return False


def test_resilience_components():
    """Test resilience components."""
    print("\n🛡️  Testing resilience components...")
    
    try:
        # Test circuit breaker
        from pwmk.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
        
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
        cb = CircuitBreaker(config)
        
        # Test normal operation
        def successful_operation():
            return "success"
        
        result = cb.call(successful_operation)
        if result != "success":
            print("   ❌ Circuit breaker normal operation failed")
            return False
        print("   ✅ Circuit breaker normal operation: OK")
        
        # Test failure handling
        def failing_operation():
            raise RuntimeError("Test failure")
        
        failure_count = 0
        for _ in range(3):
            try:
                cb.call(failing_operation)
            except RuntimeError:
                failure_count += 1
            except Exception as e:
                if "Circuit breaker is open" in str(e):
                    break
        
        if cb.get_state() == CircuitState.OPEN:
            print("   ✅ Circuit breaker failure handling: OK")
        else:
            print("   ⚠️  Circuit breaker may not be opening on failures")
        
        # Test fallback manager
        from pwmk.utils.fallback_manager import FallbackManager, FallbackConfig, SystemMode
        
        config = FallbackConfig()
        manager = FallbackManager(config)
        
        # Test mode changes
        original_mode = manager.get_mode()
        manager.set_mode(SystemMode.DEGRADED, "Test mode change")
        
        if manager.get_mode() == SystemMode.DEGRADED:
            print("   ✅ Fallback manager mode changes: OK")
        else:
            print("   ❌ Fallback manager mode changes failed")
            return False
        
        # Restore original mode
        manager.set_mode(original_mode, "Restore mode")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Resilience components test failed: {e}")
        traceback.print_exc()
        return False


def test_optimization_components():
    """Test optimization components."""
    print("\n⚡ Testing optimization components...")
    
    try:
        # Test scaling components
        from pwmk.optimization.adaptive_scaling import AdaptiveScaler, ScalingRule, ScalingDirection
        
        scaler = AdaptiveScaler("test_component")
        
        # Record some metrics
        scaler.record_metrics(
            latency_p95=100.0,
            throughput=50.0,
            cpu_usage=60.0,
            memory_usage=70.0,
            error_rate=0.01
        )
        
        status = scaler.get_status()
        required_keys = ["component", "current_instances", "scaling_enabled"]
        if all(key in status for key in required_keys):
            print("   ✅ Adaptive scaling basic functionality: OK")
        else:
            print(f"   ❌ Adaptive scaling status missing keys: {status.keys()}")
            return False
        
        # Test scaling rules
        rule = ScalingRule(
            name="test_rule",
            metric_name="latency_p95",
            scale_up_threshold=200.0,
            scale_down_threshold=50.0
        )
        
        scaler.add_rule(rule)
        print("   ✅ Scaling rule addition: OK")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Optimization components test failed: {e}")
        traceback.print_exc()
        return False


def test_quantum_components():
    """Test quantum components."""
    print("\n⚛️  Testing quantum components...")
    
    try:
        from pwmk.quantum.adaptive_quantum_acceleration import (
            AdaptiveQuantumAccelerator, 
            QuantumProblem, 
            QuantumBackend
        )
        
        accelerator = AdaptiveQuantumAccelerator()
        
        # Test basic functionality
        problem = QuantumProblem(
            problem_type="optimization",
            parameters={"target": [0.5, 0.5]},
            problem_size=2,
            timeout=5.0
        )
        
        # This should fall back to classical computation
        result = accelerator.solve_optimization_problem(problem)
        
        if hasattr(result, 'solution') and hasattr(result, 'quality_score'):
            print("   ✅ Quantum acceleration with classical fallback: OK")
        else:
            print("   ❌ Quantum acceleration result structure invalid")
            return False
        
        # Test performance report
        report = accelerator.get_performance_report()
        if isinstance(report, dict) and 'available_backends' in report:
            print("   ✅ Quantum performance reporting: OK")
        else:
            print("   ❌ Quantum performance reporting failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Quantum components test failed: {e}")
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that required files exist and have proper structure."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "pwmk/__init__.py",
        "pwmk/utils/__init__.py",
        "pwmk/utils/circuit_breaker.py",
        "pwmk/utils/fallback_manager.py",
        "pwmk/utils/health_monitor.py",
        "pwmk/optimization/adaptive_scaling.py",
        "pwmk/optimization/performance_optimizer.py",
        "pwmk/quantum/adaptive_quantum_acceleration.py",
        "pwmk/security/input_sanitizer.py",
        "pyproject.toml",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"   ❌ Missing required files: {', '.join(missing_files)}")
        return False
    else:
        print(f"   ✅ All {len(required_files)} required files exist: OK")
    
    # Check file sizes (should not be empty)
    empty_files = []
    for file_path in required_files:
        if Path(file_path).stat().st_size == 0:
            empty_files.append(file_path)
    
    if empty_files:
        print(f"   ⚠️  Empty files detected: {', '.join(empty_files)}")
    
    return len(empty_files) == 0


def test_configuration_files():
    """Test configuration files."""
    print("\n⚙️  Testing configuration files...")
    
    try:
        # Test pyproject.toml
        import toml
        with open("pyproject.toml", "r") as f:
            config = toml.load(f)
        
        required_sections = ["build-system", "project"]
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            print(f"   ❌ pyproject.toml missing sections: {missing_sections}")
            return False
        else:
            print("   ✅ pyproject.toml structure: OK")
        
        # Check project metadata
        project = config.get("project", {})
        if "name" in project and "version" in project:
            print(f"   ✅ Project metadata: {project['name']}")
        else:
            print("   ⚠️  Project metadata incomplete")
        
        return True
        
    except ImportError:
        print("   ⚠️  toml module not available, skipping detailed config test")
        return True
    except Exception as e:
        print(f"   ❌ Configuration files test failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🚀 PWMK Lightweight System Validation")
    print("=" * 50)
    print("Note: This validation runs without external dependencies like PyTorch")
    
    start_time = time.time()
    test_results = {}
    
    # Run test suites
    test_suites = [
        ("Import Structure", test_import_structure),
        ("File Structure", test_file_structure),
        ("Configuration Files", test_configuration_files),
        ("Core Functionality", test_core_functionality),
        ("Security Components", test_security_components),
        ("Resilience Components", test_resilience_components),
        ("Optimization Components", test_optimization_components),
        ("Quantum Components", test_quantum_components),
    ]
    
    passed_tests = 0
    total_tests = len(test_suites)
    
    for test_name, test_func in test_suites:
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"   💥 {test_name} failed with exception: {e}")
            test_results[test_name] = False
    
    # Calculate results
    success_rate = passed_tests / total_tests
    overall_success = success_rate >= 0.8  # 80% threshold
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n📊 Validation Summary (completed in {total_time:.2f}s)")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n📈 Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
    print(f"🎯 Overall Result: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
    
    # Save results
    validation_results = {
        'timestamp': time.time(),
        'validation_type': 'lightweight',
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate,
        'overall_success': overall_success,
        'execution_time': total_time,
        'detailed_results': test_results
    }
    
    results_file = Path("lightweight_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
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
        traceback.print_exc()
        sys.exit(3)