#!/usr/bin/env python3
"""
Comprehensive validation and quality gates for PWMK.
Runs the complete test suite and validates system health.
"""

import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pwmk.validation.comprehensive_system_test import get_test_suite, run_quality_gates
from pwmk.utils.health_monitor import get_health_monitor, start_health_monitoring
from pwmk.utils.fallback_manager import get_fallback_manager
from pwmk.optimization.adaptive_scaling import get_scaling_manager
from pwmk.optimization.performance_optimizer import get_performance_optimizer


def initialize_system_components():
    """Initialize all system components for testing."""
    print("🔧 Initializing system components...")
    
    try:
        # Initialize health monitoring
        health_monitor = get_health_monitor()
        start_health_monitoring()
        
        # Initialize fallback manager
        fallback_manager = get_fallback_manager()
        
        # Initialize scaling manager
        scaling_manager = get_scaling_manager()
        
        # Initialize performance optimizer
        performance_optimizer = get_performance_optimizer()
        
        print("✅ System components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize system components: {e}")
        return False


def run_system_health_check():
    """Run system health check."""
    print("\n🏥 Running system health check...")
    
    try:
        health_monitor = get_health_monitor()
        health_report = health_monitor.get_health_report()
        
        print(f"   Overall Status: {health_report['overall_status']}")
        print(f"   Uptime: {health_report['uptime_seconds']:.1f}s")
        print(f"   Components: {len(health_report['component_status'])}")
        
        if health_report['failed_components']:
            print(f"   ⚠️  Failed Components: {', '.join(health_report['failed_components'])}")
        
        if health_report['degraded_components']:
            print(f"   ⚠️  Degraded Components: {', '.join(health_report['degraded_components'])}")
        
        return health_report['overall_status'] in ['healthy', 'degraded']
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def run_performance_validation():
    """Run performance validation."""
    print("\n⚡ Running performance validation...")
    
    try:
        performance_optimizer = get_performance_optimizer()
        
        # Auto-optimize system
        optimization_results = performance_optimizer.auto_optimize_system()
        
        print("   Optimization Results:")
        for category, results in optimization_results.items():
            if results:
                print(f"     {category}: {len(results)} optimizations applied")
        
        # Get performance report
        performance_report = performance_optimizer.get_optimization_report()
        torch_settings = performance_report.get('torch_settings', {})
        
        print(f"   PyTorch Threads: {torch_settings.get('num_threads', 'unknown')}")
        print(f"   CUDA Available: {torch_settings.get('cuda_available', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance validation failed: {e}")
        return False


def run_security_validation():
    """Run security validation."""
    print("\n🔒 Running security validation...")
    
    try:
        from pwmk.security.input_sanitizer import get_sanitizer
        from pwmk.security.belief_validator import get_validator
        
        sanitizer = get_sanitizer()
        validator = get_validator()
        
        # Test sanitization
        test_cases = [
            "has(agent_1, key)",
            "believes(agent_2, location(treasure, room_3))",
            "safe_predicate(normal_argument)"
        ]
        
        for test_case in test_cases:
            try:
                sanitized = sanitizer.sanitize_belief_content(test_case)
                validated = validator.sanitize_and_validate(sanitized, "test_agent")
                if not validated:
                    print(f"   ⚠️  Failed to validate: {test_case}")
            except Exception as e:
                print(f"   ⚠️  Sanitization error for '{test_case}': {e}")
        
        # Test dangerous input blocking
        dangerous_inputs = [
            "__import__('os')",
            "eval('malicious code')",
            "'; DROP TABLE users; --"
        ]
        
        blocked_count = 0
        for dangerous_input in dangerous_inputs:
            try:
                sanitizer.sanitize_belief_content(dangerous_input)
                print(f"   ⚠️  Dangerous input not blocked: {dangerous_input}")
            except:
                blocked_count += 1
        
        print(f"   Blocked {blocked_count}/{len(dangerous_inputs)} dangerous inputs")
        
        return blocked_count == len(dangerous_inputs)
        
    except Exception as e:
        print(f"❌ Security validation failed: {e}")
        return False


def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running integration tests...")
    
    try:
        # Test belief store integration
        from pwmk.core.beliefs import BeliefStore
        
        store = BeliefStore()
        
        # Add some beliefs
        test_beliefs = [
            ("agent_1", "has(key, agent_1)"),
            ("agent_2", "believes(agent_1, location(treasure, room_3))"),
            ("agent_3", "at(room_2)")
        ]
        
        for agent_id, belief in test_beliefs:
            store.add_belief(agent_id, belief)
        
        # Test querying
        results = store.query("has(key, X)")
        if not results:
            print("   ⚠️  Basic belief querying failed")
            return False
        
        # Test batch operations
        batch_results = store.batch_query([
            "has(key, X)",
            "believes(Y, location(treasure, Z))"
        ])
        
        if len(batch_results) != 2:
            print("   ⚠️  Batch querying failed")
            return False
        
        print("   ✅ Belief store integration tests passed")
        
        # Test scaling integration
        scaling_manager = get_scaling_manager()
        scaler = scaling_manager.register_component("integration_test")
        
        # Record metrics
        scaler.record_metrics(
            latency_p95=150.0,
            throughput=75.0,
            cpu_usage=65.0,
            memory_usage=70.0,
            error_rate=0.02
        )
        
        status = scaler.get_status()
        if not status.get('current_instances'):
            print("   ⚠️  Scaling integration failed")
            return False
        
        print("   ✅ Scaling integration tests passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🚀 PWMK Comprehensive System Validation")
    print("=" * 50)
    
    start_time = time.time()
    validation_results = {
        'timestamp': time.time(),
        'system_initialization': False,
        'health_check': False,
        'performance_validation': False,
        'security_validation': False,
        'integration_tests': False,
        'quality_gates': False,
        'overall_success': False
    }
    
    try:
        # 1. Initialize system components
        validation_results['system_initialization'] = initialize_system_components()
        if not validation_results['system_initialization']:
            print("❌ System initialization failed - stopping validation")
            return validation_results
        
        # Wait for health monitoring to start
        time.sleep(2)
        
        # 2. System health check
        validation_results['health_check'] = run_system_health_check()
        
        # 3. Performance validation
        validation_results['performance_validation'] = run_performance_validation()
        
        # 4. Security validation
        validation_results['security_validation'] = run_security_validation()
        
        # 5. Integration tests
        validation_results['integration_tests'] = run_integration_tests()
        
        # 6. Run quality gates (critical tests)
        print("\n🛡️  Running quality gates...")
        try:
            quality_results = run_quality_gates()
            critical_failures = quality_results.get('critical_failures', [])
            
            if critical_failures:
                print(f"   ❌ {len(critical_failures)} critical failures detected:")
                for failure in critical_failures:
                    print(f"      - {failure['test_name']}: {failure['error_message']}")
                validation_results['quality_gates'] = False
            else:
                success_rate = quality_results['summary']['success_rate']
                total_tests = quality_results['summary']['total']
                passed_tests = quality_results['summary']['pass']
                
                print(f"   ✅ Quality gates passed: {passed_tests}/{total_tests} tests ({success_rate:.1%})")
                validation_results['quality_gates'] = success_rate >= 0.8  # 80% threshold
            
        except Exception as e:
            print(f"   ❌ Quality gates execution failed: {e}")
            validation_results['quality_gates'] = False
        
        # Calculate overall success
        critical_checks = [
            validation_results['system_initialization'],
            validation_results['health_check'],
            validation_results['security_validation'],
            validation_results['quality_gates']
        ]
        
        validation_results['overall_success'] = all(critical_checks)
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n📊 Validation Summary (completed in {total_time:.2f}s)")
        print("=" * 50)
        
        for check_name, result in validation_results.items():
            if check_name in ['timestamp', 'overall_success']:
                continue
            
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {check_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n🎯 Overall Result: {'✅ SUCCESS' if validation_results['overall_success'] else '❌ FAILURE'}")
        
        # Save results
        results_file = Path("validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return validation_results
        
    except Exception as e:
        print(f"\n💥 Validation failed with unexpected error: {e}")
        validation_results['overall_success'] = False
        return validation_results


if __name__ == "__main__":
    try:
        results = main()
        exit_code = 0 if results['overall_success'] else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Fatal error during validation: {e}")
        sys.exit(3)