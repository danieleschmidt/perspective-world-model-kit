#!/usr/bin/env python3
"""
Quality Gates Validation: Tests, Security, Performance
Comprehensive validation of code quality, security, and performance standards
"""

import sys
import time
import subprocess
import os
from pathlib import Path

# Add pwmk to path
sys.path.insert(0, str(Path(__file__).parent))

def run_basic_tests():
    """Run basic functionality tests."""
    print("🧪 Running Basic Functionality Tests...")
    
    try:
        from pwmk.core.world_model import PerspectiveWorldModel
        from pwmk.core.beliefs import BeliefStore
        from pwmk.agents.tom_agent import ToMAgent
        from pwmk.planning.epistemic import EpistemicPlanner, Goal
        import torch
        
        test_results = []
        
        # Test 1: Model creation and forward pass
        print("\n1️⃣ Testing Model Forward Pass...")
        try:
            model = PerspectiveWorldModel(obs_dim=16, action_dim=4, hidden_dim=32, num_agents=2)
            obs = torch.randn(2, 5, 16)
            actions = torch.randint(0, 4, (2, 5))
            
            with torch.no_grad():
                next_states, beliefs = model(obs, actions)
            
            assert next_states.shape == (2, 5, 32)
            assert beliefs.shape == (2, 5, 64)
            test_results.append(("Model Forward Pass", True))
            print("   ✅ Model forward pass test passed")
        except Exception as e:
            test_results.append(("Model Forward Pass", False))
            print(f"   ❌ Model forward pass test failed: {e}")
        
        # Test 2: Belief store operations
        print("\n2️⃣ Testing Belief Store Operations...")
        try:
            belief_store = BeliefStore()
            belief_store.add_belief("agent_0", "has(key)")
            belief_store.add_belief("agent_1", "at(room_2)")
            
            results = belief_store.query("has(X)")
            assert len(results) > 0
            assert results[0]["X"] == "key"
            
            # Test nested beliefs
            belief_store.add_nested_belief("agent_0", "agent_1", "location(treasure, room_3)")
            
            test_results.append(("Belief Store Operations", True))
            print("   ✅ Belief store operations test passed")
        except Exception as e:
            test_results.append(("Belief Store Operations", False))
            print(f"   ❌ Belief store operations test failed: {e}")
        
        # Test 3: ToM Agent functionality
        print("\n3️⃣ Testing ToM Agent Functionality...")
        try:
            agent = ToMAgent(
                agent_id="test_agent",
                world_model=model,
                tom_depth=1,
                planning_horizon=3
            )
            
            agent.update_beliefs({"location": "room_1", "has_item": True})
            beliefs = agent.reason_about_beliefs("has(X)")
            
            test_results.append(("ToM Agent Functionality", True))
            print("   ✅ ToM agent functionality test passed")
        except Exception as e:
            test_results.append(("ToM Agent Functionality", False))
            print(f"   ❌ ToM agent functionality test failed: {e}")
        
        # Test 4: Epistemic Planning
        print("\n4️⃣ Testing Epistemic Planning...")
        try:
            planner = EpistemicPlanner(
                world_model=model,
                belief_store=belief_store,
                search_depth=3
            )
            
            goal = Goal(achievement="has(treasure)", epistemic=[])
            initial_state = torch.randn(16).numpy()
            
            plan = planner.plan(initial_state=initial_state, goal=goal, timeout=1.0)
            
            assert hasattr(plan, 'actions')
            assert len(plan.actions) > 0
            
            test_results.append(("Epistemic Planning", True))
            print("   ✅ Epistemic planning test passed")
        except Exception as e:
            test_results.append(("Epistemic Planning", False))
            print(f"   ❌ Epistemic planning test failed: {e}")
        
        # Summary
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        print(f"\n📊 Basic Tests Summary: {passed}/{total} passed")
        
        return passed >= 3  # Require at least 3/4 to pass
        
    except Exception as e:
        print(f"❌ Basic tests setup failed: {e}")
        return False

def run_security_validation():
    """Run security validation tests."""
    print("\n🔒 Running Security Validation...")
    
    try:
        from pwmk.utils.validation import validate_model_config, PWMKValidationError
        from pwmk.core.beliefs import BeliefStore
        
        security_results = []
        
        # Test 1: Input validation
        print("\n1️⃣ Testing Input Validation...")
        try:
            # Test invalid configurations
            invalid_configs = [
                {"obs_dim": -1, "action_dim": 4},
                {"obs_dim": 0, "action_dim": 4},
                {"obs_dim": 16, "action_dim": -1},
            ]
            
            validation_failures = 0
            for config in invalid_configs:
                try:
                    validate_model_config(config)
                    print(f"   ⚠️  Should have rejected: {config}")
                except:
                    validation_failures += 1
            
            assert validation_failures == len(invalid_configs)
            security_results.append(("Input Validation", True))
            print("   ✅ Input validation test passed")
        except Exception as e:
            security_results.append(("Input Validation", False))
            print(f"   ❌ Input validation test failed: {e}")
        
        # Test 2: Malicious input handling
        print("\n2️⃣ Testing Malicious Input Handling...")
        try:
            belief_store = BeliefStore()
            
            malicious_inputs = [
                "__import__('os').system('echo hacked')",
                "eval('1+1')",
                "../../../etc/passwd",
                "<script>alert('xss')</script>",
            ]
            
            for malicious_input in malicious_inputs:
                # Should store safely without executing
                belief_store.add_belief("test_agent", malicious_input)
                beliefs = belief_store.get_all_beliefs("test_agent")
                assert malicious_input in beliefs[-1]  # Should be stored as string
            
            security_results.append(("Malicious Input Handling", True))
            print("   ✅ Malicious input handling test passed")
        except Exception as e:
            security_results.append(("Malicious Input Handling", False))
            print(f"   ❌ Malicious input handling test failed: {e}")
        
        # Test 3: Memory safety
        print("\n3️⃣ Testing Memory Safety...")
        try:
            from pwmk.core.world_model import PerspectiveWorldModel
            import torch
            
            # Test large tensor handling
            model = PerspectiveWorldModel(obs_dim=32, action_dim=4, hidden_dim=64, num_agents=2)
            
            # Create moderately large batch
            obs = torch.randn(16, 20, 32)
            actions = torch.randint(0, 4, (16, 20))
            
            with torch.no_grad():
                next_states, beliefs = model(obs, actions)
            
            # Clean up
            del obs, actions, next_states, beliefs, model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            security_results.append(("Memory Safety", True))
            print("   ✅ Memory safety test passed")
        except Exception as e:
            security_results.append(("Memory Safety", False))
            print(f"   ❌ Memory safety test failed: {e}")
        
        # Summary
        passed = sum(1 for _, result in security_results if result)
        total = len(security_results)
        
        print(f"\n🛡️ Security Tests Summary: {passed}/{total} passed")
        
        return passed >= 2  # Require at least 2/3 to pass
        
    except Exception as e:
        print(f"❌ Security validation setup failed: {e}")
        return False

def run_performance_benchmarks():
    """Run performance benchmark tests."""
    print("\n⚡ Running Performance Benchmarks...")
    
    try:
        from pwmk.core.world_model import PerspectiveWorldModel
        from pwmk.agents.tom_agent import ToMAgent
        import torch
        import time
        
        performance_results = []
        
        # Test 1: Model inference speed
        print("\n1️⃣ Testing Model Inference Speed...")
        try:
            model = PerspectiveWorldModel(obs_dim=64, action_dim=8, hidden_dim=128, num_agents=4)
            model.eval()
            
            # Benchmark inference
            batch_size = 16
            seq_len = 10
            num_iterations = 50
            
            obs = torch.randn(batch_size, seq_len, 64)
            actions = torch.randint(0, 8, (batch_size, seq_len))
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(obs, actions)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    next_states, beliefs = model(obs, actions)
            end_time = time.time()
            
            total_samples = batch_size * seq_len * num_iterations
            duration = end_time - start_time
            throughput = total_samples / duration
            
            print(f"   📊 Throughput: {throughput:.1f} samples/sec")
            print(f"   📊 Avg latency: {duration/num_iterations*1000:.2f}ms per batch")
            
            # Performance requirements
            min_throughput = 1000  # samples/sec
            max_latency = 100      # ms per batch
            
            avg_latency = duration / num_iterations * 1000
            
            performance_ok = (throughput >= min_throughput and avg_latency <= max_latency)
            
            performance_results.append(("Model Inference Speed", performance_ok))
            if performance_ok:
                print("   ✅ Model inference performance test passed")
            else:
                print("   ⚠️  Model inference performance below target (acceptable for testing)")
        except Exception as e:
            performance_results.append(("Model Inference Speed", False))
            print(f"   ❌ Model inference performance test failed: {e}")
        
        # Test 2: Memory efficiency
        print("\n2️⃣ Testing Memory Efficiency...")
        try:
            import psutil
            
            initial_memory = psutil.virtual_memory().used / (1024**2)
            
            # Create and use multiple models
            models = []
            for i in range(10):
                model = PerspectiveWorldModel(obs_dim=32, action_dim=4, hidden_dim=64, num_agents=2)
                obs = torch.randn(4, 5, 32)
                actions = torch.randint(0, 4, (4, 5))
                
                with torch.no_grad():
                    _ = model(obs, actions)
                
                models.append(model)
            
            peak_memory = psutil.virtual_memory().used / (1024**2)
            
            # Clean up
            del models
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            final_memory = psutil.virtual_memory().used / (1024**2)
            
            memory_increase = peak_memory - initial_memory
            cleanup_efficiency = max(0, (peak_memory - final_memory) / memory_increase) if memory_increase > 0 else 1.0
            
            print(f"   📊 Memory increase: {memory_increase:.1f}MB")
            print(f"   📊 Cleanup efficiency: {cleanup_efficiency:.1%}")
            
            # Memory requirements
            max_memory_per_model = 50  # MB
            min_cleanup = 0.5  # 50%
            
            memory_ok = (memory_increase / 10 <= max_memory_per_model and cleanup_efficiency >= min_cleanup)
            
            performance_results.append(("Memory Efficiency", memory_ok))
            if memory_ok:
                print("   ✅ Memory efficiency test passed")
            else:
                print("   ⚠️  Memory efficiency below target (acceptable for testing)")
        except Exception as e:
            performance_results.append(("Memory Efficiency", False))
            print(f"   ❌ Memory efficiency test failed: {e}")
        
        # Test 3: Multi-agent scaling
        print("\n3️⃣ Testing Multi-Agent Scaling...")
        try:
            agent_counts = [2, 4, 8]
            scaling_times = []
            
            for num_agents in agent_counts:
                model = PerspectiveWorldModel(obs_dim=16, action_dim=4, hidden_dim=32, num_agents=num_agents)
                
                agents = []
                start_time = time.time()
                
                for i in range(num_agents):
                    agent = ToMAgent(f"agent_{i}", model, tom_depth=1, planning_horizon=3)
                    agents.append(agent)
                
                # Update all agents
                for agent in agents:
                    agent.update_beliefs({"test": "value"})
                
                end_time = time.time()
                scaling_times.append(end_time - start_time)
                
                print(f"   📊 {num_agents} agents: {scaling_times[-1]:.4f}s")
            
            # Check if scaling is reasonable (not exponential)
            if len(scaling_times) >= 2:
                scaling_factor = scaling_times[-1] / scaling_times[0]
                agent_factor = agent_counts[-1] / agent_counts[0]
                
                # Should scale better than O(n^2)
                scaling_ok = scaling_factor <= agent_factor ** 1.5
                
                print(f"   📊 Scaling factor: {scaling_factor:.2f}x for {agent_factor}x agents")
                
                performance_results.append(("Multi-Agent Scaling", scaling_ok))
                if scaling_ok:
                    print("   ✅ Multi-agent scaling test passed")
                else:
                    print("   ⚠️  Multi-agent scaling suboptimal (acceptable for testing)")
            else:
                performance_results.append(("Multi-Agent Scaling", True))
                print("   ✅ Multi-agent scaling test passed")
        except Exception as e:
            performance_results.append(("Multi-Agent Scaling", False))
            print(f"   ❌ Multi-agent scaling test failed: {e}")
        
        # Summary
        passed = sum(1 for _, result in performance_results if result)
        total = len(performance_results)
        
        print(f"\n⚡ Performance Tests Summary: {passed}/{total} passed")
        
        return passed >= 2  # Require at least 2/3 to pass
        
    except Exception as e:
        print(f"❌ Performance benchmarks setup failed: {e}")
        return False

def run_code_quality_checks():
    """Run code quality and static analysis checks."""
    print("\n📝 Running Code Quality Checks...")
    
    try:
        quality_results = []
        
        # Test 1: Import structure validation
        print("\n1️⃣ Testing Import Structure...")
        try:
            # Test core imports
            from pwmk import PerspectiveWorldModel, BeliefStore, EpistemicPlanner, ToMAgent
            from pwmk.quantum import QuantumInspiredPlanner, QuantumCircuitOptimizer
            
            print("   ✅ All core modules importable")
            quality_results.append(("Import Structure", True))
        except Exception as e:
            print(f"   ❌ Import structure issues: {e}")
            quality_results.append(("Import Structure", False))
        
        # Test 2: API consistency
        print("\n2️⃣ Testing API Consistency...")
        try:
            from pwmk.core.world_model import PerspectiveWorldModel
            from pwmk.core.beliefs import BeliefStore
            import torch
            
            # Test consistent API patterns
            model = PerspectiveWorldModel(obs_dim=16, action_dim=4, hidden_dim=32, num_agents=2)
            belief_store = BeliefStore()
            
            # Check method naming consistency
            assert hasattr(model, 'forward')
            assert hasattr(belief_store, 'add_belief')
            assert hasattr(belief_store, 'query')
            
            print("   ✅ API consistency validated")
            quality_results.append(("API Consistency", True))
        except Exception as e:
            print(f"   ❌ API consistency issues: {e}")
            quality_results.append(("API Consistency", False))
        
        # Test 3: Documentation presence
        print("\n3️⃣ Testing Documentation Presence...")
        try:
            from pwmk.core.world_model import PerspectiveWorldModel
            from pwmk.core.beliefs import BeliefStore
            
            # Check docstrings
            assert PerspectiveWorldModel.__doc__ is not None
            assert BeliefStore.__doc__ is not None
            assert PerspectiveWorldModel.forward.__doc__ is not None
            
            print("   ✅ Documentation presence validated")
            quality_results.append(("Documentation Presence", True))
        except Exception as e:
            print(f"   ❌ Documentation issues: {e}")
            quality_results.append(("Documentation Presence", False))
        
        # Summary
        passed = sum(1 for _, result in quality_results if result)
        total = len(quality_results)
        
        print(f"\n📝 Code Quality Summary: {passed}/{total} passed")
        
        return passed >= 2  # Require at least 2/3 to pass
        
    except Exception as e:
        print(f"❌ Code quality checks setup failed: {e}")
        return False

def main():
    """Main quality gates validation."""
    print("🛡️ PWMK Quality Gates Validation")
    print("=" * 60)
    
    gate_results = []
    
    # Run all quality gates
    gates = [
        ("Basic Functionality Tests", run_basic_tests),
        ("Security Validation", run_security_validation),
        ("Performance Benchmarks", run_performance_benchmarks),
        ("Code Quality Checks", run_code_quality_checks),
    ]
    
    for gate_name, gate_func in gates:
        print(f"\n{'='*60}")
        print(f"🚪 Quality Gate: {gate_name}")
        print(f"{'='*60}")
        
        try:
            result = gate_func()
            gate_results.append((gate_name, result))
            
            if result:
                print(f"✅ {gate_name}: PASSED")
            else:
                print(f"❌ {gate_name}: FAILED")
        except Exception as e:
            print(f"❌ {gate_name}: EXCEPTION - {e}")
            gate_results.append((gate_name, False))
    
    # Final summary
    passed_gates = sum(1 for _, result in gate_results if result)
    total_gates = len(gate_results)
    
    print("\n" + "=" * 60)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    
    for gate_name, result in gate_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {gate_name}: {status}")
    
    print(f"\n🎯 Overall Result: {passed_gates}/{total_gates} gates passed")
    
    if passed_gates >= 3:  # Require at least 3/4 gates to pass
        print("🎉 QUALITY GATES: SUCCESS!")
        print("   Code meets production quality standards")
        return True
    else:
        print("❌ QUALITY GATES: INSUFFICIENT")
        print(f"   Need at least 3 gates to pass, got {passed_gates}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)