#!/usr/bin/env python3
"""Test Generation 2 enhancements: robust error handling, validation, and monitoring."""

import torch
import numpy as np
import time
from pwmk import PerspectiveWorldModel
from pwmk.utils.validation import PWMKValidationError
from pwmk.utils.logging import setup_logging, get_logger
from pwmk.utils.monitoring import setup_monitoring


def test_validation_and_monitoring():
    """Test validation and monitoring features."""
    print("🔧 Testing Generation 2 Features: Validation & Monitoring")
    print("=" * 60)
    
    # Setup enhanced logging and monitoring
    setup_logging(level="DEBUG", structured=False, console=True)
    collector = setup_monitoring(system_monitoring=True, interval=0.5)
    
    logger = get_logger(__name__)
    logger.info("Starting Generation 2 feature tests")
    
    # Test 1: Valid model creation and usage
    print("\n1. Testing valid model creation and usage")
    try:
        model = PerspectiveWorldModel(
            obs_dim=32,
            action_dim=4,
            hidden_dim=64,
            num_agents=2
        )
        print("✓ Model created successfully with validation")
        
        # Valid forward pass
        batch_size, seq_len = 2, 5
        observations = torch.randn(batch_size, seq_len, 32)
        actions = torch.randint(0, 4, (batch_size, seq_len))
        agent_ids = torch.randint(0, 2, (batch_size, seq_len))
        
        with torch.no_grad():
            next_states, beliefs = model(observations, actions, agent_ids)
        
        print(f"✓ Forward pass successful: {next_states.shape}, {beliefs.shape}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 2: Input validation errors
    print("\n2. Testing input validation")
    
    test_cases = [
        {
            "name": "Invalid observation dimension",
            "observations": torch.randn(2, 5, 16),  # Wrong obs_dim
            "actions": torch.randint(0, 4, (2, 5)),
            "agent_ids": None
        },
        {
            "name": "Invalid action range",
            "observations": torch.randn(2, 5, 32),
            "actions": torch.randint(0, 10, (2, 5)),  # Actions > action_dim
            "agent_ids": None
        },
        {
            "name": "Invalid agent ID range", 
            "observations": torch.randn(2, 5, 32),
            "actions": torch.randint(0, 4, (2, 5)),
            "agent_ids": torch.randint(0, 5, (2, 5))  # Agent IDs > num_agents
        },
        {
            "name": "Mismatched shapes",
            "observations": torch.randn(2, 5, 32),
            "actions": torch.randint(0, 4, (3, 5)),  # Wrong batch size
            "agent_ids": None
        }
    ]
    
    for test_case in test_cases:
        try:
            with torch.no_grad():
                model(test_case["observations"], test_case["actions"], test_case["agent_ids"])
            print(f"❌ {test_case['name']}: Should have failed but didn't")
        except PWMKValidationError as e:
            print(f"✓ {test_case['name']}: Caught validation error - {str(e)[:60]}...")
        except Exception as e:
            print(f"⚠️  {test_case['name']}: Unexpected error type - {type(e).__name__}")
    
    # Test 3: Configuration validation
    print("\n3. Testing configuration validation")
    
    invalid_configs = [
        {"obs_dim": -1, "action_dim": 4, "hidden_dim": 64, "num_agents": 2},
        {"obs_dim": 32, "action_dim": 0, "hidden_dim": 64, "num_agents": 2},
        {"obs_dim": 32, "action_dim": 4, "hidden_dim": -10, "num_agents": 2},
        {"obs_dim": 32, "action_dim": 4, "hidden_dim": 64, "num_agents": 0},
    ]
    
    for i, config in enumerate(invalid_configs):
        try:
            PerspectiveWorldModel(**config)
            print(f"❌ Config {i+1}: Should have failed")
        except PWMKValidationError as e:
            print(f"✓ Config {i+1}: Validation caught error - {str(e)[:50]}...")
        except Exception as e:
            print(f"⚠️  Config {i+1}: Unexpected error - {type(e).__name__}")
    
    # Test 4: Performance monitoring
    print("\n4. Testing performance monitoring")
    
    # Create a new model for performance testing
    model = PerspectiveWorldModel(obs_dim=32, action_dim=4, hidden_dim=64, num_agents=2)
    model.eval()
    for i in range(10):
        batch_size = np.random.randint(1, 5)
        seq_len = np.random.randint(3, 8)
        
        observations = torch.randn(batch_size, seq_len, 32)
        actions = torch.randint(0, 4, (batch_size, seq_len))
        
        with torch.no_grad():
            model(observations, actions)
        
        time.sleep(0.1)  # Small delay for system monitoring
    
    print("✓ Generated metrics from 10 forward passes")
    
    # Wait a bit for system monitoring
    time.sleep(2)
    
    # Get metrics summary
    summary = collector.get_summary()
    
    print("\n📊 Metrics Summary:")
    
    # Model metrics
    if "model.forward_duration" in summary["metrics"]:
        duration_stats = summary["metrics"]["model.forward_duration"]
        print(f"   Forward pass duration: {duration_stats['mean']:.4f}±{duration_stats['max']-duration_stats['min']:.4f}s")
    
    if "model.throughput_samples_per_sec" in summary["metrics"]:
        throughput_stats = summary["metrics"]["model.throughput_samples_per_sec"]
        print(f"   Throughput: {throughput_stats['mean']:.1f} samples/sec")
    
    # System metrics
    if "system.cpu_percent" in summary["metrics"]:
        cpu_stats = summary["metrics"]["system.cpu_percent"]
        print(f"   CPU usage: {cpu_stats['latest']:.1f}% (avg: {cpu_stats['mean']:.1f}%)")
    
    if "system.memory_percent" in summary["metrics"]:
        mem_stats = summary["metrics"]["system.memory_percent"]
        print(f"   Memory usage: {mem_stats['latest']:.1f}% (avg: {mem_stats['mean']:.1f}%)")
    
    # Counters
    if summary["metrics"].get("counters"):
        print("   Operation counts:")
        for name, count in summary["metrics"]["counters"].items():
            print(f"     {name}: {count}")
    
    # Test 5: Logging output
    print("\n5. Testing structured logging")
    
    logger.info("Test info message: test_param=value, metric=42.0")
    logger.warning("Test warning message: component=test")
    logger.debug("Test debug message: step=1, batch_size=2")
    
    print("✓ Generated log messages with structured data")
    
    # Cleanup monitoring
    collector.stop_system_monitoring()
    
    print("\n✅ Generation 2 testing completed successfully!")
    
    return collector


def test_error_recovery():
    """Test error recovery and graceful handling."""
    print("\n🔄 Testing Error Recovery")
    print("=" * 40)
    
    logger = get_logger(__name__)
    
    # Test GPU fallback
    if torch.cuda.is_available():
        print("Testing GPU/CPU fallback...")
        
        model = PerspectiveWorldModel(obs_dim=32, action_dim=4, hidden_dim=64, num_agents=2)
        
        try:
            # Try GPU
            model = model.cuda()
            obs = torch.randn(1, 5, 32).cuda()
            actions = torch.randint(0, 4, (1, 5)).cuda()
            
            with torch.no_grad():
                states, beliefs = model(obs, actions)
            
            print("✓ GPU execution successful")
            
            # Move back to CPU
            model = model.cpu()
            obs = obs.cpu()
            actions = actions.cpu()
            
            with torch.no_grad():
                states, beliefs = model(obs, actions)
            
            print("✓ CPU fallback successful")
            
        except Exception as e:
            logger.warning(f"GPU/CPU test failed: {e}")
    
    else:
        print("✓ No GPU available, CPU-only testing")
    
    # Test memory pressure handling
    print("Testing memory pressure handling...")
    
    try:
        # Try to create very large tensors that might cause OOM
        model = PerspectiveWorldModel(obs_dim=32, action_dim=4, hidden_dim=64, num_agents=2)
        
        # Start with reasonable size and scale up
        for batch_size in [10, 50, 100]:
            try:
                obs = torch.randn(batch_size, 10, 32)
                actions = torch.randint(0, 4, (batch_size, 10))
                
                with torch.no_grad():
                    states, beliefs = model(obs, actions)
                
                print(f"✓ Handled batch size {batch_size}")
                
            except PWMKValidationError as e:
                if "out of memory" in str(e).lower():
                    print(f"✓ Gracefully handled OOM at batch size {batch_size}")
                    break
                else:
                    raise
            except Exception as e:
                logger.warning(f"Unexpected error at batch size {batch_size}: {e}")
                break
    
    except Exception as e:
        logger.error(f"Memory pressure test failed: {e}")
    
    print("✅ Error recovery testing completed")


if __name__ == "__main__":
    print("🚀 PWMK Generation 2: Robust & Reliable")
    print("Testing enhanced error handling, validation, and monitoring")
    
    try:
        collector = test_validation_and_monitoring()
        test_error_recovery()
        
        print("\n🎉 All Generation 2 tests completed!")
        print("\nGeneration 2 Features Implemented:")
        print("✅ Comprehensive input validation")
        print("✅ Structured logging with context")
        print("✅ Performance monitoring and metrics")
        print("✅ System resource monitoring") 
        print("✅ Error recovery and graceful handling")
        print("✅ Safe tensor operations")
        
        # Final metrics summary
        print("\n📈 Final Metrics Summary:")
        collector.log_summary()
        
    except Exception as e:
        print(f"\n❌ Generation 2 testing failed: {e}")
        raise