#!/usr/bin/env python3
"""Test global deployment system."""

import sys
sys.path.insert(0, '/root/repo')

from pwmk.deployment import get_global_orchestrator

def test_global_deployment():
    """Test the global deployment orchestrator."""
    try:
        orch = get_global_orchestrator()
        print(f"✅ Global orchestrator initialized with {len(orch.regions)} regions")
        
        # Test request handling
        test_request = {
            'client_location': 'us-east',
            'client_id': 'test_client',
            'data': {'query': 'test belief reasoning'}
        }
        
        response = orch.handle_global_request(test_request)
        print(f"✅ Request handled: {response.get('region', 'unknown region')}")
        
        # Test metrics
        metrics = orch.get_global_metrics()
        print(f"✅ Global metrics: {metrics.total_active_regions} active regions, {metrics.total_active_instances} instances")
        
        print("🌍 Global deployment system working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Global deployment test failed: {e}")
        return False

if __name__ == "__main__":
    test_global_deployment()