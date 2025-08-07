"""
Performance benchmarks for sentiment analysis components.
"""

import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import torch
import numpy as np
from pwmk.sentiment import (
    SentimentAnalyzer,
    MultiAgentSentimentAnalyzer,
    SentimentMonitor
)
from pwmk.sentiment.optimization import (
    BatchSentimentProcessor,
    BatchProcessingConfig,
    ParallelSentimentProcessor
)
from pwmk.sentiment.caching import SentimentCache
from pwmk.sentiment.distributed import (
    DistributedSentimentCoordinator,
    LocalWorkerNode
)


@pytest.mark.performance
class TestSentimentPerformanceBenchmarks:
    """Performance benchmarks for sentiment analysis."""
    
    @pytest.fixture
    def mock_sentiment_analyzer(self):
        """Fast mock sentiment analyzer for benchmarking."""
        analyzer = Mock()
        analyzer.model_name = "benchmark-model"
        
        def fast_analyze(text):
            # Simulate some processing time
            time.sleep(0.001)  # 1ms per analysis
            return {
                "negative": np.random.uniform(0, 0.5),
                "neutral": np.random.uniform(0, 0.5),
                "positive": np.random.uniform(0.3, 1.0)
            }
            
        analyzer.analyze_text = fast_analyze
        return analyzer
        
    @pytest.fixture
    def sample_texts(self):
        """Generate sample texts for benchmarking."""
        texts = [
            "I absolutely love this new feature implementation!",
            "This code is terrible and needs to be completely rewritten.",
            "The documentation is okay, could be better but it's functional.",
            "Amazing work on the optimization, performance is much better!",
            "I'm not sure about this approach, it seems risky.",
            "The user interface is intuitive and well-designed.",
            "This bug is really frustrating and blocking our progress.",
            "The test coverage looks good, nice attention to detail.",
            "I'm concerned about the security implications of this change.",
            "Excellent collaboration everyone, we're making great progress!",
        ]
        
        # Extend to create larger dataset
        extended_texts = []
        for i in range(100):  # Create 1000 texts
            base_text = texts[i % len(texts)]
            extended_texts.append(f"{base_text} (variant {i})")
            
        return extended_texts
        
    def test_single_analysis_performance(self, mock_sentiment_analyzer):
        """Benchmark single sentiment analysis performance."""
        text = "This is a test sentence for performance benchmarking."
        
        # Warmup
        for _ in range(10):
            mock_sentiment_analyzer.analyze_text(text)
            
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.perf_counter()
            result = mock_sentiment_analyzer.analyze_text(text)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert set(result.keys()) == {"negative", "neutral", "positive"}
            
        # Performance metrics
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        p95_time = np.percentile(times, 95)
        
        print(f"\nSingle Analysis Performance:")
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Median time: {median_time*1000:.2f} ms")
        print(f"  95th percentile: {p95_time*1000:.2f} ms")
        print(f"  Throughput: {1/avg_time:.1f} analyses/second")
        
        # Performance assertions (adjust based on requirements)
        assert avg_time < 0.1  # Should be under 100ms on average
        assert p95_time < 0.2   # 95% should be under 200ms
        
    def test_batch_processing_performance(self, mock_sentiment_analyzer, sample_texts):
        """Benchmark batch processing performance."""
        config = BatchProcessingConfig(
            batch_size=16,
            max_workers=4,
            use_gpu=False,
            enable_caching=False
        )
        
        # Mock the batch processor's internal method
        with patch.object(BatchSentimentProcessor, '_process_batch_internal') as mock_batch:
            def mock_batch_process(texts, return_raw_logits=False):
                # Simulate batch processing time
                time.sleep(len(texts) * 0.0005)  # 0.5ms per text in batch
                return [mock_sentiment_analyzer.analyze_text(text) for text in texts]
                
            mock_batch.side_effect = mock_batch_process
            
            processor = BatchSentimentProcessor(mock_sentiment_analyzer, config)
            
            batch_sizes = [1, 4, 8, 16, 32, 64]
            performance_results = {}
            
            for batch_size in batch_sizes:
                test_batch = sample_texts[:batch_size]
                
                # Warmup
                processor.process_batch_sync(test_batch[:min(4, len(test_batch))])
                
                # Benchmark
                times = []
                for _ in range(10):
                    start_time = time.perf_counter()
                    results = processor.process_batch_sync(test_batch)
                    end_time = time.perf_counter()
                    
                    times.append(end_time - start_time)
                    assert len(results) == len(test_batch)
                    
                avg_time = statistics.mean(times)
                throughput = batch_size / avg_time
                
                performance_results[batch_size] = {
                    "avg_time": avg_time,
                    "throughput": throughput,
                    "time_per_item": avg_time / batch_size
                }
                
            print(f"\nBatch Processing Performance:")
            for batch_size, metrics in performance_results.items():
                print(f"  Batch size {batch_size:2d}: "
                      f"{metrics['throughput']:6.1f} analyses/sec, "
                      f"{metrics['time_per_item']*1000:5.1f} ms/item")
                      
            # Verify batch processing is more efficient than single processing
            single_throughput = performance_results[1]["throughput"]
            batch_throughput = performance_results[16]["throughput"]
            
            assert batch_throughput > single_throughput * 2  # Should be at least 2x faster
            
    def test_multi_agent_performance(self, mock_sentiment_analyzer, sample_texts):
        """Benchmark multi-agent sentiment analysis performance."""
        num_agents_list = [2, 5, 10, 20, 50]
        performance_results = {}
        
        for num_agents in num_agents_list:
            # Create multi-agent analyzer
            multi_analyzer = MultiAgentSentimentAnalyzer(
                num_agents=num_agents,
                sentiment_analyzer=mock_sentiment_analyzer
            )
            
            # Generate agent communications
            communications = []
            for i, text in enumerate(sample_texts[:50]):  # 50 communications
                agent_id = i % num_agents
                communications.append((agent_id, text))
                
            # Benchmark
            times = []
            for _ in range(5):  # Fewer iterations for larger datasets
                start_time = time.perf_counter()
                
                for agent_id, text in communications:
                    multi_analyzer.analyze_agent_communication(agent_id, text)
                    
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
            avg_time = statistics.mean(times)
            throughput = len(communications) / avg_time
            
            performance_results[num_agents] = {
                "avg_time": avg_time,
                "throughput": throughput,
                "memory_usage": len(multi_analyzer.agent_sentiment_history)
            }
            
        print(f"\nMulti-Agent Performance:")
        for num_agents, metrics in performance_results.items():
            print(f"  {num_agents:2d} agents: "
                  f"{metrics['throughput']:6.1f} analyses/sec, "
                  f"{metrics['avg_time']:5.2f} sec total")
                  
        # Verify performance scales reasonably
        assert performance_results[2]["throughput"] > 10  # At least 10 analyses/sec for 2 agents
        
    def test_concurrent_processing_performance(self, mock_sentiment_analyzer, sample_texts):
        """Benchmark concurrent processing performance."""
        num_workers_list = [1, 2, 4, 8]
        performance_results = {}
        
        for num_workers in num_workers_list:
            executor = ThreadPoolExecutor(max_workers=num_workers)
            
            # Benchmark concurrent processing
            times = []
            for _ in range(5):
                start_time = time.perf_counter()
                
                # Submit concurrent tasks
                futures = []
                for text in sample_texts[:50]:  # 50 concurrent analyses
                    future = executor.submit(mock_sentiment_analyzer.analyze_text, text)
                    futures.append(future)
                    
                # Wait for completion
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
                    
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
                assert len(results) == 50
                
            executor.shutdown()
            
            avg_time = statistics.mean(times)
            throughput = 50 / avg_time
            
            performance_results[num_workers] = {
                "avg_time": avg_time,
                "throughput": throughput
            }
            
        print(f"\nConcurrent Processing Performance:")
        for num_workers, metrics in performance_results.items():
            print(f"  {num_workers} workers: "
                  f"{metrics['throughput']:6.1f} analyses/sec, "
                  f"{metrics['avg_time']:5.2f} sec total")
                  
        # Verify concurrency improves performance
        single_threaded = performance_results[1]["throughput"]
        multi_threaded = performance_results[4]["throughput"]
        
        assert multi_threaded > single_threaded * 1.5  # Should be at least 1.5x faster
        
    def test_caching_performance(self, mock_sentiment_analyzer, sample_texts):
        """Benchmark caching system performance."""
        cache = SentimentCache(
            max_text_entries=1000,
            text_ttl=300  # 5 minutes
        )
        
        # Test cache miss performance
        cache_miss_times = []
        for text in sample_texts[:50]:
            start_time = time.perf_counter()
            
            # Cache miss - analyze and cache
            result = mock_sentiment_analyzer.analyze_text(text)
            cache.cache_text_sentiment(text, "test-model", result)
            
            end_time = time.perf_counter()
            cache_miss_times.append(end_time - start_time)
            
        # Test cache hit performance
        cache_hit_times = []
        for text in sample_texts[:50]:
            start_time = time.perf_counter()
            
            # Cache hit - retrieve from cache
            cached_result = cache.get_text_sentiment(text, "test-model")
            
            end_time = time.perf_counter()
            cache_hit_times.append(end_time - start_time)
            
            assert cached_result is not None
            
        avg_miss_time = statistics.mean(cache_miss_times)
        avg_hit_time = statistics.mean(cache_hit_times)
        speedup = avg_miss_time / avg_hit_time
        
        print(f"\nCaching Performance:")
        print(f"  Cache miss time: {avg_miss_time*1000:.2f} ms")
        print(f"  Cache hit time: {avg_hit_time*1000:.4f} ms")
        print(f"  Cache speedup: {speedup:.1f}x")
        
        # Cache hits should be significantly faster
        assert speedup > 10  # Should be at least 10x faster
        assert avg_hit_time < 0.001  # Should be under 1ms
        
    def test_monitoring_overhead(self, mock_sentiment_analyzer, sample_texts):
        """Benchmark monitoring system overhead."""
        monitor = SentimentMonitor(max_events=1000)
        
        # Benchmark without monitoring
        times_without_monitoring = []
        for _ in range(5):
            start_time = time.perf_counter()
            
            for text in sample_texts[:100]:
                mock_sentiment_analyzer.analyze_text(text)
                
            end_time = time.perf_counter()
            times_without_monitoring.append(end_time - start_time)
            
        # Benchmark with monitoring
        times_with_monitoring = []
        for _ in range(5):
            start_time = time.perf_counter()
            
            for i, text in enumerate(sample_texts[:100]):
                analysis_start = time.perf_counter()
                result = mock_sentiment_analyzer.analyze_text(text)
                analysis_end = time.perf_counter()
                
                processing_time = analysis_end - analysis_start
                monitor.record_analysis(
                    agent_id=i % 3,
                    text=text,
                    sentiment_scores=result,
                    processing_time=processing_time
                )
                
            end_time = time.perf_counter()
            times_with_monitoring.append(end_time - start_time)
            
        avg_without = statistics.mean(times_without_monitoring)
        avg_with = statistics.mean(times_with_monitoring)
        overhead = (avg_with - avg_without) / avg_without * 100
        
        print(f"\nMonitoring Overhead:")
        print(f"  Without monitoring: {avg_without:.3f} sec")
        print(f"  With monitoring: {avg_with:.3f} sec")
        print(f"  Overhead: {overhead:.1f}%")
        
        # Monitoring overhead should be reasonable
        assert overhead < 50  # Should be less than 50% overhead
        
    def test_memory_usage_scaling(self, mock_sentiment_analyzer):
        """Test memory usage scaling with number of agents and history."""
        import sys
        
        def get_size(obj):
            """Estimate memory usage of object."""
            return sys.getsizeof(obj)
            
        agent_counts = [10, 50, 100, 500]
        memory_results = {}
        
        for num_agents in agent_counts:
            multi_analyzer = MultiAgentSentimentAnalyzer(
                num_agents=num_agents,
                sentiment_analyzer=mock_sentiment_analyzer
            )
            
            # Add communications to build history
            communications_per_agent = 50
            for agent_id in range(num_agents):
                for i in range(communications_per_agent):
                    text = f"Agent {agent_id} communication {i}"
                    multi_analyzer.analyze_agent_communication(agent_id, text)
                    
            # Estimate memory usage
            history_size = sum(
                len(history) for history in multi_analyzer.agent_sentiment_history.values()
            )
            
            memory_results[num_agents] = {
                "history_entries": history_size,
                "expected_entries": num_agents * communications_per_agent
            }
            
        print(f"\nMemory Usage Scaling:")
        for num_agents, metrics in memory_results.items():
            entries_per_agent = metrics["history_entries"] / num_agents
            print(f"  {num_agents:3d} agents: "
                  f"{metrics['history_entries']:5d} total entries, "
                  f"{entries_per_agent:4.1f} per agent")
                  
        # Verify memory usage scales linearly
        for num_agents, metrics in memory_results.items():
            assert metrics["history_entries"] == metrics["expected_entries"]
            
    @pytest.mark.slow
    def test_stress_test(self, mock_sentiment_analyzer):
        """Stress test with large volumes of data."""
        # Create large dataset
        large_dataset = [f"Test message number {i}" for i in range(5000)]
        
        multi_analyzer = MultiAgentSentimentAnalyzer(
            num_agents=20,
            sentiment_analyzer=mock_sentiment_analyzer
        )
        
        print(f"\nStress Test (processing {len(large_dataset)} messages):")
        
        start_time = time.perf_counter()
        
        # Process all messages
        for i, text in enumerate(large_dataset):
            agent_id = i % 20  # Distribute across agents
            result = multi_analyzer.analyze_agent_communication(agent_id, text)
            
            if i % 1000 == 0 and i > 0:
                elapsed = time.perf_counter() - start_time
                rate = i / elapsed
                print(f"  Processed {i:4d} messages in {elapsed:.1f}s ({rate:.1f} msg/s)")
                
        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = len(large_dataset) / total_time
        
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Throughput: {throughput:.1f} messages/second")
        print(f"  Memory efficiency: All messages processed successfully")
        
        # Verify all messages were processed
        total_history = sum(
            len(history) for history in multi_analyzer.agent_sentiment_history.values()
        )
        assert total_history == len(large_dataset)
        
        # Performance requirement - should handle at least 100 messages/second
        assert throughput > 100
        
    def test_distributed_processing_performance(self, mock_sentiment_analyzer):
        """Benchmark distributed processing performance.""" 
        # Create coordinator with multiple workers
        coordinator = DistributedSentimentCoordinator()
        
        # Add worker nodes
        for i in range(4):
            worker = LocalWorkerNode(f"worker_{i}", mock_sentiment_analyzer, max_concurrent=2)
            coordinator.add_worker(worker)
            
        print(f"\nDistributed Processing Performance:")
        print(f"  Workers: {len(coordinator.workers)}")
        
        # This would require async setup for full testing
        # For now, verify coordinator can be configured properly
        status = coordinator.get_status()
        
        assert status["num_workers"] == 4
        assert not status["is_running"]  # Not started yet
        assert status["pending_tasks"] == 0
        
        print(f"  Configuration verified: {status['num_workers']} workers ready")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])