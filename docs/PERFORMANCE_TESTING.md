# Performance Testing Guide

## Overview

This document outlines the performance testing strategy for the Perspective World Model Kit (PWMK), including benchmarking methodologies, performance targets, and regression detection.

## Testing Framework

### Core Testing Tools

- **pytest-benchmark**: For microbenchmarks and function-level performance testing
- **memory-profiler**: For memory usage analysis
- **py-spy**: For production profiling
- **locust**: For load testing and stress testing
- **pytest-memray**: For detailed memory tracking

## Benchmark Categories

### 1. Core Model Performance

#### World Model Inference
```python
# Target: <10ms per forward pass (single agent)
@pytest.mark.benchmark(group="world_model")
def test_world_model_forward_pass(benchmark, world_model, sample_obs, sample_actions):
    result = benchmark(world_model.forward, sample_obs, sample_actions)
    assert result is not None
```

#### Belief Reasoning
```python
# Target: <100ms for belief queries
@pytest.mark.benchmark(group="belief_reasoning")
def test_belief_query_performance(benchmark, belief_store, complex_query):
    result = benchmark(belief_store.query, complex_query)
    assert len(result) > 0
```

#### Planning Algorithms
```python
# Target: <1s for 10-step planning horizon
@pytest.mark.benchmark(group="planning")
def test_epistemic_planning_performance(benchmark, planner, initial_state, goal):
    plan = benchmark(planner.plan, initial_state, goal)
    assert plan.success
```

### 2. Multi-Agent Scalability

#### Agent Count Scaling
```python
@pytest.mark.parametrize("num_agents", [2, 4, 8, 16, 32])
@pytest.mark.benchmark(group="scaling")
def test_multi_agent_scaling(benchmark, num_agents):
    """Test performance scaling with number of agents."""
    env = create_multi_agent_env(num_agents=num_agents)
    world_model = create_world_model(num_agents=num_agents)
    
    def run_episode():
        obs = env.reset()
        for _ in range(100):
            actions = world_model.predict_actions(obs)
            obs, _, done, _ = env.step(actions)
            if done:
                break
    
    benchmark(run_episode)
```

### 3. Memory Usage Benchmarks

#### Memory Efficiency
```python
@pytest.mark.benchmark(group="memory")
def test_memory_usage_world_model(benchmark):
    """Test memory usage during world model training."""
    
    @benchmark
    def train_world_model():
        model = PerspectiveWorldModel(
            obs_dim=64, action_dim=8, hidden_dim=256, num_agents=4
        )
        # Simulate training batch
        batch_size = 32
        obs = torch.randn(batch_size, 64)
        actions = torch.randn(batch_size, 8)
        
        loss = model.training_step(obs, actions)
        loss.backward()
        
        return model
```

### 4. Environment Performance

#### Environment Step Time
```python
@pytest.mark.benchmark(group="environments")
def test_environment_step_performance(benchmark, env_name):
    """Test environment step execution time."""
    env = create_environment(env_name)
    obs = env.reset()
    
    def step_env():
        actions = env.action_space.sample()
        return env.step(actions)
    
    benchmark(step_env)
```

## Performance Targets

### Latency Targets

| Component | Target Latency | Acceptable Threshold |
|-----------|----------------|---------------------|
| World Model Forward Pass | < 10ms | < 50ms |
| Belief Query (Simple) | < 10ms | < 50ms |
| Belief Query (Complex) | < 100ms | < 500ms |
| Planning (5 steps) | < 500ms | < 2s |
| Planning (10 steps) | < 1s | < 5s |
| Environment Step | < 1ms | < 10ms |

### Throughput Targets

| Operation | Target Throughput | Minimum Acceptable |
|-----------|------------------|-------------------|
| Belief Updates/sec | > 1000 | > 100 |
| Planning Queries/sec | > 10 | > 1 |
| Environment Steps/sec | > 1000 | > 100 |

### Memory Targets

| Component | Target Memory | Maximum Acceptable |
|-----------|---------------|--------------------|
| World Model (4 agents) | < 500MB | < 2GB |
| Belief Store (1000 facts) | < 100MB | < 500MB |
| Planning Cache | < 200MB | < 1GB |

## Benchmarking Commands

### Run All Benchmarks
```bash
# Run all performance benchmarks
pytest tests/benchmarks/ -v --benchmark-only

# Generate detailed report
pytest tests/benchmarks/ --benchmark-json=benchmark.json --benchmark-sort=mean

# Compare with baseline
pytest tests/benchmarks/ --benchmark-compare=baseline.json
```

### Memory Profiling
```bash
# Profile memory usage
python -m memory_profiler examples/train_world_model.py

# Detailed memory tracking
pytest tests/benchmarks/ --memray --benchmark-only
```

### Production Profiling
```bash
# Profile running application
py-spy record -o profile.svg -- python examples/run_simulation.py

# Top-like profiling
py-spy top --pid <process_id>
```

## Continuous Performance Monitoring

### GitHub Actions Integration

Performance benchmarks run automatically on:
- Pull requests (compare against main branch)
- Weekly scheduled runs (track performance over time)
- Release tags (establish performance baselines)

### Regression Detection

Performance regressions are detected when:
- Latency increases by > 20% compared to baseline
- Memory usage increases by > 30% compared to baseline
- Throughput decreases by > 20% compared to baseline

### Performance Dashboard

Track performance metrics over time:
- Grafana dashboard for real-time metrics
- Historical trend analysis
- Performance alerts and notifications

## Optimization Guidelines

### World Model Optimization

1. **Batch Processing**: Always use batched operations
2. **Memory Management**: Clear intermediate tensors
3. **Model Pruning**: Remove unnecessary parameters
4. **Quantization**: Use lower precision when appropriate

### Belief Reasoning Optimization

1. **Query Optimization**: Use indexed queries
2. **Cache Management**: Implement intelligent caching
3. **Parallel Processing**: Distribute belief updates
4. **Memory Pooling**: Reuse memory allocations

### Environment Optimization

1. **Vectorization**: Use vectorized environments
2. **Lazy Evaluation**: Compute only when needed
3. **State Caching**: Cache expensive computations
4. **Parallel Execution**: Run environments in parallel

## Performance Testing Best Practices

### Test Design

1. **Realistic Workloads**: Use representative test scenarios
2. **Warm-up Runs**: Allow for JIT compilation and caching
3. **Statistical Significance**: Run multiple iterations
4. **Isolation**: Avoid interference between tests

### Data Generation

1. **Reproducible Data**: Use fixed random seeds
2. **Varied Scenarios**: Test different data distributions
3. **Edge Cases**: Include boundary conditions
4. **Realistic Scale**: Match production data sizes

### Result Analysis

1. **Baseline Comparison**: Always compare against baselines
2. **Statistical Analysis**: Use proper statistical methods
3. **Trend Detection**: Monitor performance over time
4. **Root Cause Analysis**: Investigate performance changes

## Integration with Development Workflow

### Pre-commit Hooks
```bash
# Run quick performance checks before commit
make benchmark-quick
```

### Pull Request Checks
```bash
# Run comprehensive benchmarks for PR validation
make benchmark-pr
```

### Release Validation
```bash
# Full performance validation before release
make benchmark-release
```

## Troubleshooting Performance Issues

### Common Performance Bottlenecks

1. **Memory Leaks**: Use memory profilers to detect
2. **CPU Bound Operations**: Profile and optimize hot paths
3. **I/O Bottlenecks**: Optimize file and network operations
4. **GPU Utilization**: Monitor GPU usage and efficiency

### Debugging Tools

1. **cProfile**: Python built-in profiler
2. **line_profiler**: Line-by-line profiling
3. **memory_profiler**: Memory usage profiling
4. **py-spy**: Low-overhead production profiler

### Performance Analysis Workflow

1. **Identify Bottleneck**: Use profiling tools
2. **Create Benchmark**: Isolate performance issue
3. **Optimize Code**: Implement improvements
4. **Validate Improvement**: Measure performance gain
5. **Regression Test**: Ensure no new issues

## Configuration

### Benchmark Configuration

```python
# conftest.py
@pytest.fixture
def benchmark_config():
    return {
        'min_rounds': 5,
        'max_time': 10.0,
        'warmup': True,
        'warmup_iterations': 3,
        'disable_gc': True,
        'timer': time.perf_counter
    }
```

### Environment Variables

```bash
# Performance testing configuration
export PWMK_BENCHMARK_MODE=true
export PWMK_PROFILING_ENABLED=true
export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # For memory testing
```

This performance testing framework ensures PWMK maintains optimal performance while enabling continuous optimization and regression detection.