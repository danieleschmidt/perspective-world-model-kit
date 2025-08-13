"""
Self-Improving AI Agent with Meta-Learning Capabilities

Revolutionary advancement: AI agents that autonomously enhance their own capabilities
through meta-learning, self-reflection, and adaptive architecture modification.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque

from ..core.world_model import PerspectiveWorldModel
from ..core.beliefs import BeliefStore
from ..quantum.adaptive_quantum import AdaptiveQuantumAlgorithm


@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking for self-improvement."""
    accuracy: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    memory_usage: float = 0.0
    energy_efficiency: float = 0.0
    belief_reasoning_speed: float = 0.0
    planning_success_rate: float = 0.0
    adaptation_rate: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate weighted overall performance score."""
        weights = {
            'accuracy': 0.25,
            'throughput': 0.20,
            'latency': -0.15,  # Lower is better
            'memory_usage': -0.10,  # Lower is better
            'energy_efficiency': 0.15,
            'belief_reasoning_speed': 0.20,
            'planning_success_rate': 0.25,
            'adaptation_rate': 0.15
        }
        
        score = sum(getattr(self, metric) * weight 
                   for metric, weight in weights.items())
        return max(0.0, min(1.0, score))


class MetaLearner(nn.Module):
    """Meta-learning network that learns how to improve learning."""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Meta-learning network
        self.meta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # Performance prediction network
        self.performance_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)  # 8 performance metrics
        )
        
        # Architecture modification network
        self.arch_modifier = nn.Sequential(
            nn.Linear(input_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32)  # Architecture modification parameters
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through meta-learning network."""
        improved_state = self.meta_net(state)
        predicted_performance = self.performance_predictor(state)
        
        combined = torch.cat([state, predicted_performance], dim=-1)
        arch_modifications = self.arch_modifier(combined)
        
        return improved_state, predicted_performance, arch_modifications


class SelfReflection:
    """Self-reflection module for analyzing agent performance and behavior."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.decision_history = deque(maxlen=window_size)
        self.improvement_opportunities = []
        
    def record_performance(self, metrics: PerformanceMetrics, decision_context: Dict[str, Any]):
        """Record performance metrics and decision context."""
        self.performance_history.append(metrics)
        self.decision_history.append(decision_context)
        
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends and identify improvement opportunities."""
        if len(self.performance_history) < 10:
            return {"status": "insufficient_data"}
            
        recent_scores = [m.overall_score() for m in list(self.performance_history)[-20:]]
        older_scores = [m.overall_score() for m in list(self.performance_history)[-50:-20]]
        
        if not older_scores:
            return {"status": "building_baseline"}
            
        trend = np.mean(recent_scores) - np.mean(older_scores)
        volatility = np.std(recent_scores)
        
        # Identify specific weakness patterns
        weaknesses = self._identify_weaknesses()
        strengths = self._identify_strengths()
        
        return {
            "trend": trend,
            "volatility": volatility,
            "improving": trend > 0.01,
            "stable": volatility < 0.05,
            "weaknesses": weaknesses,
            "strengths": strengths,
            "improvement_potential": self._calculate_improvement_potential()
        }
    
    def _identify_weaknesses(self) -> List[str]:
        """Identify consistently weak performance areas."""
        if not self.performance_history:
            return []
            
        metrics = {}
        for perf in self.performance_history:
            for field, value in asdict(perf).items():
                if field not in metrics:
                    metrics[field] = []
                metrics[field].append(value)
        
        weaknesses = []
        for metric, values in metrics.items():
            if metric in ['latency', 'memory_usage']:  # Lower is better
                if np.mean(values) > 0.7:
                    weaknesses.append(metric)
            else:  # Higher is better
                if np.mean(values) < 0.3:
                    weaknesses.append(metric)
                    
        return weaknesses
    
    def _identify_strengths(self) -> List[str]:
        """Identify consistently strong performance areas."""
        if not self.performance_history:
            return []
            
        metrics = {}
        for perf in self.performance_history:
            for field, value in asdict(perf).items():
                if field not in metrics:
                    metrics[field] = []
                metrics[field].append(value)
        
        strengths = []
        for metric, values in metrics.items():
            if metric in ['latency', 'memory_usage']:  # Lower is better
                if np.mean(values) < 0.3:
                    strengths.append(metric)
            else:  # Higher is better
                if np.mean(values) > 0.7:
                    strengths.append(metric)
                    
        return strengths
    
    def _calculate_improvement_potential(self) -> float:
        """Calculate overall improvement potential based on current performance."""
        if not self.performance_history:
            return 1.0
            
        recent_performance = self.performance_history[-1].overall_score()
        return max(0.0, 1.0 - recent_performance)


class ArchitectureEvolution:
    """Dynamic architecture modification system."""
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.architecture_history = []
        self.modification_count = 0
        self.successful_modifications = 0
        
    def evolve_architecture(self, modification_params: torch.Tensor, 
                          current_performance: PerformanceMetrics) -> bool:
        """Evolve the architecture based on modification parameters."""
        try:
            modifications = self._interpret_modifications(modification_params)
            
            # Create architecture checkpoint
            checkpoint = self._create_checkpoint()
            
            # Apply modifications
            success = self._apply_modifications(modifications)
            
            if success:
                self.modification_count += 1
                logging.info(f"Architecture modification {self.modification_count} applied successfully")
                return True
            else:
                # Rollback on failure
                self._restore_checkpoint(checkpoint)
                logging.warning("Architecture modification failed, rolled back")
                return False
                
        except Exception as e:
            logging.error(f"Architecture evolution failed: {e}")
            return False
    
    def _interpret_modifications(self, params: torch.Tensor) -> Dict[str, Any]:
        """Interpret modification parameters into concrete changes."""
        params = params.detach().cpu().numpy()
        
        modifications = {
            'add_layers': params[0] > 0.7,
            'modify_dimensions': params[1:5],
            'adjust_dropout': max(0.0, min(0.5, params[5])),
            'change_activation': int(params[6] * 3),  # 0: ReLU, 1: GELU, 2: Swish
            'add_skip_connections': params[7] > 0.6,
            'modify_attention': params[8:16],
            'prune_connections': params[16] > 0.8,
            'add_normalization': params[17] > 0.5,
        }
        
        return modifications
    
    def _create_checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of current architecture state."""
        return {
            'state_dict': self.base_model.state_dict().copy(),
            'architecture_info': self._get_architecture_info(),
            'timestamp': time.time()
        }
    
    def _apply_modifications(self, modifications: Dict[str, Any]) -> bool:
        """Apply architecture modifications to the model."""
        try:
            # This is a simplified example - real implementation would
            # involve complex architecture modification logic
            
            if modifications.get('adjust_dropout'):
                self._modify_dropout(modifications['adjust_dropout'])
                
            if modifications.get('add_normalization'):
                self._add_batch_normalization()
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to apply modifications: {e}")
            return False
    
    def _modify_dropout(self, dropout_rate: float):
        """Modify dropout rates in the model."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
    
    def _add_batch_normalization(self):
        """Add batch normalization where applicable."""
        # Simplified implementation
        pass
    
    def _restore_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore model from checkpoint."""
        self.base_model.load_state_dict(checkpoint['state_dict'])
    
    def _get_architecture_info(self) -> Dict[str, Any]:
        """Get current architecture information."""
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layer_count': len(list(self.base_model.modules())),
            'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
        }


class SelfImprovingAgent:
    """
    Revolutionary self-improving AI agent that autonomously enhances its capabilities.
    
    This agent combines:
    - Meta-learning for learning how to learn better
    - Self-reflection for performance analysis
    - Dynamic architecture evolution
    - Quantum-inspired optimization
    - Autonomous capability enhancement
    """
    
    def __init__(self, 
                 world_model: PerspectiveWorldModel,
                 belief_store: BeliefStore,
                 meta_learning_rate: float = 1e-4,
                 improvement_threshold: float = 0.05,
                 save_dir: str = "self_improvement_logs"):
        
        self.world_model = world_model
        self.belief_store = belief_store
        self.meta_learning_rate = meta_learning_rate
        self.improvement_threshold = improvement_threshold
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize meta-learning components
        self.meta_learner = MetaLearner()
        self.meta_optimizer = torch.optim.Adam(
            self.meta_learner.parameters(), 
            lr=meta_learning_rate
        )
        
        # Initialize self-reflection and evolution components
        self.reflection = SelfReflection()
        self.architecture_evolution = ArchitectureEvolution(world_model)
        
        # Performance tracking
        self.current_metrics = PerformanceMetrics()
        self.baseline_metrics = None
        self.improvement_history = []
        
        # Quantum optimization
        self.quantum_optimizer = AdaptiveQuantumAlgorithm()
        
        # Self-improvement state
        self.improvement_cycle = 0
        self.total_improvements = 0
        self.last_improvement_time = time.time()
        
        logging.info("Self-improving agent initialized successfully")
    
    def autonomous_improvement_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete autonomous improvement cycle.
        
        Returns:
            Dict containing improvement results and metrics
        """
        cycle_start = time.time()
        self.improvement_cycle += 1
        
        logging.info(f"Starting improvement cycle {self.improvement_cycle}")
        
        # Phase 1: Performance Assessment
        current_performance = self._assess_current_performance()
        
        # Phase 2: Self-Reflection Analysis
        reflection_analysis = self.reflection.analyze_trends()
        
        # Phase 3: Identify Improvement Opportunities
        opportunities = self._identify_improvement_opportunities(
            current_performance, reflection_analysis
        )
        
        # Phase 4: Generate Improvement Strategies
        strategies = self._generate_improvement_strategies(opportunities)
        
        # Phase 5: Execute Improvements
        improvement_results = self._execute_improvements(strategies)
        
        # Phase 6: Validate Improvements
        validation_results = self._validate_improvements(current_performance)
        
        # Phase 7: Learn from Results
        self._update_meta_learning(improvement_results, validation_results)
        
        cycle_time = time.time() - cycle_start
        
        cycle_summary = {
            'cycle_id': self.improvement_cycle,
            'duration': cycle_time,
            'performance_before': asdict(current_performance),
            'performance_after': asdict(self.current_metrics),
            'reflection_analysis': reflection_analysis,
            'opportunities': opportunities,
            'strategies': strategies,
            'improvement_results': improvement_results,
            'validation_results': validation_results,
            'net_improvement': validation_results.get('net_improvement', 0.0)
        }
        
        self._save_cycle_results(cycle_summary)
        
        if cycle_summary['net_improvement'] > self.improvement_threshold:
            self.total_improvements += 1
            self.last_improvement_time = time.time()
            logging.info(f"Successful improvement achieved: {cycle_summary['net_improvement']:.4f}")
        
        return cycle_summary
    
    def _assess_current_performance(self) -> PerformanceMetrics:
        """Assess current agent performance across all metrics."""
        start_time = time.time()
        
        # Simulate performance assessment (in real implementation, this would
        # run actual benchmarks and evaluations)
        test_scenarios = self._generate_test_scenarios()
        performance_scores = []
        
        for scenario in test_scenarios:
            score = self._run_performance_test(scenario)
            performance_scores.append(score)
        
        # Calculate comprehensive metrics
        metrics = PerformanceMetrics(
            accuracy=np.mean([s.get('accuracy', 0.0) for s in performance_scores]),
            throughput=np.mean([s.get('throughput', 0.0) for s in performance_scores]),
            latency=time.time() - start_time,
            memory_usage=self._measure_memory_usage(),
            energy_efficiency=self._measure_energy_efficiency(),
            belief_reasoning_speed=self._measure_belief_reasoning_speed(),
            planning_success_rate=np.mean([s.get('planning_success', 0.0) for s in performance_scores]),
            adaptation_rate=self._measure_adaptation_rate()
        )
        
        self.current_metrics = metrics
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
        
        return metrics
    
    def _generate_test_scenarios(self) -> List[Dict[str, Any]]:
        """Generate test scenarios for performance assessment."""
        scenarios = [
            {'type': 'belief_reasoning', 'complexity': 'low'},
            {'type': 'belief_reasoning', 'complexity': 'high'},
            {'type': 'planning', 'horizon': 5},
            {'type': 'planning', 'horizon': 15},
            {'type': 'multi_agent', 'agents': 3},
            {'type': 'multi_agent', 'agents': 10},
            {'type': 'theory_of_mind', 'depth': 2},
            {'type': 'theory_of_mind', 'depth': 4},
        ]
        return scenarios
    
    def _run_performance_test(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        """Run a performance test for a specific scenario."""
        # Simplified simulation - real implementation would run actual tests
        base_score = 0.6 + np.random.normal(0, 0.1)
        
        # Add scenario-specific variations
        if scenario['type'] == 'belief_reasoning':
            complexity_factor = 0.8 if scenario['complexity'] == 'high' else 1.0
            base_score *= complexity_factor
        elif scenario['type'] == 'planning':
            horizon_factor = max(0.5, 1.0 - scenario['horizon'] * 0.03)
            base_score *= horizon_factor
        elif scenario['type'] == 'multi_agent':
            agent_factor = max(0.4, 1.0 - scenario['agents'] * 0.05)
            base_score *= agent_factor
        elif scenario['type'] == 'theory_of_mind':
            depth_factor = max(0.3, 1.0 - scenario['depth'] * 0.1)
            base_score *= depth_factor
        
        return {
            'accuracy': max(0.0, min(1.0, base_score)),
            'throughput': max(0.0, base_score * 1000),
            'planning_success': max(0.0, min(1.0, base_score * 1.1)),
        }
    
    def _measure_memory_usage(self) -> float:
        """Measure current memory usage."""
        # Simplified measurement
        return 0.3 + np.random.uniform(-0.1, 0.1)
    
    def _measure_energy_efficiency(self) -> float:
        """Measure energy efficiency."""
        return 0.7 + np.random.uniform(-0.2, 0.2)
    
    def _measure_belief_reasoning_speed(self) -> float:
        """Measure belief reasoning speed."""
        return 0.8 + np.random.uniform(-0.15, 0.15)
    
    def _measure_adaptation_rate(self) -> float:
        """Measure adaptation rate to new scenarios."""
        return 0.6 + np.random.uniform(-0.2, 0.2)
    
    def _identify_improvement_opportunities(self, 
                                         performance: PerformanceMetrics,
                                         reflection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities."""
        opportunities = []
        
        # Identify weaknesses from reflection
        for weakness in reflection.get('weaknesses', []):
            opportunities.append({
                'type': 'weakness_improvement',
                'target': weakness,
                'current_value': getattr(performance, weakness, 0.0),
                'priority': 'high'
            })
        
        # Identify trend-based opportunities
        if reflection.get('trend', 0) < -0.01:  # Declining performance
            opportunities.append({
                'type': 'performance_recovery',
                'target': 'overall_performance',
                'priority': 'critical'
            })
        
        # Identify architecture optimization opportunities
        if performance.overall_score() < 0.7:
            opportunities.append({
                'type': 'architecture_optimization',
                'target': 'model_architecture',
                'priority': 'medium'
            })
        
        # Identify quantum optimization opportunities
        if performance.throughput < 0.6:
            opportunities.append({
                'type': 'quantum_optimization',
                'target': 'computational_efficiency',
                'priority': 'high'
            })
        
        return sorted(opportunities, 
                     key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']])
    
    def _generate_improvement_strategies(self, 
                                       opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific improvement strategies."""
        strategies = []
        
        for opp in opportunities:
            if opp['type'] == 'weakness_improvement':
                strategies.append({
                    'type': 'targeted_training',
                    'target': opp['target'],
                    'method': 'focused_meta_learning',
                    'expected_improvement': 0.1
                })
            
            elif opp['type'] == 'performance_recovery':
                strategies.append({
                    'type': 'architecture_rollback_and_enhance',
                    'method': 'selective_parameter_reset',
                    'expected_improvement': 0.15
                })
            
            elif opp['type'] == 'architecture_optimization':
                strategies.append({
                    'type': 'dynamic_architecture_evolution',
                    'method': 'neural_architecture_search',
                    'expected_improvement': 0.2
                })
            
            elif opp['type'] == 'quantum_optimization':
                strategies.append({
                    'type': 'quantum_enhanced_processing',
                    'method': 'adaptive_quantum_algorithms',
                    'expected_improvement': 0.25
                })
        
        return strategies
    
    def _execute_improvements(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute improvement strategies."""
        results = {
            'executed_strategies': [],
            'failed_strategies': [],
            'total_expected_improvement': 0.0,
            'actual_improvements': {}
        }
        
        for strategy in strategies:
            try:
                success = self._execute_single_strategy(strategy)
                if success:
                    results['executed_strategies'].append(strategy)
                    results['total_expected_improvement'] += strategy.get('expected_improvement', 0.0)
                else:
                    results['failed_strategies'].append(strategy)
                    
            except Exception as e:
                logging.error(f"Strategy execution failed: {e}")
                results['failed_strategies'].append(strategy)
        
        return results
    
    def _execute_single_strategy(self, strategy: Dict[str, Any]) -> bool:
        """Execute a single improvement strategy."""
        try:
            if strategy['type'] == 'targeted_training':
                return self._execute_targeted_training(strategy)
            elif strategy['type'] == 'architecture_rollback_and_enhance':
                return self._execute_architecture_enhancement(strategy)
            elif strategy['type'] == 'dynamic_architecture_evolution':
                return self._execute_architecture_evolution(strategy)
            elif strategy['type'] == 'quantum_enhanced_processing':
                return self._execute_quantum_optimization(strategy)
            else:
                logging.warning(f"Unknown strategy type: {strategy['type']}")
                return False
                
        except Exception as e:
            logging.error(f"Single strategy execution failed: {e}")
            return False
    
    def _execute_targeted_training(self, strategy: Dict[str, Any]) -> bool:
        """Execute targeted training for specific weaknesses."""
        target = strategy.get('target')
        
        # Generate synthetic training data focused on weakness
        training_data = self._generate_targeted_training_data(target)
        
        # Perform focused meta-learning
        for batch in training_data:
            loss = self._compute_meta_learning_loss(batch, target)
            
            self.meta_optimizer.zero_grad()
            loss.backward()
            self.meta_optimizer.step()
        
        logging.info(f"Completed targeted training for {target}")
        return True
    
    def _execute_architecture_enhancement(self, strategy: Dict[str, Any]) -> bool:
        """Execute architecture enhancement strategy."""
        # Create enhanced architecture modifications
        enhancement_params = torch.randn(32) * 0.5  # Moderate modifications
        
        success = self.architecture_evolution.evolve_architecture(
            enhancement_params, 
            self.current_metrics
        )
        
        if success:
            logging.info("Architecture enhancement executed successfully")
        
        return success
    
    def _execute_architecture_evolution(self, strategy: Dict[str, Any]) -> bool:
        """Execute dynamic architecture evolution."""
        # Use meta-learner to generate optimal architecture modifications
        current_state = self._get_current_state_tensor()
        
        with torch.no_grad():
            _, _, arch_modifications = self.meta_learner(current_state)
        
        success = self.architecture_evolution.evolve_architecture(
            arch_modifications.squeeze(),
            self.current_metrics
        )
        
        return success
    
    def _execute_quantum_optimization(self, strategy: Dict[str, Any]) -> bool:
        """Execute quantum-inspired optimization."""
        # Apply quantum optimization to computational processes
        optimization_result = self.quantum_optimizer.optimize(
            problem_type='belief_reasoning',
            parameters={'complexity': 'adaptive'}
        )
        
        if optimization_result.get('success', False):
            logging.info("Quantum optimization executed successfully")
            return True
        
        return False
    
    def _generate_targeted_training_data(self, target: str) -> List[torch.Tensor]:
        """Generate training data focused on specific target weakness."""
        # Simplified data generation - real implementation would create
        # sophisticated targeted training scenarios
        data_batches = []
        
        for _ in range(10):  # 10 training batches
            batch = torch.randn(32, 256)  # Batch size 32, feature dim 256
            data_batches.append(batch)
        
        return data_batches
    
    def _compute_meta_learning_loss(self, batch: torch.Tensor, target: str) -> torch.Tensor:
        """Compute meta-learning loss for targeted improvement."""
        # Forward pass through meta-learner
        improved_state, predicted_perf, _ = self.meta_learner(batch)
        
        # Create target performance (improvement over current)
        current_value = getattr(self.current_metrics, target, 0.0)
        target_improvement = min(1.0, current_value + 0.1)  # 10% improvement
        target_tensor = torch.full((batch.size(0),), target_improvement)
        
        # Compute loss (simplified - real implementation would be more sophisticated)
        target_idx = self._get_target_index(target)
        predicted_target = predicted_perf[:, target_idx]
        
        loss = nn.MSELoss()(predicted_target, target_tensor)
        
        return loss
    
    def _get_target_index(self, target: str) -> int:
        """Get index of target metric in prediction tensor."""
        metrics = ['accuracy', 'throughput', 'latency', 'memory_usage', 
                  'energy_efficiency', 'belief_reasoning_speed', 
                  'planning_success_rate', 'adaptation_rate']
        return metrics.index(target) if target in metrics else 0
    
    def _get_current_state_tensor(self) -> torch.Tensor:
        """Get current state as tensor for meta-learning."""
        metrics_values = [
            self.current_metrics.accuracy,
            self.current_metrics.throughput,
            self.current_metrics.latency,
            self.current_metrics.memory_usage,
            self.current_metrics.energy_efficiency,
            self.current_metrics.belief_reasoning_speed,
            self.current_metrics.planning_success_rate,
            self.current_metrics.adaptation_rate
        ]
        
        # Pad to 256 dimensions (simplified)
        state = metrics_values + [0.0] * (256 - len(metrics_values))
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    def _validate_improvements(self, baseline_performance: PerformanceMetrics) -> Dict[str, Any]:
        """Validate that improvements actually improved performance."""
        # Re-assess performance after improvements
        new_performance = self._assess_current_performance()
        
        # Calculate improvements for each metric
        improvements = {}
        for field in asdict(new_performance).keys():
            old_value = getattr(baseline_performance, field)
            new_value = getattr(new_performance, field)
            
            # For latency and memory_usage, lower is better
            if field in ['latency', 'memory_usage']:
                improvement = old_value - new_value  # Positive means improvement
            else:
                improvement = new_value - old_value  # Positive means improvement
                
            improvements[field] = improvement
        
        # Calculate overall improvement
        old_score = baseline_performance.overall_score()
        new_score = new_performance.overall_score()
        net_improvement = new_score - old_score
        
        validation_results = {
            'baseline_score': old_score,
            'new_score': new_score,
            'net_improvement': net_improvement,
            'individual_improvements': improvements,
            'significant_improvement': net_improvement > self.improvement_threshold,
            'regression_detected': net_improvement < -0.02
        }
        
        # Record for reflection
        self.reflection.record_performance(new_performance, {
            'improvement_cycle': self.improvement_cycle,
            'strategies_applied': len(self.improvement_history),
            'net_improvement': net_improvement
        })
        
        return validation_results
    
    def _update_meta_learning(self, improvement_results: Dict[str, Any], 
                             validation_results: Dict[str, Any]):
        """Update meta-learning based on improvement results."""
        # Create training signal for meta-learner
        success_signal = 1.0 if validation_results['significant_improvement'] else 0.0
        
        # Update meta-learner with success/failure feedback
        current_state = self._get_current_state_tensor()
        target_success = torch.tensor([success_signal], dtype=torch.float32)
        
        # Simple success prediction loss
        _, predicted_perf, _ = self.meta_learner(current_state)
        success_loss = nn.BCEWithLogitsLoss()(
            torch.mean(predicted_perf), 
            target_success
        )
        
        self.meta_optimizer.zero_grad()
        success_loss.backward()
        self.meta_optimizer.step()
        
        # Update improvement history
        self.improvement_history.append({
            'cycle': self.improvement_cycle,
            'improvement_results': improvement_results,
            'validation_results': validation_results,
            'timestamp': time.time()
        })
    
    def _save_cycle_results(self, cycle_summary: Dict[str, Any]):
        """Save cycle results for analysis and debugging."""
        timestamp = int(time.time())
        filename = self.save_dir / f"improvement_cycle_{self.improvement_cycle}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(cycle_summary, f, indent=2, default=str)
            logging.info(f"Cycle results saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save cycle results: {e}")
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get comprehensive improvement summary."""
        if not self.baseline_metrics:
            return {"status": "no_baseline"}
        
        current_score = self.current_metrics.overall_score()
        baseline_score = self.baseline_metrics.overall_score()
        total_improvement = current_score - baseline_score
        
        return {
            'total_cycles': self.improvement_cycle,
            'successful_improvements': self.total_improvements,
            'success_rate': self.total_improvements / max(1, self.improvement_cycle),
            'baseline_score': baseline_score,
            'current_score': current_score,
            'total_improvement': total_improvement,
            'improvement_percentage': (total_improvement / max(0.01, baseline_score)) * 100,
            'last_improvement_time': self.last_improvement_time,
            'average_cycle_time': self._calculate_average_cycle_time(),
            'improvement_trajectory': self._get_improvement_trajectory()
        }
    
    def _calculate_average_cycle_time(self) -> float:
        """Calculate average cycle execution time."""
        if len(self.improvement_history) < 2:
            return 0.0
        
        times = []
        for i in range(1, len(self.improvement_history)):
            time_diff = (self.improvement_history[i]['timestamp'] - 
                        self.improvement_history[i-1]['timestamp'])
            times.append(time_diff)
        
        return np.mean(times) if times else 0.0
    
    def _get_improvement_trajectory(self) -> List[float]:
        """Get improvement trajectory over time."""
        trajectory = []
        
        if self.baseline_metrics:
            trajectory.append(self.baseline_metrics.overall_score())
        
        for entry in self.improvement_history:
            validation = entry.get('validation_results', {})
            if 'new_score' in validation:
                trajectory.append(validation['new_score'])
        
        return trajectory
    
    def continuous_self_improvement(self, 
                                  max_cycles: int = 100,
                                  improvement_goal: float = 0.9,
                                  max_time_hours: float = 24.0) -> Dict[str, Any]:
        """
        Run continuous self-improvement until goal is reached or limits exceeded.
        
        Args:
            max_cycles: Maximum number of improvement cycles
            improvement_goal: Target overall performance score
            max_time_hours: Maximum time to run (hours)
        
        Returns:
            Final improvement summary
        """
        start_time = time.time()
        max_time_seconds = max_time_hours * 3600
        
        logging.info(f"Starting continuous self-improvement")
        logging.info(f"Goal: {improvement_goal}, Max cycles: {max_cycles}, Max time: {max_time_hours}h")
        
        while (self.improvement_cycle < max_cycles and
               self.current_metrics.overall_score() < improvement_goal and
               (time.time() - start_time) < max_time_seconds):
            
            try:
                cycle_result = self.autonomous_improvement_cycle()
                
                current_score = self.current_metrics.overall_score()
                elapsed_hours = (time.time() - start_time) / 3600
                
                logging.info(f"Cycle {self.improvement_cycle}: Score {current_score:.4f}, "
                           f"Time: {elapsed_hours:.2f}h")
                
                # Early stopping if no improvement for many cycles
                if (self.improvement_cycle > 20 and 
                    time.time() - self.last_improvement_time > 7200):  # 2 hours
                    logging.info("Early stopping: No improvement for 2 hours")
                    break
                    
            except KeyboardInterrupt:
                logging.info("Self-improvement interrupted by user")
                break
            except Exception as e:
                logging.error(f"Error in improvement cycle: {e}")
                continue
        
        final_summary = self.get_improvement_summary()
        final_summary['termination_reason'] = self._get_termination_reason(
            max_cycles, improvement_goal, max_time_seconds, start_time
        )
        
        logging.info("Continuous self-improvement completed")
        logging.info(f"Final score: {self.current_metrics.overall_score():.4f}")
        logging.info(f"Total improvement: {final_summary['total_improvement']:.4f}")
        
        return final_summary
    
    def _get_termination_reason(self, max_cycles: int, improvement_goal: float,
                               max_time_seconds: float, start_time: float) -> str:
        """Determine why continuous improvement terminated."""
        if self.current_metrics.overall_score() >= improvement_goal:
            return "goal_achieved"
        elif self.improvement_cycle >= max_cycles:
            return "max_cycles_reached"
        elif (time.time() - start_time) >= max_time_seconds:
            return "time_limit_reached"
        elif time.time() - self.last_improvement_time > 7200:
            return "no_improvement_detected"
        else:
            return "unknown"


def create_self_improving_agent(world_model: PerspectiveWorldModel, 
                               belief_store: BeliefStore,
                               **kwargs) -> SelfImprovingAgent:
    """Factory function to create a self-improving agent."""
    return SelfImprovingAgent(world_model, belief_store, **kwargs)


# Example usage and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # This would normally be created from your existing models
    # For demonstration, we'll create minimal stubs
    class MockWorldModel:
        def __init__(self):
            self.parameters = lambda: []
    
    class MockBeliefStore:
        def __init__(self):
            pass
    
    # Create self-improving agent
    world_model = MockWorldModel()
    belief_store = MockBeliefStore()
    
    agent = SelfImprovingAgent(world_model, belief_store)
    
    # Run a single improvement cycle
    print("Running single improvement cycle...")
    result = agent.autonomous_improvement_cycle()
    print(f"Cycle result: {result['net_improvement']:.4f} improvement")
    
    # Get improvement summary
    summary = agent.get_improvement_summary()
    print(f"Improvement summary: {summary}")