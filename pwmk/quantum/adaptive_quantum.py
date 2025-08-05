"""
Adaptive quantum algorithms that learn and optimize quantum parameters.

Implements self-improving quantum algorithms that adapt their parameters
based on problem characteristics and performance feedback.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import time
from collections import deque
from enum import Enum

from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector


class AdaptationStrategy(Enum):
    """Different adaptation strategies for quantum parameters."""
    GRADIENT_BASED = "gradient"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "rl"
    BAYESIAN_OPTIMIZATION = "bayesian"


@dataclass
class QuantumParameters:
    """Quantum algorithm parameters that can be adapted."""
    gate_parameters: Dict[str, float]
    circuit_depth: int
    measurement_shots: int
    decoherence_rate: float
    entanglement_strength: float


@dataclass
class AdaptationResult:
    """Result from adaptive quantum optimization."""
    optimized_parameters: QuantumParameters
    performance_improvement: float
    adaptation_time: float
    learning_curve: List[float]
    parameter_history: List[QuantumParameters]


class AdaptiveQuantumAlgorithm(LoggingMixin):
    """
    Adaptive quantum algorithm that optimizes its own parameters.
    
    Uses machine learning techniques to automatically tune quantum
    algorithm parameters for optimal performance on specific problem classes.
    """
    
    def __init__(
        self,
        adaptation_strategy: AdaptationStrategy = AdaptationStrategy.GRADIENT_BASED,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1,
        memory_size: int = 1000,
        adaptation_frequency: int = 10
    ):
        super().__init__()
        
        self.adaptation_strategy = adaptation_strategy
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.memory_size = memory_size
        self.adaptation_frequency = adaptation_frequency
        
        # Parameter ranges
        self.parameter_bounds = {
            "gate_rotation": (-2*np.pi, 2*np.pi),
            "circuit_depth": (1, 20),
            "measurement_shots": (100, 10000),
            "decoherence_rate": (0.0, 0.1),
            "entanglement_strength": (0.0, 2.0)
        }
        
        # Learning components
        self._initialize_learning_components()
        
        # Performance history
        self.performance_history = deque(maxlen=memory_size)
        self.parameter_history = deque(maxlen=memory_size)
        
        # Adaptation statistics
        self.adaptation_stats = {
            "adaptations_performed": 0,
            "total_improvement": 0.0,
            "best_performance": float('-inf'),
            "parameter_convergence": []
        }
        
        self.logger.info(
            f"Initialized AdaptiveQuantumAlgorithm: strategy={adaptation_strategy.value}, "
            f"lr={learning_rate}, exploration={exploration_rate}"
        )
    
    def _initialize_learning_components(self) -> None:
        """Initialize learning components based on adaptation strategy."""
        
        if self.adaptation_strategy == AdaptationStrategy.GRADIENT_BASED:
            # Neural network for parameter optimization
            self.parameter_network = nn.Sequential(
                nn.Linear(10, 64),  # Input: problem features
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 8)   # Output: quantum parameters
            )
            self.optimizer = torch.optim.Adam(self.parameter_network.parameters(), lr=self.learning_rate)
            
        elif self.adaptation_strategy == AdaptationStrategy.EVOLUTIONARY:
            # Evolutionary algorithm parameters
            self.population_size = 20
            self.mutation_rate = 0.1
            self.crossover_rate = 0.7
            self.population = []
            
        elif self.adaptation_strategy == AdaptationStrategy.REINFORCEMENT_LEARNING:
            # Q-learning for parameter adaptation
            self.q_table = {}
            self.epsilon = self.exploration_rate
            self.gamma = 0.95  # Discount factor
            
        elif self.adaptation_strategy == AdaptationStrategy.BAYESIAN_OPTIMIZATION:
            # Gaussian process for Bayesian optimization
            self.gp_observations = []
            self.gp_parameters = []
            self.acquisition_function = "expected_improvement"
        
        self.logger.debug(f"Learning components initialized for {self.adaptation_strategy.value}")
    
    def adapt_parameters(
        self,
        current_parameters: QuantumParameters,
        problem_features: np.ndarray,
        performance_feedback: float,
        adaptation_context: Dict[str, Any]
    ) -> AdaptationResult:
        """
        Adapt quantum parameters based on performance feedback.
        
        Args:
            current_parameters: Current quantum algorithm parameters
            problem_features: Features describing the current problem
            performance_feedback: Performance metric (higher is better)
            adaptation_context: Additional context for adaptation
            
        Returns:
            AdaptationResult with optimized parameters
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting parameter adaptation: performance={performance_feedback:.6f}")
            
            # Store current performance and parameters
            self.performance_history.append(performance_feedback)
            self.parameter_history.append(current_parameters)
            
            # Choose adaptation method
            if self.adaptation_strategy == AdaptationStrategy.GRADIENT_BASED:
                result = self._gradient_based_adaptation(
                    current_parameters, problem_features, performance_feedback
                )
            elif self.adaptation_strategy == AdaptationStrategy.EVOLUTIONARY:
                result = self._evolutionary_adaptation(
                    current_parameters, problem_features, performance_feedback
                )
            elif self.adaptation_strategy == AdaptationStrategy.REINFORCEMENT_LEARNING:
                result = self._rl_based_adaptation(
                    current_parameters, problem_features, performance_feedback
                )
            elif self.adaptation_strategy == AdaptationStrategy.BAYESIAN_OPTIMIZATION:
                result = self._bayesian_optimization_adaptation(
                    current_parameters, problem_features, performance_feedback
                )
            else:
                raise ValueError(f"Unknown adaptation strategy: {self.adaptation_strategy}")
            
            adaptation_time = time.time() - start_time
            result.adaptation_time = adaptation_time
            
            # Update statistics
            improvement = result.performance_improvement
            self.adaptation_stats["adaptations_performed"] += 1
            self.adaptation_stats["total_improvement"] += improvement
            
            if performance_feedback > self.adaptation_stats["best_performance"]:
                self.adaptation_stats["best_performance"] = performance_feedback
            
            get_metrics_collector().record_quantum_operation("parameter_adaptation", adaptation_time)
            get_metrics_collector().record_metric("parameter_improvement", improvement)
            
            self.logger.info(
                f"Parameter adaptation complete: improvement={improvement:.4f}, "
                f"time={adaptation_time:.4f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Parameter adaptation failed: {e}")
            raise
    
    def _gradient_based_adaptation(
        self,
        current_parameters: QuantumParameters,
        problem_features: np.ndarray,
        performance_feedback: float
    ) -> AdaptationResult:
        """Gradient-based parameter optimization using neural networks."""
        
        # Convert problem features to tensor
        features_tensor = torch.FloatTensor(problem_features).unsqueeze(0)
        
        # Get current parameter predictions
        with torch.no_grad():
            current_pred = self.parameter_network(features_tensor).squeeze().numpy()
        
        # Create learning targets based on performance feedback
        if len(self.performance_history) > 1:
            performance_improvement = performance_feedback - self.performance_history[-2]
            
            # Adjust targets based on improvement
            if performance_improvement > 0:
                # Positive feedback: move slightly in same direction
                target_adjustment = 0.1 * performance_improvement
            else:
                # Negative feedback: move in opposite direction
                target_adjustment = -0.1 * abs(performance_improvement)
            
            # Create target parameters
            target_params = current_pred + target_adjustment * np.random.normal(0, 0.1, len(current_pred))
            target_tensor = torch.FloatTensor(target_params).unsqueeze(0)
            
            # Training step
            self.optimizer.zero_grad()
            predicted_params = self.parameter_network(features_tensor)
            loss = nn.MSELoss()(predicted_params, target_tensor)
            loss.backward()
            self.optimizer.step()
        
        # Generate optimized parameters
        with torch.no_grad():
            optimized_pred = self.parameter_network(features_tensor).squeeze().numpy()
        
        optimized_parameters = self._decode_parameters(optimized_pred)
        
        # Estimate improvement (heuristic)
        parameter_change = np.linalg.norm(optimized_pred - current_pred)
        estimated_improvement = min(parameter_change * 0.1, 0.5)  # Cap at 50% improvement
        
        return AdaptationResult(
            optimized_parameters=optimized_parameters,
            performance_improvement=estimated_improvement,
            adaptation_time=0.0,  # Will be set by caller
            learning_curve=[performance_feedback],
            parameter_history=[current_parameters, optimized_parameters]
        )
    
    def _evolutionary_adaptation(
        self,
        current_parameters: QuantumParameters,
        problem_features: np.ndarray,
        performance_feedback: float
    ) -> AdaptationResult:
        """Evolutionary algorithm for parameter optimization."""
        
        # Initialize population if empty
        if not self.population:
            self.population = [self._mutate_parameters(current_parameters) for _ in range(self.population_size)]
        
        # Evaluate population (simplified - use performance feedback)
        fitness_scores = []
        for individual in self.population:
            # Heuristic fitness based on parameter similarity to current best
            similarity = self._parameter_similarity(individual, current_parameters)
            fitness = performance_feedback * similarity + np.random.normal(0, 0.1)
            fitness_scores.append(fitness)
        
        # Selection, crossover, and mutation
        new_population = []
        
        # Keep best individuals (elitism)
        elite_count = max(1, self.population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(self.population, fitness_scores)
            parent2 = self._tournament_selection(self.population, fitness_scores)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self._crossover_parameters(parent1, parent2)
            else:
                child = parent1 if np.random.random() < 0.5 else parent2
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child = self._mutate_parameters(child)
            
            new_population.append(child)
        
        self.population = new_population
        
        # Return best individual
        best_idx = np.argmax(fitness_scores)
        best_parameters = self.population[best_idx]
        best_fitness = fitness_scores[best_idx]
        
        improvement = best_fitness - performance_feedback
        
        return AdaptationResult(
            optimized_parameters=best_parameters,
            performance_improvement=improvement,
            adaptation_time=0.0,
            learning_curve=fitness_scores,
            parameter_history=[current_parameters, best_parameters]
        )
    
    def _rl_based_adaptation(
        self,
        current_parameters: QuantumParameters,
        problem_features: np.ndarray,
        performance_feedback: float
    ) -> AdaptationResult:
        """Reinforcement learning based parameter adaptation."""
        
        # Discretize state space (simplified)
        state = self._discretize_state(problem_features, current_parameters)
        
        # Choose action (parameter adjustment)
        if np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, 8)  # 8 possible parameter adjustments
        else:
            # Exploitation: best known action
            if state in self.q_table:
                action = np.argmax(self.q_table[state])
            else:
                action = np.random.randint(0, 8)
        
        # Apply action to get new parameters
        new_parameters = self._apply_action(current_parameters, action)
        
        # Calculate reward (based on performance improvement)
        if len(self.performance_history) > 1:
            reward = performance_feedback - self.performance_history[-2]
        else:
            reward = performance_feedback
        
        # Update Q-table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(8)
        
        # Q-learning update
        next_state = self._discretize_state(problem_features, new_parameters)
        if next_state in self.q_table:
            max_next_q = np.max(self.q_table[next_state])
        else:
            max_next_q = 0.0
        
        self.q_table[state][action] += self.learning_rate * (
            reward + self.gamma * max_next_q - self.q_table[state][action]
        )
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return AdaptationResult(
            optimized_parameters=new_parameters,
            performance_improvement=reward,
            adaptation_time=0.0,
            learning_curve=[performance_feedback],
            parameter_history=[current_parameters, new_parameters]
        )
    
    def _bayesian_optimization_adaptation(
        self,
        current_parameters: QuantumParameters,
        problem_features: np.ndarray,
        performance_feedback: float
    ) -> AdaptationResult:
        """Bayesian optimization for parameter tuning."""
        
        # Add current observation
        param_vector = self._encode_parameters(current_parameters)
        self.gp_observations.append(performance_feedback)
        self.gp_parameters.append(param_vector)
        
        # Simple Bayesian optimization (without full GP implementation)
        if len(self.gp_observations) < 5:
            # Random exploration for initial points
            new_parameters = self._random_parameters()
        else:
            # Find parameters that maximize acquisition function
            new_parameters = self._optimize_acquisition_function()
        
        # Estimate improvement based on historical data
        if self.gp_observations:
            best_so_far = max(self.gp_observations)
            estimated_improvement = max(0, performance_feedback - best_so_far)  
        else:
            estimated_improvement = 0.1
        
        return AdaptationResult(
            optimized_parameters=new_parameters,
            performance_improvement=estimated_improvement,
            adaptation_time=0.0,
            learning_curve=list(self.gp_observations),
            parameter_history=[current_parameters, new_parameters]
        )
    
    def _decode_parameters(self, param_vector: np.ndarray) -> QuantumParameters:
        """Decode parameter vector into QuantumParameters object."""
        
        # Scale parameters to appropriate ranges
        scaled_params = []
        param_names = ["gate_rotation", "circuit_depth", "measurement_shots", "decoherence_rate", "entanglement_strength"]
        
        for i, name in enumerate(param_names):
            if i < len(param_vector):
                min_val, max_val = self.parameter_bounds[name]
                scaled_val = min_val + (max_val - min_val) * (np.tanh(param_vector[i]) + 1) / 2
                scaled_params.append(scaled_val)
            else:
                # Default values
                scaled_params.append(1.0)
        
        # Ensure we have enough parameters
        while len(scaled_params) < 8:
            scaled_params.append(1.0)
        
        return QuantumParameters(
            gate_parameters={"rotation": scaled_params[0], "phase": scaled_params[1], "amplitude": scaled_params[2]},
            circuit_depth=max(1, int(scaled_params[1])),
            measurement_shots=max(100, int(scaled_params[2] * 1000)),
            decoherence_rate=max(0.0, min(0.1, scaled_params[3])),
            entanglement_strength=max(0.0, min(2.0, scaled_params[4]))
        )
    
    def _encode_parameters(self, parameters: QuantumParameters) -> np.ndarray:
        """Encode QuantumParameters into vector form."""
        
        vector = [
            parameters.gate_parameters.get("rotation", 0.0),
            parameters.gate_parameters.get("phase", 0.0),
            parameters.gate_parameters.get("amplitude", 1.0),
            float(parameters.circuit_depth),
            float(parameters.measurement_shots) / 1000.0,  # Normalize
            parameters.decoherence_rate,
            parameters.entanglement_strength
        ]
        
        return np.array(vector)
    
    def _parameter_similarity(self, params1: QuantumParameters, params2: QuantumParameters) -> float:
        """Calculate similarity between two parameter sets."""
        
        vec1 = self._encode_parameters(params1)
        vec2 = self._encode_parameters(params2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms > 1e-10:
            similarity = dot_product / norms
        else:
            similarity = 1.0
        
        return (similarity + 1) / 2  # Scale to [0, 1]
    
    def _mutate_parameters(self, parameters: QuantumParameters) -> QuantumParameters:
        """Apply random mutations to parameters."""
        
        # Create copy
        new_gate_params = parameters.gate_parameters.copy()
        
        # Mutate gate parameters
        for key in new_gate_params:
            if np.random.random() < 0.3:  # 30% mutation chance
                noise = np.random.normal(0, 0.1)
                new_gate_params[key] += noise
        
        # Mutate other parameters
        new_depth = parameters.circuit_depth
        if np.random.random() < 0.2:
            new_depth = max(1, new_depth + np.random.randint(-2, 3))
        
        new_shots = parameters.measurement_shots
        if np.random.random() < 0.2:
            new_shots = max(100, int(new_shots * (1 + np.random.normal(0, 0.2))))
        
        new_decoherence = parameters.decoherence_rate
        if np.random.random() < 0.2:
            new_decoherence = max(0.0, min(0.1, new_decoherence + np.random.normal(0, 0.01)))
        
        new_entanglement = parameters.entanglement_strength
        if np.random.random() < 0.2:
            new_entanglement = max(0.0, min(2.0, new_entanglement + np.random.normal(0, 0.1)))
        
        return QuantumParameters(
            gate_parameters=new_gate_params,
            circuit_depth=new_depth,
            measurement_shots=new_shots,
            decoherence_rate=new_decoherence,
            entanglement_strength=new_entanglement
        )
    
    def _tournament_selection(
        self, 
        population: List[QuantumParameters], 
        fitness_scores: List[float]
    ) -> QuantumParameters:
        """Tournament selection for evolutionary algorithm."""
        
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return population[winner_idx]
    
    def _crossover_parameters(
        self, 
        parent1: QuantumParameters, 
        parent2: QuantumParameters
    ) -> QuantumParameters:
        """Crossover operation for parameter breeding."""
        
        # Blend crossover for gate parameters
        child_gate_params = {}
        for key in parent1.gate_parameters:
            if key in parent2.gate_parameters:
                alpha = np.random.random()
                child_gate_params[key] = (alpha * parent1.gate_parameters[key] + 
                                        (1 - alpha) * parent2.gate_parameters[key])
            else:
                child_gate_params[key] = parent1.gate_parameters[key]
        
        # Random selection for discrete parameters
        child_depth = parent1.circuit_depth if np.random.random() < 0.5 else parent2.circuit_depth
        child_shots = parent1.measurement_shots if np.random.random() < 0.5 else parent2.measurement_shots
        
        # Blend for continuous parameters
        alpha = np.random.random()
        child_decoherence = alpha * parent1.decoherence_rate + (1 - alpha) * parent2.decoherence_rate
        child_entanglement = alpha * parent1.entanglement_strength + (1 - alpha) * parent2.entanglement_strength
        
        return QuantumParameters(
            gate_parameters=child_gate_params,
            circuit_depth=child_depth,
            measurement_shots=child_shots,
            decoherence_rate=child_decoherence,
            entanglement_strength=child_entanglement
        )
    
    def _discretize_state(self, features: np.ndarray, parameters: QuantumParameters) -> str:
        """Discretize continuous state for Q-learning."""
        
        # Simple binning approach
        feature_bins = [int(f * 10) for f in features[:5]]  # Take first 5 features
        param_vector = self._encode_parameters(parameters)
        param_bins = [int(p * 10) for p in param_vector[:3]]  # Take first 3 parameters
        
        state_tuple = tuple(feature_bins + param_bins)
        return str(state_tuple)
    
    def _apply_action(self, parameters: QuantumParameters, action: int) -> QuantumParameters:
        """Apply RL action to modify parameters."""
        
        new_params = parameters
        
        # Define 8 possible actions
        if action == 0:  # Increase rotation
            new_gate_params = parameters.gate_parameters.copy()
            new_gate_params["rotation"] = new_gate_params.get("rotation", 0.0) + 0.1
            new_params = QuantumParameters(
                gate_parameters=new_gate_params,
                circuit_depth=parameters.circuit_depth,
                measurement_shots=parameters.measurement_shots,
                decoherence_rate=parameters.decoherence_rate,
                entanglement_strength=parameters.entanglement_strength
            )
        elif action == 1:  # Decrease rotation
            new_gate_params = parameters.gate_parameters.copy()
            new_gate_params["rotation"] = new_gate_params.get("rotation", 0.0) - 0.1
            new_params = QuantumParameters(
                gate_parameters=new_gate_params,
                circuit_depth=parameters.circuit_depth,
                measurement_shots=parameters.measurement_shots,
                decoherence_rate=parameters.decoherence_rate,
                entanglement_strength=parameters.entanglement_strength
            )
        elif action == 2:  # Increase circuit depth
            new_params = QuantumParameters(
                gate_parameters=parameters.gate_parameters,
                circuit_depth=min(20, parameters.circuit_depth + 1),
                measurement_shots=parameters.measurement_shots,
                decoherence_rate=parameters.decoherence_rate,
                entanglement_strength=parameters.entanglement_strength
            )
        elif action == 3:  # Decrease circuit depth
            new_params = QuantumParameters(
                gate_parameters=parameters.gate_parameters,
                circuit_depth=max(1, parameters.circuit_depth - 1),
                measurement_shots=parameters.measurement_shots,
                decoherence_rate=parameters.decoherence_rate,
                entanglement_strength=parameters.entanglement_strength
            )
        # ... implement other actions similarly
        
        return new_params
    
    def _random_parameters(self) -> QuantumParameters:
        """Generate random parameters within valid bounds."""
        
        gate_params = {
            "rotation": np.random.uniform(-np.pi, np.pi),
            "phase": np.random.uniform(0, 2*np.pi),
            "amplitude": np.random.uniform(0.5, 1.5)
        }
        
        return QuantumParameters(
            gate_parameters=gate_params,
            circuit_depth=np.random.randint(1, 11),
            measurement_shots=np.random.randint(100, 2000),
            decoherence_rate=np.random.uniform(0.0, 0.05),
            entanglement_strength=np.random.uniform(0.5, 2.0)
        )
    
    def _optimize_acquisition_function(self) -> QuantumParameters:
        """Optimize acquisition function for Bayesian optimization."""
        
        # Simple implementation: random search with bias toward good regions
        best_candidates = []
        best_scores = []
        
        for _ in range(50):  # Try 50 random candidates
            candidate = self._random_parameters()
            score = self._evaluate_acquisition(candidate)
            best_candidates.append(candidate)
            best_scores.append(score)
        
        # Return best candidate
        best_idx = np.argmax(best_scores)
        return best_candidates[best_idx]
    
    def _evaluate_acquisition(self, parameters: QuantumParameters) -> float:
        """Evaluate acquisition function for a parameter set."""
        
        # Expected improvement (simplified)
        param_vector = self._encode_parameters(parameters)
        
        if not self.gp_parameters:
            return 1.0  # High acquisition for first evaluations
        
        # Find most similar previous evaluation
        similarities = []
        for prev_params in self.gp_parameters:
            similarity = np.exp(-np.linalg.norm(param_vector - prev_params))
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        max_idx = similarities.index(max_similarity)
        predicted_performance = self.gp_observations[max_idx]
        
        # Simple acquisition: predicted performance + exploration bonus
        current_best = max(self.gp_observations) if self.gp_observations else 0.0
        improvement = max(0, predicted_performance - current_best)
        exploration_bonus = 0.1 * (1.0 - max_similarity)  # Explore dissimilar regions
        
        return improvement + exploration_bonus
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        stats = self.adaptation_stats.copy()
        
        if stats["adaptations_performed"] > 0:
            stats["avg_improvement"] = stats["total_improvement"] / stats["adaptations_performed"]
        
        if self.performance_history:
            stats["performance_trend"] = list(self.performance_history)[-10:]  # Last 10 performances
            stats["current_performance"] = self.performance_history[-1]
        
        if hasattr(self, 'parameter_network'):
            stats["network_parameters"] = sum(p.numel() for p in self.parameter_network.parameters())
        
        return stats