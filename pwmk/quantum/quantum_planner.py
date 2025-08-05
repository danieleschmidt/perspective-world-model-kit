"""
Quantum-inspired task planner implementing superposition and interference principles.

Uses quantum computing concepts adapted for classical multi-agent planning problems,
enabling exploration of multiple solution paths simultaneously.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import time
from collections import defaultdict

from ..utils.logging import LoggingMixin
from ..utils.validation import validate_tensor_shape, PWMKValidationError
from ..utils.monitoring import get_metrics_collector


@dataclass
class QuantumState:
    """Represents a quantum superposition state for planning."""
    amplitudes: np.ndarray  # Complex probability amplitudes
    actions: List[str]      # Corresponding action sequences
    phase: float           # Global phase
    entanglement_map: Dict[int, List[int]]  # Agent entanglement relationships


@dataclass 
class PlanningResult:
    """Result from quantum-inspired planning."""
    best_action_sequence: List[str]
    probability: float
    quantum_advantage: float  # Estimated speedup over classical
    interference_patterns: Dict[str, float]
    planning_time: float


class QuantumInspiredPlanner(LoggingMixin):
    """
    Quantum-inspired task planner using superposition and interference.
    
    Implements quantum computing principles for enhanced exploration of
    the planning space, enabling simultaneous evaluation of multiple
    action sequences through quantum superposition.
    """
    
    def __init__(
        self,
        num_qubits: int = 8,
        max_depth: int = 10,
        interference_threshold: float = 0.1,
        coherence_time: float = 1.0,
        num_agents: int = 2
    ):
        super().__init__()
        
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.interference_threshold = interference_threshold
        self.coherence_time = coherence_time
        self.num_agents = num_agents
        
        # Quantum state dimensions
        self.hilbert_space_dim = 2 ** num_qubits
        
        # Initialize quantum operators
        self._initialize_quantum_operators()
        
        # Planning statistics
        self.planning_stats = defaultdict(list)
        
        self.logger.info(
            f"Initialized QuantumInspiredPlanner: qubits={num_qubits}, "
            f"max_depth={max_depth}, agents={num_agents}"
        )
    
    def _initialize_quantum_operators(self) -> None:
        """Initialize quantum gates and operators for planning."""
        
        # Pauli matrices
        self.pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.eye(2, dtype=complex)
        
        # Hadamard gate for superposition
        self.hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        # Phase gates
        self.phase_gate = lambda theta: np.array([
            [1, 0], 
            [0, np.exp(1j * theta)]
        ], dtype=complex)
        
        # CNOT gate for entanglement
        self.cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        self.logger.debug("Quantum operators initialized")
    
    def create_superposition_state(
        self, 
        action_space: List[str],
        initial_weights: Optional[np.ndarray] = None
    ) -> QuantumState:
        """
        Create quantum superposition of possible action sequences.
        
        Args:
            action_space: Available actions for planning
            initial_weights: Optional prior weights for actions
            
        Returns:
            QuantumState representing superposition of action sequences
        """
        start_time = time.time()
        
        try:
            num_actions = min(len(action_space), self.hilbert_space_dim)
            
            if initial_weights is None:
                # Equal superposition (Hadamard-like initialization)
                amplitudes = np.ones(num_actions, dtype=complex) / np.sqrt(num_actions)
            else:
                # Weighted superposition based on prior knowledge
                if len(initial_weights) != num_actions:
                    raise PWMKValidationError(
                        f"Initial weights length {len(initial_weights)} != num_actions {num_actions}"
                    )
                amplitudes = np.sqrt(initial_weights).astype(complex)
                amplitudes /= np.linalg.norm(amplitudes)
            
            # Add random phases for richer interference patterns
            phases = np.random.uniform(0, 2*np.pi, num_actions)
            amplitudes *= np.exp(1j * phases)
            
            # Create entanglement map for multi-agent coordination
            entanglement_map = self._generate_entanglement_map()
            
            quantum_state = QuantumState(
                amplitudes=amplitudes,
                actions=action_space[:num_actions],
                phase=0.0,
                entanglement_map=entanglement_map
            )
            
            duration = time.time() - start_time
            get_metrics_collector().record_quantum_operation("create_superposition", duration)
            
            self.logger.debug(
                f"Created superposition state: {num_actions} actions, "
                f"entanglement_groups={len(entanglement_map)}, duration={duration:.4f}s"
            )
            
            return quantum_state
            
        except Exception as e:
            self.logger.error(f"Failed to create superposition state: {e}")
            raise
    
    def _generate_entanglement_map(self) -> Dict[int, List[int]]:
        """Generate entanglement relationships between agents."""
        entanglement_map = {}
        
        # Create entanglement based on agent proximity or interaction patterns
        for agent_id in range(self.num_agents):
            # Simple ring topology for multi-agent entanglement
            next_agent = (agent_id + 1) % self.num_agents
            entanglement_map[agent_id] = [next_agent]
            
            # Add additional entanglements for more complex coordination
            if self.num_agents > 2:
                other_agent = (agent_id + 2) % self.num_agents
                entanglement_map[agent_id].append(other_agent)
        
        return entanglement_map
    
    def apply_quantum_interference(
        self, 
        quantum_state: QuantumState,
        goal_vector: np.ndarray,
        environment_feedback: Optional[Dict[str, float]] = None
    ) -> QuantumState:
        """
        Apply quantum interference to amplify/suppress action probabilities.
        
        Args:
            quantum_state: Current quantum state
            goal_vector: Target goal encoded as vector
            environment_feedback: Optional feedback from environment
            
        Returns:
            Updated quantum state after interference
        """
        start_time = time.time()
        
        try:
            amplitudes = quantum_state.amplitudes.copy()
            
            # Goal-directed interference
            goal_alignment = self._compute_goal_alignment(quantum_state.actions, goal_vector)
            
            # Apply phase shifts based on goal alignment
            for i, alignment in enumerate(goal_alignment):
                phase_shift = alignment * np.pi / 4  # Constructive interference for aligned actions
                amplitudes[i] *= np.exp(1j * phase_shift)
            
            # Environment-based interference
            if environment_feedback:
                for i, action in enumerate(quantum_state.actions):
                    if action in environment_feedback:
                        feedback_phase = environment_feedback[action] * np.pi / 2
                        amplitudes[i] *= np.exp(1j * feedback_phase)
            
            # Apply decoherence based on coherence time
            decoherence_factor = np.exp(-time.time() / self.coherence_time)
            amplitudes *= decoherence_factor
            
            # Renormalize amplitudes
            norm = np.linalg.norm(amplitudes)
            if norm > 1e-10:
                amplitudes /= norm
            else:
                # Reinitialize if amplitudes become too small
                amplitudes = np.ones(len(amplitudes), dtype=complex) / np.sqrt(len(amplitudes))
            
            # Update quantum state
            updated_state = QuantumState(
                amplitudes=amplitudes,
                actions=quantum_state.actions,
                phase=quantum_state.phase,
                entanglement_map=quantum_state.entanglement_map
            )
            
            duration = time.time() - start_time
            get_metrics_collector().record_quantum_operation("apply_interference", duration)
            
            self.logger.debug(f"Applied quantum interference: duration={duration:.4f}s")
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Failed to apply quantum interference: {e}")
            raise
    
    def _compute_goal_alignment(self, actions: List[str], goal_vector: np.ndarray) -> List[float]:
        """Compute alignment between actions and goal."""
        alignments = []
        
        for action in actions:
            # Simple heuristic: compute alignment based on action similarity to goal encoding
            action_vector = self._encode_action(action)
            
            if len(action_vector) == len(goal_vector):
                alignment = np.dot(action_vector, goal_vector) / (
                    np.linalg.norm(action_vector) * np.linalg.norm(goal_vector) + 1e-8
                )
            else:
                # Fallback alignment based on action index
                alignment = 0.5
            
            alignments.append(alignment)
        
        return alignments
    
    def _encode_action(self, action: str) -> np.ndarray:
        """Encode action as vector for goal alignment computation."""
        # Simple encoding: use action string hash and normalize
        action_hash = hash(action) % 1000
        vector = np.array([
            action_hash % 10,
            (action_hash // 10) % 10,
            (action_hash // 100) % 10
        ], dtype=float)
        
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-8)
    
    def measure_quantum_state(
        self, 
        quantum_state: QuantumState,
        num_measurements: int = 1000
    ) -> PlanningResult:
        """
        Perform quantum measurement to collapse superposition into action sequence.
        
        Args:
            quantum_state: Quantum state to measure
            num_measurements: Number of quantum measurements for statistics
            
        Returns:
            PlanningResult with best action sequence and quantum metrics
        """
        start_time = time.time()
        
        try:
            # Calculate probabilities from amplitudes
            probabilities = np.abs(quantum_state.amplitudes) ** 2
            
            # Ensure probabilities are normalized
            probabilities /= np.sum(probabilities)
            
            # Perform multiple measurements to get statistics
            measurement_counts = defaultdict(int)
            
            for _ in range(num_measurements):
                # Sample action based on quantum probabilities
                action_idx = np.random.choice(len(quantum_state.actions), p=probabilities)
                action = quantum_state.actions[action_idx]
                measurement_counts[action] += 1
            
            # Find most frequently measured action (highest probability)
            best_action = max(measurement_counts.keys(), key=lambda k: measurement_counts[k])
            best_probability = measurement_counts[best_action] / num_measurements
            best_action_idx = quantum_state.actions.index(best_action)
            
            # Calculate quantum advantage (theoretical speedup)
            classical_complexity = len(quantum_state.actions)
            quantum_complexity = np.sqrt(len(quantum_state.actions))
            quantum_advantage = classical_complexity / quantum_complexity
            
            # Analyze interference patterns
            interference_patterns = self._analyze_interference_patterns(quantum_state)
            
            planning_time = time.time() - start_time
            
            result = PlanningResult(
                best_action_sequence=[best_action],
                probability=best_probability,
                quantum_advantage=quantum_advantage,
                interference_patterns=interference_patterns,
                planning_time=planning_time
            )
            
            # Record metrics
            get_metrics_collector().record_quantum_operation("quantum_measurement", planning_time)
            get_metrics_collector().record_metric("quantum_advantage", quantum_advantage)
            
            self.planning_stats["planning_times"].append(planning_time)
            self.planning_stats["quantum_advantages"].append(quantum_advantage)
            
            self.logger.info(
                f"Quantum measurement complete: best_action={best_action}, "
                f"probability={best_probability:.3f}, advantage={quantum_advantage:.2f}x, "
                f"time={planning_time:.4f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quantum measurement failed: {e}")
            raise
    
    def _analyze_interference_patterns(self, quantum_state: QuantumState) -> Dict[str, float]:
        """Analyze quantum interference patterns in the state."""
        patterns = {}
        
        amplitudes = quantum_state.amplitudes
        
        # Compute interference strength
        interference_strength = np.std(np.abs(amplitudes))
        patterns["interference_strength"] = float(interference_strength)
        
        # Compute phase coherence
        phases = np.angle(amplitudes)
        phase_coherence = 1.0 - np.std(phases) / np.pi
        patterns["phase_coherence"] = float(phase_coherence)
        
        # Compute entanglement measure
        entanglement_strength = len(quantum_state.entanglement_map) / self.num_agents
        patterns["entanglement_strength"] = float(entanglement_strength)
        
        return patterns
    
    def plan(
        self,
        initial_state: Dict[str, Any],
        goal: str,
        action_space: List[str],
        max_iterations: int = 100
    ) -> PlanningResult:
        """
        Main quantum-inspired planning method.
        
        Args:
            initial_state: Current environment state
            goal: Target goal description
            action_space: Available actions
            max_iterations: Maximum planning iterations
            
        Returns:
            PlanningResult with optimal action sequence
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting quantum planning: goal='{goal}', actions={len(action_space)}")
            
            # Encode goal as vector
            goal_vector = self._encode_goal(goal)
            
            # Create initial superposition state
            quantum_state = self.create_superposition_state(action_space)
            
            # Iterative quantum evolution
            for iteration in range(max_iterations):
                # Apply quantum interference based on goal and environment
                quantum_state = self.apply_quantum_interference(
                    quantum_state, 
                    goal_vector,
                    environment_feedback=self._get_environment_feedback(initial_state)
                )
                
                # Check convergence
                if iteration % 10 == 0:
                    convergence = self._check_convergence(quantum_state)
                    if convergence > 0.95:
                        self.logger.debug(f"Converged at iteration {iteration}")
                        break
            
            # Final measurement
            result = self.measure_quantum_state(quantum_state)
            
            total_time = time.time() - start_time
            result.planning_time = total_time
            
            self.logger.info(
                f"Quantum planning complete: {iteration+1} iterations, "
                f"total_time={total_time:.4f}s, quantum_advantage={result.quantum_advantage:.2f}x"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quantum planning failed: {e}")
            raise
    
    def _encode_goal(self, goal: str) -> np.ndarray:
        """Encode goal string as numerical vector."""
        # Simple encoding based on goal string characteristics
        goal_hash = hash(goal) % 10000
        vector = np.array([
            goal_hash % 100,
            (goal_hash // 100) % 100,
            len(goal) % 100
        ], dtype=float)
        
        return vector / (np.linalg.norm(vector) + 1e-8)
    
    def _get_environment_feedback(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Extract environment feedback for quantum interference."""
        feedback = {}
        
        # Simple heuristic feedback based on state properties
        if "obstacles" in state:
            feedback["move_forward"] = -0.5  # Negative feedback if obstacles present
            feedback["turn_left"] = 0.2
            feedback["turn_right"] = 0.2
        else:
            feedback["move_forward"] = 0.5   # Positive feedback if clear path
        
        return feedback
    
    def _check_convergence(self, quantum_state: QuantumState) -> float:
        """Check convergence of quantum state evolution."""
        probabilities = np.abs(quantum_state.amplitudes) ** 2
        
        # Convergence measure: entropy reduction
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(probabilities))
        
        convergence = 1.0 - entropy / (max_entropy + 1e-10)
        return float(convergence)
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive planning statistics."""
        stats = dict(self.planning_stats)
        
        if stats["planning_times"]:
            stats["avg_planning_time"] = np.mean(stats["planning_times"])
            stats["std_planning_time"] = np.std(stats["planning_times"])
        
        if stats["quantum_advantages"]:
            stats["avg_quantum_advantage"] = np.mean(stats["quantum_advantages"])
            stats["max_quantum_advantage"] = np.max(stats["quantum_advantages"])
        
        return stats