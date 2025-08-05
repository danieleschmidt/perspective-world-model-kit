"""
Quantum circuit optimization for task planning.

Implements quantum circuit design and optimization techniques for
efficient task planning in multi-agent environments.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from dataclasses import dataclass
import time
from enum import Enum

from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector


class GateType(Enum):
    """Quantum gate types for circuit construction."""
    HADAMARD = "H"
    PAULI_X = "X" 
    PAULI_Y = "Y"
    PAULI_Z = "Z"
    PHASE = "P"
    ROTATION_X = "RX"
    ROTATION_Y = "RY" 
    ROTATION_Z = "RZ"
    CNOT = "CNOT"
    TOFFOLI = "TOFFOLI"


@dataclass
class QuantumGate:
    """Represents a quantum gate in the circuit."""
    gate_type: GateType
    target_qubits: List[int]
    control_qubits: Optional[List[int]] = None
    parameters: Optional[List[float]] = None
    
    
@dataclass
class QuantumCircuit:
    """Quantum circuit for task planning optimization."""
    num_qubits: int
    gates: List[QuantumGate]
    depth: int
    
    
@dataclass
class CircuitOptimizationResult:
    """Result from quantum circuit optimization."""
    optimized_circuit: QuantumCircuit
    optimization_time: float
    gate_count_reduction: float
    depth_reduction: float
    fidelity: float


class QuantumCircuitOptimizer(LoggingMixin):
    """
    Quantum circuit optimizer for task planning applications.
    
    Optimizes quantum circuits to minimize gate count and depth while
    maintaining high fidelity for planning computations.
    """
    
    def __init__(
        self,
        max_qubits: int = 10,
        optimization_level: int = 2,
        target_fidelity: float = 0.99
    ):
        super().__init__()
        
        self.max_qubits = max_qubits
        self.optimization_level = optimization_level
        self.target_fidelity = target_fidelity
        
        # Initialize gate matrices
        self._initialize_gate_matrices()
        
        # Optimization statistics
        self.optimization_stats = {
            "circuits_optimized": 0,
            "total_gate_reduction": 0,
            "total_depth_reduction": 0,
            "avg_fidelity": []
        }
        
        self.logger.info(
            f"Initialized QuantumCircuitOptimizer: max_qubits={max_qubits}, "
            f"optimization_level={optimization_level}, target_fidelity={target_fidelity}"
        )
    
    def _initialize_gate_matrices(self) -> None:
        """Initialize quantum gate matrices."""
        
        # Single-qubit gates
        self.gate_matrices = {
            GateType.HADAMARD: np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            GateType.PAULI_X: np.array([[0, 1], [1, 0]], dtype=complex),
            GateType.PAULI_Y: np.array([[0, -1j], [1j, 0]], dtype=complex),
            GateType.PAULI_Z: np.array([[1, 0], [0, -1]], dtype=complex),
        }
        
        # Parameterized gates (functions)
        self.parameterized_gates = {
            GateType.PHASE: lambda theta: np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex),
            GateType.ROTATION_X: lambda theta: np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * self.gate_matrices[GateType.PAULI_X],
            GateType.ROTATION_Y: lambda theta: np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * self.gate_matrices[GateType.PAULI_Y],
            GateType.ROTATION_Z: lambda theta: np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * self.gate_matrices[GateType.PAULI_Z],
        }
        
        # Two-qubit gates
        self.cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        self.logger.debug("Quantum gate matrices initialized")
    
    def create_planning_circuit(
        self,
        num_agents: int,
        planning_depth: int,
        action_encoding: Dict[str, int]
    ) -> QuantumCircuit:
        """
        Create quantum circuit optimized for multi-agent task planning.
        
        Args:
            num_agents: Number of agents in the system
            planning_depth: Planning horizon depth
            action_encoding: Mapping from actions to qubit indices
            
        Returns:
            QuantumCircuit optimized for planning
        """
        start_time = time.time()
        
        try:
            # Calculate required qubits
            qubits_per_agent = max(2, int(np.ceil(np.log2(len(action_encoding)))))
            total_qubits = num_agents * qubits_per_agent
            
            if total_qubits > self.max_qubits:
                raise ValueError(f"Required qubits {total_qubits} exceeds maximum {self.max_qubits}")
            
            gates = []
            
            # Initialize superposition for all agent action spaces
            for agent in range(num_agents):
                agent_qubits = list(range(agent * qubits_per_agent, (agent + 1) * qubits_per_agent))
                
                # Create superposition
                for qubit in agent_qubits:
                    gates.append(QuantumGate(GateType.HADAMARD, [qubit]))
            
            # Add entanglement between agents for coordination
            for depth_level in range(planning_depth):
                for agent in range(num_agents - 1):
                    control_qubit = agent * qubits_per_agent
                    target_qubit = (agent + 1) * qubits_per_agent
                    
                    gates.append(QuantumGate(
                        GateType.CNOT, 
                        [target_qubit], 
                        control_qubits=[control_qubit]
                    ))
                
                # Add phase gates for planning dynamics
                for agent in range(num_agents):
                    for q in range(qubits_per_agent):
                        qubit_idx = agent * qubits_per_agent + q
                        phase = np.pi * (depth_level + 1) / (planning_depth + 1)
                        
                        gates.append(QuantumGate(
                            GateType.PHASE,
                            [qubit_idx],
                            parameters=[phase]
                        ))
            
            circuit = QuantumCircuit(
                num_qubits=total_qubits,
                gates=gates,
                depth=self._calculate_circuit_depth(gates)
            )
            
            duration = time.time() - start_time
            get_metrics_collector().record_quantum_operation("create_circuit", duration)
            
            self.logger.info(
                f"Created planning circuit: {total_qubits} qubits, {len(gates)} gates, "
                f"depth={circuit.depth}, time={duration:.4f}s"
            )
            
            return circuit
            
        except Exception as e:
            self.logger.error(f"Failed to create planning circuit: {e}")
            raise
    
    def optimize_circuit(
        self, 
        circuit: QuantumCircuit,
        optimization_passes: Optional[List[str]] = None
    ) -> CircuitOptimizationResult:
        """
        Optimize quantum circuit for reduced gate count and depth.
        
        Args:
            circuit: Input quantum circuit
            optimization_passes: List of optimization techniques to apply
            
        Returns:
            CircuitOptimizationResult with optimized circuit and metrics
        """
        start_time = time.time()
        
        try:
            if optimization_passes is None:
                optimization_passes = ["gate_fusion", "redundancy_removal", "depth_optimization"]
            
            original_gate_count = len(circuit.gates)
            original_depth = circuit.depth
            
            optimized_circuit = circuit
            
            # Apply optimization passes
            for pass_name in optimization_passes:
                if pass_name == "gate_fusion":
                    optimized_circuit = self._apply_gate_fusion(optimized_circuit)
                elif pass_name == "redundancy_removal":
                    optimized_circuit = self._remove_redundant_gates(optimized_circuit)
                elif pass_name == "depth_optimization":
                    optimized_circuit = self._optimize_circuit_depth(optimized_circuit)
                elif pass_name == "commutation_analysis":
                    optimized_circuit = self._apply_commutation_optimization(optimized_circuit)
            
            # Calculate optimization metrics
            final_gate_count = len(optimized_circuit.gates)
            final_depth = optimized_circuit.depth
            
            gate_reduction = (original_gate_count - final_gate_count) / original_gate_count
            depth_reduction = (original_depth - final_depth) / original_depth
            
            # Calculate fidelity (simplified model)
            fidelity = self._calculate_optimization_fidelity(circuit, optimized_circuit)
            
            optimization_time = time.time() - start_time
            
            result = CircuitOptimizationResult(
                optimized_circuit=optimized_circuit,
                optimization_time=optimization_time,
                gate_count_reduction=gate_reduction,
                depth_reduction=depth_reduction,
                fidelity=fidelity
            )
            
            # Update statistics
            self.optimization_stats["circuits_optimized"] += 1
            self.optimization_stats["total_gate_reduction"] += gate_reduction
            self.optimization_stats["total_depth_reduction"] += depth_reduction
            self.optimization_stats["avg_fidelity"].append(fidelity)
            
            get_metrics_collector().record_quantum_operation("circuit_optimization", optimization_time)
            get_metrics_collector().record_metric("gate_reduction", gate_reduction)
            get_metrics_collector().record_metric("depth_reduction", depth_reduction)
            
            self.logger.info(
                f"Circuit optimization complete: gate_reduction={gate_reduction:.2%}, "
                f"depth_reduction={depth_reduction:.2%}, fidelity={fidelity:.4f}, "
                f"time={optimization_time:.4f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Circuit optimization failed: {e}")
            raise
    
    def _apply_gate_fusion(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply gate fusion optimization to reduce gate count."""
        
        fused_gates = []
        skip_indices = set()
        
        for i, gate in enumerate(circuit.gates):
            if i in skip_indices:
                continue
                
            # Look for consecutive gates on same qubits that can be fused
            if i + 1 < len(circuit.gates):
                next_gate = circuit.gates[i + 1]
                
                if (gate.target_qubits == next_gate.target_qubits and
                    gate.control_qubits == next_gate.control_qubits):
                    
                    # Fuse compatible gate pairs
                    fused_gate = self._fuse_gate_pair(gate, next_gate)
                    if fused_gate:
                        fused_gates.append(fused_gate)
                        skip_indices.add(i + 1)
                        continue
            
            fused_gates.append(gate)
        
        return QuantumCircuit(
            num_qubits=circuit.num_qubits,
            gates=fused_gates,
            depth=self._calculate_circuit_depth(fused_gates)
        )
    
    def _fuse_gate_pair(self, gate1: QuantumGate, gate2: QuantumGate) -> Optional[QuantumGate]:
        """Attempt to fuse two compatible gates."""
        
        # Rotation gate fusion
        if (gate1.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z] and
            gate1.gate_type == gate2.gate_type):
            
            # Combine rotation angles
            angle1 = gate1.parameters[0] if gate1.parameters else 0
            angle2 = gate2.parameters[0] if gate2.parameters else 0
            combined_angle = angle1 + angle2
            
            return QuantumGate(
                gate_type=gate1.gate_type,
                target_qubits=gate1.target_qubits,
                control_qubits=gate1.control_qubits,
                parameters=[combined_angle]
            )
        
        # Phase gate fusion
        if gate1.gate_type == GateType.PHASE and gate2.gate_type == GateType.PHASE:
            phase1 = gate1.parameters[0] if gate1.parameters else 0
            phase2 = gate2.parameters[0] if gate2.parameters else 0
            combined_phase = phase1 + phase2
            
            return QuantumGate(
                gate_type=GateType.PHASE,
                target_qubits=gate1.target_qubits,
                parameters=[combined_phase]
            )
        
        return None
    
    def _remove_redundant_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Remove redundant gates from the circuit."""
        
        non_redundant_gates = []
        
        for gate in circuit.gates:
            # Remove identity operations
            if self._is_identity_gate(gate):
                continue
                
            # Remove pairs of inverse gates
            if (non_redundant_gates and 
                self._are_inverse_gates(non_redundant_gates[-1], gate)):
                non_redundant_gates.pop()
                continue
            
            non_redundant_gates.append(gate)
        
        return QuantumCircuit(
            num_qubits=circuit.num_qubits,
            gates=non_redundant_gates,
            depth=self._calculate_circuit_depth(non_redundant_gates)
        )
    
    def _is_identity_gate(self, gate: QuantumGate) -> bool:
        """Check if gate is effectively an identity operation."""
        
        if gate.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
            if gate.parameters and abs(gate.parameters[0]) < 1e-10:
                return True
        
        if gate.gate_type == GateType.PHASE:
            if gate.parameters and abs(gate.parameters[0]) < 1e-10:
                return True
        
        return False
    
    def _are_inverse_gates(self, gate1: QuantumGate, gate2: QuantumGate) -> bool:
        """Check if two gates are inverses of each other."""
        
        if (gate1.target_qubits != gate2.target_qubits or
            gate1.control_qubits != gate2.control_qubits):
            return False
        
        # Self-inverse gates
        self_inverse_gates = [GateType.HADAMARD, GateType.PAULI_X, GateType.PAULI_Y, GateType.PAULI_Z]
        if gate1.gate_type in self_inverse_gates and gate1.gate_type == gate2.gate_type:
            return True
        
        # Rotation gates with opposite angles
        rotation_gates = [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]
        if (gate1.gate_type in rotation_gates and gate1.gate_type == gate2.gate_type):
            if (gate1.parameters and gate2.parameters and
                abs(gate1.parameters[0] + gate2.parameters[0]) < 1e-10):
                return True
        
        return False
    
    def _optimize_circuit_depth(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize circuit depth by reordering commuting gates."""
        
        # Group gates by the qubits they act on
        qubit_gate_map = {i: [] for i in range(circuit.num_qubits)}
        
        for gate_idx, gate in enumerate(circuit.gates):
            for qubit in gate.target_qubits:
                qubit_gate_map[qubit].append((gate_idx, gate))
            
            if gate.control_qubits:
                for qubit in gate.control_qubits:
                    qubit_gate_map[qubit].append((gate_idx, gate))
        
        # Find gates that can be parallelized
        optimized_layers = []
        remaining_gates = set(range(len(circuit.gates)))
        
        while remaining_gates:
            current_layer = []
            used_qubits = set()
            
            for gate_idx in sorted(remaining_gates):
                gate = circuit.gates[gate_idx]
                gate_qubits = set(gate.target_qubits)
                if gate.control_qubits:
                    gate_qubits.update(gate.control_qubits)
                
                # Check if gate can be added to current layer
                if not gate_qubits.intersection(used_qubits):
                    current_layer.append(gate)
                    used_qubits.update(gate_qubits)
                    remaining_gates.remove(gate_idx)
            
            if current_layer:
                optimized_layers.extend(current_layer)
        
        return QuantumCircuit(
            num_qubits=circuit.num_qubits,
            gates=optimized_layers,
            depth=len([layer for layer in optimized_layers if layer])
        )
    
    def _apply_commutation_optimization(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply commutation rules to optimize gate ordering."""
        
        optimized_gates = circuit.gates.copy()
        
        # Simple commutation: move single-qubit gates before two-qubit gates when possible
        for i in range(len(optimized_gates) - 1):
            gate1 = optimized_gates[i]
            gate2 = optimized_gates[i + 1]
            
            if self._gates_commute(gate1, gate2):
                # Check if swapping improves circuit properties
                if self._should_swap_gates(gate1, gate2):
                    optimized_gates[i], optimized_gates[i + 1] = gate2, gate1
        
        return QuantumCircuit(
            num_qubits=circuit.num_qubits,
            gates=optimized_gates,
            depth=self._calculate_circuit_depth(optimized_gates)
        )
    
    def _gates_commute(self, gate1: QuantumGate, gate2: QuantumGate) -> bool:
        """Check if two gates commute."""
        
        gate1_qubits = set(gate1.target_qubits)
        if gate1.control_qubits:
            gate1_qubits.update(gate1.control_qubits)
        
        gate2_qubits = set(gate2.target_qubits)
        if gate2.control_qubits:
            gate2_qubits.update(gate2.control_qubits)
        
        # Gates commute if they act on disjoint qubit sets
        return not gate1_qubits.intersection(gate2_qubits)
    
    def _should_swap_gates(self, gate1: QuantumGate, gate2: QuantumGate) -> bool:
        """Determine if swapping gates improves circuit properties."""
        
        # Prefer single-qubit gates before two-qubit gates
        gate1_is_single = len(gate1.target_qubits) == 1 and not gate1.control_qubits
        gate2_is_single = len(gate2.target_qubits) == 1 and not gate2.control_qubits
        
        if gate2_is_single and not gate1_is_single:
            return True
        
        return False
    
    def _calculate_circuit_depth(self, gates: List[QuantumGate]) -> int:
        """Calculate the depth of a quantum circuit."""
        
        if not gates:
            return 0
        
        # Track the latest time each qubit is used
        qubit_times = {}
        
        for gate in gates:
            all_qubits = gate.target_qubits[:]
            if gate.control_qubits:
                all_qubits.extend(gate.control_qubits)
            
            # Find the latest time among all qubits used by this gate
            max_time = max([qubit_times.get(q, 0) for q in all_qubits])
            new_time = max_time + 1
            
            # Update times for all qubits used by this gate
            for qubit in all_qubits:
                qubit_times[qubit] = new_time
        
        return max(qubit_times.values()) if qubit_times else 0
    
    def _calculate_optimization_fidelity(
        self, 
        original: QuantumCircuit, 
        optimized: QuantumCircuit
    ) -> float:
        """Calculate fidelity between original and optimized circuits."""
        
        # Simplified fidelity calculation based on gate count and type preservation  
        if len(original.gates) == 0:
            return 1.0
        
        # Count gate types in both circuits
        original_gate_counts = {}
        optimized_gate_counts = {}
        
        for gate in original.gates:
            gate_type = gate.gate_type
            original_gate_counts[gate_type] = original_gate_counts.get(gate_type, 0) + 1
        
        for gate in optimized.gates:
            gate_type = gate.gate_type
            optimized_gate_counts[gate_type] = optimized_gate_counts.get(gate_type, 0) + 1
        
        # Calculate fidelity based on gate type preservation
        total_difference = 0
        total_gates = len(original.gates)
        
        all_gate_types = set(original_gate_counts.keys()) | set(optimized_gate_counts.keys())
        
        for gate_type in all_gate_types:
            orig_count = original_gate_counts.get(gate_type, 0)
            opt_count = optimized_gate_counts.get(gate_type, 0)
            total_difference += abs(orig_count - opt_count)
        
        fidelity = max(0.0, 1.0 - total_difference / (2 * total_gates))
        return fidelity
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get circuit optimization statistics."""
        stats = self.optimization_stats.copy()
        
        if stats["circuits_optimized"] > 0:
            stats["avg_gate_reduction"] = stats["total_gate_reduction"] / stats["circuits_optimized"]
            stats["avg_depth_reduction"] = stats["total_depth_reduction"] / stats["circuits_optimized"]
        
        if stats["avg_fidelity"]:
            stats["mean_fidelity"] = np.mean(stats["avg_fidelity"])
            stats["min_fidelity"] = np.min(stats["avg_fidelity"])
        
        return stats