"""
Quantum annealing scheduler for optimization problems in task planning.

Implements quantum annealing algorithms to solve complex optimization
problems that arise in multi-agent task planning scenarios.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
from dataclasses import dataclass
import time
from enum import Enum

from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector


class AnnealingSchedule(Enum):
    """Different annealing schedule types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"


@dataclass
class AnnealingProblem:
    """Quantum annealing problem specification."""
    hamiltonian: np.ndarray  # Problem Hamiltonian
    variables: List[str]     # Variable names
    constraints: List[Dict[str, Any]]  # Problem constraints
    objective_function: Callable[[np.ndarray], float]


@dataclass
class AnnealingResult:
    """Result from quantum annealing optimization."""
    best_solution: np.ndarray
    best_energy: float
    annealing_time: float
    convergence_history: List[float]
    temperature_schedule: List[float]
    quantum_tunneling_events: int


class QuantumAnnealingScheduler(LoggingMixin):
    """
    Quantum annealing scheduler for complex optimization problems.
    
    Uses quantum annealing principles to solve NP-hard optimization
    problems that arise in multi-agent task planning.
    """
    
    def __init__(
        self,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        annealing_steps: int = 1000,
        schedule_type: AnnealingSchedule = AnnealingSchedule.LINEAR,
        quantum_fluctuation_strength: float = 1.0
    ):
        super().__init__()
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.annealing_steps = annealing_steps
        self.schedule_type = schedule_type
        self.quantum_fluctuation_strength = quantum_fluctuation_strength
        
        # Annealing parameters
        self.current_temperature = initial_temperature
        self.current_step = 0
        
        # Statistics tracking
        self.annealing_stats = {
            "problems_solved": 0,
            "total_annealing_time": 0.0,
            "best_energies": [],
            "quantum_advantages": []
        }
        
        self.logger.info(
            f"Initialized QuantumAnnealingScheduler: T_initial={initial_temperature}, "
            f"T_final={final_temperature}, steps={annealing_steps}, "
            f"schedule={schedule_type.value}"
        )
    
    def solve_optimization_problem(
        self,
        problem: AnnealingProblem,
        initial_solution: Optional[np.ndarray] = None,
        num_runs: int = 10
    ) -> AnnealingResult:
        """
        Solve optimization problem using quantum annealing.
        
        Args:
            problem: AnnealingProblem specification
            initial_solution: Optional starting solution
            num_runs: Number of annealing runs for statistics
            
        Returns:
            AnnealingResult with best solution found
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting quantum annealing: variables={len(problem.variables)}, runs={num_runs}")
            
            best_solution = None
            best_energy = float('inf')
            best_convergence_history = []
            best_temperature_schedule = []
            total_tunneling_events = 0
            
            # Run multiple annealing attempts
            for run in range(num_runs):
                solution, energy, history, temp_schedule, tunneling = self._single_annealing_run(
                    problem, initial_solution
                )
                
                total_tunneling_events += tunneling
                
                if energy < best_energy:
                    best_solution = solution
                    best_energy = energy
                    best_convergence_history = history
                    best_temperature_schedule = temp_schedule
                
                self.logger.debug(f"Run {run+1}/{num_runs}: energy={energy:.6f}")
            
            annealing_time = time.time() - start_time
            
            result = AnnealingResult(
                best_solution=best_solution,
                best_energy=best_energy,
                annealing_time=annealing_time,
                convergence_history=best_convergence_history,
                temperature_schedule=best_temperature_schedule,
                quantum_tunneling_events=total_tunneling_events
            )
            
            # Update statistics
            self.annealing_stats["problems_solved"] += 1
            self.annealing_stats["total_annealing_time"] += annealing_time
            self.annealing_stats["best_energies"].append(best_energy)
            
            # Calculate quantum advantage (heuristic)
            classical_time_estimate = len(problem.variables) ** 2 * 0.001  # Simple estimate
            quantum_advantage = classical_time_estimate / annealing_time if annealing_time > 0 else 1.0
            self.annealing_stats["quantum_advantages"].append(quantum_advantage)
            
            get_metrics_collector().record_quantum_operation("quantum_annealing", annealing_time)
            get_metrics_collector().record_metric("annealing_energy", best_energy)
            get_metrics_collector().record_metric("quantum_tunneling_events", total_tunneling_events)
            
            self.logger.info(
                f"Quantum annealing complete: best_energy={best_energy:.6f}, "
                f"time={annealing_time:.4f}s, tunneling_events={total_tunneling_events}, "
                f"quantum_advantage={quantum_advantage:.2f}x"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quantum annealing failed: {e}")
            raise
    
    def _single_annealing_run(
        self,
        problem: AnnealingProblem,
        initial_solution: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, float, List[float], List[float], int]:
        """Perform a single quantum annealing run."""
        
        # Initialize solution
        if initial_solution is not None:
            current_solution = initial_solution.copy()
        else:
            current_solution = np.random.random(len(problem.variables))
        
        current_energy = problem.objective_function(current_solution)
        
        convergence_history = [current_energy]
        temperature_schedule = []
        tunneling_events = 0
        
        # Annealing loop
        for step in range(self.annealing_steps):
            # Update temperature according to schedule
            temperature = self._get_temperature(step)
            temperature_schedule.append(temperature)
            
            # Generate candidate solution with quantum fluctuations
            candidate_solution = self._generate_candidate_solution(
                current_solution, temperature, problem
            )
            
            candidate_energy = problem.objective_function(candidate_solution)
            
            # Quantum tunneling probability (beyond classical Boltzmann)
            accept_probability = self._calculate_acceptance_probability(
                current_energy, candidate_energy, temperature
            )
            
            # Quantum tunneling enhancement
            if candidate_energy > current_energy:
                quantum_tunneling_prob = self._quantum_tunneling_probability(
                    current_energy, candidate_energy, temperature
                )
                
                if np.random.random() < quantum_tunneling_prob:
                    tunneling_events += 1
                    accept_probability = 1.0  # Force acceptance via quantum tunneling
            
            # Accept or reject candidate
            if np.random.random() < accept_probability:
                current_solution = candidate_solution
                current_energy = candidate_energy
                
                # Check constraints
                if not self._satisfies_constraints(current_solution, problem.constraints):
                    # Apply constraint penalty
                    current_energy += self._constraint_penalty(current_solution, problem.constraints)
            
            convergence_history.append(current_energy)
            
            # Early termination check
            if step > 100 and len(convergence_history) > 50:
                recent_improvement = (convergence_history[-50] - convergence_history[-1]) / abs(convergence_history[-50])
                if recent_improvement < 1e-6:
                    self.logger.debug(f"Early termination at step {step} due to convergence")
                    break
        
        return current_solution, current_energy, convergence_history, temperature_schedule, tunneling_events
    
    def _get_temperature(self, step: int) -> float:
        """Get temperature according to annealing schedule."""
        
        progress = step / self.annealing_steps
        
        if self.schedule_type == AnnealingSchedule.LINEAR:
            temperature = self.initial_temperature * (1 - progress) + self.final_temperature * progress
            
        elif self.schedule_type == AnnealingSchedule.EXPONENTIAL:
            decay_rate = np.log(self.initial_temperature / self.final_temperature)
            temperature = self.initial_temperature * np.exp(-decay_rate * progress)
            
        elif self.schedule_type == AnnealingSchedule.POLYNOMIAL:
            power = 2.0  # Quadratic cooling
            temperature = self.initial_temperature * (1 - progress) ** power + self.final_temperature
            
        elif self.schedule_type == AnnealingSchedule.ADAPTIVE:
            # Adaptive schedule based on convergence rate
            if step < 100:
                temperature = self.initial_temperature
            else:
                # Slow down cooling if not converging
                convergence_rate = 0.5  # Simplified heuristic
                adaptive_factor = 1.0 + (1.0 - convergence_rate) * 0.5
                linear_temp = self.initial_temperature * (1 - progress) + self.final_temperature * progress
                temperature = linear_temp * adaptive_factor
        
        else:
            temperature = self.initial_temperature * (1 - progress) + self.final_temperature * progress
        
        return max(temperature, self.final_temperature)
    
    def _generate_candidate_solution(
        self,
        current_solution: np.ndarray,
        temperature: float,
        problem: AnnealingProblem
    ) -> np.ndarray:
        """Generate candidate solution with quantum fluctuations."""
        
        # Base random perturbation scaled by temperature
        perturbation_strength = temperature / self.initial_temperature
        base_perturbation = np.random.normal(0, perturbation_strength, size=current_solution.shape)
        
        # Quantum fluctuations (non-classical noise)
        quantum_noise = self._generate_quantum_noise(current_solution.shape, temperature)
        
        candidate = current_solution + base_perturbation + quantum_noise
        
        # Ensure candidate stays within valid bounds
        candidate = np.clip(candidate, 0.0, 1.0)
        
        return candidate
    
    def _generate_quantum_noise(self, shape: Tuple[int, ...], temperature: float) -> np.ndarray:
        """Generate quantum noise for enhanced exploration."""
        
        # Quantum fluctuations based on Heisenberg uncertainty principle
        quantum_scale = self.quantum_fluctuation_strength * np.sqrt(temperature)
        
        # Non-Gaussian quantum noise (approximating quantum field fluctuations)
        uniform_noise = np.random.uniform(-1, 1, shape)
        quantum_noise = quantum_scale * np.sign(uniform_noise) * np.sqrt(np.abs(uniform_noise))
        
        return quantum_noise
    
    def _calculate_acceptance_probability(
        self,
        current_energy: float,
        candidate_energy: float,
        temperature: float
    ) -> float:
        """Calculate acceptance probability (Metropolis-Hastings with quantum enhancement)."""
        
        if candidate_energy <= current_energy:
            return 1.0
        
        energy_difference = candidate_energy - current_energy
        
        # Classical Boltzmann factor
        if temperature > 1e-10:
            classical_prob = np.exp(-energy_difference / temperature)
        else:
            classical_prob = 0.0
        
        return classical_prob
    
    def _quantum_tunneling_probability(
        self,
        current_energy: float,
        candidate_energy: float,
        temperature: float
    ) -> float:
        """Calculate quantum tunneling probability for barrier penetration."""
        
        energy_barrier = candidate_energy - current_energy
        
        if energy_barrier <= 0:
            return 0.0
        
        # Quantum tunneling probability (simplified WKB approximation)
        # Increases tunneling probability for quantum advantage
        tunneling_strength = self.quantum_fluctuation_strength
        effective_barrier = energy_barrier / (1.0 + tunneling_strength * temperature)
        
        if effective_barrier > 10.0:  # Prevent numerical overflow
            return 0.0
        
        tunneling_prob = 0.1 * np.exp(-effective_barrier)  # Enhanced tunneling factor
        
        return min(tunneling_prob, 0.3)  # Cap at 30% for stability
    
    def _satisfies_constraints(self, solution: np.ndarray, constraints: List[Dict[str, Any]]) -> bool:
        """Check if solution satisfies all constraints."""
        
        for constraint in constraints:
            constraint_type = constraint.get("type", "")
            
            if constraint_type == "bounds":
                min_val = constraint.get("min", 0.0)
                max_val = constraint.get("max", 1.0)
                if np.any(solution < min_val) or np.any(solution > max_val):
                    return False
            
            elif constraint_type == "sum":
                target_sum = constraint.get("target", 1.0)
                tolerance = constraint.get("tolerance", 0.01)
                if abs(np.sum(solution) - target_sum) > tolerance:
                    return False
            
            elif constraint_type == "custom":
                constraint_func = constraint.get("function")
                if constraint_func and not constraint_func(solution):
                    return False
        
        return True
    
    def _constraint_penalty(self, solution: np.ndarray, constraints: List[Dict[str, Any]]) -> float:
        """Calculate penalty for constraint violations."""
        
        penalty = 0.0
        
        for constraint in constraints:
            constraint_type = constraint.get("type", "")
            weight = constraint.get("weight", 1.0)
            
            if constraint_type == "bounds":
                min_val = constraint.get("min", 0.0)
                max_val = constraint.get("max", 1.0)
                
                lower_violations = np.maximum(0, min_val - solution)
                upper_violations = np.maximum(0, solution - max_val)
                penalty += weight * (np.sum(lower_violations) + np.sum(upper_violations))
            
            elif constraint_type == "sum":
                target_sum = constraint.get("target", 1.0)
                current_sum = np.sum(solution)
                penalty += weight * abs(current_sum - target_sum)
            
            elif constraint_type == "custom":
                penalty_func = constraint.get("penalty_function")
                if penalty_func:
                    penalty += weight * penalty_func(solution)
        
        return penalty
    
    def create_task_planning_problem(
        self,
        agents: List[str],
        tasks: List[str],
        agent_capabilities: Dict[str, List[str]],
        task_priorities: Dict[str, float],
        coordination_requirements: List[Tuple[str, str]]
    ) -> AnnealingProblem:
        """
        Create quantum annealing problem for multi-agent task assignment.
        
        Args:
            agents: List of agent identifiers
            tasks: List of task identifiers
            agent_capabilities: Agent capability mappings
            task_priorities: Task priority weights
            coordination_requirements: Required agent coordination pairs
            
        Returns:
            AnnealingProblem for task planning optimization
        """
        
        try:
            num_agents = len(agents)
            num_tasks = len(tasks)
            total_variables = num_agents * num_tasks
            
            # Create variable names
            variables = []
            for i, agent in enumerate(agents):
                for j, task in enumerate(tasks):
                    variables.append(f"{agent}_{task}")
            
            # Build Hamiltonian matrix
            hamiltonian = np.zeros((total_variables, total_variables))
            
            # Diagonal terms: task priorities and agent capabilities
            for i, agent in enumerate(agents):
                for j, task in enumerate(tasks):
                    var_idx = i * num_tasks + j
                    
                    # Task priority contribution
                    priority = task_priorities.get(task, 1.0)
                    hamiltonian[var_idx, var_idx] -= priority  # Negative for minimization
                    
                    # Agent capability penalty
                    if task not in agent_capabilities.get(agent, []):
                        hamiltonian[var_idx, var_idx] += 10.0  # High penalty
            
            # Off-diagonal terms: coordination requirements
            for agent1, agent2 in coordination_requirements:
                if agent1 in agents and agent2 in agents:
                    agent1_idx = agents.index(agent1)
                    agent2_idx = agents.index(agent2)
                    
                    for j in range(num_tasks):
                        var1_idx = agent1_idx * num_tasks + j
                        var2_idx = agent2_idx * num_tasks + j
                        
                        # Encourage coordination (negative coupling)
                        hamiltonian[var1_idx, var2_idx] -= 0.5
                        hamiltonian[var2_idx, var1_idx] -= 0.5
            
            # Create objective function
            def objective_function(x: np.ndarray) -> float:
                # Reshape solution to agent-task assignment matrix
                assignment_matrix = x.reshape(num_agents, num_tasks)
                
                energy = 0.0
                
                # Task completion rewards
                for j, task in enumerate(tasks):
                    task_assigned = np.any(assignment_matrix[:, j] > 0.5)
                    if task_assigned:
                        energy -= task_priorities.get(task, 1.0)
                
                # Agent capability penalties
                for i, agent in enumerate(agents):
                    for j, task in enumerate(tasks):
                        if (assignment_matrix[i, j] > 0.5 and 
                            task not in agent_capabilities.get(agent, [])):
                            energy += 10.0
                
                # Coordination bonuses
                for agent1, agent2 in coordination_requirements:
                    if agent1 in agents and agent2 in agents:
                        agent1_idx = agents.index(agent1)
                        agent2_idx = agents.index(agent2)
                        
                        # Check if agents work on same tasks
                        coordination_score = np.dot(assignment_matrix[agent1_idx], assignment_matrix[agent2_idx])
                        energy -= 0.5 * coordination_score
                
                return energy
            
            # Create constraints
            constraints = [
                {
                    "type": "bounds",
                    "min": 0.0,
                    "max": 1.0,
                    "weight": 1.0
                },
                {
                    "type": "custom",
                    "function": lambda x: np.all(x.reshape(num_agents, num_tasks).sum(axis=0) >= 0.5),  # Each task assigned
                    "penalty_function": lambda x: np.sum(np.maximum(0, 0.5 - x.reshape(num_agents, num_tasks).sum(axis=0))),
                    "weight": 5.0
                }
            ]
            
            problem = AnnealingProblem(
                hamiltonian=hamiltonian,
                variables=variables,
                constraints=constraints,
                objective_function=objective_function
            )
            
            self.logger.info(
                f"Created task planning problem: {num_agents} agents, {num_tasks} tasks, "
                f"{total_variables} variables, {len(coordination_requirements)} coordination requirements"
            )
            
            return problem
            
        except Exception as e:
            self.logger.error(f"Failed to create task planning problem: {e}")
            raise
    
    def get_annealing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive annealing statistics."""
        stats = self.annealing_stats.copy()
        
        if stats["problems_solved"] > 0:
            stats["avg_annealing_time"] = stats["total_annealing_time"] / stats["problems_solved"]
        
        if stats["best_energies"]:
            stats["mean_energy"] = np.mean(stats["best_energies"])
            stats["best_energy_overall"] = np.min(stats["best_energies"])
        
        if stats["quantum_advantages"]:
            stats["mean_quantum_advantage"] = np.mean(stats["quantum_advantages"])
            stats["max_quantum_advantage"] = np.max(stats["quantum_advantages"])
        
        return stats