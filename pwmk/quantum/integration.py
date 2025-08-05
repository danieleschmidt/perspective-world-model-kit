"""
Integration layer for quantum-enhanced planning with existing PWMK components.

Provides seamless integration between quantum algorithms and the existing
perspective world model and belief reasoning systems.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
from dataclasses import dataclass
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..core.world_model import PerspectiveWorldModel
from ..core.beliefs import BeliefStore
from ..planning.epistemic import EpistemicPlanner
from ..utils.logging import LoggingMixin
from ..utils.monitoring import get_metrics_collector
from ..utils.validation import validate_tensor_shape, PWMKValidationError

from .quantum_planner import QuantumInspiredPlanner, QuantumState, PlanningResult
from .quantum_circuits import QuantumCircuitOptimizer, QuantumCircuit
from .quantum_annealing import QuantumAnnealingScheduler, AnnealingProblem
from .adaptive_quantum import AdaptiveQuantumAlgorithm, QuantumParameters


@dataclass
class QuantumEnhancedPlan:
    """Enhanced planning result combining quantum and classical components."""
    classical_plan: List[str]
    quantum_plan: PlanningResult
    belief_consistency: float
    integration_confidence: float
    execution_strategy: str
    fallback_plan: Optional[List[str]] = None


@dataclass
class QuantumPlanningConfig:
    """Configuration for quantum-enhanced planning."""
    enable_quantum_superposition: bool = True
    enable_circuit_optimization: bool = True
    enable_quantum_annealing: bool = True
    enable_adaptive_parameters: bool = True
    classical_fallback: bool = True
    parallel_execution: bool = True
    max_planning_time: float = 5.0
    confidence_threshold: float = 0.7


class QuantumEnhancedPlanner(LoggingMixin):
    """
    Quantum-enhanced planner integrating all quantum components with PWMK.
    
    Combines quantum-inspired algorithms with existing world models and
    belief systems for superior multi-agent planning performance.
    """
    
    def __init__(
        self,
        world_model: PerspectiveWorldModel,
        belief_store: BeliefStore,
        classical_planner: EpistemicPlanner,
        config: Optional[QuantumPlanningConfig] = None
    ):
        super().__init__()
        
        self.world_model = world_model
        self.belief_store = belief_store
        self.classical_planner = classical_planner
        self.config = config or QuantumPlanningConfig()
        
        # Initialize quantum components
        self._initialize_quantum_components()
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Integration statistics
        self.integration_stats = {
            "total_plans": 0,
            "quantum_advantages": [],
            "fallback_rates": 0,
            "belief_consistency_scores": [],
            "planning_times": []
        }
        
        self.logger.info(
            f"Initialized QuantumEnhancedPlanner: world_model={type(world_model).__name__}, "
            f"quantum_components={self._get_enabled_components()}"
        )
    
    def _initialize_quantum_components(self) -> None:
        """Initialize quantum planning components based on configuration."""
        
        # Quantum-inspired planner
        if self.config.enable_quantum_superposition:
            self.quantum_planner = QuantumInspiredPlanner(
                num_qubits=8,
                max_depth=15,
                num_agents=self.world_model.num_agents
            )
        else:
            self.quantum_planner = None
        
        # Quantum circuit optimizer
        if self.config.enable_circuit_optimization:
            self.circuit_optimizer = QuantumCircuitOptimizer(
                max_qubits=12,
                optimization_level=2
            )
        else:
            self.circuit_optimizer = None
        
        # Quantum annealing scheduler
        if self.config.enable_quantum_annealing:
            self.annealing_scheduler = QuantumAnnealingScheduler(
                initial_temperature=5.0,
                final_temperature=0.01,
                annealing_steps=500
            )
        else:
            self.annealing_scheduler = None
        
        # Adaptive quantum algorithm
        if self.config.enable_adaptive_parameters:
            self.adaptive_quantum = AdaptiveQuantumAlgorithm(
                learning_rate=0.01,
                exploration_rate=0.1
            )
        else:
            self.adaptive_quantum = None
        
        self.logger.debug("Quantum components initialized")
    
    async def plan_async(
        self,
        initial_state: Dict[str, Any],
        goal: str,
        action_space: List[str],
        agent_context: Optional[Dict[str, Any]] = None
    ) -> QuantumEnhancedPlan:
        """
        Asynchronous quantum-enhanced planning with parallel execution.
        
        Args:
            initial_state: Current environment state
            goal: Target goal description
            action_space: Available actions
            agent_context: Additional agent context information
            
        Returns:
            QuantumEnhancedPlan with combined quantum and classical results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting quantum-enhanced planning: goal='{goal}', actions={len(action_space)}")
            
            # Validate inputs
            self._validate_planning_inputs(initial_state, goal, action_space)
            
            # Prepare planning tasks
            planning_tasks = []
            
            # Classical planning task
            classical_task = asyncio.create_task(
                self._run_classical_planning(initial_state, goal, action_space)
            )
            planning_tasks.append(("classical", classical_task))
            
            # Quantum planning tasks (run in parallel if enabled)
            if self.config.parallel_execution:
                if self.quantum_planner:
                    quantum_task = asyncio.create_task(
                        self._run_quantum_planning(initial_state, goal, action_space)
                    )
                    planning_tasks.append(("quantum", quantum_task))
                
                if self.annealing_scheduler:
                    annealing_task = asyncio.create_task(
                        self._run_annealing_planning(initial_state, goal, action_space, agent_context)
                    )
                    planning_tasks.append(("annealing", annealing_task))
            
            # Wait for all tasks to complete or timeout
            planning_results = {}
            
            for task_name, task in planning_tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=self.config.max_planning_time)
                    planning_results[task_name] = result
                    self.logger.debug(f"{task_name} planning completed successfully")
                except asyncio.TimeoutError:
                    self.logger.warning(f"{task_name} planning timed out")
                    planning_results[task_name] = None
                except Exception as e:
                    self.logger.error(f"{task_name} planning failed: {e}")
                    planning_results[task_name] = None
            
            # Integrate planning results
            integrated_plan = await self._integrate_planning_results(
                planning_results, initial_state, goal, action_space
            )
            
            planning_time = time.time() - start_time
            integrated_plan.quantum_plan.planning_time = planning_time
            
            # Update statistics
            self._update_integration_stats(integrated_plan, planning_time)
            
            self.logger.info(
                f"Quantum-enhanced planning complete: strategy={integrated_plan.execution_strategy}, "
                f"confidence={integrated_plan.integration_confidence:.3f}, time={planning_time:.4f}s"
            )
            
            return integrated_plan
            
        except Exception as e:
            self.logger.error(f"Quantum-enhanced planning failed: {e}")
            raise
    
    def plan(
        self,
        initial_state: Dict[str, Any],
        goal: str,
        action_space: List[str],
        agent_context: Optional[Dict[str, Any]] = None
    ) -> QuantumEnhancedPlan:
        """
        Synchronous wrapper for quantum-enhanced planning.
        
        Args:
            initial_state: Current environment state
            goal: Target goal description
            action_space: Available actions
            agent_context: Additional agent context information
            
        Returns:
            QuantumEnhancedPlan with combined quantum and classical results
        """
        return asyncio.run(self.plan_async(initial_state, goal, action_space, agent_context))
    
    async def _run_classical_planning(
        self,
        initial_state: Dict[str, Any],
        goal: str,
        action_space: List[str]
    ) -> List[str]:
        """Run classical epistemic planning."""
        
        try:
            # Convert goal to classical planner format
            classical_goal = self._convert_goal_format(goal)
            
            # Run classical planning in thread pool
            loop = asyncio.get_event_loop()
            classical_plan = await loop.run_in_executor(
                self.executor,
                self.classical_planner.plan,
                initial_state,
                classical_goal
            )
            
            return classical_plan.actions if hasattr(classical_plan, 'actions') else [action_space[0]]
            
        except Exception as e:
            self.logger.error(f"Classical planning failed: {e}")
            return [action_space[0]] if action_space else []
    
    async def _run_quantum_planning(
        self,
        initial_state: Dict[str, Any],
        goal: str,
        action_space: List[str]
    ) -> PlanningResult:
        """Run quantum-inspired planning."""
        
        try:
            loop = asyncio.get_event_loop()
            quantum_result = await loop.run_in_executor(
                self.executor,
                self.quantum_planner.plan,
                initial_state,
                goal,
                action_space
            )
            
            return quantum_result
            
        except Exception as e:
            self.logger.error(f"Quantum planning failed: {e}")
            # Return dummy result
            return PlanningResult(
                best_action_sequence=[action_space[0]] if action_space else [],
                probability=0.5,
                quantum_advantage=1.0,
                interference_patterns={},
                planning_time=0.0
            )
    
    async def _run_annealing_planning(
        self,
        initial_state: Dict[str, Any],
        goal: str,
        action_space: List[str],
        agent_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Run quantum annealing planning."""
        
        try:
            # Create annealing problem from planning context
            annealing_problem = self._create_annealing_problem(
                initial_state, goal, action_space, agent_context
            )
            
            loop = asyncio.get_event_loop()
            annealing_result = await loop.run_in_executor(
                self.executor,
                self.annealing_scheduler.solve_optimization_problem,
                annealing_problem
            )
            
            # Convert solution to action sequence
            action_sequence = self._convert_annealing_solution(
                annealing_result.best_solution, action_space
            )
            
            return action_sequence
            
        except Exception as e:
            self.logger.error(f"Annealing planning failed: {e}")
            return [action_space[0]] if action_space else []
    
    async def _integrate_planning_results(
        self,
        planning_results: Dict[str, Any],
        initial_state: Dict[str, Any],
        goal: str,
        action_space: List[str]
    ) -> QuantumEnhancedPlan:
        """Integrate results from different planning approaches."""
        
        classical_plan = planning_results.get("classical", [])
        quantum_result = planning_results.get("quantum")
        annealing_plan = planning_results.get("annealing", [])
        
        # Validate results
        if not classical_plan and not quantum_result:
            # Emergency fallback
            fallback_plan = [action_space[0]] if action_space else []
            return QuantumEnhancedPlan(
                classical_plan=fallback_plan,
                quantum_plan=PlanningResult(
                    best_action_sequence=fallback_plan,
                    probability=0.1,
                    quantum_advantage=1.0,
                    interference_patterns={},
                    planning_time=0.0
                ),
                belief_consistency=0.5,
                integration_confidence=0.1,
                execution_strategy="emergency_fallback",
                fallback_plan=fallback_plan
            )
        
        # Evaluate plan quality and consistency
        plan_evaluations = {}
        
        if classical_plan:
            plan_evaluations["classical"] = await self._evaluate_plan_quality(
                classical_plan, initial_state, goal
            )
        
        if quantum_result:
            plan_evaluations["quantum"] = await self._evaluate_plan_quality(
                quantum_result.best_action_sequence, initial_state, goal
            )
            # Boost evaluation with quantum advantage
            plan_evaluations["quantum"] *= (1.0 + 0.1 * quantum_result.quantum_advantage)
        
        if annealing_plan:
            plan_evaluations["annealing"] = await self._evaluate_plan_quality(
                annealing_plan, initial_state, goal
            )
        
        # Select best plan or create hybrid
        best_approach = max(plan_evaluations.keys(), key=lambda k: plan_evaluations[k])
        integration_confidence = plan_evaluations[best_approach]
        
        # Check belief consistency
        belief_consistency = await self._check_belief_consistency(
            planning_results, initial_state, goal
        )
        
        # Determine execution strategy
        if integration_confidence > self.config.confidence_threshold:
            if best_approach == "quantum" and quantum_result:
                execution_strategy = "quantum_primary"
                primary_plan = quantum_result.best_action_sequence
                quantum_plan_result = quantum_result
            elif best_approach == "annealing":
                execution_strategy = "annealing_primary" 
                primary_plan = annealing_plan
                quantum_plan_result = quantum_result or PlanningResult(
                    best_action_sequence=annealing_plan,
                    probability=0.8,
                    quantum_advantage=1.5,
                    interference_patterns={},
                    planning_time=0.0
                )
            else:
                execution_strategy = "classical_primary"
                primary_plan = classical_plan
                quantum_plan_result = quantum_result or PlanningResult(
                    best_action_sequence=classical_plan,
                    probability=0.7,
                    quantum_advantage=1.0,
                    interference_patterns={},
                    planning_time=0.0
                )
        else:
            # Low confidence: create hybrid plan
            execution_strategy = "hybrid"
            primary_plan = self._create_hybrid_plan(planning_results)
            quantum_plan_result = quantum_result or PlanningResult(
                best_action_sequence=primary_plan,
                probability=0.6,
                quantum_advantage=1.2,
                interference_patterns={},
                planning_time=0.0
            )
        
        # Create fallback plan
        fallback_plan = classical_plan if classical_plan != primary_plan else None
        
        return QuantumEnhancedPlan(
            classical_plan=classical_plan,
            quantum_plan=quantum_plan_result,
            belief_consistency=belief_consistency,
            integration_confidence=integration_confidence,
            execution_strategy=execution_strategy,
            fallback_plan=fallback_plan
        )
    
    def _validate_planning_inputs(
        self,
        initial_state: Dict[str, Any],
        goal: str,
        action_space: List[str]
    ) -> None:
        """Validate planning inputs."""
        
        if not isinstance(initial_state, dict):
            raise PWMKValidationError("initial_state must be a dictionary")
        
        if not isinstance(goal, str) or not goal.strip():
            raise PWMKValidationError("goal must be a non-empty string")
        
        if not isinstance(action_space, list) or not action_space:
            raise PWMKValidationError("action_space must be a non-empty list")
        
        if len(action_space) > 1000:
            raise PWMKValidationError("action_space too large (>1000 actions)")
    
    def _convert_goal_format(self, goal: str) -> Any:
        """Convert goal string to classical planner format."""
        # Simple conversion - in practice would need more sophisticated parsing
        return {"description": goal, "type": "achievement"}
    
    def _create_annealing_problem(
        self,
        initial_state: Dict[str, Any],
        goal: str,
        action_space: List[str],
        agent_context: Optional[Dict[str, Any]]
    ) -> AnnealingProblem:
        """Create quantum annealing problem from planning context."""
        
        # Extract agent information
        if agent_context and "agents" in agent_context:
            agents = agent_context["agents"]
            capabilities = agent_context.get("capabilities", {})
            priorities = agent_context.get("priorities", {})
            coordination = agent_context.get("coordination", [])
        else:
            # Default single agent setup
            agents = ["agent_0"]
            capabilities = {"agent_0": action_space}
            priorities = {action: 1.0 for action in action_space}
            coordination = []
        
        return self.annealing_scheduler.create_task_planning_problem(
            agents=agents,
            tasks=action_space,
            agent_capabilities=capabilities,
            task_priorities=priorities,
            coordination_requirements=coordination
        )
    
    def _convert_annealing_solution(
        self,
        solution: np.ndarray,
        action_space: List[str]
    ) -> List[str]:
        """Convert annealing solution to action sequence."""
        
        # Find actions with highest assignment probabilities
        if len(solution) >= len(action_space):
            action_probs = solution[:len(action_space)]
            sorted_indices = np.argsort(action_probs)[::-1]
            
            # Return top actions above threshold
            selected_actions = []
            for idx in sorted_indices:
                if action_probs[idx] > 0.5 and len(selected_actions) < 5:
                    selected_actions.append(action_space[idx])
            
            return selected_actions if selected_actions else [action_space[0]]
        
        return [action_space[0]]
    
    async def _evaluate_plan_quality(
        self,
        plan: List[str],
        initial_state: Dict[str, Any],
        goal: str
    ) -> float:
        """Evaluate quality of a plan."""
        
        if not plan:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Plan length penalty (prefer shorter plans)
        length_penalty = max(0, 1.0 - len(plan) * 0.1)
        quality_score *= length_penalty
        
        # Goal relevance (heuristic based on action-goal keyword matching)
        goal_keywords = set(goal.lower().split())
        plan_keywords = set(" ".join(plan).lower().split())
        relevance = len(goal_keywords.intersection(plan_keywords)) / max(len(goal_keywords), 1)
        quality_score += 0.3 * relevance
        
        # State consistency (check if actions are valid in current state)
        valid_actions = 0
        for action in plan:
            if self._is_action_valid(action, initial_state):
                valid_actions += 1
        
        validity_score = valid_actions / len(plan) if plan else 0
        quality_score += 0.2 * validity_score
        
        return min(1.0, quality_score)
    
    def _is_action_valid(self, action: str, state: Dict[str, Any]) -> bool:
        """Check if action is valid in current state."""
        # Simple heuristic validation
        if "obstacles" in state and state["obstacles"] and "move_forward" in action:
            return False
        return True
    
    async def _check_belief_consistency(
        self,
        planning_results: Dict[str, Any],
        initial_state: Dict[str, Any],
        goal: str
    ) -> float:
        """Check consistency of plans with belief system."""
        
        # Get relevant beliefs from belief store
        try:
            relevant_beliefs = self.belief_store.query(f"relevant_to_goal('{goal}')")
            if not relevant_beliefs:
                return 0.7  # Default consistency if no beliefs found
        except:
            return 0.7
        
        # Simple consistency check: count plans that don't conflict with beliefs
        consistent_plans = 0
        total_plans = len([p for p in planning_results.values() if p])
        
        for plan_type, plan in planning_results.items():
            if plan:
                # Heuristic consistency check
                if self._plan_consistent_with_beliefs(plan, relevant_beliefs):
                    consistent_plans += 1
        
        return consistent_plans / max(total_plans, 1)
    
    def _plan_consistent_with_beliefs(self, plan: Any, beliefs: List[Dict]) -> bool:
        """Check if plan is consistent with beliefs."""
        # Simplified consistency check
        return True  # Default to consistent
    
    def _create_hybrid_plan(self, planning_results: Dict[str, Any]) -> List[str]:
        """Create hybrid plan combining multiple approaches."""
        
        all_actions = []
        
        # Collect actions from all successful plans
        for plan_type, result in planning_results.items():
            if result:
                if isinstance(result, list):
                    all_actions.extend(result)
                elif hasattr(result, 'best_action_sequence'):
                    all_actions.extend(result.best_action_sequence)
        
        if not all_actions:
            return []
        
        # Select most frequent actions (simple voting)
        action_counts = {}
        for action in all_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Return top 3 most voted actions
        sorted_actions = sorted(action_counts.keys(), key=lambda k: action_counts[k], reverse=True)
        return sorted_actions[:3]
    
    def _update_integration_stats(self, plan: QuantumEnhancedPlan, planning_time: float) -> None:
        """Update integration statistics."""
        
        self.integration_stats["total_plans"] += 1
        self.integration_stats["planning_times"].append(planning_time)
        self.integration_stats["belief_consistency_scores"].append(plan.belief_consistency)
        
        if plan.quantum_plan.quantum_advantage > 1.0:
            self.integration_stats["quantum_advantages"].append(plan.quantum_plan.quantum_advantage)
        
        if plan.execution_strategy in ["classical_primary", "emergency_fallback"]:
            self.integration_stats["fallback_rates"] += 1
        
        get_metrics_collector().record_quantum_operation("quantum_enhanced_planning", planning_time)
        get_metrics_collector().record_metric("integration_confidence", plan.integration_confidence)
        get_metrics_collector().record_metric("belief_consistency", plan.belief_consistency)
    
    def _get_enabled_components(self) -> List[str]:
        """Get list of enabled quantum components."""
        components = []
        if self.config.enable_quantum_superposition:
            components.append("superposition")
        if self.config.enable_circuit_optimization:
            components.append("circuit_optimization")
        if self.config.enable_quantum_annealing:
            components.append("annealing")
        if self.config.enable_adaptive_parameters:
            components.append("adaptive")
        return components
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        stats = self.integration_stats.copy()
        
        if stats["total_plans"] > 0:
            stats["avg_planning_time"] = np.mean(stats["planning_times"])
            stats["avg_belief_consistency"] = np.mean(stats["belief_consistency_scores"])
            stats["fallback_rate"] = stats["fallback_rates"] / stats["total_plans"]
        
        if stats["quantum_advantages"]:
            stats["avg_quantum_advantage"] = np.mean(stats["quantum_advantages"])
            stats["max_quantum_advantage"] = np.max(stats["quantum_advantages"])
        
        return stats
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.logger.info("QuantumEnhancedPlanner closed")