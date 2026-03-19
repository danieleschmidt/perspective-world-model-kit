"""
MultiAgentSimulator: Orchestrates multiple agents with different world
views, tracks how beliefs evolve over time, and measures belief divergence.

Each agent holds:
- A :class:`WorldModel` for their own beliefs.
- A :class:`TheoryOfMindModel` for beliefs about other agents.

At every time step each agent can receive a (possibly different) set of
observations.  After updating, the simulator computes pairwise *belief
divergence* using the Jensen–Shannon divergence (JSD), which is symmetric
and bounded to [0, 1] (when using base-2 logarithm).

Jensen–Shannon Divergence
-------------------------
For two probability distributions P and Q over the same support:
  JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = 0.5(P + Q)

For Bernoulli(p) vs Bernoulli(q) with M = 0.5*(p+q):
  JSD = 0.5 * [p log2(p/m) + (1-p) log2((1-p)/(1-m))]
      + 0.5 * [q log2(q/m) + (1-q) log2((1-q)/(1-m))]

where we treat 0 * log(0) = 0.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Tuple

from pwmk.core.beliefs import BeliefState
from pwmk.core.theory_of_mind import TheoryOfMindModel
from pwmk.core.world_model import WorldModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_xlogx(x: float) -> float:
    """x * log2(x), treating 0 * log(0) = 0."""
    if x <= 0.0:
        return 0.0
    return x * math.log2(x)


def _jsd_bernoulli(p: float, q: float) -> float:
    """Jensen–Shannon divergence between Bernoulli(p) and Bernoulli(q)."""
    m = 0.5 * (p + q)
    m1 = 0.5 * ((1 - p) + (1 - q))

    kl_p = _safe_xlogx(p) - _safe_xlogx(m) if m > 0 else 0.0
    kl_p1 = _safe_xlogx(1 - p) - _safe_xlogx(m1) if m1 > 0 else 0.0
    kl_q = _safe_xlogx(q) - _safe_xlogx(m) if m > 0 else 0.0
    kl_q1 = _safe_xlogx(1 - q) - _safe_xlogx(m1) if m1 > 0 else 0.0

    return 0.5 * (kl_p + kl_p1) + 0.5 * (kl_q + kl_q1)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultiAgentSimulator:
    """
    Simulate multiple agents with heterogeneous world views.

    Parameters
    ----------
    agent_ids : list[str]
        Names of the agents participating in the simulation.
    prior : float
        Default prior probability for all agents (default 0.5).

    Attributes
    ----------
    world_models : dict[str, WorldModel]
        Per-agent Bayesian world model.
    tom_models : dict[str, TheoryOfMindModel]
        Per-agent Theory of Mind model (models other agents).
    history : list[dict]
        Snapshot of beliefs and divergence after every call to :meth:`step`.
    """

    def __init__(self, agent_ids: List[str], prior: float = 0.5) -> None:
        if len(agent_ids) < 2:
            raise ValueError("At least 2 agents are required.")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("Agent IDs must be unique.")

        self.agent_ids = list(agent_ids)
        self._prior = prior

        self.world_models: Dict[str, WorldModel] = {
            aid: WorldModel(aid, prior=prior) for aid in agent_ids
        }
        self.tom_models: Dict[str, TheoryOfMindModel] = {
            aid: TheoryOfMindModel(aid, prior=prior) for aid in agent_ids
        }

        self.history: List[Dict] = []
        self._step: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(
        self,
        observations_per_agent: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        observations_per_agent : dict
            Mapping ``agent_id → {proposition: likelihood}``.
            An agent can observe zero or more propositions.
            ``likelihood`` is P(observation | proposition=True) ∈ (0, 1).

        Returns
        -------
        dict
            Pairwise belief divergences for all proposition pairs at this
            time step (see :meth:`get_belief_divergence`).
        """
        # 1. Each agent updates its own world model
        for agent_id, obs in observations_per_agent.items():
            if agent_id not in self.world_models:
                raise ValueError(f"Unknown agent_id: {agent_id!r}")
            wm = self.world_models[agent_id]
            for proposition, likelihood in obs.items():
                wm.observe(proposition, likelihood)

        # 2. Each agent updates its ToM model of others based on observable
        #    behaviour (we use a simple heuristic: if agent B observes
        #    proposition X with high likelihood, agent A infers that B's
        #    belief in X is now higher).
        for observer_id, obs in observations_per_agent.items():
            for other_id in self.agent_ids:
                if other_id == observer_id:
                    continue
                tom = self.tom_models[observer_id]
                if other_id in observations_per_agent:
                    other_obs = observations_per_agent[other_id]
                    for proposition, likelihood in other_obs.items():
                        # First-order: what does other_id believe?
                        # We infer it from the observation likelihood.
                        # A rough but principled proxy: their posterior.
                        other_posterior = self.world_models[other_id].get_posterior(
                            proposition
                        )
                        tom.update_belief_about_other(
                            other_id, proposition, other_posterior, order=1
                        )
                        # Second-order: what does other_id think we believe?
                        # Use our own posterior as a proxy for what's
                        # mutually inferable.
                        own_posterior = self.world_models[observer_id].get_posterior(
                            proposition
                        )
                        tom.update_belief_about_other(
                            other_id, proposition, own_posterior, order=2
                        )

        # 3. Record history
        divergence = self.get_belief_divergence()
        snapshot = {
            "step": self._step,
            "beliefs": {
                aid: self.world_models[aid].belief_state.to_dict()
                for aid in self.agent_ids
            },
            "divergence": divergence,
            "tom_first_order": {
                aid: {
                    other: self.tom_models[aid].get_first_order_state(other).to_dict()
                    if self.tom_models[aid].get_first_order_state(other)
                    else {}
                    for other in self.agent_ids
                    if other != aid
                }
                for aid in self.agent_ids
            },
        }
        self.history.append(snapshot)
        self._step += 1
        return divergence

    def get_belief_divergence(self) -> Dict[str, float]:
        """
        Compute pairwise Jensen–Shannon divergence between all agent pairs
        for every proposition that at least one agent has an explicit belief
        about.

        Returns
        -------
        dict
            Keys are ``"{agent_i}_{agent_j}_{proposition}"`` strings;
            values are JSD ∈ [0, 1].  A value of 0 means the two agents
            agree perfectly; 1 means maximum disagreement.
        """
        # Gather all propositions with explicit beliefs
        all_propositions: set[str] = set()
        for aid in self.agent_ids:
            all_propositions.update(
                self.world_models[aid].belief_state.known_propositions()
            )

        divergences: Dict[str, float] = {}
        n = len(self.agent_ids)
        for i in range(n):
            for j in range(i + 1, n):
                ai = self.agent_ids[i]
                aj = self.agent_ids[j]
                for prop in sorted(all_propositions):
                    p = self.world_models[ai].get_posterior(prop)
                    q = self.world_models[aj].get_posterior(prop)
                    key = f"{ai}_{aj}_{prop}"
                    divergences[key] = _jsd_bernoulli(p, q)

        return divergences

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        """Current simulation time step."""
        return self._step

    def belief_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return current beliefs for all agents as a nested dict."""
        return {
            aid: self.world_models[aid].belief_state.to_dict()
            for aid in self.agent_ids
        }

    def __repr__(self) -> str:
        return (
            f"MultiAgentSimulator(agents={self.agent_ids}, "
            f"step={self._step})"
        )
