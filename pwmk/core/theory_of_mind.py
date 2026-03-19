"""
TheoryOfMindModel: Agent A maintains a model of Agent B's mental state.

Theory of Mind (ToM) is the cognitive ability to attribute mental states
(beliefs, desires, intentions) to others and to understand that those
mental states may differ from one's own.

This implementation supports:
- **First-order ToM**: "What does Agent B believe about proposition X?"
- **Second-order ToM**: "What does Agent B believe that I (Agent A) believe
  about proposition X?"

Both levels are tracked as probability distributions (BeliefState objects)
so they can evolve continuously over time.

Reference
---------
Premack & Woodruff (1978), "Does the chimpanzee have a theory of mind?"
Baron-Cohen et al. (1985), "Does the autistic child have a 'theory of mind'?"
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from pwmk.core.beliefs import BeliefState


class TheoryOfMindModel:
    """
    Maintains ToM models of other agents.

    Agent *owner_id* holds:
    - first-order beliefs:  ``owner_id`` thinks ``other_agent`` believes P(X)
    - second-order beliefs: ``owner_id`` thinks ``other_agent`` thinks
      ``owner_id`` believes P(X)

    Parameters
    ----------
    owner_id : str
        The agent that *holds* this ToM model.
    prior : float
        Prior probability for unknown propositions (default 0.5).
    """

    def __init__(self, owner_id: str, prior: float = 0.5) -> None:
        self.owner_id = owner_id
        self._prior = prior

        # first_order[other_agent_id] → BeliefState representing what
        # owner thinks other_agent believes
        self._first_order: Dict[str, BeliefState] = {}

        # second_order[other_agent_id] → BeliefState representing what
        # owner thinks other_agent thinks owner believes
        self._second_order: Dict[str, BeliefState] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_first(self, agent_id: str) -> BeliefState:
        if agent_id not in self._first_order:
            self._first_order[agent_id] = BeliefState(
                f"{self.owner_id}_believes_{agent_id}_believes",
                prior=self._prior,
            )
        return self._first_order[agent_id]

    def _get_or_create_second(self, agent_id: str) -> BeliefState:
        if agent_id not in self._second_order:
            self._second_order[agent_id] = BeliefState(
                f"{self.owner_id}_believes_{agent_id}_believes_{self.owner_id}_believes",
                prior=self._prior,
            )
        return self._second_order[agent_id]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update_belief_about_other(
        self,
        agent_id: str,
        proposition: str,
        probability: float,
        order: int = 1,
    ) -> None:
        """
        Update the ToM model for *agent_id* at the given *order*.

        Parameters
        ----------
        agent_id : str
            The other agent being modelled.
        proposition : str
            The world proposition.
        probability : float
            The probability we ascribe to the other agent holding that belief.
        order : int
            1 = first-order ("what B believes"),
            2 = second-order ("what B thinks I believe").
        """
        if order == 1:
            self._get_or_create_first(agent_id).update(proposition, probability)
        elif order == 2:
            self._get_or_create_second(agent_id).update(proposition, probability)
        else:
            raise ValueError(f"order must be 1 or 2, got {order}")

    def get_belief_about_other(
        self,
        agent_id: str,
        proposition: str,
        order: int = 1,
    ) -> float:
        """
        Return the probability we ascribe to *agent_id* believing
        *proposition* is True.

        Parameters
        ----------
        agent_id : str
            The other agent.
        proposition : str
            World proposition.
        order : int
            1 for first-order, 2 for second-order.

        Returns
        -------
        float
            Probability in [0, 1].  Returns prior (0.5) for unknown agents
            or unknown propositions.
        """
        if order == 1:
            if agent_id not in self._first_order:
                return self._prior
            return self._first_order[agent_id].get(proposition)
        elif order == 2:
            if agent_id not in self._second_order:
                return self._prior
            return self._second_order[agent_id].get(proposition)
        else:
            raise ValueError(f"order must be 1 or 2, got {order}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def tracked_agents(self) -> list[str]:
        """Return all agent IDs currently being modelled."""
        return list(
            set(self._first_order.keys()) | set(self._second_order.keys())
        )

    def get_first_order_state(self, agent_id: str) -> Optional[BeliefState]:
        """Return the first-order BeliefState for *agent_id*, or None."""
        return self._first_order.get(agent_id)

    def get_second_order_state(self, agent_id: str) -> Optional[BeliefState]:
        """Return the second-order BeliefState for *agent_id*, or None."""
        return self._second_order.get(agent_id)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return a nested dict summarising all ToM beliefs.

        Returns
        -------
        dict
            ``{agent_id: {"first_order": {...}, "second_order": {...}}}``
        """
        result: Dict[str, Dict[str, float]] = {}
        all_agents = self.tracked_agents()
        for agent_id in all_agents:
            result[agent_id] = {}
            if agent_id in self._first_order:
                result[agent_id]["first_order"] = (
                    self._first_order[agent_id].to_dict()
                )
            if agent_id in self._second_order:
                result[agent_id]["second_order"] = (
                    self._second_order[agent_id].to_dict()
                )
        return result

    def __repr__(self) -> str:
        return (
            f"TheoryOfMindModel(owner={self.owner_id!r}, "
            f"tracked_agents={self.tracked_agents()})"
        )
