"""
BeliefState: Represents an agent's beliefs about world propositions
as probability distributions.

Each proposition maps to a probability in [0, 1]:
  - 0.0  → agent is certain the proposition is False
  - 1.0  → agent is certain the proposition is True
  - 0.5  → agent has no preference / maximum uncertainty

Unknown propositions default to 0.5 (uniform prior).
"""

from __future__ import annotations

from typing import Dict, Optional


class BeliefState:
    """
    Probability-distribution belief state over a set of propositions.

    Propositions are arbitrary strings (e.g. ``"door_is_open"``,
    ``"agent_b_has_key"``).

    Parameters
    ----------
    agent_id : str
        Identifier of the agent that holds these beliefs.
    prior : float
        Default probability assigned to unknown propositions (default 0.5).
    """

    def __init__(self, agent_id: str, prior: float = 0.5) -> None:
        if not 0.0 <= prior <= 1.0:
            raise ValueError(f"prior must be in [0, 1], got {prior}")
        self.agent_id = agent_id
        self._prior = prior
        self._beliefs: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, proposition: str, probability: float) -> None:
        """
        Set the probability of *proposition* being True.

        Parameters
        ----------
        proposition : str
            The proposition to update.
        probability : float
            New probability in [0, 1].
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability must be in [0, 1], got {probability}"
            )
        self._beliefs[proposition] = probability

    def get(self, proposition: str) -> float:
        """
        Return the probability of *proposition* being True.

        Returns the prior (default 0.5) for unknown propositions.
        """
        return self._beliefs.get(proposition, self._prior)

    def to_dict(self) -> Dict[str, float]:
        """Return a copy of all explicitly-set belief probabilities."""
        return dict(self._beliefs)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def known_propositions(self) -> list[str]:
        """Return all propositions with an explicit probability."""
        return list(self._beliefs.keys())

    def entropy(self, proposition: str) -> float:
        """
        Binary entropy H(p) for a single proposition.

        H = -p log2(p) - (1-p) log2(1-p)
        Returns 0 for p=0 or p=1 (certain beliefs).
        """
        import math

        p = self.get(proposition)
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def copy(self) -> "BeliefState":
        """Return a deep copy."""
        bs = BeliefState(self.agent_id, prior=self._prior)
        bs._beliefs = dict(self._beliefs)
        return bs

    def __repr__(self) -> str:
        items = ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(self._beliefs.items())
        )
        return f"BeliefState(agent_id={self.agent_id!r}, beliefs={{{items}}})"
