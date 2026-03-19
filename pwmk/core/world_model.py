"""
WorldModel: Integrates observations and maintains a Bayesian belief state
over world propositions.

Bayesian Update
---------------
Given:
  - prior P(proposition)
  - observation likelihood P(obs | proposition=True) = likelihood

We update:
  P(proposition | obs) ∝ likelihood × P(proposition)

The complementary term for the False branch is (1 - likelihood) × (1 - P(proposition)).
After normalisation the posterior is:

    P(proposition | obs) = (likelihood × p) / (likelihood × p + (1-likelihood) × (1-p))

This assumes a binary world where the observation is equally likely to be
made whether the proposition is true (with probability `likelihood`) or false
(with probability `1 - likelihood`).  It is a simple but analytically clean
model appropriate for research demonstrations.
"""

from __future__ import annotations

from pwmk.core.beliefs import BeliefState


class WorldModel:
    """
    Bayesian world model for a single agent.

    The agent maintains a :class:`BeliefState` and updates it each time
    a new observation arrives.

    Parameters
    ----------
    agent_id : str
        Identifier of the owning agent.
    prior : float
        Default prior probability for unknown propositions (default 0.5).
    """

    def __init__(self, agent_id: str, prior: float = 0.5) -> None:
        self.agent_id = agent_id
        self._beliefs = BeliefState(agent_id, prior=prior)
        self._observation_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe(self, proposition: str, likelihood: float) -> float:
        """
        Incorporate a new observation and return the updated posterior.

        Parameters
        ----------
        proposition : str
            The world proposition the observation is about.
        likelihood : float
            P(observation | proposition=True).  Must be in (0, 1).
            Values near 1.0 mean "observing this strongly supports the
            proposition being True"; values near 0.0 mean the observation
            is evidence *against* it.

        Returns
        -------
        float
            Updated posterior probability P(proposition | observation).
        """
        if not 0.0 < likelihood < 1.0:
            raise ValueError(
                f"likelihood must be strictly in (0, 1), got {likelihood}"
            )

        p_prior = self._beliefs.get(proposition)
        p_not_prior = 1.0 - p_prior

        # Unnormalised posteriors
        support_true = likelihood * p_prior
        support_false = (1.0 - likelihood) * p_not_prior

        normaliser = support_true + support_false
        if normaliser == 0.0:
            posterior = 0.5
        else:
            posterior = support_true / normaliser

        self._beliefs.update(proposition, posterior)
        self._observation_count += 1
        return posterior

    def get_posterior(self, proposition: str) -> float:
        """Return the current posterior probability of *proposition*."""
        return self._beliefs.get(proposition)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def belief_state(self) -> BeliefState:
        """The underlying :class:`BeliefState` object."""
        return self._beliefs

    @property
    def observation_count(self) -> int:
        """Total number of observations processed."""
        return self._observation_count

    def __repr__(self) -> str:
        return (
            f"WorldModel(agent_id={self.agent_id!r}, "
            f"observations={self._observation_count}, "
            f"propositions={len(self._beliefs.known_propositions())})"
        )
