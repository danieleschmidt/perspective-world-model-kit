"""
Tests for the Perspective World Model Kit core components.

Covers:
- BeliefState: probability storage, update, get, to_dict, entropy
- WorldModel: Bayesian update via observe(), get_posterior()
- TheoryOfMindModel: first-order and second-order ToM
- MultiAgentSimulator: step(), get_belief_divergence(), history
"""

import math
import pytest

from pwmk.core.beliefs import BeliefState
from pwmk.core.world_model import WorldModel
from pwmk.core.theory_of_mind import TheoryOfMindModel
from pwmk.core.simulator import MultiAgentSimulator, _jsd_bernoulli


# ===========================================================================
# BeliefState tests
# ===========================================================================

class TestBeliefState:
    def test_default_prior(self):
        bs = BeliefState("agent_x")
        assert bs.get("unknown_prop") == 0.5

    def test_custom_prior(self):
        bs = BeliefState("agent_x", prior=0.3)
        assert bs.get("novel_prop") == 0.3

    def test_update_and_get(self):
        bs = BeliefState("agent_x")
        bs.update("door_open", 0.8)
        assert bs.get("door_open") == pytest.approx(0.8)

    def test_update_overwrites(self):
        bs = BeliefState("agent_x")
        bs.update("door_open", 0.8)
        bs.update("door_open", 0.2)
        assert bs.get("door_open") == pytest.approx(0.2)

    def test_to_dict(self):
        bs = BeliefState("agent_x")
        bs.update("a", 0.7)
        bs.update("b", 0.3)
        d = bs.to_dict()
        assert d == {"a": pytest.approx(0.7), "b": pytest.approx(0.3)}

    def test_to_dict_is_copy(self):
        bs = BeliefState("agent_x")
        bs.update("a", 0.5)
        d = bs.to_dict()
        d["a"] = 0.9
        assert bs.get("a") == pytest.approx(0.5)

    def test_probability_bounds_valid(self):
        bs = BeliefState("a")
        bs.update("p", 0.0)   # boundary OK
        bs.update("p", 1.0)   # boundary OK

    def test_probability_out_of_bounds(self):
        bs = BeliefState("a")
        with pytest.raises(ValueError):
            bs.update("p", 1.1)
        with pytest.raises(ValueError):
            bs.update("p", -0.1)

    def test_entropy_maximum(self):
        """Entropy is 1 bit at p=0.5."""
        bs = BeliefState("a")
        bs.update("p", 0.5)
        assert bs.entropy("p") == pytest.approx(1.0, abs=1e-9)

    def test_entropy_zero_at_certainty(self):
        bs = BeliefState("a")
        bs.update("p", 0.0)
        assert bs.entropy("p") == pytest.approx(0.0)
        bs.update("p", 1.0)
        assert bs.entropy("p") == pytest.approx(0.0)

    def test_known_propositions(self):
        bs = BeliefState("a")
        bs.update("x", 0.5)
        bs.update("y", 0.9)
        assert set(bs.known_propositions()) == {"x", "y"}

    def test_copy_independence(self):
        bs = BeliefState("a")
        bs.update("p", 0.6)
        bs2 = bs.copy()
        bs2.update("p", 0.1)
        assert bs.get("p") == pytest.approx(0.6)

    def test_invalid_prior(self):
        with pytest.raises(ValueError):
            BeliefState("a", prior=1.5)


# ===========================================================================
# WorldModel tests
# ===========================================================================

class TestWorldModel:
    def test_initial_posterior_is_prior(self):
        wm = WorldModel("agent_x", prior=0.5)
        assert wm.get_posterior("any_prop") == pytest.approx(0.5)

    def test_bayesian_update_high_likelihood(self):
        """Strong evidence should push posterior significantly above prior."""
        wm = WorldModel("agent_x", prior=0.5)
        posterior = wm.observe("door_open", likelihood=0.9)
        assert posterior > 0.7

    def test_bayesian_update_low_likelihood(self):
        """Likelihood < 0.5 should push posterior below prior."""
        wm = WorldModel("agent_x", prior=0.5)
        posterior = wm.observe("door_open", likelihood=0.1)
        assert posterior < 0.3

    def test_multiple_observations_converge(self):
        """Repeated strong observations should push posterior near 1."""
        wm = WorldModel("agent_x", prior=0.5)
        for _ in range(10):
            p = wm.observe("key_here", likelihood=0.95)
        assert p > 0.99

    def test_repeated_weak_evidence_converges_to_zero(self):
        """Repeated evidence against should push posterior near 0."""
        wm = WorldModel("agent_x", prior=0.5)
        for _ in range(10):
            p = wm.observe("key_here", likelihood=0.05)
        assert p < 0.01

    def test_observation_count(self):
        wm = WorldModel("agent_x")
        assert wm.observation_count == 0
        wm.observe("p", 0.8)
        wm.observe("q", 0.6)
        assert wm.observation_count == 2

    def test_likelihood_boundary_raises(self):
        wm = WorldModel("a")
        with pytest.raises(ValueError):
            wm.observe("p", likelihood=0.0)
        with pytest.raises(ValueError):
            wm.observe("p", likelihood=1.0)

    def test_belief_state_accessible(self):
        wm = WorldModel("a")
        wm.observe("p", 0.8)
        assert isinstance(wm.belief_state, BeliefState)
        assert "p" in wm.belief_state.known_propositions()

    def test_get_posterior_matches_observe_return(self):
        wm = WorldModel("a")
        p1 = wm.observe("prop", 0.75)
        p2 = wm.get_posterior("prop")
        assert p1 == pytest.approx(p2)

    def test_custom_prior(self):
        wm = WorldModel("a", prior=0.8)
        # With strong prior (0.8) and moderate likelihood (0.7):
        # posterior should still be > 0.5
        p = wm.observe("x", 0.7)
        assert p > 0.8  # Bayesian update raises a high prior further


# ===========================================================================
# TheoryOfMindModel tests
# ===========================================================================

class TestTheoryOfMindModel:
    def test_default_prior_returned_for_unknown(self):
        tom = TheoryOfMindModel("agent_a")
        assert tom.get_belief_about_other("agent_b", "prop_x") == pytest.approx(0.5)

    def test_update_and_get_first_order(self):
        tom = TheoryOfMindModel("agent_a")
        tom.update_belief_about_other("agent_b", "key_hidden", 0.75, order=1)
        assert tom.get_belief_about_other("agent_b", "key_hidden", order=1) == pytest.approx(0.75)

    def test_update_and_get_second_order(self):
        tom = TheoryOfMindModel("agent_a")
        tom.update_belief_about_other("agent_b", "key_hidden", 0.6, order=2)
        assert tom.get_belief_about_other("agent_b", "key_hidden", order=2) == pytest.approx(0.6)

    def test_first_and_second_order_independent(self):
        tom = TheoryOfMindModel("agent_a")
        tom.update_belief_about_other("agent_b", "prop", 0.8, order=1)
        tom.update_belief_about_other("agent_b", "prop", 0.3, order=2)
        assert tom.get_belief_about_other("agent_b", "prop", order=1) == pytest.approx(0.8)
        assert tom.get_belief_about_other("agent_b", "prop", order=2) == pytest.approx(0.3)

    def test_multiple_agents_tracked(self):
        tom = TheoryOfMindModel("agent_a")
        tom.update_belief_about_other("agent_b", "p", 0.7)
        tom.update_belief_about_other("agent_c", "p", 0.4)
        assert set(tom.tracked_agents()) == {"agent_b", "agent_c"}

    def test_invalid_order_raises(self):
        tom = TheoryOfMindModel("a")
        with pytest.raises(ValueError):
            tom.update_belief_about_other("b", "p", 0.5, order=3)
        with pytest.raises(ValueError):
            tom.get_belief_about_other("b", "p", order=0)

    def test_summary_structure(self):
        tom = TheoryOfMindModel("agent_a")
        tom.update_belief_about_other("agent_b", "prop", 0.7, order=1)
        tom.update_belief_about_other("agent_b", "prop", 0.4, order=2)
        summary = tom.summary()
        assert "agent_b" in summary
        assert "first_order" in summary["agent_b"]
        assert "second_order" in summary["agent_b"]

    def test_get_first_order_state_returns_belief_state(self):
        tom = TheoryOfMindModel("a")
        tom.update_belief_about_other("b", "p", 0.6)
        state = tom.get_first_order_state("b")
        assert isinstance(state, BeliefState)
        assert state.get("p") == pytest.approx(0.6)

    def test_unknown_agent_returns_none(self):
        tom = TheoryOfMindModel("a")
        assert tom.get_first_order_state("nonexistent") is None
        assert tom.get_second_order_state("nonexistent") is None


# ===========================================================================
# MultiAgentSimulator tests
# ===========================================================================

class TestMultiAgentSimulator:
    def test_creation(self):
        sim = MultiAgentSimulator(["a", "b", "c"])
        assert set(sim.agent_ids) == {"a", "b", "c"}
        assert sim.current_step == 0

    def test_requires_at_least_two_agents(self):
        with pytest.raises(ValueError):
            MultiAgentSimulator(["solo"])

    def test_unique_agent_ids_required(self):
        with pytest.raises(ValueError):
            MultiAgentSimulator(["a", "a"])

    def test_step_increments(self):
        sim = MultiAgentSimulator(["a", "b"])
        sim.step({"a": {"p": 0.8}, "b": {}})
        assert sim.current_step == 1
        sim.step({"a": {}, "b": {"p": 0.6}})
        assert sim.current_step == 2

    def test_beliefs_update_after_step(self):
        sim = MultiAgentSimulator(["a", "b"])
        sim.step({"a": {"prop": 0.9}, "b": {}})
        # Agent a should now have a belief higher than prior
        p = sim.world_models["a"].get_posterior("prop")
        assert p > 0.5

    def test_divergence_zero_without_observations(self):
        """All agents at prior → divergence is 0 (identical distributions)."""
        sim = MultiAgentSimulator(["a", "b"])
        # Don't give any agent any observations; divergence should be 0
        div = sim.get_belief_divergence()
        assert div == {}  # No propositions tracked yet

    def test_divergence_nonzero_after_private_observation(self):
        """Agent a sees strong evidence; agent b sees none → high divergence."""
        sim = MultiAgentSimulator(["a", "b"])
        for _ in range(5):
            sim.step({"a": {"secret": 0.95}, "b": {}})
        div = sim.get_belief_divergence()
        key = "a_b_secret"
        assert key in div
        assert div[key] > 0.05

    def test_divergence_decreases_when_agents_converge(self):
        """If both agents see the same evidence, their beliefs should converge."""
        sim = MultiAgentSimulator(["a", "b"])
        # Give different observations initially
        sim.step({"a": {"prop": 0.9}, "b": {"prop": 0.2}})
        div_initial = sim.get_belief_divergence().get("a_b_prop", 0.0)
        # Now give both agents the same strong evidence
        for _ in range(5):
            sim.step({"a": {"prop": 0.9}, "b": {"prop": 0.9}})
        div_final = sim.get_belief_divergence().get("a_b_prop", 0.0)
        assert div_final < div_initial

    def test_history_recorded(self):
        sim = MultiAgentSimulator(["a", "b"])
        sim.step({"a": {"p": 0.8}, "b": {"p": 0.6}})
        sim.step({"a": {"p": 0.9}, "b": {"p": 0.7}})
        assert len(sim.history) == 2
        assert sim.history[0]["step"] == 0
        assert sim.history[1]["step"] == 1

    def test_history_contains_beliefs_and_divergence(self):
        sim = MultiAgentSimulator(["a", "b"])
        sim.step({"a": {"prop": 0.8}, "b": {}})
        snap = sim.history[0]
        assert "beliefs" in snap
        assert "divergence" in snap
        assert "a" in snap["beliefs"]
        assert "b" in snap["beliefs"]

    def test_unknown_agent_in_observations_raises(self):
        sim = MultiAgentSimulator(["a", "b"])
        with pytest.raises(ValueError):
            sim.step({"a": {"p": 0.8}, "unknown": {"p": 0.5}})

    def test_belief_snapshot(self):
        sim = MultiAgentSimulator(["a", "b"])
        sim.step({"a": {"x": 0.7}, "b": {"x": 0.4}})
        snap = sim.belief_snapshot()
        assert set(snap.keys()) == {"a", "b"}
        assert "x" in snap["a"]

    def test_tom_updated_after_step(self):
        """After a step, Tom model should have first-order beliefs about observed agents."""
        sim = MultiAgentSimulator(["a", "b"])
        sim.step({"a": {"p": 0.9}, "b": {"p": 0.4}})
        # Agent a's ToM should now model agent b
        tom_a = sim.tom_models["a"]
        assert "b" in tom_a.tracked_agents()


# ===========================================================================
# JSD helper tests
# ===========================================================================

class TestJSD:
    def test_identical_distributions(self):
        assert _jsd_bernoulli(0.7, 0.7) == pytest.approx(0.0, abs=1e-9)

    def test_maximum_divergence(self):
        """p=0, q=1 should give JSD=1."""
        # Using very close to 0 and 1 (actual 0/1 requires log(0)=0 convention)
        jsd = _jsd_bernoulli(0.0, 1.0)
        # p=0 → log2(0)=0 by convention; JSD should equal 1 bit
        assert jsd == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self):
        assert _jsd_bernoulli(0.3, 0.7) == pytest.approx(
            _jsd_bernoulli(0.7, 0.3), abs=1e-9
        )

    def test_bounded(self):
        for p, q in [(0.1, 0.9), (0.5, 0.8), (0.0, 0.5), (0.5, 1.0)]:
            jsd = _jsd_bernoulli(p, q)
            assert 0.0 <= jsd <= 1.0 + 1e-9
