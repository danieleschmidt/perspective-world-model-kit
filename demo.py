#!/usr/bin/env python3
"""
Theory of Mind Demo: 3-Agent Scenario
======================================

Scenario
--------
Three agents are operating in a shared environment where a key item is hidden
in one of two locations: either the *red box* or the *blue box*.

- **Agent A** has direct access to a sensor and *knows* the key is in the red
  box (strong private information).
- **Agent B** cannot see inside the boxes but can observe Agent A's movements
  and tries to infer where the key is.
- **Agent C** has only noisy environmental observations and forms its own
  independent belief.

We run 8 time steps and print how beliefs evolve, how Agent B's Theory of
Mind model of Agent A develops, and how divergence between agents changes.
"""

from pwmk import BeliefState, WorldModel, TheoryOfMindModel, MultiAgentSimulator


PROPOSITION = "key_in_red_box"


def print_header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step: int, beliefs: dict, divergence: dict, tom_a: TheoryOfMindModel) -> None:
    print(f"\n--- Step {step} ---")
    for agent_id, bs in beliefs.items():
        p = bs.get(PROPOSITION)
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        print(f"  {agent_id:8s}  P({PROPOSITION}) = {p:.3f}  [{bar}]")

    # Show Agent B's ToM model of Agent A (first-order)
    b_thinks_a = tom_a.get_belief_about_other("agent_a", PROPOSITION, order=1)
    b_thinks_a_thinks_b = tom_a.get_belief_about_other("agent_a", PROPOSITION, order=2)
    print(
        f"\n  [ToM] B thinks A believes key_in_red_box = {b_thinks_a:.3f}"
        f"  |  B thinks A thinks B believes = {b_thinks_a_thinks_b:.3f}"
    )

    # Print pairwise divergences for our proposition
    relevant = {k: v for k, v in divergence.items() if PROPOSITION in k}
    if relevant:
        print("  [JSD]", "  ".join(f"{k.split('_'+PROPOSITION)[0]}={v:.4f}" for k, v in relevant.items()))


def main() -> None:
    print_header("Perspective World Model Kit — Theory of Mind Demo")

    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------
    agents = ["agent_a", "agent_b", "agent_c"]
    sim = MultiAgentSimulator(agents, prior=0.5)

    # Agent B's ToM model — we'll track it separately for detailed output
    tom_b = sim.tom_models["agent_b"]

    print(f"\nScenario: '{PROPOSITION}'")
    print("  Agent A: Private sensor — key IS in the red box (high-confidence obs)")
    print("  Agent B: Observes A's behaviour, infers belief indirectly")
    print("  Agent C: Noisy environment sensor, independent observations\n")

    # -------------------------------------------------------------------
    # Observation schedule:
    #
    # Agent A: consistently observes strong evidence for the proposition
    # Agent B: initially uncertain, gradually gets indirect signal
    # Agent C: sees mixed evidence (noise)
    # -------------------------------------------------------------------
    observation_schedule = [
        # Step 0: A has strong sensor hit; B and C have no info
        {
            "agent_a": {PROPOSITION: 0.95},
            "agent_b": {},
            "agent_c": {PROPOSITION: 0.55},
        },
        # Step 1: A re-confirms; B picks up a weak indirect signal
        {
            "agent_a": {PROPOSITION: 0.92},
            "agent_b": {PROPOSITION: 0.60},
            "agent_c": {PROPOSITION: 0.45},
        },
        # Step 2: B strengthens inference; C sees conflicting signal
        {
            "agent_a": {PROPOSITION: 0.90},
            "agent_b": {PROPOSITION: 0.68},
            "agent_c": {PROPOSITION: 0.40},
        },
        # Step 3: B gets clearer indirect evidence
        {
            "agent_a": {PROPOSITION: 0.93},
            "agent_b": {PROPOSITION: 0.75},
            "agent_c": {PROPOSITION: 0.52},
        },
        # Step 4: B becomes fairly confident; C remains uncertain
        {
            "agent_a": {PROPOSITION: 0.91},
            "agent_b": {PROPOSITION: 0.80},
            "agent_c": {PROPOSITION: 0.48},
        },
        # Step 5: B continues to converge
        {
            "agent_a": {PROPOSITION: 0.94},
            "agent_b": {PROPOSITION: 0.84},
            "agent_c": {PROPOSITION: 0.55},
        },
        # Step 6: B now has strong belief
        {
            "agent_a": {PROPOSITION: 0.95},
            "agent_b": {PROPOSITION: 0.88},
            "agent_c": {PROPOSITION: 0.50},
        },
        # Step 7: Final step — beliefs stabilise
        {
            "agent_a": {PROPOSITION: 0.96},
            "agent_b": {PROPOSITION: 0.90},
            "agent_c": {PROPOSITION: 0.53},
        },
    ]

    # -------------------------------------------------------------------
    # Run simulation
    # -------------------------------------------------------------------
    print_header("Simulation — Belief Evolution over 8 Steps")

    # Cache belief states for printing (WorldModel holds them)
    belief_states = {aid: sim.world_models[aid].belief_state for aid in agents}

    for t, obs in enumerate(observation_schedule):
        divergence = sim.step(obs)
        print_step(t, belief_states, divergence, tom_b)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print_header("Final Beliefs")
    for agent_id in agents:
        p = sim.world_models[agent_id].get_posterior(PROPOSITION)
        print(f"  {agent_id}: P(key_in_red_box) = {p:.4f}")

    print_header("Theory of Mind Summary (Agent B's model of Agent A)")
    tom_summary = tom_b.summary()
    if "agent_a" in tom_summary:
        fo = tom_summary["agent_a"].get("first_order", {})
        so = tom_summary["agent_a"].get("second_order", {})
        print(f"  First-order  (B thinks A believes):       {fo}")
        print(f"  Second-order (B thinks A thinks B believes): {so}")

    print_header("Belief Divergence Trajectory")
    print(f"  {'Step':>4}  {'A↔B (JSD)':>12}  {'A↔C (JSD)':>12}  {'B↔C (JSD)':>12}")
    for snap in sim.history:
        s = snap["step"]
        div = snap["divergence"]
        ab_key = f"agent_a_agent_b_{PROPOSITION}"
        ac_key = f"agent_a_agent_c_{PROPOSITION}"
        bc_key = f"agent_b_agent_c_{PROPOSITION}"
        print(
            f"  {s:>4}  "
            f"{div.get(ab_key, 0.0):>12.5f}  "
            f"{div.get(ac_key, 0.0):>12.5f}  "
            f"{div.get(bc_key, 0.0):>12.5f}"
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
