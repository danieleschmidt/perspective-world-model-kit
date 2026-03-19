# Perspective World Model Kit (PWMK)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

Neuro-symbolic world models with **Theory of Mind** belief tracking for
multi-agent systems.  Pure NumPy/SciPy — no deep-learning framework required.

---

## What Is Theory of Mind?

**Theory of Mind (ToM)** is the cognitive capacity to attribute mental states —
beliefs, desires, intentions — to others, and to understand that those states
may differ from one's own.

> *"Agent A can model what Agent B believes, and can even model what Agent B
> thinks Agent A believes."*

In multi-agent AI this matters because rational, cooperative or adversarial
behaviour all depend on agents reasoning about the epistemic state of peers.
A robot that knows *you* don't know where the tool is will behave differently
from one that assumes shared omniscience.

This library provides a clean, research-ready implementation of hierarchical
ToM belief tracking built on Bayesian probability.

---

## Architecture

```
pwmk/
└── core/
    ├── beliefs.py        – BeliefState  (probability distributions over propositions)
    ├── world_model.py    – WorldModel   (Bayesian belief update from observations)
    ├── theory_of_mind.py – TheoryOfMindModel  (first- and second-order ToM)
    └── simulator.py      – MultiAgentSimulator (orchestration + divergence tracking)
```

### Core Components

#### `BeliefState`
Represents an agent's beliefs as a mapping from *propositions* (strings) to
probabilities in \[0, 1\].

```python
from pwmk import BeliefState

bs = BeliefState("agent_a")
bs.update("key_in_red_box", 0.85)   # agent strongly believes it
bs.get("key_in_red_box")             # → 0.85
bs.get("door_open")                  # → 0.5  (uniform prior for unknown props)
bs.to_dict()                         # → {"key_in_red_box": 0.85}
```

#### `WorldModel`
Wraps a `BeliefState` and updates it via **Bayesian inference** each time an
observation arrives.

```
P(prop | obs) ∝ P(obs | prop=True) × P(prop)
```

```python
from pwmk import WorldModel

wm = WorldModel("agent_a")
wm.observe("key_in_red_box", likelihood=0.9)   # strong evidence → posterior rises
wm.get_posterior("key_in_red_box")             # e.g. → 0.818
```

*`likelihood`* = P(observation | proposition is True).  Repeated
high-likelihood observations drive the posterior toward 1; repeated
low-likelihood observations drive it toward 0.

#### `TheoryOfMindModel`
Agent A maintains a probabilistic model of what Agent B believes
(first-order ToM) and what Agent B thinks Agent A believes
(second-order ToM).

```python
from pwmk import TheoryOfMindModel

tom = TheoryOfMindModel("agent_a")

# A thinks B believes the key is in the red box with 0.7 probability
tom.update_belief_about_other("agent_b", "key_in_red_box", 0.7, order=1)

# A thinks B thinks A believes the key is there with 0.5 probability
tom.update_belief_about_other("agent_b", "key_in_red_box", 0.5, order=2)

tom.get_belief_about_other("agent_b", "key_in_red_box", order=1)  # → 0.7
```

#### `MultiAgentSimulator`
Orchestrates 2–*N* agents.  At each step, every agent may receive
*different* observations, updating their respective world models.  The
simulator also propagates observable behaviour into each agent's ToM models
and tracks **belief divergence** using the Jensen–Shannon divergence (JSD).

```python
from pwmk import MultiAgentSimulator

sim = MultiAgentSimulator(["agent_a", "agent_b", "agent_c"])

# Each agent sees different likelihoods for the same proposition
sim.step({
    "agent_a": {"key_in_red_box": 0.95},   # direct sensor
    "agent_b": {"key_in_red_box": 0.60},   # indirect inference
    "agent_c": {"key_in_red_box": 0.45},   # conflicting noise
})

sim.get_belief_divergence()
# → {"agent_a_agent_b_key_in_red_box": 0.123, ...}
```

---

## Belief Divergence (JSD)

The **Jensen–Shannon Divergence** measures how far apart two probability
distributions are.  For two Bernoulli beliefs p and q:

```
JSD(p ‖ q) = ½ KL(p ‖ m) + ½ KL(q ‖ m),   m = ½(p + q)
```

- JSD = 0   → agents agree perfectly
- JSD = 1   → maximum possible disagreement (one certain True, other certain False)

This symmetric, bounded measure lets us track convergence or divergence of
agent beliefs over time.

---

## Installation

```bash
git clone https://github.com/danieleschmidt/perspective-world-model-kit
cd perspective-world-model-kit
pip install numpy scipy
pip install -e .    # optional: install as package
```

**Dependencies:** `numpy`, `scipy` only.  Python 3.9+.

---

## Quick Start

```bash
~/anaconda3/bin/python3 demo.py
```

The demo runs a 3-agent scenario (8 time steps) where:
- **Agent A** has private sensor access to a hidden key's location.
- **Agent B** observes Agent A's behaviour and infers its belief indirectly.
- **Agent C** has independent noisy observations.

You'll see belief evolution, ToM updates, and divergence trajectories printed
to the terminal.

---

## Running Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/test_world_model.py -v
```

---

## Neuro-Symbolic Approach

PWMK combines **neural-style probabilistic representations** (continuous
belief distributions) with **symbolic-style structured reasoning** (explicit
propositions, Bayesian update rules, hierarchical ToM).

This contrasts with:
- *Pure neural* approaches (e.g., LSTM world models) that lack interpretable
  belief representations.
- *Pure symbolic* planners that cannot handle uncertainty gracefully.

The result is a framework amenable to:
- **Formal analysis** of ToM depth and convergence.
- **Integration with neural encoders** that map raw observations to
  proposition likelihoods.
- **Scalable multi-agent simulations** with provably correct Bayesian
  belief updates.

---

## Research Context

This toolkit targets AI researchers studying:
- **Multi-agent communication and coordination** (cooperative & adversarial)
- **Epistemic planning** (planning under uncertainty about others' beliefs)
- **Emergent social cognition** in simulated agents
- **Interpretable AI** — belief states are human-readable dictionaries

Potential publication venues: AAAI, NeurIPS, ICLR (multi-agent track).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
