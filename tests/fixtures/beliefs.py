"""Test fixtures for belief systems and reasoning."""
import pytest
from typing import Dict, List, Any, Set, Tuple
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_predicates() -> List[str]:
    """Sample predicates for belief reasoning tests."""
    return [
        "has(X, Y)",
        "at(X, Y)", 
        "believes(X, Y)",
        "knows(X, Y)",
        "sees(X, Y)",
        "can_reach(X, Y)",
        "connected(X, Y)",
        "locked(X)",
        "open(X)",
        "agent(X)",
        "object(X)",
        "location(X)"
    ]


@pytest.fixture
def sample_facts() -> List[str]:
    """Sample facts for belief reasoning tests."""
    return [
        "agent(agent_0)",
        "agent(agent_1)",
        "agent(agent_2)",
        "object(key)",
        "object(treasure)",
        "object(door)",
        "location(room_1)",
        "location(room_2)",
        "location(room_3)",
        "has(agent_0, key)",
        "at(agent_0, room_1)",
        "at(agent_1, room_2)",
        "at(treasure, room_3)",
        "connected(room_1, room_2)",
        "connected(room_2, room_3)",
        "locked(door)",
        "sees(agent_0, agent_1)",
        "knows(agent_0, has(agent_0, key))",
        "believes(agent_1, at(treasure, room_3))"
    ]


@pytest.fixture
def sample_rules() -> List[str]:
    """Sample rules for belief reasoning tests."""
    return [
        "can_reach(X, Y) :- at(X, Z), connected(Z, Y).",
        "can_unlock(X, Y) :- has(X, key), at(X, Z), connected(Z, Y), locked(Y).",
        "visible(X, Y) :- at(X, Z), at(Y, Z).",
        "knows(X, Y) :- sees(X, Y).",
        "believes(X, Y) :- knows(X, Y).",
        "accessible(X, Y) :- can_reach(X, Y), not(locked(Y)).",
        "accessible(X, Y) :- can_reach(X, Y), can_unlock(X, Y)."
    ]


@pytest.fixture
def nested_beliefs() -> Dict[str, List[str]]:
    """Sample nested beliefs for testing higher-order theory of mind."""
    return {
        "agent_0": [
            "believes(agent_1, has(agent_2, key))",
            "believes(agent_1, believes(agent_2, at(treasure, room_3)))",
            "knows(agent_0, believes(agent_1, at(agent_0, room_1)))"
        ],
        "agent_1": [
            "believes(agent_0, knows(agent_1, has(agent_2, key)))",
            "believes(agent_2, believes(agent_0, at(treasure, room_3)))",
            "knows(agent_1, believes(agent_0, sees(agent_1, treasure)))"
        ],
        "agent_2": [
            "believes(agent_0, believes(agent_1, has(agent_2, key)))",
            "knows(agent_2, believes(agent_1, believes(agent_0, at(treasure, room_3))))"
        ]
    }


@pytest.fixture
def mock_prolog_engine() -> Mock:
    """Mock Prolog engine for testing."""
    engine = Mock()
    
    # Mock query results
    def mock_query(query_str: str) -> List[Dict[str, str]]:
        # Simple mock responses based on query patterns
        if "has(agent_0, key)" in query_str:
            return [{"success": True}]
        elif "has(X, key)" in query_str:
            return [{"X": "agent_0"}]
        elif "at(X, Y)" in query_str:
            return [
                {"X": "agent_0", "Y": "room_1"},
                {"X": "agent_1", "Y": "room_2"},
                {"X": "treasure", "Y": "room_3"}
            ]
        else:
            return []
    
    engine.query = Mock(side_effect=mock_query)
    engine.assert_fact = Mock()
    engine.retract_fact = Mock()
    engine.add_rule = Mock()
    engine.clear = Mock()
    engine.consult = Mock()
    
    return engine


@pytest.fixture
def mock_belief_store() -> Mock:
    """Mock belief store for testing."""
    store = Mock()
    store.beliefs = {}
    store.predicates = set()
    
    def mock_add_belief(agent: str, belief: str):
        if agent not in store.beliefs:
            store.beliefs[agent] = []
        store.beliefs[agent].append(belief)
    
    def mock_get_beliefs(agent: str) -> List[str]:
        return store.beliefs.get(agent, [])
    
    def mock_query_beliefs(query: str) -> List[Dict[str, Any]]:
        # Simple pattern matching for common queries
        if "has(" in query:
            return [{"agent": "agent_0", "object": "key"}]
        elif "at(" in query:
            return [
                {"agent": "agent_0", "location": "room_1"},
                {"agent": "agent_1", "location": "room_2"}
            ]
        else:
            return []
    
    store.add_belief = Mock(side_effect=mock_add_belief)
    store.get_beliefs = Mock(side_effect=mock_get_beliefs)
    store.query = Mock(side_effect=mock_query_beliefs)
    store.clear = Mock()
    store.update_beliefs = Mock()
    
    return store


@pytest.fixture
def belief_update_scenarios() -> List[Dict[str, Any]]:
    """Scenarios for testing belief updates."""
    return [
        {
            "name": "direct_observation",
            "initial_beliefs": ["at(agent_0, room_1)"],
            "observation": "sees(agent_0, treasure)",
            "expected_new_beliefs": ["at(treasure, room_1)", "knows(agent_0, at(treasure, room_1))"],
            "expected_retractions": []
        },
        {
            "name": "belief_revision",
            "initial_beliefs": ["believes(agent_0, at(treasure, room_2))"],
            "observation": "sees(agent_0, at(treasure, room_3))",
            "expected_new_beliefs": ["at(treasure, room_3)", "knows(agent_0, at(treasure, room_3))"],
            "expected_retractions": ["believes(agent_0, at(treasure, room_2))"]
        },
        {
            "name": "nested_belief_update",
            "initial_beliefs": ["believes(agent_1, has(agent_2, key))"],
            "observation": "sees(agent_0, not(has(agent_2, key)))",
            "expected_new_beliefs": ["knows(agent_0, not(has(agent_2, key)))"],
            "expected_retractions": []
        }
    ]


@pytest.fixture
def epistemic_queries() -> List[Dict[str, Any]]:
    """Sample epistemic queries for testing."""
    return [
        {
            "query": "believes(agent_0, has(agent_1, key))",
            "expected_result": True,
            "description": "First-order belief query"
        },
        {
            "query": "believes(agent_0, believes(agent_1, at(treasure, room_3)))",
            "expected_result": True,
            "description": "Second-order belief query"
        },
        {
            "query": "knows(agent_0, believes(agent_1, believes(agent_2, has(agent_0, key))))",
            "expected_result": False,
            "description": "Third-order belief query"
        },
        {
            "query": "common_knowledge(has(agent_0, key))",
            "expected_result": False,
            "description": "Common knowledge query"
        }
    ]


@pytest.fixture
def belief_consistency_tests() -> List[Dict[str, Any]]:
    """Test cases for belief consistency checking."""
    return [
        {
            "name": "consistent_beliefs",
            "beliefs": [
                "has(agent_0, key)",
                "at(agent_0, room_1)",
                "believes(agent_1, has(agent_0, key))"
            ],
            "expected_consistent": True
        },
        {
            "name": "contradictory_beliefs",
            "beliefs": [
                "has(agent_0, key)",
                "not(has(agent_0, key))"
            ],
            "expected_consistent": False
        },
        {
            "name": "inconsistent_nested_beliefs",
            "beliefs": [
                "knows(agent_0, has(agent_1, key))",
                "believes(agent_0, not(has(agent_1, key)))"
            ],
            "expected_consistent": False
        }
    ]


@pytest.fixture
def mock_epistemic_reasoner() -> Mock:
    """Mock epistemic reasoner for testing."""
    reasoner = Mock()
    
    def mock_reason(beliefs: List[str], query: str) -> Dict[str, Any]:
        return {
            "result": True,
            "confidence": 0.85,
            "proof": ["step1", "step2", "step3"],
            "assumptions": ["closed_world", "unique_names"]
        }
    
    reasoner.reason = Mock(side_effect=mock_reason)
    reasoner.check_consistency = Mock(return_value=True)
    reasoner.infer_beliefs = Mock(return_value=["inferred_belief_1", "inferred_belief_2"])
    reasoner.explain_reasoning = Mock(return_value="explanation")
    
    return reasoner


@pytest.fixture
def tom_agent_configs() -> Dict[str, Dict[str, Any]]:
    """Configurations for Theory of Mind agents."""
    return {
        "basic": {
            "tom_depth": 1,
            "belief_update_rate": 1.0,
            "reasoning_steps": 10,
            "memory_size": 100
        },
        "advanced": {
            "tom_depth": 3,
            "belief_update_rate": 0.8,
            "reasoning_steps": 20,
            "memory_size": 500,
            "use_attention": True,
            "attention_heads": 4
        },
        "minimal": {
            "tom_depth": 0,
            "belief_update_rate": 1.0,
            "reasoning_steps": 5,
            "memory_size": 50
        }
    }


@pytest.fixture
def belief_graph_data() -> Dict[str, Any]:
    """Sample belief graph data for testing graph-based reasoning."""
    return {
        "nodes": [
            {"id": "agent_0", "type": "agent"},
            {"id": "agent_1", "type": "agent"},
            {"id": "key", "type": "object"},
            {"id": "treasure", "type": "object"},
            {"id": "room_1", "type": "location"},
            {"id": "room_2", "type": "location"}
        ],
        "edges": [
            {"source": "agent_0", "target": "key", "relation": "has", "belief_holder": "agent_0"},
            {"source": "agent_1", "target": "room_2", "relation": "at", "belief_holder": "agent_1"},
            {"source": "treasure", "target": "room_1", "relation": "at", "belief_holder": "agent_0"},
            {"source": "agent_0", "target": "agent_1", "relation": "believes_has_key", "belief_holder": "agent_0"}
        ],
        "belief_states": {
            "agent_0": ["has(agent_0, key)", "at(treasure, room_1)", "believes(agent_1, at(agent_1, room_2))"],
            "agent_1": ["at(agent_1, room_2)", "believes(agent_0, has(agent_0, key))"]
        }
    }