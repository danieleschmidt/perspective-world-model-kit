"""Belief store and reasoning system."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import time

from ..security.input_sanitizer import get_sanitizer, SecurityError
from ..security.belief_validator import get_validator
from ..utils.logging import get_logger
from ..utils.monitoring import get_metrics_collector
from ..optimization.caching import get_cache_manager
from ..optimization.parallel_processing import get_parallel_processor


@dataclass
class BeliefState:
    """Represents the belief state of an agent."""
    agent_id: str
    beliefs: Dict[str, Any]
    confidence: Dict[str, float]
    timestamp: int


class BeliefStore:
    """
    Manages beliefs and provides reasoning capabilities.
    
    Supports first-order and nested belief reasoning using
    a simplified Prolog-like interface.
    """
    
    def __init__(self, backend: str = "simple"):
        self.backend = backend
        self.facts: Dict[str, List[str]] = {}
        self.rules: List[str] = []
        self.beliefs: Dict[str, BeliefState] = {}
        
        # Initialize security and monitoring
        self.sanitizer = get_sanitizer()
        self.validator = get_validator()
        self.logger = get_logger(self.__class__.__name__)
        self.metrics = get_metrics_collector()
        
        # Performance optimization
        self.cache_manager = get_cache_manager()
        self.parallel_processor = get_parallel_processor()
        self.enable_caching = True
        self.enable_parallel = True
        
        # Performance tracking
        self.query_count = 0
        self.belief_count = 0
        
    def add_belief(self, agent_id: str, belief: str) -> None:
        """Add a belief for a specific agent with security validation."""
        start_time = time.time()
        
        try:
            # Security validation
            clean_agent_id = self.sanitizer.sanitize_agent_id(agent_id)
            clean_belief = self.sanitizer.sanitize_belief_content(belief)
            
            # Belief validation
            validated_belief = self.validator.sanitize_and_validate(clean_belief, clean_agent_id)
            
            # Add to store
            if clean_agent_id not in self.facts:
                self.facts[clean_agent_id] = []
            
            # Check for duplicates
            if validated_belief not in self.facts[clean_agent_id]:
                self.facts[clean_agent_id].append(validated_belief)
                self.belief_count += 1
                
                self.logger.debug(f"Added belief for {clean_agent_id}: {validated_belief}")
            else:
                self.logger.debug(f"Duplicate belief ignored for {clean_agent_id}: {validated_belief}")
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_belief_operation("add", clean_agent_id, duration)
            
        except SecurityError as e:
            self.logger.warning(f"Security error adding belief: {e}")
            self.metrics.monitor.increment_counter("belief_add_security_errors")
            raise
        except Exception as e:
            self.logger.error(f"Error adding belief: {e}")
            self.metrics.monitor.increment_counter("belief_add_general_errors")
            raise
        
    def add_rule(self, rule: str) -> None:
        """Add a reasoning rule."""
        self.rules.append(rule)
        
    def query(self, query_str: str) -> List[Dict[str, str]]:
        """
        Query the belief store with enhanced pattern matching and security.
        
        Args:
            query_str: Query in simplified Prolog syntax
            
        Returns:
            List of variable bindings that satisfy the query
        """
        start_time = time.time()
        
        try:
            # Security validation
            clean_query = self.sanitizer.sanitize_belief_query(query_str)
            
            # Empty query check
            if not clean_query.strip():
                self.logger.debug("Empty query received")
                return []
            
            self.query_count += 1
            self.logger.debug(f"Processing query: {clean_query}")
            
            results = []
            
            # Handle nested belief queries  
            if "believes(" in clean_query:
                return self._query_nested_beliefs(clean_query)
            
            # Extract variables (uppercase identifiers)
            variables = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', clean_query)
            
            # Pattern matching against all facts
            for agent_id, facts in self.facts.items():
                for fact in facts:
                    if self._matches_pattern(fact, clean_query):
                        binding = self._extract_bindings(fact, clean_query, variables)
                        if binding:
                            binding['agent'] = agent_id  # Add agent context
                            results.append(binding)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_results = []
            for result in results:
                key = tuple(sorted(result.items()))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_belief_operation("query", "system", duration)
            self.metrics.monitor.record_metric("belief_query_results", len(unique_results))
            
            self.logger.debug(f"Query completed: {len(unique_results)} results in {duration:.4f}s")
            return unique_results
            
        except SecurityError as e:
            self.logger.warning(f"Security error in query: {e}")
            self.metrics.monitor.increment_counter("belief_query_security_errors")
            raise
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            self.metrics.monitor.increment_counter("belief_query_general_errors")
            raise
        
    def get_belief_state(self, agent_id: str) -> Optional[BeliefState]:
        """Get the current belief state for an agent."""
        if agent_id in self.beliefs:
            return self.beliefs[agent_id]
        return None
        
    def update_beliefs(
        self, 
        agent_id: str, 
        observations: Dict[str, Any]
    ) -> None:
        """Update agent beliefs based on new observations."""
        # Simple belief update mechanism
        for key, value in observations.items():
            belief = f"{key}({value})"
            self.add_belief(agent_id, belief)
            
    def _matches_pattern(self, fact: str, pattern: str) -> bool:
        """Check if a fact matches a query pattern."""
        # Simplified pattern matching
        fact_clean = re.sub(r'\([^)]*\)', '(X)', fact)
        pattern_clean = re.sub(r'\([^)]*\)', '(X)', pattern)
        return fact_clean == pattern_clean
        
    def _extract_bindings(
        self, 
        fact: str, 
        pattern: str, 
        variables: List[str]
    ) -> Optional[Dict[str, str]]:
        """Extract variable bindings from a matching fact."""
        # Simplified binding extraction
        binding = {}
        
        # Extract terms from fact and pattern
        fact_terms = re.findall(r'\(([^)]*)\)', fact)
        pattern_terms = re.findall(r'\(([^)]*)\)', pattern)
        
        if len(fact_terms) == len(pattern_terms):
            for i, (fact_term, pattern_term) in enumerate(zip(fact_terms, pattern_terms)):
                if pattern_term in variables:
                    binding[pattern_term] = fact_term
                    
        return binding if binding else None
    
    def _query_nested_beliefs(self, query_str: str) -> List[Dict[str, str]]:
        """Handle nested belief queries like believes(A, believes(B, X))."""
        results = []
        
        # Extract nested structure
        nested_match = re.search(r'believes\(([^,]+),\s*(.+)\)', query_str)
        if not nested_match:
            return results
            
        believer = nested_match.group(1).strip()
        belief_content = nested_match.group(2).strip()
        
        # Search through facts for matching beliefs
        for agent_id, facts in self.facts.items():
            if believer == agent_id or believer.isupper():  # Variable or exact match
                for fact in facts:
                    if self._matches_pattern(fact, belief_content):
                        binding = {'Believer': agent_id} if believer.isupper() else {}
                        # Extract additional variables from the belief content
                        content_vars = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', belief_content)
                        content_binding = self._extract_bindings(fact, belief_content, content_vars)
                        if content_binding:
                            binding.update(content_binding)
                            results.append(binding)
        
        return results
    
    def add_nested_belief(self, believer: str, target_agent: str, belief: str) -> None:
        """Add a nested belief (A believes that B believes X)."""
        nested_belief = f"believes({target_agent}, {belief})"
        self.add_belief(believer, nested_belief)
    
    def get_all_beliefs(self, agent_id: str) -> List[str]:
        """Get all beliefs for a specific agent."""
        return self.facts.get(agent_id, [])
    
    def clear_beliefs(self, agent_id: str) -> None:
        """Clear all beliefs for a specific agent."""
        if agent_id in self.facts:
            self.facts[agent_id] = []
            
    def belief_exists(self, agent_id: str, belief: str) -> bool:
        """Check if a specific belief exists for an agent."""
        return belief in self.facts.get(agent_id, [])
    
    def batch_query(self, queries: List[str], use_parallel: bool = None) -> List[List[Dict[str, str]]]:
        """
        Execute multiple queries in batch, optionally using parallel processing.
        
        Args:
            queries: List of query strings
            use_parallel: Override parallel processing setting
            
        Returns:
            List of query results in the same order as input queries
        """
        start_time = time.time()
        use_parallel = use_parallel if use_parallel is not None else self.enable_parallel
        
        try:
            if use_parallel and len(queries) > 2:
                # Use parallel processing for multiple queries
                belief_stores = [self] * len(queries)
                futures = self.parallel_processor.submit_belief_query_batch(belief_stores, queries)
                raw_results = self.parallel_processor.wait_for_completion(futures, timeout=30.0)
                
                # Extract actual results from task results
                results = []
                for raw_result in raw_results:
                    if raw_result and len(raw_result) == 2:
                        task_id, result = raw_result
                        results.append(result if result is not None else [])
                    else:
                        results.append([])
                        
            else:
                # Sequential processing
                results = []
                for query in queries:
                    try:
                        result = self.query(query)
                        results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Batch query failed for '{query}': {e}")
                        results.append([])
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.monitor.record_metric("batch_query_duration", duration)
            self.metrics.monitor.record_metric("batch_query_size", len(queries))
            self.metrics.monitor.increment_counter("batch_queries")
            
            self.logger.debug(
                f"Batch query completed: {len(queries)} queries in {duration:.4f}s "
                f"(parallel={use_parallel})"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch query failed: {e}")
            self.metrics.monitor.increment_counter("batch_query_failures")
            # Return empty results for all queries on failure
            return [[] for _ in queries]
    
    def batch_add_beliefs(
        self, 
        agent_beliefs: List[Tuple[str, str]], 
        use_parallel: bool = None
    ) -> List[bool]:
        """
        Add multiple beliefs in batch, optionally using parallel processing.
        
        Args:
            agent_beliefs: List of (agent_id, belief) tuples
            use_parallel: Override parallel processing setting
            
        Returns:
            List of success flags in the same order as input
        """
        start_time = time.time()
        use_parallel = use_parallel if use_parallel is not None else self.enable_parallel
        
        try:
            if use_parallel and len(agent_beliefs) > 2:
                # Use parallel processing
                belief_stores = [self] * len(agent_beliefs)
                agent_ids = [ab[0] for ab in agent_beliefs]
                beliefs = [ab[1] for ab in agent_beliefs]
                
                futures = self.parallel_processor.submit_belief_update_batch(
                    belief_stores, agent_ids, beliefs
                )
                raw_results = self.parallel_processor.wait_for_completion(futures, timeout=30.0)
                
                # Extract success flags
                results = []
                for raw_result in raw_results:
                    if raw_result and len(raw_result) == 2:
                        task_id, success = raw_result
                        results.append(success)
                    else:
                        results.append(False)
                        
            else:
                # Sequential processing
                results = []
                for agent_id, belief in agent_beliefs:
                    try:
                        self.add_belief(agent_id, belief)
                        results.append(True)
                    except Exception as e:
                        self.logger.warning(f"Batch add failed for {agent_id}: {e}")
                        results.append(False)
            
            # Record metrics
            duration = time.time() - start_time
            success_count = sum(results)
            self.metrics.monitor.record_metric("batch_add_duration", duration)
            self.metrics.monitor.record_metric("batch_add_size", len(agent_beliefs))
            self.metrics.monitor.record_metric("batch_add_success_rate", success_count / len(results))
            self.metrics.monitor.increment_counter("batch_adds")
            
            self.logger.debug(
                f"Batch add completed: {success_count}/{len(agent_beliefs)} successful "
                f"in {duration:.4f}s (parallel={use_parallel})"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch add beliefs failed: {e}")
            self.metrics.monitor.increment_counter("batch_add_failures")
            return [False] * len(agent_beliefs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance and scaling statistics."""
        return {
            "query_count": self.query_count,
            "belief_count": self.belief_count,
            "total_agents": len(self.facts),
            "total_facts": sum(len(facts) for facts in self.facts.values()),
            "caching_enabled": self.enable_caching,
            "parallel_enabled": self.enable_parallel,
            "parallel_stats": self.parallel_processor.get_stats(),
            "cache_stats": {
                "enabled": self.cache_manager.is_enabled(),
                "stats": getattr(self.cache_manager, 'get_stats', lambda: {})()
            }
        }