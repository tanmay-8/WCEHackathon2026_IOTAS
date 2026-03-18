"""
Graph Retrieval Service - Production-grade controlled retrieval.

Architecture:
- Mode-based queries (no wildcard explosion)
- Real hop distance calculation
- User-isolated traversal
- Timeline filtering
- Configurable scoring weights
- Deferred reinforcement (updated after answer generation)
"""

from typing import Dict, List, Any, Tuple, Optional
import time
import math
from datetime import datetime, timezone
from neo4j import GraphDatabase
from neo4j.time import DateTime
from config.settings import Settings
from services.graph.query_understanding import QueryUnderstanding, RetrievalMode


class GraphRetrieval:
    """
    Production-grade graph retrieval with controlled traversal.
    
    Design Principles:
    - No wildcard path explosion
    - Mode-based targeted queries
    - Real hop distance scoring
    - Strict user isolation
    - O(edges) complexity, not O(paths)
    """
    
    # Configurable scoring weights (sum = 1.0)
    SCORE_WEIGHTS = {
        "graph_distance": 0.4,
        "recency": 0.3,
        "confidence": 0.2,
        "reinforcement": 0.1
    }
    
    # Recency decay parameter
    RECENCY_DECAY_LAMBDA = 0.1  # Exponential decay rate

    
    def __init__(self):
        """Initialize Neo4j connection and query understanding."""
        try:
            self.driver = GraphDatabase.driver(
                Settings.NEO4J_URI,
                auth=(Settings.NEO4J_USER, Settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            self.driver = None
        
        self.query_understanding = QueryUnderstanding()
    
    def retrieve(
        self, 
        user_id: str, 
        query: str, 
        max_depth: int = 3
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Retrieve relevant graph context using controlled mode-based queries.
        
        Args:
            user_id: User identifier
            query: User's query text
            max_depth: Maximum hops (unused, kept for compatibility)
            
        Returns:
            Tuple of (retrieved_nodes, retrieval_time_ms)
        """
        if not self.driver:
            print("Warning: Neo4j driver not initialized, returning empty results")
            return [], 0.0
        
        start_time = time.time()
        retrieved_nodes = []
        
        try:
            with self.driver.session() as session:
                # 1. Classify query mode
                mode, recommended_depth = self.query_understanding.classify_query(query)
                depth, top_k = self._build_retrieval_plan(
                    mode=mode,
                    query_text=query,
                    requested_depth=max_depth,
                    recommended_depth=recommended_depth
                )
                print(f"Query mode: {mode.value}, depth: {depth}, top_k: {top_k}")
                
                # 2. Extract timeline filter (optional)
                start_date = self.query_understanding.extract_timeline(query)
                
                # 3. Execute mode-specific retrieval with inline hop distance calculation
                raw_nodes = self._execute_mode_based_retrieval(
                    session=session,
                    user_id=user_id,
                    mode=mode,
                    start_date=start_date,
                    depth=depth,
                    top_k=top_k
                )

                # 4. Apply scoring and ranking
                retrieved_nodes = self._score_and_rank_nodes(
                    raw_nodes, query
                )
        
        except Exception as e:
            print(f"Error during graph retrieval: {e}")
            import traceback
            traceback.print_exc()
        
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        return retrieved_nodes, retrieval_time_ms

    def _build_retrieval_plan(
        self,
        mode: RetrievalMode,
        query_text: str,
        requested_depth: int,
        recommended_depth: int
    ) -> Tuple[int, int]:
        """
        Build adaptive retrieval depth and top-k based on mode and query complexity.
        """
        token_count = len((query_text or "").split())

        depth = max(1, min(5, max(requested_depth, recommended_depth)))

        if mode == RetrievalMode.DIRECT_LOOKUP:
            depth = min(depth, 2)
            top_k = 50
        elif mode == RetrievalMode.AGGREGATION:
            depth = min(3, max(depth, 2))
            top_k = 90
        else:  # RELATIONAL_REASONING
            depth = max(3, depth)
            top_k = 130

        # Increase budget for longer, likely multi-part queries.
        if token_count >= 14:
            top_k += 20
        if token_count >= 24:
            top_k += 20

        return depth, min(top_k, 200)
    
    def _serialize_neo4j_types(self, obj: Any) -> Any:
        """Convert Neo4j types to JSON-serializable Python types."""
        if isinstance(obj, DateTime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_neo4j_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_neo4j_types(item) for item in obj]
        else:
            return obj
    
    def _execute_mode_based_retrieval(
        self,
        session,
        user_id: str,
        mode: RetrievalMode,
        start_date: Optional[datetime],
        depth: int,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute controlled retrieval based on query mode.
        
        NO WILDCARD PATHS - Each mode has specific traversal.
        """
        params = {
            "user_id": user_id,
            "top_k": top_k,
            "start_date": start_date
        }

        node_time_filter = "AND ($start_date IS NULL OR coalesce(n.timestamp, n.last_reinforced, n.created_at) >= $start_date)"
        
        if mode == RetrievalMode.DIRECT_LOOKUP:
            # Simple entity lookup (e.g., "What assets do I own?")
            query = f"""
            MATCH (u:User {{id: $user_id}})
            OPTIONAL MATCH (u)-[:OWNS_MESSAGE]->(m:Message)
            WHERE m.user_id = $user_id
            OPTIONAL MATCH (u)-[:MADE_TRANSACTION]->(t:Transaction)
            WHERE t.user_id = $user_id AND coalesce(t.superseded, false) = false
            OPTIONAL MATCH (t)-[:AFFECTS_ASSET]->(a:Asset)
            WHERE a.user_id = $user_id
            OPTIONAL MATCH (m)-[:DERIVED_FACT]->(f:Fact)
            WHERE f.user_id = $user_id
            WITH u, collect(DISTINCT m) + collect(DISTINCT t) + collect(DISTINCT a) + collect(DISTINCT f) as nodes
            UNWIND nodes as n
            WITH u, n
            WHERE n IS NOT NULL {node_time_filter}
            OPTIONAL MATCH path = shortestPath((u)-[*1..{depth}]-(n))
            WITH n, coalesce(length(path), {depth} + 1) AS hops
            WITH n, hops,
                 CASE
                     WHEN n:Fact THEN 'fact_memory'
                     WHEN n:Transaction THEN 'transaction_link'
                     WHEN n:Asset THEN 'asset_link'
                     WHEN n:Message THEN 'message_context'
                     ELSE 'direct_lookup'
                 END AS matched_by
            RETURN DISTINCT n, hops, matched_by
            ORDER BY hops ASC
            LIMIT $top_k
            """
        
        elif mode == RetrievalMode.AGGREGATION:
            # Aggregation queries (e.g., "How much have I invested?")
            query = f"""
            MATCH (u:User {{id: $user_id}})
            MATCH (u)-[:MADE_TRANSACTION]->(t:Transaction)
                        WHERE t.user_id = $user_id AND coalesce(t.superseded, false) = false
                            AND ($start_date IS NULL OR coalesce(t.timestamp, t.last_reinforced, t.created_at) >= $start_date)
            OPTIONAL MATCH (t)-[:AFFECTS_ASSET]->(a:Asset)
            WHERE a.user_id = $user_id
            OPTIONAL MATCH (f:Fact)-[:CONFIRMS]->(t)
            WHERE f.user_id = $user_id
            WITH u, collect(DISTINCT t) + collect(DISTINCT a) + collect(DISTINCT f) as nodes
            UNWIND nodes as n
                        WITH u, n
                        WHERE n IS NOT NULL {node_time_filter}
                        OPTIONAL MATCH path = shortestPath((u)-[*1..{depth}]-(n))
                        WITH n, coalesce(length(path), {depth} + 1) AS hops
                        WITH n, hops,
                                 CASE
                                         WHEN n:Transaction THEN 'aggregation_transaction'
                                         WHEN n:Fact THEN 'aggregation_fact'
                                         WHEN n:Asset THEN 'aggregation_asset'
                                         ELSE 'aggregation_context'
                                 END AS matched_by
                        RETURN DISTINCT n, hops, matched_by
                        ORDER BY hops ASC
                        LIMIT $top_k
            """
        
        elif mode == RetrievalMode.RELATIONAL_REASONING:
            # Multi-hop reasoning (e.g., "Is investment aligned with goal?")
            query = f"""
            MATCH (u:User {{id: $user_id}})
            OPTIONAL MATCH (u)-[:MADE_TRANSACTION]->(t:Transaction)-[:AFFECTS_ASSET]->(a:Asset)
                        WHERE t.user_id = $user_id AND a.user_id = $user_id
                            AND coalesce(t.superseded, false) = false
                            AND ($start_date IS NULL OR coalesce(t.timestamp, t.last_reinforced, t.created_at) >= $start_date)
            OPTIONAL MATCH (a)-[:CONTRIBUTES_TO]->(g:Goal)
            WHERE g.user_id = $user_id
            OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)
            WHERE p.user_id = $user_id
            OPTIONAL MATCH (f:Fact)-[:CONFIRMS]->(t)
            WHERE f.user_id = $user_id
            OPTIONAL MATCH (f2:Fact)-[:RELATES_TO]->(a)
            WHERE f2.user_id = $user_id
              WITH u, collect(DISTINCT t) + collect(DISTINCT a) + collect(DISTINCT g) + 
                 collect(DISTINCT p) + collect(DISTINCT f) + collect(DISTINCT f2) as nodes
            UNWIND nodes as n
              WITH u, n
              WHERE n IS NOT NULL {node_time_filter}
              OPTIONAL MATCH path = shortestPath((u)-[*1..{depth}]-(n))
              WITH n, coalesce(length(path), {depth} + 1) AS hops
              WITH n, hops,
                  CASE
                     WHEN n:Goal THEN 'goal_reasoning'
                     WHEN n:Preference THEN 'preference_reasoning'
                     WHEN n:Asset THEN 'asset_reasoning'
                     WHEN n:Fact THEN 'fact_reasoning'
                     ELSE 'relational_context'
                  END AS matched_by
              RETURN DISTINCT n, hops, matched_by
              ORDER BY hops ASC
              LIMIT $top_k
            """
        
        else:
            # Fallback to direct lookup
            query = f"""
            MATCH (u:User {{id: $user_id}})
            MATCH (u)-[:MADE_TRANSACTION]->(t:Transaction)
            WHERE t.user_id = $user_id AND coalesce(t.superseded, false) = false
            OPTIONAL MATCH path = shortestPath((u)-[*1..{depth}]-(t))
            WITH t AS n, coalesce(length(path), {depth} + 1) AS hops
            RETURN n, hops, 'fallback_transaction' AS matched_by
            ORDER BY hops ASC
            LIMIT $top_k
            """
        
        result = session.run(query, **params)
        
        # Format nodes
        nodes = []
        for record in result:
            node = record.get("n")
            if node:
                nodes.append({
                    "type": list(node.labels)[0] if node.labels else "Unknown",
                    "properties": self._serialize_neo4j_types(dict(node)),
                    "neo4j_id": node.id,
                    "hop_distance": int(record.get("hops", depth + 1)),
                    "retrieval_trace": {
                        "mode": mode.value,
                        "matched_by": record.get("matched_by", "unknown"),
                        "depth_used": depth,
                        "top_k_used": top_k,
                        "timeline_filter_applied": start_date is not None
                    }
                })
        
        print(f"[DEBUG] Mode: {mode.value}, Retrieved {len(nodes)} nodes (depth={depth}, top_k={top_k})")
        return nodes
    
    def _score_and_rank_nodes(
        self, 
        nodes: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Score and rank nodes using normalized multi-factor scoring.
        
        New Formula (all components normalized 0-1):
        score = w_graph × graph_score + 
                w_recency × recency_score + 
                w_confidence × confidence +
                w_reinforcement × reinforcement_score
        
        Where:
        - graph_score = 1 / (hops + 1)
        - recency_score = exp(-λ × days_ago)
        - confidence = node.confidence
        - reinforcement_score = min(1, log(1 + count) / log(10))
        """
        scored_nodes = []
        now = datetime.now(timezone.utc)
        
        for node in nodes:
            props = node.get("properties", {})
            node_type = node.get("type", "Unknown")
            
            # Skip User nodes from ranking
            if node_type == "User":
                continue
            
            # 1. Graph Distance Score (inverse of hops)
            hops = node.get("hop_distance", 3)
            graph_score = 1.0 / (hops + 1)
            
            # 2. Recency Score (exponential decay)
            recency_score = 0.5  # Default
            last_reinforced = props.get("last_reinforced")
            days_ago = 0.0
            if last_reinforced:
                if isinstance(last_reinforced, str):
                    try:
                        last_dt = datetime.fromisoformat(last_reinforced.replace("Z", "+00:00"))
                        days_ago = max(0.0, (now - last_dt).total_seconds() / 86400)  # Days as float
                        recency_score = math.exp(-self.RECENCY_DECAY_LAMBDA * days_ago)
                    except:
                        pass

            # Fallback to timestamp/created_at when last_reinforced is missing
            if days_ago == 0.0 and not last_reinforced:
                fallback_dt = props.get("timestamp") or props.get("created_at")
                if isinstance(fallback_dt, str):
                    try:
                        parsed_dt = datetime.fromisoformat(fallback_dt.replace("Z", "+00:00"))
                        days_ago = max(0.0, (now - parsed_dt).total_seconds() / 86400)
                    except:
                        pass
            
            # 3. Confidence Score (already normalized 0-1)
            confidence = float(props.get("confidence", 0.5))
            
            # 4. Reinforcement Score (logarithmic scaling)
            reinforcement_count = int(props.get("reinforcement_count", 0))
            if reinforcement_count > 0:
                reinforcement_score = min(1.0, math.log(1 + reinforcement_count) / math.log(10))
            else:
                reinforcement_score = 0.0

            # Calculate weighted final score
            final_score = (
                self.SCORE_WEIGHTS["graph_distance"] * graph_score +
                self.SCORE_WEIGHTS["recency"] * recency_score +
                self.SCORE_WEIGHTS["confidence"] * confidence +
                self.SCORE_WEIGHTS["reinforcement"] * reinforcement_score
            )
            
            # Add scoring details to node
            node["retrieval_score"] = round(final_score, 3)
            node["score_breakdown"] = {
                "graph_distance": round(graph_score, 3),
                "recency": round(recency_score, 3),
                "confidence": round(confidence, 3),
                "reinforcement": round(reinforcement_score, 3),
                "days_since_reinforced": round(days_ago, 3),
                "hop_distance": hops,
                "trace": node.get("retrieval_trace", {})
            }
            
            # Add snippet for explainability
            node["snippet"] = self._create_snippet(node_type, props)
            
            scored_nodes.append(node)
        
        # Sort by score descending
        scored_nodes.sort(key=lambda x: x.get("retrieval_score", 0), reverse=True)
        
        print(f"[DEBUG] Scored and ranked {len(scored_nodes)} nodes (top score: {scored_nodes[0]['retrieval_score'] if scored_nodes else 'N/A'})")
        return scored_nodes
    
    def _create_snippet(self, node_type: str, props: Dict[str, Any]) -> str:
        """Create human-readable snippet for node."""
        if node_type == "Transaction":
            amount = props.get("amount", 0)
            tx_type = props.get("transaction_type", "transaction")
            return f"{tx_type.capitalize()} of ₹{amount:,.0f}"
        
        elif node_type == "Asset":
            name = props.get("name", "Unknown")
            asset_type = props.get("asset_type", "asset")
            return f"{name} ({asset_type})"
        
        elif node_type == "Fact":
            text = props.get("text", "")
            return text[:60] + "..." if len(text) > 60 else text
        
        elif node_type == "Goal":
            name = props.get("name", "Unknown goal")
            return f"Goal: {name}"
        
        elif node_type == "Message":
            text = props.get("text", "")
            return text[:50] + "..." if len(text) > 50 else text
        
        else:
            return props.get("name", props.get("text", node_type))
    
    def reinforce_cited_nodes(self, user_id: str, node_ids: List[str]):
        """
        Update reinforcement for nodes cited in LLM answer.
        
        Called AFTER answer generation (deferred reinforcement).
        
        Args:
            user_id: User identifier
            node_ids: List of node IDs that were cited in answer
        """
        if not self.driver or not node_ids:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $node_ids as node_id
                MATCH (n {id: node_id, user_id: $user_id})
                SET n.last_reinforced = datetime(),
                    n.reinforcement_count = coalesce(n.reinforcement_count, 0) + 1
                RETURN count(n) as updated_count
                """
                
                result = session.run(query, user_id=user_id, node_ids=node_ids)
                record = result.single()
                count = record["updated_count"] if record else 0
                print(f"Reinforced {count} cited nodes")
        
        except Exception as e:
            print(f"Error reinforcing nodes: {e}")
    
    def detect_contradictions(self, user_id: str, new_fact_text: str, entity_name: str) -> List[Dict[str, Any]]:
        """
        Detect contradictions with existing facts about the same entity.
        
        Args:
            user_id: User identifier
            new_fact_text: The new fact text to check
            entity_name: The entity this fact relates to
            
        Returns:
            List of potentially contradicting facts
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                # Find existing facts about the same entity with strict user isolation
                query = """
                MATCH (f:Fact {user_id: $user_id})
                MATCH (f)-[:RELATES_TO]->(e {name: $entity_name, user_id: $user_id})
                WHERE f.text <> $new_fact_text
                RETURN f.id as fact_id, f.text as fact_text, f.confidence as confidence
                ORDER BY f.timestamp DESC
                LIMIT 5
                """
                
                result = session.run(
                    query,
                    user_id=user_id,
                    new_fact_text=new_fact_text,
                    entity_name=entity_name
                )
                
                contradictions = []
                for record in result:
                    contradictions.append({
                        "fact_id": record["fact_id"],
                        "fact_text": record["fact_text"],
                        "confidence": record["confidence"]
                    })
                
                return contradictions
        
        except Exception as e:
            print(f"Error in contradiction detection: {e}")
            return []
    
    def mark_contradiction(self, old_fact_id: str, new_fact_id: str, user_id: str, reduce_confidence: bool = True):
        """
        Mark two facts as contradicting and optionally reduce confidence of old fact.
        
        Args:
            old_fact_id: ID of the older fact
            new_fact_id: ID of the newer fact
            user_id: User identifier for isolation
            reduce_confidence: Whether to reduce confidence of old fact
        """
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (old:Fact {id: $old_fact_id, user_id: $user_id})
                MATCH (new:Fact {id: $new_fact_id, user_id: $user_id})
                MERGE (old)-[:CONTRADICTS]->(new)
                """
                
                if reduce_confidence:
                    query += """
                    SET old.confidence = old.confidence * 0.5
                    """
                
                session.run(query, old_fact_id=old_fact_id, new_fact_id=new_fact_id, user_id=user_id)
                print(f"Marked contradiction between {old_fact_id} and {new_fact_id}")
        
        except Exception as e:
            print(f"Error marking contradiction: {e}")
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
