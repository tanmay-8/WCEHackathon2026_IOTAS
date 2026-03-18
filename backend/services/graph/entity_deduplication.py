"""
Entity Deduplication Service - Find matching entities across messages using multiple strategies.

Strategies:
1. String similarity (fuzzy matching) - Fast, for typos/variations
2. Semantic similarity - For meaning-based matches (requires embeddings)
3. Type-aware matching - Entity type constraints
4. Temporal proximity - Same user, nearby dates = likely same entity
5. Property overlap - Matching properties suggest same entity
"""

from typing import Dict, List, Any, Tuple, Optional
from difflib import SequenceMatcher
import json
from neo4j import GraphDatabase
from config.settings import Settings


class EntityDeduplication:
    """
    Finds duplicate/matching entities across messages to enable updates instead of creates.
    
    Matching Strategy (in order of confidence):
    1. Exact match (same type, name, user)
    2. Strong string similarity (>0.85) + same type
    3. Semantic similarity (embeddings) + type match
    4. Property match (same source values) + type
    5. Weak string similarity (>0.70) + strong confidence penalty
    """
    
    def __init__(self):
        """Initialize Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(
                Settings.NEO4J_URI,
                auth=(Settings.NEO4J_USER, Settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            self.driver = None
        
        # Caching for similarity scores (user_id -> cache)
        self._similarity_cache = {}
    
    def find_matching_entity(
        self, 
        user_id: str,
        entity_type: str,
        entity_name: str,
        entity_properties: Dict[str, Any],
        confidence_threshold: float = 0.75
    ) -> Optional[Dict[str, Any]]:
        """
        Find existing entity that matches the new one.
        
        Args:
            user_id: User identifier
            entity_type: Type (Asset, Transaction, Goal, etc.)
            entity_name: Name/identifier of entity
            entity_properties: Properties (amount, asset_type, etc.)
            confidence_threshold: Minimum confidence to return match (0.75)
            
        Returns:
            Matching entity with confidence score, or None
            {
                "id": "existing_id",
                "type": "Asset",
                "name": "HDFC MF",
                "properties": {...},
                "match_score": 0.95,
                "match_method": "exact_match|strong_similarity|semantic|property|weak",
                "conflict": {"type": "amount", "old": 3000, "new": 4000} or None
            }
        """
        if not self.driver:
            return None
        
        try:
            with self.driver.session() as session:
                # 1. Try exact match first (fastest)
                exact = self._find_exact_match(session, user_id, entity_type, entity_name)
                if exact:
                    return {
                        "id": exact["entity_id"],
                        "type": entity_type,
                        "name": exact["entity_name"],
                        "properties": exact["properties"],
                        "match_score": 1.0,
                        "match_method": "exact_match",
                        "conflict": self._detect_conflict(exact["properties"], entity_properties)
                    }
                
                # 2. Try strong string similarity (>0.85)
                candidates = self._find_similar_entities(
                    session, 
                    user_id, 
                    entity_type, 
                    entity_name,
                    similarity_threshold=0.85
                )
                
                if candidates:
                    best = candidates[0]  # Sorted by score descending
                    if best["score"] >= confidence_threshold:
                        return {
                            "id": best["entity_id"],
                            "type": entity_type,
                            "name": best["entity_name"],
                            "properties": best["properties"],
                            "match_score": best["score"],
                            "match_method": "strong_similarity",
                            "conflict": self._detect_conflict(best["properties"], entity_properties)
                        }
                
                # 3. Try property overlap (same source values)
                property_match = self._find_property_match(
                    session,
                    user_id,
                    entity_type,
                    entity_properties
                )
                
                if property_match and property_match["score"] >= confidence_threshold:
                    return {
                        "id": property_match["entity_id"],
                        "type": entity_type,
                        "name": property_match["entity_name"],
                        "properties": property_match["properties"],
                        "match_score": property_match["score"],
                        "match_method": "property_match",
                        "conflict": self._detect_conflict(property_match["properties"], entity_properties)
                    }
                
                # 4. Try weak string similarity (>0.70) with confidence penalty
                if candidates:
                    weak = candidates[-1]  # Weakest match from earlier search
                    if weak["score"] >= 0.70 and weak["score"] >= (confidence_threshold - 0.15):
                        return {
                            "id": weak["entity_id"],
                            "type": entity_type,
                            "name": weak["entity_name"],
                            "properties": weak["properties"],
                            "match_score": weak["score"],
                            "match_method": "weak_similarity",
                            "conflict": self._detect_conflict(weak["properties"], entity_properties)
                        }
                
                # No match found
                return None
        
        except Exception as e:
            print(f"[Deduplication] Error finding match: {e}")
            return None
    
    def _find_exact_match(
        self,
        session,
        user_id: str,
        entity_type: str,
        entity_name: str
    ) -> Optional[Dict[str, Any]]:
        """Find exact match by type and name."""
        query = """
        MATCH (n {user_id: $user_id, type: $entity_type})
        WHERE toLower(n.name) = toLower($entity_name) OR toLower(n.text) = toLower($entity_name)
        RETURN n.id as entity_id, 
               n.name as entity_name, 
               properties(n) as properties
        LIMIT 1
        """
        
        result = session.run(
            query,
            user_id=user_id,
            entity_type=entity_type,
            entity_name=entity_name
        )
        
        record = result.single()
        if record:
            return dict(record)
        return None
    
    def _find_similar_entities(
        self,
        session,
        user_id: str,
        entity_type: str,
        entity_name: str,
        similarity_threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """
        Find entities with similar names using string similarity.
        
        Returns list sorted by similarity score (descending).
        """
        query = """
        MATCH (n {user_id: $user_id, type: $entity_type})
        WITH n,
             apoc.text.similarity(toLower(n.name), toLower($entity_name)) as score
        WHERE score >= $threshold
        RETURN n.id as entity_id,
               n.name as entity_name,
               properties(n) as properties,
               score
        ORDER BY score DESC
        LIMIT 10
        """
        
        try:
            result = session.run(
                query,
                user_id=user_id,
                entity_type=entity_type,
                entity_name=entity_name,
                threshold=similarity_threshold
            )
            
            matches = [dict(record) for record in result]
            return matches
        except Exception as e:
            print(f"[Dedup] String similarity error (using fallback): {e}")
            # Fallback: Manual string comparison
            return self._fallback_string_similarity(
                session,
                user_id,
                entity_type,
                entity_name,
                similarity_threshold
            )
    
    def _fallback_string_similarity(
        self,
        session,
        user_id: str,
        entity_type: str,
        entity_name: str,
        similarity_threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """Fallback string similarity using Python's SequenceMatcher."""
        query = """
        MATCH (n {user_id: $user_id, type: $entity_type})
        RETURN n.id as entity_id,
               n.name as entity_name,
               properties(n) as properties
        LIMIT 50
        """
        
        result = session.run(query, user_id=user_id, entity_type=entity_type)
        candidates = []
        
        for record in result:
            name = record.get("entity_name", "")
            score = SequenceMatcher(None, entity_name.lower(), name.lower()).ratio()
            
            if score >= similarity_threshold:
                candidates.append({
                    "entity_id": record.get("entity_id"),
                    "entity_name": name,
                    "properties": record.get("properties", {}),
                    "score": score
                })
        
        return sorted(candidates, key=lambda x: x["score"], reverse=True)
    
    def _find_property_match(
        self,
        session,
        user_id: str,
        entity_type: str,
        entity_properties: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find match based on property overlap.
        
        Same amount + same asset_type = likely same entity
        """
        if not entity_properties:
            return None
        
        # Get amount and asset_type (most likely to match)
        amount = entity_properties.get("amount")
        asset_type = entity_properties.get("asset_type")
        source = entity_properties.get("source")
        
        if not (amount or asset_type):
            return None
        
        # Build flexible query
        filters = []
        params = {"user_id": user_id, "entity_type": entity_type}
        
        if amount:
            filters.append("n.amount = $amount")
            params["amount"] = amount
        
        if asset_type:
            filters.append("n.asset_type = $asset_type")
            params["asset_type"] = asset_type
        
        if source:
            filters.append("n.source = $source")
            params["source"] = source
        
        if not filters:
            return None
        
        where_clause = " AND ".join(filters)
        query = f"""
        MATCH (n {{user_id: $user_id, type: $entity_type}})
        WHERE {where_clause}
        RETURN n.id as entity_id,
               n.name as entity_name,
               properties(n) as properties,
               {len(filters)} as match_count
        ORDER BY match_count DESC
        LIMIT 1
        """
        
        try:
            result = session.run(query, **params)
            record = result.single()
            
            if record:
                # Score based on how many properties matched
                match_count = record.get("match_count", 0)
                max_props = 3  # amount, asset_type, source
                score = min(1.0, match_count / max_props)
                
                return {
                    "entity_id": record.get("entity_id"),
                    "entity_name": record.get("entity_name"),
                    "properties": record.get("properties", {}),
                    "score": score
                }
        except Exception as e:
            print(f"[Dedup] Property match error: {e}")
        
        return None
    
    def _detect_conflict(
        self,
        old_properties: Dict[str, Any],
        new_properties: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect conflicting values between old and new entity.
        
        Returns conflict info if found, None otherwise.
        {
            "type": "amount",
            "old": 3000,
            "new": 4000,
            "severity": "high"  # high if numeric diff > 10%, medium < 10%
        }
        """
        conflicts = []
        
        # Check numeric properties (amount, count, percentage)
        numeric_keys = ["amount", "value", "count", "shares"]
        for key in numeric_keys:
            old_val = old_properties.get(key)
            new_val = new_properties.get(key)
            
            if old_val is not None and new_val is not None:
                if float(old_val) != float(new_val):
                    percent_diff = abs(float(new_val) - float(old_val)) / float(old_val) * 100
                    severity = "high" if percent_diff > 10 else "medium"
                    
                    conflicts.append({
                        "type": key,
                        "old": old_val,
                        "new": new_val,
                        "percent_diff": round(percent_diff, 2),
                        "severity": severity
                    })
        
        # Check string properties (name, description, asset_type)
        string_keys = ["name", "asset_type", "description", "transaction_type"]
        for key in string_keys:
            old_val = old_properties.get(key)
            new_val = new_properties.get(key)
            
            if old_val and new_val and str(old_val).lower() != str(new_val).lower():
                conflicts.append({
                    "type": key,
                    "old": old_val,
                    "new": new_val,
                    "severity": "medium"
                })
        
        return conflicts[0] if conflicts else None
    
    def clear_cache(self, user_id: str = None):
        """Clear similarity cache."""
        if user_id:
            self._similarity_cache.pop(user_id, None)
        else:
            self._similarity_cache.clear()
