"""
Contradiction Detection Service - Identify when user is correcting/contradicting previous statements.

Detects:
1. Explicit corrections ("No, that was wrong", "Actually", "I meant")
2. Implicit corrections (conflicting numeric values)
3. Conflicting statements (goal changed, asset sold then re-bought)
4. Deprecations (old statement vs new statement for same entity)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from neo4j import GraphDatabase
from config.settings import Settings

load_dotenv()


class ContradictionDetector:
    """
    Detects corrections and contradictions in user messages.
    
    Detection Methods:
    1. Keyword-based: Direct phrases like "no", "actually", "I meant"
    2. LLM-based: Understand implicit corrections
    3. Graph-based: Find conflicting facts in database
    4. Numeric: Check for value conflicts in extracted entities
    """
    
    # Keywords that suggest correction
    CORRECTION_KEYWORDS = [
        "no", "actually", "wait", "correction", "wrong", "mistaken",
        "meant", "sorry", "let me correct", "that was wrong",
        "not", "rather", "instead of", "i meant", "should be",
        "updated", "changed", "revised", "correcting myself"
    ]
    
    # Keywords that suggest deprecation
    DEPRECATION_KEYWORDS = [
        "no longer", "sold", "liquidated", "closed", "exited",
        "cancelled", "stopped investing", "divested"
    ]
    
    def __init__(self):
        """Initialize services."""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.llm = None
        
        try:
            self.driver = GraphDatabase.driver(
                Settings.NEO4J_URI,
                auth=(Settings.NEO4J_USER, Settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            self.driver = None
    
    def detect_correction(
        self,
        current_message: str,
        user_id: str,
        extracted_entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect if current message is correcting previous statements.
        
        Args:
            current_message: User's current message
            user_id: User ID
            extracted_entities: Entities extracted from current message
            
        Returns:
            {
                "is_correction": bool,
                "correction_type": "explicit|implicit|conflict|deprecation",
                "confidence": 0.0-1.0,
                "previous_statement": "...",  # What's being corrected
                "corrections": [
                    {
                        "entity": "HDFC",
                        "property": "amount",
                        "old_value": 3000,
                        "new_value": 4000,
                        "justification": "User said 'my HDFC investment was 4000'"
                    }
                ],
                "strategy": "update|deprecate|merge",  # How to handle
                "requires_user_confirmation": bool
            }
        """
        start_total = time.time()
        message_lower = current_message.lower()
        
        # 1. Check for explicit correction keywords
        start = time.time()
        explicit_score = self._check_explicit_keywords(message_lower)
        explicit_time = (time.time() - start) * 1000
        
        # 2. Check for deprecation keywords
        start = time.time()
        deprecation_score = self._check_deprecation_keywords(message_lower)
        deprecation_time = (time.time() - start) * 1000
        
        # 3. LLM-based detection (if available)
        llm_score = 0.0
        llm_context = ""
        llm_time = 0.0
        if self.llm:
            start = time.time()
            llm_score, llm_context = self._llm_detect_correction(current_message)
            llm_time = (time.time() - start) * 1000
        
        # 4. Graph-based conflict detection
        start = time.time()
        graph_conflicts = self._detect_graph_conflicts(
            user_id,
            extracted_entities
        )
        graph_time = (time.time() - start) * 1000
        
        # 5. Aggregated decision
        total_time = (time.time() - start_total) * 1000
        is_correction = (explicit_score > 0.5 or deprecation_score > 0.5 or 
                        llm_score > 0.6 or len(graph_conflicts) > 0)
        
        if not is_correction:
            print(f"[ContradictionDetector] NO correction. Explicit:{explicit_time:.1f}ms Deprecation:{deprecation_time:.1f}ms LLM:{llm_time:.1f}ms Graph:{graph_time:.1f}ms TOTAL:{total_time:.1f}ms")
            return {
                "is_correction": False,
                "confidence": 0.0,
                "corrections": []
            }
        
        # Determine correction type and strategy
        if explicit_score > deprecation_score:
            correction_type = "explicit"
            strategy = "update"
        elif deprecation_score > explicit_score:
            correction_type = "deprecation"
            strategy = "deprecate"
        elif len(graph_conflicts) > 0:
            correction_type = "conflict"
            strategy = "merge"
        else:
            correction_type = "implicit"
            strategy = "update"
        
        # Aggregate confidence
        confidence = max(explicit_score, deprecation_score, llm_score)
        if len(graph_conflicts) > 0:
            confidence = (confidence + 1.0) / 2
        
        print(f"[ContradictionDetector] ✓ IS CORRECTION! Type:{correction_type} Confidence:{confidence:.2f}. Explicit:{explicit_time:.1f}ms Deprecation:{deprecation_time:.1f}ms LLM:{llm_time:.1f}ms Graph:{graph_time:.1f}ms TOTAL:{total_time:.1f}ms")
        
        return {
            "is_correction": True,
            "correction_type": correction_type,
            "confidence": min(1.0, confidence),
            "strategy": strategy,
            "corrections": graph_conflicts,
            "requires_user_confirmation": confidence < 0.80,
            "llm_context": llm_context if llm_context else None
        }
    
    def _check_explicit_keywords(self, message_lower: str) -> float:
        """Score for explicit correction keywords."""
        matches = sum(1 for kw in self.CORRECTION_KEYWORDS if kw in message_lower)
        return min(1.0, matches * 0.3)
    
    def _check_deprecation_keywords(self, message_lower: str) -> float:
        """Score for deprecation keywords."""
        matches = sum(1 for kw in self.DEPRECATION_KEYWORDS if kw in message_lower)
        return min(1.0, matches * 0.4)
    
    def _llm_detect_correction(self, message: str) -> tuple[float, str]:
        """
        Use LLM to detect implicit corrections.
        
        Returns (confidence, context_explanation)
        """
        if not self.llm:
            return 0.0, ""
        
        try:
            prompt = f"""Analyze if this message is correcting or contradicting previous statements.

Message: "{message}"

Respond with JSON:
{{
    "is_correction": boolean,
    "correction_type": "explicit|implicit|none",
    "confidence": 0.0-1.0,
    "explanation": "why this is/isn't a correction"
}}

Only return JSON, no other text."""
            
            response = self.llm.generate_content(prompt)
            
            import json
            result = json.loads(response.text)
            
            confidence = result.get("confidence", 0.0) if result.get("is_correction") else 0.0
            explanation = result.get("explanation", "")
            
            return confidence, explanation
        
        except Exception as e:
            print(f"[Contradiction] LLM error: {e}")
            return 0.0, ""
    
    def _detect_graph_conflicts(
        self,
        user_id: str,
        extracted_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between extracted entities and existing graph.
        
        Returns list of conflicts found.
        """
        if not self.driver or not extracted_entities:
            return []
        
        conflicts = []
        
        try:
            with self.driver.session() as session:
                for entity in extracted_entities:
                    entity_type = entity.get("type")
                    entity_name = entity.get("value")
                    entity_props = entity.get("properties", {})
                    
                    # Find existing similar entities
                    query = """
                    MATCH (existing {user_id: $user_id, type: $entity_type})
                    WHERE toLower(existing.name) = toLower($entity_name) OR 
                          toLower(existing.text) = toLower($entity_name)
                    OPTIONAL MATCH (existing)-[r]-(connected)
                    RETURN existing, properties(existing) as props, count(r) as relationship_count
                    LIMIT 5
                    """
                    
                    result = session.run(
                        query,
                        user_id=user_id,
                        entity_type=entity_type,
                        entity_name=entity_name
                    )
                    
                    for record in result:
                        existing_props = record.get("props", {})
                        
                        # Check for numeric conflicts
                        for key in ["amount", "value", "count"]:
                            old_val = existing_props.get(key)
                            new_val = entity_props.get(key)
                            
                            if old_val is not None and new_val is not None:
                                if float(old_val) != float(new_val):
                                    percent_diff = abs(float(new_val) - float(old_val)) / float(old_val) * 100
                                    
                                    conflicts.append({
                                        "entity": entity_name,
                                        "entity_type": entity_type,
                                        "property": key,
                                        "old_value": old_val,
                                        "new_value": new_val,
                                        "percent_diff": round(percent_diff, 2),
                                        "severity": "high" if percent_diff > 10 else "medium"
                                    })
        
        except Exception as e:
            print(f"[Contradiction] Graph conflict detection error: {e}")
        
        return conflicts
    
    def classify_correction_strategy(
        self,
        correction_type: str,
        corrections: List[Dict[str, Any]],
        confidence: float
    ) -> str:
        """
        Classify what action to take.
        
        Returns: "update"|"deprecate"|"merge"|"manual_review"
        """
        if confidence < 0.60:
            return "manual_review"
        
        if correction_type == "deprecation":
            return "deprecate"
        
        if correction_type == "explicit" and corrections:
            return "update"
        
        if len(corrections) > 2:  # Multiple conflicting properties
            return "merge"
        
        return "update"
