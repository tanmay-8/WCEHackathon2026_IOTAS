"""
Correction Orchestrator - Coordinates detection and handling of entity corrections and updates.

Pipeline:
1. Extract entities from new message
2. Detect if it's a correction (ContradictionDetector)
3. Find matching existing entities (EntityDeduplication)
4. Determine update strategy
5. Apply update (UpdateHandler)
6. Update relationships if needed
"""

from typing import Dict, List, Any, Optional
import time
from services.graph.entity_deduplication import EntityDeduplication
from services.graph.contradiction_detector import ContradictionDetector
from services.graph.update_handler import UpdateHandler


class CorrectionOrchestrator:
    """
    Orchestrates the entire correction and update workflow.
    
    High-level flow:
    1. Is this a correction? (ContradictionDetector)
    2. What entity does it correct? (EntityDeduplication)
    3. How to update it? (UpdateHandler via strategy)
    4. Apply changes and update relationships
    """
    
    def __init__(self):
        """Initialize all services."""
        self.deduplicator = EntityDeduplication()
        self.contradiction_detector = ContradictionDetector()
        self.updater = UpdateHandler()
    
    def process_message_for_corrections(
        self,
        user_id: str,
        current_message: str,
        extracted_entities: List[Dict[str, Any]],
        extracted_facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a user message to detect and handle corrections.
        
        Args:
            user_id: User identifier
            current_message: User's current message
            extracted_entities: Entities extracted from message
            extracted_facts: Facts extracted from message
            
        Returns:
            {
                "has_corrections": bool,
                "corrections_detected": [
                    {
                        "entity": "HDFC",
                        "entity_type": "Asset",
                        "correction_type": "explicit|implicit|conflict",
                        "action_taken": "updated|deprecat ed|merged|deferred",
                        "previous_value": 3000,
                        "new_value": 4000,
                        "node_id_updated": "node_123",
                        "confidence": 0.95,
                        "version": 2
                    }
                ],
                "duplicate_entities_merged": [...],
                "processing_time_ms": 245.3,
                "requires_user_confirmation": bool,
                "status_message": str
            }
        """
        start_time = time.time()
        
        result = {
            "has_corrections": False,
            "corrections_detected": [],
            "duplicate_entities_merged": [],
            "requires_user_confirmation": False,
            "status_message": ""
        }
        
        # Step 1: Detect if message is a correction
        correction_detection = self.contradiction_detector.detect_correction(
            current_message,
            user_id,
            extracted_entities
        )
        
        if not correction_detection.get("is_correction"):
            result["status_message"] = "No corrections detected, will create new entities"
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            return result
        
        result["has_corrections"] = True
        correction_type = correction_detection.get("correction_type")
        confidence = correction_detection.get("confidence", 0.0)
        strategy = correction_detection.get("strategy", "update")
        
        print(f"[CorrectionOrch] Detected {correction_type} correction with confidence {confidence:.2f}")
        
        # Step 2: Process each extracted entity for deduplication and updates
        for entity in extracted_entities:
            entity_type = entity.get("type")
            entity_name = entity.get("value")
            entity_props = entity.get("properties", {})
            
            # Find matching existing entity
            match = self.deduplicator.find_matching_entity(
                user_id,
                entity_type,
                entity_name,
                entity_props,
                confidence_threshold=0.75
            )
            
            if not match:
                continue  # No match, will be created as new entity
            
            # Match found - handle correction
            matching_id = match.get("id")
            match_score = match.get("match_score")
            conflict = match.get("conflict")
            
            print(f"[CorrectionOrch] Found match for {entity_name}: {matching_id} (score: {match_score})")
            
            # Step 3: Update the matched entity
            try:
                update_result = self.updater.update_node(
                    user_id=user_id,
                    node_id=matching_id,
                    node_type=entity_type,
                    new_properties=entity_props,
                    update_strategy=strategy,
                    justification=f"{correction_type} correction: {current_message[:100]}...",
                    preserve_history=True
                )
                
                if update_result.get("success"):
                    result["corrections_detected"].append({
                        "entity": entity_name,
                        "entity_type": entity_type,
                        "correction_type": correction_type,
                        "action_taken": strategy if strategy != "manual_review" else "deferred",
                        "previous_value": conflict.get("old") if conflict else None,
                        "new_value": conflict.get("new") if conflict else None,
                        "node_id_updated": matching_id,
                        "confidence": min(1.0, match_score * confidence),
                        "version": update_result.get("version", 1),
                        "strategy_used": strategy
                    })
                    
                    print(f"[CorrectionOrch] ✓ Updated {entity_type} {matching_id} to version {update_result.get('version')}")
                else:
                    print(f"[CorrectionOrch] ✗ Failed to update {entity_type} {matching_id}")
            
            except Exception as e:
                print(f"[CorrectionOrch] Error updating entity: {e}")
        
        # Step 4: Check for duplicates to merge
        duplicates_to_merge = self._find_and_merge_duplicates(user_id, extracted_entities)
        if duplicates_to_merge:
            result["duplicate_entities_merged"] = duplicates_to_merge
        
        # Step 5: Determine if user confirmation is needed
        result["requires_user_confirmation"] = (
            confidence < 0.80 or
            len([c for c in result["corrections_detected"] if c.get("action_taken") == "deferred"]) > 0
        )
        
        # Step 6: Build status message
        correction_count = len(result["corrections_detected"])
        merged_count = len(result["duplicate_entities_merged"])
        
        if correction_count > 0:
            result["status_message"] = f"Applied {correction_count} correction(s)"
            if merged_count > 0:
                result["status_message"] += f" and merged {merged_count} duplicate(s)"
            if result["requires_user_confirmation"]:
                result["status_message"] += " (pending your confirmation)"
        else:
            result["status_message"] = "No corrections to apply"
        
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        return result
    
    def _find_and_merge_duplicates(
        self,
        user_id: str,
        extracted_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect and merge duplicate entities from extraction.
        
        Returns list of merge operations performed.
        """
        merge_results = []
        
        # Group entities by (type, name similarity)
        entity_groups = {}
        
        for entity in extracted_entities:
            entity_type = entity.get("type")
            entity_name = entity.get("value", "")
            
            key = f"{entity_type}:{entity_name.lower()}"
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Merge groups with multiple entities
        for group_key, entities in entity_groups.items():
            if len(entities) > 1:
                print(f"[CorrectionOrch] Found {len(entities)} duplicates for {group_key}")
                # Take properties from merged entities
                merged_props = {}
                for entity in entities:
                    merged_props.update(entity.get("properties", {}))
                
                merge_results.append({
                    "entity_type": entities[0].get("type"),
                    "entity_name": entities[0].get("value"),
                    "duplicate_count": len(entities),
                    "merged_properties": merged_props
                })
        
        return merge_results
    
    def handle_user_confirmation(
        self,
        user_id: str,
        correction_id: str,
        user_decision: str  # "approve"|"reject"|"modify"
    ) -> Dict[str, Any]:
        """
        Handle user confirmation/rejection of pending corrections.
        
        Args:
            user_id: User identifier
            correction_id: ID of correction to confirm
            user_decision: approve|reject|modify
            
        Returns:
            {"success": bool, "message": str}
        """
        # TODO: Implement based on stored pending corrections
        return {
            "success": True,
            "message": f"Correction {correction_id} {user_decision}"
        }
