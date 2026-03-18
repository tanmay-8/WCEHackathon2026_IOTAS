"""
Memory Orchestrator - Coordinates memory ingestion workflow with correction detection and handling.

Enhanced Flow:
1. Extract entities from text (LLM)
2. DETECT CORRECTIONS (NEW: ContradictionDetector + EntityDeduplication)
3. HANDLE CORRECTIONS (NEW: UpdateHandler = update instead of create)
4. Ingest new entities into graph (Neo4j)
5. Return metrics on corrections applied

Result: No more duplicate nodes! Updates instead of creates!
"""

from typing import Dict, Any
import time
from services.extraction.llm_extractor import LLMExtractor
from services.graph.ingestion import GraphIngestion
from services.graph.correction_orchestrator import CorrectionOrchestrator


class MemoryOrchestrator:
    """
    Orchestrates complete memory ingestion workflow with smart deduplication.
    
    NEW: Detects when user is correcting previous statements and updates existing
    nodes instead of creating duplicates!
    
    Example:
    Msg 1: "I invested 3000 in HDFC"
      → Creates Transaction node
    Msg 2: "No my HDFC investment was 4000"
      → Detects correction, UPDATES previous Transaction node to 4000
      → No duplicate created!
      → Space saved, accuracy improved
    """
    
    def __init__(self):
        """Initialize all required services."""
        self.extractor = LLMExtractor()
        self.graph_ingestion = GraphIngestion()
        self.correction_orchestrator = CorrectionOrchestrator()  # NEW
    
    def ingest_memory(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Execute complete memory ingestion workflow with correction handling.
        
        Args:
            user_id: User identifier
            message: User's message containing information to store
            
        Returns:
            Dictionary with ingestion results
            {
                "nodes_created": int,
                "nodes_updated": int,  # NEW
                "relationships_created": int,
                "facts_created": int,
                "chunks_indexed": int,
                "corrections_applied": int,  # NEW
                "duplicates_merged": int,  # NEW
                "messages": [...]  # Detailed messages
            }
        """
        start_time = time.time()
        
        # Step 1: Extract structured data from text
        start = time.time()
        extracted_data = self.extractor.extract(message, user_id)
        extract_time = (time.time() - start) * 1000
        
        facts = extracted_data.get("facts", [])
        nodes = extracted_data.get("nodes", [])
        relationships = extracted_data.get("relationships", [])
        
        print(f"[MemoryOrch] Step 1 - Extract: {extract_time:.1f}ms ({len(nodes)} nodes, {len(facts)} facts)")
        
        # Step 2: NEW - Detect and handle corrections
        start = time.time()
        correction_result = self.correction_orchestrator.process_message_for_corrections(
            user_id=user_id,
            current_message=message,
            extracted_entities=nodes,
            extracted_facts=facts
        )
        correction_time = (time.time() - start) * 1000
        
        corrections_applied = len(correction_result.get("corrections_detected", []))
        duplicates_merged = len(correction_result.get("duplicate_entities_merged", []))
        
        print(f"[MemoryOrch] Step 2 - Correction Check: {correction_time:.1f}ms (corrections_applied={corrections_applied}, duplicates_merged={duplicates_merged})")
        
        # Step 2b: Filter out entities that were corrected (don't create new ones)
        corrected_entity_ids = {
            c.get("entity") for c in correction_result.get("corrections_detected", [])
        }
        nodes_to_create = [
            node for node in nodes
            if node.get("value") not in corrected_entity_ids
        ]
        
        # Step 3: Ingest only new entities (not corrections) into graph
        start = time.time()
        graph_result = self.graph_ingestion.ingest_memory(
            user_id=user_id,
            message_text=message,
            facts=facts,
            nodes=nodes_to_create,  # Only uncorrected entities
            relationships=relationships
        )
        ingestion_time = (time.time() - start) * 1000
        
        print(f"[MemoryOrch] Step 3 - Graph Ingestion: {ingestion_time:.1f}ms (nodes_created={graph_result.get('nodes_created', 0)})")
        
        # Step 4: Build comprehensive result
        processing_time_ms = (time.time() - start_time) * 1000
        
        result = {
            "nodes_created": graph_result.get("nodes_created", 0),
            "nodes_updated": corrections_applied,  # NEW
            "relationships_created": graph_result.get("relationships_created", 0),
            "facts_created": graph_result.get("facts_created", 0),
            "chunks_indexed": 0,  # TODO: Vector indexing
            "corrections_applied": corrections_applied,  # NEW
            "duplicates_merged": duplicates_merged,  # NEW
            "processing_time_ms": round(processing_time_ms, 2),
            "messages": [
                correction_result.get("status_message"),
            ],
            "requires_user_confirmation": correction_result.get("requires_user_confirmation", False),
            "correction_details": correction_result.get("corrections_detected", [])
        }
        
        # Add detailed messages for each correction
        for correction in correction_result.get("corrections_detected", []):
            entity = correction.get("entity")
            new_value = correction.get("new_value")
            old_value = correction.get("previous_value")
            version = correction.get("version", 1)
            
            msg = f"Updated {entity}: {old_value} → {new_value} (v{version})"
            result["messages"].append(msg)
        
        print(f"[MemoryOrch] ✓ COMPLETE! Extract:{extract_time:.1f}ms Corrections:{correction_time:.1f}ms Ingestion:{ingestion_time:.1f}ms | Corrections:{corrections_applied} NewNodes:{graph_result.get('nodes_created', 0)} | TOTAL:{processing_time_ms:.1f}ms")
        
        return result
    
    def close(self):
        """Close all service connections."""
        if hasattr(self, 'graph_ingestion'):
            self.graph_ingestion.close()
