"""
Memory Orchestrator - Coordinates memory ingestion workflow.

Flow:
1. Extract entities from text (LLM)
2. Ingest into graph (Neo4j)
3. Ingest into vector database (Milvus)
"""

from typing import Dict, Any
from services.extraction.llm_extractor import LLMExtractor
from services.graph.ingestion import GraphIngestion
from services.vector.retrieval import VectorRetrieval
from services.vector.milvus_service import get_milvus_service


class MemoryOrchestrator:
    """
    Orchestrates complete memory ingestion workflow.
    """

    def __init__(self):
        """Initialize all required services."""
        self.extractor = LLMExtractor()
        self.graph_ingestion = GraphIngestion()
        self.vector_retrieval = VectorRetrieval()
        self.milvus_service = get_milvus_service()

    def ingest_memory(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Execute complete memory ingestion workflow.

        Args:
            user_id: User identifier
            message: User's message containing information to store

        Returns:
            Dictionary with ingestion results
        """
        # Step 1: Extract structured data from text
        extracted_data = self.extractor.extract(message, user_id)
        facts = extracted_data.get("facts", [])
        nodes = extracted_data.get("nodes", [])
        relationships = extracted_data.get("relationships", [])

        # Step 2: Ingest into graph (Message + Facts + Entities)
        graph_result = self.graph_ingestion.ingest_memory(
            user_id=user_id,
            message_text=message,
            facts=facts,
            nodes=nodes,
            relationships=relationships
        )

        # Step 3: Ingest vector chunks for semantic retrieval
        related_node_ids = []
        for fact in facts:
            fact_id = fact.get("id")
            if fact_id:
                related_node_ids.append(fact_id)
        for node in nodes:
            node_id = node.get("properties", {}).get("id")
            if node_id:
                related_node_ids.append(node_id)

        chunks_indexed = self.vector_retrieval.ingest_message(
            user_id=user_id,
            message_text=message,
            related_node_ids=list(set(related_node_ids))
        )

        # Step 4: Ingest into Milvus vector database
        vector_ids_indexed = 0
        if self.milvus_service:
            try:
                # Chunk the message
                chunk_size = 450
                overlap = 60
                chunks = self._chunk_text(
                    message, chunk_size=chunk_size, overlap=overlap)

                if chunks:
                    batch_data = [
                        {
                            "text": chunk,
                            "chunk_index": idx,
                            "source_type": "chat",
                            "confidence": 0.7,
                            "metadata": {"facts": [f['id'] for f in facts]}
                        }
                        for idx, chunk in enumerate(chunks)
                    ]
                    vector_ids = self.milvus_service.ingest_batch(
                        user_id, batch_data)
                    vector_ids_indexed = len(vector_ids)
            except Exception as e:
                print(f"Warning: Milvus ingestion failed: {e}")

        return {
            "nodes_created": graph_result.get("nodes_created", 0),
            "relationships_created": graph_result.get("relationships_created", 0),
            "facts_created": graph_result.get("facts_created", 0),
            "chunks_indexed": chunks_indexed,
            "vectors_indexed": vector_ids_indexed
        }

    def _chunk_text(self, text: str, chunk_size: int = 450, overlap: int = 60) -> list:
        """Split text into overlapping chunks."""
        if not text or chunk_size <= 0:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            start = end - overlap if end < text_len else text_len

        return chunks

    def close(self):
        """Close all service connections."""
        self.graph_ingestion.close()
        self.vector_retrieval.close()
        if self.milvus_service:
            self.milvus_service.close()
