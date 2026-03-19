"""Retrieval orchestrator with reciprocal rank fusion (RRF) for hybrid context."""

from typing import Dict, Any, List, Tuple
import time
from config.settings import Settings
from services.graph.retrieval import GraphRetrieval
from services.llm.answer_generator import AnswerGenerator
from services.vector.retrieval import VectorRetrieval


class RetrievalOrchestrator:
    """
    Orchestrates complete query retrieval and answer generation workflow.
    """

    RRF_K = 60

    def __init__(self):
        """Initialize all required services."""
        self.graph_retrieval = GraphRetrieval()
        self.vector_retrieval = VectorRetrieval()
        self.answer_generator = AnswerGenerator()

    def retrieve_and_answer(
        self,
        user_id: str,
        query: str
    ) -> Tuple[str, Dict[str, float], List[Dict[str, Any]]]:
        """
        Execute complete retrieval and answer generation workflow.

        Args:
            user_id: User identifier
            query: User's question

        Returns:
            Tuple of (answer, metrics, memory_citations)
        """
        # Step 1: Retrieve from graph (with timing)
        graph_start = time.time()
        graph_context, _ = self.graph_retrieval.retrieve(
            user_id=user_id,
            query=query,
            max_depth=3  # Adaptive based on query mode
        )
        graph_query_ms = (time.time() - graph_start) * 1000

        # Step 2: Vector retrieval
        vector_context, vector_search_ms = self.vector_retrieval.search(
            user_id=user_id,
            query=query
        )

        # Step 3: Assemble context with reciprocal rank fusion
        context_start = time.time()
        fused_results = self._fuse_rrf(graph_context, vector_context)
        formatted_context = [item["payload"]
                             for item in fused_results if item["source"] == "graph"]
        fused_vector_context = [item["payload"]
                                for item in fused_results if item["source"] == "vector"]
        context_assembly_ms = (time.time() - context_start) * 1000

        # Step 4: Generate answer using LLM (with timing)
        llm_start = time.time()
        answer = self.answer_generator.generate(
            query=query,
            graph_context=formatted_context,
            vector_context=fused_vector_context
        )
        llm_generation_ms = (time.time() - llm_start) * 1000

        # Step 5: Assemble detailed metrics
        retrieval_ms = graph_query_ms + vector_search_ms + context_assembly_ms

        metrics = {
            "graph_query_ms": round(graph_query_ms, 2),
            "vector_search_ms": round(vector_search_ms, 2),
            "context_assembly_ms": round(context_assembly_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "llm_generation_ms": round(llm_generation_ms, 2)
        }

        # Step 6: Format memory citations with scores
        memory_citations = self._format_memory_citations(fused_results)

        # Step 7: DEFERRED REINFORCEMENT - Update cited nodes after answer generation
        cited_node_ids = [node["properties"]["id"] for node in formatted_context[:10]
                          if "properties" in node and "id" in node["properties"]]
        if cited_node_ids:
            self.graph_retrieval.reinforce_cited_nodes(user_id, cited_node_ids)

        return answer, metrics, memory_citations

    def _fuse_rrf(
        self,
        graph_context: List[Dict[str, Any]],
        vector_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fuse graph and vector lists using reciprocal rank fusion."""
        fused: List[Dict[str, Any]] = []

        for rank, node in enumerate(graph_context, start=1):
            rrf_score = 1.0 / (self.RRF_K + rank)
            fused.append(
                {
                    "source": "graph",
                    "payload": node,
                    "fusion_score": round(rrf_score, 6),
                    "rank": rank
                }
            )

        for rank, chunk in enumerate(vector_context, start=1):
            chunk["retrieval_score"] = chunk.get(
                "retrieval_score", chunk.get("similarity", 0.0))
            rrf_score = 1.0 / (self.RRF_K + rank)
            fused.append(
                {
                    "source": "vector",
                    "payload": chunk,
                    "fusion_score": round(rrf_score, 6),
                    "rank": rank
                }
            )

        fused.sort(key=lambda item: item.get(
            "fusion_score", 0.0), reverse=True)
        return fused[: Settings.DEFAULT_TOP_K * 2]

    def _format_memory_citations(self, fused_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format memory citations with retrieval scores for explainability.

        Returns:
            List of formatted memory citations with snippets and hop distance
        """
        citations = []

        for item in fused_results:
            if item.get("source") == "graph":
                node = item.get("payload", {})
                node_type = node.get("type", "Unknown")
                props = node.get("properties", {})
                score = node.get("retrieval_score",
                                 item.get("fusion_score", 0.0))
                snippet = node.get("snippet", "")
                hop_distance = node.get("score_breakdown", {}).get(
                    "hop_distance", "N/A")

                citation = {
                    "node_type": node_type,
                    "retrieval_score": score,
                    "hop_distance": hop_distance,
                    "snippet": snippet,
                    "properties": {},
                    "score_breakdown": {
                        **(node.get("score_breakdown", {}) or {}),
                        "rrf_score": item.get("fusion_score", 0.0),
                        "source": "graph",
                        "rank": item.get("rank")
                    }
                }

                if node_type == "Fact":
                    citation["properties"] = {
                        "text": props.get("text", ""),
                        "confidence": props.get("confidence", 0.0),
                        "reinforcement_count": props.get("reinforcement_count", 0)
                    }
                elif node_type == "Transaction":
                    citation["properties"] = {
                        "amount": props.get("amount", 0),
                        "transaction_type": props.get("transaction_type", ""),
                        "confidence": props.get("confidence", 0.0)
                    }
                elif node_type in ["Asset", "Goal", "Entity"]:
                    citation["properties"] = {
                        key: value for key, value in props.items()
                        if key in ["name", "text", "confidence", "id"]
                    }

                citations.append(citation)
                continue

            chunk = item.get("payload", {})
            citations.append(
                {
                    "node_type": "DocumentChunk",
                    "retrieval_score": chunk.get("retrieval_score", chunk.get("similarity", 0.0)),
                    "hop_distance": "vector",
                    "snippet": chunk.get("text", "")[:120],
                    "properties": {
                        "chunk_id": chunk.get("id"),
                        "similarity": chunk.get("similarity", 0.0),
                        "source": "vector"
                    },
                    "score_breakdown": {
                        "vector_similarity": chunk.get("similarity", 0.0),
                        "rrf_score": item.get("fusion_score", 0.0),
                        "source": "vector",
                        "rank": item.get("rank")
                    }
                }
            )

        citations.sort(key=lambda item: item.get(
            "retrieval_score", 0.0), reverse=True)
        return citations[:10]

    def close(self):
        """Close all service connections."""
        self.graph_retrieval.close()
        self.vector_retrieval.close()
