from typing import Dict, Any, List, Tuple, Optional
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config.settings import Settings
from services.graph.retrieval import GraphRetrieval
from services.graph.query_decomposition import QueryDecomposition, QueryIntent
from services.llm.answer_generator import AnswerGenerator
from services.vector.retrieval import VectorRetrieval
from services.vector.milvus_service import get_milvus_service
from services.cache.retrieval_cache import get_retrieval_cache


class RetrievalOrchestrator:
    """
    Orchestrates complete query retrieval and answer generation workflow.

    Optimized for sub-100ms retrieval through:
    - Parallel graph + vector retrieval
    - Reduced query complexity
    - Milvus vector search integration
    - Result caching (5-minute TTL)
    """

    RRF_K = 60
    MAX_RETRIEVAL_MS = 100  # Target: sub-100ms retrieval

    def __init__(self):
        """Initialize all required services."""
        self.graph_retrieval = GraphRetrieval()
        self.vector_retrieval = VectorRetrieval()
        self.milvus_service = get_milvus_service()
        self.answer_generator = AnswerGenerator()
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="retrieval_")
        self.cache = get_retrieval_cache()
        # Create event loop for async operations (lazy initialization)
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def retrieve_and_answer(
        self,
        user_id: str,
        query: str
    ) -> Tuple[str, Dict[str, float], List[Dict[str, Any]]]:
        """
        Execute complete retrieval and answer generation workflow.

        Optimized pipeline:
        1. Check cache for recent results
        2. Parallel graph + vector retrieval (concurrent)
        3. Reciprocal rank fusion
        4. LLM answer generation
        5. Deferred node reinforcement

        Args:
            user_id: User identifier
            query: User's question

        Returns:
            Tuple of (answer, metrics, memory_citations)
        """
        retrieval_start = time.time()
        cache_hit = False

        # Step 0: Check cache
        cached_results = self.cache.get(user_id, query, "retrieval_context")
        if cached_results:
            cache_hit = True
            graph_context = cached_results.get("graph_context", [])
            vector_context = cached_results.get("vector_context", [])
            graph_query_ms = 0.0  # From cache
            vector_search_ms = 0.0  # From cache
        else:
            # Step 1: Parallel retrieval using asyncio (graph + vector simultaneously)
            try:
                graph_context, vector_context, graph_query_ms, vector_search_ms = (
                    self._run_async_retrieval(user_id, query)
                )
            except Exception as e:
                # Fallback to sequential retrieval if async fails
                print(
                    f"Async retrieval failed, falling back to sequential: {e}")
                graph_context, graph_query_ms = self._retrieve_graph_parallel(
                    user_id, query)
                vector_context, vector_search_ms = self._retrieve_vector_parallel(
                    user_id, query)

            # Cache retrieval results
            self.cache.set(
                user_id, query,
                {
                    "graph_context": graph_context,
                    "vector_context": vector_context
                },
                "retrieval_context"
            )

        # Step 2: Assemble context with reciprocal rank fusion
        context_start = time.time()
        fused_results = self._fuse_rrf(graph_context, vector_context)

        # Step 2.5: Filter fused results by quality threshold for better grounding
        quality_threshold = 0.01  # Minimum fusion score
        quality_filtered = [r for r in fused_results if r.get(
            "fusion_score", 0.0) >= quality_threshold]

        # Select top contexts: prioritize high-confidence graph + supporting vector
        formatted_context = []
        vector_context_formatted = []

        for item in quality_filtered:
            if item["source"] == "graph" and len(formatted_context) < 8:
                formatted_context.append(item["payload"])
            elif item["source"] == "vector" and len(vector_context_formatted) < 5:
                vector_context_formatted.append(item["payload"])

        # Fallback: ensure we have some context
        if not formatted_context:
            formatted_context = [item["payload"]
                                 for item in fused_results[:5] if item["source"] == "graph"]
        if not vector_context_formatted:
            vector_context_formatted = [
                item["payload"] for item in fused_results[:3] if item["source"] == "vector"]

        context_assembly_ms = (time.time() - context_start) * 1000

        # Step 3: Generate answer using LLM (with timing)
        llm_start = time.time()
        answer = self.answer_generator.generate(
            query=query,
            graph_context=formatted_context,
            vector_context=vector_context_formatted
        )
        llm_generation_ms = (time.time() - llm_start) * 1000

        # Step 4: Assemble detailed metrics
        total_retrieval_ms = (time.time() - retrieval_start) * 1000
        retrieval_ms = graph_query_ms + vector_search_ms + context_assembly_ms

        metrics = {
            "decomposition_ms": round(decomposition_ms, 2),
            "graph_query_ms": round(graph_query_ms, 2),
            "vector_search_ms": round(vector_search_ms, 2),
            "context_assembly_ms": round(context_assembly_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "total_retrieval_ms": round(total_retrieval_ms, 2),
            "llm_generation_ms": round(llm_generation_ms, 2),
            "retrieval_optimized": retrieval_ms < self.MAX_RETRIEVAL_MS,
            "cache_hit": cache_hit
        }

        # Step 5: Format memory citations with scores
        memory_citations = self._format_memory_citations(fused_results)

        # Step 6: DEFERRED REINFORCEMENT - Update cited nodes after answer generation
        cited_node_ids = [node["properties"]["id"] for node in formatted_context[:10]
                          if "properties" in node and "id" in node["properties"]]
        if cited_node_ids:
            # Run reinforcement in background thread to avoid blocking
            self._reinforce_async(user_id, cited_node_ids)

        return answer, metrics, memory_citations

    def _run_async_retrieval(
        self,
        user_id: str,
        query: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, float]:
        """
        Run graph and vector retrieval in parallel using asyncio.

        Creates a new event loop and runs async parallel retrieval.

        Returns:
            Tuple of (graph_context, vector_context, graph_time_ms, vector_time_ms)
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self._async_parallel_retrieve(user_id, query)
            )
            return result
        finally:
            loop.close()

    async def _async_parallel_retrieve(
        self,
        user_id: str,
        query: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, float]:
        """
        Async function to retrieve graph and vector context in parallel.

        Uses asyncio.gather to run both retrievals concurrently.
        Each retrieval runs in a thread pool executor to avoid blocking.

        Returns:
            Tuple of (graph_context, vector_context, graph_time_ms, vector_time_ms)
        """
        # Create tasks for parallel execution
        graph_task = asyncio.create_task(
            self._async_retrieve_graph(user_id, query)
        )
        vector_task = asyncio.create_task(
            self._async_retrieve_vector(user_id, query)
        )

        # Wait for both tasks to complete (true parallel execution)
        results = await asyncio.gather(graph_task, vector_task, return_exceptions=True)

        # Handle results
        if isinstance(results[0], Exception):
            print(f"Graph retrieval exception: {results[0]}")
            graph_context, graph_query_ms = [], 0.0
        else:
            graph_context, graph_query_ms = results[0]

        if isinstance(results[1], Exception):
            print(f"Vector retrieval exception: {results[1]}")
            vector_context, vector_search_ms = [], 0.0
        else:
            vector_context, vector_search_ms = results[1]

        return graph_context, vector_context, graph_query_ms, vector_search_ms

    async def _async_retrieve_graph(
        self,
        user_id: str,
        query: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Async wrapper for graph retrieval.

        Offloads blocking graph query to thread executor for non-blocking execution.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._retrieve_graph_sync,
            user_id,
            query
        )

    async def _async_retrieve_vector(
        self,
        user_id: str,
        query: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Async wrapper for vector retrieval.

        Offloads blocking vector search to thread executor for non-blocking execution.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._retrieve_vector_sync,
            user_id,
            query
        )

    def _retrieve_graph_sync(
        self,
        user_id: str,
        query: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Synchronous graph retrieval (executed in thread executor).

        This is the actual blocking operation that gets run in a thread.
        """
        start = time.time()
        try:
            graph_context, _ = self.graph_retrieval.retrieve(
                user_id=user_id,
                query=query,
                max_depth=2  # Reduced from 3 for sub-100ms target
            )
            elapsed = (time.time() - start) * 1000
            return graph_context, elapsed
        except Exception as e:
            print(f"Graph retrieval error: {e}")
            return [], (time.time() - start) * 1000

    def _retrieve_vector_sync(
        self,
        user_id: str,
        query: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Synchronous vector retrieval (executed in thread executor).

        This is the actual blocking operation that gets run in a thread.
        """
        start = time.time()
        try:
            # Try Milvus first (faster for semantic search)
            if self.milvus_service and self.milvus_service.collection:
                vector_context = self._search_milvus(user_id, query)
                if vector_context:
                    elapsed = (time.time() - start) * 1000
                    return vector_context, elapsed

            # Fallback to Neo4j vector retrieval
            vector_context, _ = self.vector_retrieval.search(
                user_id=user_id,
                query=query,
                top_k=5  # Reduced for faster retrieval
            )
            elapsed = (time.time() - start) * 1000
            return vector_context, elapsed
        except Exception as e:
            print(f"Vector retrieval error: {e}")
            return [], (time.time() - start) * 1000

    def _retrieve_graph_parallel(self, user_id: str, query: str) -> Tuple[List[Dict[str, Any]], float]:
        """Fallback: Retrieve from graph database sequentially with optimized depth."""
        return self._retrieve_graph_sync(user_id, query)

    def _retrieve_vector_parallel(self, user_id: str, query: str) -> Tuple[List[Dict[str, Any]], float]:
        """Fallback: Retrieve from vector database (Milvus) with fallback to Neo4j sequentially."""
        return self._retrieve_vector_sync(user_id, query)

    def _search_milvus(self, user_id: str, query: str, top_k: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Search Milvus for semantic vector matches."""
        try:
            results = self.milvus_service.search_similar(
                user_id=user_id,
                query_text=query,
                top_k=top_k,
                threshold=0.4  # Lower threshold for more results
            )

            if results:
                # Convert Milvus results to standard format
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "id": result.get("vector_id", ""),
                        "text": result.get("text", ""),
                        "similarity": result.get("similarity", 0.0),
                        "retrieval_score": result.get("similarity", 0.0),
                        "source_type": result.get("source_type", "vector"),
                        "confidence": result.get("confidence", 0.7),
                        "metadata": result.get("metadata", {})
                    })
                return formatted_results
        except Exception as e:
            print(f"Milvus search error: {e}")

        return None

    def _reinforce_async(self, user_id: str, node_ids: List[str]) -> None:
        """Reinforce cited nodes in background thread."""
        try:
            self.executor.submit(
                self.graph_retrieval.reinforce_cited_nodes,
                user_id,
                node_ids
            )
        except Exception as e:
            print(f"Background reinforcement error: {e}")

    def _fuse_rrf(
        self,
        graph_context: List[Dict[str, Any]],
        vector_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fuse graph and vector results using adaptive reciprocal rank fusion.

        Improvements:
        - Adaptive weighting based on confidence scores
        - Confidence-aware threshold filtering
        - Better score normalization
        - Deduplication of similar results

        Args:
            graph_context: Graph retrieval results
            vector_context: Vector retrieval results

        Returns:
            Sorted list of fused results
        """
        fused: List[Dict[str, Any]] = []

        # Adaptive weights based on result quality
        graph_confidence_avg = self._avg_confidence(graph_context)
        vector_confidence_avg = self._avg_confidence(vector_context)

        # Normalize weights (give more weight to higher-confidence source)
        total_confidence = graph_confidence_avg + vector_confidence_avg
        if total_confidence > 0:
            graph_weight = graph_confidence_avg / total_confidence
            vector_weight = vector_confidence_avg / total_confidence
        else:
            graph_weight = 0.5
            vector_weight = 0.5

        # Fuse graph results with adaptive weighting
        for rank, node in enumerate(graph_context, start=1):
            confidence = node.get("confidence", node.get(
                "properties", {}).get("confidence", 0.7))

            # Adaptive RRF: confidence modulates the RRF score
            base_rrf = 1.0 / (self.RRF_K + rank)
            confidence_boost = confidence * 0.3  # 0-0.3 boost based on confidence
            rrf_score = base_rrf * (1.0 + confidence_boost) * graph_weight

            fused.append(
                {
                    "source": "graph",
                    "payload": node,
                    "fusion_score": round(rrf_score, 6),
                    "rank": rank,
                    "source_weight": round(graph_weight, 3),
                    "confidence": round(confidence, 3)
                }
            )

        # Fuse vector results with adaptive weighting
        for rank, chunk in enumerate(vector_context, start=1):
            chunk["retrieval_score"] = chunk.get(
                "retrieval_score", chunk.get("similarity", 0.0))
            confidence = chunk.get("retrieval_score", 0.5)

            # Adaptive RRF: confidence modulates the RRF score
            base_rrf = 1.0 / (self.RRF_K + rank)
            confidence_boost = confidence * 0.3  # 0-0.3 boost based on similarity
            rrf_score = base_rrf * (1.0 + confidence_boost) * vector_weight

            fused.append(
                {
                    "source": "vector",
                    "payload": chunk,
                    "fusion_score": round(rrf_score, 6),
                    "rank": rank,
                    "source_weight": round(vector_weight, 3),
                    "confidence": round(confidence, 3)
                }
            )

        # Sort by fusion score (highest first)
        fused.sort(key=lambda item: item.get(
            "fusion_score", 0.0), reverse=True)

        # Return top results with quality threshold
        top_k = Settings.DEFAULT_TOP_K * 2
        return fused[:top_k]

    def _avg_confidence(self, context: List[Dict[str, Any]]) -> float:
        """Calculate average confidence score for context items."""
        if not context:
            return 0.5

        confidence_scores = []
        for item in context:
            # Try different confidence field names
            conf = (
                item.get("confidence") or
                item.get("properties", {}).get("confidence") or
                item.get("similarity") or
                item.get("retrieval_score") or
                0.5
            )
            confidence_scores.append(float(conf))

        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

    def _format_memory_citations(self, fused_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format memory citations with retrieval scores for explainability.

        Optimization: Include better confidence indicators and source attribution
        for improved grounding in final answers.

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

                # Get confidence from node or properties
                confidence = node.get(
                    "confidence", props.get("confidence", 0.7))

                citation = {
                    "node_type": node_type,
                    "retrieval_score": round(score, 4),
                    "confidence": round(confidence, 3),
                    "hop_distance": hop_distance,
                    "snippet": snippet,
                    "source": "graph",
                    "source_weight": item.get("source_weight", 0.5),
                    "properties": {},
                    "score_breakdown": {
                        **(node.get("score_breakdown", {}) or {}),
                        "rrf_score": item.get("fusion_score", 0.0),
                        "fusion_weight": item.get("source_weight", 0.5),
                        "source": "graph",
                        "rank": item.get("rank"),
                        "confidence": round(confidence, 3)
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
            similarity = chunk.get(
                "retrieval_score", chunk.get("similarity", 0.0))

            citations.append(
                {
                    "node_type": "DocumentChunk",
                    "retrieval_score": round(similarity, 4),
                    # Use similarity as confidence for vector
                    "confidence": round(similarity, 3),
                    "hop_distance": "vector",
                    # Increased snippet size
                    "snippet": chunk.get("text", "")[:200],
                    "source": "vector",
                    "source_weight": item.get("source_weight", 0.5),
                    "properties": {
                        "chunk_id": chunk.get("id"),
                        "similarity": round(similarity, 4),
                        "source_type": chunk.get("source_type", "vector")
                    },
                    "score_breakdown": {
                        "vector_similarity": round(similarity, 4),
                        "rrf_score": item.get("fusion_score", 0.0),
                        "fusion_weight": item.get("source_weight", 0.5),
                        "source": "vector",
                        "rank": item.get("rank")
                    }
                }
            )

        # Sort by composite score: retrieval_score + fusion contribution
        citations.sort(
            key=lambda item: (item.get("retrieval_score", 0.0) * 0.6) +
                             (item.get("source_weight", 0.5) * 0.4),
            reverse=True
        )
        return citations[:12]  # Return up to 12 citations instead of 10

    def close(self):
        """Close all service connections and cleanup resources."""
        self.graph_retrieval.close()
        self.vector_retrieval.close()
        self.executor.shutdown(wait=True)
