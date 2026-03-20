from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from enum import Enum


class IntentType(str, Enum):
    """Intent classification for chat requests"""
    MEMORY = "MEMORY"
    QUESTION = "QUESTION"
    BOTH = "BOTH"


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    user_id: str = Field(
        ...,
        description="User UUID from the frontend context. This field is accepted for compatibility.",
        example="f4a2fca4-39d0-4028-8fb5-4f0b84b6c9d5"
    )
    message: str = Field(
        ...,
        min_length=1,
        description="User message to ingest as memory and/or answer as a question.",
        example="How much have I invested this month?"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation identifier reserved for future multi-thread support.",
        example=None
    )


class RetrievalMetrics(BaseModel):
    """Detailed metrics for retrieval and generation performance"""
    graph_query_ms: float = Field(
        ..., description="Time spent on graph retrieval in milliseconds.", example=34.2)
    vector_search_ms: float = Field(
        ..., description="Time spent on vector retrieval in milliseconds.", example=0.0)
    context_assembly_ms: float = Field(
        ..., description="Time to assemble context in milliseconds.", example=5.4)
    retrieval_ms: float = Field(
        ..., description="Total retrieval time in milliseconds.", example=45.1)
    llm_generation_ms: float = Field(
        ..., description="LLM answer generation time in milliseconds.", example=502.7)


class MemoryCitation(BaseModel):
    """Memory citation with retrieval score for explainability"""
    node_type: str = Field(...,
                           description="Type of cited graph node.", example="Fact")
    retrieval_score: float = Field(
        ..., description="Final ranking score for the citation.", example=0.91)
    hop_distance: Any = Field(
        ..., description="Shortest path hop distance from user context.", example=1)
    snippet: str = Field(..., description="Human-readable citation snippet.",
                         example="Investment in HDFC Mutual Fund")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Raw node properties used for explainability.")
    score_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional weighted score components: graph_distance, recency, confidence, reinforcement."
    )
    source: str = Field(
        default="hybrid", description="Source of the citation: 'graph', 'vector', or 'hybrid'.", example="graph")


class MemoryStorageResult(BaseModel):
    """Result of memory storage operation"""
    nodes_created: int = Field(...,
                               description="Number of graph nodes created.", example=4)
    relationships_created: int = Field(
        ..., description="Number of graph relationships created.", example=3)
    facts_created: int = Field(...,
                               description="Number of normalized facts extracted.", example=2)
    chunks_indexed: int = Field(
        ..., description="Number of chunks indexed for retrieval.", example=1)


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    intent: IntentType
    answer: Optional[str] = None
    memory_storage: Optional[MemoryStorageResult] = None
    retrieval_metrics: Optional[RetrievalMetrics] = None
    memory_citations: Optional[List[MemoryCitation]] = None
    message: str


# ── Document Upload Models ──────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Response from document upload and extraction"""
    success: bool
    filename: str
    format: str
    extraction_method: str
    text_preview: str
    extracted_text: str  # Full extracted text from document
    metadata: dict
    message: str
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None


class DocumentIngestionRequest(BaseModel):
    """Request to ingest extracted document text into graph"""
    user_id: str
    document_text: str
    document_name: str
    document_format: str
    metadata: Optional[dict] = None


class DocumentIngestionResponse(BaseModel):
    """Response from document ingestion into graph"""
    success: bool
    document_name: str
    extracted_text: str
    extraction_stats: dict
    llm_extraction: Optional[dict] = None
    memory_storage: Optional[MemoryStorageResult] = None
    message: str
