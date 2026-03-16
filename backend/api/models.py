from pydantic import BaseModel
from typing import Optional, List, Any
from enum import Enum


class IntentType(str, Enum):
    """Intent classification for chat requests"""
    MEMORY = "MEMORY"
    QUESTION = "QUESTION"
    BOTH = "BOTH"


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    user_id: str
    message: str
    conversation_id: Optional[str] = None


class RetrievalMetrics(BaseModel):
    """Detailed metrics for retrieval and generation performance"""
    graph_query_ms: float
    vector_search_ms: float
    context_assembly_ms: float
    retrieval_ms: float
    llm_generation_ms: float


class MemoryCitation(BaseModel):
    """Memory citation with retrieval score for explainability"""
    node_type: str
    retrieval_score: float
    hop_distance: Any  # Can be int or "N/A"
    snippet: str
    properties: dict
    score_breakdown: Optional[dict] = None  # graph_distance, recency, confidence, reinforcement


class MemoryStorageResult(BaseModel):
    """Result of memory storage operation"""
    nodes_created: int
    relationships_created: int
    facts_created: int
    chunks_indexed: int


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
    llm_extraction: Optional[dict] = None  # Actual facts, entities, relationships from LLM
    memory_storage: Optional[MemoryStorageResult] = None
    retrieval_metrics: Optional[RetrievalMetrics] = None
    message: str
