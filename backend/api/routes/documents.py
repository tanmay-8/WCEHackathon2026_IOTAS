"""
Document Upload Route - Handle file uploads and extraction.

Workflow:
1. User uploads document (PDF, DOCX, etc.)
2. Store file in S3
3. Extract text using TextExtractor
4. Process extracted text through LLM extractor
5. Ingest structured data into Neo4j graph

All follows the existing ingestion pipeline for consistency.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Header, status, Depends
from typing import Optional
from pathlib import Path
import tempfile
import os

from api.models import (
    DocumentUploadResponse,
    DocumentIngestionRequest,
    DocumentIngestionResponse,
    MemoryStorageResult,
)
from services.extraction.text_extractor import text_extractor
from services.extraction.llm_extractor import LLMExtractor
from services.graph.ingestion import GraphIngestion
from services.auth.auth_service import auth_service
from services.storage.s3_storage import s3_storage
from services.database.user_service import UserService

router = APIRouter()

# Initialize services
llm_extractor = LLMExtractor()
graph_ingestion = GraphIngestion()
user_service = UserService()


# ── Authentication Helper ───────────────────────────────────────────────────

async def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    """Extract user ID from JWT token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token claims",
        )
    return user_id


def get_neo4j_user_id(pg_user_id: str) -> str:
    """
    Convert PostgreSQL user ID to Neo4j user ID.

    Args:
        pg_user_id: PostgreSQL user UUID

    Returns:
        neo4j_user_id from the user record

    Raises:
        HTTPException: If user not found or not linked to Neo4j
    """
    user = user_service.get_user_by_id(pg_user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    neo4j_user_id = user.get("neo4j_user_id")
    if not neo4j_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not linked to knowledge graph"
        )

    return neo4j_user_id


# ── Document Upload & Extraction ────────────────────────────────────────────

@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["documents"],
    summary="Upload and extract document",
    description="Upload PDF, DOCX, or other supported formats and extract text."
)
async def upload_document(
    file: UploadFile = File(...),
    pg_user_id: str = Depends(get_current_user_id)
) -> DocumentUploadResponse:
    """
    Upload a document and extract text.

    Supports: PDF, DOCX, DOCX (legacy), plain text, images (future OCR)

    Args:
        file: Uploaded file
        user_id: Current user ID (from JWT)

    Returns:
        DocumentUploadResponse with extracted text preview and metadata
    """
    try:
        # Validate file type
        filename = file.filename or "unknown"
        file_ext = Path(filename).suffix.lower()

        if file_ext not in text_extractor.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_ext}. "
                f"Supported: {', '.join(text_extractor.SUPPORTED_FORMATS.keys())}"
            )

        # Read file bytes
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )

        if len(file_bytes) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large. Maximum size: 50MB"
            )

        # Extract text from uploaded file
        extracted_text, metadata = text_extractor.extract_from_bytes(
            file_bytes, filename)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract any text from the document"
            )

        # Upload to S3 (best effort; extraction should still work if S3 is disabled)
        s3_key = None
        s3_url = None
        try:
            content_type = file.content_type or 'application/octet-stream'
            s3_key, s3_url = s3_storage.upload_document(
                file_bytes=file_bytes,
                filename=filename,
                user_id=pg_user_id,
                content_type=content_type
            )
        except ValueError as e:
            metadata['s3_warning'] = str(e)

        return DocumentUploadResponse(
            success=True,
            filename=filename,
            format=metadata.get('format', 'Unknown'),
            extraction_method=metadata.get('extraction_method', 'Unknown'),
            text_preview=extracted_text[:500],  # First 500 chars
            extracted_text=extracted_text,  # Full extracted text
            metadata=metadata,
            message=f"Successfully extracted {len(extracted_text)} characters from {filename}",
            s3_key=s3_key,
            s3_url=s3_url
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


# ── Document Text Ingestion into Graph ──────────────────────────────────────

@router.post(
    "/documents/ingest",
    response_model=DocumentIngestionResponse,
    tags=["documents"],
    summary="Ingest extracted document text into graph",
    description="Process extracted document text through LLM and ingest into Neo4j graph."
)
async def ingest_document(
    request: DocumentIngestionRequest,
    pg_user_id: str = Depends(get_current_user_id)
) -> DocumentIngestionResponse:
    """
    Ingest extracted document text into the knowledge graph.

    Flow:
    1. Process document text through LLM extractor
    2. Generate structured facts and entities
    3. Ingest into Neo4j via memory orchestrator

    Args:
        request: DocumentIngestionRequest with extracted text
        user_id: Current user ID (from JWT)

    Returns:
        DocumentIngestionResponse with ingestion results
    """
    try:
        # Ensure user_id matches
        if request.user_id != pg_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot ingest document for different user"
            )

        # Use neo4j_user_id for graph extraction + ingestion consistency
        neo4j_user_id = get_neo4j_user_id(pg_user_id)

        if not request.document_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document text is empty"
            )

        # Process through LLM extractor (same as chat flow)
        extraction_result = llm_extractor.extract(
            request.document_text, neo4j_user_id)

        facts = extraction_result.get("facts", [])
        nodes = extraction_result.get("nodes", [])
        relationships = extraction_result.get("relationships", [])

        # Ingest into graph via graph ingestion service
        ingestion_stats = graph_ingestion.ingest_memory(
            user_id=neo4j_user_id,
            message_text=f"Document: {request.document_name}",
            facts=facts,
            nodes=nodes,
            relationships=relationships,
            skip_contradiction_detection=True
        )

        return DocumentIngestionResponse(
            success=True,
            document_name=request.document_name,
            extracted_text=request.document_text,  # Raw text from request
            extraction_stats={
                "facts_extracted": len(facts),
                "entities_extracted": len(nodes),
                "relationships_extracted": len(relationships),
                "text_length": len(request.document_text)
            },
            llm_extraction={
                "facts": facts,
                "entities": nodes,
                "relationships": relationships
            },
            memory_storage=MemoryStorageResult(
                nodes_created=ingestion_stats.get("nodes_created", 0),
                relationships_created=ingestion_stats.get(
                    "relationships_created", 0),
                facts_created=ingestion_stats.get("facts_created", 0),
                chunks_indexed=0  # Future: add chunking support
            ),
            message=f"Successfully ingested {request.document_name} into knowledge graph"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting document: {str(e)}"
        )


# ── Combined Upload + Ingest ───────────────────────────────────────────────

@router.post(
    "/documents/upload-and-ingest",
    response_model=DocumentIngestionResponse,
    tags=["documents"],
    summary="Upload document and fully ingest into graph",
    description="End-to-end: upload file → extract text → process → ingest into Neo4j."
)
async def upload_and_ingest_document(
    file: UploadFile = File(...),
    pg_user_id: str = Depends(get_current_user_id)
) -> DocumentIngestionResponse:
    """
    Complete end-to-end document processing.

    Steps:
    1. Extract text from uploaded file
    2. Process through LLM extractor
    3. Ingest structured data into Neo4j

    Args:
        file: Uploaded document file
        user_id: Current user ID (from JWT)

    Returns:
        DocumentIngestionResponse with complete ingestion results
    """
    try:
        # Step 1: Extract text from file
        filename = file.filename or "unknown"
        file_ext = Path(filename).suffix.lower()

        if file_ext not in text_extractor.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {file_ext}"
            )

        file_bytes = await file.read()

        if not file_bytes or len(file_bytes) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file size"
            )

        # Get neo4j_user_id
        neo4j_user_id = get_neo4j_user_id(pg_user_id)

        # Extract text
        extracted_text, metadata = text_extractor.extract_from_bytes(
            file_bytes, filename)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract text from document"
            )

        # Upload to S3 (best effort; continue if disabled)
        try:
            content_type = file.content_type or 'application/octet-stream'
            s3_key, s3_url = s3_storage.upload_document(
                file_bytes=file_bytes,
                filename=filename,
                user_id=pg_user_id,
                content_type=content_type
            )
            # Store S3 info in metadata for future reference
            metadata['s3_key'] = s3_key
            metadata['s3_url'] = s3_url
        except ValueError as e:
            metadata['s3_warning'] = str(e)

        # Step 2: Process through LLM using neo4j_user_id for consistency with chat flow
        extraction_result = llm_extractor.extract(
            extracted_text, neo4j_user_id)
        facts = extraction_result.get("facts", [])
        nodes = extraction_result.get("nodes", [])
        relationships = extraction_result.get("relationships", [])

        # Step 3: Ingest into graph using neo4j_user_id to match existing chat data
        ingestion_stats = graph_ingestion.ingest_memory(
            user_id=neo4j_user_id,
            message_text=f"Document: {filename}",
            facts=facts,
            nodes=nodes,
            relationships=relationships,
            skip_contradiction_detection=True
        )

        return DocumentIngestionResponse(
            success=True,
            document_name=filename,
            extracted_text=extracted_text,  # Raw text extracted from PDF
            extraction_stats={
                "facts_extracted": len(facts),
                "entities_extracted": len(nodes),
                "relationships_extracted": len(relationships),
                "text_length": len(extracted_text),
                "format": metadata.get('format'),
                "extraction_method": metadata.get('extraction_method')
            },
            llm_extraction={
                "facts": facts,
                "entities": nodes,
                "relationships": relationships
            },
            memory_storage=MemoryStorageResult(
                nodes_created=ingestion_stats.get("nodes_created", 0),
                relationships_created=ingestion_stats.get(
                    "relationships_created", 0),
                facts_created=ingestion_stats.get("facts_created", 0),
                chunks_indexed=0
            ),
            message=f"Successfully uploaded and ingested {filename}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )
