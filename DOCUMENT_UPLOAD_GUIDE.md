# Document Upload & Ingestion Feature

## Overview

GraphMind now supports uploading documents (PDF, DOCX, TXT, and future image OCR) and automatically extracting, processing, and ingesting their content into your knowledge graph.

The feature integrates seamlessly with the existing ingestion pipeline, treating document text the same way as chat messages—extracting facts, entities, and relationships, then storing them in Neo4j.

---

## Architecture

### Flow Diagram

```
User Upload (Frontend)
    ↓
File Validation
    ↓
TextExtractor Service (Backend)
    ├─ PDF extraction (PyPDF2)
    ├─ DOCX extraction (python-docx)
    ├─ Plain text reading
    └─ OCR support (future)
    ↓
Extracted Text + Metadata
    ↓
LLMExtractor Service
    (Same as chat flow)
    ├─ Extract facts
    ├─ Identify entities
    └─ Generate relationships
    ↓
Memory Orchestrator
    (Same as chat flow)
    ├─ Create message node
    ├─ Create fact nodes
    ├─ MERGE entity nodes
    └─ Create relationships
    ↓
Neo4j Graph Database
    ↓
Success Response with Stats
```

---

## Backend Implementation

### 1. Text Extraction Service
**File:** `backend/services/extraction/text_extractor.py`

**Supported Formats:**
- **PDF** (.pdf) - Extracts text using PyPDF2
- **Word Documents** (.docx, .doc) - Uses python-docx
- **Plain Text** (.txt) - Direct file reading
- **Images** (.png, .jpg, .jpeg) - OCR support via PaddleOCR

**Key Features:**
- Handles files up to 50MB
- Provides metadata (format, extraction method, text preview)
- Chunking utility for large documents
- User-friendly error messages

**Example Usage:**

```python
from services.extraction.text_extractor import text_extractor

# Extract from file path
text, metadata = text_extractor.extract_from_file("/path/to/document.pdf")

# Extract from file bytes (uploaded files)
text, metadata = text_extractor.extract_from_bytes(file_bytes, "document.pdf")

# Chunk large text for LLM context
chunks = text_extractor.chunk_text(text, chunk_size=1000, overlap=100)
```

**Metadata Returned:**
```json
{
  "format": "PDF",
  "extraction_method": "PyPDF2",
  "filename": "financial_report.pdf",
  "text_preview": "First 200 characters...",
  "total_pages": 10
}
```

---

### 2. API Routes
**File:** `backend/api/routes/documents.py`

#### Upload & Extract
```
POST /documents/upload

Request:
  - file: multipart/form-data (PDF, DOCX, TXT, Images)

Response:
{
  "success": true,
  "filename": "financial_report.pdf",
  "format": "PDF",
  "extraction_method": "PyPDF2",
  "text_preview": "...",
  "metadata": {...}
}

Status Codes:
  - 200: Success
  - 400: Unsupported format or empty file
  - 413: File too large (>50MB)
  - 500: Extraction error
```

#### Ingest Extracted Text
```
POST /documents/ingest

Request:
{
  "user_id": "user_123",
  "document_text": "...",
  "document_name": "financial_report.pdf",
  "document_format": "PDF",
  "metadata": {...}  // optional
}

Response:
{
  "success": true,
  "document_name": "financial_report.pdf",
  "extraction_stats": {
    "facts_extracted": 15,
    "entities_extracted": 8,
    "relationships_extracted": 12,
    "text_length": 5000
  },
  "memory_storage": {
    "nodes_created": 25,
    "relationships_created": 18,
    "facts_created": 15,
    "chunks_indexed": 0
  }
}

Status Codes:
  - 200: Success
  - 403: Unauthorized (user mismatch)
  - 500: Ingestion error
```

#### Combined Upload & Ingest
```
POST /documents/upload-and-ingest

Request:
  - file: multipart/form-data

Response:
{
  "success": true,
  "document_name": "financial_report.pdf",
  "extraction_stats": {...},
  "memory_storage": {...}
}

Status Codes: Same as above
```

---

### 3. API Models
**File:** `backend/api/models.py`

```python
class DocumentUploadResponse(BaseModel):
    """Response from document upload and extraction"""
    success: bool
    filename: str
    format: str
    extraction_method: str
    text_preview: str
    metadata: dict
    message: str

class DocumentIngestionRequest(BaseModel):
    """Request to ingest extracted text"""
    user_id: str
    document_text: str
    document_name: str
    document_format: str
    metadata: Optional[dict] = None

class DocumentIngestionResponse(BaseModel):
    """Response from document ingestion"""
    success: bool
    document_name: str
    extraction_stats: dict
    memory_storage: Optional[MemoryStorageResult] = None
    retrieval_metrics: Optional[RetrievalMetrics] = None
    message: str
```

---

## Frontend Implementation

### 1. Document Upload Component
**File:** `frontend/src/components/DocumentUpload.tsx`

**Features:**
- File input with drag-and-drop support
- Real-time upload progress
- Error handling with user-friendly messages
- Automatic format validation
- Size limit enforcement (50MB)

**Props:**

```typescript
interface DocumentUploadProps {
  onUploadSuccess?: (data: any) => void;
  onUploadError?: (error: string) => void;
  userId: string;
}
```

**Example Usage:**

```tsx
<DocumentUpload
  userId={user.user_id}
  onUploadSuccess={(data) => console.log('Uploaded:', data)}
  onUploadError={(error) => console.error('Error:', error)}
/>
```

**Component Features:**
- Progress bar during upload
- Supports: PDF, DOCX, DOC, TXT, PNG, JPG, JPEG
- Shows upload stats on success
- Error display with retry option
- Auto-reset after successful upload

---

### 2. Integration in Chat Page
**File:** `frontend/src/pages/Chat.tsx`

The DocumentUpload component is integrated directly into the Chat interface:

**Features:**
- **Upload Button** - Toggle button (📎 icon) in the input area
- **Upload Panel** - Collapsible section above the main input
- **Success Notification** - Shows extraction stats:
  - Facts extracted
  - Entities identified
  - Relationships created
  - Text length
- **Error Display** - Clear error messages with suggested fixes

**Usage:**
1. Click the upload icon next to the text input
2. Upload a document
3. View extraction stats in the success notification
4. Continue chatting—document content is now in your knowledge graph

---

### 3. API Integration
**File:** `frontend/src/lib/api.ts`

```typescript
export const documentAPI = {
  // Extract text from document
  uploadDocument: async (file: File) => {...}
  
  // Ingest extracted text into graph
  ingestDocument: async (
    documentText: string,
    documentName: string,
    documentFormat: string,
    userId: string,
    metadata?: Record<string, any>
  ) => {...}
  
  // Combined upload + ingest (recommended)
  uploadAndIngest: async (file: File) => {...}
};
```

---

## Integration with Existing Pipeline

The document upload feature **reuses the entire existing ingestion pipeline**:

1. **Text Extraction** → `TextExtractor` service
2. **LLM Processing** → Reuses `LLMExtractor` (same as chat)
3. **Graph Ingestion** → Reuses `MemoryOrchestrator` (same as chat)
4. **Data Structure** → Same Neo4j schema
5. **Deferred Reinforcement** → Same ranking & citation system

**Key Benefit:** No changes to core ingestion logic. Documents flow through identical processing as manual chat messages.

---

## Usage Examples

### Backend - Python/FastAPI

**Extract text from document:**
```python
from services.extraction.text_extractor import text_extractor

text, metadata = text_extractor.extract_from_bytes(file_bytes, "report.pdf")
print(f"Extracted {len(text)} characters from {metadata['format']} file")
```

**Upload and process document:**
```bash
curl -X POST "http://localhost:8000/documents/upload-and-ingest" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@report.pdf"
```

### Frontend - React/TypeScript

**Show upload dialog:**
```tsx
const [showUpload, setShowUpload] = useState(false);

<button onClick={() => setShowUpload(!showUpload)}>
  Upload Document
</button>

{showUpload && (
  <DocumentUpload
    userId={user.user_id}
    onUploadSuccess={(result) => {
      console.log('Ingested:', result.extraction_stats);
    }}
  />
)}
```

---

## Supported File Types & Limitations

| Format | Status | Max Size | Limitations |
|--------|--------|----------|-------------|
| PDF | ✅ Supported | 50MB | Text-based PDFs only (scanned PDFs need OCR) |
| DOCX | ✅ Supported | 50MB | Tables converted to pipe-separated format |
| DOC | ✅ Supported | 50MB | Legacy format (use DOCX when possible) |
| TXT | ✅ Supported | 50MB | UTF-8 encoding required |
| PNG/JPG/JPEG | ✅ Supported | 50MB | Extracts text via PaddleOCR |

---

## Recent Changes & Migration Notes

### What Was Added
- `backend/services/extraction/text_extractor.py` - Text extraction service
- `backend/api/routes/documents.py` - Document upload API routes
- `frontend/src/components/DocumentUpload.tsx` - Upload UI component
- Updated `backend/api/models.py` - New request/response types
- Updated `frontend/src/lib/api.ts` - Document API functions
- Updated `frontend/src/pages/Chat.tsx` - Integrated upload UI
- Updated `backend/requirements.txt` - Added PyPDF2, python-docx

### What Wasn't Changed
- ✅ **Core ingestion pipeline** - All original logic preserved
- ✅ **Neo4j schema** - No changes to graph structure
- ✅ **LLMExtractor** - Uses existing extraction logic
- ✅ **MemoryOrchestrator** - Reuses existing flow
- ✅ **GraphRetrieval** - No changes to retrieval
- ✅ **Chat endpoint** - Completely independent
- ✅ **Authentication** - Integrated seamlessly

---

## Error Handling

### Common Errors & Solutions

**"Unsupported file format"**
- Cause: File extension not supported
- Fix: Use PDF, DOCX, DOC, TXT, or PNG/JPG (future)

**"File too large"**
- Cause: File exceeds 50MB
- Fix: Split large files or compress before upload

**"Could not extract any text"**
- Cause: Scanned PDF or image-based document
- Fix: Wait for OCR support (planned) or convert to text first

**"Invalid token"**
- Cause: Expired or missing JWT
- Fix: Re-authenticate and retry

**"Error during extraction"**
- Cause: Corrupted file or unsupported encoding
- Fix: Verify file integrity and try again

---

## Future Enhancements

### Planned Features
1. **Batch Upload** - Process multiple documents in one request
3. **Document Chunking** - Automatic chunking for vector retrieval
4. **Extraction Preview** - Show extracted text before ingestion
5. **Format Preservation** - Better handling of tables and structured data
6. **Progress Tracking** - Real-time progress for large files
6. **Document History** - Track which documents were ingested
7. **Selective Ingestion** - Choose which sections to ingest

### Potential Optimizations
- Streaming upload for very large files
- Async processing queue for bulk uploads
- Document vectorization for semantic search
- Deduplication across multiple document uploads
- Document-level confidence scoring

---

## Testing

### Backend Tests

```bash
# Test text extraction
python -m pytest tests/test_text_extractor.py

# Test API routes
python -m pytest tests/test_documents_api.py

# Test end-to-end flow
python -m pytest tests/test_document_ingestion_e2e.py
```

### Frontend Tests

```bash
# Test DocumentUpload component
npm test -- DocumentUpload.test.tsx

# Test Chat integration
npm test -- Chat.test.tsx
```

### Manual Testing Checklist

- [ ] Upload PDF document
- [ ] Upload DOCX document
- [ ] Upload TXT document
- [ ] Verify extraction stats
- [ ] Check Neo4j graph for ingested data
- [ ] Verify facts/entities in retrieval
- [ ] Test with large files (>10MB)
- [ ] Test with unsupported formats
- [ ] Test with empty files
- [ ] Test with corrupted files
- [ ] Verify error messages
- [ ] Test without authentication

---

## Configuration

### Environment Variables

No new environment variables required. Uses existing:
- `NEO4J_URI` - Graph database connection
- `GEMINI_API_KEY` - LLM extraction

### Dependencies Added

```txt
PyPDF2==3.0.1
python-docx==0.8.11
pillow==10.1.0
paddlepaddle>=2.6.0
paddleocr==2.7.0.3
pdf2image==1.16.3
```

---

## Questions & Support

### FAQ

**Q: Does document upload replace chat?**
A: No, documents complement chat. Both feed into the same knowledge graph.

**Q: Can I upload multiple documents at once?**
A: Currently one at a time (batch upload planned).

**Q: How long does extraction take?**
A: Depends on file size. PDFs typically: <1s for <1MB, few seconds for larger files.

**Q: Are documents stored?**
A: Only the extracted content is stored in Neo4j. Original file is not persisted.

**Q: Can I delete document content?**
A: Currently, no. Plan to add document deletion (will remove related nodes).

**Q: What about confidential documents?**
A: Ensure you trust the system before uploading sensitive content. Content goes through LLM extraction.

---

## Appendix: Schema Reference

### Document-related Nodes (Neo4j)

```cypher
# Message node (created for document)
(Message {
  id: "msg_...",
  user_id: "user_123",
  text: "Document: financial_report.pdf",
  timestamp: datetime(),
  source_type: "document",  // New source type
  created_at: datetime()
})

# Facts extracted from document
(Fact {
  id: "fact_...",
  user_id: "user_123",
  text: "Invested ₹50,000 in HDFC",
  confidence: 0.95,
  timestamp: datetime(),
  reinforcement_count: 0,
  source: "document",  // Track source
  document_name: "financial_report.pdf"
})

# Relationships (same as chat)
(Message)-[:DERIVED_FACT]->(Fact)
(Fact)-[:CONFIRMS]->(Transaction)
```

###Relationships Flow

```cypher
# Document source tracking
(Message {source_type: "document"})-[:DERIVED_FACT]->(Fact)-[:CONFIRMS]->(Entity)

# Same retrieval ranking applies
# No schema changes, pure data
```

---

## Conclusion

Document upload is fully integrated with GraphMind's existing architecture, reusing all core services while providing a new ingestion pathway for non-chat content. The feature is extensible and ready for enhancements like OCR and batch processing.
