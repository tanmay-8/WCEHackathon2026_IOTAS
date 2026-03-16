# OCR Feature Guide - Scanned PDF Support

## Overview

GraphMind now supports **Optical Character Recognition (OCR)** for extracting text from scanned PDFs and image files. This uses **PaddleOCR**, a free and open-source OCR engine.

## What's Supported

### Document Types

| Format | Method | Status |
|--------|--------|--------|
| **PDF (Text-based)** | Direct text extraction (PyPDF2) | ✅ Fast |
| **PDF (Scanned)** | OCR (PaddleOCR) | ✅ Automatic detection |
| **Images** (.png, .jpg, .jpeg) | OCR (PaddleOCR) | ✅ Full support |
| **Word** (.docx, .doc) | Direct extraction | ✅ Fast |
| **Plain Text** (.txt) | Direct extraction | ✅ Fast |

## How It Works

### Automatic Scanned PDF Detection

When you upload a PDF:

1. **First attempt**: Direct text extraction using PyPDF2
2. **Detection**: If extracted text is < 100 characters, it's marked as "scanned"
3. **Fallback**: Automatically switches to OCR processing
4. **Result**: Pages are converted to images and text is extracted using PaddleOCR

### OCR Processing Flow

```
Scanned PDF/Image
    ↓
Convert to Images (pdf2image)
    ↓
Run PaddleOCR on each page
    ↓
Filter results (confidence > 0.3)
    ↓
Combine text from all pages
    ↓
Process through LLM (extract facts, entities, relationships)
    ↓
Store in Neo4j Knowledge Graph
```

## Installation

### 1. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

This installs:
- `paddleocr==2.7.0.3` - OCR engine
- `pdf2image==1.16.3` - PDF to image conversion

### 2. System Requirements

#### For PDF to Image Conversion

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download from https://github.com/oschwartz10612/poppler-windows/releases/ and add to PATH

### 3. First Run

The first time you use OCR, PaddleOCR will download required models (~100MB):
- English detection and recognition models
- This is automatic and happens once

## Usage

### Via Frontend

1. Click **📎 Upload Document** in Chat
2. Select a PDF or image file
3. Wait for processing (showing progress %)
4. See results:
   - ✅ Extraction method (PyPDF2 or PaddleOCR)
   - ✅ Whether it was scanned
   - ✅ Extracted facts, entities, relationships

### Via API

**Upload endpoint** returns:
```json
{
  "success": true,
  "filename": "scanned_document.pdf",
  "extracted_text": "Full text extracted via OCR...",
  "metadata": {
    "extraction_method": "PaddleOCR (Scanned PDF)",
    "is_scanned": true,
    "format": "PDF (Scanned)",
    "total_pages": 5
  }
}
```

## API Response Fields

### DocumentUploadResponse

```python
{
  "success": bool,
  "filename": str,
  "format": str,  # "PDF (Scanned)" for OCR
  "extraction_method": str,  # "PaddleOCR (Scanned PDF)" or "PyPDF2"
  "text_preview": str,
  "metadata": {
    "is_scanned": bool,  # True if OCR was used
    "extraction_method": str,
    "total_pages": int,
    ...
  },
  "s3_key": str,
  "s3_url": str
}
```

## Performance & Limitations

### Performance

| Type | Time per Page | Notes |
|------|---------------|-------|
| Text PDF | Instant | Direct extraction |
| Scanned PDF | 1-3 seconds | Depends on image quality |
| Image | 1-3 seconds | Depends on resolution |

### Accuracy

- **High quality scans**: 95%+ accuracy
- **Compressed images**: 80-90% accuracy
- **Poor quality/blurry**: 50-70% accuracy

### Limitations

1. **Language**: Currently English only (can be extended to multilingual)
2. **Rotation**: Automatically handles rotated text
3. **Handwriting**: Not supported (OCR is for printed text)
4. **Complex layouts**: May struggle with heavy tables/graphics

## Troubleshooting

### Issue: "PaddleOCR not initialized"

**Solution**: Make sure dependencies are installed:
```bash
pip install paddleocr pdf2image
```

### Issue: "pdf2image: Error - poppler not found"

**Solution**: Install poppler (see System Requirements above)

### Issue: Slow OCR processing

**Cause**: First run downloads models (~100MB). Subsequent runs are faster.

**Solution**: Be patient on first run, or download models manually:
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
```

### Issue: "Could not extract any text"

**Cause**: OCR couldn't recognize text (very low quality, non-text content)

**Solution**:
- Improve image quality/scanning resolution
- Ensure text is printed (not handwritten)
- Check if file is actually a document

## Advanced Configuration

### Changing OCR Language

In `backend/services/extraction/text_extractor.py`, line ~56:

```python
# For multilingual support
self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # Chinese
self.ocr = PaddleOCR(use_angle_cls=True, lang='es')  # Spanish
```

Supported languages: `en`, `ch`, `fr`, `de`, `es`, `pt`, `ru`, `ar`, `hi`, `ja`, `ko`, etc.

### Adjusting Confidence Threshold

In `backend/services/extraction/text_extractor.py`, line ~450:

```python
if confidence > 0.3:  # Change this value (0.0 to 1.0)
    text_lines.append(text)
```

- Lower value: Include more text (less accurate)
- Higher value: Only high-confidence text (may miss content)

## Future Enhancements

- [ ] Multilingual OCR support
- [ ] Handwriting recognition
- [ ] Table extraction with structure preservation
- [ ] Performance optimization (batch processing)
- [ ] Configuration via UI
- [ ] Language detection (automatic)

## References

- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
- **pdf2image**: https://github.com/Belval/pdf2image
- **Poppler**: https://poppler.freedesktop.org/

## Cost

✅ **Completely free**
- PaddleOCR: MIT License (free & open source)
- No API calls
- No cloud dependencies
- All processing happens locally
