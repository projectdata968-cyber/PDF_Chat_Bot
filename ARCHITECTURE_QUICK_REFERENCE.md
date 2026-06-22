# OCR RAG PDFBot - Architecture Quick Reference

## 1. Request Flow: PDF Upload → Answer Generation

### **Ingestion Flow (Upload Endpoint)**
```
User uploads PDFs
    ↓ [DocumentProcessor]
Validate → Save → Detect Type → Extract Text
    ↓
If Digital PDF → PDFParser (PyMuPDF)
If Scanned PDF → OCRProcessor (PaddleOCR)
    ↓
Enrich Metadata → Chunk Text (500 tokens) → Generate Embeddings
    ↓ [Embedding Model: SentenceTransformer BAAI/bge-small-en-v1.5]
Create 384-dim vectors
    ↓ [ChromaDB]
Store in SQLite vector database
    ↓
Return: {documents_processed, chunks_created, vectorstore_count}
```

### **Query Flow (Chat Endpoint)**
```
User asks question
    ↓ [Retriever]
Embed query → Search ChromaDB (top-5 similar chunks)
    ↓ [RAG Pipeline]
Build context from retrieved chunks with citations
    ↓ [Prompt Engineering]
System: "You are a helpful PDF assistant"
Context: [Retrieved chunks formatted]
Question: [User question]
    ↓ [OpenRouter LLM]
Generate answer using openrouter/auto (temperature=0.2)
    ↓
Return: {question, answer, sources}
```

---

## 2. Document Processor Orchestration

**Key Decision**: Detect PDF type and route to appropriate extractor

```python
# Pseudo-code
if is_scanned_pdf(pdf_path):  # Check if has extractable text
    # Heavy computation: Convert pages to 300 DPI images + OCR
    docs = ocr_processor.extract_text_with_ocr(pdf_path)
else:
    # Fast: Native PDF text extraction
    docs = pdf_parser.extract_text(pdf_path)

# Then: Enrich metadata, chunk, embed, store
```

**Why Separate?**
- Digital PDFs: Fast, native text extraction
- Scanned PDFs: Requires image → text conversion (PaddleOCR)
- Allows disabling OCR for speed when processing digital PDFs only
- Enables switching OCR engines independently

---

## 3. RAG Pipeline Orchestration

**4-Step Process:**
1. **Retrieve**: Query embedding → similarity search in ChromaDB
2. **Build Context**: Format retrieved chunks with citations
3. **Construct Prompt**: System instructions + context + question
4. **Generate**: Send prompt to OpenRouter LLM → get answer

**Why Separate Components?**
- **Retriever**: Can add reranking, filtering, caching
- **Context Builder**: Can implement different formatting strategies
- **LLM Client**: Can swap between providers (OpenAI, Anthropic, local LLM)

---

## 4. ChromaDB Storage Strategy

**Location**: `data/vector_store/chroma.sqlite3` (persistent SQLite)

**Schema for Each Stored Chunk:**
```python
{
    "id": "doc_<uuid>",                    # Unique ID
    "text": "chunk content",               # Text content
    "embedding": [0.1, 0.2, ...],         # 384-dim normalized vector
    "metadata": {
        "page": 5,                         # Page number
        "filename": "report.pdf",          # Original filename
        "citation": "report.pdf (Page 5)", # Citation format
        "ocr": false,                      # Was OCR used?
        "tokens": 450,                     # Token count
        "document_id": 1                   # Sequential ID
    }
}
```

**Search Method**: Cosine similarity over embeddings + metadata filtering

**Scaling Path**: SQLite → Pinecone / Weaviate / Milvus

---

## 5. OpenRouter Integration Strategy

**Current Setup:**
- **Model**: `openrouter/auto` (auto-selects cheapest free model)
- **Temperature**: 0.2 (factual, low randomness)
- **System Role**: "You are a helpful PDF assistant"
- **API**: OpenAI SDK-compatible

**Prompt Pattern:**
```
System: You are a helpful PDF assistant.
User: Use context to answer. Cite sources [Source: Filename (Page X)].

Context:
[Retrieved chunks with citations]

Question:
{user_question}
```

**Scaling Paths:**
- Higher quality: Switch to GPT-4/Claude on paid tier
- Lower cost: Use smaller models (Llama, Mistral)
- Lower latency: Use local LLM (Ollama, vLLM)

---

## 6. Data Flow Between Modules

```
PDF File (binary)
    ↓ save to temp/
    ↓ [DocumentProcessor] → detect type
    ├─ Digital? → [PDFParser] → per-page text
    └─ Scanned? → [OCRProcessor] → per-page OCR
    ↓
Per-page documents: {text, metadata}
    ↓ [MetadataExtractor] → enrich metadata
    ↓
Enriched documents: {text, metadata}
    ↓ [TextChunker] → token-aware chunking
    ↓
Chunks: {text, metadata: {tokens, citation, page, ...}}
    ↓ [EmbeddingModel] → batch encode
    ↓
Chunks + Embeddings (384-dim vectors)
    ↓ [ChromaDB] → store with ID + text + embedding + metadata
    ↓
Vector database (persistent)

---

Query Path:
User Question
    ↓ [EmbeddingModel] → 384-dim query vector
    ↓ [ChromaDB] → similarity search (top-5)
    ↓
Retrieved chunks + metadata
    ↓ [RAGPipeline.build_context] → format with citations
    ↓
Context string + Question
    ↓ [OpenRouter LLM] → generate answer
    ↓
Answer + Sources (from metadata)
```

---

## 7. Why Each Module Exists

| Module | Why | Can Be Replaced With |
|--------|-----|-------------------|
| **PDFParser** | Fast native PDF text extraction | pdfplumber, PyPDF2 |
| **OCRProcessor** | Extract text from image-based PDFs | Tesseract, EasyOCR |
| **TextChunker** | Token-aware splitting (important for LLM context windows) | LangChain's RecursiveCharacterTextSplitter |
| **MetadataExtractor** | Standardize document metadata (citations, document IDs) | Custom enrichment logic |
| **EmbeddingModel** | Generate semantic vectors for retrieval | OpenAI embeddings API, Cohere |
| **ChromaDB** | Persistent vector storage | Pinecone, Weaviate, Milvus |
| **Retriever** | Semantic search + filtering | Different ranking/reranking strategies |
| **RAGPipeline** | Orchestrate retrieval + generation | Different RAG strategies (multi-turn, reasoning) |
| **OpenRouterClient** | LLM API abstraction | OpenAI, Anthropic, local LLM |

---

## 8. Separation of Concerns

Each module has **single responsibility**:

- **Document Extraction**: PDFParser, OCRProcessor (text → chunks)
- **Document Preparation**: MetadataExtractor, TextChunker (format → storage)
- **Vectorization**: EmbeddingModel (text → embeddings)
- **Storage**: ChromaDB (embeddings ↔ disk)
- **Retrieval**: Retriever (query → context)
- **Generation**: RAGPipeline + OpenRouterClient (context + question → answer)
- **API**: Routes (HTTP ↔ logic)

**Benefits:**
- ✅ Easy to test (mock dependencies)
- ✅ Easy to replace (swap implementations)
- ✅ Easy to scale (optimize independently)
- ✅ Easy to maintain (clear boundaries)

---

## 9. Production Scaling Strategy

### Phase 1: Current (Development)
- ✅ PDF storage: Local filesystem
- ✅ Vector DB: SQLite
- ✅ Embeddings: CPU (SentenceTransformer)
- ✅ LLM: OpenRouter free tier
- 📊 Capacity: ~1K documents

### Phase 2: 10K+ Documents
```
Current               →         Production
─────────────────────────────────────────
Local filesystem      →         S3 / Azure Blob
SQLite                →         Pinecone / Weaviate
CPU embeddings        →         GPU batch processing
OpenRouter free       →         Cached responses
Sync processing       →         Async queue (Celery)
Single FastAPI        →         Kubernetes deployment
```

### Phase 3: 100K+ Documents
- Multi-tenancy with RBAC
- Distributed OCR processing
- Query result caching
- Hybrid search (BM25 + vector)
- Advanced RAG (query expansion, reranking)

---

## 10. Architecture Benefits

| Principle | Implementation | Benefit |
|-----------|----------------|---------|
| **Layered Architecture** | API → Processing → Retrieval → LLM | Clear separation, easy to debug |
| **Strategy Pattern** | Select extraction based on PDF type | Flexible, testable |
| **Facade Pattern** | RAGPipeline hides complexity | Simple interface |
| **Dependency Injection** | Singleton instances in routes | Easy to mock for testing |
| **Configuration Management** | Centralized settings.py | Environment-aware, secure |
| **Logging** | Centralized logger | Consistent monitoring |
| **Async I/O** | aiofiles for file operations | Scalable to high concurrency |
| **Metadata Preservation** | Citation + source tracking | Traceability + fact-checking |

---

## Quick Start for Developers

### Understanding the Flow
1. **Need to add a new PDF extraction method?** → Modify `DocumentProcessor` extraction logic
2. **Need to improve retrieval quality?** → Tune `Retriever` or `EmbeddingModel`
3. **Need to customize answers?** → Modify `RAGPipeline.build_context()` or prompt in `OpenRouterClient`
4. **Need to switch LLM providers?** → Replace `OpenRouterClient` with new implementation
5. **Need to scale to 100K docs?** → Replace `ChromaDB` with Pinecone in `ChromaVectorStore`

### Testing Strategy
- **Unit tests**: Mock `DocumentProcessor`, `Retriever`, `OpenRouterClient`
- **Integration tests**: Test full upload → chat flow
- **Performance tests**: Measure embedding generation, retrieval latency

---

## Configuration Overview

| Setting | Location | Purpose |
|---------|----------|---------|
| API Keys | `.env` → `config/settings.py` | OpenRouter credentials |
| Model Selection | `config/settings.py` | Default LLM model |
| Storage Paths | `config/settings.py` | Temp files, vector DB |
| Chunk Size | `core/chunker.py` | 500 tokens default |
| Embedding Model | `embeddings/embedding_model.py` | BAAI/bge-small-en-v1.5 |
| LLM Temperature | `llm/openrouter_client.py` | 0.2 (factual) |
| Top-K Results | `api/schemas.py` | Default 5, max 20 |
| OCR Threads | `core/ocr_processor.py` | 4 parallel threads |

---

## File Locations

```
backend/
├── main.py                           # App entry point
├── api/
│   ├── routes.py                     # HTTP endpoints (5 endpoints)
│   └── schemas.py                    # Request/response validation
├── config/
│   └── settings.py                   # Configuration management
├── core/
│   ├── document_processor.py         # Ingestion orchestration
│   ├── pdf_parser.py                 # Digital PDF extraction
│   ├── ocr_processor.py              # Scanned PDF extraction
│   ├── chunker.py                    # Token-aware chunking
│   └── metadata_extractor.py         # Metadata enrichment
├── embeddings/
│   └── embedding_model.py            # Vector generation
├── vectordb/
│   └── chroma_store.py               # ChromaDB integration
├── rag/
│   ├── rag_pipeline.py               # QA orchestration
│   └── retriever.py                  # Semantic search
├── llm/
│   └── openrouter_client.py          # LLM API integration
└── utils/
    └── logger.py                     # Centralized logging
```

---

## Key Metrics & Monitoring

```
Ingestion Pipeline:
- PDF processing time (per page)
- OCR confidence score (for scanned PDFs)
- Chunk size distribution (tokens)
- Embedding generation time (batch)

Query Pipeline:
- Retrieval latency (embedding + search)
- Context quality (relevance of top-5)
- LLM generation time (tokens/sec)
- Answer accuracy (user feedback)

System Health:
- Vector DB size (number of chunks, disk usage)
- API response time (P50, P95, P99)
- Error rates (by endpoint)
- Cache hit rate
```

