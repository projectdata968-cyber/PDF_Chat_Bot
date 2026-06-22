# OCR RAG PDFBot Architecture

## Table of Contents
1. [Overview](#overview)
2. [Request Flow](#request-flow)
3. [Module Responsibilities](#module-responsibilities)
4. [Data Flow](#data-flow)
5. [Separation of Concerns](#separation-of-concerns)
6. [Key Orchestration Patterns](#key-orchestration-patterns)
7. [Storage Strategy](#storage-strategy)
8. [LLM Integration Strategy](#llm-integration-strategy)
9. [Production Scaling Strategy](#production-scaling-strategy)
10. [System Architecture Diagram](#system-architecture-diagram)

---

## Overview

OCR RAG PDFBot is a **Retrieval-Augmented Generation (RAG) system** designed to process both **digital and scanned PDFs** through a unified pipeline. It enables users to:

- **Upload PDFs** (digital or scanned)
- **Process documents** with intelligent extraction (native PDF parsing or OCR)
- **Generate embeddings** for semantic search
- **Store embeddings** in a persistent vector database
- **Retrieve relevant context** via similarity search
- **Generate answers** using an LLM (OpenRouter)

The architecture is built on **layered separation of concerns**, with each module handling a specific responsibility in the document processing and retrieval pipeline.

---

## Request Flow

### 1. PDF Upload to Answer Generation (End-to-End)

```
User Request (POST /upload_and_process_pdfs)
    ↓
[API Router] - FastAPI endpoint validates request
    ↓
[Document Processor] - Orchestrates ingestion pipeline
    ├─ Validate PDF file (format, size)
    ├─ Save uploaded file to temp directory
    ├─ Detect PDF type (digital vs. scanned)
    │   ├─ IF digital → [PDF Parser] (extract text with PyMuPDF)
    │   └─ IF scanned → [OCR Processor] (extract via PaddleOCR)
    ├─ Extract per-page metadata
    └─ Return documents with metadata
    ↓
[Metadata Extractor] - Enrich document metadata
    ├─ Add document ID
    ├─ Add filename
    ├─ Add citation format (filename + page number)
    └─ Return enriched documents
    ↓
[Text Chunker] - Token-aware segmentation
    ├─ Count tokens (using tiktoken)
    ├─ Split documents into overlapping chunks (500-token chunks with 50-token overlap)
    ├─ Preserve metadata for each chunk
    └─ Return chunks with token counts
    ↓
[Embedding Model] - Generate vector embeddings
    ├─ Use SentenceTransformer (BAAI/bge-small-en-v1.5)
    ├─ Encode all chunks to embeddings (normalized)
    └─ Return embedding vectors
    ↓
[ChromaDB Vector Store] - Persist embeddings
    ├─ Generate unique document IDs
    ├─ Store documents, embeddings, and metadata
    ├─ Create collection if not exists
    └─ Return storage success
    ↓
API Response (StandardAPIResponse)
    └─ Return ingestion stats (documents_processed, chunks_created, vectorstore_count)
```

### 2. Chat Query to Answer Generation (QA Flow)

```
User Request (POST /chat)
    ↓
[API Router] - Validates ChatRequest
    ↓
[RAG Pipeline] - Orchestrates Q&A pipeline
    ├─ Query → [Retriever]
    │   ├─ Embed query using [Embedding Model]
    │   ├─ Semantic search in [ChromaDB Vector Store]
    │   ├─ Retrieve top-k (default=5) similar chunks
    │   └─ Return documents with metadata
    │
    ├─ Build Context from retrieved chunks
    │   ├─ Format retrieved documents
    │   ├─ Add citation information
    │   └─ Concatenate into context string
    │
    ├─ Construct Prompt
    │   ├─ Add system role (PDF assistant)
    │   ├─ Include retrieved context
    │   ├─ Append user question
    │   └─ Request source citations
    │
    └─ Generate Response
        ├─ Send prompt to [OpenRouter Client]
        ├─ Use model: openrouter/auto (free tier)
        ├─ Apply temperature=0.2 (low randomness)
        └─ Return LLM-generated answer
    ↓
API Response (StandardAPIResponse)
    └─ Return QA result (question, answer, sources)
```

---

## Module Responsibilities

### **Tier 1: API & Configuration Layer**

#### `api/routes.py`
**Purpose:** HTTP endpoint definitions and request routing
- Exposes 5 API endpoints:
  - `GET /health` - Health check
  - `POST /upload_and_process_pdfs` - Ingest & process PDFs
  - `GET /vector_store/count` - Get stored document count
  - `POST /vector_store/search` - Raw vector similarity search
  - `POST /chat` - RAG-based Q&A
- Instantiates singleton services (DocumentProcessor, EmbeddingModel, ChromaVectorStore, RAGPipeline)
- Handles request/response serialization
- Implements error handling with structured API responses

#### `api/schemas.py`
**Purpose:** Request/response validation and documentation
- `ChatRequest` - Validates user questions (min_length=1, top_k=1-20)
- `SearchQueryRequest` - Validates vector search queries
- `PDFProcessResponse` - Ingestion result schema
- `StandardAPIResponse[T]` - Generic response wrapper with type safety
- Provides OpenAPI documentation through Pydantic models

#### `config/settings.py`
**Purpose:** Centralized configuration management
- Loads environment variables from `.env`
- Defines file storage paths (temp upload directory, vector store directory)
- Stores API keys and base URLs (OpenRouter)
- Defines model options and default models
- Provides single source of truth for configuration

### **Tier 2: Document Processing Layer**

#### `core/document_processor.py`
**Purpose:** Orchestration of the document ingestion pipeline
**Key Responsibilities:**
1. **File Validation**
   - Checks file format (must be `.pdf`)
   - Validates file size (max 200 MB)
2. **File Persistence**
   - Saves uploaded files to temp directory using async I/O
   - Manages file paths for downstream processing
3. **PDF Type Detection**
   - Uses `is_scanned_pdf()` to determine extraction strategy
   - Checks if PDF has extractable text (>20 chars per page)
   - Returns `True` for scanned PDFs, `False` for digital PDFs
4. **Extraction Strategy Selection**
   - Routes to `PDFParser` for digital PDFs (faster, better quality)
   - Routes to `OCRProcessor` for scanned PDFs (slower, handles images)
5. **Metadata Enrichment**
   - Ensures consistent metadata structure
   - Sets filename, source path, OCR flag, citation format
   - Prepares documents for chunking
6. **Pipeline Coordination**
   - Calls `MetadataExtractor` to enrich metadata
   - Calls `TextChunker` to split documents
   - Returns final chunks ready for embedding

#### `core/pdf_parser.py`
**Purpose:** Extract text from digital (native text) PDFs
- Uses PyMuPDF (fitz) for fast, reliable text extraction
- Iterates through PDF pages
- Preserves page numbers in metadata
- Returns structured documents with metadata
- **Why separate:** Allows different implementations for different PDF types; OCR is computationally expensive and should only run on scanned PDFs

#### `core/ocr_processor.py`
**Purpose:** Extract text from scanned PDFs using vision AI
**Process:**
1. Convert PDF pages to high-resolution images (300 DPI)
2. Run PaddleOCR on each image with optimized settings:
   - `cpu_threads=4` for parallel processing
   - `det_limit_side_len=960` for cleaner text grouping
   - `confidence > 0.40` threshold for quality filtering
3. Extract text with confidence filtering
4. Return structured documents with OCR metadata
- **Why separate:** OCR is a distinct capability requiring different libraries; keeping it separate enables:
  - Disabling OCR for performance on digital PDFs only
  - Switching to alternative OCR engines
  - Monitoring OCR-specific metrics

#### `core/chunker.py`
**Purpose:** Token-aware text segmentation
**Strategy:**
- Splits documents into **500-token chunks** with **50-token overlap**
- Uses tiktoken to count tokens (encoder: `cl100k_base`)
- Preserves metadata for each chunk (including token count)
- Overlap ensures context continuity across chunk boundaries
- **Why this matters:**
  - Prevents context fragmentation (overlap)
  - Respects LLM context windows
  - Enables efficient embedding generation (embeddings-friendly sizes)
  - Token count tracking allows monitoring embedding costs

#### `core/metadata_extractor.py`
**Purpose:** Enrich and standardize document metadata
- Extracts filename from source path
- Generates unique document ID
- Creates standardized citation format: `filename (Page X)`
- Ensures consistent metadata structure across all chunks
- **Why separate:** Metadata handling is distinct from text extraction; allows:
  - Custom citation formats
  - Additional metadata enrichment (timestamps, content hashes)
  - Versioning of metadata structure

### **Tier 3: Embedding & Vector Storage Layer**

#### `embeddings/embedding_model.py`
**Purpose:** Generate semantic embeddings for documents and queries
- Uses **SentenceTransformer** with pre-trained model `BAAI/bge-small-en-v1.5`
  - Lightweight (~33M parameters, 42 MB)
  - Fast inference on CPU
  - Normalized embeddings for cosine similarity (384-dim vectors)
- Methods:
  - `embed_documents(texts)` - Batch embedding for storage
  - `embed_query(query)` - Single query embedding for retrieval
- Normalization ensures optimal dot-product similarity scoring
- **Why this model:**
  - Bilingual (English + Chinese)
  - Excellent retrieval performance on MTEB benchmarks
  - Runs efficiently on CPU (no GPU required)
  - Open source and free

#### `vectordb/chroma_store.py`
**Purpose:** Persistent vector database management
**Storage Strategy:**
- Uses **ChromaDB** with persistent SQLite backend
- Storage path: `data/vector_store/` (survives application restarts)
- Single collection: `pdf_documents` (all documents in one collection)
- Schema for each stored chunk:
  ```
  {
    "id": "doc_<uuid>",              # Unique identifier
    "text": "chunk content",          # Actual text
    "embedding": [0.1, 0.2, ...],   # 384-dim vector
    "metadata": {
      "page": 1,
      "filename": "doc.pdf",
      "citation": "doc.pdf (Page 1)",
      "ocr": false,
      "tokens": 450,
      "document_id": 1
    }
  }
  ```
- Methods:
  - `add_documents(chunks, embeddings)` - Batch insert
  - `similarity_search(query_embedding, filename, top_k)` - Vector search with optional filename filter

### **Tier 4: Retrieval-Augmented Generation (RAG) Layer**

#### `rag/retriever.py`
**Purpose:** Semantic search and context retrieval
**Process:**
1. Embed user query using `EmbeddingModel`
2. Search ChromaDB for similar chunks (top-k, default=5)
3. Optional filtering by filename (single-document search)
4. Return retrieved documents with metadata
- **Why separate:** Retrieval is distinct from generation; allows:
  - Swapping different retrieval strategies
  - Adding reranking or filtering
  - Monitoring retrieval quality independently

#### `rag/rag_pipeline.py`
**Purpose:** Orchestrate retrieval-augmented generation
**Process:**
1. **Retrieve:** Call `Retriever` to get top-k similar chunks
2. **Build Context:** Format retrieved chunks with citations
   ```
   [Source: filename (Page X)]
   <chunk text>
   
   [Source: filename (Page Y)]
   <chunk text>
   ```
3. **Construct Prompt:** Create instruction to LLM
   ```
   System: You are a helpful PDF assistant.
   User: Use the provided context to answer...
   [Context from retrieved chunks]
   Question: <user query>
   Answer clearly and cite sources.
   ```
4. **Generate:** Call `OpenRouterClient` to generate answer
5. **Return:** Structured response with question, answer, and sources
- **Why separate:** Allows different RAG strategies:
  - Adding intermediate steps (reranking, summarization)
  - Chain-of-thought reasoning
  - Multi-step queries

### **Tier 5: LLM Integration Layer**

#### `llm/openrouter_client.py`
**Purpose:** LLM API abstraction and response generation
- Uses **OpenRouter API** for model access
- Current model: `openrouter/auto` (free tier - auto-selects cheapest)
- Configuration:
  - `temperature=0.2` - Low randomness (factual answers)
  - System role: "You are a helpful PDF assistant"
  - Uses OpenAI SDK (compatible API)
- Abstracts LLM implementation:
  - Can switch models (e.g., to paid models for better quality)
  - Can add preprocessing (prompt templates, validation)
  - Can implement token counting/cost tracking
- **Why OpenRouter:**
  - Free tier available for testing
  - Supports multiple LLM providers
  - Automatic model selection
  - Cost-efficient

### **Tier 6: Utilities**

#### `utils/logger.py`
**Purpose:** Centralized logging configuration
- Provides single logger instance for entire application
- Format: `[timestamp] [level] - message`
- Stream output to console (DEBUG level)
- Enables easy integration with log aggregation (ELK, CloudWatch)
- **Why centralized:** Allows:
  - Consistent log format
  - Easy switching to file/remote logging
  - Performance monitoring and debugging

### **Tier 7: Application Entry Point**

#### `main.py`
**Purpose:** FastAPI application initialization and lifecycle management
- Configures PaddleOCR environment flags for optimal performance
- Defines application lifespan (startup/shutdown)
- Initializes ChromaDB on startup
- Registers routers
- Runs UV server on `http://127.0.0.1:8000`

---

## Data Flow

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                 (Upload PDFs / Ask Questions)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                    HTTP Requests
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   INGESTION PATH                    QUERY PATH
   (Upload Endpoint)                 (Chat Endpoint)
        │                                 │
        ├─ File Validation               │
        ├─ PDF Type Detection            │
        │   ├─ Digital? → PDF Parser     │
        │   └─ Scanned? → OCR Processor  │
        │                                 │
        ├─ Text Extraction               │
        ├─ Metadata Enrichment           │
        ├─ Chunking (500-token)          │
        │                                 │
        ├─ Embedding Generation          ◄──┐
        │                                    │
        ├─ ChromaDB Storage                  │
        │                                    │
        └────────────────┬───────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
      Vector DB Storage          Query Embedding
      (Persistent SQLite)        (Via Embedding Model)
            │                         │
            └────────────┬────────────┘
                         │
                    Similarity Search
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
      Retrieved Chunks          User Question
            │                         │
            └────────────┬────────────┘
                         │
                 Context Building
                         │
                    Prompt Assembly
                         │
                         ▼
                  OpenRouter LLM API
                         │
                         ▼
                  LLM-Generated Answer
                         │
                         ▼
           ┌──────────────────────────┐
           │   Response to User       │
           │ (Answer + Sources)       │
           └──────────────────────────┘
```

### Detailed Data Transformations

#### Ingestion Pipeline

```
Raw PDF File (Binary)
    │
    ├─ Saved to: temp/uploaded_files/filename.pdf
    │
    ▼
PDF Type Detection (is_scanned_pdf)
    │
    ├─ Digital PDF (has extractable text)
    │   │
    │   ├─ Input: PDF file path
    │   ├─ Process: PyMuPDF extract text per page
    │   └─ Output: List[{text, metadata}]
    │       └─ metadata: {page, source, ocr: false}
    │
    └─ Scanned PDF (image-based)
        │
        ├─ Input: PDF file path
        ├─ Process: PaddleOCR per page (300 DPI)
        └─ Output: List[{text, metadata}]
            └─ metadata: {page, source, ocr: true}
    │
    ▼
Metadata Enrichment
    │
    ├─ Input: Document list + file object
    ├─ Process:
    │   ├─ Extract filename from file object
    │   ├─ Add document_id (incremental)
    │   ├─ Add citation: "filename (Page X)"
    │   └─ Overwrite filename & source fields
    └─ Output: List[{text, metadata}]
        └─ metadata: {page, source, ocr, filename, citation, document_id}
    │
    ▼
Text Chunking (Token-Aware)
    │
    ├─ Input: Document list with metadata
    ├─ Process:
    │   ├─ For each document:
    │   │   ├─ Split text into words
    │   │   ├─ Group into 500-word segments
    │   │   ├─ Apply 50-word overlap
    │   │   ├─ Count tokens via tiktoken
    │   │   └─ Create chunk with metadata
    │   └─ Flatten all chunks
    └─ Output: List[{text, metadata}]
        └─ metadata: {..., tokens: <count>}
    │
    ▼
Embedding Generation
    │
    ├─ Input: List of chunk texts
    ├─ Process:
    │   ├─ Batch encode with SentenceTransformer
    │   ├─ Normalize embeddings (L2)
    │   └─ Output 384-dimensional vectors
    └─ Output: List[List[float]]
        └─ Each vector: [0.123, -0.456, ...]
    │
    ▼
ChromaDB Storage
    │
    ├─ Input: Chunks + embeddings
    ├─ Process:
    │   ├─ Generate UUIDs for each chunk
    │   ├─ Prepare batch insert
    │   ├─ Store in: data/vector_store/chroma.sqlite3
    │   └─ Index by embedding for similarity
    └─ Output: Storage complete
        └─ API returns: {documents_processed, chunks_created, vectorstore_count}
```

#### Query Pipeline

```
User Question (String)
    │
    ▼
Embedding Encoding
    │
    ├─ Input: Query text
    ├─ Process: SentenceTransformer.encode(query, normalize=True)
    └─ Output: 384-dim normalized vector
    │
    ▼
Vector Similarity Search
    │
    ├─ Input: Query vector, top_k=5
    ├─ Process:
    │   ├─ Cosine similarity over all stored embeddings
    │   ├─ Return top-5 most similar chunks
    │   ├─ Include metadata for each result
    │   └─ Optional: filter by filename
    └─ Output:
        ├─ documents: [text1, text2, ..., text5]
        └─ metadatas: [{citation, page, ...}, ...]
    │
    ▼
Context Building
    │
    ├─ Input: Retrieved docs + metadatas
    ├─ Process:
    │   ├─ For each (doc, metadata) pair:
    │   │   ├─ Extract citation from metadata
    │   │   └─ Format: "[Source: citation]\n{doc}"
    │   └─ Join all formatted chunks with "\n\n"
    └─ Output: Context string
        └─ "[Source: file.pdf (Page 1)]\nText content...\n\n[Source: file.pdf (Page 2)]\nMore content..."
    │
    ▼
Prompt Assembly
    │
    ├─ Template:
    │   """
    │   Use the provided context to answer the question...
    │   
    │   Context:
    │   {context}
    │   
    │   Question:
    │   {query}
    │   
    │   Answer clearly and cite sources...
    │   """
    └─ Output: Full prompt ready for LLM
    │
    ▼
LLM Generation (OpenRouter)
    │
    ├─ Input: System message + prompt
    ├─ Process:
    │   ├─ Send to OpenRouter API
    │   ├─ Model: openrouter/auto (free)
    │   ├─ Temperature: 0.2 (factual)
    │   └─ Wait for response
    └─ Output: LLM-generated answer string
    │
    ▼
Response Assembly
    │
    ├─ Input: Question + answer + sources
    ├─ Process: Package into response dict
    └─ Output:
        {
          "question": "User's question",
          "answer": "LLM's answer with citations",
          "sources": [...metadata from retrieved chunks...]
        }
    │
    ▼
HTTP Response (200 OK)
    │
    └─ StandardAPIResponse[QAResult]
        {
          "status": "success",
          "data": {
            "question": "...",
            "answer": "...",
            "sources": [...]
          }
        }
```

---

## Separation of Concerns

This architecture exemplifies **separation of concerns** through layered, modular design:

| Module | Concern | Reason for Separation | Reusability |
|--------|---------|----------------------|------------|
| **PDF Parser** | Extract text from digital PDFs | Different extraction strategy from OCR; PyMuPDF is fast and deterministic | Can use for non-RAG text extraction |
| **OCR Processor** | Extract text from scanned PDFs | Requires heavy ML library; expensive computation | Can use for any OCR task |
| **Chunker** | Token-aware text segmentation | Orthogonal to extraction; reusable for different chunk sizes/overlap | Different chunking strategies |
| **Metadata Extractor** | Standardize document metadata | Metadata enrichment is independent of text extraction | Custom metadata handling |
| **Embedding Model** | Generate semantic vectors | Decoupled from storage; can switch embedding models | Different embedding services |
| **Vector Store** | Persist and search embeddings | Database abstraction; hides ChromaDB implementation details | Swap for Pinecone, Weaviate, etc. |
| **Retriever** | Semantic search logic | Retrieval strategy independent from generation | Different ranking/filtering |
| **RAG Pipeline** | Orchestrate retrieval + generation | Coordinates retriever + LLM; handles prompt engineering | Different RAG strategies |
| **LLM Client** | LLM API communication | Abstracts OpenRouter API; enables model switching | Different LLM providers |
| **Routes** | HTTP endpoint definitions | API layer independent from business logic | Easy to add endpoints |
| **Schemas** | Request/response validation | Separates validation from processing | Reusable Pydantic models |
| **Settings** | Configuration management | Centralized config; easy environment switching | Monolithic config file |
| **Logger** | Logging infrastructure | Consistent logging across app | Easy to integrate with ELK/CloudWatch |

### Benefits of This Separation

1. **Testability**: Each module can be unit-tested independently
   - Mock different PDF types
   - Test chunking without embeddings
   - Test retrieval without LLM calls

2. **Replaceability**: Swap implementations without changing other modules
   - Replace PyMuPDF with pdfplumber
   - Replace PaddleOCR with Tesseract
   - Replace ChromaDB with Pinecone
   - Replace OpenRouter with Anthropic/OpenAI

3. **Scalability**: Different modules can scale independently
   - Cache embeddings separately
   - Replicate vector DB
   - Load-balance LLM calls

4. **Performance Optimization**: Profile and optimize individual modules
   - Batch embeddings
   - Add retrieval caching
   - Rate-limit OCR processing

5. **Maintainability**: Clear responsibility boundaries
   - Easy to locate bugs
   - Easy to understand code
   - Easy to add features

---

## Key Orchestration Patterns

### 1. DocumentProcessor Orchestration (Ingestion)

**Pattern**: Sequential composition with strategy selection

```python
async def process_documents(self, files):
    # 1. Save files
    file_paths = await self.save_uploaded_files(files)
    
    # 2. Detect type and apply appropriate strategy
    for file_obj, file_path in zip(files, file_paths):
        scanned = self.is_scanned_pdf(file_path)
        
        if scanned:
            docs = self.ocr_processor.extract_text_with_ocr(file_path)
        else:
            docs = self.pdf_parser.extract_text(file_path)
        
        # 3. Enrich metadata
        for doc in docs:
            doc["metadata"]["filename"] = file_obj.filename
            doc["metadata"]["citation"] = f"{file_obj.filename} (Page {page})"
    
    # 4. Enrich and chunk
    enriched_docs = MetadataExtractor.enrich_metadata(all_docs)
    chunks = self.chunker.split_documents(enriched_docs)
    
    return chunks
```

**Key Patterns:**
- **Strategy Pattern**: Select extraction strategy based on PDF type
- **Decorator Pattern**: Enrich metadata at each stage
- **Pipeline Pattern**: Sequential processing with clear handoffs

### 2. RAGPipeline Orchestration (Query)

**Pattern**: Sequential orchestration with context building

```python
def ask(self, query: str) -> dict:
    # 1. Retrieve
    retrieved_docs = self.retriever.retrieve(query)
    
    # 2. Build context
    context = self.build_context(retrieved_docs)
    
    # 3. Construct prompt
    prompt = f"""
    Use provided context to answer...
    
    Context: {context}
    Question: {query}
    """
    
    # 4. Generate
    answer = self.llm.generate_response(prompt)
    
    # 5. Package
    return {
        "question": query,
        "answer": answer,
        "sources": retrieved_docs["metadatas"]
    }
```

**Key Patterns:**
- **Facade Pattern**: RAGPipeline abstracts complex multi-step process
- **Composition**: Combines Retriever + LLM
- **Immutable Context**: Retrieved docs don't change during generation

---

## Storage Strategy

### ChromaDB Design

**Persistence Model:**
- **Backend**: SQLite database (`data/vector_store/chroma.sqlite3`)
- **Survives**: Application restarts
- **Collection**: Single `pdf_documents` collection holds all documents
- **Indexing**: Automatic HNSW index for efficient similarity search

**Schema for Stored Chunks:**

```python
{
    "id": "doc_abc123def456",           # Unique UUID-based ID
    "text": "Chunk content...",         # The actual text
    "embedding": [0.123, -0.456, ...], # 384-dimensional vector
    "metadata": {
        "page": 5,                      # Page number in original PDF
        "filename": "report.pdf",       # Original filename
        "source": "/path/to/report.pdf",# Full file path
        "ocr": false,                   # Was OCR used?
        "document_id": 1,               # Sequential ID
        "tokens": 450,                  # Token count
        "citation": "report.pdf (Page 5)" # Formatted citation
    }
}
```

**Query Strategy:**
```python
# Similarity search
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5,                    # Top-5 most similar
    where={"filename": "report.pdf"} # Optional: filter by filename
)
```

**Advantages of ChromaDB:**
1. **Persistence**: SQLite survives process restarts
2. **No External Dependencies**: Self-contained vector DB
3. **Simple Schema**: Flexible metadata storage
4. **Performance**: HNSW indexing for fast similarity search
5. **Easy Filtering**: Metadata filtering via `where` clause

**Limitations & Scaling Considerations:**
- **Single Machine**: SQLite not suitable for distributed systems
- **Production Path**: Replace with:
  - **Pinecone** (managed cloud vector DB)
  - **Weaviate** (self-hosted, distributed)
  - **Milvus** (open-source vector DB)
  - **Qdrant** (high-performance vector DB)

---

## LLM Integration Strategy

### OpenRouter Architecture

**Why OpenRouter?**
1. **Free Tier**: Allows development/testing without cost
2. **Provider Agnostic**: Access multiple LLM providers through single API
3. **Auto-selection**: `openrouter/auto` automatically chooses cheapest model
4. **OpenAI Compatible**: Uses standard OpenAI SDK (easy to migrate)

**Current Configuration:**

```python
OpenRouterClient:
  - Model: "openrouter/auto" (auto-selects free model)
  - Temperature: 0.2 (factual, low randomness)
  - System Role: "You are a helpful PDF assistant"
  - API Key: From environment variable
  - Base URL: https://openrouter.ai/api/v1
```

**Prompt Engineering:**

```python
prompt = f"""
Use the provided context to answer the question. 
If the context does not contain the answer, state that you do not know.

Context:
{context_from_retrieval}

Question:
{user_question}

Answer clearly and cite sources using the format [Source: Filename (Page X)].
"""
```

**Key Features:**
1. **Context Inclusion**: Retrieved chunks included as context
2. **Citation Requirements**: Explicit instruction to cite sources
3. **Fallback Handling**: "I don't know" for unanswerable questions
4. **Source Tracking**: Metadata preserved throughout pipeline

**Scaling & Optimization Paths:**

| Issue | Solution |
|-------|----------|
| Cost (high volume) | Switch to cheaper models or self-hosted LLM |
| Latency (real-time requirements) | Use faster models or local LLMs (Ollama/Llama2) |
| Quality (complex reasoning) | Use GPT-4 or Claude via OpenRouter paid tier |
| Token limits (long documents) | Implement answer summarization or multi-turn retrieval |

---

## Production Scaling Strategy

### Phase 1: Current Architecture (Development)
- **PDF Storage**: Local filesystem (`temp/uploaded_files/`)
- **Vector DB**: Local SQLite (`data/vector_store/`)
- **Embeddings**: CPU-based (SentenceTransformer)
- **LLM**: OpenRouter free tier
- **Capacity**: ~1000 documents (rough estimate)

### Phase 2: Scale to 10K+ Documents

**Bottlenecks & Solutions:**

| Bottleneck | Current | Solution |
|-----------|---------|----------|
| **File Storage** | Local filesystem | Add S3/Azure Blob Storage |
| **Vector DB** | SQLite (single machine) | Migrate to Pinecone/Weaviate |
| **Embeddings** | Sync CPU processing | Batch processing + GPU acceleration |
| **LLM Latency** | OpenRouter free (slow) | Use faster models or local LLM |

**Recommended Architecture:**

```
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Application (Kubernetes)          │
│                   - Horizontal Pod Autoscaling               │
└──────────────────────────────────────────────────────────────┘
            │                               │
            ▼                               ▼
    ┌─────────────────┐        ┌──────────────────────┐
    │   S3/Blob Store │        │  Message Queue       │
    │  (PDF files)    │        │  (Celery/RabbitMQ)   │
    │                 │        │  (async jobs)        │
    └─────────────────┘        └──────────────────────┘
            │                               │
            │                               ▼
            │                    ┌──────────────────────┐
            │                    │  Worker Pod          │
            │                    │  (Embedding gen)     │
            │                    │  (OCR processing)    │
            │                    └──────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
                  ┌──────────────────────┐
                  │  Pinecone / Weaviate │
                  │  (Vector Database)   │
                  └──────────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │  OpenRouter / Local  │
                  │  LLM API             │
                  └──────────────────────┘
```

**Implementation Steps:**

1. **Async Job Queue**
   ```python
   @app.post("/upload_and_process_pdfs")
   async def upload_pdfs(files):
       # Save to S3
       # Queue for processing
       # Return job_id immediately
       # Client polls for status
   ```

2. **Batch Embedding Generation**
   ```python
   # Worker process
   @celery_app.task
   def process_pdf_batch(job_ids):
       # Process multiple PDFs in parallel
       # Generate embeddings in batches
       # Store in Pinecone
   ```

3. **GPU-Accelerated Embeddings**
   ```python
   # Use CUDA for faster embedding generation
   # Model: Large embedding model (e1-large, Jina-large)
   # Batch size: 128-256
   ```

4. **Caching Layer**
   ```python
   # Cache retrieved contexts (Redis)
   # Cache LLM responses for common queries
   # Cache embeddings for repeated documents
   ```

### Phase 3: Enterprise Scale (100K+ Documents)

**Additional Considerations:**

1. **Multi-tenancy**
   - Separate vector DB namespaces per tenant
   - RBAC for document access
   - Billing per tenant

2. **Distributed Processing**
   - Kubernetes job scheduling
   - Distributed OCR processing
   - Batch embedding generation

3. **Monitoring & Observability**
   - Query latency tracking
   - Embedding quality metrics
   - LLM cost monitoring

4. **Advanced RAG**
   - Hybrid search (BM25 + vector)
   - Query rewriting / expansion
   - Multi-step retrieval
   - Answer ranking / reranking

---

## System Architecture Diagram

### Complete System View

```
╔══════════════════════════════════════════════════════════════════╗
║                        CLIENT LAYER                              ║
║                  (Web/Desktop/Mobile App)                        ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                    HTTP/REST (FastAPI)
                              │
╔══════════════════════════════════════════════════════════════════╗
║                    APPLICATION LAYER (main.py)                   ║
║                 - App initialization                             ║
║                 - Lifespan management                            ║
║                 - Router registration                            ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                    ┌─────────┴─────────┐
                    │                   │
╔═══════════════════════════════╗  ╔══════════════════════════╗
║     API ROUTER LAYER          ║  ║  CONFIG LAYER            ║
║  (api/routes.py)              ║  ║  (config/settings.py)    ║
║                               ║  ║  - API keys              ║
║  Endpoints:                   ║  ║  - Model selection       ║
║  - POST /upload_and_process   ║  ║  - Storage paths         ║
║  - POST /chat                 ║  ║  - LLM settings          ║
║  - POST /vector_store/search  ║  ╚══════════════════════════╝
║  - GET /vector_store/count    ║
║  - GET /health                ║
╚═══════════════════════════════╝
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
  UPLOAD         CHAT
  FLOW           FLOW
    │               │
    ▼               │
╔═══════════════════════════════════════════════════════════════╗
║              DOCUMENT PROCESSING LAYER                        ║
║           (core/document_processor.py)                        ║
║                                                               ║
║  1. Validate PDF                                             ║
║  2. Save to temp directory                                   ║
║  3. Detect PDF type (digital vs scanned)                     ║
║  4. Route to appropriate extractor                           ║
╚═══════════════════════════════════════════════════════════════╝
    │
    ├─────────────────┬────────────────────┐
    │                 │                    │
    ▼                 ▼                    ▼
DIGITAL          SCANNED             METADATA
PDF PATH         PDF PATH            ENRICHMENT
    │                 │                    │
    ▼                 ▼                    ▼
┌───────────┐    ┌───────────────┐   ┌──────────────┐
│PDF Parser │    │ OCR Processor │   │Metadata      │
│(PyMuPDF)  │    │ (PaddleOCR)   │   │Extractor     │
│           │    │               │   │              │
│Extract    │    │Convert pages  │   │Add:          │
│text per   │    │to images      │   │- document_id │
│page       │    │Run OCR        │   │- filename    │
│           │    │Filter by conf │   │- citation    │
└───────────┘    └───────────────┘   └──────────────┘
    │                 │                    │
    └─────────────────┴────────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │ TEXT CHUNKER             │
        │ (core/chunker.py)        │
        │                          │
        │ - Token counting         │
        │ - 500-token chunks       │
        │ - 50-token overlap       │
        └──────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │ EMBEDDING MODEL          │
        │ (embeddings/embedding_   │
        │  model.py)               │
        │                          │
        │ SentenceTransformer      │
        │ BAAI/bge-small-en-v1.5   │
        │ 384-dim vectors          │
        └──────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │ CHROMADB VECTOR STORE    │
        │ (vectordb/chroma_store)  │
        │                          │
        │ Storage:                 │
        │ data/vector_store/       │
        │ chroma.sqlite3           │
        └──────────────────────────┘
                      │
                      └─────────────────┐
                                        │
                    (Upload Flow Complete)
                                        │
                ┌───────────────────────┘
                │
                ▼
         RETURNS TO API
         (Upload Response)
    │
    │
    ▼ (CHAT flow continues below)
╔══════════════════════════════════════════════════════════════╗
║              RAG PIPELINE LAYER                              ║
║           (rag/rag_pipeline.py)                              ║
║                                                              ║
║  1. Retrieve similar chunks                                 ║
║  2. Build context from chunks                               ║
║  3. Construct prompt                                        ║
║  4. Generate LLM response                                   ║
║  5. Return answer with sources                              ║
╚══════════════════════════════════════════════════════════════╝
    │
    ├─────────────────┬────────────────────┐
    │                 │                    │
    ▼                 ▼                    ▼
RETRIEVER       CONTEXT          RAG
(rag/retriever)  BUILDER          PIPELINE
    │                 │                    │
    ▼                 ▼                    ▼
┌───────────┐   ┌──────────┐        ┌──────────┐
│Embed      │   │Format    │        │System    │
│query      │   │retrieved │        │prompt    │
│           │   │chunks    │        │          │
│Search     │   │Add       │        │Include   │
│ChromaDB   │   │citations │        │context   │
│Top-5      │   │          │        │Include   │
│           │   │          │        │question  │
└───────────┘   └──────────┘        └──────────┘
                                            │
                                            ▼
                            ┌───────────────────────┐
                            │  LLM CLIENT           │
                            │ (llm/openrouter_      │
                            │  client.py)           │
                            │                       │
                            │ OpenRouter API        │
                            │ Model: auto           │
                            │ Temp: 0.2             │
                            └───────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────┐
                            │  LLM Response         │
                            │  (with citations)     │
                            └───────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────┐
                            │ RAG Pipeline          │
                            │ Response Assembly     │
                            │                       │
                            │ {                     │
                            │  question: "...",     │
                            │  answer: "...",       │
                            │  sources: [...]       │
                            │ }                     │
                            └───────────────────────┘
                                            │
                                            ▼
                                    RETURNS TO API
                                   (Chat Response)
```

### Module Dependencies

```
main.py
├─ fastapi
├─ api.routes
│  ├─ core.document_processor
│  │  ├─ core.pdf_parser
│  │  ├─ core.ocr_processor
│  │  ├─ core.chunker
│  │  └─ core.metadata_extractor
│  ├─ embeddings.embedding_model
│  ├─ vectordb.chroma_store
│  ├─ rag.rag_pipeline
│  │  ├─ rag.retriever
│  │  │  ├─ embeddings.embedding_model
│  │  │  └─ vectordb.chroma_store
│  │  └─ llm.openrouter_client
│  └─ api.schemas
├─ config.settings
└─ utils.logger
```

---

## Summary

The **OCR RAG PDFBot** demonstrates enterprise-grade architecture principles:

✅ **Separation of Concerns**: Each module has single, clear responsibility  
✅ **Scalability**: Independent scaling of each component  
✅ **Testability**: Isolated modules for unit testing  
✅ **Replaceability**: Swap implementations without affecting others  
✅ **Maintainability**: Clear interfaces and data contracts  
✅ **Observability**: Centralized logging and error handling  
✅ **Production-Ready**: Path to enterprise deployment (Kubernetes, cloud DBs, queues)

