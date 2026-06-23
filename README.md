# OCR RAG PDF Assistant

A full-stack application for intelligent PDF question-answering using Optical Character Recognition (OCR) and Retrieval-Augmented Generation (RAG). Upload PDFs, process them with automatic OCR detection, and ask natural language questions to retrieve accurate answers with source references.

## Overview

This application combines advanced PDF processing with modern AI techniques to enable intelligent document understanding:

- **Automatic PDF Detection**: Distinguishes between text-based and scanned PDFs
- **OCR Processing**: Extracts text from scanned documents using PaddleOCR
- **Vector Embeddings**: Creates semantic embeddings for all document chunks
- **RAG Pipeline**: Retrieves relevant context and generates answers using OpenRouter LLM
- **Multi-Document Support**: Query across multiple PDFs with selective document filtering
- **Source Tracking**: References exact file names and page numbers in responses

## Features

- ✅ **Dual PDF Processing**
  - Direct text extraction for native PDFs (PyMuPDF)
  - PaddleOCR v4 for scanned/image-based PDFs with configurable preprocessing
  - Automatic detection of PDF type
  - Max PDF file size: 200 MB

- ✅ **Intelligent Chunking**
  - Token-aware document splitting (500 tokens, 50-token overlap)
  - Tiktoken-based token counting for accuracy
  - Metadata preservation (filename, page number, OCR flag)

- ✅ **Vector Database**
  - ChromaDB with SQLite persistence
  - Anti-duplication strategy (overwrites old vectors for same document)
  - Efficient similarity search

- ✅ **RAG Pipeline**
  - Semantic search using BAAI/bge-small-en-v1.5 embeddings
  - Context-aware LLM response generation
  - Automatic source extraction with page references

- ✅ **Multi-Document Queries**
  - Search across all documents or select specific PDFs
  - Configurable retrieval depth (top_k: 1-20 chunks)
  - Unified document filtering

- ✅ **Interactive Web Interface**
  - Streamlit-based frontend for easy interaction
  - Persistent chat history (max 20 messages)
  - Document management (list, delete, chunk overview)
  - Real-time vector store metrics

- ✅ **Production Deployment**
  - Docker containerization for Hugging Face Spaces
  - GitHub Actions CI/CD pipeline for automated backend sync
  - RESTful API with comprehensive error handling

## System Architecture

### Data Processing Pipeline

```
PDF Upload
    ↓
Document Processor (Validation & Type Detection)
    ├→ Text PDF → PDFParser (PyMuPDF)
    └→ Scanned PDF → OCRProcessor (PaddleOCR)
    ↓
Metadata Enrichment (Filename, Page, Document ID)
    ↓
TextChunker (Token-aware, 500 tokens/chunk)
    ↓
EmbeddingModel (BAAI/bge-small-en-v1.5)
    ↓
ChromaVectorStore (Persistent SQLite)
```

### Query Pipeline

```
User Question
    ↓
Retriever (Embed query, search ChromaDB with optional filters)
    ↓
RAG Pipeline (Build context from retrieved chunks)
    ↓
OpenRouter LLM (Generate answer with system prompt)
    ↓
Source Extraction (Deduplicate by filename & page)
    ↓
Response (Answer + Sources)
```

### Component Communication

```
Frontend (Streamlit)
    ↓ HTTP Requests
Backend (FastAPI)
    ├→ Document Processor
    ├→ Vector Store (ChromaDB)
    ├→ Embedding Model
    ├→ RAG Pipeline
    └→ LLM Client (OpenRouter API)
```

## Technology Stack

### Backend
- **Framework**: FastAPI + Uvicorn
- **Vector Database**: ChromaDB 1.5.9 (SQLite-based)
- **Embeddings**: Sentence-Transformers (BAAI/bge-small-en-v1.5)
- **LLM**: OpenRouter API
  - Default model: `openrouter/auto` (load-balanced)
  - Free available models:
    - `meta-llama/llama-3.3-70b-instruct:free`
    - `deepseek/deepseek-r1:free`
    - `qwen/qwen-2.5-coder-32b-instruct:free`
    - `google/gemma-3-27b-it:free`
- **OCR**: PaddleOCR v4
  - Image preprocessing: 75 DPI, max 800px width with aspect ratio preservation
  - Text detection confidence threshold: 0.4 (40%)
  - Configuration: `PP-OCRv4`, `det_limit_side_len=640`, 2 CPU threads
- **PDF Processing**: PyMuPDF (fitz), PyPDFium2
- **Image Processing**: OpenCV, Pillow, NumPy
- **Token Counting**: Tiktoken
- **Async I/O**: Aiofiles
- **Logging**: Python logging with coloredlogs

### Frontend
- **Framework**: Streamlit
- **HTTP Client**: Requests
- **Configuration**: Python-dotenv

### Deployment
- **Container**: Docker (Python 3.10-slim)
- **Orchestration**: Hugging Face Spaces
- **CI/CD**: GitHub Actions
- **Source Control**: Git

## Repository Structure

```
pdf_reader_bot/
├── backend/                    # FastAPI server
│   ├── api/
│   │   ├── routes.py          # API endpoint definitions
│   │   └── schemas.py          # Pydantic request/response models
│   ├── config/
│   │   └── settings.py         # Pydantic BaseSettings configuration
│   ├── core/
│   │   ├── document_processor.py  # PDF validation & routing
│   │   ├── pdf_parser.py          # Text PDF extraction
│   │   ├── ocr_processor.py        # Scanned PDF OCR
│   │   ├── chunker.py             # Token-aware chunking
│   │   └── metadata_extractor.py   # Metadata enrichment
│   ├── embeddings/
│   │   └── embedding_model.py      # Sentence-Transformers wrapper
│   ├── llm/
│   │   └── openrouter_client.py    # OpenRouter LLM client
│   ├── rag/
│   │   ├── rag_pipeline.py         # RAG orchestration
│   │   └── retriever.py            # Vector search with filtering
│   ├── vectordb/
│   │   └── chroma_store.py         # ChromaDB persistence
│   ├── utils/
│   │   └── logger.py               # Logging setup
│   ├── main.py                     # FastAPI app entry point
│   ├── Dockerfile                  # Docker configuration
│   └── requirements.txt            # Python dependencies
│
├── frontend/                   # Streamlit application
│   ├── services/
│   │   └── api_client.py      # HTTP wrapper for backend endpoints
│   ├── utils/
│   │   └── config.py          # Frontend configuration
│   ├── app.py                 # Streamlit app entry point
│   └── requirements.txt        # Frontend dependencies
│
├── data/
│   └── vector_store/          # ChromaDB persistent storage (SQLite)
│
├── temp/
│   └── uploaded_files/        # Temporary uploaded PDF storage
│
├── .github/
│   └── workflows/
│       └── sync.yml           # GitHub Actions deployment workflow
│
├── .gitignore                 # Git exclusion rules
├── README.md                  # This file
└── pdf_bot_env/               # Python virtual environment
```

## Installation

### Prerequisites

- Python 3.10+
- Git
- OpenRouter API key (for LLM functionality)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/projectdata968-cyber/PDF_Chat_Bot.git
   cd pdf_reader_bot
   ```

2. **Activate the virtual environment**
   ```bash
   source pdf_bot_env/bin/activate  # On Windows: pdf_bot_env\Scripts\activate
   ```
   Note: The virtual environment (`pdf_bot_env/`) is already included in the repository.

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

4. **Install frontend dependencies**
   ```bash
   cd frontend
   pip install -r requirements.txt
   cd ..
   ```

5. **Create environment configuration**
   Create a `.env` file in the root directory with the following content:
   ```
   # Required
   OPENROUTER_API_KEY=your_openrouter_api_key

   # Optional (defaults provided)
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   DEFAULT_MODEL=openrouter/auto
   ENV=development
   DEBUG=true
   UPLOAD_DIR=./temp/uploaded_files
   VECTOR_DB_DIR=./data/vector_store
   BACKEND_URL=http://localhost:8000
   ```

## Environment Variables

### Backend Variables

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `OPENROUTER_API_KEY` | API key for OpenRouter LLM service | — | Yes |
| `OPENROUTER_BASE_URL` | OpenRouter API endpoint | `https://openrouter.ai/api/v1` | No |
| `DEFAULT_MODEL` | Default LLM model identifier | `openrouter/auto` | No |
| `ENV` | Environment type (development/production) | `development` | No |
| `DEBUG` | Enable debug logging | `true` | No |
| `UPLOAD_DIR` | Temporary PDF upload directory | `./temp/uploaded_files` | No |
| `VECTOR_DB_DIR` | ChromaDB persistent storage directory | `./data/vector_store` | No |

### Frontend Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `BACKEND_URL` | FastAPI backend server URL | `http://localhost:8000` |

### Obtaining Credentials

1. **OpenRouter API Key**:
   - Visit [OpenRouter.ai](https://openrouter.ai)
   - Sign up for a free account
   - Navigate to API keys section
   - Generate and copy your API key
   - Paste into `.env` file

## Running Instructions

### Local Development

#### Terminal 1: Start Backend Server
```bash
cd backend
python main.py
```
- Server runs on `http://localhost:8000`
- API documentation available at `http://localhost:8000/docs` (Swagger UI)

#### Terminal 2: Start Frontend Application
```bash
cd frontend
streamlit run app.py
```
- Frontend runs on `http://localhost:8501`
- Auto-opens in browser

### Production with Docker

#### Build Docker Image
```bash
cd backend
docker build -t pdf-chatbot-backend:latest .
```

#### Run Docker Container
```bash
docker run -p 7860:7860 \
  -e OPENROUTER_API_KEY=your_api_key_here \
  -v $(pwd)/data:/app/data \
  pdf-chatbot-backend:latest
```
- Backend accessible on `http://localhost:7860`
- Vector store persisted in `./data` volume

## API Communication

### Health Check
```bash
GET http://localhost:8000/health
```
Response:
```json
{
  "status": "success",
  "data": "ok",
  "message": "Service is healthy"
}
```

### Upload & Process PDFs
```bash
POST http://localhost:8000/upload_and_process_pdfs
Content-Type: multipart/form-data

files: [pdf_file_1, pdf_file_2, ...]
```
Response:
```json
{
  "status": "success",
  "data": {
    "documents_processed": 2,
    "chunks_created": 150,
    "vectorstore_count": 350
  },
  "message": "PDFs processed successfully"
}
```

### Chat (RAG Query)
```bash
POST http://localhost:8000/chat
Content-Type: application/json

{
  "message": "What is the main topic of the document?",
  "top_k": 5,
  "documents": ["file1.pdf", "file2.pdf"]
}
```
Response:
```json
{
  "status": "success",
  "data": {
    "question": "What is the main topic of the document?",
    "answer": "The document discusses machine learning fundamentals...",
    "sources": [
      {"file_name": "file1.pdf", "page": 3},
      {"file_name": "file1.pdf", "page": 7}
    ]
  }
}
```

### List Documents
```bash
GET http://localhost:8000/documents
```
Response:
```json
{
  "status": "success",
  "data": [
    {"filename": "document1.pdf", "document_id": 1, "chunks": 45},
    {"filename": "document2.pdf", "document_id": 2, "chunks": 38}
  ]
}
```

### Delete Document
```bash
DELETE http://localhost:8000/documents/document1.pdf
```
Response:
```json
{
  "status": "success",
  "message": "document1.pdf deleted"
}
```

### Vector Store Count
```bash
GET http://localhost:8000/vector_store/count
```
Response:
```json
{
  "status": "success",
  "data": 83
}
```

## Deployment

### Streamlit Frontend Deployment

**Option 1: Streamlit Cloud** (Recommended for quick deployment)
1. Push repository to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Create new app → Select repository
4. Configure environment variables in cloud settings
5. Set `BACKEND_URL` to your deployed backend URL

**Option 2: Self-Hosted**
```bash
pip install streamlit
streamlit run frontend/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501
```

### Hugging Face Spaces Backend Deployment

The backend is deployed as a Docker container on Hugging Face Spaces:

1. **Space Setup**:
   - Created at: [Nimesh968/pdf-chatbot-backend](https://huggingface.co/spaces/Nimesh968/pdf-chatbot-backend)
   - Space type: Docker
   - Persistent storage enabled for vector DB

2. **GitHub Actions CI/CD**:
   - Trigger: Push to `master` branch with changes in `backend/**`
   - Action: Automatically syncs backend code to HF Space
   - Requires: `HF_TOKEN` secret configured in GitHub

3. **Container Configuration**:
   - Base image: `python:3.10-slim`
   - Port: 7860 (Hugging Face standard)
   - System dependencies: OpenCV, OCR libraries installed via apt-get
   - Entrypoint: Uvicorn FastAPI application

4. **Environment Variables on HF Spaces**:
   - Configure in Space settings:
     - `OPENROUTER_API_KEY`: Your API key
     - Any other environment variables as needed
   - Variables are injected at runtime

### Docker Usage

**Local Testing**:
```bash
# Build
docker build -t pdf-chatbot-backend:latest backend/

# Run with volume for persistence
docker run -p 7860:7860 \
  -e OPENROUTER_API_KEY=<your-key> \
  -v $(pwd)/data:/app/data \
  pdf-chatbot-backend:latest
```

**Dockerfile Details**:
- Base: `python:3.10-slim`
- Install: build-essential, libgl1, libglib2.0-0, libgomp1 (OCR/OpenCV support)
- Entrypoint: `uvicorn main:app --host 0.0.0.0 --port 7860`

## Future Improvements

- [ ] **Multi-language OCR Support**: Extend PaddleOCR to support more languages
- [ ] **Hybrid Search**: Combine keyword-based BM25 with semantic search
- [ ] **Fine-tuned Models**: Add option for domain-specific fine-tuned embeddings/LLMs
- [ ] **Caching Layer**: Implement Redis for query/response caching
- [ ] **Advanced Filtering**: Add date range, metadata, and custom field filtering
- [ ] **Chat Persistence**: Store conversations in database for audit trail
- [ ] **Rate Limiting**: Implement token-based rate limiting for API
- [ ] **Monitoring Dashboard**: Add Prometheus metrics and Grafana dashboards
- [ ] **Multiple Language Support**: Frontend UI in multiple languages
- [ ] **Batch Processing**: Support for scheduled batch PDF processing
- [ ] **Web Scraping Integration**: Auto-ingest web content as PDFs
- [ ] **Advanced RAG**: Implement routing, multi-hop retrieval, reranking

## License

MIT

---

## Support

For issues, questions, or contributions, please refer to the GitHub repository documentation or contact the project maintainers.
