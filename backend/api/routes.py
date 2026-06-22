import os
from typing import Annotated
from pydantic import WithJsonSchema

from fastapi import (
    APIRouter,
    UploadFile as FastAPIUploadFile,
    File,
)

from api.schemas import (
    ChatRequest,
    SearchQueryRequest,
    StandardAPIResponse,
    PDFProcessResponse,
)

from config.settings import Settings
from core.document_processor import DocumentProcessor
from embeddings.embedding_model import EmbeddingModel
from vectordb.chroma_store import ChromaVectorStore
from rag.rag_pipeline import RAGPipeline
from utils.logger import logger


router = APIRouter()


# =========================
# SINGLETON INSTANCES
# =========================
document_processor = DocumentProcessor()
embedding_model = EmbeddingModel()
vector_store = ChromaVectorStore()
rag_pipeline = RAGPipeline()


# =========================
# HEALTH CHECK
# =========================
@router.get(
    "/health",
    response_model=StandardAPIResponse[str]
)
async def health_check():

    return StandardAPIResponse(
        status="success",
        data="ok",
        message="Service is healthy",
    )

UploadFile = Annotated[FastAPIUploadFile, WithJsonSchema({"type": "string", "format": "binary"})]

# =========================
# PDF INGESTION PIPELINE
# =========================
@router.post(
    "/upload_and_process_pdfs",
    response_model=StandardAPIResponse[PDFProcessResponse]
)
async def upload_and_process_pdfs(
    files: Annotated[list[UploadFile], File(...)]
):

    try:

        logger.info(f"Received {len(files)} PDFs")

        # 1. Process PDFs (OCR + parsing + chunking)
        chunks = await document_processor.process_documents(files)

        logger.info(f"Generated {len(chunks)} chunks")

        # 2. Extract text
        texts = [c["text"] for c in chunks]

        # 3. Embeddings
        embeddings = embedding_model.embed_documents(texts)

        # 4. Store in vector DB
        vector_store.add_documents(chunks, embeddings)

        # 5. Structured ingestion response
        response = PDFProcessResponse(
            documents_processed=len(files),
            chunks_created=len(chunks),
            vectorstore_count=vector_store.collection.count(),
        )

        return StandardAPIResponse[PDFProcessResponse](
            status="success",
            data=response,
            message="PDFs processed successfully",
        )

    except Exception as e:

        logger.exception("PDF ingestion failed")

        return StandardAPIResponse[PDFProcessResponse](
            status="error",
            message=str(e),
        )


# =========================
# VECTOR DB COUNT
# =========================
@router.get(
    "/vector_store/count",
    response_model=StandardAPIResponse
)
async def get_vectorstore_count():

    try:

        count = vector_store.collection.count()

        return StandardAPIResponse(
            status="success",
            data=count,
        )

    except Exception as e:

        logger.exception("Vector count failed")

        return StandardAPIResponse(
            status="error",
            message=str(e),
        )


# =========================
# VECTOR SEARCH (RAW)
# =========================
@router.post(
    "/vector_store/search",
    response_model=StandardAPIResponse
)
async def vector_search(
    request: SearchQueryRequest
):

    try:

        query_embedding = embedding_model.embed_query(
            request.query
        )

        results = vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k,
        )

        return StandardAPIResponse(
            status="success",
            data=results,
        )

    except Exception as e:

        logger.exception("Vector search failed")

        return StandardAPIResponse(
            status="error",
            message=str(e),
        )


# =========================
# CHAT (RAG PIPELINE)
# =========================
@router.post(
    "/chat",
    response_model=StandardAPIResponse
)
async def chat(
    request: ChatRequest
):

    try:

        logger.info(f"Chat query: {request.message}")
        top_k_val = getattr(request, "top_k", 5) 

        response = rag_pipeline.ask(
            query=request.message,
            top_k=top_k_val,
            documents=request.documents
        )

        return StandardAPIResponse(
            status="success",
            data=response,
        )

    except Exception as e:

        logger.exception("Chat failed")

        return StandardAPIResponse(
            status="error",
            message=str(e),
        )

# =========================
# stored documents
# =========================
@router.get("/documents")
async def list_documents():

    try:

        results = vector_store.collection.get(
            include=["metadatas"]
        )

        docs = {}

        for meta in results["metadatas"]:

            filename = meta["filename"]

            if filename not in docs:
                docs[filename] = {
                    "filename": filename,
                    "document_id": meta.get("document_id"),
                    "chunks": 0
                }

            docs[filename]["chunks"] += 1

        return StandardAPIResponse(
            status="success",
            data=list(docs.values())
        )

    except Exception as e:

        return StandardAPIResponse(
            status="error",
            message=str(e)
        )
    

# =========================
# delete documents
# =========================
@router.delete("/documents/{filename}")
async def delete_document(filename: str):

    try:

        results = vector_store.collection.get(
            where={"filename": filename}
        )

        ids = results["ids"]

        if ids:
            vector_store.collection.delete(
                ids=ids
            )

        settings = Settings() 
        pdf_path = os.path.join(
            settings.TEMPFILE_UPLOAD_DIRECTORY,
            filename
        )

        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return StandardAPIResponse(
            status="success",
            message=f"{filename} deleted"
        )

    except Exception as e:

        return StandardAPIResponse(
            status="error",
            message=str(e)
        )