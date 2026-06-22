import os
import sys
from pathlib import Path

os.environ["FLAGS_use_pir"] = "0"         # Bypasses the buggy new executor layout entirely
os.environ["FLAGS_enable_onednn"] = "1"   # Activates fast processor acceleration safely

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import router
from vectordb.chroma_store import ChromaVectorStore
from config.settings import settings
from utils.logger import logger

# =========================
# APP LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting OCR RAG PDFBot...")

    os.makedirs(settings.TEMPFILE_UPLOAD_DIRECTORY, exist_ok=True)
    os.makedirs(settings.VECTORSTORE_DIRECTORY, exist_ok=True)

    # Initialize ChromaDB
    app.state.vector_store = ChromaVectorStore()

    logger.info("ChromaDB initialized.")
    logger.info("Application startup complete.")

    yield

    logger.info("Shutting down application...")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="OCR RAG PDFBot",
    description=(
        "Chat with multiple PDFs "
        "using OCR + RAG"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================
# ROUTERS
# =========================
app.include_router(router)


# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
async def root():
    return {"message": "OCR RAG PDFBot API running"}


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    logger.info("Running FastAPI server...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )