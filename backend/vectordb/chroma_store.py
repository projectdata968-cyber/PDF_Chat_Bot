import os
import uuid
import chromadb
from config.settings import settings


class ChromaVectorStore:

    def __init__(self):

        os.makedirs(settings.VECTORSTORE_DIRECTORY, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=settings.VECTORSTORE_DIRECTORY
        )

        self.collection = self.client.get_or_create_collection(
            name="pdf_documents"
        )

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ):

        # ========================================================
        # ANTI-DUPLICATION EXTRACTION & PURGE
        # ========================================================
        if chunks and "metadata" in chunks[0]:
            # Extract the filename from the first chunk's metadata
            filename = chunks[0]["metadata"].get("filename")
            
            if filename:
                # Search if vectors already exist for this filename
                existing = self.collection.get(
                    where={"filename": filename}
                )
                # If old chunks exist, delete them first
                if existing and existing.get("ids"):
                    self.collection.delete(ids=existing["ids"])
        # ========================================================
        
        ids = [
            f"doc_{uuid.uuid4().hex}"
            for _ in range(len(chunks))
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            chunk["metadata"]
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict = None,
    ):

        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }

        if where:
            search_kwargs["where"] = where

        results = self.collection.query(**search_kwargs)

        return results