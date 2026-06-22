from embeddings.embedding_model import EmbeddingModel
from vectordb.chroma_store import ChromaVectorStore


class Retriever:

    def __init__(self):

        self.embedding_model = (
            EmbeddingModel()
        )

        self.vector_store = (
            ChromaVectorStore()
        )

    def retrieve(
    self,
    query: str,
    filename: str = None,
    top_k: int = 5,
    documents: list[str] = None
) -> dict:

        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        # =========================
        # UNIFIED DOCUMENT FILTER
        # =========================
        where_clause = None

        if documents:

            where_clause = {
                "filename": {
                    "$in": documents
                }
            }

        elif filename:

            where_clause = {
                "filename": filename
            }

        # =========================
        # CHROMA SEARCH
        # =========================
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where_clause
        )
        
        # =========================
        # FORMAT RESPONSE
        # =========================
        if results and "documents" in results and results["documents"]:

            return {
                "documents": results["documents"][0],
                "metadatas": results["metadatas"][0]
                if results.get("metadatas") else []
            }

        return {
            "documents": [],
            "metadatas": []
        }