from rag.retriever import Retriever
from llm.openrouter_client import OpenRouterClient


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.llm = OpenRouterClient()

    def build_context(self, retrieved_docs) -> str:

        if not retrieved_docs or not isinstance(
            retrieved_docs,
            dict
        ):
            return ""

        context_parts = []

        # OCR parser layout
        if "rec_texts" in retrieved_docs:

            texts = retrieved_docs["rec_texts"]

            if isinstance(texts, list):

                for idx, text in enumerate(
                    texts,
                    start=1
                ):
                    if text:
                        context_parts.append(
                            f"[SOURCE {idx}]\n{text}"
                        )

        # ChromaDB layout
        elif "documents" in retrieved_docs:

            documents = retrieved_docs["documents"]

            if (
                documents
                and isinstance(
                    documents,
                    list
                )
            ):

                target_docs = (
                    documents[0]
                    if (
                        isinstance(
                            documents[0],
                            list
                        )
                    )
                    else documents
                )

                for idx, doc in enumerate(
                    target_docs,
                    start=1
                ):
                    if doc:
                        context_parts.append(
                            f"[SOURCE {idx}]\n{str(doc)}"
                        )

        return "\n\n".join(
            context_parts
        )

    def extract_sources(
        self,
        retrieved_docs,
        default_filename=None
    ):

        filename_val = (
            default_filename
            or "Unknown File"
        )

        sources = []
        seen = set()

        try:

            metadatas = (
                retrieved_docs.get(
                    "metadatas",
                    []
                )
            )

            for meta in metadatas:

                if not isinstance(
                    meta,
                    dict
                ):
                    continue

                file_name = meta.get(
                    "filename",
                    filename_val
                )

                page = meta.get(
                    "page",
                    meta.get(
                        "pdf_page"
                    )
                )

                source_key = (
                    file_name,
                    page
                )

                if (
                    source_key
                    not in seen
                ):
                    seen.add(
                        source_key
                    )

                    sources.append(
                        {
                            "file_name": file_name,
                            "page": page
                        }
                    )

        except Exception as e:

            print(
                f"[SOURCE ERROR] {e}",
                flush=True
            )

        return sources

    def ask(
        self,
        query: str,
        filename: str = None,
        top_k: int = 5,
        documents: list[str] = None
    ) -> dict:

        retrieved_docs = (
            self.retriever.retrieve(
                query=query,
                filename=filename,
                top_k=top_k,
                documents=documents
            )
        )

        context = self.build_context(
            retrieved_docs
        )

        if not context.strip():

            return {
                "question": query,
                "answer": (
                    "No relevant context found "
                    "in the uploaded documents."
                ),
                "sources": []
            }

        prompt = f"""
                        Use the provided context to answer the question.

                        Rules:
                        - Answer only from the provided context.
                        - If the answer is not in the context, say you do not know.
                        - Do NOT mention source names.
                        - Do NOT mention page numbers.
                        - Do NOT generate citations.

                        Context:
                        {context}

                        Question:
                        {query}
                        """

        answer = (
            self.llm.generate_response(
                prompt
            )
        )

        lower_answer = (
            answer.lower()
        )

        if (
            "do not know"
            in lower_answer
            or "dont know"
            in lower_answer
            or "no relevant context"
            in lower_answer
        ):

            return {
                "question": query,
                "answer": (
                    "No relevant context found "
                    "in the uploaded documents."
                ),
                "sources": []
            }
        
        # print("\n[DEBUG] Retrieved Docs:",
        #         retrieved_docs,
        #         flush=True
        #     )

        sources = self.extract_sources(
            retrieved_docs,
            filename
        )

        # print("\n[DEBUG] Sources:",
        #         sources,
        #         flush=True
        #     )

        return {
            "question": query,
            "answer": answer,
            "sources": sources
        }