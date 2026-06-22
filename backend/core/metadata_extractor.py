import os


class MetadataExtractor:

    @staticmethod
    def enrich_metadata(
        docs: list[dict]
    ) -> list[dict]:

        enriched_docs = []

        for idx, doc in enumerate(docs):

            source_path = (
                doc["metadata"]["source"]
            )

            filename = os.path.basename(
                source_path
            )

            enriched_metadata = {
                **doc["metadata"],
                "document_id": idx + 1,
                "filename": filename,
                "pdf_page": doc["metadata"]["page"]
            }

            enriched_docs.append({
                "text": doc["text"],
                "metadata": enriched_metadata
            })

        return enriched_docs