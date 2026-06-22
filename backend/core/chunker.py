import tiktoken

from config.settings import settings
from utils.logger import logger


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def count_tokens(
        self,
        text: str,
        model: str = settings.DEFAULT_MODEL
    ) -> int:

        try:
            encoding = tiktoken.encoding_for_model(
                model
            )

        except KeyError:

            encoding = tiktoken.get_encoding(
                "cl100k_base"
            )

        return len(
            encoding.encode(text)
        )

    def split_documents(
        self,
        docs: list[dict]
    ) -> list[dict]:

        chunks = []

        for doc in docs:

            words = doc["text"].split()

            start = 0

            while start < len(words):

                end = start + self.chunk_size

                chunk_text = " ".join(
                    words[start:end]
                )

                token_count = (
                    self.count_tokens(
                        chunk_text
                    )
                )

                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "tokens": token_count,
                    }
                })

                start += (
                    self.chunk_size
                    - self.chunk_overlap
                )

        logger.info(
            f"Created {len(chunks)} chunks"
        )

        return chunks