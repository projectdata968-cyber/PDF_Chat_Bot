import os
import aiofiles
import fitz

from fastapi import UploadFile

from config.settings import settings
from .pdf_parser import PDFParser
from .ocr_processor import OCRProcessor
from .chunker import TextChunker
from .metadata_extractor import MetadataExtractor
from utils.logger import logger


class DocumentProcessor:

    def __init__(self):

        self.pdf_parser = PDFParser()

        self.ocr_processor = OCRProcessor()

        self.chunker = TextChunker()

    def validate_pdf(
        self,
        file: UploadFile,
        max_size_mb: int = 200
    ):

        if not file.filename.lower().endswith(
            ".pdf"
        ):
            raise ValueError(
                f"{file.filename} "
                f"is not a PDF."
            )

        file_size_mb = (
            len(file.file.read())
            / (1024 * 1024)
        )

        file.file.seek(0)

        if file_size_mb > max_size_mb:

            raise ValueError(
                f"{file.filename} "
                f"exceeds size limit."
            )

    async def save_uploaded_files(
        self,
        files: list[UploadFile]
    ) -> list[str]:

        os.makedirs(
            settings.TEMPFILE_UPLOAD_DIRECTORY,
            exist_ok=True
        )

        file_paths = []

        for file in files:

            self.validate_pdf(file)

            file_path = os.path.join(
                settings.TEMPFILE_UPLOAD_DIRECTORY,
                file.filename
            )

            async with aiofiles.open(
                file_path,
                "wb"
            ) as f:

                content = await file.read()

                await f.write(content)

            file_paths.append(file_path)

        return file_paths

    def is_scanned_pdf(
        self,
        pdf_path: str
    ) -> bool:

        pdf = fitz.open(pdf_path)

        for page in pdf:

            text = page.get_text("text")

            if len(text.strip()) > 20:

                return False

        return True

    async def process_documents(
        self,
        files: list[UploadFile]
    ) -> list[dict]:

        file_paths = (
            await self.save_uploaded_files(
                files
            )
        )

        all_docs = []
        processed_files = set()

        # Zip file objects with their saved paths to retain filename access
        for file_obj, file_path in zip(files, file_paths):

            if file_obj.filename in processed_files:

                logger.warning(
                    f"Skipping duplicate file: {file_obj.filename}"
                )

                continue

            processed_files.add(
                file_obj.filename
            )

            scanned = self.is_scanned_pdf(
                file_path
            )

            if scanned:

                logger.info(
                    f"Scanned PDF detected: "
                    f"{file_path}"
                )

                docs = (
                    self.ocr_processor
                    .extract_text_with_ocr(
                        file_path
                    )
                )

            else:

                docs = (
                    self.pdf_parser
                    .extract_text(
                        file_path
                    )
                )
                
            for doc in docs:
                # Ensure a metadata dictionary exists
                if "metadata" not in doc or doc["metadata"] is None:
                    doc["metadata"] = {}
                
                # Overwrite metadata values to guarantee correctness
                doc["metadata"]["filename"] = file_obj.filename
                doc["metadata"]["source"] = file_path
                doc["metadata"]["ocr"] = scanned
                # Ensure standard citation structure matches the file
                page_num = doc["metadata"].get("page", 1)
                doc["metadata"]["citation"] = f"{file_obj.filename} (Page {page_num})"

            all_docs.extend(docs)

        enriched_docs = (
            MetadataExtractor
            .enrich_metadata(all_docs)
        )

        chunks = (
            self.chunker
            .split_documents(
                enriched_docs
            )
        )

        return chunks