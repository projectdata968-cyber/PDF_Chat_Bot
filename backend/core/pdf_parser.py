import fitz

from utils.logger import logger


class PDFParser:

    def extract_text(
        self,
        pdf_path: str
    ) -> list[dict]:

        docs = []

        pdf = fitz.open(pdf_path)

        for page_num, page in enumerate(pdf):

            text = page.get_text("text")

            if text.strip():

                docs.append({
                    "text": text,
                    "metadata": {
                        "page": page_num + 1,
                        "source": pdf_path,
                        "ocr": False,
                    }
                })

        logger.info(
            f"Parsed PDF: {pdf_path}"
        )

        return docs