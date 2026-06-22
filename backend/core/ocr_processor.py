import time
import fitz
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from utils.logger import logger

class OCRProcessor:

    def __init__(self):

        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            enable_mkldnn=False,
            cpu_threads=2,             # Parallel processing threads for better performance
            ocr_version="PP-OCRv4",
            det_limit_side_len=640,    # Limits massive layout images for cleaner text grouping
            det_limit_type="max",       # Uses max dimension limit to handle various page sizes effectively
        )

    def extract_text_with_ocr(self, pdf_path: str) -> list[dict]:
        docs = []
        pdf = fitz.open(pdf_path)

        for page_num, page in enumerate(pdf):

            logger.info(f"Starting OCR page {page_num + 1}/{len(pdf)}")
            pix = page.get_pixmap(dpi=75)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            MAX_WIDTH = 800

            if img.width > MAX_WIDTH:
                ratio = MAX_WIDTH / img.width

                img = img.resize(
                    (
                        MAX_WIDTH,
                        int(img.height * ratio)
                    )
                )
            if page_num == 0:
                logger.info(
                    f"Image size after resize: {img.width}x{img.height}"
                )

            img_np = np.array(img)

            start_time = time.time()
            result = self.ocr.ocr(img_np)
            end_time = time.time()
            logger.info(f"Finished OCR page {page_num + 1}/{len(pdf)} in {end_time - start_time:.2f} seconds")

            extracted_text = []

            # Check if PaddleOCR returned valid lines
            if result and isinstance(result, list):
                # PaddleOCR sometimes wraps results inside an outer list container [[line1, line2]]
                lines = result[0] if isinstance(result[0], list) else result
                
                for line in lines:
                    # Case A: Standard PaddleOCR structure -> [ [[x,y], [x,y]], ("Text String", 0.99) ]
                    if isinstance(line, list) or isinstance(line, tuple):
                        if len(line) > 1 and (isinstance(line[1], list) or isinstance(line[1], tuple)):
                            text = line[1][0]
                            confidence = line[1][1] if len(line[1]) > 1 else 1.0
                        else:
                            text = str(line)
                            confidence = 1.0
                    # Case B: Raw string or fallback text fallback item
                    else:
                        text = str(line)
                        confidence = 1.0

                    # Keep text blocks that pass confidence threshold or fallback text values
                    if confidence > 0.40 and text.strip():
                        extracted_text.append(text)

            full_text = "\n".join(extracted_text)

            if full_text.strip():
                docs.append({
                    "text": full_text,
                    "metadata": {
                        "page": page_num + 1,
                        "source": pdf_path,
                        "ocr": True,
                    }
                })

        logger.info(f"OCR completed: {pdf_path}")
        return docs
