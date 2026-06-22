from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000"
)