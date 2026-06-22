from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = Field(..., env="OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = Field(
        "https://openrouter.ai/api/v1",
        env="OPENROUTER_BASE_URL",
    )
    DEFAULT_MODEL: str = Field("openrouter/auto", env="DEFAULT_MODEL")

    ENV: str = Field("development", env="ENV")
    DEBUG: bool = Field(True, env="DEBUG")

    TEMPFILE_UPLOAD_DIRECTORY: str = Field(
        default=str(BASE_DIR / "temp" / "uploaded_files"),
        env="UPLOAD_DIR",
    )
    VECTORSTORE_DIRECTORY: str = Field(
        default=str(BASE_DIR / "data" / "vector_store"),
        env="VECTOR_DB_DIR",
    )

    MODEL_OPTIONS: dict = {
        "openrouter": {
            "playground": "https://openrouter.ai/models",
            "models": [
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-r1:free",
                "qwen/qwen-2.5-coder-32b-instruct:free",
                "google/gemma-3-27b-it:free",
            ],
        }
    }

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()