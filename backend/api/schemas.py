from typing import (
    Any,
    List,
    Optional,
    Literal,
    Generic, 
    TypeVar,
)

from pydantic import (
    BaseModel,
    Field,
)

T = TypeVar("T")

# =========================
# CHAT REQUEST
# =========================
class ChatRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        description="User question"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of retrieved chunks"
    )

    documents: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of selected PDFs. "
            "None = all documents"
        )
    )


# =========================
# VECTOR SEARCH REQUEST
# =========================
class SearchQueryRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        description="Search query"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    filename: Optional[str] = Field(
        default=None,
        description="Optional filename to isolate context to a single document"
    )


# =========================
# PDF INGEST RESPONSE
# =========================
class PDFProcessResponse(BaseModel):

    documents_processed: int

    chunks_created: int

    vectorstore_count: int


# =========================
# STANDARD API RESPONSE
# =========================
class StandardAPIResponse(BaseModel, Generic[T]):

    status: Literal[
        "success",
        "error"
    ]

    data: Optional[T] = None

    message: Optional[str] = None