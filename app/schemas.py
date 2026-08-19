from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ============================================================
# Auth
# ============================================================
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = Field(default=None, min_length=6)


# ============================================================
# Projects
# ============================================================
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    description: str = ""
    clinical_topic: str = Field("", description="e.g. Adult Hypertension Management")


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    clinical_topic: str
    created_at: datetime
    document_count: int = 0

    class Config:
        from_attributes = True


# ============================================================
# Documents  (system-owned sources, NOT end-user uploads)
# ============================================================
class DocumentRegister(BaseModel):
    title: str
    # لازم واحد منهم بس: source_ref (اسم ملف موجود جوه SYSTEM_SOURCES_DIR)
    # أو source_url (رابط رسمي زي who.int / cdc.gov / nice.org.uk)
    source_ref: Optional[str] = Field(
        None, description="Filename already present inside the system's SYSTEM_SOURCES_DIR"
    )
    source_url: Optional[str] = Field(
        None, description="Official guideline URL (WHO/CDC/NICE/USPSTF) to fetch server-side"
    )
    publisher: str = ""


class DocumentResponse(BaseModel):
    id: str
    project_id: str
    title: str
    source_ref: Optional[str]
    source_url: Optional[str]
    publisher: str
    status: str
    page_count: int
    chunks_indexed: int
    error: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentIngestRequest(BaseModel):
    reset: bool = Field(False, description="لو True بيمسح الـ chunks القديمة بتاعة المستند ده قبل ما يعيد فهرستها")


class JobResponse(BaseModel):
    id: str
    document_id: str
    status: str
    progress: int
    current_stage: str
    chunks_indexed: int
    error: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# Conversations / Messages
# ============================================================
class ConversationCreate(BaseModel):
    title: str = "New conversation"


class ConversationResponse(BaseModel):
    id: str
    project_id: str
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)


class Citation(BaseModel):
    document: str
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    citations: List[Citation] = []
    confidence: Optional[str] = None
    refused: bool = False
    risk_flag: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# Retrieve (debug)
# ============================================================
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document: str
    section: Optional[str] = None
    page: Optional[int] = None
    score: float
    content: str


class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]
    max_score: float
    confidence: str


# ============================================================
# Evaluations
# ============================================================
class EvalCase(BaseModel):
    question: str
    expected_keywords: List[str] = Field(
        default_factory=list,
        description="كلمات/فريزات المفروض تظهر في الإجابة أو في الـ chunks المسترجعة عشان نعتبرها صح",
    )
    expect_refusal: bool = False


class EvaluationRequest(BaseModel):
    cases: List[EvalCase]
    top_k: int = 5


class EvaluationCaseResult(BaseModel):
    question: str
    retrieved_hit: bool
    citation_count: int
    faithful: bool
    refused: bool
    expected_refusal: bool
    passed: bool


class EvaluationResponse(BaseModel):
    total_cases: int
    precision_at_k: float
    citation_accuracy: float
    unsupported_claim_rate: float
    results: List[EvaluationCaseResult]
