import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text,
)
from sqlalchemy.orm import relationship

from .database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStatus(str, enum.Enum):
    REGISTERED = "registered"   # اتسجل في السيستم بس لسه ما اتعملوش ingest
    QUEUED = "queued"
    INGESTING = "ingesting"
    INGESTED = "ingested"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    # بس الأدمن يقدر يسجل/يعمل ingest لمستندات النظام — عادي users بس بيسألوا
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_now)


class Project(Base):
    """
    مشروع = نطاق سريري واحد (مثلا Adult Hypertension Management)، زي ما مطلوب في
    الـ scope بتاع الأجندة. كل project له اندكس/vector store منفصل.
    """
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    clinical_topic = Column(String, default="")
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=_now)

    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="project", cascade="all, delete-orphan")


class Document(Base):
    """
    مستند رسمي داخل السيستم. الملف نفسه موجود في SYSTEM_SOURCES_DIR (أو بيتنزل من
    source_url رسمي) — مفيش multipart upload من اليوزر هنا خالص.
    """
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)

    title = Column(String, nullable=False)
    # اسم الملف جوه SYSTEM_SOURCES_DIR (لو موجود فعلا على السيستم)
    source_ref = Column(String, nullable=True)
    # أو رابط رسمي (WHO/CDC/NICE/USPSTF) يتنزل منه وقت الـ ingest
    source_url = Column(String, nullable=True)
    publisher = Column(String, default="")

    status = Column(Enum(DocumentStatus), default=DocumentStatus.REGISTERED)
    page_count = Column(Integer, default=0)
    chunks_indexed = Column(Integer, default=0)
    error = Column(Text, default="")

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    project = relationship("Project", back_populates="documents")
    jobs = relationship("IngestJob", back_populates="document", cascade="all, delete-orphan")


class IngestJob(Base):
    """Async job بيتتبع خطوات الـ ingest (Ingestion -> Chunking -> Embeddings -> Index)."""
    __tablename__ = "ingest_jobs"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)

    status = Column(Enum(JobStatus), default=JobStatus.QUEUED)
    progress = Column(Integer, default=0)  # 0-100
    current_stage = Column(String, default="queued")
    chunks_indexed = Column(Integer, default=0)
    error = Column(Text, default="")

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    document = relationship("Document", back_populates="jobs")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String, default="New conversation")
    created_at = Column(DateTime, default=_now)

    project = relationship("Project", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """كل رسالة (سؤال أو رد) بتتسجل هنا = الـ 'Save Logs' في الـ pipeline."""
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)

    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)

    # للردود بس (assistant)
    citations = Column(Text, default="[]")       # JSON string: [{document, section, page, chunk_id}, ...]
    confidence = Column(String, nullable=True)     # High / Medium / Low / Insufficient Evidence
    refused = Column(Boolean, default=False)
    risk_flag = Column(String, nullable=True)      # allowed / needs_caution / refuse
    retrieval_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=_now)

    conversation = relationship("Conversation", back_populates="messages")
