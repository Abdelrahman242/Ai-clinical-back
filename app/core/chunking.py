"""
app/core/chunking.py
---------------------
Ingestion & Chunking (Day 1-2 من الأجندة):

1. Data Sourcing  -> بيحصل قبل كده في routers/documents.py (تسجيل مصدر النظام)
2. PDF Cleaning   -> resolve_source_path / fetch_if_url هنا
3. Section-aware chunking -> 400-800 token chunks بتحافظ على حدود الـ section
4. Metadata schema -> document_name, page_number, section_title, chunk_id, source_url
   بتتخزن مع كل chunk في الـ vector store.
"""

import os
import re
import urllib.request
import uuid
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from bs4 import BeautifulSoup
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHUNK_OVERLAP, CHUNK_SIZE, SYSTEM_SOURCES_DIR

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".html", ".htm"}

# هيدرات شائعة في أدلة WHO/CDC/NICE بنستخدمها كإشارة لبداية section جديد
_SECTION_HEADING_RE = re.compile(
    r"^\s*(\d+(\.\d+)*\s+[A-Z].{3,80}|[A-Z][A-Z \-/]{6,80})\s*$"
)


def project_sources_dir(project_id: str) -> Path:
    """
    كل مشروع له فولدر منفصل جوه SYSTEM_SOURCES_DIR باسم الـ project_id.
    ده الفولدر اللي المفروض تحط فيه ملفات المشروع ده عشان الـ auto-ingest
    يكتشفها ويفهرسها لوحده (شوف core/auto_ingest.py).
    """
    path = SYSTEM_SOURCES_DIR / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_source_path(project_id: str, source_ref: str) -> Path:
    """
    بيرجع مسار الملف جوه data/sources/<project_id>/ بس. متعمدين معندناش أي API
    بتاخد مسار مطلق من اليوزر عشان محدش يقدر يقرا ملفات بره فولدر النظام.
    """
    safe_name = os.path.basename(source_ref)
    path = project_sources_dir(project_id) / safe_name
    if not path.exists():
        raise FileNotFoundError(
            f"الملف '{safe_name}' مش موجود جوه data/sources/{project_id}/. "
            "حط الملف هناك (هيتفهرس لوحده تلقائيًا) أو استخدم source_url بدل كده."
        )
    return path


def fetch_if_url(project_id: str, source_url: str, dest_filename: str) -> Path:
    """بينزل ملف رسمي من رابط (WHO/CDC/NICE/USPSTF) ويحفظه جوه فولدر المشروع."""
    dest = project_sources_dir(project_id) / dest_filename
    if not dest.exists():
        urllib.request.urlretrieve(source_url, dest)  # noqa: S310 (server-side controlled URL)
    return dest


def _guess_section_title(text: str) -> str:
    for line in text.splitlines()[:5]:
        line = line.strip()
        if line and _SECTION_HEADING_RE.match(line):
            return line[:120]
    return ""


def load_and_split_file(
    file_path: Path,
    document_name: str,
    source_url: str = "",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[LCDocument]:
    """
    بيقرأ الملف (PDF/TXT/HTML) وبيقسمه لـ chunks بتحافظ على حدود الـ section،
    وبيحط metadata schema كامل مع كل chunk.
    """
    ext = file_path.suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"نوع الملف غير مدعوم: {ext} (المتاح حاليًا: pdf, txt, html, htm)")

    if ext == ".pdf":
        loader = PyMuPDFLoader(str(file_path))
        pages = loader.load()  # صفحة/عنصر لكل Document, بيحافظ على page number في metadata
    elif ext in {".html", ".htm"}:
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        text = soup.get_text("\\n", strip=True)
        pages = [LCDocument(page_content=text, metadata={"page": 0})]
    else:
        loader = TextLoader(str(file_path), encoding="utf-8")
        pages = loader.load()  # صفحة/عنصر لكل Document, بيحافظ على page number في metadata

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[LCDocument] = []
    current_section = ""

    for page in pages:
        page_number = page.metadata.get("page", 0) + 1  # PyMuPDFLoader بيبدأ من صفر
        heading = _guess_section_title(page.page_content)
        if heading:
            current_section = heading

        for sub in splitter.split_documents([page]):
            sub.metadata.update({
                "document_name": document_name,
                "page_number": page_number,
                "section_title": current_section or "General",
                "chunk_id": uuid.uuid4().hex[:12],
                "source_url": source_url,
                # نخلي "source" برضه عشان توافق أي كود قديم بيقرا المفتاح ده
                "source": document_name,
            })
            all_chunks.append(sub)

    return all_chunks
