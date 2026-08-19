"""
app/config.py
--------------
كل الإعدادات الثابتة للسيستم في مكان واحد.
مفيش أي مسار فيه اليوزر بيرفع ملف مباشرة على الـ RAG store — الملفات (الأدلة الطبية
الرسمية) موجودة جوه السيستم في SYSTEM_SOURCES_DIR، والـ admin بس هو اللي يسجّلها
كـ Document عن طريق الـ API (POST /projects/{id}/documents) بالاسم أو الرابط الرسمي،
مش عن طريق upload من اليوزر العادي.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# System-owned document storage (NOT a user-upload folder)
# ------------------------------------------------------------------
# المطور/الأدمن هو اللي يحط ملفات الـ PDF الرسمية هنا وقت الـ deployment
# (WHO / CDC / NICE / USPSTF ...). الـ API بيتعامل معاها كـ "source_ref"
# مش كـ file بيتبعت من اليوزر.
SYSTEM_SOURCES_DIR = Path(os.getenv("SYSTEM_SOURCES_DIR", BASE_DIR / "data" / "sources"))
SYSTEM_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

# كل مشروع (project) بياخد فولدر منفصل لفهرسه (index) جوه المسار ده
VECTORSTORE_ROOT = Path(os.getenv("VECTORSTORE_ROOT", BASE_DIR / "data" / "vectorstores"))
VECTORSTORE_ROOT.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'clinical_rag.db'}")

# ------------------------------------------------------------------
# Embeddings / LLM
# ------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# ملحوظة: Groq عملوا deprecate لـ llama-3.3-70b-versatile و llama-3.1-8b-instant.
# البديل المتاح حاليًا: openai/gpt-oss-120b (قوي) أو openai/gpt-oss-20b (أسرع/أخف).
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "10"))

# ------------------------------------------------------------------
# Safety & guardrail thresholds  (Day 4 من الأجندة)
# ------------------------------------------------------------------
# لو أعلى similarity score رجع أقل من كده -> نرفض ونقول "insufficient evidence"
RETRIEVAL_CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.35"))

# لو متوسط الـ score بين الحد ده والـ threshold الأساسي -> نجاوب بس نحط confidence = Low
RETRIEVAL_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_LOW_CONFIDENCE_THRESHOLD", "0.55"))

# كلمات بتدل إن السؤال حالة طوارئ / خارج نطاق النظام تماما -> Refuse/Redirect فورًا
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "suicidal", "suicide",
    "severe bleeding", "unconscious", "overdose", "stroke", "heart attack",
    "ألم في الصدر", "مش قادر اتنفس", "نزيف شديد", "فاقد الوعي", "جرعة زايدة",
]

# مواضيع خارج نطاق المشروع (طب أطفال، جراحة، إلخ) — يترفضوا بأدب برسالة redirect
OUT_OF_SCOPE_HINT_KEYWORDS = ["surgery", "pediatric dosage", "جرعة أطفال", "عملية جراحية"]

# ------------------------------------------------------------------
# Auto-ingest (بدون تدخل الأدمن)
# ------------------------------------------------------------------
# لو True، السيرفر بيدوّر لوحده كل AUTO_INGEST_INTERVAL_SECONDS على فولدر
# كل مشروع (data/sources/<project_id>/) وأي ملف جديد بيتسجل ويتفهرس تلقائيًا.
AUTO_INGEST_ENABLED = os.getenv("AUTO_INGEST_ENABLED", "true").lower() in ("1", "true", "yes")
AUTO_INGEST_INTERVAL_SECONDS = int(os.getenv("AUTO_INGEST_INTERVAL_SECONDS", "15"))

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
