from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.auto_ingest import start_background_scanner
from .database import Base, engine
from .routers import (
    auth_router,
    conversations,
    documents,
    evaluations,
    health,
    projects,
    retrieve,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Clinical RAG Copilot API",
    description=(
        "Retrieval-Augmented Generation API grounded in official clinical guidelines. "
        "مستندات النظام (الأدلة الرسمية) بتتفهرس تلقائيًا بمجرد ما تتحط جوه "
        "data/sources/<project_id>/ — مفيش رفع ملف من المستخدم ولا حتى ضغطة زرار من الأدمن."
    ),
    version="2.1.0",
)


@app.on_event("startup")
def _launch_auto_ingest_scanner():
    """
    بيشغّل thread خلفي بيدوّر باستمرار على فولدرات مستندات كل مشروع، وأي ملف
    جديد يلاقيه بيسجّله ويعمله ingest أوتوماتيك من غير أي تدخل من الأدمن.
    """
    start_background_scanner()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth_router.router)
app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(conversations.router)
app.include_router(retrieve.router)
app.include_router(evaluations.router)


@app.get("/")
def root():
    return {"status": "running", "docs": "/docs"}
