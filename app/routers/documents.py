from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import auth, models, schemas
from ..database import get_db
from ..core.access import get_accessible_document, get_accessible_project
from ..core.jobs import run_ingest_job

router = APIRouter(tags=["Documents"])


# ============================================================
# POST/GET /api/v1/projects/{project_id}/documents
# ------------------------------------------------------------
# ملحوظة: الطريقة الأساسية الجديدة للتسجيل هي auto-ingest — تحط الملف في
# data/sources/{project_id}/ والسيستم بيسجله ويفهرسه لوحده تلقائيًا (شوف
# app/core/auto_ingest.py). الـ endpoint ده اتسيب موجود بس كـ fallback يدوي
# (مفيد أساسًا لتسجيل مستند بـ source_url بدل ما تحتاج تنزّله يدوي، أو لو
# عايز تتحكم في العنوان/الناشر يدويًا قبل ما يوصله الـ scanner).
# متاح للأدمن بس عشان الملفات تفضل متحكم فيها من جوه السيستم، وبيتأكد كمان
# إن المشروع ده أصلاً موجود (require_admin بيفتح لكل مشاريع السيستم للأدمن).
# ============================================================
@router.post(
    "/api/v1/projects/{project_id}/documents",
    response_model=schemas.DocumentResponse,
)
def register_document(
    project_id: str,
    payload: schemas.DocumentRegister,
    db: Session = Depends(get_db),
    admin: models.User = Depends(auth.require_admin),
):
    get_accessible_project(db, project_id, admin)

    if not payload.source_ref and not payload.source_url:
        raise HTTPException(
            status_code=400,
            detail="لازم تحدد source_ref (ملف موجود جوه مخزن مستندات النظام) أو source_url (رابط رسمي)",
        )

    document = models.Document(
        project_id=project_id,
        title=payload.title,
        source_ref=payload.source_ref,
        source_url=payload.source_url,
        publisher=payload.publisher,
        status=models.DocumentStatus.REGISTERED,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


@router.get(
    "/api/v1/projects/{project_id}/documents",
    response_model=List[schemas.DocumentResponse],
)
def list_documents(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    get_accessible_project(db, project_id, current_user)

    return (
        db.query(models.Document)
        .filter(models.Document.project_id == project_id)
        .order_by(models.Document.created_at.desc())
        .all()
    )


# ============================================================
# POST /api/v1/projects/{project_id}/reindex
# ------------------------------------------------------------
# Queues a reset=True ingest job for every registered document in the
# project. The work runs in background tasks so the HTTP request returns
# quickly and the caller can poll each job through /api/v1/jobs/{job_id}.
# ============================================================
@router.post(
    "/api/v1/projects/{project_id}/reindex",
    response_model=schemas.BulkReindexResponse,
)
def reindex_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin: models.User = Depends(auth.require_admin),
):
    get_accessible_project(db, project_id, admin)
    documents = (
        db.query(models.Document)
        .filter(models.Document.project_id == project_id)
        .order_by(models.Document.created_at)
        .all()
    )

    jobs = []
    for document in documents:
        job = models.IngestJob(
            document_id=document.id,
            status=models.JobStatus.QUEUED,
            current_stage="queued",
        )
        document.status = models.DocumentStatus.QUEUED
        db.add(job)
        jobs.append((job, document))

    db.commit()
    for job, document in jobs:
        db.refresh(job)
        background_tasks.add_task(run_ingest_job, job.id, document.id, True)

    return schemas.BulkReindexResponse(
        project_id=project_id,
        queued_jobs=[job for job, _ in jobs],
    )


# ============================================================
# POST /api/v1/documents/{document_id}/ingest
# ============================================================
@router.post(
    "/api/v1/documents/{document_id}/ingest",
    response_model=schemas.JobResponse,
)
def ingest_document(
    document_id: str,
    payload: schemas.DocumentIngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin: models.User = Depends(auth.require_admin),
):
    document = get_accessible_document(db, document_id, admin)

    job = models.IngestJob(document_id=document.id, status=models.JobStatus.QUEUED)
    document.status = models.DocumentStatus.QUEUED
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(run_ingest_job, job.id, document.id, payload.reset)
    return job


# ============================================================
# GET /api/v1/documents/{document_id}/status
# ============================================================
@router.get(
    "/api/v1/documents/{document_id}/status",
    response_model=schemas.DocumentResponse,
)
def document_status(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    return get_accessible_document(db, document_id, current_user)


# ============================================================
# GET /api/v1/jobs/{job_id}
# ============================================================
@router.get("/api/v1/jobs/{job_id}", response_model=schemas.JobResponse)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    job = db.query(models.IngestJob).filter(models.IngestJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="الـ job غير موجود")

    # الـ job مربوط بمستند مربوط بمشروع — بنتأكد إن اليوزر يملك المشروع ده
    get_accessible_document(db, job.document_id, current_user)
    return job
