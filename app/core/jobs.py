"""
app/core/jobs.py
------------------
تشغيل الـ ingestion كـ background job وتحديث حالته في الداتابيز خطوة بخطوة،
عشان GET /jobs/{job_id} و GET /documents/{document_id}/status يقدروا يتابعوا
التقدم (زي أي async pipeline حقيقي).
"""

from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from .. import models
from ..database import SessionLocal
from . import chunking, vectorstore


def run_ingest_job(job_id: str, document_id: str, reset: bool) -> None:
    db: Session = SessionLocal()
    try:
        job = db.query(models.IngestJob).filter(models.IngestJob.id == job_id).first()
        document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if job is None or document is None:
            return

        def set_stage(stage: str, progress: int):
            job.current_stage = stage
            job.progress = progress
            job.status = models.JobStatus.RUNNING
            document.status = models.DocumentStatus.INGESTING
            db.commit()

        try:
            set_stage("resolving_source", 10)
            if document.source_ref:
                path = chunking.resolve_source_path(document.project_id, document.source_ref)
            elif document.source_url:
                url_path = urlparse(document.source_url).path.lower()
                url_name = Path(url_path).name
                if url_name.endswith((".pdf", ".txt", ".html", ".htm")):
                    filename = url_name
                elif "/bitstreams/" in url_path or url_path.endswith("/content"):
                    filename = f"{document.id}.pdf"
                else:
                    filename = f"{document.id}.html"
                path = chunking.fetch_if_url(document.project_id, document.source_url, filename)
            else:
                raise ValueError("المستند مفيهوش source_ref ولا source_url")

            if reset:
                set_stage("removing_old_chunks", 20)
                vectorstore.remove_document_chunks(document.project_id, document.title)

            set_stage("chunking", 40)
            chunks = chunking.load_and_split_file(
                path, document_name=document.title, source_url=document.source_url or ""
            )

            set_stage("embedding_and_indexing", 75)
            n = vectorstore.add_chunks(document.project_id, chunks)

            set_stage("finalizing", 95)
            document.chunks_indexed = (0 if reset else document.chunks_indexed) + n
            document.status = models.DocumentStatus.INGESTED
            document.error = ""

            job.status = models.JobStatus.SUCCEEDED
            job.progress = 100
            job.current_stage = "done"
            job.chunks_indexed = n
            db.commit()

        except Exception as exc:  # noqa: BLE001
            job.status = models.JobStatus.FAILED
            job.error = str(exc)
            job.current_stage = "failed"
            document.status = models.DocumentStatus.FAILED
            document.error = str(exc)
            db.commit()
    finally:
        db.close()
