"""Rebuild every registered document with the configured embedding model.

Run once after changing EMBEDDING_MODEL or EMBEDDING_NORMALIZE:
    python scripts/reindex_all_sources.py
"""
from app import models
from app.config import EMBEDDING_MODEL, EMBEDDING_NORMALIZE
from app.core.jobs import run_ingest_job
from app.database import SessionLocal


def main() -> None:
    db = SessionLocal()
    try:
        documents = db.query(models.Document).order_by(models.Document.created_at).all()
        print(
            f"Reindexing {len(documents)} documents with {EMBEDDING_MODEL} "
            f"(normalize={EMBEDDING_NORMALIZE})"
        )
        for document in documents:
            job = models.IngestJob(
                document_id=document.id,
                status=models.JobStatus.QUEUED,
                current_stage="queued",
            )
            document.status = models.DocumentStatus.QUEUED
            db.add(job)
            db.commit()
            db.refresh(job)
            run_ingest_job(job.id, document.id, reset=True)
            print(f"reindexed document={document.id} title={document.title}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
