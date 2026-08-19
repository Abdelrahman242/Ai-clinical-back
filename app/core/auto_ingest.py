"""
app/core/auto_ingest.py
--------------------------
بدل ما الأدمن يسجل الملف من الواجهة (Register) وبعدين يدوس فهرسة (Ingest)
يدوي، السيرفر نفسه بيدوّر كل شوية ثواني على فولدر كل مشروع
(data/sources/<project_id>/) وأي ملف جديد يلاقيه (PDF/TXT) بيعمله:

    1) تسجيل تلقائي كـ Document
    2) Ingest فوري (chunking -> embeddings -> index)

من غير أي ضغطة زرار. الأدمن بس بيحط الملف في الفولدر ويستنى شوية.
"""

import logging
import threading
import time
from pathlib import Path

from sqlalchemy.orm import Session

from .. import models
from ..config import AUTO_INGEST_ENABLED, AUTO_INGEST_INTERVAL_SECONDS
from ..database import SessionLocal
from . import chunking
from .jobs import run_ingest_job

logger = logging.getLogger("auto_ingest")


def _discover_new_files(db: Session, project: models.Project) -> list[Path]:
    folder = chunking.project_sources_dir(project.id)

    already_tracked = {
        d.source_ref
        for d in db.query(models.Document).filter(models.Document.project_id == project.id).all()
        if d.source_ref
    }

    new_files = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in chunking.ALLOWED_EXTENSIONS:
            continue
        if path.name in already_tracked:
            continue
        new_files.append(path)

    return new_files


def scan_and_ingest_new_files() -> None:
    """
    Scan واحدة على كل المشاريع: بتسجل وتفهرس أي ملف جديد لقته في فولدر كل
    مشروع. آمنة تتنادى كذا مرة (idempotent) — أي ملف اتسجل قبل كده بيتخطى.
    """
    db: Session = SessionLocal()
    try:
        projects = db.query(models.Project).all()
        for project in projects:
            for file_path in _discover_new_files(db, project):
                logger.info("auto-ingest: found new file %s (project=%s)", file_path.name, project.id)

                document = models.Document(
                    project_id=project.id,
                    title=file_path.stem.replace("_", " ").strip() or file_path.name,
                    source_ref=file_path.name,
                    status=models.DocumentStatus.QUEUED,
                )
                db.add(document)
                db.commit()
                db.refresh(document)

                job = models.IngestJob(document_id=document.id, status=models.JobStatus.QUEUED)
                db.add(job)
                db.commit()
                db.refresh(job)

                # بيتنفذ هنا مباشرة (مش background task منفصلة) لأننا أصلاً
                # جوه thread خلفي مستقل عن الـ request/response cycle.
                run_ingest_job(job.id, document.id, reset=False)
    except Exception:  # noqa: BLE001
        logger.exception("auto-ingest scan failed")
    finally:
        db.close()


def _loop() -> None:
    logger.info(
        "auto-ingest: background scanner started (كل %s ثانية)", AUTO_INGEST_INTERVAL_SECONDS
    )
    while True:
        scan_and_ingest_new_files()
        time.sleep(AUTO_INGEST_INTERVAL_SECONDS)


def start_background_scanner() -> None:
    if not AUTO_INGEST_ENABLED:
        logger.info("auto-ingest: متوقف (AUTO_INGEST_ENABLED=false)")
        return
    thread = threading.Thread(target=_loop, name="auto-ingest-scanner", daemon=True)
    thread.start()
