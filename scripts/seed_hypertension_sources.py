#!/usr/bin/env python3
"""Seed the curated hypertension source catalog into an existing project.

Usage:
    python scripts/seed_hypertension_sources.py <project_id>

The script is intentionally catalog-driven: it only downloads URLs declared in
 data/hypertension_sources.json, skips documents already registered by URL, and
 runs the same ingestion pipeline used by the API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import models
from app.core.jobs import run_ingest_job
from app.database import SessionLocal

BASE_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = BASE_DIR / "data" / "hypertension_sources.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed official hypertension sources")
    parser.add_argument("project_id", help="Existing project ID for the hypertension knowledge base")
    args = parser.parse_args()

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    db = SessionLocal()
    try:
        project = db.query(models.Project).filter(models.Project.id == args.project_id).first()
        if project is None:
            raise SystemExit(f"Project not found: {args.project_id}")

        added = 0
        skipped = 0
        for source in catalog:
            existing = (
                db.query(models.Document)
                .filter(
                    models.Document.project_id == project.id,
                    models.Document.source_url == source["url"],
                )
                .first()
            )
            if existing is not None:
                skipped += 1
                continue

            document = models.Document(
                project_id=project.id,
                title=source["title"],
                source_url=source["url"],
                publisher=source["publisher"],
                status=models.DocumentStatus.QUEUED,
            )
            db.add(document)
            db.commit()
            db.refresh(document)

            job = models.IngestJob(
                document_id=document.id,
                status=models.JobStatus.QUEUED,
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            run_ingest_job(job.id, document.id, reset=False)
            added += 1
            print(f"ingested: {source['title']}")

        print(f"Completed: added={added}, skipped={skipped}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
