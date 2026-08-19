from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db

router = APIRouter(tags=["Health"])


@router.get("/health")
def health():
    """Liveness — هل الـ process شغال أصلا."""
    return {"status": "ok"}


@router.get("/ready")
def ready(db: Session = Depends(get_db)):
    """Readiness — هل السيستم جاهز يستقبل traffic (الداتابيز متاحة)."""
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False

    return {"status": "ready" if db_ok else "not_ready", "database": db_ok}
