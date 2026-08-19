from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import auth, models, schemas
from ..database import get_db
from ..core.access import get_accessible_project
from ..core.chunking import project_sources_dir

router = APIRouter(prefix="/api/v1/projects", tags=["Projects"])


@router.post("", response_model=schemas.ProjectResponse)
def create_project(
    payload: schemas.ProjectCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    project = models.Project(
        name=payload.name,
        description=payload.description,
        clinical_topic=payload.clinical_topic,
        created_by=current_user.id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    # بنجهز فولدر مستندات المشروع فورًا عشان الأدمن يعرف يحط الملفات فيه على طول
    project_sources_dir(project.id)

    return _to_response(project)


@router.get("", response_model=List[schemas.ProjectResponse])
def list_projects(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """
    كل يوزر بيشوف مشاريعه هو بس (اللي عملها بنفسه). الأدمن وحده بيشوف كل
    المشاريع في السيستم — عشان يقدر يدير مستنداتها.
    """
    query = db.query(models.Project)
    if not current_user.is_admin:
        query = query.filter(models.Project.created_by == current_user.id)

    projects = query.order_by(models.Project.created_at.desc()).all()
    return [_to_response(p) for p in projects]


@router.get("/{project_id}", response_model=schemas.ProjectResponse)
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    project = get_accessible_project(db, project_id, current_user)
    return _to_response(project)


def _to_response(project: models.Project) -> schemas.ProjectResponse:
    return schemas.ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        clinical_topic=project.clinical_topic,
        created_at=project.created_at,
        document_count=len(project.documents),
    )
