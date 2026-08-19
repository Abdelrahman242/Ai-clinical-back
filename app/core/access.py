"""
Centralized authorization helpers.

Project endpoints preserve a useful 403 for an existing project the caller
cannot manage. Child resources are intentionally existence-masked with 404 so
conversation/document IDs cannot be used to discover another user's data.
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session

from .. import models


def get_accessible_project(db: Session, project_id: str, user: models.User) -> models.Project:
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="المشروع غير موجود")
    if not user.is_admin and project.created_by != user.id:
        raise HTTPException(status_code=403, detail="مش معاك صلاحية توصل للمشروع ده")
    return project


def get_accessible_conversation(
    db: Session, conversation_id: str, user: models.User
) -> models.Conversation:
    conversation = (
        db.query(models.Conversation)
        .filter(models.Conversation.id == conversation_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="المحادثة غير موجودة")

    project = db.query(models.Project).filter(models.Project.id == conversation.project_id).first()
    if not project or (not user.is_admin and project.created_by != user.id):
        raise HTTPException(status_code=404, detail="المحادثة غير موجودة")
    return conversation


def get_accessible_document(db: Session, document_id: str, user: models.User) -> models.Document:
    document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="المستند غير موجود")

    project = db.query(models.Project).filter(models.Project.id == document.project_id).first()
    if not project or (not user.is_admin and project.created_by != user.id):
        raise HTTPException(status_code=404, detail="المستند غير موجود")
    return document
