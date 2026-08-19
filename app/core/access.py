"""
app/core/access.py
---------------------
كل مشروع مملوك لليوزر اللي عمله (Project.created_by). أي حاجة تحت المشروع
(مستندات، محادثات، رسايل) بترث نفس الصلاحية: صاحب المشروع أو الأدمن بس هم
اللي يقدروا يوصلوها. أي يوزر تاني بيرجعله 403/404 حتى لو عنده الـ ID بالظبط.
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


def get_accessible_conversation(db: Session, conversation_id: str, user: models.User) -> models.Conversation:
    conversation = (
        db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="المحادثة غير موجودة")
    # بيتأكد إن اليوزر عنده صلاحية على المشروع الأب — بيرمي 403/404 لوحده لو لأ
    get_accessible_project(db, conversation.project_id, user)
    return conversation


def get_accessible_document(db: Session, document_id: str, user: models.User) -> models.Document:
    document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="المستند غير موجود")
    get_accessible_project(db, document.project_id, user)
    return document
