import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import auth, models, schemas
from ..database import get_db
from ..core.pipeline import run_pipeline

router = APIRouter(tags=["Conversations"])


# ------------------------------------------------------------------
# مش مذكورة صراحة في اللستة اللي بعتها، بس لازمة عشان نقدر نبدأ محادثة
# قبل ما نبعتلها رسايل — مضفتها هنا برضه.
# ------------------------------------------------------------------
@router.post(
    "/api/v1/projects/{project_id}/conversations",
    response_model=schemas.ConversationResponse,
)
def create_conversation(
    project_id: str,
    payload: schemas.ConversationCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="المشروع غير موجود")

    conversation = models.Conversation(
        project_id=project_id, user_id=current_user.id, title=payload.title
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.get(
    "/api/v1/projects/{project_id}/conversations",
    response_model=List[schemas.ConversationResponse],
)
def list_conversations(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    return (
        db.query(models.Conversation)
        .filter(models.Conversation.project_id == project_id)
        .order_by(models.Conversation.created_at.desc())
        .all()
    )


# ============================================================
# POST /api/v1/conversations/{conversation_id}/messages
# ------------------------------------------------------------
# هنا الـ pipeline كله بيتنفذ:
# Validate/Auth -> Retrieve -> Safety Threshold -> Generation
# -> Validate Citations -> Save Logs -> Return Answer/Refusal
# ============================================================
@router.post(
    "/api/v1/conversations/{conversation_id}/messages",
    response_model=schemas.MessageResponse,
)
def post_message(
    conversation_id: str,
    payload: schemas.MessageCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    conversation = (
        db.query(models.Conversation)
        .filter(models.Conversation.id == conversation_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="المحادثة غير موجودة")

    # آخر 10 رسايل عشان الـ context history
    past = (
        db.query(models.Message)
        .filter(models.Message.conversation_id == conversation_id)
        .order_by(models.Message.created_at.desc())
        .limit(10)
        .all()
    )
    past.reverse()
    history = [{"role": m.role.value, "content": m.content} for m in past]

    # 1) Save the user's message (log)
    user_msg = models.Message(
        conversation_id=conversation_id,
        role=models.MessageRole.USER,
        content=payload.query,
    )
    db.add(user_msg)
    db.commit()

    # 2) Run the full pipeline
    result = run_pipeline(
        project_id=conversation.project_id,
        query=payload.query,
        top_k=payload.top_k,
        chat_history=history,
    )

    # 3) Save the assistant's answer (log) — Answer or Refusal
    citations_json = json.dumps(
        [c.__dict__ for c in result.citations], ensure_ascii=False
    )
    assistant_msg = models.Message(
        conversation_id=conversation_id,
        role=models.MessageRole.ASSISTANT,
        content=result.answer,
        citations=citations_json,
        confidence=result.confidence,
        refused=result.refused,
        risk_flag=result.risk_flag,
        retrieval_score=result.max_retrieval_score,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return _to_message_response(assistant_msg)


@router.get(
    "/api/v1/conversations/{conversation_id}/messages",
    response_model=List[schemas.MessageResponse],
)
def get_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    messages = (
        db.query(models.Message)
        .filter(models.Message.conversation_id == conversation_id)
        .order_by(models.Message.created_at)
        .all()
    )
    return [_to_message_response(m) for m in messages]


def _to_message_response(m: models.Message) -> schemas.MessageResponse:
    try:
        citations = json.loads(m.citations or "[]")
    except json.JSONDecodeError:
        citations = []

    return schemas.MessageResponse(
        id=m.id,
        conversation_id=m.conversation_id,
        role=m.role.value,
        content=m.content,
        citations=citations,
        confidence=m.confidence,
        refused=bool(m.refused),
        risk_flag=m.risk_flag,
        created_at=m.created_at,
    )
