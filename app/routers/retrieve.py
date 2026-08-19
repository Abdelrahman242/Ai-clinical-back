from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import auth, models, schemas
from ..database import get_db
from ..core import safety
from ..core.access import get_accessible_project
from ..core.vectorstore import similarity_search_with_score

router = APIRouter(tags=["Retrieve (debug)"])


@router.post(
    "/api/v1/projects/{project_id}/retrieve",
    response_model=schemas.RetrieveResponse,
)
def debug_retrieve(
    project_id: str,
    payload: schemas.RetrieveRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """
    مفيد للـ debug: بيرجع الـ chunks المسترجعة وسكورها وكل الـ metadata من غير
    ما يعدي على الـ LLM خالص — عشان تقدر تتأكد إن الـ retrieval نفسه شغال صح.
    """
    get_accessible_project(db, project_id, current_user)

    results = similarity_search_with_score(project_id, payload.query, k=payload.top_k)
    max_score = max((s for _, s in results), default=0.0)

    return schemas.RetrieveResponse(
        query=payload.query,
        max_score=round(max_score, 4),
        confidence=safety.confidence_from_score(max_score),
        results=[
            schemas.RetrievedChunk(
                chunk_id=doc.metadata.get("chunk_id", ""),
                document=doc.metadata.get("document_name", doc.metadata.get("source", "unknown")),
                section=doc.metadata.get("section_title"),
                page=doc.metadata.get("page_number"),
                score=round(score, 4),
                content=doc.page_content,
            )
            for doc, score in results
        ],
    )
