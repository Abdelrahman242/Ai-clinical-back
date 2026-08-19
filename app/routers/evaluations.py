from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import auth, models, schemas
from ..database import get_db
from ..core.pipeline import run_pipeline
from ..core import safety

router = APIRouter(tags=["Evaluations"])


@router.post(
    "/api/v1/projects/{project_id}/evaluations",
    response_model=schemas.EvaluationResponse,
)
def run_evaluation(
    project_id: str,
    payload: schemas.EvaluationRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """
    Empirical Evaluation Dashboard (زي slide 10 في الأجندة):
    - Retrieval Precision@K   -> retrieved_hit لكل حالة
    - Citation Accuracy       -> citation_count > 0 في الحالات المتوقع فيها إجابة
    - Faithfulness / Unsupported Claim Rate -> safety.unsupported_claims
    """
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="المشروع غير موجود")

    if not payload.cases:
        raise HTTPException(status_code=400, detail="لازم تبعت case واحدة على الأقل")

    results = []
    hits = 0
    cited = 0
    total_answered = 0
    unsupported_total = 0
    claim_total = 0

    for case in payload.cases:
        result = run_pipeline(project_id=project_id, query=case.question, top_k=payload.top_k)

        expected_kw_hit = True
        if case.expected_keywords:
            haystack = (result.answer + " " + " ".join(c.document for c in result.citations)).lower()
            expected_kw_hit = any(kw.lower() in haystack for kw in case.expected_keywords)

        retrieved_hit = bool(result.citations) and expected_kw_hit
        citation_count = len(result.citations)

        unsupported = safety.unsupported_claims(result.answer, [c.document for c in result.citations])
        faithful = len(unsupported) == 0

        if not result.refused:
            total_answered += 1
            if citation_count > 0:
                cited += 1
            claim_total += 1
            unsupported_total += len(unsupported)
        if retrieved_hit:
            hits += 1

        passed = (
            (case.expect_refusal and result.refused)
            or (not case.expect_refusal and retrieved_hit and faithful)
        )

        results.append(
            schemas.EvaluationCaseResult(
                question=case.question,
                retrieved_hit=retrieved_hit,
                citation_count=citation_count,
                faithful=faithful,
                refused=result.refused,
                expected_refusal=case.expect_refusal,
                passed=passed,
            )
        )

    n = len(payload.cases)
    return schemas.EvaluationResponse(
        total_cases=n,
        precision_at_k=round(hits / n, 4),
        citation_accuracy=round(cited / total_answered, 4) if total_answered else 0.0,
        unsupported_claim_rate=round(unsupported_total / claim_total, 4) if claim_total else 0.0,
        results=results,
    )
