"""
app/core/pipeline.py
----------------------
بيطبق نفس الـ flow اللي في الأجندة:

User -> Frontend -> Backend API
     -> Validate/Auth        (بيحصل في الـ router عن طريق auth.get_current_user)
     -> Retrieve Context     (vectorstore.similarity_search_with_score)
     -> Safety Threshold     (safety.classify_input_risk + safety.confidence_from_score)
     -> Generation Model     (llm.get_llm)
     -> Validate Answer/Citations (safety.unsupported_claims)
     -> Save Logs            (بيرجع للـ router يخزنه كـ Message)
     -> Return Answer or Refusal
"""

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from . import safety
from .llm import get_llm, get_prompt_template, get_small_talk_prompt
from ..config import ENABLE_QUERY_EXPANSION
from .vectorstore import similarity_search_with_score


@dataclass
class PipelineCitation:
    document: str
    section: Optional[str]
    page: Optional[int]
    chunk_id: Optional[str]
    score: float


@dataclass
class PipelineResult:
    answer: str
    citations: List[PipelineCitation] = field(default_factory=list)
    confidence: Optional[str] = safety.CONFIDENCE_INSUFFICIENT
    refused: bool = False
    risk_flag: str = safety.RISK_ALLOWED
    max_retrieval_score: float = 0.0


def _build_lc_history(chat_history: list):
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in chat_history
    ]


def run_pipeline(
    project_id: str,
    query: str,
    top_k: int = 5,
    chat_history: Optional[list] = None,
) -> PipelineResult:
    chat_history = chat_history or []

    # ----------------------------------------------------------------
    # 1) Input Risk Classification
    # ----------------------------------------------------------------
    risk = safety.classify_input_risk(query)
    if risk.risk == safety.RISK_REFUSE:
        return PipelineResult(
            answer=(
                "معلش، السؤال ده برة نطاق النظام أو بيوصف حالة طارئة. "
                f"{risk.reason} من فضلك تواصل مع طبيب/جهة مختصة مباشرة."
            ),
            refused=True,
            risk_flag=risk.risk,
            confidence=safety.CONFIDENCE_INSUFFICIENT,
        )

    # ----------------------------------------------------------------
    # 1.5) كلام عابر/تحية — يرد بشكل طبيعي، من غير ما يعدي على منطق
    # الرفض بتاع الأسئلة الطبية (مفيش retrieval ولا confidence threshold هنا)
    # ----------------------------------------------------------------
    if safety.is_small_talk(query):
        llm = get_llm()
        prompt = get_small_talk_prompt()
        final_prompt = prompt.invoke({
            "question": query,
            "chat_history": _build_lc_history(chat_history),
        })
        response = llm.invoke(final_prompt)
        return PipelineResult(
            answer=response.content,
            citations=[],
            confidence=None,
            refused=False,
            risk_flag=risk.risk,
            max_retrieval_score=0.0,
        )

    # ----------------------------------------------------------------
    # 2) Retrieve Context
    # ----------------------------------------------------------------
    retrieval_query = _expand_hypertension_query(query) if ENABLE_QUERY_EXPANSION else query
    try:
        retrieved = similarity_search_with_score(project_id, retrieval_query, k=top_k)
    except Exception:  # noqa: BLE001
        return PipelineResult(
            answer=(
                "حصلت مشكلة في قراءة فهرس المصادر. تأكد أن مصادر ضغط الدم تم تحميلها "
                "وعمل ingest لها على نفس المشروع، ثم أعد المحاولة."
            ),
            refused=True,
            risk_flag=risk.risk,
            confidence=safety.CONFIDENCE_INSUFFICIENT,
        )
    max_score = max((score for _, score in retrieved), default=0.0)

    # ----------------------------------------------------------------
    # 3) Safety / Retrieval Confidence Threshold
    # ----------------------------------------------------------------
    confidence = safety.confidence_from_score(max_score)
    if confidence == safety.CONFIDENCE_INSUFFICIENT:
        return PipelineResult(
            answer=(
                "معنديش أدلة كافية من المستندات الرسمية المسجلة في المشروع ده "
                "عشان أجاوب على السؤال بثقة. من فضلك أعد صياغة السؤال أو تأكد إن "
                "الدليل المناسب اتسجل واتعمله ingest."
            ),
            refused=True,
            risk_flag=risk.risk,
            confidence=confidence,
            max_retrieval_score=max_score,
        )

    context_text = "\n\n".join(doc.page_content for doc, _ in retrieved)

    # ----------------------------------------------------------------
    # 4) Grounded Generation
    # ----------------------------------------------------------------
    try:
        llm = get_llm()
        prompt = get_prompt_template()
    except Exception:  # noqa: BLE001
        return PipelineResult(
            answer=(
                "حصلت مشكلة في تشغيل نموذج الإجابة. تأكد من ضبط إعدادات مزود "
                "النموذج على الخادم ثم أعد المحاولة."
            ),
            refused=True,
            risk_flag=risk.risk,
            confidence=safety.CONFIDENCE_LOW,
            max_retrieval_score=max_score,
        )

    final_prompt = prompt.invoke({
        "context": context_text,
        "question": query,
        "chat_history": _build_lc_history(chat_history),
    })
    try:
        response = llm.invoke(final_prompt)
        answer_text = response.content
    except Exception:  # noqa: BLE001
        return PipelineResult(
            answer=(
                "المصادر اتوجدت، لكن حصل عطل مؤقت أثناء توليد الإجابة. "
                "أعد المحاولة بعد لحظات، وإذا استمرت المشكلة راجع إعدادات نموذج اللغة."
            ),
            refused=True,
            risk_flag=risk.risk,
            confidence=safety.CONFIDENCE_LOW,
            max_retrieval_score=max_score,
        )

    # ----------------------------------------------------------------
    # 5) Validate Answer / Citations (unsupported claim detection)
    # ----------------------------------------------------------------
    retrieved_texts = [doc.page_content for doc, _ in retrieved]
    unsupported = safety.unsupported_claims(answer_text, retrieved_texts)
    if unsupported:
        # نخفّض الثقة بدل ما نرفض بالكامل، ونحذر المستخدم صراحة
        confidence = safety.CONFIDENCE_LOW
        answer_text += (
            "\n\n⚠️ تنبيه: جزء من الإجابة دي مش متأكدين إنه مدعوم بالكامل بالنص "
            "المسترجع من الدليل الرسمي — راجع المصادر تحت قبل ما تعتمد عليها."
        )

    citations = [
        PipelineCitation(
            document=doc.metadata.get("document_name", doc.metadata.get("source", "unknown")),
            section=doc.metadata.get("section_title"),
            page=doc.metadata.get("page_number"),
            chunk_id=doc.metadata.get("chunk_id"),
            score=round(score, 4),
        )
        for doc, score in retrieved
    ]

    return PipelineResult(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        refused=False,
        risk_flag=risk.risk,
        max_retrieval_score=max_score,
    )


def _expand_hypertension_query(query: str) -> str:
    """Optional legacy expansion for deployments that explicitly enable it."""
    q = query.lower()
    hypertension_terms = (
        "ضغط الدم", "ضغط", "blood pressure", "hypertension", "high blood pressure",
        "الضغط المرتفع", "الضغط العالي", "الضغط المنخفض", "الانقباضي", "الانبساطي",
    )
    if any(term in q for term in hypertension_terms):
        return query + " hypertension high blood pressure blood pressure systolic diastolic treatment diagnosis measurement"
    return query
