"""
app/core/pipeline.py
----------------------
Flow:
User -> Frontend -> Backend API
     -> Validate/Auth        (بيحصل في الـ router عن طريق auth.get_current_user)
     -> Retrieve Context     (vectorstore.similarity_search_with_score)
     -> Safety Check         (بس حالات الطوارئ/خارج النطاق بترفض — مش نقص الأدلة)
     -> Generation Model     (مبني على الدليل لو موجود، وإلا من المعرفة العامة)
     -> Validate Citations   (safety.unsupported_claims — للإجابات المبنية على دليل بس)
     -> Save Logs
     -> Return Answer or Refusal (رفض بس لحالات الطوارئ الحقيقية)
"""

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from . import safety
from .llm import get_llm, get_prompt_template, get_small_talk_prompt
from .vectorstore import similarity_search_with_score

# لو أعلى score أقل من كده، بنعتبر إن مفيش سياق مفيد ونروح على المعرفة العامة
GENERAL_KNOWLEDGE_FALLBACK_LABEL = "General Knowledge"


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
    # 1.5) كلام عابر/تحية
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
    retrieved = similarity_search_with_score(project_id, query, k=top_k)
    max_score = max((score for _, score in retrieved), default=0.0)
    confidence = safety.confidence_from_score(max_score)

    has_useful_context = (
    confidence != safety.CONFIDENCE_INSUFFICIENT
    and bool(retrieved))

    context_text = (
    "\n\n".join(doc.page_content for doc, _ in retrieved)
    if has_useful_context
    else "")

    # بنبني الـ citations من أي مستندات اتسترجعت، بغض النظر عن الـ threshold
    # بتاع has_useful_context — عشان المستخدم يشوف المصادر القريبة حتى لو
    # الموديل جاوب من معرفته العامة بدل ما يعتمد عليها في الـ prompt.
    citations = [
    PipelineCitation(
        document=doc.metadata.get(
            "document_name",
            doc.metadata.get("source", "unknown")
        ),
        section=doc.metadata.get("section_title"),
        page=doc.metadata.get("page_number"),
        chunk_id=doc.metadata.get("chunk_id"),
        score=round(score, 4),
    )
    for doc, score in retrieved]

    # ----------------------------------------------------------------
    # 3) Generation
    # ----------------------------------------------------------------
    llm = get_llm()
    prompt = get_prompt_template()

    final_prompt = prompt.invoke({
        "context": context_text if has_useful_context else "(لا يوجد سياق ذي صلة من المستندات المسجلة)",
        "question": query,
        "chat_history": _build_lc_history(chat_history),
    })
    response = llm.invoke(final_prompt)
    answer_text = response.content

    if not has_useful_context:
        return PipelineResult(
            answer=answer_text,
            citations=citations,
            confidence=GENERAL_KNOWLEDGE_FALLBACK_LABEL,
            refused=False,
            risk_flag=risk.risk,
            max_retrieval_score=max_score,
        )

    # ----------------------------------------------------------------
    # 4) Validate Answer / Citations (unsupported claim detection)
    # ----------------------------------------------------------------
    retrieved_texts = [doc.page_content for doc, _ in retrieved]
    unsupported = safety.unsupported_claims(answer_text, retrieved_texts)
    if unsupported:
        confidence = safety.CONFIDENCE_LOW
        answer_text += (
            "\n\n⚠️ تنبيه: جزء من الإجابة دي مش متأكدين إنه مدعوم بالكامل بالنص "
            "المسترجع من الدليل الرسمي — راجع المصادر تحت قبل ما تعتمد عليها."
        )

    return PipelineResult(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        refused=False,
        risk_flag=risk.risk,
        max_retrieval_score=max_score,
    )
