"""
app/core/pipeline.py
----------------------
Flow:
User -> Frontend -> Backend API
     -> Validate/Auth
     -> Retrieve Context
     -> Safety Check
     -> Generation Model
     -> Validate Citations
     -> Save Logs
     -> Return Answer or Refusal
"""

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from . import safety
from .llm import get_llm, get_prompt_template, get_small_talk_prompt
from .vectorstore import similarity_search_with_score


# لو أعلى score أقل من threshold،
# بنعتبر إن مفيش سياق مفيد ونروح للـ General Knowledge
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
        HumanMessage(content=m["content"])
        if m["role"] == "user"
        else AIMessage(content=m["content"])
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
    # 1.5) Small Talk / Greetings
    # ----------------------------------------------------------------

    if safety.is_small_talk(query):

        llm = get_llm()
        prompt = get_small_talk_prompt()

        final_prompt = prompt.invoke(
            {
                "question": query,
                "chat_history": _build_lc_history(chat_history),
            }
        )

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

    retrieved = similarity_search_with_score(
        project_id,
        query,
        k=top_k,
    )

    # ----------------------------------------------------------------
    # DEBUG
    # ----------------------------------------------------------------

    print("\n========== RETRIEVAL DEBUG ==========")
    print("Project ID:", project_id)
    print("Query:", query)
    print("Retrieved count:", len(retrieved))

    for doc, score in retrieved:
        print(
            "Document:",
            doc.metadata.get(
                "document_name",
                doc.metadata.get("source", "unknown"),
            ),
            "| Score:",
            score,
            "| Page:",
            doc.metadata.get("page_number"),
            "| Section:",
            doc.metadata.get("section_title"),
            "| Chunk:",
            doc.metadata.get("chunk_id"),
        )

    print("=====================================\n")

    # ----------------------------------------------------------------
    # Calculate retrieval score / confidence
    # ----------------------------------------------------------------

    max_score = max(
        (score for _, score in retrieved),
        default=0.0,
    )

    confidence = safety.confidence_from_score(max_score)

    # ----------------------------------------------------------------
    # Determine whether retrieved context is useful enough
    # ----------------------------------------------------------------

    has_useful_context = (
        confidence != safety.CONFIDENCE_INSUFFICIENT
        and bool(retrieved)
    )

    # ----------------------------------------------------------------
    # Build context ONLY if retrieval is useful
    # ----------------------------------------------------------------

    context_text = (
        "\n\n".join(
            doc.page_content
            for doc, _ in retrieved
        )
        if has_useful_context
        else ""
    )

    # ----------------------------------------------------------------
    # Build citations from ALL retrieved documents
    #
    # IMPORTANT:
    # Citations are returned even if the retrieval score is below
    # the useful-context threshold.
    # ----------------------------------------------------------------

    citations = [
        PipelineCitation(
            document=doc.metadata.get(
                "document_name",
                doc.metadata.get("source", "unknown"),
            ),
            section=doc.metadata.get("section_title"),
            page=doc.metadata.get("page_number"),
            chunk_id=doc.metadata.get("chunk_id"),
            score=round(score, 4),
        )
        for doc, score in retrieved
    ]

    # ----------------------------------------------------------------
    # 3) Generation
    # ----------------------------------------------------------------

    llm = get_llm()
    prompt = get_prompt_template()

    final_prompt = prompt.invoke(
        {
            "context": (
                context_text
                if has_useful_context
                else "(لا يوجد سياق ذي صلة من المستندات المسجلة)"
            ),
            "question": query,
            "chat_history": _build_lc_history(chat_history),
        }
    )

    response = llm.invoke(final_prompt)

    answer_text = response.content

    # ----------------------------------------------------------------
    # 3.5) General Knowledge Fallback
    # ----------------------------------------------------------------

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
    # 4) Validate Answer / Unsupported Claims
    # ----------------------------------------------------------------

    retrieved_texts = [
        doc.page_content
        for doc, _ in retrieved
    ]

    unsupported = safety.unsupported_claims(
        answer_text,
        retrieved_texts,
    )

    if unsupported:

        confidence = safety.CONFIDENCE_LOW

        answer_text += (
            "\n\n⚠️ تنبيه: جزء من الإجابة دي مش متأكدين "
            "إنه مدعوم بالكامل بالنص المسترجع من الدليل الرسمي "
            "— راجع المصادر تحت قبل ما تعتمد عليها."
        )

    # ----------------------------------------------------------------
    # 5) Final Result
    # ----------------------------------------------------------------

    return PipelineResult(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        refused=False,
        risk_flag=risk.risk,
        max_retrieval_score=max_score,
    )
