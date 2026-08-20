"""
app/core/pipeline.py

Flow:
User
  -> Safety / Risk Classification
  -> Retrieve relevant clinical documents
  -> Decide whether retrieved evidence is sufficient
  -> Generate answer using retrieved evidence
  -> Validate answer against retrieved evidence
  -> Return answer + citations
"""

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from . import safety
from .llm import get_llm, get_prompt_template, get_small_talk_prompt
from .vectorstore import similarity_search_with_score


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


def _build_citations(retrieved) -> List[PipelineCitation]:
    """
    Build citations from every document returned by the vectorstore.

    Important:
    A citation means that the document was retrieved.
    It does NOT automatically mean that the final answer was fully
    supported by that document.
    """

    return [
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


def run_pipeline(
    project_id: str,
    query: str,
    top_k: int = 5,
    chat_history: Optional[list] = None,
) -> PipelineResult:

    chat_history = chat_history or []

    # ================================================================
    # 1) INPUT RISK CLASSIFICATION
    # ================================================================

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

    # ================================================================
    # 1.5) SMALL TALK
    # ================================================================

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

    # ================================================================
    # 2) RETRIEVE CLINICAL CONTEXT
    # ================================================================

    retrieved = similarity_search_with_score(
        project_id,
        query,
        k=top_k,
    )

    max_score = max(
        (score for _, score in retrieved),
        default=0.0,
    )

    confidence = safety.confidence_from_score(max_score)

    # ================================================================
    # DEBUG
    # ================================================================

    print("\n========== RETRIEVAL DEBUG ==========")
    print("Project ID:", project_id)
    print("Query:", query)
    print("Retrieved count:", len(retrieved))
    print("Max score:", max_score)
    print("Confidence:", confidence)

    for doc, score in retrieved:
        print(
            "Document:",
            doc.metadata.get(
                "document_name",
                doc.metadata.get("source", "unknown"),
            ),
            "| Score:",
            round(score, 4),
            "| Page:",
            doc.metadata.get("page_number"),
            "| Section:",
            doc.metadata.get("section_title"),
            "| Chunk:",
            doc.metadata.get("chunk_id"),
        )

    print("=====================================\n")

    # ================================================================
    # 3) BUILD CITATIONS
    #
    # Citations are ALWAYS built from retrieved documents.
    # They are NOT dependent on the confidence threshold.
    # ================================================================

    citations = _build_citations(retrieved)

    # ================================================================
    # 4) DETERMINE WHETHER EVIDENCE IS STRONG ENOUGH
    # ================================================================

    has_useful_context = (
        bool(retrieved)
        and confidence != safety.CONFIDENCE_INSUFFICIENT
    )

    # ================================================================
    # 5) BUILD CONTEXT FOR LLM
    # ================================================================

    if has_useful_context:

        context_text = "\n\n".join(
            doc.page_content
            for doc, _ in retrieved
        )

    else:

        context_text = (
            "لا يوجد سياق طبي مسترجع بدرجة كافية من المصادر المسجلة."
        )

    # ================================================================
    # 6) GENERATION
    # ================================================================

    llm = get_llm()
    prompt = get_prompt_template()

    final_prompt = prompt.invoke(
        {
            "context": context_text,
            "question": query,
            "chat_history": _build_lc_history(chat_history),
        }
    )

    response = llm.invoke(final_prompt)

    answer_text = response.content

    # ================================================================
    # 7) FALLBACK
    # ================================================================

    if not has_useful_context:

        return PipelineResult(
            answer=answer_text,
            citations=citations,
            confidence=GENERAL_KNOWLEDGE_FALLBACK_LABEL,
            refused=False,
            risk_flag=risk.risk,
            max_retrieval_score=max_score,
        )

    # ================================================================
    # 8) VALIDATE ANSWER AGAINST RETRIEVED EVIDENCE
    # ================================================================

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
            "\n\n⚠️ تنبيه: بعض المعلومات في الإجابة "
            "قد لا تكون مدعومة بالكامل بالنص المسترجع. "
            "يرجى مراجعة المصادر المذكورة."
        )

    # ================================================================
    # 9) FINAL RESULT
    # ================================================================

    return PipelineResult(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        refused=False,
        risk_flag=risk.risk,
        max_retrieval_score=max_score,
    )
