"""
app/core/pipeline.py

Medical RAG pipeline:
User
 -> Safety
 -> Retrieve documents
 -> Generate ONLY from retrieved documents
 -> Validate answer against retrieved documents
 -> Return answer + citations
"""

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from . import safety
from .llm import get_llm, get_prompt_template, get_small_talk_prompt
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
        HumanMessage(content=m["content"])
        if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in chat_history
    ]


def _build_citations(retrieved) -> List[PipelineCitation]:
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
    # 1) SAFETY
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
    # 2) SMALL TALK
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
    # 3) RETRIEVE
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
    print("Retrieved:", len(retrieved))
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
    # 4) NO SOURCE -> DO NOT ANSWER FROM GENERAL KNOWLEDGE
    # ================================================================

    if not retrieved:

        return PipelineResult(
            answer=(
                "لا توجد معلومات كافية في المصادر الطبية المسجلة "
                "للإجابة عن هذا السؤال."
            ),
            citations=[],
            confidence=safety.CONFIDENCE_INSUFFICIENT,
            refused=False,
            risk_flag=risk.risk,
            max_retrieval_score=0.0,
        )

    # ================================================================
    # 5) BUILD CITATIONS
    # ================================================================

    citations = _build_citations(retrieved)

    # ================================================================
    # 6) BUILD CONTEXT
    #
    # IMPORTANT:
    # كل الـretrieved documents تدخل للـLLM.
    # مفيش General Knowledge fallback.
    # ================================================================

    context_text = "\n\n".join(
        doc.page_content
        for doc, _ in retrieved
    )

    # ================================================================
    # 7) GENERATE FROM SOURCE ONLY
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
    # 8) VALIDATE AGAINST SOURCE
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

        answer_text = (
            "لم أجد في المصادر المسترجعة معلومات كافية "
            "لدعم جميع تفاصيل الإجابة المطلوبة."
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
