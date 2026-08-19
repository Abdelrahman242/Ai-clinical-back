"""
app/core/safety.py
--------------------
Safety & Guardrail Workflow (Day 4 من الأجندة):

1. Input Risk Classification   -> classify_input_risk()
2. Retrieval Confidence Threshold -> confidence_from_score()
3. Unsupported Claim Detection  -> unsupported_claims()
"""

import re
from dataclasses import dataclass
from typing import List

from ..config import (
    EMERGENCY_KEYWORDS,
    OUT_OF_SCOPE_HINT_KEYWORDS,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    RETRIEVAL_LOW_CONFIDENCE_THRESHOLD,
)

RISK_ALLOWED = "allowed"
RISK_NEEDS_CAUTION = "needs_caution"
RISK_REFUSE = "refuse"

CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"
CONFIDENCE_INSUFFICIENT = "Insufficient Evidence"

_PATIENT_SCENARIO_HINTS = ["my patient", "i have", "i feel", "i am experiencing", "مريضي", "بحس", "عندي"]

# كلام عابر/تحية — مفروض ياخد رد ودود عادي، مش يترفض بـ "أدلة غير كافية"
_SMALL_TALK_PATTERNS = [
    # تحيات وسؤال عن الحال
    "ازيك", "إزيك", "ازايك", "إزايك", "عامل ايه", "عامل إيه", "اخبارك", "أخبارك",
    "ايه الاخبار", "إيه الأخبار", "ايه اخبارك", "كيفك", "كيف حالك", "شلونك",
    "صباح الخير", "مساء الخير", "السلام عليكم", "اهلا", "أهلا", "هاي", "هلا",
    "hi", "hello", "hey", "how are you", "what's up", "whats up", "good morning",
    "good evening",
    # سؤال عن هوية/قدرات النظام
    "مين انت", "من أنت", "انت مين", "إنت مين", "بتعمل ايه", "تقدر تعمل ايه",
    "who are you", "what can you do", "what do you do",
    # شكر/إنهاء
    "شكرا", "شكرًا", "متشكر", "تسلم", "الله يعافيك", "thanks", "thank you",
    "bye", "مع السلامة", "باي",
]


def is_small_talk(query: str) -> bool:
    """
    بيتعرف على كلام عابر/تحية/شكر مش سؤال طبي فعلي — عشان النظام يرد عليه
    بشكل طبيعي بدل ما يترفض بمنطق "الأدلة غير كافية" اللي مخصص للأسئلة
    الطبية بس. فحص بسيط بالكلمات المفتاحية، مقصود إنه يكون واسع شوية عشان
    يمسك أشكال مختلفة من العامية.
    """
    q = query.strip().lower()
    if not q:
        return False
    # سؤال طويل ومفصل غالبًا سؤال طبي حقيقي مش تحية، حتى لو فيه كلمة زي "ازيك" جواه
    if len(q) > 60:
        return False
    return any(pattern in q for pattern in _SMALL_TALK_PATTERNS)


@dataclass
class RiskAssessment:
    risk: str
    reason: str


def classify_input_risk(query: str) -> RiskAssessment:
    q = query.lower()

    for kw in EMERGENCY_KEYWORDS:
        if kw.lower() in q:
            return RiskAssessment(
                risk=RISK_REFUSE,
                reason="السؤال بيوصف حالة طوارئ محتملة — لازم تواصل فوري مع خدمات الطوارئ/الطبيب، مش أداة بحث في أدلة.",
            )

    for kw in OUT_OF_SCOPE_HINT_KEYWORDS:
        if kw.lower() in q:
            return RiskAssessment(
                risk=RISK_REFUSE,
                reason="السؤال خارج نطاق المشروع الحالي (الموضوع السريري المحدد فقط).",
            )

    if any(hint in q for hint in _PATIENT_SCENARIO_HINTS):
        return RiskAssessment(
            risk=RISK_NEEDS_CAUTION,
            reason="السؤال بيوصف سيناريو مريض شخصي — الإجابة هتتحط كمعلومة من الدليل الرسمي بس، مش تشخيص أو قرار علاجي.",
        )

    return RiskAssessment(risk=RISK_ALLOWED, reason="")


def confidence_from_score(max_score: float) -> str:
    if max_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        return CONFIDENCE_INSUFFICIENT
    if max_score < RETRIEVAL_LOW_CONFIDENCE_THRESHOLD:
        return CONFIDENCE_LOW
    if max_score < 0.75:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_HIGH


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+")


def unsupported_claims(answer_text: str, retrieved_texts: List[str], min_overlap: float = 0.12) -> List[str]:
    """
    فحص بسيط لـ faithfulness: كل جملة في الإجابة لازم يكون فيها overlap معقول
    (كلمات مشتركة) مع نص الـ chunks المسترجعة. الجمل اللي معندهاش overlap
    كافي بترجع كـ "unsupported".
    ده heuristic خفيف مناسب لـ MVP هاكاثون — مش بديل لتقييم بشري كامل.
    """
    if not retrieved_texts:
        return _SENTENCE_SPLIT_RE.split(answer_text.strip())

    context_words = set(re.findall(r"[a-zA-Z\u0600-\u06FF]{3,}", " ".join(retrieved_texts).lower()))
    unsupported = []

    for sentence in _SENTENCE_SPLIT_RE.split(answer_text.strip()):
        sentence = sentence.strip()
        if len(sentence) < 8:
            continue
        words = re.findall(r"[a-zA-Z\u0600-\u06FF]{3,}", sentence.lower())
        if not words:
            continue
        overlap = len(set(words) & context_words) / len(set(words))
        if overlap < min_overlap:
            unsupported.append(sentence)

    return unsupported
