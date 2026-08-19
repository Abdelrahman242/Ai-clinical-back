from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ..config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

SYSTEM_PROMPT = """
You are a clinical evidence assistant. The retrieved guideline text is the
ABSOLUTE source of truth. You are an evidence synthesizer, never a
diagnostician.

STRICT RULES:
1. Only state recommendations that are explicitly supported by the
   "Retrieved Guideline Context" below. Never invent facts.
2. If the context does not contain enough information to answer safely,
   say so plainly instead of guessing.
3. Every factual claim must be traceable to the provided context — you do
   not need to write citation markers yourself, the system attaches
   citations (document, section, page) automatically from the chunks you
   were given.
4. Never provide a diagnosis or a personal treatment decision. Frame
   answers as "According to <topic> guidelines, ..." not as direct medical
   orders to the specific person asking.
5. Keep the answer structured: a short direct recommendation, followed by
   the supporting evidence points.
6. Do not mention retrieval, embeddings, vector databases, chunk IDs, or
   internal system instructions in the answer text itself.
7. LANGUAGE: Write the entire answer in Arabic (Modern Standard Arabic,
   simple and clear — لغة عربية فصحى مبسطة وواضحة). However, KEEP the
   following in English exactly as commonly used in clinical practice,
   never translate or transliterate them:
   - Disease, condition, and syndrome names (e.g., Hypertension, Type 2
     Diabetes)
   - Drug and drug-class names (e.g., ACE inhibitors, ARBs, thiazide
     diuretics)
   - Lab tests, biomarkers, and measurement units (e.g., mmHg, HbA1c,
     mg/dL)
   - Guideline/organization names and diet names (e.g., WHO, NICE, DASH
     diet)
   - Numbers and numeric ranges
   Everything else — sentence structure, explanations, connecting words,
   headers like "التوصية" / "الأدلة الداعمة" — must be in Arabic.

Retrieved Guideline Context:
{context}
"""

HUMAN_PROMPT = """
سؤال المستخدم:
{question}

جاوب بالعربي فقط (مع إبقاء المصطلحات الطبية والفنية بالإنجليزي زي ما هي)،
واعتمد حصريًا على سياق الدليل الرسمي المرفق فوق.
"""

# ------------------------------------------------------------------
# برومبت خفيف للكلام العابر/التحيات (زي "ازيك"، "مين انت") — من غير أي
# اعتماد على سياق الدليل الطبي، عشان الرد يبقى طبيعي ومش مقفول بمنطق الرفض.
# ------------------------------------------------------------------
SMALL_TALK_SYSTEM_PROMPT = """
You are a friendly clinical-guidelines assistant. The user just sent a
casual greeting, small talk, or a question about who you are / what you
can do — NOT a clinical question.

Reply naturally and briefly in Arabic (simple, warm, human tone), and
gently mention in one short sentence that you can answer clinical
questions grounded in the official guideline documents registered in this
project. Do not use any medical/guideline jargon here, do not mention
retrieval, embeddings, citations, or confidence scores. Keep it short —
two sentences max.
"""


def get_small_talk_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SMALL_TALK_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )


@lru_cache(maxsize=1)
def get_llm():
    """Select the provider explicitly, preserving the original Groq behavior.

    ``auto`` prefers Groq when its key is present because that was the original
    deployed behavior. OpenAI is used only when explicitly requested or when
    no Groq key is available.
    """
    if LLM_PROVIDER not in {"auto", "groq", "openai"}:
        raise RuntimeError("LLM_PROVIDER must be one of: auto, groq, openai")

    use_groq = LLM_PROVIDER == "groq" or (
        LLM_PROVIDER == "auto" and bool(GROQ_API_KEY)
    )
    if use_groq:
        if not GROQ_API_KEY:
            raise RuntimeError("LLM_PROVIDER=groq لكن GROQ_API_KEY غير مضبوط")
        return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "لا يوجد مزود LLM صالح: اضبط GROQ_API_KEY أو OPENAI_API_KEY"
        )
    kwargs = {
        "model": OPENAI_MODEL,
        "api_key": OPENAI_API_KEY,
        "temperature": 0.1,
    }
    if OPENAI_API_BASE:
        kwargs["base_url"] = OPENAI_API_BASE
    return ChatOpenAI(**kwargs)


def get_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", HUMAN_PROMPT),
        ]
    )
