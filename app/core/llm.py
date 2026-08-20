from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from ..config import GROQ_API_KEY, LLM_MODEL

SYSTEM_PROMPT = """
You are a clinical assistant. Prefer the "Retrieved Guideline Context"
below when it's relevant — but if it doesn't fully answer the question,
still answer using your own general medical knowledge instead of refusing.

Rules:
1. If the context answers the question, base your answer on it.
2. If the context is empty, irrelevant, or incomplete, answer from your
   own general medical knowledge — don't refuse just because the context
   is thin.
3. Never give a personal diagnosis or treatment order — frame things as
   general medical information, not direct orders to the specific person
   asking.
4. Write the whole answer in Arabic (simple, clear). Keep disease names,
   drug names, lab tests, units, and organization/guideline names in
   English exactly as used in clinical practice (e.g., Hypertension, ACE
   inhibitors, mmHg, WHO).
5. Don't mention retrieval, embeddings, or internal system details.

Retrieved Guideline Context:
{context}
"""

HUMAN_PROMPT = """
سؤال المستخدم:
{question}

جاوب بالعربي (المصطلحات الطبية بالإنجليزي زي ما هي).
"""

# ------------------------------------------------------------------
# برومبت خفيف للكلام العابر/التحيات (زي "ازيك"، "مين انت") — من غير أي
# اعتماد على سياق الدليل الطبي، عشان الرد يبقى طبيعي ومش مقفول بمنطق الرفض.
# ------------------------------------------------------------------
SMALL_TALK_SYSTEM_PROMPT = """
You are a friendly clinical assistant. The user sent a casual greeting or
small talk — not a clinical question. Reply naturally and briefly in
Arabic. Keep it short — two sentences max.
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
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY مش متظبط. حط قيمته في .env (خد الـ key من "
            "https://console.groq.com/keys) وأعد تشغيل السيرفر."
        )
    return ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


def get_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", HUMAN_PROMPT),
        ]
    )
