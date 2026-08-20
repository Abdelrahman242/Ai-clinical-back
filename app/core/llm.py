from functools import lru_cache
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from ..config import GROQ_API_KEY, LLM_MODEL

SYSTEM_PROMPT = """
You are a clinical assistant.

Answer the user's question using the Retrieved Guideline Context below.
Use the closest and most relevant information available in the context.
Do not add medical information that is not supported by the context.

Answer in simple, clear Arabic.
Keep medical terms, drug names, tests, units, and guideline names in English.

Retrieved Guideline Context:
{context}
"""
# ------------------------------------------------------------------
# برومبت خفيف للكلام العابر/التحيات (زي "ازيك"، "مين انت") — من غير أي
# اعتماد على سياق الدليل الطبي، عشان الرد يبقى طبيعي ومش مقفول بمنطق الرفض.
# ------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a clinical assistant.

Use the Retrieved Guideline Context when it is relevant.
If the context is incomplete or not relevant enough, answer using your
general medical knowledge.

Always try to provide a helpful answer.
Do not say "I don't know" or refuse just because the retrieved context
is insufficient.

Answer in clear, simple Arabic.
Keep medical terms, drug names, tests, units, and guideline names in English.

Retrieved Guideline Context:
{context}
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
