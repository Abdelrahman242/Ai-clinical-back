from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from ..config import GROQ_API_KEY, LLM_MODEL

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

Retrieved Guideline Context:
{context}
"""

HUMAN_PROMPT = """
User Question:
{question}

Answer strictly using the guideline context above.
"""


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
