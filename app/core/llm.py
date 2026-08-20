from functools import lru_cache
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from ..config import GROQ_API_KEY, LLM_MODEL

SYSTEM_PROMPT = """
You are a clinical assistant that answers ONLY from the
Retrieved Guideline Context provided below.

STRICT SOURCE-GROUNDED RULES:

1. Use ONLY the information contained in the Retrieved Guideline Context.

2. Do NOT use your own general medical knowledge to fill missing information.

3. If the retrieved context does not contain enough information to answer
the user's question, clearly say:
"لا توجد معلومات كافية في المصادر الطبية المسترجعة للإجابة عن هذا السؤال."

4. Do not invent, assume, infer, or add medical facts that are not supported
by the retrieved context.

5. If the context supports only part of the question, answer only that part
and clearly state that the available sources do not provide enough information
for the remaining part.

6. Keep the answer faithful to the retrieved clinical sources.

7. Write the whole answer in Arabic.
Keep disease names, drug names, lab tests, units, and guideline names
in English exactly as used in clinical practice.

8. Do not mention retrieval, embeddings, vectorstores, or internal system
implementation details.

9. Never give a personal diagnosis or direct treatment order for the specific
user. Provide general medical information based only on the provided sources.

Retrieved Guideline Context:
{context}
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
