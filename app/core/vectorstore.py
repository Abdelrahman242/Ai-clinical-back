"""
app/core/vectorstore.py
-------------------------
كل project له فولدر index منفصل جوه VECTORSTORE_ROOT/<project_id>/.
كده كل نطاق سريري (clinical topic) معزول عن التاني، ومفيش تسريب سياق بين المشاريع.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument

from ..config import VECTORSTORE_ROOT
from .embeddings import get_embeddings


def _project_dir(project_id: str) -> Path:
    path = VECTORSTORE_ROOT / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_file(project_id: str) -> Path:
    return _project_dir(project_id) / "index.faiss"


def has_index(project_id: str) -> bool:
    return _index_file(project_id).exists()


def load_vectorstore(project_id: str) -> Optional[FAISS]:
    if not has_index(project_id):
        return None
    return FAISS.load_local(
        str(_project_dir(project_id)),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def add_chunks(project_id: str, chunks: List[LCDocument]) -> int:
    """بيضيف chunks جديدة لفهرس المشروع (بينشئه لو مش موجود)."""
    if not chunks:
        return 0

    embeddings = get_embeddings()
    store = load_vectorstore(project_id)

    if store is None:
        store = FAISS.from_documents(chunks, embeddings)
    else:
        store.add_documents(chunks)

    store.save_local(str(_project_dir(project_id)))
    return len(chunks)


def remove_document_chunks(project_id: str, document_name: str) -> None:
    """بيعيد بناء الفهرس من غير chunks المستند ده (مستخدم مع reset=True وقت الـ re-ingest)."""
    store = load_vectorstore(project_id)
    if store is None:
        return

    kept = [
        (doc_id, store.docstore.search(doc_id))
        for doc_id in store.index_to_docstore_id.values()
    ]
    remaining_docs = [
        doc for _, doc in kept
        if doc is not None and doc.metadata.get("document_name") != document_name
    ]

    if remaining_docs:
        new_store = FAISS.from_documents(remaining_docs, get_embeddings())
        new_store.save_local(str(_project_dir(project_id)))
    else:
        import shutil
        shutil.rmtree(_project_dir(project_id), ignore_errors=True)
        _project_dir(project_id)


def similarity_search_with_score(
    project_id: str, query: str, k: int = 5
) -> List[Tuple[LCDocument, float]]:
    """
    بيرجع (chunk, similarity_score) مرتبة من الأعلى للأقل.
    FAISS بترجع L2 distance، فبنحولها لـ similarity تقريبية 0-1 عشان تتقارن
    بالـ thresholds بسهولة.
    """
    store = load_vectorstore(project_id)
    if store is None:
        return []

    raw = store.similarity_search_with_score(query, k=k)
    results = []
    for doc, distance in raw:
        # نحوّل لـ float بايثون عادي هنا عشان أي حاجة تانية في السيستم (زي json.dumps)
        # متتعقدش بسبب numpy.float32 اللي FAISS بيرجعها
        distance = float(distance)
        # With normalized embeddings, FAISS L2 distance maps to cosine
        # similarity as cos = 1 - distance^2 / 2. Clamp for safe thresholds.
        similarity = 1.0 - (max(distance, 0.0) ** 2) / 2.0
        similarity = max(0.0, min(1.0, similarity))
        results.append((doc, float(similarity)))
    return results
