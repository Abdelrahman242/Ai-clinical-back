from langchain_core.documents import Document

from app.core.embeddings import get_embeddings

embeddings = get_embeddings()
documents = [
    Document(page_content="Hypertension is persistently elevated blood pressure.", metadata={"id": "hypertension"}),
    Document(page_content="Aspirin can reduce platelet aggregation in selected cardiovascular settings.", metadata={"id": "aspirin"}),
]
query = "ما الحالة التي تعني أن ضغط الدم يظل مرتفعًا بشكل مستمر؟"
query_vector = embeddings.embed_query(query)
document_vectors = embeddings.embed_documents([doc.page_content for doc in documents])

def cosine(left, right):
    return sum(a * b for a, b in zip(left, right))

scores = [cosine(query_vector, vector) for vector in document_vectors]
assert scores[0] > scores[1], scores
print(f"semantic_retrieval=passed hypertension_score={scores[0]:.4f} unrelated_score={scores[1]:.4f}")
