from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from ..config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings():
    """Cached embedding model instance (loaded once per process)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
