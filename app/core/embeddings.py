from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from ..config import EMBEDDING_MODEL, EMBEDDING_NORMALIZE


@lru_cache(maxsize=1)
def get_embeddings():
    """Use one normalized multilingual embedding model for indexing and queries."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
    )
