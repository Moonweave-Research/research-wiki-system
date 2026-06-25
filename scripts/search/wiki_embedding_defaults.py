"""Embedding model defaults shared across the indexing/query stack.

Lives in its own module to break the circular import between
wiki_fact_embeddings (the index builder) and wiki_query (which uses
the same default for its CLI). No other dependencies — keep it that way.
"""

DEFAULT_FASTEMBED_MODEL = "intfloat/multilingual-e5-large"

EMBEDDING_DIM_BY_MODEL: dict[str, int] = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-small-en-v1.5": 384,
}
