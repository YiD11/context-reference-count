"""Default embedding implementation using sentence-transformers."""

from functools import cached_property

from context_ref.embedding.base import EmbeddingFunction


class DefaultEmbedding(EmbeddingFunction):
    """
    Default embedding using sentence-transformers.

    Uses the all-MiniLM-L6-v2 model by default, which provides a good
    balance between speed and quality for semantic similarity tasks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    @cached_property
    def _encoder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._model_name)

    @property
    def dimension(self) -> int:
        return self._encoder.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        embedding = self._encoder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._encoder.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
