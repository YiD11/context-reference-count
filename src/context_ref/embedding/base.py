"""Base embedding function interface."""

from abc import ABC, abstractmethod


class EmbeddingFunction(ABC):
    """Abstract base class for embedding functions."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default implementation calls embed() for each."""
        return [self.embed(text) for text in texts]
