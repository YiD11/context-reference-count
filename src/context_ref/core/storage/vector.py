"""Vector store interface for similarity search."""

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add entries to the vector store."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search for similar embeddings.

        Returns dict with keys: "ids", "distances", "metadatas", "documents".
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete entries by ID."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...
