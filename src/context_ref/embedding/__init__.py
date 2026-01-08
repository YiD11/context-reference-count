"""Embedding function implementations."""

from context_ref.embedding.base import EmbeddingFunction
from context_ref.embedding.default import DefaultEmbedding

__all__ = ["EmbeddingFunction", "DefaultEmbedding"]
