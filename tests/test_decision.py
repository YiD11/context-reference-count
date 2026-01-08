"""Tests for decision making logic with split reference counting."""

import uuid

import pytest

from context_ref.core.cache import ToolCache
from context_ref.core.config import CacheConfig
from context_ref.core.storage import ChromaVectorStore, MemoryStorageBackend
from context_ref.embedding.base import EmbeddingFunction
from context_ref.interceptor.wrapper import (
    CacheDecision,
    ToolInterceptor,
)


class MockEmbedding(EmbeddingFunction):
    """Mock embedding that returns consistent vectors for same input."""

    @property
    def dimension(self) -> int:
        return 16

    def embed(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 32, 2)]


def create_test_cache(embedding_func=None) -> ToolCache:
    """Create a ToolCache with isolated storage for testing."""
    collection_name = f"test_decision_{uuid.uuid4().hex[:8]}"
    storage = MemoryStorageBackend()
    vector_store = ChromaVectorStore(collection_name=collection_name)
    config = CacheConfig(
        similarity_threshold=0.5,
        reuse_threshold=0.95,
    )
    return ToolCache(
        config=config,
        embedding_func=embedding_func or MockEmbedding(),
        storage=storage,
        vector_store=vector_store,
    )


class TestToolInterceptorDecision:
    """Test cases for ToolInterceptor decision making."""

    @pytest.fixture
    def cache(self) -> ToolCache:
        return create_test_cache()

    @pytest.fixture
    def interceptor(self, cache: ToolCache) -> ToolInterceptor:
        return ToolInterceptor(cache=cache)

    def test_decision_types(self, cache: ToolCache, interceptor: ToolInterceptor) -> None:
        """Test EXECUTE, REUSE, and PROVIDE_CONTEXT decisions."""
        # EXECUTE - no cache
        result = interceptor.decide("search", {"query": "test"})
        assert result.decision == CacheDecision.EXECUTE
        assert result.cache_hit is None
        assert result.context_hints is None

        # Save entry and test REUSE
        cache.save(
            tool_name="search",
            input_args={"query": "test"},
            output="cached result",
        )

        result = interceptor.decide("search", {"query": "test"})
        assert result.decision == CacheDecision.REUSE
        assert result.cache_hit is not None
        assert result.cache_hit.entry.output == "cached result"
        assert result.cache_hit.similarity >= 0.95

    def test_reuse_increments_reuse_count(
        self, cache: ToolCache, interceptor: ToolInterceptor
    ) -> None:
        """Test that REUSE decision increments reuse_count, not provide_context_count."""
        entry = cache.save(
            tool_name="search",
            input_args={"query": "test"},
            output="cached result",
        )

        data = cache.storage.get(entry.id)
        assert data is not None
        assert data["reuse_count"] == 0
        assert data["provide_context_count"] == 0

        # Make multiple REUSE decisions
        for i in range(1, 6):
            result = interceptor.decide("search", {"query": "test"})
            assert result.decision == CacheDecision.REUSE

            data = cache.storage.get(entry.id)
            assert data is not None
            assert data["reuse_count"] == i
            assert data["provide_context_count"] == 0

    def test_context_increments_context_count(self, cache: ToolCache) -> None:
        """Test that PROVIDE_CONTEXT decision increments provide_context_count."""
        collection_name = f"test_context_{uuid.uuid4().hex[:8]}"
        storage = MemoryStorageBackend()
        vector_store = ChromaVectorStore(collection_name=collection_name)
        config = CacheConfig(
            similarity_threshold=0.3,
            reuse_threshold=0.99,
        )
        cache = ToolCache(
            config=config,
            embedding_func=MockEmbedding(),
            storage=storage,
            vector_store=vector_store,
        )
        interceptor = ToolInterceptor(cache=cache)

        entry = cache.save(
            tool_name="search",
            input_args={"query": "python programming"},
            output="cached result",
        )

        data = cache.storage.get(entry.id)
        assert data is not None
        assert data["reuse_count"] == 0
        assert data["provide_context_count"] == 0

        result = interceptor.decide("search", {"query": "python programming basics"})

        if result.decision == CacheDecision.PROVIDE_CONTEXT:
            assert result.context_hints is not None
            for hit in result.context_hints:
                hit_data = cache.storage.get(hit.entry.id)
                assert hit_data is not None
                assert hit_data["provide_context_count"] >= 1

    def test_format_context_hints(
        self, cache: ToolCache, interceptor: ToolInterceptor
    ) -> None:
        """Test formatting of context hints with reference counts."""
        entry = cache.save(
            tool_name="search",
            input_args={"query": "test query"},
            output="test result",
        )

        cache.increment_reuse(entry.id)
        cache.increment_reuse(entry.id)
        cache.increment_context(entry.id)

        hits = cache.search("search", {"query": "test query"})
        formatted = interceptor.format_context_hints(hits)

        assert "Historical tool usage suggestions" in formatted
        assert "search" in formatted
        assert "reuse: 2" in formatted
        assert "context: 1" in formatted
        assert "total_refs: 3" in formatted

        # Empty hints should return empty string
        assert interceptor.format_context_hints([]) == ""

    def test_score_increases_with_references(
        self, cache: ToolCache, interceptor: ToolInterceptor
    ) -> None:
        """Test that entry score increases as it gets more references."""
        entry = cache.save(
            tool_name="search",
            input_args={"query": "score test"},
            output="result",
        )

        initial_score = cache.storage.get_score(entry.id)
        assert initial_score is not None

        for _ in range(3):
            interceptor.decide("search", {"query": "score test"})

        final_score = cache.storage.get_score(entry.id)
        assert final_score is not None
        assert final_score > initial_score
