"""Tests for ToolCache."""

import uuid

import pytest

from context_ref.core.cache import ToolCache
from context_ref.core.config import CacheConfig
from context_ref.core.storage import ChromaVectorStore, MemoryStorageBackend
from context_ref.embedding.base import EmbeddingFunction


class MockEmbedding(EmbeddingFunction):
    """Mock embedding function for testing with hash-based differentiation."""

    @property
    def dimension(self) -> int:
        return 16

    def embed(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 32, 2)]


def create_test_cache(embedding_func=None, eviction_policy="score", max_size=10) -> ToolCache:
    """Create a ToolCache with isolated storage for testing."""
    collection_name = f"test_cache_{uuid.uuid4().hex[:8]}"
    storage = MemoryStorageBackend()
    vector_store = ChromaVectorStore(collection_name=collection_name)
    config = CacheConfig(
        similarity_threshold=0.5,
        reuse_threshold=0.95,
        max_cache_size=max_size,
        eviction_policy=eviction_policy,
    )
    return ToolCache(
        config=config,
        embedding_func=embedding_func or MockEmbedding(),
        storage=storage,
        vector_store=vector_store,
    )


class TestToolCache:
    """Test cases for ToolCache."""

    @pytest.fixture
    def cache(self) -> ToolCache:
        return create_test_cache()

    def test_save_and_search(self, cache: ToolCache) -> None:
        """Test saving and searching cache entries."""
        cache.save(
            tool_name="search",
            input_args={"query": "what is python"},
            output="Python is a programming language",
        )

        hits = cache.search(
            tool_name="search",
            input_args={"query": "what is python"},
        )

        assert len(hits) >= 1
        assert hits[0].entry.tool_name == "search"
        assert hits[0].entry.output == "Python is a programming language"

    def test_save_updates_existing_entry(self, cache: ToolCache) -> None:
        """Test that saving same input updates existing entry."""
        cache.save(
            tool_name="search",
            input_args={"query": "test"},
            output="result1",
        )
        entry = cache.save(
            tool_name="search",
            input_args={"query": "test"},
            output="result2",
        )

        assert entry.reuse_count == 0
        assert entry.provide_context_count == 0
        assert entry.output == "result2"

    def test_increment_reference_counts(self, cache: ToolCache) -> None:
        """Test incrementing reuse and context counts."""
        entry = cache.save(
            tool_name="search",
            input_args={"query": "test"},
            output="result",
        )

        cache.increment_reuse(entry.id)
        cache.increment_reuse(entry.id)
        data = cache.storage.get(entry.id)
        assert data is not None
        assert data["reuse_count"] == 2
        assert data["provide_context_count"] == 0

        cache.increment_context(entry.id)
        data = cache.storage.get(entry.id)
        assert data is not None
        assert data["reuse_count"] == 2
        assert data["provide_context_count"] == 1

    def test_search_filters_by_tool_name(self, cache: ToolCache) -> None:
        """Test that search results are filtered by tool name."""
        cache.save(
            tool_name="search",
            input_args={"query": "test"},
            output="search result",
        )
        cache.save(
            tool_name="calculator",
            input_args={"query": "test"},
            output="calc result",
        )

        search_hits = cache.search(
            tool_name="search",
            input_args={"query": "test"},
        )
        calc_hits = cache.search(
            tool_name="calculator",
            input_args={"query": "test"},
        )

        assert all(h.entry.tool_name == "search" for h in search_hits)
        assert all(h.entry.tool_name == "calculator" for h in calc_hits)

    def test_get_best_match(self, cache: ToolCache) -> None:
        """Test getting best match from cache."""
        cache.save(
            tool_name="search",
            input_args={"query": "hello world"},
            output="Hello World!",
        )

        hit = cache.get_best_match(
            tool_name="search",
            input_args={"query": "hello world"},
        )

        assert hit is not None
        assert hit.entry.output == "Hello World!"

        hit_none = cache.get_best_match(
            tool_name="nonexistent",
            input_args={"query": "test"},
        )
        assert hit_none is None

    def test_eviction_policies(self) -> None:
        """Test different eviction policies."""
        for policy in ["lru", "lfu", "fifo", "score"]:
            cache = create_test_cache(eviction_policy=policy, max_size=3)

            for i in range(5):
                cache.save(
                    tool_name="search",
                    input_args={"query": f"query_{i}"},
                    output=f"result_{i}",
                )

            assert cache.storage.size() <= 3

    def test_clear_cache(self, cache: ToolCache) -> None:
        """Test clearing cache."""
        cache.save(
            tool_name="search",
            input_args={"query": "test"},
            output="result",
        )
        cache.clear()

        stats = cache.stats()
        assert stats["total_entries"] == 0

    def test_stats(self, cache: ToolCache) -> None:
        """Test cache statistics."""
        for i in range(3):
            cache.save(
                tool_name="search",
                input_args={"query": f"query_{i}"},
                output=f"result_{i}",
            )

        entries = list(cache.storage.keys())
        cache.increment_reuse(entries[0])
        cache.increment_reuse(entries[0])
        cache.increment_context(entries[1])

        stats = cache.stats()
        assert stats["total_entries"] == 3
        assert stats["total_reuse_count"] == 2
        assert stats["total_context_count"] == 1
        assert stats["total_references"] == 3

    def test_weighted_score_calculation(self, cache: ToolCache) -> None:
        """Test that weighted score considers both reuse and context counts."""
        entry = cache.save(
            tool_name="search",
            input_args={"query": "score test"},
            output="result",
        )

        initial_score = cache.storage.get_score(entry.id)
        assert initial_score is not None

        cache.increment_reuse(entry.id)
        reuse_score = cache.storage.get_score(entry.id)
        assert reuse_score is not None
        assert reuse_score > initial_score

        entry2 = cache.save(
            tool_name="search",
            input_args={"query": "score test 2"},
            output="result2",
        )
        cache.increment_context(entry2.id)
        context_score = cache.storage.get_score(entry2.id)
        assert context_score is not None

    def test_entry_has_uuid(self, cache: ToolCache) -> None:
        """Test that saved entries have UUID."""
        entry = cache.save(
            tool_name="search",
            input_args={"query": "uuid test"},
            output="result",
        )
        assert entry.uuid is not None
        assert len(entry.uuid) == 36

    def test_score_updates_on_reference_increment(self, cache: ToolCache) -> None:
        """Test that score is updated when reference counts change."""
        entry = cache.save(
            tool_name="search",
            input_args={"query": "test"},
            output="result",
        )

        score_before = cache.storage.get_score(entry.id)

        for _ in range(5):
            cache.increment_reuse(entry.id)

        score_after = cache.storage.get_score(entry.id)
        assert score_after > score_before
