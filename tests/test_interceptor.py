"""Tests for tool interceptor with split reference counting."""

import uuid

import pytest

from context_ref.core.cache import ToolCache
from context_ref.core.config import CacheConfig
from context_ref.core.storage import ChromaVectorStore, MemoryStorageBackend
from context_ref.embedding.base import EmbeddingFunction
from context_ref.interceptor.wrapper import ToolInterceptor


class MockEmbedding(EmbeddingFunction):
    """Mock embedding for testing with better differentiation."""

    @property
    def dimension(self) -> int:
        return 16

    def embed(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 32, 2)]


def create_test_cache(embedding_func=None) -> ToolCache:
    """Create a ToolCache with isolated storage for testing."""
    collection_name = f"test_interceptor_{uuid.uuid4().hex[:8]}"
    storage = MemoryStorageBackend()
    vector_store = ChromaVectorStore(collection_name=collection_name)
    config = CacheConfig(
        similarity_threshold=0.5,
        reuse_threshold=0.95,
        eviction_policy="score",
    )
    return ToolCache(
        config=config,
        embedding_func=embedding_func or MockEmbedding(),
        storage=storage,
        vector_store=vector_store,
    )


class TestToolInterceptor:
    """Test cases for ToolInterceptor."""

    @pytest.fixture
    def interceptor(self) -> ToolInterceptor:
        config = CacheConfig(
            similarity_threshold=0.5,
            reuse_threshold=0.95,
        )
        cache = create_test_cache()
        return ToolInterceptor(cache=cache, config=config)

    def test_wrap_tool_caches_result(self, interceptor: ToolInterceptor) -> None:
        """Test that wrapped tool caches results and reuses them."""
        call_count = 0

        def my_tool(query: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Result for {query}"

        wrapped = interceptor.wrap_tool(my_tool)

        result1 = wrapped(query="test")
        assert result1 == "Result for test"
        assert call_count == 1

        result2 = wrapped(query="test")
        assert result2 == "Result for test"
        assert call_count == 1

    def test_wrap_tool_preserves_metadata(self, interceptor: ToolInterceptor) -> None:
        """Test that wrapped tool preserves function name and docstring."""
        def my_tool(query: str) -> str:
            """My tool docstring."""
            return query

        wrapped = interceptor.wrap_tool(my_tool)

        assert wrapped.__name__ == "my_tool"
        assert wrapped.__doc__ == "My tool docstring."

    def test_stats_tracking(self, interceptor: ToolInterceptor) -> None:
        """Test that stats are tracked correctly."""
        def my_tool(query: str) -> str:
            return query

        wrapped = interceptor.wrap_tool(my_tool)

        wrapped(query="first")
        wrapped(query="first")
        wrapped(query="second")

        stats = interceptor.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["total_entries"] == 2
        assert "hit_rate" in stats
        assert stats["hit_rate"] == pytest.approx(1/3)

        interceptor.reset_stats()
        stats = interceptor.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_callbacks(self) -> None:
        """Test that callbacks are called on cache hits and misses."""
        hit_calls: list[tuple] = []
        miss_calls: list[tuple] = []

        def on_hit(name: str, args: dict, output) -> None:
            hit_calls.append((name, args, output))

        def on_miss(name: str, args: dict) -> None:
            miss_calls.append((name, args))

        config = CacheConfig(
            similarity_threshold=0.5,
            reuse_threshold=0.95,
        )
        cache = create_test_cache()
        interceptor = ToolInterceptor(
            cache=cache,
            config=config,
            on_cache_hit=on_hit,
            on_cache_miss=on_miss,
        )

        def my_tool(query: str) -> str:
            return f"Result: {query}"

        wrapped = interceptor.wrap_tool(my_tool)

        wrapped(query="test")
        assert len(miss_calls) == 1
        assert miss_calls[0][0] == "my_tool"

        wrapped(query="test")
        assert len(hit_calls) == 1
        assert hit_calls[0][0] == "my_tool"

    def test_cache_hit_increments_reuse_count(self, interceptor: ToolInterceptor) -> None:
        """Test that cache hits increment reuse_count correctly."""
        def my_tool(query: str) -> str:
            return f"Result: {query}"

        wrapped = interceptor.wrap_tool(my_tool)

        wrapped(query="test")

        entries = list(interceptor.cache.storage.keys())
        entry_id = entries[0]
        data = interceptor.cache.storage.get(entry_id)
        assert data is not None
        assert data["reuse_count"] == 0
        assert data["provide_context_count"] == 0

        for i in range(1, 6):
            wrapped(query="test")
            data = interceptor.cache.storage.get(entry_id)
            assert data is not None
            assert data["reuse_count"] == i
            assert data["provide_context_count"] == 0

    def test_different_queries_different_entries(self, interceptor: ToolInterceptor) -> None:
        """Test that different queries create different cache entries."""
        def my_tool(query: str) -> str:
            return f"Result: {query}"

        wrapped = interceptor.wrap_tool(my_tool)

        wrapped(query="first")
        wrapped(query="second")
        wrapped(query="third")

        stats = interceptor.stats
        assert stats["misses"] == 3
        assert stats["hits"] == 0
        assert interceptor.cache.storage.size() == 3

    def test_wrapped_tool_with_args(self, interceptor: ToolInterceptor) -> None:
        """Test wrapped tool with positional and keyword arguments."""
        def my_tool(a: int, b: int, limit: int = 10) -> str:
            return f"{a+b}:{limit}"

        wrapped = interceptor.wrap_tool(my_tool)

        result1 = wrapped(1, 2, limit=5)
        assert result1 == "3:5"

        result2 = wrapped(1, 2, limit=5)
        assert result2 == "3:5"

        stats = interceptor.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_score_updates_on_hit(self, interceptor: ToolInterceptor) -> None:
        """Test that entry score is updated on cache hit."""
        def my_tool(query: str) -> str:
            return query

        wrapped = interceptor.wrap_tool(my_tool)

        wrapped(query="test")

        entries = list(interceptor.cache.storage.keys())
        entry_id = entries[0]
        initial_score = interceptor.cache.storage.get_score(entry_id)

        for _ in range(3):
            wrapped(query="test")

        final_score = interceptor.cache.storage.get_score(entry_id)

        assert final_score is not None
        assert initial_score is not None
        assert final_score > initial_score
