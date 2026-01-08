"""Tests for cache entry models."""

from datetime import datetime

from context_ref.core.models import CacheEntry, CacheHit


class TestCacheEntry:
    """Test cases for CacheEntry."""

    def test_create_entry(self) -> None:
        entry = CacheEntry(
            id="test123",
            tool_name="search",
            input_text='{"query": "test"}',
            input_args={"query": "test"},
            output="result",
        )
        assert entry.id == "test123"
        assert entry.tool_name == "search"
        assert entry.reuse_count == 0
        assert entry.provide_context_count == 0
        assert entry.total_reference_count == 0
        assert entry.success is True
        assert entry.uuid is not None

    def test_increment_counts(self) -> None:
        """Test incrementing reuse and context counts."""
        entry = CacheEntry(
            id="test123",
            tool_name="search",
            input_text='{"query": "test"}',
            input_args={"query": "test"},
            output="result",
        )
        old_time = entry.last_accessed_at

        # Test increment_reuse
        entry.increment_reuse()
        assert entry.reuse_count == 1
        assert entry.provide_context_count == 0
        assert entry.total_reference_count == 1
        assert entry.last_accessed_at >= old_time

        # Test increment_context
        entry.increment_context()
        assert entry.reuse_count == 1
        assert entry.provide_context_count == 1
        assert entry.total_reference_count == 2

        # Test multiple increments
        entry.increment_reuse()
        entry.increment_context()
        assert entry.reuse_count == 2
        assert entry.provide_context_count == 2
        assert entry.total_reference_count == 4

    def test_serialization_roundtrip(self) -> None:
        """Test to_dict/from_dict and to_json/from_json roundtrip."""
        entry = CacheEntry(
            id="test123",
            tool_name="search",
            input_text='{"query": "test"}',
            input_args={"query": "test"},
            output={"results": ["a", "b", "c"]},
            reuse_count=5,
            provide_context_count=3,
        )

        # Test dict roundtrip
        data = entry.to_dict()
        assert data["id"] == "test123"
        assert data["reuse_count"] == 5
        assert data["provide_context_count"] == 3
        assert "uuid" in data
        assert "created_at" in data

        restored_from_dict = CacheEntry.from_dict(data)
        assert restored_from_dict.id == entry.id
        assert restored_from_dict.reuse_count == entry.reuse_count
        assert restored_from_dict.provide_context_count == entry.provide_context_count

        # Test JSON roundtrip
        json_str = entry.to_json()
        restored_from_json = CacheEntry.from_json(json_str)
        assert restored_from_json.id == entry.id
        assert restored_from_json.output == entry.output
        assert restored_from_json.reuse_count == entry.reuse_count

    def test_from_dict_backward_compatibility(self) -> None:
        """Test backward compatibility with old data format missing counts."""
        data = {
            "id": "test123",
            "tool_name": "search",
            "input_text": '{"query": "test"}',
            "input_args": {"query": "test"},
            "output": "result",
        }
        entry = CacheEntry.from_dict(data)
        assert entry.reuse_count == 0
        assert entry.provide_context_count == 0


class TestCacheHit:
    """Test cases for CacheHit."""

    def test_should_reuse_decision(self) -> None:
        """Test should_reuse decision based on similarity."""
        entry = CacheEntry(
            id="test123",
            tool_name="search",
            input_text='{"query": "test"}',
            input_args={"query": "test"},
            output="result",
        )

        # High similarity should reuse
        hit_high = CacheHit(entry=entry, similarity=0.98, weighted_score=0.9)
        assert hit_high.should_reuse is True

        # Low similarity should not reuse
        hit_low = CacheHit(entry=entry, similarity=0.85, weighted_score=0.8)
        assert hit_low.should_reuse is False

    def test_cache_hit_to_dict(self) -> None:
        """Test CacheHit serialization."""
        entry = CacheEntry(
            id="test123",
            tool_name="search",
            input_text='{"query": "test"}',
            input_args={"query": "test"},
            output="result",
            reuse_count=3,
            provide_context_count=2,
        )
        hit = CacheHit(entry=entry, similarity=0.95, weighted_score=0.88)
        data = hit.to_dict()

        assert data["similarity"] == 0.95
        assert data["weighted_score"] == 0.88
        assert data["entry"]["reuse_count"] == 3
        assert data["entry"]["provide_context_count"] == 2
