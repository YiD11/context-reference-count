"""Tests for cache configuration."""

import pytest
from pydantic import ValidationError

from context_ref.core.config import CacheConfig


class TestCacheConfig:
    """Test cases for CacheConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CacheConfig()
        assert config.similarity_threshold == 0.75
        assert config.reuse_threshold == 0.95
        assert config.max_cache_size == 1000
        assert config.embedding_dimension == 384
        assert config.reuse_context_factor == 0.6
        assert config.time_decay_lambda == 0.01
        assert config.top_k == 5
        assert config.eviction_policy == "score"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = CacheConfig(
            similarity_threshold=0.8,
            reuse_threshold=0.9,
            max_cache_size=500,
            reuse_context_factor=0.7,
            time_decay_lambda=0.02,
            eviction_policy="lru",
        )
        assert config.similarity_threshold == 0.8
        assert config.reuse_threshold == 0.9
        assert config.max_cache_size == 500
        assert config.reuse_context_factor == 0.7
        assert config.time_decay_lambda == 0.02
        assert config.eviction_policy == "lru"

    def test_threshold_validations(self) -> None:
        """Test threshold range and order validations."""
        # similarity_threshold out of range
        with pytest.raises(ValidationError):
            CacheConfig(similarity_threshold=1.5)
        with pytest.raises(ValidationError):
            CacheConfig(similarity_threshold=-0.1)

        # reuse_threshold out of range
        with pytest.raises(ValidationError):
            CacheConfig(reuse_threshold=1.5)
        with pytest.raises(ValidationError):
            CacheConfig(reuse_threshold=-0.1)

        # threshold order violation
        with pytest.raises(ValueError, match="similarity_threshold .* must be <="):
            CacheConfig(similarity_threshold=0.95, reuse_threshold=0.8)

    def test_parameter_validations(self) -> None:
        """Test validations for other parameters."""
        # reuse_context_factor out of range
        with pytest.raises(ValidationError):
            CacheConfig(reuse_context_factor=1.5)
        with pytest.raises(ValidationError):
            CacheConfig(reuse_context_factor=-0.1)

        # max_cache_size must be positive
        with pytest.raises(ValidationError):
            CacheConfig(max_cache_size=0)
        with pytest.raises(ValidationError):
            CacheConfig(max_cache_size=-10)

        # time_decay_lambda must be non-negative
        with pytest.raises(ValidationError):
            CacheConfig(time_decay_lambda=-0.01)

    def test_boundary_values(self) -> None:
        """Test valid boundary values."""
        config = CacheConfig(
            similarity_threshold=0.0,
            reuse_threshold=1.0,
            reuse_context_factor=0.0,
            time_decay_lambda=0.0,
            max_cache_size=1,
        )
        assert config.similarity_threshold == 0.0
        assert config.reuse_threshold == 1.0
        assert config.reuse_context_factor == 0.0
        assert config.time_decay_lambda == 0.0
        assert config.max_cache_size == 1
