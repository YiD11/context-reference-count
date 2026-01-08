"""Configuration for the cache system with pydantic-based settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheConfig(BaseModel):
    """Configuration for the tool cache system.

    Attributes:
        similarity_threshold: Minimum similarity for cache hits (0-1)
        reuse_threshold: Minimum similarity for direct reuse (0-1)
        max_cache_size: Maximum number of entries in cache
        embedding_dimension: Dimension of embedding vectors
        reuse_context_factor: Balance factor between reuse and context counts (0-1)
            - Higher values weight reuse_count more heavily
            - Lower values weight provide_context_count more heavily
        time_decay_lambda: Decay rate for recency factor (per hour)
        top_k: Number of candidates to retrieve for similarity search
        eviction_policy: Cache eviction strategy
        persist_path: Path for persistent storage (optional)
        collection_name: Name of the vector store collection (deprecated, use vector_store.collection_name)
        redis_score_key: Key prefix for Redis ZSET scores (deprecated, use storage.prefix)
        storage: Storage backend configuration
        vector_store: Vector store configuration
    """

    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for cache hits (0-1)",
    )
    reuse_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for direct reuse (0-1)",
    )
    max_cache_size: int = Field(
        default=1000, gt=0, description="Maximum number of entries in cache"
    )
    embedding_dimension: int = Field(
        default=384, gt=0, description="Dimension of embedding vectors"
    )
    reuse_context_factor: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Balance factor between reuse and context counts (0-1)",
    )
    time_decay_lambda: float = Field(
        default=0.01, ge=0.0, description="Decay rate for recency factor (per hour)"
    )
    top_k: int = Field(
        default=5,
        gt=0,
        description="Number of candidates to retrieve for similarity search",
    )
    eviction_policy: Literal["lru", "lfu", "fifo", "score"] = Field(
        default="score", description="Cache eviction strategy"
    )
    persist_path: Optional[str] = Field(
        default=None, description="Path for persistent storage (optional)"
    )
    collection_name: str = Field(
        default="tool_cache",
        description="Name of the vector store collection (deprecated)",
    )
    redis_score_key: str = Field(
        default="cache_scores",
        description="Key prefix for Redis ZSET scores (deprecated)",
    )
    storage: Optional[StorageBackendConfig] = Field(
        default=None, description="Storage backend configuration"
    )
    vector_store: Optional[VectorStoreConfig] = Field(
        default=None, description="Vector store configuration"
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "CacheConfig":
        """Validate that similarity_threshold <= reuse_threshold."""
        if self.similarity_threshold > self.reuse_threshold:
            raise ValueError(
                f"similarity_threshold ({self.similarity_threshold}) must be <= "
                f"reuse_threshold ({self.reuse_threshold})"
            )
        # Initialize storage config if not provided
        if self.storage is None:
            self.storage = StorageBackendConfig()
        # Initialize vector store config if not provided
        if self.vector_store is None:
            self.vector_store = VectorStoreConfig(collection_name=self.collection_name)
        return self


class ChromaConfig(BaseSettings):
    """Configuration for ChromaDB connectivity."""

    model_config = SettingsConfigDict(
        env_prefix="CHROMADB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: str = Field(
        default="ephemeral",
        description="ChromaDB mode: 'client', 'persistent', or 'ephemeral'",
    )
    host: Optional[str] = Field(
        default=None, description="ChromaDB server host (for client mode)"
    )
    port: Optional[int] = Field(
        default=None, description="ChromaDB server port (for client mode)"
    )
    path: Optional[str] = Field(
        default=None, description="Persistent storage path (for persistent mode)"
    )
    auth_provider: Optional[str] = Field(
        default=None, description="ChromaDB authentication provider"
    )
    auth_credentials_provider: Optional[str] = Field(
        default=None, description="ChromaDB authentication credentials provider"
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate ChromaDB mode."""
        valid_modes = {"client", "persistent", "ephemeral"}
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {v}")
        return v

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "ChromaConfig":
        """Validate that required fields are set based on mode."""
        if self.mode == "client":
            if not self.host:
                self.host = "localhost"
            if not self.port:
                self.port = 8000
        return self

    def is_client_mode(self) -> bool:
        """Check if ChromaDB is configured in client mode."""
        return self.mode == "client"

    def is_persistent_mode(self) -> bool:
        """Check if ChromaDB is configured in persistent mode."""
        return self.mode == "persistent"

    def is_ephemeral_mode(self) -> bool:
        """Check if ChromaDB is configured in ephemeral mode."""
        return self.mode == "ephemeral"


class RedisConfig(BaseSettings):
    """Configuration for Redis connectivity."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: Optional[str] = Field(default=None, description="Redis connection URL")
    host: Optional[str] = Field(default=None, description="Redis server host")
    port: int = Field(default=6379, description="Redis server port")
    db: int = Field(default=0, ge=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")

    @model_validator(mode="after")
    def validate_connection_info(self) -> "RedisConfig":
        """Validate that either url or host is provided."""
        if not self.url and not self.host:
            # Redis is optional, return None-like config
            pass
        return self

    def is_configured(self) -> bool:
        """Check if Redis is configured."""
        return self.url is not None or self.host is not None

    def is_url_based(self) -> bool:
        """Check if Redis is configured using URL."""
        return self.url is not None

    def is_host_based(self) -> bool:
        """Check if Redis is configured using host."""
        return self.host is not None


class StorageBackendConfig(BaseModel):
    """Configuration for storage backend selection and settings.

    Attributes:
        backend_type: Type of storage backend ('memory' or 'redis')
        redis: Redis connection configuration (required if backend_type='redis')
        prefix: Key prefix for storage backend
    """

    backend_type: Literal["memory", "redis"] = Field(
        default="memory", description="Storage backend type: 'memory' or 'redis'"
    )
    redis: Optional[RedisConfig] = Field(
        default=None,
        description="Redis configuration (required if backend_type='redis')",
    )
    prefix: str = Field(
        default="context_ref:", description="Key prefix for storage backend"
    )

    @model_validator(mode="after")
    def validate_redis_required(self) -> "StorageBackendConfig":
        """Ensure Redis config is provided when backend_type is redis."""
        if self.backend_type == "redis" and self.redis is None:
            # Auto-create from environment
            self.redis = RedisConfig()
            if not self.redis.is_configured():
                raise ValueError(
                    "Redis configuration required when backend_type='redis'. "
                    "Set REDIS_URL or REDIS_HOST environment variable."
                )
        return self


class VectorStoreConfig(BaseModel):
    """Configuration for vector store selection and settings.

    Attributes:
        store_type: Type of vector store ('chroma' or 'memory')
        chroma: ChromaDB configuration (used if store_type='chroma')
        collection_name: Name of the vector store collection
    """

    store_type: Literal["chroma", "memory"] = Field(
        default="chroma", description="Vector store type: 'chroma' or 'memory'"
    )
    chroma: Optional["ChromaConfig"] = Field(
        default=None, description="ChromaDB configuration (used if store_type='chroma')"
    )
    collection_name: str = Field(
        default="tool_cache", description="Name of the vector store collection"
    )

    @model_validator(mode="after")
    def validate_chroma_required(self) -> "VectorStoreConfig":
        """Ensure Chroma config is provided when store_type is chroma."""
        if self.store_type == "chroma" and self.chroma is None:
            self.chroma = ChromaConfig()
        return self


def get_chroma_config() -> ChromaConfig:
    """Get ChromaDB configuration from environment variables."""
    return ChromaConfig()


def get_redis_config() -> Optional[RedisConfig]:
    """Get Redis configuration from environment variables, or None if not configured."""
    config = RedisConfig()
    if not config.is_configured():
        return None
    return config


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get environment variable as boolean.

    This is kept for backward compatibility with existing code.
    """
    import os

    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "yes", "1", "on")
