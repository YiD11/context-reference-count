"""Factory functions for creating storage and vector store instances from config."""

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from context_ref.core.config import StorageBackendConfig, VectorStoreConfig
    from context_ref.core.storage.memory import MemoryStorageBackend
    from context_ref.core.storage.redis import RedisStorageBackend
    from context_ref.core.storage.vector import VectorStore

    StorageType = Union[MemoryStorageBackend, RedisStorageBackend]


def create_storage_backend(config: "StorageBackendConfig") -> "StorageType":
    """Create storage backend instance from config.

    Args:
        config: Storage backend configuration

    Returns:
        Storage backend instance (MemoryStorageBackend or RedisStorageBackend)

    Raises:
        ValueError: If redis backend is selected but redis config is missing
    """
    if config.backend_type == "redis":
        from context_ref.core.storage.redis import RedisStorageBackend

        redis_config = config.redis
        if redis_config is None:
            raise ValueError("Redis config required for redis backend")

        if redis_config.is_url_based():
            return RedisStorageBackend(url=redis_config.url, prefix=config.prefix)
        else:
            return RedisStorageBackend(
                host=redis_config.host or "localhost",
                port=redis_config.port,
                db=redis_config.db,
                password=redis_config.password,
                prefix=config.prefix,
            )
    else:
        from context_ref.core.storage.memory import MemoryStorageBackend

        return MemoryStorageBackend()


def create_vector_store(config: "VectorStoreConfig") -> "VectorStore":
    """Create vector store instance from config.

    Args:
        config: Vector store configuration

    Returns:
        Vector store instance (ChromaVectorStore or in-memory fallback)
    """
    if config.store_type == "chroma":
        from context_ref.core.storage.chroma import ChromaVectorStore

        chroma_config = config.chroma
        if chroma_config is None:
            return ChromaVectorStore(collection_name=config.collection_name)

        if chroma_config.is_client_mode():
            return ChromaVectorStore(
                collection_name=config.collection_name,
                host=chroma_config.host or "localhost",
                port=chroma_config.port or 8000,
                mode="client",
            )
        elif chroma_config.is_persistent_mode():
            return ChromaVectorStore(
                collection_name=config.collection_name,
                path=chroma_config.path,
                mode="persistent",
            )
        else:
            return ChromaVectorStore(
                collection_name=config.collection_name, mode="ephemeral"
            )
    else:
        # Memory vector store - use ChromaVectorStore in ephemeral mode as fallback
        from context_ref.core.storage.chroma import ChromaVectorStore

        return ChromaVectorStore(
            collection_name=config.collection_name, mode="ephemeral"
        )
