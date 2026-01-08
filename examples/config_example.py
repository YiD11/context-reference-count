from context_ref.core import (
    CacheConfig,
    ChromaConfig,
    RedisConfig,
    StorageBackendConfig,
    ToolCache,
    VectorStoreConfig,
)


def example_memory_cache():
    config = CacheConfig(
        similarity_threshold=0.8,
        storage=StorageBackendConfig(backend_type="memory"),
        vector_store=VectorStoreConfig(
            store_type="chroma",
            chroma=ChromaConfig(mode="ephemeral"),
            collection_name="my_cache",
        ),
    )

    cache = ToolCache(config=config)

    cache.save("search", {"query": "python"}, {"results": ["result1", "result2"]})
    hit = cache.get_best_match("search", {"query": "python tutorial"})
    print(hit)
    cache.close()


def example_redis_cache():
    """使用 Redis 存储和客户端模式 ChromaDB 的缓存。"""
    config = CacheConfig(
        similarity_threshold=0.8,
        storage=StorageBackendConfig(
            backend_type="redis",
            redis=RedisConfig(host="localhost", port=6379, db=0, password=None),
            prefix="my_app:",
        ),
        vector_store=VectorStoreConfig(
            store_type="chroma",
            chroma=ChromaConfig(mode="client", host="localhost", port=8000),
            collection_name="my_cache",
        ),
    )

    cache = ToolCache(config=config)

    cache.save("search", {"query": "python"}, {"results": ["result1", "result2"]})
    hit = cache.get_best_match("search", {"query": "python tutorial"})
    print(hit)
    cache.close()


def example_redis_url_cache():
    """使用 Redis URL 连接的缓存。"""
    config = CacheConfig(
        similarity_threshold=0.8,
        storage=StorageBackendConfig(
            backend_type="redis", redis=RedisConfig(url="redis://localhost:6379/0")
        ),
        vector_store=VectorStoreConfig(
            store_type="chroma", chroma=ChromaConfig(mode="ephemeral")
        ),
    )

    cache = ToolCache(config=config)
    cache.close()


def example_env_based_cache():
    """从环境变量读取配置（向后兼容）。

    需要设置环境变量:
    - REDIS_HOST=localhost
    - REDIS_PORT=6379
    - CHROMADB_MODE=client
    - CHROMADB_HOST=localhost
    - CHROMADB_PORT=8000
    """
    config = CacheConfig(
        similarity_threshold=0.8,
        storage=StorageBackendConfig(
            backend_type="redis",
        ),
        vector_store=VectorStoreConfig(
            store_type="chroma",
        ),
    )

    cache = ToolCache(config=config)
    cache.close()


def example_default_cache():
    cache = ToolCache()
    config = CacheConfig(similarity_threshold=0.85)
    cache = ToolCache(config=config)

    cache.close()


def example_persistent_chroma():
    config = CacheConfig(
        storage=StorageBackendConfig(backend_type="memory"),
        vector_store=VectorStoreConfig(
            store_type="chroma",
            chroma=ChromaConfig(mode="persistent", path="./chroma_data"),
            collection_name="persistent_cache",
        ),
    )

    cache = ToolCache(config=config)
    cache.close()
