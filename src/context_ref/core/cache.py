"""Tool cache implementation with semantic similarity and reference counting."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TYPE_CHECKING

from context_ref.core.config import CacheConfig
from context_ref.core.models import CacheEntry, CacheHit
from context_ref.core.storage.memory import MemoryStorageBackend
from context_ref.core.storage.vector import VectorStore
from context_ref.core.utils import (
    compute_weighted_score,
    generate_cache_id,
    serialize_args,
)
from context_ref.embedding.base import EmbeddingFunction

if TYPE_CHECKING:
    from context_ref.core.storage.redis import RedisStorageBackend

    StorageType = MemoryStorageBackend | RedisStorageBackend


class ToolCache:
    """Semantic cache for tool calls with reference counting.

    Combines vector similarity search with reference counting to optimize
    tool call caching and retrieval decisions.

    The cache uses two backends:
    - VectorStore: For similarity search using embeddings
    - StorageBackend: For storing and retrieving full entry data with scores

    Score Formula:
        score = similarity + normalized_ref * recency_factor

        where:
        - normalized_ref = log(weighted_count + 1) / log(100)
        - weighted_count = factor * reuse_count + (1 - factor) * provide_context_count
        - recency_factor = exp(-lambda * delta_t_hours)
        - factor = config.reuse_context_factor

    Example:
        # Default in-memory cache
        cache = ToolCache()

        # With custom config
        from context_ref.core.config import CacheConfig, StorageBackendConfig, VectorStoreConfig

        config = CacheConfig(
            similarity_threshold=0.8,
            storage=StorageBackendConfig(backend_type="redis"),
            vector_store=VectorStoreConfig(store_type="chroma")
        )
        cache = ToolCache(config=config)

        # Legacy: With explicit storage and vector_store instances
        from context_ref.core.storage.redis import RedisStorageBackend
        from context_ref.core.storage.chroma import ChromaVectorStore

        storage = RedisStorageBackend.from_env()
        vector_store = ChromaVectorStore(collection_name="my_cache")
        cache = ToolCache(
            storage=storage,
            vector_store=vector_store,
            config=CacheConfig(similarity_threshold=0.8)
        )

        # Save and retrieve
        cache.save("search", {"query": "python"}, {"results": [...]})
        hit = cache.get_best_match("search", {"query": "python tutorial"})
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
        embedding_func: EmbeddingFunction | None = None,
        storage: StorageType | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.config = config or CacheConfig()
        self._embedding_func = embedding_func

        # Use provided storage/vector_store instances or create from config
        if storage is not None:
            self._storage = storage
        else:
            from context_ref.core.storage.config import create_storage_backend

            # config.storage is guaranteed to exist due to CacheConfig validation
            assert self.config.storage is not None
            self._storage = create_storage_backend(self.config.storage)

        if vector_store is not None:
            self._vector_store = vector_store
        else:
            self._vector_store = self._create_vector_store_from_config()

    def _create_vector_store_from_config(self) -> VectorStore | None:
        """Create vector store from config (lazy initialization supported)."""
        if self.config.vector_store:
            from context_ref.core.storage.config import create_vector_store

            return create_vector_store(self.config.vector_store)
        return None

    @property
    def embedding_func(self) -> EmbeddingFunction:
        if self._embedding_func is None:
            from context_ref.embedding.default import DefaultEmbedding

            self._embedding_func = DefaultEmbedding()
        return self._embedding_func

    @property
    def storage(self) -> Any:
        return self._storage

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            from context_ref.core.storage.chroma import ChromaVectorStore

            # Use collection_name from config
            collection_name = self.config.collection_name
            if self.config.vector_store:
                collection_name = self.config.vector_store.collection_name

            self._vector_store = ChromaVectorStore.from_env()
            self._vector_store._collection_name = collection_name
        return self._vector_store

    def _compute_entry_score(self, entry: CacheEntry, similarity: float = 1.0) -> float:
        """Compute score for an entry with given similarity."""
        return compute_weighted_score(
            similarity=similarity,
            reuse_count=entry.reuse_count,
            provide_context_count=entry.provide_context_count,
            last_accessed=entry.last_accessed_at,
            reuse_context_factor=self.config.reuse_context_factor,
            time_decay_lambda=self.config.time_decay_lambda,
        )

    def search(
        self,
        tool_name: str,
        input_args: dict[str, Any],
        top_k: int | None = None,
    ) -> list[CacheHit]:
        """Search for similar cached tool calls.

        Returns a list of cache hits sorted by weighted score.
        """
        input_text = serialize_args(input_args)
        embedding = self.embedding_func.embed(input_text)
        k = top_k or self.config.top_k

        results = self.vector_store.search(
            query_embedding=embedding,
            k=k,
            filter={"tool_name": tool_name},
        )

        if not results.get("ids") or not results["ids"][0]:
            return []

        hits: list[CacheHit] = []
        for idx, entry_id in enumerate(results["ids"][0]):
            entry_data = self._storage.get(entry_id)
            if entry_data is None:
                try:
                    if not self._storage.exists(entry_id):
                        self.vector_store.delete(ids=[entry_id])
                except Exception:
                    pass
                continue

            entry = CacheEntry.from_dict(entry_data)
            distance = results["distances"][0][idx] if results["distances"] else 0.0
            similarity = 1.0 - distance

            if similarity < self.config.similarity_threshold:
                continue

            weighted_score = compute_weighted_score(
                similarity=similarity,
                reuse_count=entry.reuse_count,
                provide_context_count=entry.provide_context_count,
                last_accessed=entry.last_accessed_at,
                reuse_context_factor=self.config.reuse_context_factor,
                time_decay_lambda=self.config.time_decay_lambda,
            )
            hits.append(
                CacheHit(
                    entry=entry, similarity=similarity, weighted_score=weighted_score
                )
            )

        hits.sort(key=lambda h: h.weighted_score, reverse=True)
        return hits

    def _touch_entry(self, entry_id: str) -> None:
        """Refresh access time and derived score for an entry."""
        updated = self._storage.update_access_time(entry_id)
        if not updated:
            return

        entry_data = self._storage.get(entry_id)
        if entry_data is None:
            return

        entry = CacheEntry.from_dict(entry_data)
        score = self._compute_entry_score(entry)
        self._storage.update_score(entry_id, score)

    def get_best_match(
        self,
        tool_name: str,
        input_args: dict[str, Any],
    ) -> CacheHit | None:
        """Get the best matching cache entry if similarity threshold is met."""
        hits = self.search(tool_name, input_args, top_k=1)
        if hits and hits[0].similarity >= self.config.similarity_threshold:
            self._touch_entry(hits[0].entry.id)
            return hits[0]
        return None

    def save(
        self,
        tool_name: str,
        input_args: dict[str, Any],
        output: Any,
        success: bool = True,
    ) -> CacheEntry:
        """Save a tool call result to cache."""
        input_text = serialize_args(input_args)
        entry_id = generate_cache_id(tool_name, input_text)

        existing = self._storage.get(entry_id)
        if existing is not None:
            entry = CacheEntry.from_dict(existing)
            entry.output = output
            entry.success = success
            entry.last_accessed_at = datetime.now()

            score = self._compute_entry_score(entry)
            self._storage.set(entry_id, entry.to_dict(), score=score)
            return entry

        embedding = self.embedding_func.embed(input_text)
        entry = CacheEntry(
            id=entry_id,
            tool_name=tool_name,
            input_text=input_text,
            input_args=input_args,
            output=output,
            embedding=embedding,
            success=success,
        )

        score = self._compute_entry_score(entry)
        self.vector_store.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[input_text],
            metadata=[{"tool_name": tool_name}],
        )
        try:
            self._storage.set(entry_id, entry.to_dict(), score=score)
        except Exception:
            try:
                self.vector_store.delete(ids=[entry_id])
            except Exception:
                pass
            raise

        self._maybe_evict()
        return entry

    def increment_reuse(self, entry_id: str) -> bool:
        """Increment the reuse count for a cache entry.

        Called when an entry is directly reused (high similarity match).

        Args:
            entry_id: The unique identifier for the entry.

        Returns:
            True if increment succeeded, False if entry doesn't exist.
        """
        success = self._storage.increment_reuse(entry_id)
        if success:
            entry_data = self._storage.get(entry_id)
            if entry_data:
                entry = CacheEntry.from_dict(entry_data)
                score = self._compute_entry_score(entry)
                self._storage.update_score(entry_id, score)
        return success

    def increment_context(self, entry_id: str) -> bool:
        """Increment the provide_context count for a cache entry.

        Called when an entry is provided as context hint (medium similarity match).

        Args:
            entry_id: The unique identifier for the entry.

        Returns:
            True if increment succeeded, False if entry doesn't exist.
        """
        success = self._storage.increment_context(entry_id)
        if success:
            entry_data = self._storage.get(entry_id)
            if entry_data:
                entry = CacheEntry.from_dict(entry_data)
                score = self._compute_entry_score(entry)
                self._storage.update_score(entry_id, score)
        return success

    def increment_reference(self, entry_id: str) -> bool:
        """Increment the reference count for a cache entry (backward compatibility).

        This is an alias for increment_reuse.

        Args:
            entry_id: The unique identifier for the entry.

        Returns:
            True if increment succeeded, False if entry doesn't exist.
        """
        return self.increment_reuse(entry_id)

    def _maybe_evict(self) -> None:
        """Evict entries if cache size exceeds limit."""
        if self._storage.size() <= self.config.max_cache_size:
            return

        num_to_evict = self._storage.size() - self.config.max_cache_size

        if self.config.eviction_policy == "score":
            evict_keys = self._storage.get_bottom_by_score(num_to_evict)
        elif self.config.eviction_policy == "lru":
            evict_keys = self._storage.get_oldest_by_access(num_to_evict)
        elif self.config.eviction_policy == "lfu":
            evict_keys = self._storage.get_least_used(num_to_evict)
        else:
            evict_keys = self._storage.get_oldest_by_creation(num_to_evict)

        for entry_id in evict_keys:
            self._evict_entry(entry_id)

    def _evict_entry(self, entry_id: str) -> None:
        if self._storage.exists(entry_id):
            self._storage.delete(entry_id)
            self.vector_store.delete(ids=[entry_id])

    def clear(self) -> None:
        """Clear all cache entries."""
        self._storage.clear()
        self.vector_store.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        entries = list(self._storage.keys())
        if not entries:
            return {
                "total_entries": 0,
                "total_reuse_count": 0,
                "total_context_count": 0,
                "total_references": 0,
                "avg_reference_count": 0.0,
            }

        total_reuse = 0
        total_context = 0
        max_refs = 0
        for key in entries:
            data = self._storage.get(key)
            if data:
                reuse = data.get("reuse_count", 0)
                context = data.get("provide_context_count", 0)
                total_reuse += reuse
                total_context += context
                max_refs = max(max_refs, reuse + context)

        total_refs = total_reuse + total_context
        return {
            "total_entries": len(entries),
            "total_reuse_count": total_reuse,
            "total_context_count": total_context,
            "total_references": total_refs,
            "avg_reference_count": total_refs / len(entries),
            "max_reference_count": max_refs,
        }

    def close(self) -> None:
        """Clean up resources."""
        self._storage.close()
        self.vector_store.close()

    def __enter__(self) -> "ToolCache":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
        return None
