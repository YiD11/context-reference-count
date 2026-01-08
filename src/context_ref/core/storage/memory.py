"""In-memory storage backend implementation.

Simplified implementation for development and testing.
"""

from datetime import datetime
from typing import Any, Iterator


class MemoryStorageBackend:
    """In-memory storage backend using Python dict.

    Thread-safe implementation for local caching.
    For production, use RedisStorageBackend.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.RLock()
        self._data: dict[str, dict[str, Any]] = {}
        self._scores: dict[str, float] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._data.get(key)
            return entry.copy() if entry else None

    def set(self, key: str, value: dict[str, Any], score: float = 0.0) -> None:
        with self._lock:
            self._data[key] = value.copy()
            self._scores[key] = score

    def delete(self, key: str) -> bool:
        with self._lock:
            deleted = self._data.pop(key, None) is not None
            self._scores.pop(key, None)
            return deleted

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def keys(self) -> Iterator[str]:
        with self._lock:
            yield from list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._scores.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def update_score(self, key: str, score: float) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            self._scores[key] = score
            return True

    def get_score(self, key: str) -> float | None:
        with self._lock:
            return self._scores.get(key)

    def update_access_time(self, key: str) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            entry["last_accessed_at"] = datetime.now()
            return True

    def increment_reuse(self, key: str) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            entry["reuse_count"] = entry.get("reuse_count", 0) + 1
            entry["last_accessed_at"] = datetime.now()
            return True

    def increment_context(self, key: str) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            entry["provide_context_count"] = entry.get("provide_context_count", 0) + 1
            entry["last_accessed_at"] = datetime.now()
            return True

    def decrement_reference(self, key: str) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            reuse = entry.get("reuse_count", 0)
            context = entry.get("provide_context_count", 0)
            if reuse > 0:
                entry["reuse_count"] = reuse - 1
            elif context > 0:
                entry["provide_context_count"] = context - 1
            return True

    def get_bottom_by_score(self, n: int = 1) -> list[str]:
        """Get keys with lowest scores (for eviction)."""
        with self._lock:
            sorted_items = sorted(self._scores.items(), key=lambda x: x[1])
            return [k for k, _ in sorted_items[:n]]

    def get_oldest_by_access(self, n: int = 1) -> list[str]:
        """Get keys by oldest access time (LRU eviction)."""
        with self._lock:
            items = []
            for key, entry in self._data.items():
                last = entry.get("last_accessed_at", datetime.min)
                if isinstance(last, str):
                    last = datetime.fromisoformat(last)
                items.append((key, last))
            items.sort(key=lambda x: x[1])
            return [k for k, _ in items[:n]]

    def get_least_used(self, n: int = 1) -> list[str]:
        """Get keys by lowest reference count (LFU eviction)."""
        with self._lock:
            items = []
            for key, entry in self._data.items():
                total = entry.get("reuse_count", 0) + entry.get("provide_context_count", 0)
                items.append((key, total))
            items.sort(key=lambda x: x[1])
            return [k for k, _ in items[:n]]

    def get_oldest_by_creation(self, n: int = 1) -> list[str]:
        """Get keys by creation time (FIFO eviction)."""
        with self._lock:
            items = []
            for key, entry in self._data.items():
                created = entry.get("created_at", datetime.min)
                if isinstance(created, str):
                    created = datetime.fromisoformat(created)
                items.append((key, created))
            items.sort(key=lambda x: x[1])
            return [k for k, _ in items[:n]]

    def close(self) -> None:
        pass
