"""Redis storage backend implementation.

ZSET-based storage for production caching with score management.
"""

import json
import redis
from datetime import datetime
from typing import Any, Iterator

from context_ref.core.config import RedisConfig, get_redis_config


class RedisStorageBackend:
    """Redis storage backend with ZSET-based score management.

    Uses Redis data structures:
    - String: Store cache entry JSON (key: {prefix}entry:{id})
    - ZSET: Store entry scores for ranking (key: {prefix}scores)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        url: str | None = None,
        prefix: str = "context_ref:",
    ) -> None:
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._url = url
        self._prefix = prefix
        self._client: redis.Redis | None = None

    @classmethod
    def from_env(cls, config: RedisConfig | None = None) -> "RedisStorageBackend":
        """Create backend from environment configuration."""
        if config is None:
            config = get_redis_config()
        if config is None:
            raise ValueError(
                "Redis not configured. Set REDIS_URL or REDIS_HOST environment variable."
            )
        if config.is_url_based():
            return cls(url=config.url)
        return cls(host=config.host or "localhost", port=config.port, db=config.db)

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            if self._url:
                self._client = redis.from_url(self._url, decode_responses=True)
            else:
                self._client = redis.Redis(
                    host=self._host,
                    port=self._port,
                    db=self._db,
                    password=self._password,
                    decode_responses=True,
                )
        return self._client

    def _entry_key(self, key: str) -> str:
        return f"{self._prefix}entry:{key}"

    def _scores_key(self) -> str:
        return f"{self._prefix}scores"

    def _serialize(self, value: dict[str, Any]) -> str:
        data = value.copy()
        for k, v in data.items():
            if isinstance(v, datetime):
                data[k] = v.isoformat()
        return json.dumps(data, default=str)

    def _deserialize(self, data: str | None) -> dict[str, Any] | None:
        if data is None:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    def get(self, key: str) -> dict[str, Any] | None:
        client = self._get_client()
        data = client.get(self._entry_key(key))
        return self._deserialize(data)

    def set(self, key: str, value: dict[str, Any], score: float = 0.0) -> None:
        client = self._get_client()
        pipe = client.pipeline()
        pipe.set(self._entry_key(key), self._serialize(value))
        pipe.zadd(self._scores_key(), {key: score})
        pipe.execute()

    def delete(self, key: str) -> bool:
        client = self._get_client()
        pipe = client.pipeline()
        pipe.delete(self._entry_key(key))
        pipe.zrem(self._scores_key(), key)
        results = pipe.execute()
        return results[0] > 0

    def exists(self, key: str) -> bool:
        client = self._get_client()
        return client.exists(self._entry_key(key)) > 0

    def keys(self) -> Iterator[str]:
        client = self._get_client()
        pattern = f"{self._prefix}entry:*"
        prefix_len = len(f"{self._prefix}entry:")
        for key in client.scan_iter(match=pattern):
            yield key[prefix_len:]

    def clear(self) -> None:
        client = self._get_client()
        for key in client.scan_iter(match=f"{self._prefix}entry:*"):
            client.delete(key)
        client.delete(self._scores_key())

    def size(self) -> int:
        client = self._get_client()
        return client.zcard(self._scores_key())

    def update_score(self, key: str, score: float) -> bool:
        client = self._get_client()
        if not client.exists(self._entry_key(key)):
            return False
        client.zadd(self._scores_key(), {key: score})
        return True

    def get_score(self, key: str) -> float | None:
        client = self._get_client()
        return client.zscore(self._scores_key(), key)

    def update_access_time(self, key: str) -> bool:
        data = self.get(key)
        if data is None:
            return False
        data["last_accessed_at"] = datetime.now().isoformat()
        client = self._get_client()
        client.set(self._entry_key(key), self._serialize(data))
        return True

    def increment_reuse(self, key: str) -> bool:
        data = self.get(key)
        if data is None:
            return False
        data["reuse_count"] = data.get("reuse_count", 0) + 1
        data["last_accessed_at"] = datetime.now().isoformat()
        client = self._get_client()
        client.set(self._entry_key(key), self._serialize(data))
        return True

    def increment_context(self, key: str) -> bool:
        data = self.get(key)
        if data is None:
            return False
        data["provide_context_count"] = data.get("provide_context_count", 0) + 1
        data["last_accessed_at"] = datetime.now().isoformat()
        client = self._get_client()
        client.set(self._entry_key(key), self._serialize(data))
        return True

    def decrement_reference(self, key: str) -> bool:
        data = self.get(key)
        if data is None:
            return False
        reuse = data.get("reuse_count", 0)
        context = data.get("provide_context_count", 0)
        if reuse > 0:
            data["reuse_count"] = reuse - 1
        elif context > 0:
            data["provide_context_count"] = context - 1
        client = self._get_client()
        client.set(self._entry_key(key), self._serialize(data))
        return True

    def get_bottom_by_score(self, n: int = 1) -> list[str]:
        """Get keys with lowest scores (for eviction)."""
        client = self._get_client()
        return list(client.zrange(self._scores_key(), 0, n - 1))

    def get_oldest_by_access(self, n: int = 1) -> list[str]:
        """Get keys by oldest access time (LRU eviction)."""
        entries = []
        for key in self.keys():
            data = self.get(key)
            if data:
                last = data.get("last_accessed_at")
                if isinstance(last, str):
                    last = datetime.fromisoformat(last)
                elif last is None:
                    last = datetime.min
                entries.append((key, last))
        entries.sort(key=lambda x: x[1])
        return [k for k, _ in entries[:n]]

    def get_least_used(self, n: int = 1) -> list[str]:
        """Get keys by lowest reference count (LFU eviction)."""
        entries = []
        for key in self.keys():
            data = self.get(key)
            if data:
                total = data.get("reuse_count", 0) + data.get("provide_context_count", 0)
                entries.append((key, total))
        entries.sort(key=lambda x: x[1])
        return [k for k, _ in entries[:n]]

    def get_oldest_by_creation(self, n: int = 1) -> list[str]:
        """Get keys by creation time (FIFO eviction)."""
        entries = []
        for key in self.keys():
            data = self.get(key)
            if data:
                created = data.get("created_at")
                if isinstance(created, str):
                    created = datetime.fromisoformat(created)
                elif created is None:
                    created = datetime.min
                entries.append((key, created))
        entries.sort(key=lambda x: x[1])
        return [k for k, _ in entries[:n]]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
