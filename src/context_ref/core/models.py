"""Data models for cache entries and results."""

import json
import uuid as uuid_lib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _generate_uuid() -> str:
    """Generate a new UUID for cache entries."""
    return str(uuid_lib.uuid4())


@dataclass
class CacheEntry:
    """Represents a cached tool call entry.

    Attributes:
        id: Unique identifier (hash-based, for deduplication)
        uuid: UUID for Redis ZSET storage
        tool_name: Name of the tool that was called
        input_text: Serialized input arguments
        input_args: Original input arguments dict
        output: Tool call result
        embedding: Vector embedding of input_text
        reuse_count: Times this entry was directly reused (high similarity)
        provide_context_count: Times this entry was provided as context hint
        created_at: When the entry was first created
        last_accessed_at: When the entry was last accessed
        success: Whether the tool call succeeded
    """

    id: str
    tool_name: str
    input_text: str
    input_args: dict[str, Any]
    output: Any
    uuid: str = field(default_factory=_generate_uuid)
    embedding: list[float] | None = None
    reuse_count: int = 0
    provide_context_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    success: bool = True

    @property
    def total_reference_count(self) -> int:
        """Total number of times this entry was referenced."""
        return self.reuse_count + self.provide_context_count

    def increment_reuse(self) -> None:
        """Increment reuse count and update access time."""
        self.reuse_count += 1
        self.last_accessed_at = datetime.now()

    def increment_context(self) -> None:
        """Increment provide_context count and update access time."""
        self.provide_context_count += 1
        self.last_accessed_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Serialize entry to dict for JSON storage."""
        return {
            "id": self.id,
            "uuid": self.uuid,
            "tool_name": self.tool_name,
            "input_text": self.input_text,
            "input_args": self.input_args,
            "output": self.output,
            "embedding": self.embedding,
            "reuse_count": self.reuse_count,
            "provide_context_count": self.provide_context_count,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "success": self.success,
        }

    def to_json(self) -> str:
        """Serialize entry to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Deserialize entry from dict."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        last_accessed_at = data.get("last_accessed_at")
        if isinstance(last_accessed_at, str):
            last_accessed_at = datetime.fromisoformat(last_accessed_at)
        elif last_accessed_at is None:
            last_accessed_at = datetime.now()

        return cls(
            id=data["id"],
            uuid=data.get("uuid", _generate_uuid()),
            tool_name=data["tool_name"],
            input_text=data["input_text"],
            input_args=data["input_args"],
            output=data["output"],
            embedding=data.get("embedding"),
            reuse_count=data.get("reuse_count", 0),
            provide_context_count=data.get("provide_context_count", 0),
            created_at=created_at,
            last_accessed_at=last_accessed_at,
            success=data.get("success", True),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CacheEntry":
        """Deserialize entry from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class CacheHit:
    """Represents a cache hit with similarity score and weighted score."""

    entry: CacheEntry
    similarity: float
    weighted_score: float

    @property
    def should_reuse(self) -> bool:
        """Whether this hit should be directly reused (default threshold 0.95)."""
        return self.similarity >= 0.95

    def to_dict(self) -> dict[str, Any]:
        """Serialize cache hit to dict."""
        return {
            "entry": self.entry.to_dict(),
            "similarity": self.similarity,
            "weighted_score": self.weighted_score,
        }
