"""
Context Reference Count Plugin

A semantic caching plugin for LLM agents that reduces redundant tool calls
through similarity-based caching and reference counting.

Features:
- Split reference counting: tracks reuse_count and provide_context_count separately
- Score-based cache management with configurable formula
- Redis storage backend with ZSET for score-based ranking
- Pluggable storage and vector store backends
"""

from context_ref.core.cache import ToolCache
from context_ref.core.config import CacheConfig
from context_ref.core.models import CacheEntry, CacheHit
from context_ref.interceptor.wrapper import CacheDecision, DecisionResult, ToolInterceptor

__version__ = "0.1.0"
__all__ = [
    "ToolCache",
    "CacheConfig",
    "CacheEntry",
    "CacheHit",
    "CacheDecision",
    "DecisionResult",
    "ToolInterceptor",
]
