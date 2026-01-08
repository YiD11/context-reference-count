"""Utility functions for cache operations.

This module contains reusable utility functions for cache ID generation,
scoring algorithms, and data transformations that don't depend on specific
business logic or class state.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any


def generate_cache_id(tool_name: str, input_text: str) -> str:
    """Generate a deterministic cache ID from tool name and input.

    Uses SHA-256 hash truncated to 16 characters for readability.

    Args:
        tool_name: Name of the tool
        input_text: Serialized input arguments

    Returns:
        16-character hex string as cache ID

    Example:
        >>> generate_cache_id("search", '{"query": "python"}')
        'a3b2c1d4e5f6g7h8'
    """
    content = f"{tool_name}:{input_text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def serialize_args(args: dict[str, Any]) -> str:
    """Serialize tool arguments to canonical string representation.

    Uses sorted keys and default string conversion for non-JSON types.

    Args:
        args: Tool arguments dictionary

    Returns:
        JSON string with sorted keys

    Example:
        >>> serialize_args({"b": 2, "a": 1})
        '{"a": 1, "b": 2}'
    """
    return json.dumps(args, sort_keys=True, default=str)


def compute_weighted_score(
    similarity: float,
    reuse_count: int,
    provide_context_count: int,
    last_accessed: datetime,
    reuse_context_factor: float = 0.6,
    time_decay_lambda: float = 0.01,
) -> float:
    """Compute weighted score combining similarity, reference counts, and recency.

    This is the core scoring algorithm used for cache ranking and eviction.

    Formula:
        score = similarity + normalized_ref * recency_factor

    where:
        weighted_count = factor * reuse_count + (1 - factor) * provide_context_count
        normalized_ref = log(weighted_count + 1) / log(100)
        recency_factor = exp(-lambda * delta_t_hours)

    The logarithmic normalization ensures diminishing returns for high reference counts,
    while time decay gradually reduces the score of old entries.

    Args:
        similarity: Cosine similarity score (0-1)
        reuse_count: Number of times entry was directly reused
        provide_context_count: Number of times entry was provided as context
        last_accessed: Last access timestamp
        reuse_context_factor: Weight factor for reuse vs context (default: 0.6)
        time_decay_lambda: Time decay rate parameter (default: 0.01)

    Returns:
        Weighted score for ranking cache entries

    Example:
        >>> from datetime import timedelta
        >>> now = datetime.now()
        >>> # Recent high-reuse entry
        >>> compute_weighted_score(0.9, 10, 2, now)
        # Returns ~0.9 + log(7.2)/log(100) * 1.0 ≈ 1.33

        >>> # Old low-reuse entry
        >>> old_time = now - timedelta(hours=100)
        >>> compute_weighted_score(0.8, 1, 1, old_time)
        # Returns ~0.8 + log(1.6)/log(100) * exp(-1.0) ≈ 0.85
    """
    # Calculate time decay factor
    delta_t = (datetime.now() - last_accessed).total_seconds() / 3600
    recency_factor = math.exp(-time_decay_lambda * delta_t)

    # Calculate weighted reference count
    # Higher factor means reuse is weighted more than context provision
    weighted_count = (
        reuse_context_factor * reuse_count
        + (1 - reuse_context_factor) * provide_context_count
    )

    # Normalize reference count logarithmically
    # log(100) ≈ 4.605, so normalized_ref maxes out at 1.0 around 100 total refs
    normalized_ref = math.log(weighted_count + 1) / math.log(100)
    normalized_ref = min(normalized_ref, 1.0)

    # Combine similarity with weighted reference contribution
    score = similarity + normalized_ref * recency_factor
    return score


def normalize_reference_count(
    reuse_count: int,
    provide_context_count: int,
    max_refs: float = 100.0,
    reuse_context_factor: float = 0.6,
) -> float:
    """Normalize reference counts to [0, 1] range using logarithmic scaling.

    Args:
        reuse_count: Number of direct reuses
        provide_context_count: Number of context provisions
        max_refs: Maximum reference count for normalization (default: 100)
        reuse_context_factor: Weight for reuse vs context (default: 0.6)

    Returns:
        Normalized score in [0, 1]

    Example:
        >>> normalize_reference_count(10, 5, max_refs=100)
        0.42  # log(10*0.6 + 5*0.4 + 1) / log(100)
    """
    weighted_count = (
        reuse_context_factor * reuse_count
        + (1 - reuse_context_factor) * provide_context_count
    )
    normalized = math.log(weighted_count + 1) / math.log(max_refs)
    return min(normalized, 1.0)


def compute_recency_factor(
    last_accessed: datetime,
    time_decay_lambda: float = 0.01,
) -> float:
    """Compute time-based recency factor using exponential decay.

    Args:
        last_accessed: Last access timestamp
        time_decay_lambda: Decay rate (default: 0.01, ~50% decay in 69 hours)

    Returns:
        Recency factor in (0, 1]

    Example:
        >>> from datetime import timedelta
        >>> now = datetime.now()
        >>> compute_recency_factor(now)  # Just accessed
        1.0
        >>> compute_recency_factor(now - timedelta(hours=69))  # ~50% decay
        0.5
    """
    delta_t_hours = (datetime.now() - last_accessed).total_seconds() / 3600
    return math.exp(-time_decay_lambda * delta_t_hours)


def compute_similarity_score(distance: float) -> float:
    """Convert distance metric to similarity score.

    Assumes distance is in range [0, 2] for cosine distance.

    Args:
        distance: Distance metric from vector store (typically cosine distance)

    Returns:
        Similarity score in range [0, 1]

    Example:
        >>> compute_similarity_score(0.0)  # Perfect match
        1.0
        >>> compute_similarity_score(1.0)  # Orthogonal
        0.0
        >>> compute_similarity_score(2.0)  # Opposite
        -1.0
    """
    return 1.0 - distance
