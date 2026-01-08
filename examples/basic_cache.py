"""
Example: Basic cache usage without LangGraph.

This example shows how to use the ToolCache directly for
caching and retrieving tool call results.
"""

from context_ref import ToolCache, CacheConfig


def main():
    config = CacheConfig(
        similarity_threshold=0.7,
        max_cache_size=100,
        similarity_weight=0.6,
        reference_weight=0.3,
        recency_weight=0.1,
    )

    cache = ToolCache(config=config)

    print("=== Saving tool call results ===")

    cache.save(
        tool_name="search",
        input_args={"query": "what is machine learning"},
        output="Machine learning is a subset of AI...",
    )

    cache.save(
        tool_name="search",
        input_args={"query": "explain deep learning"},
        output="Deep learning uses neural networks...",
    )

    for i in range(5):
        cache.save(
            tool_name="search",
            input_args={"query": "what is machine learning"},
            output="Machine learning is a subset of AI...",
        )

    print("=== Searching for similar queries ===")

    hits = cache.search(
        tool_name="search",
        input_args={"query": "tell me about machine learning"},
    )

    for hit in hits:
        print(f"\nSimilarity: {hit.similarity:.3f}")
        print(f"Reference count: {hit.entry.reference_count}")
        print(f"Weighted score: {hit.weighted_score:.3f}")
        print(f"Output: {hit.entry.output[:50]}...")

    print("\n=== Cache Statistics ===")
    stats = cache.stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== Best Match ===")
    best = cache.get_best_match(
        tool_name="search",
        input_args={"query": "what is ML"},
    )

    if best:
        print(f"Found match with similarity {best.similarity:.3f}")
        print(f"Should reuse directly: {best.should_reuse}")
    else:
        print("No match found above threshold")


if __name__ == "__main__":
    main()
