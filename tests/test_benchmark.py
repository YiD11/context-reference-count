"""Pytest tests for cache hit rate benchmarking using real datasets."""

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.benchmark import (
    load_queries,
    print_result,
    run_benchmark,
    save_result,
)
from context_ref.core.config import CacheConfig


@pytest.fixture
def sample_queries() -> list[dict[str, Any]]:
    """Sample queries for testing cache functionality."""
    return [
        {
            "id": "1",
            "query": "How do I install Python?",
            "tool_name": "search",
            "input_args": {"query": "How do I install Python?"},
        },
        {
            "id": "2",
            "query": "What is machine learning?",
            "tool_name": "search",
            "input_args": {"query": "What is machine learning?"},
        },
        {
            "id": "3",
            "query": "Explain neural networks",
            "tool_name": "search",
            "input_args": {"query": "Explain neural networks"},
        },
    ]


class TestCacheBenchmark:
    """Tests for benchmark execution."""

    def test_run_benchmark(self, sample_queries: list[dict[str, Any]]) -> None:
        """Test running benchmark and verifying statistics."""
        result = run_benchmark(sample_queries, dataset_name="test_simple")

        assert result["total_queries"] == 3
        assert result["cache_hits"] >= 0
        assert result["cache_misses"] >= 0
        assert 0 <= result["hit_rate"] <= 1
        assert result["avg_query_time"] >= 0
        assert result["total_time"] >= 0

        # Verify hit rate calculation
        expected_total = result["cache_hits"] + result["cache_misses"]
        assert result["total_queries"] == expected_total
        if result["total_queries"] > 0:
            assert (
                abs(result["hit_rate"] - (result["cache_hits"] / result["total_queries"]))
                < 0.001
            )

    def test_split_reference_counting_stats(self, sample_queries: list[dict[str, Any]]) -> None:
        """Test new statistics fields for split reference counting."""
        result = run_benchmark(sample_queries, dataset_name="new_stats")

        assert "reuse_count" in result
        assert "context_count" in result
        assert "execute_count" in result
        assert "total_reuse_count" in result
        assert "total_context_count" in result
        assert result["cache_entries"] > 0

        # Decision distribution should sum to total queries
        total = result["reuse_count"] + result["context_count"] + result["execute_count"]
        assert total == result["total_queries"]


class TestResultSaving:
    """Tests for result saving."""

    def test_save_results(
        self, sample_queries: list[dict[str, Any]], tmp_path: Path
    ) -> None:
        """Test saving results to JSON and CSV."""
        result = run_benchmark(sample_queries, dataset_name="save_test")

        # Test JSON format
        json_path = tmp_path / "benchmark_results.json"
        save_result(result, json_path)
        assert json_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data["dataset_name"] == "save_test"
        assert saved_data["total_queries"] == 3
        assert "hit_rate" in saved_data
        assert "timestamp" in saved_data

        # Test CSV format
        csv_path = tmp_path / "benchmark_results.csv"
        save_result(result, csv_path, format="csv")
        assert csv_path.exists()


class TestDatasetLoading:
    """Tests for dataset loading functions."""

    def test_load_programming_qa_dataset(self) -> None:
        """Test loading programming Q&A dataset."""
        queries = load_queries("programming-qa", limit=20)

        assert len(queries) == 20
        assert all("id" in q for q in queries)
        assert all("query" in q for q in queries)
        assert all("tool_name" in q for q in queries)
        assert all("input_args" in q for q in queries)

    def test_load_sample_dataset(self) -> None:
        """Test loading sample dataset."""
        queries = load_queries("sample", limit=10)

        assert len(queries) == 10
        for q in queries:
            assert "id" in q
            assert "query" in q
            assert "tool_name" in q

    def test_invalid_dataset(self) -> None:
        """Test that invalid dataset name raises error."""
        with pytest.raises(ValueError, match="不支持的数据集"):
            load_queries("invalid_dataset", limit=10)


class TestBenchmarkScenarios:
    """Integration tests for various benchmark scenarios."""

    def test_exact_match_caching(self) -> None:
        """Test that exact matches are cached."""
        queries = [
            {
                "id": "1",
                "query": "Test query",
                "tool_name": "test",
                "input_args": {"query": "Test query"},
            },
            {
                "id": "2",
                "query": "Test query",
                "tool_name": "test",
                "input_args": {"query": "Test query"},
            },
        ]

        result = run_benchmark(queries, dataset_name="exact_match")
        assert result["hit_rate"] >= 0.5

    def test_config_variations(self) -> None:
        """Test benchmark with different configurations."""
        queries = load_queries("sample", limit=10)

        # High threshold config
        high_config = CacheConfig(similarity_threshold=0.9, reuse_threshold=0.95)
        result_high = run_benchmark(queries, config=high_config, dataset_name="high_threshold")
        assert result_high["total_queries"] == 10

        # Low threshold config
        low_config = CacheConfig(similarity_threshold=0.5, reuse_threshold=0.8)
        result_low = run_benchmark(queries, config=low_config, dataset_name="low_threshold")
        assert result_low["total_queries"] == 10


def test_quick_benchmark() -> None:
    """Quick test that can be run without pytest."""
    queries = [
        {
            "id": "1",
            "query": "How do I install Python?",
            "tool_name": "search",
            "input_args": {"query": "How do I install Python?"},
        },
        {
            "id": "2",
            "query": "What is machine learning?",
            "tool_name": "search",
            "input_args": {"query": "What is machine learning?"},
        },
    ]

    result = run_benchmark(queries, dataset_name="quick")

    assert result["total_queries"] == 2
    assert "hit_rate" in result
    assert "reuse_count" in result
    assert "context_count" in result


if __name__ == "__main__":
    test_quick_benchmark()
