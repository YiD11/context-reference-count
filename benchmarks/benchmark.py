"""基准测试 - 简化版。

读取数据 → 运行 workflow → 输出结果
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from context_ref.core.cache import ToolCache, CacheEntry
from context_ref.core.config import CacheConfig
from context_ref.core.storage.memory import MemoryStorageBackend
from context_ref.interceptor.wrapper import CacheDecision, ToolInterceptor


def load_queries(dataset: str, limit: int = 100, **kwargs) -> list[dict[str, Any]]:
    """加载查询数据。

    Args:
        dataset: 数据集名称 (programming-qa, toolbench, sample)
        limit: 查询数量
        **kwargs: 数据集特定参数 (subset, data_dir, categories)

    Returns:
        查询列表，每个查询包含 tool_name 和 input_args
    """
    if dataset == "programming-qa":
        return _load_programming_qa(limit)
    elif dataset == "toolbench":
        return _load_toolbench(limit, **kwargs)
    elif dataset == "sample":
        return _load_sample(limit)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")


def _load_programming_qa(limit: int) -> list[dict]:
    """加载编程问答数据。"""
    queries = []
    categories = {
        "python": [
            "How to install Python on Windows?",
            "How do I install Python 3.11 on Windows 10?",
            "Python installation guide for Windows 11",
            "pip install command not found Python",
            "How to add Python to PATH on Windows?",
        ],
        "git": [
            "How to undo last git commit?",
            "Undo last commit but keep changes",
            "Git reset soft head~1",
            "How to revert a commit in git?",
            "Git commit --amend to modify last commit",
        ],
        "docker": [
            "Docker container exited with code 0",
            "Docker container stops immediately",
            "How to keep container running?",
            "Docker run --detach flag explained",
            "Docker CMD vs ENTRYPOINT difference",
        ],
        "sql": [
            "How to join two tables in SQL?",
            "SQL INNER JOIN vs LEFT JOIN",
            "Join multiple tables in SQL",
            "SQL query to combine tables",
            "PostgreSQL join syntax",
        ],
        "database": [
            "How to connect to PostgreSQL?",
            "PostgreSQL connection string format",
            "SQLAlchemy database connection setup",
            "Database connection pooling Python",
            "psycopg2 connection example",
        ],
    }

    query_id = 1
    for category, texts in categories.items():
        for text in texts:
            queries.append(
                {
                    "id": str(query_id),
                    "query": text,
                    "tool_name": "search",
                    "input_args": {"query": text},
                    "category": category,
                }
            )
            query_id += 1
            if len(queries) >= limit:
                return queries
    return queries


def _load_toolbench(
    limit: int,
    subset: str = "G1",
    data_dir: str | None = None,
    categories: list[str] | None = None,
) -> list[dict]:
    """加载 ToolBench 数据。"""
    from benchmarks.toolbench import format_for_benchmark, load_toolbench_queries

    if data_dir is None:
        data_dir = "./benchmarks/tool_bench_data"

    data_dir = Path(data_dir)

    queries = load_toolbench_queries(
        data_dir, subset=subset, limit=limit, categories=categories
    )
    return format_for_benchmark(queries)


def _load_sample(limit: int) -> list[dict]:
    """加载示例数据。"""
    samples = [
        ("search", {"query": "Python list comprehension"}),
        ("search", {"query": "Python list comprehension tutorial"}),
        ("search", {"query": "How to use list comprehension"}),
        ("weather", {"location": "San Francisco"}),
        ("weather", {"location": "SF", "units": "metric"}),
        ("calculator", {"operation": "add", "a": 2, "b": 2}),
        ("calculator", {"expression": "2 + 2"}),
        ("translate", {"text": "hello", "target": "es"}),
        ("translate", {"text": "hello", "target": "spanish"}),
        ("stock", {"symbol": "AAPL"}),
    ]

    queries = []
    for i in range(limit):
        tool_name, input_args = samples[i % len(samples)]
        queries.append(
            {
                "id": str(i + 1),
                "query": f"Sample query {i + 1}",
                "tool_name": tool_name,
                "input_args": input_args,
            }
        )
    return queries


def run_benchmark(
    queries: list[dict[str, Any]],
    config: CacheConfig | None = None,
    executor: Callable[[str, dict], tuple[bool, Any]] | None = None,
    storage_backend: MemoryStorageBackend | None = None,
    dataset_name: str = "benchmark",
    enable_compare: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """运行基准测试 workflow。

    Args:
        queries: 查询列表
        config: 缓存配置
        executor: API 执行器，(tool_name, args) -> (success, output)
        dataset_name: 数据集名称
        enable_compare: 是否启用性能对比测试
        quiet: 是否静默模式

    Returns:
        结果字典
    """
    config = config or CacheConfig()
    storage = storage_backend or MemoryStorageBackend()
    cache = ToolCache(config=config, storage=storage)
    interceptor = ToolInterceptor(cache=cache, config=config)

    reuse_count = 0
    context_count = 0
    execute_count = 0
    start_time = time.time()
    from tqdm import tqdm

    with tqdm(queries, desc="benchmark", leave=False) as pbar:
        for query in pbar:
            decision_result = interceptor.decide(
                query["tool_name"],
                query["input_args"],
            )

            if decision_result.decision == CacheDecision.REUSE:
                reuse_count += 1

            elif decision_result.decision == CacheDecision.PROVIDE_CONTEXT:
                context_count += 1
                context_ids = set()
                if decision_result.context_hints:
                    for hint in decision_result.context_hints:
                        context_ids.add(hint.entry.id)

                saved_entry = _execute_and_save(cache, executor, query)
                if saved_entry.id in context_ids:
                    cache.storage.decrement_reference(saved_entry.id)

            else:
                execute_count += 1
                _execute_and_save(cache, executor, query)

            pbar.set_postfix_str(
                f"Reuse: {reuse_count}, Context: {context_count}, Exec: {execute_count}"
            )

    total_time = time.time() - start_time
    stats = cache.stats()

    # 性能对比测试：对重用的工具进行实际执行测试
    avg_execution_time = 0.0
    total_time_saved = 0.0
    if enable_compare and reuse_count > 0 and executor:
        if not quiet:
            print("\n开始性能对比测试...")

        # 收集所有被重用的查询
        reused_queries = []
        for query in queries:
            decision_result = interceptor.decide(
                query["tool_name"],
                query["input_args"],
            )
            if decision_result.decision == CacheDecision.REUSE:
                reused_queries.append(query)

        # 对每个重用的查询进行5次实际执行测试
        execution_times = []
        from tqdm import tqdm

        sample_size = min(len(reused_queries), 100)  # 最多测试100个样本
        with tqdm(
            reused_queries[:sample_size],
            desc="性能对比",
            disable=quiet,
            leave=False,
        ) as pbar:
            for query in pbar:
                times = []
                for _ in range(5):
                    start = time.time()
                    try:
                        executor(query["tool_name"], query["input_args"])
                    except Exception:
                        pass  # 忽略执行错误，只关注时间
                    times.append(time.time() - start)

                # 使用平均时间
                avg_time = sum(times) / len(times) if times else 0
                execution_times.append(avg_time)

        if execution_times:
            avg_execution_time = sum(execution_times) / len(execution_times)
            # 计算节约的时间：重用次数 * 平均执行时间
            total_time_saved = reuse_count * avg_execution_time

            if not quiet:
                print(f"平均单次执行时间：{avg_execution_time * 1000:.2f}ms")
                print(f"重用节约总时间：{total_time_saved:.2f}s")

    result = {
        "dataset_name": dataset_name,
        "total_queries": len(queries),
        "cache_hits": reuse_count + context_count,
        "cache_misses": execute_count,
        "hit_rate": (reuse_count + context_count) / len(queries) if queries else 0.0,
        "reuse_count": reuse_count,
        "context_count": context_count,
        "execute_count": execute_count,
        "avg_query_time": total_time / len(queries) if queries else 0.0,
        "total_time": total_time,
        "cache_entries": stats["total_entries"],
        "avg_references": stats["avg_reference_count"],
        "total_reuse_count": stats.get("total_reuse_count", 0),
        "total_context_count": stats.get("total_context_count", 0),
        "timestamp": datetime.now().isoformat(),
    }

    # 添加性能对比数据
    if enable_compare and reuse_count > 0:
        result["avg_execution_time"] = avg_execution_time
        result["total_time_saved"] = total_time_saved
        result["time_saved_percentage"] = (
            total_time_saved / (total_time + total_time_saved) * 100
            if (total_time + total_time_saved) > 0
            else 0.0
        )

    return result


def _execute_and_save(
    cache: ToolCache,
    executor: Callable | None,
    query: dict[str, Any],
) -> CacheEntry:
    """执行并保存结果。"""
    tool_name = query["tool_name"]
    input_args = query["input_args"]

    if executor:
        try:
            success, output = executor(tool_name, input_args)
        except Exception as e:
            success = False
            output = {"error": str(e)}
    else:
        success = True
        output = {
            "result": f"Mock response for {tool_name}",
            "args": input_args,
            "timestamp": datetime.now().isoformat(),
        }

    return cache.save(tool_name, input_args, output, success=success)


def print_result(result: dict[str, Any]) -> None:
    """打印结果到控制台。"""
    print(f"\n{'=' * 60}")
    print("基准测试结果")
    print(f"{'=' * 60}")
    print(f"数据集：{result['dataset_name']}")
    print(f"总查询数：{result['total_queries']}")
    print(f"缓存命中：{result['cache_hits']}")
    print(f"缓存未命中：{result['cache_misses']}")
    print(f"命中率：{result['hit_rate']:.2%}")

    print(f"\n决策类型分布：")
    total = result["total_queries"]
    for name, key in [
        ("REUSE（直接重用）", "reuse_count"),
        ("PROVIDE_CONTEXT（提供上下文）", "context_count"),
        ("EXECUTE（执行调用）", "execute_count"),
    ]:
        count = result[key]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {name}:")
        print(f"    数量：{count}")
        print(f"    比例：{pct:.1f}%")

    print(f"\n性能指标：")
    print(f"  平均查询时间：{result['avg_query_time'] * 1000:.2f}ms")
    print(f"  总耗时：{result['total_time']:.2f}s")
    print(f"  缓存条目数：{result['cache_entries']}")
    print(f"  平均引用计数：{result['avg_references']:.2f}")
    if "total_reuse_count" in result:
        print(f"  总重用计数：{result['total_reuse_count']}")
    if "total_context_count" in result:
        print(f"  总上下文计数：{result['total_context_count']}")

    # 性能对比数据
    if "avg_execution_time" in result:
        print(f"\n性能对比（启用 --enable-compare）：")
        print(f"  平均单次执行时间：{result['avg_execution_time'] * 1000:.2f}ms")
        print(f"  工具重用节约总时间：{result['total_time_saved']:.2f}s")
        print(f"  时间节约百分比：{result['time_saved_percentage']:.2f}%")


def save_result(
    result: dict[str, Any], output_path: str | Path, format: str = "json"
) -> None:
    """保存结果到文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    elif format == "csv":
        import csv

        file_exists = output_path.exists()
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)
    else:
        raise ValueError(f"不支持的格式: {format}")
