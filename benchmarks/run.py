#!/usr/bin/env python3
"""基准测试命令行工具。

用法：
    # 模拟调用
    python benchmarks/run.py --dataset toolbench --num-queries 10

    # 真实 API 调用
    export RAPIDAPI_KEY=your_key
    python benchmarks/run.py --dataset toolbench --num-queries 5 --real-api
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path="benchmarks/.env")  # RAPIDAPI_KEY loaded from benchmarks/.env

from benchmarks.benchmark import load_queries, print_result, run_benchmark, save_result
from context_ref.core.config import CacheConfig


def main():
    parser = argparse.ArgumentParser(description="运行基准测试")
    parser.add_argument(
        "--dataset",
        choices=["programming-qa", "toolbench", "sample"],
        default="programming-qa",
        help="数据集类型",
    )
    parser.add_argument("--num-queries", type=int, default=100, help="查询数量")
    parser.add_argument(
        "--subset", choices=["G1", "G2", "G3"], default="G1", help="ToolBench子集"
    )
    parser.add_argument("--data-dir", help="ToolBench数据目录")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="ToolBench类别筛选（空格分隔，例如：Weather Finance Translation），默认全选",
    )
    parser.add_argument(
        "--real-api", action="store_true", help="使用真实API调用（需要RAPIDAPI_KEY）"
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.8, help="相似度阈值"
    )
    parser.add_argument("--reuse-threshold", type=float, default=0.95, help="重用阈值")
    parser.add_argument(
        "--max-cache-size", type=int, default=1000, help="最大缓存条目数"
    )
    parser.add_argument(
        "--output-format", choices=["json", "csv"], default="json", help="输出格式"
    )
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument(
        "--enable-compare",
        action="store_true",
        help="启用性能对比测试（对重用的工具进行5次实际执行测试并计算时间节约）",
    )

    args = parser.parse_args()

    if not args.quiet:
        print(f"数据集：{args.dataset}")
        if args.dataset == "toolbench":
            print(f"子集：{args.subset}")
            print(f"API模式：{'真实调用' if args.real_api else '模拟调用'}")
            if args.categories:
                print(f"类别筛选：{', '.join(args.categories)}")
        print(f"查询数量：{args.num_queries}\n")

    queries = load_queries(
        args.dataset,
        limit=args.num_queries,
        subset=args.subset,
        data_dir=args.data_dir,
        categories=args.categories,
    )

    if not args.quiet:
        print(f"已加载 {len(queries)} 个查询")

    executor = None
    if args.real_api and args.dataset == "toolbench":
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            print(
                "错误：使用 --real-api 需要设置 RAPIDAPI_KEY 环境变量", file=sys.stderr
            )
            sys.exit(1)

        from benchmarks.executor import ToolBenchAPIExecutor

        tool_dir = Path(args.data_dir or "benchmarks/tool_bench_data") / "toolenv/tools"
        if not tool_dir.exists():
            print(f"错误：工具目录不存在: {tool_dir}", file=sys.stderr)
            sys.exit(1)

        api_executor = ToolBenchAPIExecutor(tool_dir, api_key)

        def executor(tool_name: str, input_args: dict):
            api_name = input_args.get("api_name", tool_name)
            category = input_args.get("category")
            arguments = input_args.get("arguments", {})
            result = api_executor.execute_api(tool_name, api_name, arguments, category)
            return result.success, result.output

        if not args.quiet:
            print(f"已初始化真实 API 执行器\n")

    if not args.quiet:
        print("开始运行基准测试...\n")

    config = CacheConfig(
        similarity_threshold=args.similarity_threshold,
        reuse_threshold=args.reuse_threshold,
        max_cache_size=args.max_cache_size,
    )

    result = run_benchmark(
        queries,
        config,
        executor,
        dataset_name=args.dataset,
        enable_compare=args.enable_compare,
        quiet=args.quiet,
    )

    if not args.quiet:
        print_result(result)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path("benchmarks/results")
            / f"{args.dataset}_{args.subset}_benchmark.{args.output_format}"
        )

    save_result(result, output_path, format=args.output_format)

    if not args.quiet:
        print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
