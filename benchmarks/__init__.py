"""基准测试模块。

简化的接口：
- load_queries: 加载数据
- run_benchmark: 运行 workflow
- print_result / save_result: 输出结果
"""

from benchmarks.benchmark import (
    load_queries,
    print_result,
    run_benchmark,
    save_result,
)

__all__ = [
    "load_queries",
    "run_benchmark",
    "print_result",
    "save_result",
]
