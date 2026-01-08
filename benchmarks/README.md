# Benchmark

缓存命中率测试，评估工具调用缓存效果。

## 运行

```bash
python benchmarks/run.py --dataset sample --num-queries 50
python benchmarks/run.py --dataset programming-qa --num-queries 100
python benchmarks/run.py --dataset toolbench --num-queries 200 --subset G1
```

## Python API

```python
from benchmarks import run_benchmark

result = run_benchmark("sample", limit=50)
print(f"命中率: {result.hit_rate:.2%}")
```

## 数据集

| 数据集 | 说明 |
|-------|------|
| sample | 测试数据 |
| programming-qa | 编程问答 |
| toolbench | [ToolBench](https://github.com/OpenBMB/ToolBench) 真实API数据 |

## 输出

结果保存至 `benchmarks/results/`，格式：

```json
{"hit_rate": 0.65, "reuse_count": 150, "context_count": 80}
```