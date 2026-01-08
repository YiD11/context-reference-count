# Context Reference Count

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LLM工具调用的语义缓存，减少重复调用。

## 为什么用

- 语义相似查询复用缓存结果
- 分离式引用计数（复用/上下文提示）
- Redis/内存存储 + ChromaDB向量检索
- LangGraph无缝集成

## 安装

```bash
uv add context-ref
```

## 快速开始

### LangGraph

```python
from langgraph.prebuilt import ToolNode
from context_ref import ToolInterceptor, CacheConfig

interceptor = ToolInterceptor(
    config=CacheConfig(similarity_threshold=0.75)
)

tool_node = ToolNode(
    tools=[my_tool],
    wrap_tool_call=interceptor.create_wrapper(),
)
```

### 单个函数

```python
from context_ref import ToolInterceptor

interceptor = ToolInterceptor()

@interceptor.wrap_tool
def search(query: str) -> str:
    return api_call(query)

search("python")  # 缓存
search("python")  # 命中
```

## 配置

| 参数 | 默认 | 说明 |
|-----|-----|-----|
| similarity_threshold | 0.75 | 最小相似度 |
| reuse_threshold | 0.95 | 直接复用阈值 |
| max_cache_size | 1000 | 最大条目数 |
| eviction_policy | score | 淘汰策略 |

环境变量：`REDIS_URL`, `CHROMADB_MODE`, `APP_PORT`

## API服务

```bash
uvicorn context_ref.api:app --port 8080
```

## References

This project is inspired by:

- [GPTCache](https://github.com/zilliztech/GPTCache) - Semantic caching for LLM queries
- [Letta](https://github.com/letta-ai/letta)
- [MemGPT/Letta](https://github.com/letta-ai/letta) - Memory management for LLM agents
- [Asteria](https://arxiv.org/abs/2501.xxxxx) - Semantic-aware caching for agentic tool access
- [AgentReuse](https://arxiv.org/abs/2512.21309) - Plan reuse for LLM agents