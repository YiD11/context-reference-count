"""Context Reference Count API 服务."""

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from context_ref.core.cache import ToolCache
from context_ref.core.config import CacheConfig
from context_ref.embedding.default import DefaultEmbedding

app = FastAPI(
    title="Context Reference Count API",
    description="语义缓存插件 API 服务",
    version="0.1.0",
)

_cache: Optional[ToolCache] = None


def get_cache() -> ToolCache:
    """获取或创建缓存实例."""
    global _cache
    if _cache is None:
        config = CacheConfig()
        _cache = ToolCache(config=config, embedding_func=DefaultEmbedding())
    return _cache


class CacheSearchRequest(BaseModel):
    tool_name: str
    input_args: Dict[str, Any]
    top_k: Optional[int] = 5


class CacheSaveRequest(BaseModel):
    tool_name: str
    input_args: Dict[str, Any]
    output: Any
    success: bool = True


class CacheHitResponse(BaseModel):
    entry_id: str
    tool_name: str
    similarity: float
    weighted_score: float
    reference_count: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    total_references: int
    avg_reference_count: float
    max_reference_count: int


@app.post("/api/cache/search")
async def search_cache(request: CacheSearchRequest) -> List[CacheHitResponse]:
    """搜索相似的缓存条目."""
    cache = get_cache()
    hits = cache.search(request.tool_name, request.input_args, request.top_k)

    return [
        CacheHitResponse(
            entry_id=hit.entry.id,
            tool_name=hit.entry.tool_name,
            similarity=hit.similarity,
            weighted_score=hit.weighted_score,
            reference_count=hit.entry.total_reference_count,
        )
        for hit in hits
    ]


@app.post("/api/cache/save")
async def save_cache(request: CacheSaveRequest) -> Dict[str, str]:
    """保存缓存条目."""
    cache = get_cache()
    entry = cache.save(
        tool_name=request.tool_name,
        input_args=request.input_args,
        output=request.output,
        success=request.success,
    )
    return {"entry_id": entry.id, "message": "Saved successfully"}


@app.get("/api/cache/stats")
async def get_cache_stats() -> CacheStatsResponse:
    """获取缓存统计信息."""
    cache = get_cache()
    stats = cache.stats()
    return CacheStatsResponse(**stats)


@app.delete("/api/cache")
async def clear_cache() -> Dict[str, str]:
    """清空所有缓存."""
    cache = get_cache()
    cache.clear()
    return {"message": "Cache cleared successfully"}


@app.get("/health")
async def health_check():
    """健康检查."""
    return {"status": "healthy", "service": "context-ref-api"}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8080"))

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
    )
