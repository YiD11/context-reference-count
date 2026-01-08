FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（sentence-transformers 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY pyproject.toml ./
COPY uv.lock ./

# 安装 uv 包管理器
RUN pip install --no-cache-dir uv

# 安装项目依赖
RUN uv pip install --system -e ".[langchain]"

# 安装 FastAPI 和 uvicorn
RUN uv pip install --system fastapi uvicorn

# 复制源代码
COPY src/ ./src/

# 设置环境变量
ENV PYTHONPATH=/app \
    CHROMA_DB_URL=http://chromadb:8000 \
    REDIS_URL=redis://redis:6379

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["python", "-m", "uvicorn", "src.context_ref.api:app", "--host", "0.0.0.0", "--port", "8080"]
