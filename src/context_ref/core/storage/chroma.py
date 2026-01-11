"""ChromaDB vector store implementation."""

from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from context_ref.core.config import get_chroma_config
from context_ref.core.storage.vector import VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store for similarity search."""

    def __init__(
        self,
        collection_name: str = "tool_cache",
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        mode: str = "ephemeral",
    ) -> None:
        self._collection_name = collection_name
        self._host = host
        self._port = port
        self._path = path
        self._mode = mode
        self._client: ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
    
    def get_client(self) -> ClientAPI:
        self._init_client()
        assert isinstance(self._client, ClientAPI)
        return self._client
    
    def get_collection(self) -> chromadb.Collection:
        self._init_client()
        assert isinstance(self._collection, chromadb.Collection)
        return self._collection

    def _init_client(self) -> None:
        if self._client is not None:
            return

        settings = Settings(anonymized_telemetry=False)

        if self._mode == "client" and self._host:
            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port or 8000,
                settings=settings,
            )
        elif self._mode == "persistent" and self._path:
            self._client = chromadb.PersistentClient(path=self._path, settings=settings)
        else:
            self._client = chromadb.Client(settings=settings)

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self._init_client()
        collection = self.get_collection()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
        )

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> chromadb.QueryResult:
        self._init_client()
        collection = self.get_collection()
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
            include=["distances", "metadatas", "documents"],
        )

    def delete(self, ids: list[str]) -> None:
        if self._collection is not None:
            self._collection.delete(ids=ids)

    def clear(self) -> None:
        c = self.get_client()
        c.delete_collection(self._collection_name)
        self._collection = c.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        self._client = None
        self._collection = None

    @classmethod
    def from_env(cls) -> "ChromaVectorStore":
        """Create from environment configuration."""
        config = get_chroma_config()
        if config.is_client_mode():
            return cls(
                host=config.host or "localhost",
                port=config.port or 8000,
                mode="client",
            )
        if config.is_persistent_mode():
            return cls(path=config.path, mode="persistent")
        return cls(mode="ephemeral")
