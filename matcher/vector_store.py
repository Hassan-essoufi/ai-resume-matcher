"""Vector store for resumes and jobs.

QdrantStore  : production (Docker service), persistent, fast top-K search.
InMemoryStore: same interface with numpy, for local runs and tests.
Each record keeps its vector plus a payload (filename, scrubbed text, extracted profile).
"""
import uuid
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams


class VectorStore(Protocol):
    def upsert(self, collection: str, item_id: str, vector: np.ndarray, payload: dict) -> None: ...
    def get(self, collection: str, item_id: str) -> tuple[np.ndarray, dict] | None: ...
    def search(self, collection: str, vector: np.ndarray, k: int) -> list[tuple[str, float, np.ndarray, dict]]: ...
    def list_ids(self, collection: str) -> list[str]: ...
    def delete(self, collection: str, item_id: str) -> None: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, tuple[np.ndarray, dict]]] = {}

    def upsert(self, collection, item_id, vector, payload):
        self._data.setdefault(collection, {})[item_id] = (np.asarray(vector, dtype=np.float32), payload)

    def get(self, collection, item_id):
        return self._data.get(collection, {}).get(item_id)

    def search(self, collection, vector, k):
        items = self._data.get(collection, {})
        if not items:
            return []
        ids = list(items)
        mat = np.stack([items[i][0] for i in ids])
        scores = mat @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[:k]
        return [(ids[j], float(scores[j]), items[ids[j]][0], items[ids[j]][1]) for j in order]

    def list_ids(self, collection):
        return list(self._data.get(collection, {}))

    def delete(self, collection, item_id):
        self._data.get(collection, {}).pop(item_id, None)


class QdrantStore:
    def __init__(self, url: str, dim: int, api_key: str = ""):
        self.client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        self.dim = dim
        self._ready: set[str] = set()

    @staticmethod
    def _point_id(item_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, item_id))   # Qdrant ids must be UUIDs or ints

    def _ensure(self, collection: str) -> None:
        if collection in self._ready:
            return
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))
        self._ready.add(collection)

    def upsert(self, collection, item_id, vector, payload):
        self._ensure(collection)
        self.client.upsert(collection, points=[PointStruct(
            id=self._point_id(item_id), vector=np.asarray(vector).tolist(), payload={**payload, "item_id": item_id})])

    def get(self, collection, item_id):
        self._ensure(collection)
        pts = self.client.retrieve(collection, ids=[self._point_id(item_id)], with_vectors=True)
        return (np.array(pts[0].vector, dtype=np.float32), pts[0].payload) if pts else None

    def search(self, collection, vector, k):
        self._ensure(collection)
        res = self.client.query_points(collection, query=np.asarray(vector).tolist(), limit=k,
                                       with_payload=True, with_vectors=True).points
        return [(p.payload["item_id"], float(p.score), np.array(p.vector, dtype=np.float32), p.payload) for p in res]

    def list_ids(self, collection):
        self._ensure(collection)
        ids, offset = [], None
        while True:
            pts, offset = self.client.scroll(collection, limit=256, offset=offset, with_payload=["item_id"])
            ids += [p.payload["item_id"] for p in pts]
            if offset is None:
                return ids

    def delete(self, collection, item_id):
        self._ensure(collection)
        self.client.delete(collection, points_selector=PointIdsList(points=[self._point_id(item_id)]))
