
import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms


class OpenAIEmbedder:
    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, client):
        self.client = client
        self.dim = self.DIMS.get(client.embedding_model, 1536)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.array(self.client.embed(texts), dtype=np.float32))


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9+#.]+", text.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _normalize(out)


def profile_text(summary: str, titles: list[str], skills: list[str], body: str, max_body: int = 3000) -> str:
    """What gets embedded: the structured essentials first, then the (scrubbed) raw text."""
    head = " | ".join(filter(None, [summary, ", ".join(titles), "skills: " + ", ".join(skills)]))
    return f"{head}\n{body[:max_body]}"
