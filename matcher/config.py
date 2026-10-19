from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_timeout_s: float = 60
    llm_max_retries: int = 3

    # "openai" in production; "hashing" = offline, for tests/CI only
    embedding_provider: str = "openai"

    # Vector store: "qdrant" in production, "memory" for tests
    vector_store: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Redis cache of LLM/embedding results, so the same text is never paid for twice
    redis_url: str = ""
    cache_ttl_days: int = 30

    # API security: clients must send this in the X-API-Key header (empty = auth disabled)
    api_key: str = ""

    # Retrieval: how many resumes to shortlist by vector search before scoring
    retrieve_top_k: int = 50

    # Scoring weights (sum to 1)
    w_semantic: float = 0.35
    w_required: float = 0.40
    w_nice: float = 0.10
    w_experience: float = 0.15

    # Cosine similarity of text embeddings is rescaled from [floor, ceiling] to [0, 1]
    semantic_floor: float = 0.15
    semantic_ceiling: float = 0.75

    # Two different skill names count as the same skill above this embedding similarity
    skill_similarity_threshold: float = 0.80

    scrub_pii: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
