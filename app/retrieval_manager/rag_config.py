from pydantic_settings import BaseSettings, SettingsConfigDict

class RAGConfig(BaseSettings):
    """Configuration for Retrieval and Re-ranking models."""

    # RAG defaults
    DEFAULT_QUERY_TOP_K: int = 10
    RETRIEVAL_DISTANCE_THRESHOLD: float | None = None

    model_config = SettingsConfigDict(env_file=".env")

rag_config = RAGConfig()