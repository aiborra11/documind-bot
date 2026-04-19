from pydantic_settings import BaseSettings, SettingsConfigDict

class RAGConfig(BaseSettings):
    """Configuration for Retrieval and Re-ranking models."""

    # RAG defaults
    DEFAULT_INITIAL_TOP_K: int = 10
    DEFAULT_FINAL_TOP_N: int = 3

    # Cross-Encoder
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    RETRIEVAL_DISTANCE_THRESHOLD: float | None = None     # NOTE: We are applying cosine distance so, 0 = perfect match.

    model_config = SettingsConfigDict(env_file=".env")

rag_config = RAGConfig()