from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    """Configuration for the document ingestion pipeline."""
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MIN_CHUNK_LENGTH: int = 50
    MIN_ALPHANUMERIC_RATIO: float = 0.5
    ALLOWED_EXTENSIONS: set[str] = {"pdf"}
    MAX_FILE_SIZE_MB: int = 20
    UPLOAD_CHUNK_SIZE_BYTES: int = 1024 * 1024
    TEMP_FILE_SUFFIX: str = ".pdf"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env")

ingestion_config = IngestionConfig()