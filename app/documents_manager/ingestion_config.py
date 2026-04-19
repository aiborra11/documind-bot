from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    """Configuration for the document ingestion pipeline."""
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    #TODO: We must validate what are we filtering out with these thresholds.
    MIN_CHUNK_LENGTH: int = 50              # Drops tiny chunks coming from headers, footers, or page numbers that don't provide meaningful context.
    MIN_ALPHANUMERIC_RATIO: float = 0.4     # Ensures chunks contain a reasonable amount of text vs. symbols, which helps preserve tables/markdown while filtering out noise.
    ALLOWED_EXTENSIONS: set[str] = {"pdf", ".xls", ".xlsx", ".doc", ".docx", ".txt"}
    MAX_FILE_SIZE_MB: int = 35              
    UPLOAD_CHUNK_SIZE_BYTES: int = 1024 * 1024
    TEMP_FILE_SUFFIX: str = ".pdf"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env")

ingestion_config = IngestionConfig()