from pydantic import BaseModel, Field

from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseConfig(BaseSettings):
    # Database Settings
    COLLECTION_NAME: str = "financial_docs"
    VECTOR_DB_PATH: str = "./data/chroma_db"

    METADATA_DISTANCE_SPACE: str = "hnsw:space"
    DISTANCE_METRIC: str = "cosine"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env")

db_config = DatabaseConfig()


