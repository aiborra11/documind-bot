from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    APP_NAME: str = "Documind Bot API"
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    DEBUG: bool = True
    RAW_DATA_PATH: str = "data/raw"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

app_settings = AppConfig()
