from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "NIRMAAN AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()