from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    TOGETHER_API_KEY: str = ""
    VECTORSTORE_PATH: str = "./vectorstore"
    MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 500

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings() 