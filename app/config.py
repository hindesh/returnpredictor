from pathlib import Path
from pydantic_settings import BaseSettings

# Absolute project root — works locally and on Vercel (/var/task/)
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    database_url: str = "postgresql://returnshield:returnshield_pass@localhost:5432/returnshield"
    model_dir: str = "trained_models"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
