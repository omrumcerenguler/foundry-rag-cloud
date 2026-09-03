"""Environment-backed, validated application configuration."""

import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")


class Settings(BaseModel):
    """Runtime settings for local and Azure cloud deployments."""

    rag_mode: Literal["LOCAL", "AZURE_CLOUD"] = "LOCAL"
    database_path: str = str(BASE_DIR / "stores" / "rag.db")
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    api_key: str | None = None
    foundry_model_path: str | None = None
    local_embedding_model: str = "qwen3-embedding-0.6b"
    local_chat_model: str = "Phi-3.5-mini"
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_api_version: str = "2024-10-21"
    azure_embedding_deployment: str = "text-embedding-3-small"
    azure_chat_deployment: str = "gpt-4.1-mini"
    azure_embedding_dimension: int = Field(default=1536, gt=0)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load supported environment variables and validate their values."""
        values: dict[str, object] = {}
        mappings = {
            "RAG_MODE": "rag_mode",
            "RAG_DATABASE_PATH": "database_path",
            "RAG_CONFIDENCE_THRESHOLD": "confidence_threshold",
            "API_KEY": "api_key",
            "FOUNDRY_MODEL_PATH": "foundry_model_path",
            "LOCAL_EMBEDDING_MODEL": "local_embedding_model",
            "LOCAL_CHAT_MODEL": "local_chat_model",
            "AZURE_OPENAI_ENDPOINT": "azure_endpoint",
            "AZURE_OPENAI_API_KEY": "azure_api_key",
            "AZURE_OPENAI_API_VERSION": "azure_api_version",
            "AZURE_EMBEDDING_DEPLOYMENT": "azure_embedding_deployment",
            "AZURE_CHAT_DEPLOYMENT": "azure_chat_deployment",
            "AZURE_EMBEDDING_DIMENSION": "azure_embedding_dimension",
        }
        for environment_name, field_name in mappings.items():
            if (value := os.getenv(environment_name)) is not None:
                values[field_name] = value
        if "rag_mode" in values:
            values["rag_mode"] = str(values["rag_mode"]).upper()
        if "RAG_MODE" not in values and sys.platform.startswith("linux"):
            values["rag_mode"] = "AZURE_CLOUD"
        if "database_path" in values:
            database_path = Path(str(values["database_path"]))
            values["database_path"] = str(
                database_path
                if database_path.is_absolute()
                else BASE_DIR / database_path
            )
        return cls.model_validate(values)
