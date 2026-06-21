"""Typed configuration loaded from config/config.yaml + .env.

Why this exists: every parameter that you'll want to tweak during experiments
(chunk size, top_k, which reranker, max agent iterations) lives in one YAML file,
and every secret/provider choice lives in .env. No magic numbers scattered in code.
"""
from __future__ import annotations

from pathlib import Path
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Env(BaseSettings):
    """Secrets and provider choices from .env."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash-lite"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vector_store_dir: str = "./data/chroma"


class Config(BaseModel):
    """Everything from config.yaml as a nested object."""
    data: dict
    retrieval: dict
    agent: dict
    generation: dict
    evaluation: dict

    @classmethod
    def load(cls, path: str | Path = ROOT / "config" / "config.yaml") -> "Config":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


# Import these anywhere: `from src.config import CONFIG, ENV`
CONFIG = Config.load()
ENV = Env()
