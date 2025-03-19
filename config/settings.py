"""Configuration settings for the Document QA Bot."""

import os
from typing import Dict, Any


def get_settings() -> Dict[str, Any]:
    """
    Get application settings from environment variables.
    
    Returns:
        Dictionary of settings
    """
    return {
        "telegram_token": os.environ.get("TELEGRAM_TOKEN"),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "data_dir": os.environ.get("DATA_DIR", "data"),
        "embedding_provider": os.environ.get("EMBEDDING_PROVIDER", "GEMINI"),
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
        "max_chunk_size": int(os.environ.get("MAX_CHUNK_SIZE", "500")),
        "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", "100")),
        "num_chunks_retrieve": int(os.environ.get("NUM_CHUNKS_RETRIEVE", "20")),
        "num_rerank": int(os.environ.get("NUM_RERANK", "150")),
        "use_reranking": os.environ.get("USE_RERANKING", "1") == "1",
        "batch_size": int(os.environ.get("BATCH_SIZE", "3")),
        "anthropic_rate": float(os.environ.get("ANTHROPIC_RATE", "0.5")),
        "embedding_rate": float(os.environ.get("EMBEDDING_RATE", "1.0")),
    }


def validate_settings(settings: Dict[str, Any]) -> None:
    """
    Validate settings and print warnings for missing keys.
    
    Args:
        settings: Dictionary of settings to validate
    """
    if not settings.get("telegram_token"):
        print("WARNING: TELEGRAM_TOKEN environment variable is not set")
    
    if not settings.get("anthropic_api_key"):
        print("WARNING: ANTHROPIC_API_KEY environment variable is not set")
    
    embedding_provider = settings.get("embedding_provider", "GEMINI")
    if embedding_provider == "GEMINI" and not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable is not set")
    elif embedding_provider == "VOYAGE" and not os.environ.get("VOYAGE_API_KEY"):
        print("WARNING: VOYAGE_API_KEY environment variable is not set")
    elif embedding_provider == "OPENAI" and not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set")
    
    if settings.get("use_reranking") and not os.environ.get("COHERE_API_KEY"):
        print("WARNING: COHERE_API_KEY environment variable is not set, but reranking is enabled")
