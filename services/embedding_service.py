"""
Embedding Service

This module provides embedding generation for different providers,
abstracting away the implementation details.
"""

import os
import logging
import asyncio
from typing import List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from different providers."""
    
    def __init__(self, provider: str = "GEMINI"):
        """
        Initialize the embedding service.
        
        Args:
            provider: Provider for embeddings (GEMINI, VOYAGE, OPENAI)
        """
        self.provider = provider
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> Any:
        """
        Initialize the appropriate embedding client based on the provider.
        
        Returns:
            Initialized embedding client
        """
        if self.provider == "GEMINI":
            # Initialize Gemini embedding client
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                return genai
            except ImportError:
                logger.warning("Gemini SDK not installed. Falling back to OpenAI embeddings.")
                self.provider = "OPENAI"
        
        if self.provider == "VOYAGE":
            # Initialize Voyage embedding client
            try:
                import voyageai
                return voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
            except ImportError:
                logger.warning("Voyage SDK not installed. Falling back to OpenAI embeddings.")
                self.provider = "OPENAI"
        
        # Default to OpenAI embeddings
        if self.provider == "OPENAI":
            try:
                from openai import OpenAI
                return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("No embedding provider available. Please install either 'google-generativeai', 'voyageai', or 'openai'.")
        
        raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    async def create_embedding(self, text: str, type_hint: str = "document") -> Optional[List[float]]:
        """
        Create an embedding for the text using the configured provider.
        
        Args:
            text: Text to create embedding for
            type_hint: Type of embedding to create ("document" or "query")
            
        Returns:
            Vector embedding as a list of floats, or None if creation failed
        """
        try:
            # Different handling based on embedding provider
            if self.provider == "GEMINI":
                # Run in a thread pool to avoid blocking the event loop
                def get_embedding():
                    task_type = "retrieval_document" if type_hint == "document" else "retrieval_query"
                    result = self.client.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type=task_type
                    )
                    return result["embedding"]
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    embedding = await loop.run_in_executor(pool, get_embedding)
                return embedding
            
            elif self.provider == "VOYAGE":
                # Run in a thread pool to avoid blocking the event loop
                def get_embedding():
                    input_type = "document" if type_hint == "document" else "query"
                    response = self.client.embed(
                        texts=[text],
                        model="voyage-large-2",
                        input_type=input_type
                    )
                    return response.embeddings[0]
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    embedding = await loop.run_in_executor(pool, get_embedding)
                return embedding
            
            elif self.provider == "OPENAI":
                # Run in a thread pool to avoid blocking the event loop
                def get_embedding():
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    return response.data[0].embedding
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    embedding = await loop.run_in_executor(pool, get_embedding)
                return embedding
            
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
        
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None