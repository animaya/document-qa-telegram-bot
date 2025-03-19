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
from services.rate_limiter import RateLimitedClient

# Setup logging
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from different providers."""
    
    """Service for generating embeddings from different providers."""
    
    def __init__(self, provider: str = "GEMINI"):
        """
        Initialize the embedding service.
        
        Args:
            provider: Provider for embeddings (GEMINI, VOYAGE, OPENAI)
        """
        self.provider = provider
        self.client = self._initialize_client()
        
        # Add rate limiters for each embedding provider
        self.rate_limited_client = RateLimitedClient(
            name=f"embeddings_{provider.lower()}",
            rate=1.0,  # 1 request per second
            capacity=5
        )
    
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
        Create an embedding for the text using the configured provider with rate limiting.
        
        Args:
            text: Text to create embedding for
            type_hint: Type of embedding to create ("document" or "query")
            
        Returns:
            Vector embedding as a list of floats, or None if creation failed
        """
        try:
            # Define provider-specific embedding functions
            
            async def get_gemini_embedding():
                def execute():
                    task_type = "retrieval_document" if type_hint == "document" else "retrieval_query"
                    result = self.client.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type=task_type
                    )
                    return result["embedding"]
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    return await loop.run_in_executor(pool, execute)
            
            async def get_voyage_embedding():
                def execute():
                    input_type = "document" if type_hint == "document" else "query"
                    response = self.client.embed(
                        texts=[text],
                        model="voyage-large-2",
                        input_type=input_type
                    )
                    return response.embeddings[0]
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    return await loop.run_in_executor(pool, execute)
            
            async def get_openai_embedding():
                def execute():
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    return response.data[0].embedding
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    return await loop.run_in_executor(pool, execute)
            
            # Choose the appropriate embedding function based on provider
            if self.provider == "GEMINI":
                func = get_gemini_embedding
            elif self.provider == "VOYAGE":
                func = get_voyage_embedding
            elif self.provider == "OPENAI":
                func = get_openai_embedding
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
            
            # Call with rate limiting and retries
            return await self.rate_limited_client.call_with_retry(
                func,
                max_retries=3,
                initial_backoff=1.0
            )
        
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None