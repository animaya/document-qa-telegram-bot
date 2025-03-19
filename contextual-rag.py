"""
Contextual RAG Module

This module implements Anthropic's Contextual Retrieval approach,
combining Contextual Embeddings and Contextual BM25 for improved retrieval.

The implementation follows the recommendations from Anthropic's document
on Contextual Retrieval, including hybrid search and reranking for optimal results.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json
import faiss
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from anthropic import Anthropic

# Import local modules
from document_qa_telegram_bot import Chunk, SessionManager

# Check if Cohere is available for reranking
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logger.warning("Cohere SDK not installed. Reranking will not be available.")

# Setup logging
logger = logging.getLogger(__name__)

# Download NLTK data for tokenization
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

class ContextualRAG:
    """
    Implements Contextual Retrieval using a hybrid of semantic and lexical search.
    """
    
    def __init__(
        self, 
        session_manager: SessionManager,
        anthropic_api_key: str,
        embedding_provider: str = "GEMINI",
        model: str = "claude-3-haiku-20240307",
        num_chunks_retrieve: int = 20,
        num_rerank: int = 150,
        use_reranking: bool = True
    ):
        """
        Initialize the Contextual RAG system.
        
        Args:
            session_manager: SessionManager instance to access processed chunks
            anthropic_api_key: API key for Anthropic
            embedding_provider: Provider for embeddings (GEMINI, VOYAGE, OPENAI)
            model: Anthropic model to use for answer generation
            num_chunks_retrieve: Number of chunks to retrieve
            num_rerank: Number of chunks to consider for reranking
            use_reranking: Whether to use reranking (requires Cohere API key)
        """
        self.session_manager = session_manager
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        self.embedding_provider = embedding_provider
        self.model = model
        self.num_chunks_retrieve = num_chunks_retrieve
        self.num_rerank = num_rerank
        self.use_reranking = use_reranking and COHERE_AVAILABLE
        
        # Initialize embedding client
        self.embedding_client = self._initialize_embedding_client()
        
        # Initialize reranker if available
        self.reranker = None
        if self.use_reranking and COHERE_AVAILABLE:
            cohere_api_key = os.environ.get("COHERE_API_KEY")
            if cohere_api_key:
                try:
                    self.reranker = cohere.Client(cohere_api_key)
                except Exception as e:
                    logger.error(f"Failed to initialize Cohere client: {e}")
                    self.use_reranking = False
            else:
                logger.warning("COHERE_API_KEY not set. Reranking will not be used.")
                self.use_reranking = False
        
        # Initialize FAISS index and BM25 index for each user session
        self.user_indices = {}  # Map of user_id to (FAISS index, BM25 index, chunk_map)
    
    def _initialize_embedding_client(self) -> Any:
        """
        Initialize the appropriate embedding client based on the provider.
        
        Returns:
            Initialized embedding client
        """
        if self.embedding_provider == "GEMINI":
            # Initialize Gemini embedding client
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                return genai
            except ImportError:
                logger.warning("Gemini SDK not installed. Falling back to OpenAI embeddings.")
                self.embedding_provider = "OPENAI"
        
        if self.embedding_provider == "VOYAGE":
            # Initialize Voyage embedding client
            try:
                import voyageai
                return voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
            except ImportError:
                logger.warning("Voyage SDK not installed. Falling back to OpenAI embeddings.")
                self.embedding_provider = "OPENAI"
        
        # Default to OpenAI embeddings
        if self.embedding_provider == "OPENAI":
            try:
                from openai import OpenAI
                return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("No embedding provider available. Please install either 'google-generativeai', 'voyageai', or 'openai'.")
        
        raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
    
    async def prepare_indices(self, user_id: int) -> bool:
        """
        Prepare FAISS and BM25 indices for a user's session.
        
        Args:
            user_id: ID of the user
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all chunks for the user's session
            chunks = self.session_manager.get_session_chunks(user_id)
            if not chunks:
                logger.warning(f"No chunks found for user {user_id}")
                return False
            
            # Create a mapping from index to chunk
            chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
            
            # Create FAISS index for semantic search
            embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]
            if not embeddings:
                logger.warning(f"No embeddings found for user {user_id}")
                return False
            
            # Convert embeddings to numpy array
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_np)
            
            # Create BM25 index for lexical search
            tokenized_chunks = []
            for chunk in chunks:
                if chunk.contextualized_content:
                    # Tokenize the contextualized content
                    tokens = word_tokenize(chunk.contextualized_content.lower())
                    tokenized_chunks.append(tokens)
                else:
                    # Fallback to original content
                    tokens = word_tokenize(chunk.text_content.lower())
                    tokenized_chunks.append(tokens)
            
            bm25 = BM25Okapi(tokenized_chunks)
            
            # Store indices for this user
            self.user_indices[user_id] = (index, bm25, chunk_map)
            
            logger.info(f"Indices prepared for user {user_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing indices for user {user_id}: {e}")
            return False
    
    async def query(self, user_id: int, query_text: str) -> Tuple[List[Chunk], str]:
        """
        Query the system with a user's question.
        
        Args:
            user_id: ID of the user
            query_text: User's question
            
        Returns:
            Tuple of (retrieved chunks, generated answer)
        """
        try:
            # Ensure indices are prepared
            if user_id not in self.user_indices:
                success = await self.prepare_indices(user_id)
                if not success:
                    return [], "I'm sorry, but I couldn't access the document information. Please try uploading your documents again."
            
            # Get indices for this user
            index, bm25, chunk_map = self.user_indices[user_id]
            
            # Create embedding for the query
            query_embedding = await self._create_embedding(query_text)
            if query_embedding is None:
                return [], "I'm sorry, but I'm having trouble processing your question. Please try again."
            
            # Convert query embedding to numpy array
            query_embedding_np = np.array([query_embedding]).astype('float32')
            
            # Semantic search with FAISS
            k = min(self.num_rerank, len(chunk_map))  # Number of results to retrieve
            semantic_distances, semantic_indices = index.search(query_embedding_np, k)
            semantic_indices = semantic_indices[0].tolist()
            
            # Lexical search with BM25
            tokenized_query = word_tokenize(query_text.lower())
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Get top BM25 results
            bm25_indices = np.argsort(bm25_scores)[::-1][:k].tolist()
            
            # Combine results using rank fusion
            combined_indices = list(set(semantic_indices + bm25_indices))
            
            # If reranking is enabled, rerank the combined results
            final_indices = combined_indices
            if self.use_reranking and self.reranker and len(combined_indices) > 0:
                # Get texts for reranking
                texts = [chunk_map[i].contextualized_content for i in combined_indices if i in chunk_map]
                
                # Rerank
                rerank_results = await self._rerank(query_text, texts)
                
                # Map back to original indices
                if rerank_results:
                    # Rerank results are sorted by relevance
                    reranked_indices = [combined_indices[i] for i in rerank_results]
                    final_indices = reranked_indices[:self.num_chunks_retrieve]
                else:
                    # Fallback to combined results
                    final_indices = combined_indices[:self.num_chunks_retrieve]
            else:
                # Just take the top results from the combined list
                final_indices = combined_indices[:self.num_chunks_retrieve]
            
            # Get the retrieved chunks
            retrieved_chunks = [chunk_map[i] for i in final_indices if i in chunk_map]
            
            # Generate answer using retrieved chunks
            answer = await self._generate_answer(query_text, retrieved_chunks)
            
            return retrieved_chunks, answer
            
        except Exception as e:
            logger.error(f"Error querying for user {user_id}: {e}")
            return [], "I'm sorry, but I encountered an error while processing your question. Please try again."
    
    async def _create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create an embedding for the text using the configured provider.
        
        Args:
            text: Text to create embedding for
            
        Returns:
            Vector embedding as a list of floats, or None if creation failed
        """
        try:
            # Different handling based on embedding provider
            if self.embedding_provider == "GEMINI":
                # Run in a thread pool to avoid blocking the event loop
                def get_embedding():
                    result = self.embedding_client.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_query"  # Use query embedding for queries
                    )
                    return result["embedding"]
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    embedding = await loop.run_in_executor(pool, get_embedding)
                return embedding
            
            elif self.embedding_provider == "VOYAGE":
                # Run in a thread pool to avoid blocking the event loop
                def get_embedding():
                    response = self.embedding_client.embed(
                        texts=[text],
                        model="voyage-large-2",
                        input_type="query"  # Use query embedding for queries
                    )
                    return response.embeddings[0]
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    embedding = await loop.run_in_executor(pool, get_embedding)
                return embedding
            
            elif self.embedding_provider == "OPENAI":
                # Run in a thread pool to avoid blocking the event loop
                def get_embedding():
                    response = self.embedding_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    return response.data[0].embedding
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    embedding = await loop.run_in_executor(pool, get_embedding)
                return embedding
            
            else:
                raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
        
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None
    
    async def _rerank(self, query: str, texts: List[str]) -> Optional[List[int]]:
        """
        Rerank texts using Cohere's reranking API.
        
        Args:
            query: User's question
            texts: List of texts to rerank
            
        Returns:
            List of indices sorted by relevance, or None if reranking failed
        """
        if not self.use_reranking or not self.reranker or not texts:
            return None
        
        try:
            # Run in a thread pool to avoid blocking the event loop
            def do_rerank():
                response = self.reranker.rerank(
                    query=query,
                    documents=texts,
                    top_n=min(self.num_chunks_retrieve, len(texts)),
                    model="rerank-english-v2.0"
                )
                # Extract indices from results
                return [r.index for r in response.results]
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                indices = await loop.run_in_executor(pool, do_rerank)
            return indices
        
        except Exception as e:
            logger.error(f"Error reranking: {e}")
            return None
    
    async def _generate_answer(self, query: str, chunks: List[Chunk]) -> str:
        """
        Generate an answer using Anthropic's Claude and the retrieved chunks.
        
        Args:
            query: User's question
            chunks: List of retrieved chunks
            
        Returns:
            Generated answer
        """
        try:
            # Format chunks as context
            context = ""
            for i, chunk in enumerate(chunks):
                # Use the contextualized content if available
                content = chunk.contextualized_content if chunk.contextualized_content else chunk.text_content
                context += f"\n\nCHUNK {i+1}:\n{content}\n"
            
            # Prepare the prompt for answer generation
            prompt = f"""
I'll help you answer a question based on the provided document chunks. 

Here are the relevant parts of the documents:
{context}

The user's question is: {query}

Instructions:
1. Base your answer specifically on the information from these document chunks.
2. If the documents don't contain the information needed to answer the question, say so clearly.
3. Don't reference the chunk numbers in your answer.
4. Provide concise and accurate information without unnecessary elaboration.
5. If you need to quote specific information from the documents, do so directly.
6. Maintain a helpful and informative tone.
"""
            
            # Call Anthropic API to generate answer
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and return the generated answer
            return response.content[0].text.strip()
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, but I encountered an error while generating an answer. Please try again."