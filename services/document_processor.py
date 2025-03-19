"""
Document Processor Module

This module handles the extraction, chunking, and contextualization of documents
according to Anthropic's recommendations for Contextual Retrieval.

The implementation follows Anthropic's workflows:
- Prompt chaining for document processing (extract → chunk → contextualize → embed)
- Orchestrator-workers pattern for parallel processing when possible
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
from anthropic import Anthropic
import numpy as np

# Import local modules
from document_qa_bot.core.document import Document
from document_qa_bot.core.chunk import Chunk
from document_qa_bot.core.session_manager import SessionManager
from document_qa_bot.services.embedding_service import EmbeddingService

# Setup logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles the extraction, chunking, and processing of documents.
    """
    
    def __init__(
        self, 
        session_manager: SessionManager, 
        anthropic_api_key: str,
        embedding_provider: str = "GEMINI",
        model: str = "claude-3-haiku-20240307", 
        max_chunk_size: int = 500, 
        chunk_overlap: int = 100
    ):
        """
        Initialize the document processor.
        
        Args:
            session_manager: SessionManager instance to store processed chunks
            anthropic_api_key: API key for Anthropic
            embedding_provider: Provider for embeddings (GEMINI, VOYAGE, OPENAI)
            model: Anthropic model to use for contextualization
            max_chunk_size: Maximum size of chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.session_manager = session_manager
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        self.embedding_service = EmbeddingService(embedding_provider)
        self.model = model
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def process_document(self, document: Document) -> bool:
        """
        Process a document: extract text, chunk, contextualize, and create embeddings.
        
        Args:
            document: Document object to process
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Extract text from PDF
            document_text = await self._extract_text(document.file_path)
            if not document_text:
                logger.warning(f"No text extracted from document: {document.filename}")
                return False
            
            # Add document metadata
            document.metadata["total_length"] = len(document_text)
            document.metadata["extracted_text"] = document_text[:1000] + "..." if len(document_text) > 1000 else document_text
            
            # Create chunks
            chunks = await self._create_chunks(document_text, document.document_id)
            if not chunks:
                logger.warning(f"No chunks created from document: {document.filename}")
                return False
            
            # Process chunks in parallel to add context and embeddings
            processed_chunks = await self._process_chunks(chunks, document_text)
            if not processed_chunks:
                logger.warning(f"Failed to process chunks for document: {document.filename}")
                return False
            
            # Add chunks to session manager
            chunk_ids = self.session_manager.add_chunks(document.document_id, processed_chunks)
            
            logger.info(f"Successfully processed document: {document.filename} with {len(chunk_ids)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {document.filename}: {e}")
            return False
    
    async def _extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        # Run in a thread pool to avoid blocking the event loop
        def extract():
            try:
                text = ""
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
                return text
            except Exception as e:
                logger.error(f"Error extracting text from {file_path}: {e}")
                return ""
        
        # Run synchronously in a separate thread
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, extract)
        return result
    
    async def _create_chunks(self, text: str, document_id: str) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split into chunks
            document_id: ID of the document
            
        Returns:
            List of Chunk objects
        """
        # Simple splitting strategy - can be improved with better chunking
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, finish the current chunk
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                # Create a new chunk
                chunks.append(Chunk(
                    document_id=document_id,
                    text_content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    metadata={"start_paragraph": chunk_index * self.max_chunk_size}
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append(Chunk(
                document_id=document_id,
                text_content=current_chunk.strip(),
                chunk_index=chunk_index,
                metadata={"start_paragraph": chunk_index * self.max_chunk_size}
            ))
        
        return chunks
    
    async def _process_chunks(self, chunks: List[Chunk], full_document: str) -> List[Chunk]:
        """
        Process chunks in parallel: add context and create embeddings.
        
        Args:
            chunks: List of Chunk objects to process
            full_document: Full text of the document for context generation
            
        Returns:
            List of processed Chunk objects
        """
        # Create tasks for processing each chunk
        tasks = []
        for chunk in chunks:
            tasks.append(self._process_chunk(chunk, full_document))
        
        # Process chunks in parallel
        processed_chunks = await asyncio.gather(*tasks)
        return [chunk for chunk in processed_chunks if chunk is not None]
    
    async def _process_chunk(self, chunk: Chunk, full_document: str) -> Optional[Chunk]:
        """
        Process a single chunk: add context and create embedding.
        
        Args:
            chunk: Chunk object to process
            full_document: Full text of the document for context generation
            
        Returns:
            Processed Chunk object or None if processing failed
        """
        try:
            # Add context to the chunk
            context = await self._generate_context(chunk.text_content, full_document)
            if context:
                chunk.add_context(context)
            else:
                # If context generation failed, use a simple context
                simple_context = f"This is chunk {chunk.chunk_index} from the document."
                chunk.add_context(simple_context)
            
            # Create embedding for the contextualized content
            embedding = await self.embedding_service.create_embedding(chunk.contextualized_content, "document")
            if embedding:
                chunk.add_embedding(embedding)
            else:
                logger.warning(f"Failed to create embedding for chunk {chunk.chunk_id}")
                return None
            
            return chunk
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
            return None
    
    async def _generate_context(self, chunk_text: str, full_document: str) -> Optional[str]:
        """
        Generate contextual information for a chunk using Anthropic's Claude.
        
        Args:
            chunk_text: Text content of the chunk
            full_document: Full text of the document
            
        Returns:
            Generated contextual text or None if generation failed
        """
        try:
            # Prepare the prompt for context generation
            # Following Anthropic's recommendation for contextualizing chunks
            prompt = f"""

{full_document[:10000]}...


Here is the chunk we want to situate within the whole document:

{chunk_text}


Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
            
            # Call Anthropic API to generate context
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and return the generated context
            return response.content[0].text.strip()
        
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return None