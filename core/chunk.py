"""Chunk class definitions for Document QA Bot."""

import uuid
from typing import Dict, List, Any, Optional


class Chunk:
    """Represents a chunk of text from a document with contextual information."""
    
    def __init__(self, document_id: str, text_content: str, chunk_index: int, metadata: Dict[str, Any] = None):
        """
        Initialize a new chunk.
        
        Args:
            document_id: ID of the parent document
            text_content: Original text content
            chunk_index: Index of the chunk in the document
            metadata: Additional metadata about the chunk
        """
        self.chunk_id = str(uuid.uuid4())
        self.document_id = document_id
        self.text_content = text_content
        self.chunk_index = chunk_index
        self.contextual_text: Optional[str] = None
        self.contextualized_content: Optional[str] = None
        self.embedding: Optional[List[float]] = None
        self.metadata = metadata or {}
    
    def add_context(self, contextual_text: str) -> None:
        """
        Add contextual information to the chunk.
        
        Args:
            contextual_text: Generated context for the chunk
        """
        self.contextual_text = contextual_text
        self.contextualized_content = f"{contextual_text}\n\n{self.text_content}"
    
    def add_embedding(self, embedding: List[float]) -> None:
        """
        Add vector embedding to the chunk.
        
        Args:
            embedding: Vector representation of the contextualized content
        """
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text_content": self.text_content,
            "chunk_index": self.chunk_index,
            "contextual_text": self.contextual_text,
            "contextualized_content": self.contextualized_content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create chunk from dictionary."""
        chunk = cls(data["document_id"], data["text_content"], data["chunk_index"], data["metadata"])
        chunk.chunk_id = data["chunk_id"]
        chunk.contextual_text = data["contextual_text"]
        chunk.contextualized_content = data["contextualized_content"]
        return chunk