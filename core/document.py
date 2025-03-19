"""Document class definitions for the Document QA Bot."""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional


class Document:
    """Represents a document with associated metadata and processed chunks."""
    
    def __init__(self, file_path: str, filename: str, user_id: int):
        """
        Initialize a new document.
        
        Args:
            file_path: Path to the saved document
            filename: Original filename
            user_id: ID of the user who uploaded the document
        """
        self.document_id = str(uuid.uuid4())
        self.file_path = file_path
        self.filename = filename
        self.user_id = user_id
        self.upload_time = datetime.now()
        self.processed = False
        self.chunk_ids: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def mark_processed(self, chunk_ids: List[str]) -> None:
        """
        Mark document as processed and store chunk IDs.
        
        Args:
            chunk_ids: List of IDs for chunks created from this document
        """
        self.processed = True
        self.chunk_ids = chunk_ids
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "filename": self.filename,
            "user_id": self.user_id,
            "upload_time": self.upload_time.isoformat(),
            "processed": self.processed,
            "chunk_ids": self.chunk_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        doc = cls(data["file_path"], data["filename"], data["user_id"])
        doc.document_id = data["document_id"]
        doc.upload_time = datetime.fromisoformat(data["upload_time"])
        doc.processed = data["processed"]
        doc.chunk_ids = data["chunk_ids"]
        doc.metadata = data["metadata"]
        return doc