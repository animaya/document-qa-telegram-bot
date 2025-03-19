"""Session manager for handling user sessions, documents, and chunks."""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.session import Session
from core.document import Document
from core.chunk import Chunk

# Setup logging
logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions and associated data."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.sessions: Dict[int, Session] = {}  # Map of user_id to Session
        self.documents: Dict[str, Document] = {}  # Map of document_id to Document
        self.chunks: Dict[str, Chunk] = {}  # Map of chunk_id to Chunk
    
    def get_or_create_session(self, user_id: int) -> Session:
        """
        Get the user's session or create a new one if it doesn't exist.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Session object for the user
        """
        if user_id not in self.sessions or self.sessions[user_id].is_expired():
            self.sessions[user_id] = Session(user_id)
        return self.sessions[user_id]
    
    def add_document(self, file_path: str, filename: str, user_id: int) -> Document:
        """
        Add a new document and associate it with the user's session.
        
        Args:
            file_path: Path to the saved document
            filename: Original filename
            user_id: ID of the user who uploaded the document
            
        Returns:
            Document object for the uploaded document
        """
        document = Document(file_path, filename, user_id)
        self.documents[document.document_id] = document
        
        # Add document to user's session
        session = self.get_or_create_session(user_id)
        session.add_document(document.document_id)
        
        return document
    
    def add_chunks(self, document_id: str, chunks: List[Chunk]) -> List[str]:
        """
        Add chunks for a document.
        
        Args:
            document_id: ID of the parent document
            chunks: List of Chunk objects
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = []
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            chunk_ids.append(chunk.chunk_id)
        
        # Update document with chunk IDs
        if document_id in self.documents:
            self.documents[document_id].mark_processed(chunk_ids)
        
        return chunk_ids
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(document_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        return self.chunks.get(chunk_id)
    
    def get_session_documents(self, user_id: int) -> List[Document]:
        """Get all documents for a user's current session."""
        session = self.get_or_create_session(user_id)
        return [self.documents[doc_id] for doc_id in session.document_ids if doc_id in self.documents]
    
    def get_session_chunks(self, user_id: int) -> List[Chunk]:
        """Get all chunks for a user's current session."""
        documents = self.get_session_documents(user_id)
        chunks = []
        for doc in documents:
            for chunk_id in doc.chunk_ids:
                if chunk_id in self.chunks:
                    chunks.append(self.chunks[chunk_id])
        return chunks
    
    def cleanup_expired_sessions(self) -> None:
        """Remove expired sessions and their associated data."""
        expired_users = [user_id for user_id, session in self.sessions.items() if session.is_expired()]
        
        for user_id in expired_users:
            # Get document IDs for this session
            session = self.sessions[user_id]
            doc_ids = session.document_ids.copy()
            
            # Get chunk IDs for these documents
            chunk_ids = []
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    chunk_ids.extend(self.documents[doc_id].chunk_ids)
                    
                    # Delete document file
                    try:
                        os.remove(self.documents[doc_id].file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete document file: {e}")
                    
                    # Remove document
                    del self.documents[doc_id]
            
            # Remove chunks
            for chunk_id in chunk_ids:
                if chunk_id in self.chunks:
                    del self.chunks[chunk_id]
            
            # Remove session
            del self.sessions[user_id]
            
            logger.info(f"Cleaned up expired session for user {user_id}")