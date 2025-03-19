"""
Document QA Telegram Bot

This module implements a Telegram bot that processes PDF documents and answers
questions about them using Anthropic's Claude Haiku model and advanced
Retrieval-Augmented Generation (RAG) techniques.

The system uses a workflow based on Anthropic's recommendations for agent design:
- Augmented LLM for the QA component
- Orchestrator-workers for the overall architecture
- Prompt chaining for document processing
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_SESSION_IDLE_TIME = timedelta(hours=2)  # Sessions expire after 2 hours of inactivity
DATA_DIR = os.environ.get("DATA_DIR", "data")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "GEMINI")  # Options: GEMINI, VOYAGE, OPENAI
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
MAX_CHUNK_SIZE = 500  # Maximum tokens per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks
NUM_CHUNKS_RETRIEVE = 20  # Number of chunks to retrieve
NUM_RERANK = 150  # Number of chunks to rerank

# Ensure data directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "documents"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "sessions"), exist_ok=True)

class Session:
    """Represents a user session with associated documents and conversation history."""
    
    def __init__(self, user_id: int):
        """
        Initialize a new session for a user.
        
        Args:
            user_id: Telegram user ID
        """
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id
        self.start_time = datetime.now()
        self.last_activity = datetime.now()
        self.document_ids: List[str] = []
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_document(self, document_id: str) -> None:
        """
        Add a document to the session.
        
        Args:
            document_id: ID of the document to add
        """
        if document_id not in self.document_ids:
            self.document_ids.append(document_id)
        self.update_activity()
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message sender ('user' or 'assistant')
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.update_activity()
    
    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if the session has expired due to inactivity."""
        return datetime.now() - self.last_activity > MAX_SESSION_IDLE_TIME
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "document_ids": self.document_ids,
            "conversation_history": self.conversation_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create session from dictionary."""
        session = cls(data["user_id"])
        session.session_id = data["session_id"]
        session.start_time = datetime.fromisoformat(data["start_time"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.document_ids = data["document_ids"]
        session.conversation_history = data["conversation_history"]
        return session


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