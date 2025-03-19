"""Session management for the Document QA Bot."""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


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
        return datetime.now() - self.last_activity > timedelta(hours=2)  # 2 hours inactivity
    
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