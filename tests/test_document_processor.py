"""Tests for the document processor module."""

import unittest
import asyncio
import os
import tempfile
from unittest.mock import MagicMock, patch

from core.session_manager import SessionManager
from services.document_processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for the DocumentProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.session_manager = SessionManager()
        self.anthropic_api_key = "test_key"
        self.processor = DocumentProcessor(
            session_manager=self.session_manager,
            anthropic_api_key=self.anthropic_api_key,
            embedding_provider="OPENAI",  # Use OpenAI for testing as it's more common
            model="claude-3-haiku-20240307",
            max_chunk_size=200,
            chunk_overlap=50
        )
    
    @patch('document_qa_telegram_bot.services.embedding_service.EmbeddingService.create_embedding')
    @patch('document_qa_telegram_bot.services.document_processor.DocumentProcessor._generate_context')
    @patch('document_qa_telegram_bot.services.document_processor.DocumentProcessor._extract_text')
    def test_process_document(self, mock_extract_text, mock_generate_context, mock_create_embedding):
        """Test document processing workflow."""
        # Mock the extraction to return some text
        mock_extract_text.return_value = asyncio.Future()
        mock_extract_text.return_value.set_result("This is a test document.\n\nIt has multiple paragraphs.\n\nEach paragraph should be a chunk.")
        
        # Mock the context generation
        mock_generate_context.return_value = asyncio.Future()
        mock_generate_context.return_value.set_result("Context for the chunk.")
        
        # Mock the embedding creation
        mock_create_embedding.return_value = asyncio.Future()
        mock_create_embedding.return_value.set_result([0.1, 0.2, 0.3])
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Create a test document
            document = self.session_manager.add_document(
                file_path=temp_file_path,
                filename="test.pdf",
                user_id=123
            )
            
            # Process the document
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.processor.process_document(document))
            
            # Check the result
            self.assertTrue(result)
            self.assertTrue(document.processed)
            self.assertTrue(len(document.chunk_ids) > 0)
            
            # Verify calls
            mock_extract_text.assert_called_once_with(temp_file_path)
            mock_generate_context.assert_called()
            mock_create_embedding.assert_called()
            
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


if __name__ == '__main__':
    unittest.main()