"""
Telegram Bot Interface

This module provides a Telegram bot interface for the Document QA system,
allowing users to upload documents and ask questions about them.

The implementation follows Anthropic's Orchestrator-Workers architecture,
coordinating the various components for document processing and query answering.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import tempfile
from datetime import datetime
import json

from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Import local modules
from core.session_manager import SessionManager
from services.document_processor import DocumentProcessor
from services.contextual_rag import ContextualRAG

# Setup logging
logger = logging.getLogger(__name__)


class DocumentQABot:
    """
    Telegram bot for document QA using Anthropic's Claude and Contextual RAG.
    """
    
    def __init__(
    self,
    telegram_token: str,
    anthropic_api_key: str,
    data_dir: str = "data",
    embedding_provider: str = "GEMINI",
    model: str = "claude-3-haiku-20240307",
    max_chunk_size: int = 500,
    chunk_overlap: int = 100,
    num_chunks_retrieve: int = 20,
    num_rerank: int = 150,
    use_reranking: bool = True,
    batch_size: int = 3,
    anthropic_rate: float = 0.5,
    embedding_rate: float = 1.0,
    connection_timeout: float = 30.0,  # Add this parameter
    connection_retries: int = 3        # Add this parameter
):
        """
        Initialize the Document QA bot.
        
        Args:
            telegram_token: Telegram bot token
            anthropic_api_key: Anthropic API key
            data_dir: Directory to store data
            embedding_provider: Provider for embeddings (GEMINI, VOYAGE, OPENAI)
            model: Anthropic model to use
            max_chunk_size: Maximum size of chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            num_chunks_retrieve: Number of chunks to retrieve
            num_rerank: Number of chunks to consider for reranking
            use_reranking: Whether to use reranking
        """
        self.telegram_token = telegram_token
        self.anthropic_api_key = anthropic_api_key
        self.data_dir = data_dir
        self.embedding_provider = embedding_provider
        self.model = model
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_chunks_retrieve = num_chunks_retrieve
        self.num_rerank = num_rerank
        self.use_reranking = use_reranking
        self.batch_size = batch_size
        self.connection_timeout = connection_timeout
        self.connection_retries = connection_retries
        
        # Initialize components
        self.session_manager = SessionManager()
        self.document_processor = DocumentProcessor(
              session_manager=self.session_manager,
              anthropic_api_key=anthropic_api_key,
              embedding_provider=embedding_provider,
              model=model,
              max_chunk_size=max_chunk_size,
              chunk_overlap=chunk_overlap,
              batch_size=batch_size,
              anthropic_rate=anthropic_rate,
    
        )
        self.contextual_rag = ContextualRAG(
            session_manager=self.session_manager,
            anthropic_api_key=anthropic_api_key,
            embedding_provider=embedding_provider,
            model=model,
            num_chunks_retrieve=num_chunks_retrieve,
            num_rerank=num_rerank,
            use_reranking=use_reranking
        )
        
        # Initialize processing status
        self.processing_status = {}  # Map of document_id to processing status
        
        # Initialize the Telegram application
        self.application = (Application.builder()
        .token(telegram_token)
        .connect_timeout(self.connection_timeout)
        .read_timeout(self.connection_timeout)
        .write_timeout(self.connection_timeout)
        .build())
        self.max_retries = connection_retries
        # Register handlers
        self.register_handlers()
        
        # Schedule periodic cleanup
        self.application.job_queue.run_repeating(self.cleanup_expired_sessions, interval=3600, first=3600)
    
    def register_handlers(self) -> None:
        """Register message handlers for the Telegram bot."""
        # Command handlers
      
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("reset", self.reset_command))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.Document.PDF, self.handle_document))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    async def start(self) -> None:
        """Start the bot with retry logic."""
        logger.info("Starting the bot...")
        
        retry_count = 0
    
        while retry_count < self.max_retries:
            try:
                await self.application.initialize()
                await self.application.start()
                await self.application.updater.start_polling()
                logger.info("Bot started successfully!")
                return
            except telegram.error.TimedOut as e:
                retry_count += 1
                wait_time = 5 * retry_count  # Incremental backoff
                logger.warning(f"Connection timed out. Retrying in {wait_time} seconds (attempt {retry_count}/{self.max_retries})...")
                logger.debug(f"Error details: {str(e)}")
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Failed to start bot: {str(e)}")
                raise
        logger.error("Failed to start the bot after multiple attempts. Please check your network connection and Telegram token.")
        raise ConnectionError("Could not connect to Telegram API after multiple attempts")
    
    async def stop(self) -> None:
        """Stop the bot."""
        logger.info("Stopping the bot...")
        await self.application.stop()
        await self.application.updater.stop()
        await self.application.shutdown()
        logger.info("Bot stopped!")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        # Get or create session for this user
        user_id = update.effective_user.id
        self.session_manager.get_or_create_session(user_id)
        
        # Send welcome message
        welcome_text = (
            "👋 Welcome to the Document QA Bot!\n\n"
            "I can help you get answers from your PDF documents. Here's how to use me:\n\n"
            "1. 📄 Upload one or more PDF documents\n"
            "2. ⏳ Wait for me to process them (I'll let you know when it's done)\n"
            "3. ❓ Ask me any questions about the content of your documents\n\n"
            "To get started, simply upload a PDF document."
        )
        await update.message.reply_text(welcome_text)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = (
            "📚 Document QA Bot Help 📚\n\n"
            "Commands:\n"
            "/start - Start the bot and create a new session\n"
            "/help - Show this help message\n"
            "/status - Check the status of your current session\n"
            "/reset - Reset your session and delete all associated documents\n\n"
            "How to use:\n"
            "1. Upload PDF documents (multiple documents supported)\n"
            "2. Wait for processing to complete\n"
            "3. Ask questions about the documents\n\n"
            "Tips:\n"
            "- For best results, ask specific questions\n"
            "- The bot will search across all uploaded documents\n"
            "- Your session and documents will be automatically deleted after 2 hours of inactivity\n"
        )
        await update.message.reply_text(help_text)
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        user_id = update.effective_user.id
        session = self.session_manager.get_or_create_session(user_id)
        
        # Get documents for this session
        documents = self.session_manager.get_session_documents(user_id)
        
        if not documents:
            status_text = "You haven't uploaded any documents in this session yet."
            await update.message.reply_text(status_text)
            return
        
        # Create status message
        status_text = f"📊 Session Status\n\nActive documents: {len(documents)}\n\n"
        
        for i, doc in enumerate(documents, 1):
            status = "✅ Processed" if doc.processed else "⏳ Processing..."
            status_text += f"{i}. {doc.filename} - {status}\n"
        
        status_text += f"\nSession will expire after 2 hours of inactivity."
        
        await update.message.reply_text(status_text)
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /reset command."""
        user_id = update.effective_user.id
        
        # Get current session
        session = self.session_manager.sessions.get(user_id)
        if not session:
            await update.message.reply_text("No active session to reset.")
            return
        
        # Clean up session data
        self.session_manager.cleanup_expired_sessions()
        
        # Create a new session
        self.session_manager.get_or_create_session(user_id)
        
        # Clear indices for this user
        if user_id in self.contextual_rag.user_indices:
            del self.contextual_rag.user_indices[user_id]
        
        await update.message.reply_text("Your session has been reset. All documents have been deleted. You can upload new documents now.")
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle PDF document uploads."""
        user_id = update.effective_user.id
        document = update.message.document
        
        # Check if document is a PDF
        if not document.file_name.lower().endswith('.pdf'):
            await update.message.reply_text("Please upload a PDF document.")
            return
        
        # Send a "processing" message
        processing_message = await update.message.reply_text(f"Receiving document: {document.file_name}...")
        
        try:
            # Download the document
            new_file = await context.bot.get_file(document.file_id)
            
            # Create directory for user if it doesn't exist
            user_dir = os.path.join(self.data_dir, "documents", str(user_id))
            os.makedirs(user_dir, exist_ok=True)
            
            # Generate a filename for the downloaded document
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = ''.join(c if c.isalnum() else '_' for c in document.file_name)
            file_path = os.path.join(user_dir, f"{timestamp}_{safe_filename}")
            
            # Download the file
            await new_file.download_to_drive(custom_path=file_path)
            
            # Update processing message
            await processing_message.edit_text(f"Received document: {document.file_name}\nNow processing...")
            
            # Add document to session
            doc = self.session_manager.add_document(file_path, document.file_name, user_id)
            
            # Process document in the background
            asyncio.create_task(self._process_document(doc, processing_message))
            
        except Exception as e:
            logger.error(f"Error handling document: {e}")
            await processing_message.edit_text(f"Error processing document: {str(e)}")
    
    async def _process_document(self, document: Any, message: Any) -> None:
        """
        Process a document in the background.
        
        Args:
            document: Document object to process
            message: Telegram message to update with status
        """
        try:
            # Process the document
            success = await self.document_processor.process_document(document)
            
            if success:
                # Update processing message
                await message.edit_text(f"✅ Document processed successfully: {document.filename}\n\nYou can now ask questions about the content.")
                
                # Prepare indices for the user
                user_id = document.user_id
                await self.contextual_rag.prepare_indices(user_id)
            else:
                # Update processing message with error
                await message.edit_text(f"⚠️ Failed to process document: {document.filename}\n\nPlease try uploading the document again.")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            await message.edit_text(f"⚠️ Error processing document: {document.filename}\n\nError: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages as questions."""
        user_id = update.effective_user.id
        query_text = update.message.text
        
        # Get session
        session = self.session_manager.get_or_create_session(user_id)
        
        # Add message to conversation history
        session.add_message("user", query_text)
        
        # Check if user has uploaded documents
        documents = self.session_manager.get_session_documents(user_id)
        if not documents:
            await update.message.reply_text(
                "You haven't uploaded any documents yet. Please upload a PDF document first."
            )
            return
        
        # Check if all documents are processed
        unprocessed = [doc for doc in documents if not doc.processed]
        if unprocessed:
            await update.message.reply_text(
                "Some documents are still being processed. Please wait for processing to complete before asking questions."
            )
            return
        
        # Send "thinking" message
        thinking_message = await update.message.reply_text("Thinking...")
        
        try:
            # Query the system
            retrieved_chunks, answer = await self.contextual_rag.query(user_id, query_text)
            
            # Add assistant response to conversation history
            session.add_message("assistant", answer)
            
            # Update thinking message with answer
            await thinking_message.edit_text(answer)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await thinking_message.edit_text(
                "I'm sorry, but I encountered an error while processing your question. Please try again."
            )
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the Telegram bot."""
        logger.error(f"Error: {context.error}")
        
        if update and isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "I'm sorry, but an error occurred. Please try again later."
            )
    
    async def cleanup_expired_sessions(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Periodically clean up expired sessions."""
        self.session_manager.cleanup_expired_sessions()
        logger.info("Cleaned up expired sessions")

    ## Directory Structure
