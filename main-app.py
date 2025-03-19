"""
Document QA Telegram Bot - Main Application

This is the main module to run the Document QA Telegram Bot, which allows users
to upload PDF documents and ask questions about them using Claude and Contextual RAG.

Usage:
    python main.py

Environment Variables:
    TELEGRAM_TOKEN: Your Telegram bot token
    ANTHROPIC_API_KEY: Your Anthropic API key
    GEMINI_API_KEY: Your Google Gemini API key (optional)
    VOYAGE_API_KEY: Your Voyage API key (optional)
    OPENAI_API_KEY: Your OpenAI API key (optional)
    COHERE_API_KEY: Your Cohere API key (optional)
    DATA_DIR: Directory to store data (default: "data")
    EMBEDDING_PROVIDER: Provider for embeddings (GEMINI, VOYAGE, OPENAI) (default: GEMINI)
    ANTHROPIC_MODEL: Anthropic model to use (default: claude-3-haiku-20240307)
    USE_RERANKING: Whether to use reranking (1 or 0) (default: 1)
"""

import os
import logging
import asyncio
import argparse
from typing import Optional
import signal

# Import local modules
from telegram_bot import DocumentQABot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define default values
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_PROVIDER = "GEMINI"
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"
DEFAULT_MAX_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_NUM_CHUNKS_RETRIEVE = 20
DEFAULT_NUM_RERANK = 150
DEFAULT_USE_RERANKING = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document QA Telegram Bot")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("DATA_DIR", DEFAULT_DATA_DIR),
        help="Directory to store data"
    )
    
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["GEMINI", "VOYAGE", "OPENAI"],
        default=os.environ.get("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
        help="Provider for embeddings"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        help="Anthropic model to use"
    )
    
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=int(os.environ.get("MAX_CHUNK_SIZE", DEFAULT_MAX_CHUNK_SIZE)),
        help="Maximum size of chunks in tokens"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.environ.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
        help="Overlap between chunks in tokens"
    )
    
    parser.add_argument(
        "--num-chunks-retrieve",
        type=int,
        default=int(os.environ.get("NUM_CHUNKS_RETRIEVE", DEFAULT_NUM_CHUNKS_RETRIEVE)),
        help="Number of chunks to retrieve"
    )
    
    parser.add_argument(
        "--num-rerank",
        type=int,
        default=int(os.environ.get("NUM_RERANK", DEFAULT_NUM_RERANK)),
        help="Number of chunks to consider for reranking"
    )
    
    parser.add_argument(
        "--use-reranking",
        action="store_true",
        default=os.environ.get("USE_RERANKING", "1") == "1",
        help="Whether to use reranking"
    )
    
    return parser.parse_args()

async def main():
    """Main function to run the bot."""
    # Parse arguments
    args = parse_args()
    
    # Check for required environment variables
    telegram_token = os.environ.get("TELEGRAM_TOKEN")
    if not telegram_token:
        logger.error("TELEGRAM_TOKEN environment variable is required")
        return
    
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is required")
        return
    
    # Check for embedding API keys based on provider
    if args.embedding_provider == "GEMINI" and not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY environment variable is not set, but GEMINI is selected as embedding provider")
    elif args.embedding_provider == "VOYAGE" and not os.environ.get("VOYAGE_API_KEY"):
        logger.warning("VOYAGE_API_KEY environment variable is not set, but VOYAGE is selected as embedding provider")
    elif args.embedding_provider == "OPENAI" and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable is not set, but OPENAI is selected as embedding provider")
    
    # Check for Cohere API key if reranking is enabled
    if args.use_reranking and not os.environ.get("COHERE_API_KEY"):
        logger.warning("COHERE_API_KEY environment variable is not set, but reranking is enabled")
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Initialize bot
    bot = DocumentQABot(
        telegram_token=telegram_token,
        anthropic_api_key=anthropic_api_key,
        data_dir=args.data_dir,
        embedding_provider=args.embedding_provider,
        model=args.model,
        max_chunk_size=args.max_chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_chunks_retrieve=args.num_chunks_retrieve,
        num_rerank=args.num_rerank,
        use_reranking=args.use_reranking
    )
    
    # Register signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(bot, loop)))
    
    # Start the bot
    logger.info("Starting Document QA Telegram Bot...")
    logger.info(f"Configuration:")
    logger.info(f"- Data directory: {args.data_dir}")
    logger.info(f"- Embedding provider: {args.embedding_provider}")
    logger.info(f"- Anthropic model: {args.model}")
    logger.info(f"- Max chunk size: {args.max_chunk_size}")
    logger.info(f"- Chunk overlap: {args.chunk_overlap}")
    logger.info(f"- Num chunks retrieve: {args.num_chunks_retrieve}")
    logger.info(f"- Num rerank: {args.num_rerank}")
    logger.info(f"- Use reranking: {args.use_reranking}")
    
    await bot.start()
    
    # Keep the event loop running
    while True:
        await asyncio.sleep(3600)  # Sleep for an hour

async def shutdown(bot: DocumentQABot, loop: asyncio.AbstractEventLoop):
    """Shutdown the bot gracefully."""
    logger.info("Shutting down...")
    await bot.stop()
    logger.info("Bot stopped")
    loop.stop()

if __name__ == "__main__":
    asyncio.run(main())