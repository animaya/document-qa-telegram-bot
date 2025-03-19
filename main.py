"""
Document QA Telegram Bot - Main Application

This is the main module to run the Document QA Telegram Bot, which allows users
to upload PDF documents and ask questions about them using Claude and Contextual RAG.

Usage:
    python -m document_qa_bot.main

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
import signal
import argparse
from typing import Optional

# Import local modules
from document_qa_bot.bot.telegram_bot import DocumentQABot
from document_qa_bot.config.settings import get_settings, validate_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document QA Telegram Bot")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory to store data (overrides DATA_DIR environment variable)"
    )
    
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["GEMINI", "VOYAGE", "OPENAI"],
        help="Provider for embeddings (overrides EMBEDDING_PROVIDER environment variable)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Anthropic model to use (overrides ANTHROPIC_MODEL environment variable)"
    )
    
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        help="Maximum size of chunks in tokens (overrides MAX_CHUNK_SIZE environment variable)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Overlap between chunks in tokens (overrides CHUNK_OVERLAP environment variable)"
    )
    
    parser.add_argument(
        "--num-chunks-retrieve",
        type=int,
        help="Number of chunks to retrieve (overrides NUM_CHUNKS_RETRIEVE environment variable)"
    )
    
    parser.add_argument(
        "--num-rerank",
        type=int,
        help="Number of chunks to consider for reranking (overrides NUM_RERANK environment variable)"
    )
    
    parser.add_argument(
        "--use-reranking",
        action="store_true",
        default=None,
        help="Whether to use reranking (overrides USE_RERANKING environment variable)"
    )
    
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable reranking"
    )
    
    return parser.parse_args()


async def main():
    """Main function to run the bot."""
    # Parse arguments
    args = parse_args()
    
    # Get settings from environment variables
    settings = get_settings()
    
    # Override settings with command line arguments
    if args.data_dir:
        settings["data_dir"] = args.data_dir
    if args.embedding_provider:
        settings["embedding_provider"] = args.embedding_provider
    if args.model:
        settings["model"] = args.model
    if args.max_chunk_size:
        settings["max_chunk_size"] = args.max_chunk_size
    if args.chunk_overlap:
        settings["chunk_overlap"] = args.chunk_overlap
    if args.num_chunks_retrieve:
        settings["num_chunks_retrieve"] = args.num_chunks_retrieve
    if args.num_rerank:
        settings["num_rerank"] = args.num_rerank
    if args.use_reranking is not None:
        settings["use_reranking"] = args.use_reranking
    if args.no_reranking:
        settings["use_reranking"] = False
    
    # Validate settings
    validate_settings(settings)
    
    # Create data directory if it doesn't exist
    os.makedirs(settings["data_dir"], exist_ok=True)
    os.makedirs(os.path.join(settings["data_dir"], "documents"), exist_ok=True)
    
    # Check for required settings
    if not settings.get("telegram_token"):
        logger.error("TELEGRAM_TOKEN environment variable is required")
        return
    
    if not settings.get("anthropic_api_key"):
        logger.error("ANTHROPIC_API_KEY environment variable is required")
        return
    
    # Initialize bot
    bot = DocumentQABot(
        telegram_token=settings["telegram_token"],
        anthropic_api_key=settings["anthropic_api_key"],
        data_dir=settings["data_dir"],
        embedding_provider=settings["embedding_provider"],
        model=settings["model"],
        max_chunk_size=settings["max_chunk_size"],
        chunk_overlap=settings["chunk_overlap"],
        num_chunks_retrieve=settings["num_chunks_retrieve"],
        num_rerank=settings["num_rerank"],
        use_reranking=settings["use_reranking"]
    )
    
    # Register signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(bot, loop)))
    
    # Start the bot
    logger.info("Starting Document QA Telegram Bot...")
    logger.info(f"Configuration:")
    logger.info(f"- Data directory: {settings['data_dir']}")
    logger.info(f"- Embedding provider: {settings['embedding_provider']}")
    logger.info(f"- Anthropic model: {settings['model']}")
    logger.info(f"- Max chunk size: {settings['max_chunk_size']}")
    logger.info(f"- Chunk overlap: {settings['chunk_overlap']}")
    logger.info(f"- Num chunks retrieve: {settings['num_chunks_retrieve']}")
    logger.info(f"- Num rerank: {settings['num_rerank']}")
    logger.info(f"- Use reranking: {settings['use_reranking']}")
    
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