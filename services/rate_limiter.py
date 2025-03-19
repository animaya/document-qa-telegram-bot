"""
Rate limiter service for API calls to external services.

This module provides a token bucket rate limiter implementation to control
the frequency of API calls and avoid rate limit errors.
"""

import time
import asyncio
import logging
import random
from typing import Callable, Any, Optional, TypeVar, Awaitable, Dict

# Setup logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

class TokenBucketRateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, rate: float, capacity: int = 60, initial_tokens: Optional[int] = None):
        """
        Initialize a token bucket rate limiter.
        
        Args:
            rate: Tokens per second to add to the bucket
            capacity: Maximum bucket size
            initial_tokens: Initial number of tokens (defaults to capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity if initial_tokens is None else initial_tokens
        self.last_time = time.time()
        self.lock = asyncio.Lock()
        
        logger.info(f"Initialized rate limiter with rate={rate} tokens/sec, capacity={capacity}")
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time spent waiting
        """
        async with self.lock:
            # Update tokens based on elapsed time
            now = time.time()
            elapsed = now - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = now
            
            # If not enough tokens, calculate wait time
            wait_time = 0
            if tokens > self.tokens:
                # How much time to wait to get required tokens
                wait_time = (tokens - self.tokens) / self.rate
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {tokens} tokens")
                
            # If wait time > 0, wait and update
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.tokens = max(0, self.capacity - tokens)
                self.last_time = time.time()
            else:
                # If no wait, just consume tokens
                self.tokens -= tokens
            
            return wait_time


class RateLimitedClient:
    """Wrapper for API clients to add rate limiting."""
    
    def __init__(self, name: str, rate: float = 0.5, capacity: int = 10):
        """
        Initialize a rate-limited client wrapper.
        
        Args:
            name: Name of the client for logging
            rate: Tokens per second to add to the bucket
            capacity: Maximum bucket size
        """
        self.name = name
        self.rate_limiter = TokenBucketRateLimiter(rate=rate, capacity=capacity)
        self.service_limiters: Dict[str, TokenBucketRateLimiter] = {}
        logger.info(f"Initialized rate-limited client for {name}")
    
    def add_service_limiter(self, service: str, rate: float, capacity: int = 10):
        """Add a rate limiter for a specific service endpoint."""
        self.service_limiters[service] = TokenBucketRateLimiter(rate=rate, capacity=capacity)
    
    async def call_with_retry(
        self, 
        func: Callable[..., Awaitable[T]],
        *args: Any,
        service: str = "",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        jitter_factor: float = 0.1,
        **kwargs: Any
    ) -> T:
        """
        Call a function with rate limiting and exponential backoff.
        
        Args:
            func: Async function to call
            *args: Arguments to pass to the function
            service: Specific service endpoint (uses specific limiter if provided)
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff delay in seconds
            jitter_factor: Random jitter multiplier for backoff
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If all retries fail
        """
        # Determine which rate limiter to use
        limiter = self.service_limiters.get(service, self.rate_limiter)
        
        # Acquire token from rate limiter
        await limiter.acquire()
        
        # Try the call with exponential backoff for retries
        backoff = initial_backoff
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Check if it's a rate limit error
                is_rate_limit = (
                    "429" in str(e) or 
                    "Too Many Requests" in str(e) or 
                    "rate_limit_error" in str(e)
                )
                
                if not is_rate_limit and retries > 0:
                    # If it's not a rate limit error, re-raise after first attempt
                    raise
                
                retries += 1
                if retries > max_retries:
                    logger.warning(f"Failed after {max_retries} retries: {str(e)}")
                    raise
                
                # Calculate backoff with jitter
                jitter = random.uniform(0, jitter_factor * backoff)
                delay = backoff + jitter
                
                logger.warning(f"{self.name} call failed (attempt {retries}/{max_retries}): {str(e)}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
                
                # Increase backoff for next attempt
                backoff *= 2
        
        # Should never reach here
        assert last_exception is not None
        raise last_exception