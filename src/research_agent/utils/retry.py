"""
Async retry utilities with exponential backoff.

Provides retry_with_backoff for resilient HTTP calls across all tool modules.
"""

import asyncio
import logging
import random
from typing import Callable

import httpx

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple = (429, 503, 504)
):
    """
    Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute (should return httpx.Response)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to prevent thundering herd
        retry_on: HTTP status codes to retry on

    Returns:
        The response from the function

    Raises:
        httpx.HTTPError: If all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = await func()

            # Check if we should retry based on status code
            if response.status_code in retry_on:
                if attempt < max_retries:
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    # Check for Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass

                    logger.warning(
                        f"Request failed with {response.status_code}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    response.raise_for_status()

            return response

        except httpx.HTTPError as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                if jitter:
                    delay = delay * (0.5 + random.random())
                logger.warning(
                    f"Request error: {e}, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
            else:
                raise

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")
