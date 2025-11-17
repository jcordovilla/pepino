"""
Retry utilities for handling transient failures.

Provides decorators and utilities for implementing exponential backoff retry logic
for database operations and other potentially failing operations.
"""

import logging
import sqlite3
import time
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_on_error(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (sqlite3.OperationalError, sqlite3.DatabaseError),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on specific exceptions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        exceptions: Tuple of exception types to catch and retry (default: SQLite errors)
        on_retry: Optional callback function(exception, attempt_number) called before each retry

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_on_error(max_attempts=3, backoff_factor=2.0)
        ... def query_database(query):
        ...     return execute_query(query)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    # If this is the last attempt, don't retry
                    if attempt >= max_attempts - 1:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        break

                    # Calculate delay with exponential backoff
                    delay = initial_delay * (backoff_factor ** attempt)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")

                    # Wait before retrying
                    time.sleep(delay)

                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(
                        f"Unexpected exception in {func.__name__} (not retrying): {e}"
                    )
                    raise

            # If we get here, all retries failed
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"Function {func.__name__} failed without raising an exception")

        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (sqlite3.OperationalError,)
):
    """
    Async version of retry decorator.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        initial_delay: Initial delay in seconds
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated async function with retry logic
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio

            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt >= max_attempts - 1:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        break

                    delay = initial_delay * (backoff_factor ** attempt)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(
                        f"Unexpected exception in {func.__name__} (not retrying): {e}"
                    )
                    raise

            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"Function {func.__name__} failed without raising an exception")

        return wrapper
    return decorator


class RetryConfig:
    """
    Configuration for retry behavior.

    Allows for global retry configuration that can be applied to multiple operations.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)

    def should_retry(self, attempt: int) -> bool:
        """Check if we should retry for the given attempt number."""
        return attempt < self.max_attempts


# Default retry configuration for database operations
DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    backoff_factor=2.0,
    initial_delay=1.0,
    max_delay=30.0
)


# Default retry configuration for API calls
API_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    backoff_factor=1.5,
    initial_delay=0.5,
    max_delay=10.0
)


def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception is retryable.

    Args:
        exception: Exception to check

    Returns:
        True if the exception is likely transient and retryable
    """
    # SQLite errors that are typically transient
    retryable_sqlite_errors = (
        sqlite3.OperationalError,  # Database locked, etc.
        sqlite3.DatabaseError,      # General database errors
    )

    if isinstance(exception, retryable_sqlite_errors):
        error_msg = str(exception).lower()
        # These specific errors are retryable
        retryable_messages = [
            'database is locked',
            'disk i/o error',
            'attempt to write a readonly database',
        ]
        return any(msg in error_msg for msg in retryable_messages)

    return False


def log_retry_attempt(exception: Exception, attempt: int):
    """
    Default callback for logging retry attempts.

    Args:
        exception: The exception that triggered the retry
        attempt: Current attempt number
    """
    logger.info(f"Retry attempt {attempt} due to: {type(exception).__name__}: {exception}")
