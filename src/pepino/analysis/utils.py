"""
Analysis utilities for common operations across analyzers.

Provides safe, error-resistant utility functions for common operations
like timestamp parsing, formatting, and calculations.
"""

import logging
from datetime import datetime
from typing import Optional, Union

logger = logging.getLogger(__name__)


def safe_timestamp_to_iso(timestamp: Union[str, datetime, None]) -> Optional[str]:
    """
    Safely convert timestamp to ISO format string.
    
    Handles cases where timestamp might be:
    - None (returns None)
    - Already a string (returns as-is)
    - A datetime object (converts to ISO format)
    - Any other type (attempts conversion)
    
    Args:
        timestamp: Timestamp value to convert
        
    Returns:
        ISO format string or None
    """
    if not timestamp:
        return None
    
    # If it's already a string, return as-is
    if isinstance(timestamp, str):
        return timestamp
    
    # If it's a datetime object, convert to ISO format
    if hasattr(timestamp, 'isoformat'):
        return timestamp.isoformat()
    
    # Fallback - try to convert string to datetime then back to ISO
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.isoformat()
    except (ValueError, AttributeError):
        pass
    
    # Last resort - convert to string
    return str(timestamp)


def safe_datetime_from_iso(
    timestamp: Union[str, datetime, None],
    default: Optional[datetime] = None,
    raise_on_error: bool = False
) -> Optional[datetime]:
    """
    Safely convert ISO timestamp string to datetime object with enhanced error handling.

    Args:
        timestamp: ISO timestamp string or datetime object
        default: Default value to return if conversion fails
        raise_on_error: If True, raise ValueError on parsing errors

    Returns:
        datetime object, default value, or None if conversion fails

    Raises:
        ValueError: If raise_on_error is True and parsing fails

    Examples:
        >>> safe_datetime_from_iso("2024-01-15T10:30:00Z")
        datetime.datetime(2024, 1, 15, 10, 30, tzinfo=datetime.timezone.utc)

        >>> safe_datetime_from_iso("invalid", default=datetime.now())
        datetime.datetime(...)  # Returns default
    """
    if not timestamp:
        return default

    if isinstance(timestamp, datetime):
        return timestamp

    if isinstance(timestamp, str):
        try:
            # Handle common timestamp formats
            if timestamp.endswith('Z'):
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(timestamp)
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug(f"Failed to parse timestamp '{timestamp}': {e}")
            if raise_on_error:
                raise ValueError(f"Invalid timestamp format: {timestamp}") from e
            return default

    return default


def format_duration_days(first_message: Optional[str], last_message: Optional[str]) -> int:
    """
    Calculate duration in days between first and last message.

    Args:
        first_message: ISO timestamp string of first message
        last_message: ISO timestamp string of last message

    Returns:
        Number of days between messages, or 0 if calculation fails
    """
    try:
        if not first_message or not last_message:
            return 0

        first_dt = safe_datetime_from_iso(first_message)
        last_dt = safe_datetime_from_iso(last_message)

        if first_dt and last_dt:
            return (last_dt - first_dt).days + 1

    except Exception as e:
        logger.debug(f"Failed to calculate duration: {e}")
        pass

    return 0


def calculate_time_delta(
    start: Optional[str],
    end: Optional[str],
    unit: str = "hours"
) -> float:
    """
    Calculate time difference between two ISO timestamp strings.

    Args:
        start: Start timestamp (ISO format)
        end: End timestamp (ISO format)
        unit: Unit of time to return ('seconds', 'minutes', 'hours', 'days')

    Returns:
        Time difference in specified unit, or 0.0 if calculation fails

    Examples:
        >>> calculate_time_delta("2024-01-15T10:00:00Z", "2024-01-15T12:00:00Z", "hours")
        2.0
    """
    start_dt = safe_datetime_from_iso(start)
    end_dt = safe_datetime_from_iso(end)

    if not start_dt or not end_dt:
        return 0.0

    try:
        delta = end_dt - start_dt
        seconds = delta.total_seconds()

        if unit == "seconds":
            return seconds
        elif unit == "minutes":
            return seconds / 60
        elif unit == "hours":
            return seconds / 3600
        elif unit == "days":
            return seconds / 86400
        else:
            logger.warning(f"Unknown time unit: {unit}, returning seconds")
            return seconds

    except (ValueError, AttributeError, TypeError) as e:
        logger.warning(f"Failed to calculate time delta: {e}")
        return 0.0 