"""
Analysis utilities for common operations across analyzers.
"""

from datetime import datetime
from typing import Optional, Union


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


def safe_datetime_from_iso(timestamp: Union[str, datetime, None]) -> Optional[datetime]:
    """
    Safely convert ISO timestamp string to datetime object.
    
    Args:
        timestamp: ISO timestamp string
        
    Returns:
        datetime object or None
    """
    if not timestamp:
        return None
        
    if isinstance(timestamp, datetime):
        return timestamp
        
    if isinstance(timestamp, str):
        try:
            # Handle common timestamp formats
            if timestamp.endswith('Z'):
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(timestamp)
        except ValueError:
            pass
    
    return None


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
            
    except Exception:
        pass
        
    return 0 