"""
Input validation models for analysis requests.

Provides Pydantic-based validation to ensure data integrity and security
before analysis operations are performed.
"""

import re
from typing import Optional
from pydantic import BaseModel, Field, validator


class UserAnalysisRequest(BaseModel):
    """
    Validated request for user analysis.

    Ensures username format is valid and parameters are within acceptable ranges.
    """
    username: str = Field(..., min_length=1, max_length=100, description="Discord username or display name")
    days: Optional[int] = Field(None, ge=1, le=365, description="Number of days to analyze")
    include_patterns: bool = Field(True, description="Include time pattern analysis")
    include_semantic: bool = Field(True, description="Include semantic/topic analysis")

    @validator('username')
    def validate_username(cls, v):
        """Validate username format and sanitize input."""
        if not v or not v.strip():
            raise ValueError('Username cannot be empty')

        # Remove leading/trailing whitespace
        v = v.strip()

        # Check for SQL injection attempts
        if any(pattern in v.lower() for pattern in [';', '--', '/*', '*/', 'drop ', 'delete ', 'insert ', 'update ']):
            raise ValueError('Invalid characters in username')

        return v

    @validator('days')
    def validate_days(cls, v):
        """Validate days parameter."""
        if v is not None and v < 1:
            raise ValueError('days must be at least 1')
        return v


class ChannelAnalysisRequest(BaseModel):
    """
    Validated request for channel analysis.

    Ensures channel name is valid and parameters are within acceptable ranges.
    """
    channel_name: str = Field(..., min_length=1, max_length=100, description="Discord channel name")
    days: Optional[int] = Field(None, ge=1, le=365, description="Number of days to analyze")
    include_topics: bool = Field(False, description="Include topic analysis")
    include_engagement: bool = Field(True, description="Include engagement metrics")
    include_health: bool = Field(True, description="Include health metrics")

    @validator('channel_name')
    def validate_channel_name(cls, v):
        """Validate channel name format and sanitize input."""
        if not v or not v.strip():
            raise ValueError('Channel name cannot be empty')

        # Remove leading/trailing whitespace
        v = v.strip()

        # Check for SQL injection attempts
        if any(pattern in v.lower() for pattern in [';', '--', '/*', '*/', 'drop ', 'delete ', 'insert ', 'update ']):
            raise ValueError('Invalid characters in channel name')

        # Discord channel names have specific format
        # They can contain letters, numbers, hyphens, and underscores
        # But we'll be more permissive to handle display names
        if len(v) > 100:
            raise ValueError('Channel name too long (max 100 characters)')

        return v

    @validator('days')
    def validate_days(cls, v):
        """Validate days parameter."""
        if v is not None and v < 1:
            raise ValueError('days must be at least 1')
        return v


class TopicAnalysisRequest(BaseModel):
    """
    Validated request for topic analysis.

    Ensures parameters are valid for topic extraction.
    """
    channel_name: Optional[str] = Field(None, max_length=100, description="Channel name to analyze")
    days: Optional[int] = Field(None, ge=1, le=365, description="Number of days to analyze")
    limit: int = Field(1000, ge=10, le=10000, description="Maximum messages to analyze")
    min_topic_size: int = Field(5, ge=2, le=50, description="Minimum messages per topic")

    @validator('channel_name')
    def validate_channel_name(cls, v):
        """Validate channel name if provided."""
        if v is None:
            return v

        if not v.strip():
            raise ValueError('Channel name cannot be empty string')

        v = v.strip()

        # Check for SQL injection attempts
        if any(pattern in v.lower() for pattern in [';', '--', '/*', '*/', 'drop ', 'delete ']):
            raise ValueError('Invalid characters in channel name')

        return v

    @validator('limit')
    def validate_limit(cls, v):
        """Ensure limit is reasonable."""
        if v < 10:
            raise ValueError('Limit too small (minimum 10)')
        if v > 10000:
            raise ValueError('Limit too large (maximum 10000)')
        return v


class TemporalAnalysisRequest(BaseModel):
    """
    Validated request for temporal analysis.

    Ensures parameters are valid for time-based analysis.
    """
    channel_name: Optional[str] = Field(None, max_length=100, description="Channel name to analyze")
    user_name: Optional[str] = Field(None, max_length=100, description="User name to analyze")
    days: Optional[int] = Field(None, ge=1, le=365, description="Number of days to analyze")
    granularity: str = Field("day", description="Time granularity (hour/day/week)")

    @validator('granularity')
    def validate_granularity(cls, v):
        """Validate granularity parameter."""
        valid_values = ['hour', 'day', 'week']
        if v not in valid_values:
            raise ValueError(f'granularity must be one of: {", ".join(valid_values)}')
        return v

    @validator('channel_name', 'user_name')
    def validate_name(cls, v):
        """Validate channel/user name if provided."""
        if v is None:
            return v

        if not v.strip():
            raise ValueError('Name cannot be empty string')

        v = v.strip()

        # Check for SQL injection attempts
        if any(pattern in v.lower() for pattern in [';', '--', '/*', '*/']):
            raise ValueError('Invalid characters in name')

        return v


def validate_user_analysis_input(username: str, days: Optional[int] = None, **kwargs) -> UserAnalysisRequest:
    """
    Convenience function to validate user analysis input.

    Args:
        username: Username to analyze
        days: Optional days back to analyze
        **kwargs: Additional parameters

    Returns:
        Validated UserAnalysisRequest

    Raises:
        ValueError: If validation fails
    """
    return UserAnalysisRequest(username=username, days=days, **kwargs)


def validate_channel_analysis_input(channel_name: str, days: Optional[int] = None, **kwargs) -> ChannelAnalysisRequest:
    """
    Convenience function to validate channel analysis input.

    Args:
        channel_name: Channel name to analyze
        days: Optional days back to analyze
        **kwargs: Additional parameters

    Returns:
        Validated ChannelAnalysisRequest

    Raises:
        ValueError: If validation fails
    """
    return ChannelAnalysisRequest(channel_name=channel_name, days=days, **kwargs)
