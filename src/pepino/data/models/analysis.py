"""
Analysis data models using Pydantic for validation and serialization.
Models for analysis tables that support NLP, embeddings, and statistics.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class MessageEmbedding(BaseModel):
    """Represents a message embedding for similarity analysis."""

    message_id: str = Field(..., description="Message ID")
    embedding: bytes = Field(..., description="Embedding vector as bytes")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class MessageTopic(BaseModel):
    """Represents a topic assignment for a message."""

    message_id: str = Field(..., description="Message ID")
    topic_id: int = Field(..., description="Topic ID")
    confidence: float = Field(..., description="Confidence score")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1."""
        return max(0.0, min(1.0, v))

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class WordFrequency(BaseModel):
    """Represents word frequency statistics for a channel."""

    word: str = Field(..., description="Word")
    channel_id: str = Field(..., description="Channel ID")
    frequency: int = Field(..., description="Word frequency")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, v):
        """Ensure frequency is non-negative."""
        return max(0, v)

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class UserStatistics(BaseModel):
    """Represents user statistics for a channel."""

    user_id: str = Field(..., description="User ID")
    channel_id: str = Field(..., description="Channel ID")
    message_count: int = Field(..., description="Message count")
    avg_message_length: float = Field(..., description="Average message length")
    active_hours: str = Field(..., description="Active hours as JSON string")
    last_active: datetime = Field(..., description="Last activity timestamp")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )

    @field_validator("message_count")
    @classmethod
    def validate_message_count(cls, v):
        """Ensure message count is non-negative."""
        return max(0, v)

    @field_validator("avg_message_length")
    @classmethod
    def validate_avg_length(cls, v):
        """Ensure average length is non-negative."""
        return max(0.0, v)

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class ConversationChain(BaseModel):
    """Represents a conversation chain/thread."""

    chain_id: int = Field(..., description="Chain ID")
    root_message_id: str = Field(..., description="Root message ID")
    last_message_id: str = Field(..., description="Last message ID")
    message_count: int = Field(..., description="Messages in chain")
    created_at: datetime = Field(..., description="Creation timestamp")

    @field_validator("message_count")
    @classmethod
    def validate_message_count(cls, v):
        """Ensure message count is positive."""
        return max(1, v)

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class MessageTemporalStats(BaseModel):
    """Represents temporal statistics for messages in a channel."""

    channel_id: str = Field(..., description="Channel ID")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    message_count: int = Field(..., description="Message count for the date")
    active_users: int = Field(..., description="Active users count")
    avg_message_length: float = Field(..., description="Average message length")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )

    @field_validator("message_count", "active_users")
    @classmethod
    def validate_counts(cls, v):
        """Ensure counts are non-negative."""
        return max(0, v)

    @field_validator("avg_message_length")
    @classmethod
    def validate_avg_length(cls, v):
        """Ensure average length is non-negative."""
        return max(0.0, v)

    class Config:
        """Pydantic configuration."""

        from_attributes = True
