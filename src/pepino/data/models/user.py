"""
User data model using Pydantic for validation and serialization.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class User(BaseModel):
    """Represents a Discord user with automatic validation."""

    author_id: str = Field(..., description="Author ID")
    author_name: str = Field(..., description="Author name")
    author_display_name: Optional[str] = Field(None, description="Author display name")
    message_count: int = Field(default=0, description="Number of messages")
    channels_active: int = Field(default=0, description="Number of active channels")
    avg_message_length: float = Field(default=0.0, description="Average message length")
    first_message_date: Optional[str] = Field(None, description="First message date")
    last_message_date: Optional[str] = Field(None, description="Last message date")

    @field_validator("avg_message_length")
    @classmethod
    def validate_avg_length(cls, v):
        """Ensure average length is non-negative."""
        return max(0.0, v)

    @field_validator("message_count", "channels_active")
    @classmethod
    def validate_counts(cls, v):
        """Ensure counts are non-negative."""
        return max(0, v)

    @property
    def display_name(self) -> str:
        """Get the display name (preferred) or author name."""
        return self.author_display_name or self.author_name

    @property
    def is_active(self) -> bool:
        """Check if the user is considered active."""
        return self.message_count > 0

    class Config:
        """Pydantic configuration."""

        from_attributes = True  # For compatibility with ORMs
