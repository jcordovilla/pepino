"""
Channel data model using Pydantic for validation and serialization.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Channel(BaseModel):
    """Represents a Discord channel with automatic validation."""

    channel_id: str = Field(..., description="Channel ID")
    channel_name: str = Field(..., description="Channel name")
    message_count: int = Field(default=0, description="Number of messages")
    unique_users: int = Field(default=0, description="Number of unique users")
    avg_message_length: float = Field(default=0.0, description="Average message length")
    first_message: Optional[str] = Field(None, description="First message timestamp")
    last_message: Optional[str] = Field(None, description="Last message timestamp")

    @field_validator("message_count", "unique_users")
    @classmethod
    def validate_counts(cls, v):
        """Ensure counts are non-negative."""
        return max(0, v)

    @field_validator("avg_message_length")
    @classmethod
    def validate_avg_length(cls, v):
        """Ensure average length is non-negative."""
        return max(0.0, v)

    @property
    def is_active(self) -> bool:
        """Check if the channel is considered active."""
        return self.message_count > 0

    @property
    def activity_level(self) -> str:
        """Get the activity level of the channel."""
        if self.message_count == 0:
            return "inactive"
        elif self.message_count < 100:
            return "low"
        elif self.message_count < 1000:
            return "medium"
        else:
            return "high"

    class Config:
        """Pydantic configuration."""

        from_attributes = True  # For compatibility with ORMs
