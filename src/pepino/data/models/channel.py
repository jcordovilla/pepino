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

    @classmethod
    def from_db_row(cls, row):
        """Create a Channel instance from a database row (dict or tuple)."""
        if isinstance(row, dict):
            return cls(
                channel_id=row.get("channel_id") or row.get("id"),
                channel_name=row.get("channel_name"),
                message_count=row.get("message_count", 0),
                unique_users=row.get("unique_users", 0),
                avg_message_length=row.get("avg_message_length", 0.0),
                first_message=row.get("first_message"),
                last_message=row.get("last_message"),
            )
        elif isinstance(row, (list, tuple)):
            # Assume order: channel_id, channel_name, message_count, unique_users, avg_message_length, first_message, last_message
            return cls(
                channel_id=row[0],
                channel_name=row[1],
                message_count=row[2] if len(row) > 2 else 0,
                unique_users=row[3] if len(row) > 3 else 0,
                avg_message_length=row[4] if len(row) > 4 else 0.0,
                first_message=row[5] if len(row) > 5 else None,
                last_message=row[6] if len(row) > 6 else None,
            )
        else:
            raise TypeError(f"Unsupported row type for Channel: {type(row)}")

    class Config:
        """Pydantic configuration."""

        from_attributes = True  # For compatibility with ORMs
