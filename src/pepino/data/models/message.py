"""
Message data model using Pydantic for validation and serialization.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """Represents a Discord message with automatic validation."""

    id: str = Field(..., description="Message ID")
    channel_id: str = Field(..., description="Channel ID")
    channel_name: str = Field(..., description="Channel name")
    author_id: str = Field(..., description="Author ID")
    author_name: str = Field(..., description="Author name")
    author_display_name: Optional[str] = Field(None, description="Author display name")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    edited_timestamp: Optional[datetime] = Field(None, description="Edited timestamp")
    author_is_bot: bool = Field(default=False, description="Is bot message")
    is_system: bool = Field(default=False, description="Is system message")
    is_webhook: bool = Field(default=False, description="Is webhook message")
    has_attachments: bool = Field(default=False, description="Has attachments")
    has_embeds: bool = Field(default=False, description="Has embeds")
    has_stickers: bool = Field(default=False, description="Has stickers")
    has_mentions: bool = Field(default=False, description="Has mentions")
    has_reactions: bool = Field(default=False, description="Has reactions")
    has_reference: bool = Field(default=False, description="Has reference")
    referenced_message_id: Optional[str] = Field(
        None, description="Referenced message ID"
    )

    @field_validator("timestamp", "edited_timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        """Parse timestamp from string to datetime."""
        if isinstance(v, str):
            # Handle ISO format with Z suffix
            if v.endswith("Z"):
                v = v.replace("Z", "+00:00")
            return datetime.fromisoformat(v)
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Validate and clean content."""
        if v is not None and len(v.strip()) == 0:
            return ""
        return v

    @property
    def display_name(self) -> str:
        """Get the display name (preferred) or author name."""
        return self.author_display_name or self.author_name

    @property
    def is_human(self) -> bool:
        """Check if the message is from a human user."""
        return not self.author_is_bot and not self.is_system and not self.is_webhook

    @property
    def content_length(self) -> int:
        """Get the length of the message content."""
        return len(self.content) if self.content else 0

    @classmethod
    def from_db_row(cls, row) -> "Message":
        """Create Message from database row with automatic validation."""
        # Convert row to dict if it's not already
        if not isinstance(row, dict):
            # Assume it's a tuple or list-like object
            # This is a fallback for compatibility
            return cls.model_validate(
                {
                    "id": row[0],
                    "channel_id": row[1],
                    "channel_name": row[2],
                    "author_id": row[3],
                    "author_name": row[4],
                    "author_display_name": row[5],
                    "content": row[6],
                    "timestamp": row[7],
                    "edited_timestamp": row[8],
                    "author_is_bot": row[9],
                    "is_system": row[10],
                    "is_webhook": row[11],
                    "has_attachments": row[12],
                    "has_embeds": row[13],
                    "has_stickers": row[14],
                    "has_mentions": row[15],
                    "has_reactions": row[16],
                    "has_reference": row[17],
                    "referenced_message_id": row[18] if len(row) > 18 else None,
                }
            )

        return cls.model_validate(row)

    class Config:
        """Pydantic configuration."""

        from_attributes = True  # For compatibility with ORMs
        json_encoders = {datetime: lambda v: v.isoformat()}
