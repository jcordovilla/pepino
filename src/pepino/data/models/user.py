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

    @classmethod
    def from_db_row(cls, row):
        """Create a User instance from a database row (dict, tuple, or sqlite3.Row)."""
        if isinstance(row, dict):
            return cls(
                author_id=row.get("author_id") or row.get("id"),
                author_name=row.get("author_name"),
                author_display_name=row.get("author_display_name"),
                message_count=row.get("message_count", 0),
                channels_active=row.get("channels_active", 0),
                avg_message_length=row.get("avg_message_length", 0.0) or 0.0,
                first_message_date=row.get("first_message_date"),
                last_message_date=row.get("last_message_date"),
            )
        elif hasattr(row, 'keys') and hasattr(row, '__getitem__'):
            # Handle sqlite3.Row objects (they have keys() and __getitem__)
            return cls(
                author_id=row["author_id"] if "author_id" in row else (row["id"] if "id" in row else None),
                author_name=row["author_name"] if "author_name" in row else None,
                author_display_name=row["author_display_name"] if "author_display_name" in row else None,
                message_count=row["message_count"] if "message_count" in row else 0,
                channels_active=row["channels_active"] if "channels_active" in row else 0,
                avg_message_length=row["avg_message_length"] if "avg_message_length" in row else 0.0,
                first_message_date=row["first_message_date"] if "first_message_date" in row else None,
                last_message_date=row["last_message_date"] if "last_message_date" in row else None,
            )
        elif isinstance(row, (list, tuple)):
            # Handle case where row is a list/tuple of dicts (from execute_query)
            if len(row) > 0 and isinstance(row[0], dict):
                return cls.from_db_row(row[0])
            # Assume order: author_id, author_name, author_display_name, message_count, channels_active, avg_message_length, first_message_date, last_message_date
            return cls(
                author_id=row[0],
                author_name=row[1],
                author_display_name=row[2] if len(row) > 2 else None,
                message_count=row[3] if len(row) > 3 else 0,
                channels_active=row[4] if len(row) > 4 else 0,
                avg_message_length=(row[5] or 0.0) if len(row) > 5 else 0.0,
                first_message_date=row[6] if len(row) > 6 else None,
                last_message_date=row[7] if len(row) > 7 else None,
            )
        else:
            raise TypeError(f"Unsupported row type for User: {type(row)}")

    class Config:
        """Pydantic configuration."""

        from_attributes = True  # For compatibility with ORMs
