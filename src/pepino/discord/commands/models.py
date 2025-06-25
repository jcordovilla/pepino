"""
Discord command parameter models using Pydantic for validation.
Models for validating and parsing Discord slash command parameters.
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# Enums for Discord command choices
class AnalysisType(str, Enum):
    """Analysis types for Discord sync_and_analyze command."""

    USER = "user"
    CHANNEL = "channel"
    TOPICS = "topics"
    TRENDS = "trends"
    TOP_USERS = "top_users"


# Base Discord command parameter models
class BaseDiscordParams(BaseModel):
    """Base class for all Discord command parameters."""

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Reject unknown fields


class ListUsersParams(BaseDiscordParams):
    """Parameters for Discord list_users command."""

    limit: int = Field(default=50, description="Number of users to show")

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v):
        """Ensure reasonable limits for Discord display."""
        if v > 999:
            return 999
        elif v < 1:
            return 50
        return v


class ListChannelsParams(BaseDiscordParams):
    """Parameters for Discord list_channels command."""

    limit: int = Field(default=25, description="Number of channels to show")

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v):
        """Ensure reasonable limits for Discord display."""
        if v > 50:
            return 50
        elif v < 1:
            return 25
        return v


class ChannelAnalysisParams(BaseDiscordParams):
    """Parameters for Discord channel_analysis command."""

    channel: str = Field(
        ..., min_length=1, max_length=100, description="Channel name to analyze"
    )
    include_chart: bool = Field(default=True, description="Include an activity chart")

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        """Clean and validate channel name."""
        if not v or not v.strip():
            raise ValueError("Channel name cannot be empty")

        # Remove # prefix if present
        cleaned = v.strip().lstrip("#")
        if not cleaned:
            raise ValueError("Channel name cannot be just '#'")

        return cleaned


class UserAnalysisParams(BaseDiscordParams):
    """Parameters for Discord user_analysis command."""

    user: str = Field(
        ..., min_length=1, max_length=100, description="Username to analyze"
    )
    include_chart: bool = Field(default=True, description="Include an activity chart")

    @field_validator("user")
    @classmethod
    def validate_user(cls, v):
        """Clean and validate username."""
        if not v or not v.strip():
            raise ValueError("Username cannot be empty")

        # Remove @ prefix if present
        cleaned = v.strip().lstrip("@")
        if not cleaned:
            raise ValueError("Username cannot be just '@'")

        return cleaned


class TopicsAnalysisParams(BaseDiscordParams):
    """Parameters for Discord topics_analysis command."""

    channel: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional channel name to filter analysis",
    )
    days_back: int = Field(
        default=30, ge=1, le=365, description="Days to look back for analysis"
    )
    min_word_length: int = Field(
        default=4, ge=3, le=20, description="Minimum word length for topic extraction"
    )

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        """Clean and validate optional channel name."""
        if v is None:
            return None

        if not v.strip():
            return None

        # Remove # prefix if present
        cleaned = v.strip().lstrip("#")
        return cleaned if cleaned else None


class ActivityTrendsParams(BaseDiscordParams):
    """Parameters for Discord activity_trends command."""

    include_chart: bool = Field(default=True, description="Include a trends chart")
    days_back: int = Field(
        default=30, ge=7, le=365, description="Days to analyze for trends"
    )
    granularity: Literal["day", "week"] = Field(
        default="day", description="Time granularity for trends"
    )


class SyncAndAnalyzeParams(BaseDiscordParams):
    """Parameters for Discord sync_and_analyze command."""

    analysis_type: AnalysisType = Field(
        ..., description="Type of analysis to run after sync"
    )
    target: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Target for analysis (user or channel name)",
    )
    force_sync: bool = Field(
        default=False, description="Force sync even if data appears fresh"
    )

    @field_validator("target")
    @classmethod
    def validate_target(cls, v, info):
        """Validate target based on analysis type."""
        if v is None:
            return None

        cleaned = v.strip()
        if not cleaned:
            return None

        # Clean prefixes
        if cleaned.startswith("#"):
            cleaned = cleaned[1:]
        elif cleaned.startswith("@"):
            cleaned = cleaned[1:]

        return cleaned if cleaned else None

    def validate_analysis_requirements(self) -> None:
        """Validate that required targets are provided for specific analysis types."""
        if self.analysis_type in [AnalysisType.USER, AnalysisType.CHANNEL]:
            if not self.target:
                analysis_name = (
                    "user" if self.analysis_type == AnalysisType.USER else "channel"
                )
                raise ValueError(
                    f"Target {analysis_name} name is required for {analysis_name} analysis"
                )


# Response status models for tracking Discord command execution
class CommandStatus(BaseModel):
    """Status tracking for long-running Discord commands."""

    command: str = Field(..., description="Command name")
    user_id: str = Field(..., description="Discord user ID")
    guild_id: Optional[str] = Field(None, description="Discord guild ID")
    channel_id: Optional[str] = Field(None, description="Discord channel ID")
    started_at: str = Field(..., description="Start timestamp")
    status: Literal["running", "completed", "failed"] = Field(
        ..., description="Current status"
    )
    progress: Optional[str] = Field(None, description="Progress message")
    error: Optional[str] = Field(None, description="Error message if failed")
