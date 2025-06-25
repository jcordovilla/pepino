"""
CLI command parameter models using Pydantic for validation.
Models for validating and parsing CLI command parameters.
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# Enums for CLI choices
class OutputFormat(str, Enum):
    """Output format options for CLI commands."""

    JSON = "json"
    CSV = "csv"
    TEXT = "text"


# Base CLI parameter models
class BaseCLIParams(BaseModel):
    """Base class for all CLI command parameters."""

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Reject unknown fields


class ListUsersParams(BaseCLIParams):
    """Parameters for CLI list users command."""

    limit: int = Field(default=50, description="Number of users to show")

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v):
        """Ensure reasonable limits."""
        if v > 999:
            return 999
        elif v < 1:
            return 50
        return v


class ListChannelsParams(BaseCLIParams):
    """Parameters for CLI list channels command."""

    limit: int = Field(default=25, description="Number of channels to show")

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v):
        """Ensure reasonable limits."""
        if v > 50:
            return 50
        elif v < 1:
            return 25
        return v


class AnalysisOutputParams(BaseCLIParams):
    """Common output parameters for analysis commands."""

    output: Optional[str] = Field(default=None, description="Output file path")
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON, description="Output format"
    )


class UserAnalysisParams(AnalysisOutputParams):
    """Parameters for CLI user analysis command."""

    user: Optional[str] = Field(
        default=None, max_length=100, description="Username to analyze"
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Number of top users to show"
    )

    @field_validator("user")
    @classmethod
    def validate_user(cls, v):
        """Clean and validate username."""
        if v is None:
            return None

        cleaned = v.strip()
        if not cleaned:
            return None

        # Remove @ prefix if present
        if cleaned.startswith("@"):
            cleaned = cleaned[1:]

        return cleaned if cleaned else None


class ChannelAnalysisParams(AnalysisOutputParams):
    """Parameters for CLI channel analysis command."""

    channel: Optional[str] = Field(
        default=None, max_length=100, description="Channel name to analyze"
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Number of top channels to show"
    )

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        """Clean and validate channel name."""
        if v is None:
            return None

        cleaned = v.strip()
        if not cleaned:
            return None

        # Remove # prefix if present
        if cleaned.startswith("#"):
            cleaned = cleaned[1:]

        return cleaned if cleaned else None


class TopicsAnalysisParams(AnalysisOutputParams):
    """Parameters for CLI topics analysis command."""

    channel: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional channel name to filter analysis",
    )
    n_topics: int = Field(
        default=10, ge=1, le=50, description="Number of topics to extract"
    )
    days_back: int = Field(
        default=30, ge=1, le=365, description="Days to look back for analysis"
    )

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        """Clean and validate optional channel name."""
        if v is None:
            return None

        cleaned = v.strip()
        if not cleaned:
            return None

        # Remove # prefix if present
        if cleaned.startswith("#"):
            cleaned = cleaned[1:]

        return cleaned if cleaned else None


class TemporalAnalysisParams(AnalysisOutputParams):
    """Parameters for CLI temporal analysis command."""

    channel: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional channel name to filter analysis",
    )
    days_back: int = Field(
        default=30, ge=1, le=365, description="Days to analyze for trends"
    )
    granularity: Literal["day", "week"] = Field(
        default="day", description="Time granularity for analysis"
    )

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        """Clean and validate optional channel name."""
        if v is None:
            return None

        cleaned = v.strip()
        if not cleaned:
            return None

        # Remove # prefix if present
        if cleaned.startswith("#"):
            cleaned = cleaned[1:]

        return cleaned if cleaned else None
