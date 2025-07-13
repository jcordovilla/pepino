"""
Sync operation models for sync repository Discord synchronization operations.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class GuildSyncInfo(BaseModel):
    """Information about a guild sync operation."""

    guild_name: str
    guild_id: str
    sync_start_time: str
    sync_end_time: Optional[str] = None


class ChannelSkipInfo(BaseModel):
    """Information about a skipped channel."""

    guild_name: str
    channel_name: str
    channel_id: str
    reason: str
    timestamp: str


class SyncError(BaseModel):
    """Sync error information."""

    error: str
    guild_name: Optional[str] = None
    channel_name: Optional[str] = None
    channel_id: Optional[str] = None
    timestamp: str


class SyncLogEntry(BaseModel):
    """Complete sync log entry."""

    timestamp: str
    guilds_synced: List[GuildSyncInfo] = Field(default_factory=list)
    channels_skipped: List[ChannelSkipInfo] = Field(default_factory=list)
    errors: List[SyncError] = Field(default_factory=list)
    total_messages_synced: int = 0
    sync_start_time: str
    sync_end_time: Optional[str] = None
    sync_duration_seconds: Optional[float] = None 