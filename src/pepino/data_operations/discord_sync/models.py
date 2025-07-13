"""
Sync operation models for Discord synchronization.

Note: Core sync models have been moved to pepino.data.models.sync
to avoid circular imports. This module re-exports them for backward compatibility.
"""

from datetime import datetime
from typing import List, Literal, Optional, Union

from pydantic import BaseModel

# Import from data models to avoid circular dependency
from pepino.data.models.sync import (
    ChannelSkipInfo,
    GuildSyncInfo,
    SyncError,
    SyncLogEntry,
)


class SyncResult(BaseModel):
    """Base sync result model."""

    sync_performed: bool
    duration: float = 0.0
    new_messages: int = 0
    last_sync: Optional[datetime] = None


class IncrementalSyncResult(SyncResult):
    """Result of incremental sync operation."""

    reason: Optional[str] = None
    updated_channels: int = 0
    sync_log: Optional[SyncLogEntry] = None
    error: Optional[str] = None


class FullSyncResult(SyncResult):
    """Result of full sync operation."""

    type: Literal["full"] = "full"
    sync_log: Optional[SyncLogEntry] = None
    error: Optional[str] = None


SyncResultType = Union[IncrementalSyncResult, FullSyncResult]
