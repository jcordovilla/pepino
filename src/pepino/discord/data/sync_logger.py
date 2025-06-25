"""
Sync logging functionality for Discord data fetching.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..sync.models import ChannelSkipInfo, GuildSyncInfo, SyncError, SyncLogEntry

logger = logging.getLogger(__name__)


class SyncLogger:
    """Handles logging of Discord data sync operations"""

    def __init__(self):
        now = datetime.now(timezone.utc).isoformat()
        self.sync_log_entry = SyncLogEntry(timestamp=now, sync_start_time=now)

    def add_guild_sync(self, guild_name: str, guild_id: str) -> None:
        """Add a guild to the sync log"""
        guild_sync = GuildSyncInfo(
            guild_name=guild_name,
            guild_id=guild_id,
            sync_start_time=datetime.now(timezone.utc).isoformat(),
        )
        self.sync_log_entry.guilds_synced.append(guild_sync)

    def complete_guild_sync(self, guild_name: str) -> None:
        """Mark a guild sync as complete"""
        for guild in self.sync_log_entry.guilds_synced:
            if guild.guild_name == guild_name:
                guild.sync_end_time = datetime.now(timezone.utc).isoformat()
                break

    def add_channel_skip(
        self, guild_name: str, channel_name: str, channel_id: str, reason: str
    ) -> None:
        """Add a skipped channel to the sync log"""
        skip_info = ChannelSkipInfo(
            guild_name=guild_name,
            channel_name=channel_name,
            channel_id=channel_id,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.sync_log_entry.channels_skipped.append(skip_info)

    def add_error(
        self,
        error: str,
        guild_name: str = None,
        channel_name: str = None,
        channel_id: str = None,
    ) -> None:
        """Add an error to the sync log"""
        error_info = SyncError(
            error=error,
            guild_name=guild_name,
            channel_name=channel_name,
            channel_id=channel_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.sync_log_entry.errors.append(error_info)

    def add_messages_synced(self, count: int) -> None:
        """Add to the total messages synced count"""
        self.sync_log_entry.total_messages_synced += count

    def finalize_sync(self) -> None:
        """Finalize the sync log entry"""
        sync_end_time = datetime.now(timezone.utc)
        self.sync_log_entry.sync_end_time = sync_end_time.isoformat()

        # Calculate duration
        sync_start_time = datetime.fromisoformat(self.sync_log_entry.sync_start_time)
        self.sync_log_entry.sync_duration_seconds = (
            sync_end_time - sync_start_time
        ).total_seconds()

    def save_to_file(self) -> None:
        """Save sync log to file"""
        os.makedirs("logs", exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path = os.path.join("logs", f"sync_log_{date_str}.jsonl")

        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write(self.sync_log_entry.model_dump_json() + "\n")

    def get_log_entry(self) -> SyncLogEntry:
        """Get the current sync log entry"""
        return self.sync_log_entry
