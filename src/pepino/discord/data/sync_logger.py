"""
Sync logging functionality for Discord data fetching.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from pepino.logging_config import get_logger
from ..sync.models import ChannelSkipInfo, GuildSyncInfo, SyncError, SyncLogEntry


class SyncLogger:
    """Handles logging of Discord data sync operations using standard logging"""

    def __init__(self):
        self.logger = get_logger("pepino.discord.sync")
        now = datetime.now(timezone.utc).isoformat()
        self.sync_log_entry = SyncLogEntry(timestamp=now, sync_start_time=now)
        
        # Log sync start
        self.logger.info("Discord sync operation started")

    def add_guild_sync(self, guild_name: str, guild_id: str) -> None:
        """Add a guild to the sync log and log the event"""
        guild_sync = GuildSyncInfo(
            guild_name=guild_name,
            guild_id=guild_id,
            sync_start_time=datetime.now(timezone.utc).isoformat(),
        )
        self.sync_log_entry.guilds_synced.append(guild_sync)
        
        # Log guild sync start
        self.logger.info(f"Starting guild sync: {guild_name} (ID: {guild_id})")

    def complete_guild_sync(self, guild_name: str) -> None:
        """Mark a guild sync as complete and log the event"""
        for guild in self.sync_log_entry.guilds_synced:
            if guild.guild_name == guild_name:
                guild.sync_end_time = datetime.now(timezone.utc).isoformat()
                break
        
        # Log guild sync completion
        self.logger.info(f"Completed guild sync: {guild_name}")

    def add_channel_skip(
        self, guild_name: str, channel_name: str, channel_id: str, reason: str
    ) -> None:
        """Add a skipped channel to the sync log and log the event"""
        skip_info = ChannelSkipInfo(
            guild_name=guild_name,
            channel_name=channel_name,
            channel_id=channel_id,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.sync_log_entry.channels_skipped.append(skip_info)
        
        # Log channel skip
        self.logger.warning(f"Skipped channel: #{channel_name} in {guild_name} - {reason}")

    def add_error(
        self,
        error: str,
        guild_name: str = None,
        channel_name: str = None,
        channel_id: str = None,
    ) -> None:
        """Add an error to the sync log and log the event"""
        error_info = SyncError(
            error=error,
            guild_name=guild_name,
            channel_name=channel_name,
            channel_id=channel_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.sync_log_entry.errors.append(error_info)
        
        # Log error
        context = f" in {guild_name}" if guild_name else ""
        context += f" channel #{channel_name}" if channel_name else ""
        self.logger.error(f"Sync error{context}: {error}")

    def add_messages_synced(self, count: int) -> None:
        """Add to the total messages synced count and log the event"""
        self.sync_log_entry.total_messages_synced += count
        
        # Log message sync progress
        self.logger.info(f"Synced {count} messages")

    def finalize_sync(self) -> None:
        """Finalize the sync log entry and log the summary"""
        sync_end_time = datetime.now(timezone.utc)
        self.sync_log_entry.sync_end_time = sync_end_time.isoformat()

        # Calculate duration
        sync_start_time = datetime.fromisoformat(self.sync_log_entry.sync_start_time)
        self.sync_log_entry.sync_duration_seconds = (
            sync_end_time - sync_start_time
        ).total_seconds()
        
        # Log sync completion with summary
        total_messages = self.sync_log_entry.total_messages_synced
        duration = self.sync_log_entry.sync_duration_seconds
        self.logger.info(f"Discord sync operation completed - {total_messages} messages in {duration:.1f}s")

    def get_log_entry(self) -> SyncLogEntry:
        """Get the current sync log entry"""
        return self.sync_log_entry
