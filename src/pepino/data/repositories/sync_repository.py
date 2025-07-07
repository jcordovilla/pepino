"""
Sync repository for handling sync log operations.
"""

import json
from typing import Optional, List

from ...logging_config import get_logger

from ..database.manager import DatabaseManager
from ..models.sync import SyncLogEntry, GuildSyncInfo, ChannelSkipInfo, SyncError

logger = get_logger(__name__)


class SyncRepository:
    """Repository for sync log data access."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def save_sync_log(self, sync_log: SyncLogEntry) -> None:
        """Save sync log entry to database"""
        query = """
            INSERT INTO sync_logs (
                timestamp, guilds_synced, channels_skipped, errors, total_messages_synced
            ) VALUES (?, ?, ?, ?, ?)
        """

        # Convert Pydantic models to JSON for database storage
        guilds_synced_json = json.dumps([guild.model_dump() for guild in sync_log.guilds_synced])
        channels_skipped_json = json.dumps([channel.model_dump() for channel in sync_log.channels_skipped])
        errors_json = json.dumps([error.model_dump() for error in sync_log.errors])

        values = (
            sync_log.timestamp,
            guilds_synced_json,
            channels_skipped_json,
            errors_json,
            sync_log.total_messages_synced,
        )

        self.db_manager.execute_query(query, values, fetch_one=False, fetch_all=False)

    def get_last_sync_log(self) -> Optional[SyncLogEntry]:
        """Get the most recent sync log entry"""
        query = """
            SELECT timestamp, guilds_synced, channels_skipped, errors, total_messages_synced
            FROM sync_logs
            ORDER BY timestamp DESC
            LIMIT 1
        """

        try:
            result = self.db_manager.execute_query(query, fetch_one=True, fetch_all=False)
            if result:
                # Parse JSON data back into Pydantic models
                guilds_synced_data = json.loads(result[1]) if result[1] else []
                channels_skipped_data = json.loads(result[2]) if result[2] else []
                errors_data = json.loads(result[3]) if result[3] else []
                
                return SyncLogEntry(
                    timestamp=result[0],
                    guilds_synced=[GuildSyncInfo(**guild) for guild in guilds_synced_data],
                    channels_skipped=[ChannelSkipInfo(**channel) for channel in channels_skipped_data],
                    errors=[SyncError(**error) for error in errors_data],
                    total_messages_synced=result[4],
                )
            return None
        except Exception as e:
            logger.warning(f"Error fetching last sync log: {e}")
            return None

    def get_all_sync_logs(self, limit: int = 100) -> List[SyncLogEntry]:
        """Get all sync log entries"""
        query = """
            SELECT timestamp, guilds_synced, channels_skipped, errors, total_messages_synced
            FROM sync_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """

        try:
            results = self.db_manager.execute_query(query, (limit,), fetch_one=False, fetch_all=True)
            sync_logs = []
            
            for result in results:
                # Parse JSON data back into Pydantic models
                guilds_synced_data = json.loads(result[1]) if result[1] else []
                channels_skipped_data = json.loads(result[2]) if result[2] else []
                errors_data = json.loads(result[3]) if result[3] else []
                
                sync_logs.append(SyncLogEntry(
                    timestamp=result[0],
                    guilds_synced=[GuildSyncInfo(**guild) for guild in guilds_synced_data],
                    channels_skipped=[ChannelSkipInfo(**channel) for channel in channels_skipped_data],
                    errors=[SyncError(**error) for error in errors_data],
                    total_messages_synced=result[4],
                ))
            
            return sync_logs
        except Exception as e:
            logger.warning(f"Error fetching sync logs: {e}")
            return []

    def clear_all_sync_logs(self):
        """Delete all sync log data from the database."""
        self.db_manager.execute_query("DELETE FROM sync_logs", fetch_one=False, fetch_all=False)
