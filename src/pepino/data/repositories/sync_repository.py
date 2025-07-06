"""
Sync repository for handling sync log operations.
"""

import json
from typing import Any, Dict, Optional

from ...logging_config import get_logger

from ..database.manager import DatabaseManager
from ..models.sync import SyncLogEntry

logger = get_logger(__name__)


class SyncRepository:
    """Repository for sync log data access."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def save_sync_log(self, sync_log_entry: SyncLogEntry) -> None:
        """Save sync log entry to database"""
        query = """
            INSERT INTO sync_logs (
                timestamp, guilds_synced, channels_skipped, errors, total_messages_synced
            ) VALUES (?, ?, ?, ?, ?)
        """

        # Convert Pydantic lists to JSON for database storage
        guilds_synced_json = json.dumps(
            [guild.model_dump() for guild in sync_log_entry.guilds_synced]
        )
        channels_skipped_json = json.dumps(
            [skip.model_dump() for skip in sync_log_entry.channels_skipped]
        )
        errors_json = json.dumps(
            [error.model_dump() for error in sync_log_entry.errors]
        )

        values = (
            sync_log_entry.timestamp,
            guilds_synced_json,
            channels_skipped_json,
            errors_json,
            sync_log_entry.total_messages_synced,
        )

        self.db_manager.execute_query(query, values, fetch_all=False)

    def get_last_sync_log(self) -> Optional[Dict[str, Any]]:
        """Get the most recent sync log entry"""
        query = """
            SELECT timestamp, guilds_synced, channels_skipped, errors, total_messages_synced
            FROM sync_logs
            ORDER BY timestamp DESC
            LIMIT 1
        """

        try:
            result = self.db_manager.execute_query(query, fetch_one=True)
            if result:
                return {
                    "completed_at": result[0],  # timestamp
                    "guilds_synced": json.loads(result[1])
                    if result[1]
                    else [],  # guilds_synced
                    "channels_skipped": json.loads(result[2])
                    if result[2]
                    else [],  # channels_skipped
                    "errors": json.loads(result[3]) if result[3] else [],  # errors
                    "total_messages_synced": result[4],  # total_messages_synced
                }
            return None
        except Exception as e:
            logger.warning(f"Error fetching last sync log: {e}")
            return None
