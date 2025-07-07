"""
Sync repository for handling sync log operations.
"""

import json
from typing import Any, Dict, Optional, List

from pepino.logging_config import get_logger

from ..database.manager import DatabaseManager

logger = get_logger(__name__)


class SyncRepository:
    """Repository for sync log data access."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def save_sync_log(self, sync_log_data: Dict[str, Any]) -> None:
        """Save sync log entry to database"""
        query = """
            INSERT INTO sync_logs (
                timestamp, guilds_synced, channels_skipped, errors, total_messages_synced
            ) VALUES (?, ?, ?, ?, ?)
        """

        # Convert data to JSON for database storage
        guilds_synced_json = json.dumps(sync_log_data.get("guilds_synced", []))
        channels_skipped_json = json.dumps(sync_log_data.get("channels_skipped", []))
        errors_json = json.dumps(sync_log_data.get("errors", []))

        values = (
            sync_log_data.get("timestamp"),
            guilds_synced_json,
            channels_skipped_json,
            errors_json,
            sync_log_data.get("total_messages_synced", 0),
        )

        self.db_manager.execute_query(query, values, fetch_one=False, fetch_all=False)

    def get_last_sync_log(self) -> Optional[Dict[str, Any]]:
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

    def get_all_sync_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
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
                sync_logs.append({
                    "timestamp": result[0],
                    "guilds_synced": json.loads(result[1]) if result[1] else [],
                    "channels_skipped": json.loads(result[2]) if result[2] else [],
                    "errors": json.loads(result[3]) if result[3] else [],
                    "total_messages_synced": result[4],
                })
            
            return sync_logs
        except Exception as e:
            logger.warning(f"Error fetching sync logs: {e}")
            return []

    def clear_all_sync_logs(self):
        """Delete all sync log data from the database."""
        self.db_manager.execute_query("DELETE FROM sync_logs", fetch_one=False, fetch_all=False)
